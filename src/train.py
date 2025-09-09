import torch
import os
import numpy as np
import time
from tqdm import tqdm
import gc
import logging

from src.loss import LaplaceHomoscedasticLoss
from data.dataloader import get_clusters_dataloader, get_single_cluster_dataloader

LOGGER = logging.getLogger("experiment")


def objective(
    trial,
    model,
    num_epochs,
    cluster,
    cluster_names,
    config,
    device,
    device_data,
    augment,
    single_cluster=False,
):
    # Hyperparameter optimization
    lr_model = trial.suggest_float("lr_model", 1e-6, 1e-4, log=True)
    lr_loss_T = trial.suggest_float("lr_loss_T", 1e-6, 1e-5, log=True)
    lr_loss_P = trial.suggest_float("lr_loss_P", 1e-6, 1e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=False)

    criterion = LaplaceHomoscedasticLoss(
        init_logb_T=np.log(config["training"]["init_b_T"]),
        init_logb_P=np.log(config["training"]["init_b_P"]),
    ).to(device)

    # Separate optimizers
    optimizer_model = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(), lr=lr_model, weight_decay=weight_decay
    )
    # New: Separate optimizers for eta_T and eta_P
    optimizer_loss_T = getattr(torch.optim, config["training"]["optimizer"])(
        [criterion.eta_T], lr=lr_loss_T, weight_decay=0.0
    )
    optimizer_loss_P = getattr(torch.optim, config["training"]["optimizer"])(
        [criterion.eta_P], lr=lr_loss_P, weight_decay=0.0
    )

    # Separate schedulers
    scheduler_model = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_model, **config["training"]["scheduler_params"]
    )
    scheduler_loss_T = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_loss_T, **config["training"]["scheduler_params"]
    )
    scheduler_loss_P = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_loss_P, **config["training"]["scheduler_params"]
    )

    if single_cluster:
        cluster_dataloaders = get_single_cluster_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            cluster_name=cluster,
            vars=config["experiment"]["vars"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            augment=augment,
        )
    else:
        cluster_dataloaders = get_clusters_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            cluster_names=cluster_names,
            vars=config["experiment"]["vars"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            augment=augment,
        )

    train_loader = cluster_dataloaders["train"]
    val_loader = cluster_dataloaders["val"]

    # Training loop
    for _ in range(num_epochs):
        _, _, _, _, _, _, _, _, _, val_loss = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_model=optimizer_model,
            optimizer_loss_T=optimizer_loss_T,
            optimizer_loss_P=optimizer_loss_P,
            scheduler_model=scheduler_model,
            scheduler_loss_T=scheduler_loss_T,
            scheduler_loss_P=scheduler_loss_P,
            criterion=criterion,
            device=device,
        )

    # Empty GPU memory
    for split in ["train", "val", "test"]:
        dataset = cluster_dataloaders[split].dataset
        if hasattr(dataset, "datasets"):
            for d in dataset.datasets:
                d.unload_from_gpu()
        else:
            dataset.unload_from_gpu()
    del train_loader, val_loader, cluster_dataloaders, model, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss


def _train_step(
    model,
    train_loader,
    val_loader,
    optimizer_model,
    optimizer_loss_T,
    optimizer_loss_P,
    scheduler_model,
    scheduler_loss_T,
    scheduler_loss_P,
    criterion,
    device,
):
    """
    A single training and validation step for one epoch with separate optimizers.
    Includes NaN/Inf protection and safe logging.
    """

    # Training
    train_loss = 0.0
    temp_loss = 0.0
    precip_loss = 0.0
    b_T_accum = 0.0
    b_P_accum = 0.0

    model.train()
    for inputs, coarse_inputs, elev, mask, targets in train_loader:
        inputs = inputs.to(device)
        coarse_inputs = coarse_inputs.to(device)
        elev = elev.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer_model.zero_grad(set_to_none=True)
        optimizer_loss_T.zero_grad(set_to_none=True)
        optimizer_loss_P.zero_grad(set_to_none=True)

        pred_T, pred_P = model(inputs, coarse_inputs, elev, mask)

        loss, mae_T, mae_P, b_T, b_P = criterion(pred_T, targets[:, 0:1], pred_P, targets[:, 1:2])

        # Guard against invalid loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Skipping training batch due to invalid loss: {loss.item()}")
            continue

        loss.backward()
        optimizer_model.step()
        optimizer_loss_T.step()
        optimizer_loss_P.step()

        # Accumulate metrics
        b_T_accum += b_T.cpu().item()
        b_P_accum += b_P.cpu().item()
        temp_loss += mae_T.cpu().item()
        precip_loss += mae_P.cpu().item()
        train_loss += loss.detach().cpu().item()

    # Average training metrics
    num_batches = max(1, len(train_loader))
    b_T = b_T_accum / num_batches
    b_P = b_P_accum / num_batches
    temp_loss /= num_batches
    precip_loss /= num_batches
    train_loss /= num_batches

    # Validation
    model.eval()
    val_loss = 0.0
    val_temp_loss = 0.0
    val_precip_loss = 0.0
    val_b_T_accum = 0.0
    val_b_P_accum = 0.0

    with torch.no_grad():
        for inputs, coarse_inputs, elev, mask, targets in val_loader:
            inputs = inputs.to(device)
            coarse_inputs = coarse_inputs.to(device)
            elev = elev.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            pred_T, pred_P = model(inputs, coarse_inputs, elev, mask)

            loss, mae_T, mae_P, b_T, b_P = criterion(pred_T, targets[:, 0:1], pred_P, targets[:, 1:2])

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Skipping validation batch due to invalid loss: {loss.item()}")
                continue

            val_b_T_accum += b_T.cpu().item()
            val_b_P_accum += b_P.cpu().item()
            val_temp_loss += mae_T.cpu().item()
            val_precip_loss += mae_P.cpu().item()
            val_loss += loss.detach().cpu().item()

    num_val_batches = max(1, len(val_loader))
    val_b_T = val_b_T_accum / num_val_batches
    val_b_P = val_b_P_accum / num_val_batches
    val_loss /= num_val_batches
    val_temp_loss /= num_val_batches
    val_precip_loss /= num_val_batches

    # Step schedulers if provided
    if scheduler_model is not None:
        scheduler_model.step(val_loss)
    if scheduler_loss_T is not None:
        scheduler_loss_T.step(val_loss)
    if scheduler_loss_P is not None:
        scheduler_loss_P.step(val_loss)

    torch.cuda.empty_cache()

    return (
        b_T,
        b_P,
        temp_loss,
        precip_loss,
        train_loss,
        val_b_T,
        val_b_P,
        val_temp_loss,
        val_precip_loss,
        val_loss,
    )


def train_model(
    model,
    cluster_name,
    num_epochs,
    train_loader,
    val_loader,
    config,
    device,
    save_path,
):
    """Train the model with separate optimizers for model and loss parameters.
    Includes checkpointing, early stopping, and logging.
    """

    criterion = LaplaceHomoscedasticLoss(
        init_logb_T=torch.tensor(np.log(config["training"]["init_b_T"])),
        init_logb_P=torch.tensor(np.log(config["training"]["init_b_P"])),
    ).to(device)

    # Separate optimizers for model and loss parameters
    optimizer_model = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(),
        lr=config["domain_specific"][cluster_name]["optimizer_params"]["lr_model"],
        weight_decay=config["domain_specific"][cluster_name]["optimizer_params"].get("weight_decay", 0.0),
    )
    optimizer_loss_T = getattr(torch.optim, config["training"]["optimizer"])(
        [p for n, p in criterion.named_parameters() if p.requires_grad and "eta_T" in n],
        lr=config["domain_specific"][cluster_name]["loss_params"]["lr_loss_T"],
        weight_decay=config["domain_specific"][cluster_name]["loss_params"].get("weight_decay", 0.0),
    )
    optimizer_loss_P = getattr(torch.optim, config["training"]["optimizer"])(
        [p for n, p in criterion.named_parameters() if p.requires_grad and "eta_P" in n],
        lr=config["domain_specific"][cluster_name]["loss_params"]["lr_loss_P"],
        weight_decay=config["domain_specific"][cluster_name]["loss_params"].get("weight_decay", 0.0),
    )

    scheduler_model = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_model, **config["training"]["scheduler_params"]
    )
    scheduler_loss_T = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_loss_T, **config["training"]["scheduler_params"]
    )
    scheduler_loss_P = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer_loss_P, **config["training"]["scheduler_params"]
    )

    # Early stopping and logging setup
    early_stopping = config["training"]["early_stopping"]
    patience = config["training"]["early_stopping_params"]["patience"]

    list_b_T, list_b_P = [], []
    val_list_b_P, val_list_b_T = [], []
    train_losses, train_temp_losses, train_precip_losses = [], [], []
    val_losses, val_temp_losses, val_precip_losses = [], [], []

    # Checkpoint loading logic needs to be updated to handle two optimizers/schedulers
    cluster_dir = os.path.join(save_path, cluster_name)
    if os.path.exists(os.path.join(cluster_dir, "last_snapshot.pth")):
        LOGGER.info(f"TRAINING: Loading model checkpoint from {cluster_dir} ...")
        checkpoint = torch.load(os.path.join(cluster_dir, "last_snapshot.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Load states for both optimizers and schedulers
        optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])
        optimizer_loss_T.load_state_dict(checkpoint["optimizer_loss_T_state_dict"])
        optimizer_loss_P.load_state_dict(checkpoint["optimizer_loss_P_state_dict"])
        scheduler_model.load_state_dict(checkpoint["scheduler_model_state_dict"])
        scheduler_loss_T.load_state_dict(checkpoint["scheduler_loss_T_state_dict"])
        scheduler_loss_P.load_state_dict(checkpoint["scheduler_loss_P_state_dict"])
        start_epoch = checkpoint["epoch"]
        if os.path.exists(os.path.join(cluster_dir, "train_losses.npy")):
            train_losses = list(np.load(os.path.join(cluster_dir, "train_losses.npy")))
        if os.path.exists(os.path.join(cluster_dir, "val_losses.npy")):
            val_losses = list(np.load(os.path.join(cluster_dir, "val_losses.npy")))
        early_stop_counter = checkpoint.get("early_stop_counter", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        LOGGER.info(f"TRAINING: Resuming training from epoch {start_epoch + 1}")
    else:
        LOGGER.info("TRAINING: No checkpoint found, starting fresh training.")
        os.makedirs(os.path.join(save_path, cluster_name), exist_ok=True)
        start_epoch = 0
        early_stop_counter = 0
        best_val_loss = float("inf")

    log_file = os.path.join(cluster_dir, "training_log.csv")
    if os.path.exists(log_file):
        with open(log_file, "a") as f:
            f.write(f"Resuming training from epoch {start_epoch+1}\n")
    else:
        with open(log_file, "w") as f:
            f.write(
                "Epoch,Train Loss,Validation Loss,Temperature Train Loss,Precipitation Train Loss,Temperature Validation Loss,Precipitation Validation Loss,Model LR,Loss LR,Epoch Time\n"
            )

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Progress:"):
        model.train()
        epoch_start_time = time.time()

        (
            b_T,
            b_P,
            temp_loss,
            precip_loss,
            train_loss,
            val_b_T,
            val_b_P,
            val_temp_loss,
            val_precip_loss,
            val_loss,
        ) = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_model=optimizer_model,
            optimizer_loss_T=optimizer_loss_T,
            optimizer_loss_P=optimizer_loss_P,
            scheduler_model=scheduler_model,
            scheduler_loss_T=scheduler_loss_T,
            scheduler_loss_P=scheduler_loss_P,
            criterion=criterion,
            device=device,
        )

        list_b_T.append(float(b_T))
        list_b_P.append(float(b_P))
        train_temp_losses.append(float(temp_loss))
        train_precip_losses.append(float(precip_loss))
        train_losses.append(float(train_loss))
        val_list_b_P.append(float(val_b_P))
        val_list_b_T.append(float(val_b_T))
        val_temp_losses.append(float(val_temp_loss))
        val_precip_losses.append(float(val_precip_loss))
        val_losses.append(float(val_loss))

        # Early stopping and saving logic based on overall validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model snapshot
            snapshot_path = os.path.join(cluster_dir, "best_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_model_state_dict": optimizer_model.state_dict(),
                    "optimizer_loss_T_state_dict": optimizer_loss_T.state_dict(),
                    "optimizer_loss_P_state_dict": optimizer_loss_P.state_dict(),
                    "scheduler_model_state_dict": scheduler_model.state_dict(),
                    "scheduler_loss_T_state_dict": scheduler_loss_T.state_dict(),
                    "scheduler_loss_P_state_dict": scheduler_loss_P.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_temp_loss": val_temp_loss,
                    "val_precip_loss": val_precip_loss,
                    "b_T": list_b_T[-1],
                    "b_P": list_b_P[-1],
                },
                snapshot_path,
            )
        else:
            early_stop_counter += 1

        epoch_time = time.time() - epoch_start_time
        current_lr_model = optimizer_model.param_groups[0]["lr"]
        current_lr_loss_T = optimizer_loss_T.param_groups[0]["lr"]
        current_lr_loss_P = optimizer_loss_P.param_groups[0]["lr"]
        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_temp_losses[-1]:.6f},{train_precip_losses[-1]:.6f},{val_temp_losses[-1]:.6f},{val_precip_losses[-1]:.6f},{current_lr_model:.6e},{current_lr_loss_T:.6e},{current_lr_loss_P:.6e},{epoch_time:.2f}\n"
            )

        if early_stopping and (early_stop_counter >= patience):
            LOGGER.info("TRAINING: Early stopping triggered (Validation loss stagnation).")
            last_snapshot_path = os.path.join(cluster_dir, "last_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_model_state_dict": optimizer_model.state_dict(),
                    "optimizer_loss_T_state_dict": optimizer_loss_T.state_dict(),
                    "optimizer_loss_P_state_dict": optimizer_loss_P.state_dict(),
                    "scheduler_model_state_dict": scheduler_model.state_dict(),
                    "scheduler_loss_T_state_dict": scheduler_loss_T.state_dict(),
                    "scheduler_loss_P_state_dict": scheduler_loss_P.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_temp_loss": val_temp_loss,
                    "val_precip_loss": val_precip_loss,
                    "b_T": list_b_T[-1],
                    "b_P": list_b_P[-1],
                    "early_stop_counter": early_stop_counter,
                },
                last_snapshot_path,
            )
            LOGGER.info(f"TRAINING: Last model state saved to {last_snapshot_path}")
            break

        torch.cuda.empty_cache()

    LOGGER.info(f"TRAINING: Training complete! Best model saved as: {snapshot_path}")

    def to_numpy_safe(x):
        """Convert list of tensors/floats to NumPy array safely."""
        return np.array([t.detach().cpu().item() if torch.is_tensor(t) else float(t) for t in x])

    np.save(os.path.join(cluster_dir, "b_T.npy"), to_numpy_safe(list_b_T))
    np.save(os.path.join(cluster_dir, "b_P.npy"), to_numpy_safe(list_b_P))
    np.save(os.path.join(cluster_dir, "train_losses.npy"), to_numpy_safe(train_losses))
    np.save(
        os.path.join(cluster_dir, "train_temp_losses.npy"),
        to_numpy_safe(train_temp_losses),
    )
    np.save(
        os.path.join(cluster_dir, "train_precip_losses.npy"),
        to_numpy_safe(train_precip_losses),
    )
    np.save(os.path.join(cluster_dir, "val_b_T.npy"), to_numpy_safe(val_list_b_T))
    np.save(os.path.join(cluster_dir, "val_b_P.npy"), to_numpy_safe(val_list_b_P))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), to_numpy_safe(val_losses))
    np.save(os.path.join(cluster_dir, "val_temp_losses.npy"), to_numpy_safe(val_temp_losses))
    np.save(
        os.path.join(cluster_dir, "val_precip_losses.npy"),
        to_numpy_safe(val_precip_losses),
    )
