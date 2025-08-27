import torch
import os
import numpy as np
import time
from tqdm import tqdm
import gc
import yaml
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
    # New: Suggest a ratio instead of an independent lr_loss
    lr_ratio = trial.suggest_float("lr_ratio", 0.1, 1.0)
    lr_loss = lr_model * lr_ratio
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Use Laplace homoscedastic loss
    criterion = LaplaceHomoscedasticLoss().to(device)

    # Optimizer with separate param groups
    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        [
            {
                "params": model.parameters(),
                "lr": lr_model,
                "weight_decay": weight_decay,
            },
            {
                "params": [criterion.log_b_T, criterion.log_b_P],
                "lr": lr_loss,
                "weight_decay": 0.0,
            },
        ]
    )

    # Scheduler applied to all param groups
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer, **config["training"]["scheduler_params"]
    )

    # Data loaders
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
            excluded_cluster=cluster,
            cluster_names=cluster_names,
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
            optimizer=optimizer,
            scheduler=scheduler,
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
    optimizer,
    scheduler,
    criterion,
    device,
):
    # --------------------
    # Training
    # --------------------
    train_loss = 0.0
    temp_loss = 0.0
    precip_loss = 0.0
    b_T_accum = 0.0
    b_P_accum = 0.0

    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Channel 0 = temperature, channel 1 = precipitation
        loss, mae_T, mae_P, b_T, b_P = criterion(
            outputs[:, 0:1, :, :],
            targets[:, 0:1, :, :],
            outputs[:, 1:2, :, :],
            targets[:, 1:2, :, :],
        )

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        b_T_accum += b_T.detach().cpu().item()
        b_P_accum += b_P.detach().cpu().item()
        temp_loss += mae_T.detach().cpu().item()
        precip_loss += mae_P.detach().cpu().item()
        train_loss += loss.detach().cpu().item()

    # Average metrics
    num_batches = len(train_loader)
    b_T = b_T_accum / num_batches
    b_P = b_P_accum / num_batches
    temp_loss /= num_batches
    precip_loss /= num_batches
    train_loss /= num_batches

    # --------------------
    # Validation
    # --------------------
    model.eval()
    val_loss = 0.0
    val_temp_loss = 0.0
    val_precip_loss = 0.0
    val_b_T_accum = 0.0
    val_b_P_accum = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss, mae_T, mae_P, b_T, b_P = criterion(
                outputs[:, 0:1, :, :],
                targets[:, 0:1, :, :],
                outputs[:, 1:2, :, :],
                targets[:, 1:2, :, :],
            )

            val_b_T_accum += b_T.detach().cpu().item()
            val_b_P_accum += b_P.detach().cpu().item()
            val_temp_loss += mae_T.detach().cpu().item()
            val_precip_loss += mae_P.detach().cpu().item()
            val_loss += loss.detach().cpu().item()

    num_val_batches = len(val_loader)
    val_b_T = val_b_T_accum / num_val_batches
    val_b_P = val_b_P_accum / num_val_batches
    val_loss /= num_val_batches
    val_temp_loss /= num_val_batches
    val_precip_loss /= num_val_batches

    # Step scheduler
    scheduler.step(val_loss)

    # Free memory
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


# Train loop
def train_model(
    model,
    excluding_cluster,
    num_epochs,
    train_loader,
    val_loader,
    config,
    device,
    save_path,
):
    # Load optimizer/scheduler configs
    if os.path.exists(os.path.join(save_path, "config.yaml")):
        config = yaml.safe_load(open(os.path.join(save_path, "config.yaml"), "r"))

    # Use Laplace homoscedastic loss
    criterion = LaplaceHomoscedasticLoss().to(device)

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        [
            {
                "params": model.parameters(),
                **config["domain_specific"][excluding_cluster]["optimizer_params"],
            },
            {
                "params": [criterion.log_b_T, criterion.log_b_P],
                **config["domain_specific"][excluding_cluster]["loss_params"],
            },
        ]
    )

    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer, **config["training"]["scheduler_params"]
    )

    early_stopping = config["training"]["early_stopping"]
    patience = config["training"]["early_stopping_params"]["patience"]

    list_b_T, list_b_P = [], []
    val_list_b_P, val_list_b_T = [], []
    (
        train_losses,
        train_temp_losses,
        train_precip_losses,
    ) = ([], [], [])
    val_losses, val_temp_losses, val_precip_losses = [], [], []

    # Load checkpoint if exists
    cluster_dir = os.path.join(save_path, excluding_cluster)
    if os.path.exists(os.path.join(cluster_dir, "last_snapshot.pth")):
        LOGGER.info(f"TRAINING: Loading model checkpoint from {cluster_dir} ...")
        checkpoint = torch.load(
            os.path.join(cluster_dir, "last_snapshot.pth"), map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        if os.path.exists(os.path.join(cluster_dir, "train_losses.npy")):
            train_losses = list(np.load(os.path.join(cluster_dir, "train_losses.npy")))
        if os.path.exists(os.path.join(cluster_dir, "val_losses.npy")):
            val_losses = list(np.load(os.path.join(cluster_dir, "val_losses.npy")))
        early_stop_counter_T = checkpoint.get("early_stop_counter_T", 0)
        early_stop_counter_P = checkpoint.get("early_stop_counter_P", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        best_val_temp_loss = checkpoint.get("val_temp_loss", float("inf"))
        best_val_precip_loss = checkpoint.get("val_precip_loss", float("inf"))
        LOGGER.info(f"TRAINING: Resuming training from epoch {start_epoch+1}")
    else:
        LOGGER.info("TRAINING: No checkpoint found, starting fresh training.")
        os.makedirs(os.path.join(save_path, excluding_cluster), exist_ok=True)
        start_epoch = 0
        early_stop_counter_T = 0
        early_stop_counter_P = 0
        best_val_loss = float("inf")
        best_val_temp_loss = float("inf")
        best_val_precip_loss = float("inf")

    log_file = os.path.join(cluster_dir, "training_log.csv")
    if os.path.exists(log_file):
        with open(log_file, "a") as f:
            f.write(f"Resuming training from epoch {start_epoch+1}\n")
    else:
        with open(log_file, "w") as f:
            f.write("Epoch,Train Loss,Validation Loss,Learning Rate,Epoch Time\n")

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
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )
        list_b_T.append(b_T if isinstance(b_T, float) else b_T.detach().cpu().item())
        list_b_P.append(b_P if isinstance(b_P, float) else b_P.detach().cpu().item())
        train_temp_losses.append(temp_loss)
        train_precip_losses.append(precip_loss)
        train_losses.append(train_loss)
        val_list_b_P.append(val_b_P)
        val_list_b_T.append(val_b_T)
        val_temp_losses.append(val_temp_loss)
        val_precip_losses.append(val_precip_loss)
        val_losses.append(val_loss)

        # -------------------------------
        # Early stopping on both MAEs
        # -------------------------------
        if val_temp_loss < best_val_temp_loss:
            best_val_temp_loss = val_temp_loss
            early_stop_counter_T = 0
        else:
            early_stop_counter_T += 1

        if val_precip_loss < best_val_precip_loss:
            best_val_precip_loss = val_precip_loss
            early_stop_counter_P = 0
        else:
            early_stop_counter_P += 1

        # Save best snapshot if either metric improved
        if (
            val_loss < best_val_loss
            or val_temp_loss <= best_val_temp_loss
            or val_precip_loss <= best_val_precip_loss
        ):
            best_val_loss = min(best_val_loss, val_loss)
            snapshot_path = os.path.join(cluster_dir, "best_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_temp_loss": val_temp_loss,
                    "val_precip_loss": val_precip_loss,
                    "early_stop_counter_T": early_stop_counter_T,
                    "early_stop_counter_P": early_stop_counter_P,
                },
                snapshot_path,
            )

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]
        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n"
            )

        # Trigger early stop if either patience exceeded
        if early_stopping and (
            early_stop_counter_T >= patience or early_stop_counter_P >= patience
        ):
            LOGGER.info("TRAINING: Early stopping triggered (MAE stagnation).")
            last_snapshot_path = os.path.join(cluster_dir, "last_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_temp_loss": val_temp_loss,
                    "val_precip_loss": val_precip_loss,
                    "early_stop_counter_T": early_stop_counter_T,
                    "early_stop_counter_P": early_stop_counter_P,
                },
                last_snapshot_path,
            )
            LOGGER.info(f"TRAINING: Last model state saved to {last_snapshot_path}")
            break

        torch.cuda.empty_cache()

    LOGGER.info(f"TRAINING: Training complete! Best model saved as: {snapshot_path}")

    np.save(os.path.join(cluster_dir, "b_T.npy"), np.array(list_b_T))
    np.save(os.path.join(cluster_dir, "b_P.npy"), np.array(list_b_P))
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(
        os.path.join(cluster_dir, "train_temp_losses.npy"), np.array(train_temp_losses)
    )
    np.save(
        os.path.join(cluster_dir, "train_precip_losses.npy"),
        np.array(train_precip_losses),
    )
    np.save(os.path.join(cluster_dir, "val_b_T.npy"), np.array(val_list_b_T))
    np.save(os.path.join(cluster_dir, "val_b_P.npy"), np.array(val_list_b_P))

    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(cluster_dir, "val_temp_losses.npy"), np.array(val_temp_losses))
    np.save(
        os.path.join(cluster_dir, "val_precip_losses.npy"), np.array(val_precip_losses)
    )
