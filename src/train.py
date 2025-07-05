import torch
import os
import numpy as np
import time
from tqdm import tqdm
import gc
import yaml
import logging

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
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer, **config["training"]["scheduler_params"]
    )
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    if single_cluster:
        LOGGER.info(f"OPTIMIZATION: Training on single cluster {cluster} ...")
        cluster_dataloaders = get_single_cluster_dataloader(
            data_path=config["paths"]["data_path"],
            elev_dir=config["paths"]["elev_path"],
            cluster_name=cluster,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            use_theta_e=config["training"]["use_theta_e"],
            device=device_data,
            augment=augment,
        )
        train_loader = cluster_dataloaders["train"]
        val_loader = cluster_dataloaders["val"]
    else:
        LOGGER.info(f"OPTIMIZATION: Training excluding cluster {cluster} ...")
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

    for _ in range(num_epochs):
        _, val_loss = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )

    # Empty GPU memory
    LOGGER.info("OPTIMIZATION: Emptying GPU memory ...")
    for split in ["train", "val", "test"]:
        dataset = cluster_dataloaders[split].dataset
        if hasattr(dataset, "datasets"):  # likely a ConcatDataset
            for d in dataset.datasets:
                d.unload_from_gpu()
        else:
            dataset.unload_from_gpu()
    del train_loader, val_loader, cluster_dataloaders, model
    torch.cuda.empty_cache()
    gc.collect()
    LOGGER.info("OPTIMIZATION: GPU memory emptied ...")

    return val_loss


# Train and val step
def _train_step(
    model, train_loader, val_loader, optimizer, scheduler, criterion, device
):

    # Training step
    train_loss = 0.0
    for temperature, elevation, target in tqdm(train_loader):
        temperature = temperature.to(device)
        elevation = elevation.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(temperature, elevation)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
        del output, loss  # Free memory

    train_loss /= len(train_loader)

    # Validation Step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for temperature, elevation, target in val_loader:
            temperature, elevation, target = (
                temperature.to(device),
                elevation.to(device),
                target.to(device),
            )
            output = model(temperature, elevation)
            val_loss += criterion(output, target).detach().item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    torch.cuda.empty_cache()

    return train_loss, val_loss


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

    # Load optimizer and scheduler model and domain specific configs
    if os.path.exists(os.path.join(save_path, "config.yaml")):
        config = yaml.safe_load(open(os.path.join(save_path, "config.yaml"), "r"))

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(),
        **config["domain_specific"][excluding_cluster]["optimizer_params"],
    )
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(
        optimizer, **config["training"]["scheduler_params"]
    )
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    early_stopping = config["training"]["early_stopping"]
    patience = config["training"]["early_stopping_params"]["patience"]
    rolling_mean_threshold = config["training"]["early_stopping_params"][
        "rolling_mean_threshold"
    ]
    rolling_mean_window = config["training"]["early_stopping_params"][
        "rolling_mean_window"
    ]

    train_losses, val_losses = [], []

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
        early_stop_counter = checkpoint.get("early_stop_counter", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        LOGGER.info(f"TRAINING: Resuming training from epoch {start_epoch+1}")
    else:
        LOGGER.info("TRAINING: No checkpoint found, starting fresh training.")
        os.makedirs(os.path.join(save_path, excluding_cluster), exist_ok=True)
        start_epoch = 0
        early_stop_counter = 0
        best_val_loss = float("inf")

    # Logging
    log_file = os.path.join(cluster_dir, "training_log.csv")
    if os.path.exists(log_file):
        with open(log_file, "a") as f:
            f.write(f"Resuming training from epoch {start_epoch+1}\n")
    else:
        with open(log_file, "w") as f:
            f.write("Epoch,Train Loss,Validation Loss,Learning Rate,Epoch Time\n")

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs)):

        model.train()
        epoch_start_time = time.time()
        train_loss, val_loss = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Compute the rolling average of the val loss
        if rolling_mean_threshold is not None:
            if len(val_losses) > rolling_mean_window:
                rolling_mean_val_loss = np.mean(val_losses[-rolling_mean_window:])
            else:
                rolling_mean_val_loss = np.mean(val_losses)
            if rolling_mean_val_loss < rolling_mean_threshold:
                early_stop_counter += 1

        # Save the latest best model snapshot
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            snapshot_path = os.path.join(cluster_dir, "best_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "early_stop_counter": early_stop_counter,
                },
                snapshot_path,
            )
            early_stop_counter = 0
        else:
            # Stop when n=patience consecutive epochs < best_val_loss
            early_stop_counter += 1

        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]
        # current_lr = scheduler.get_last_lr() # TODO: fix this
        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n"
            )

        # Early Stopping
        if early_stopping and early_stop_counter >= patience:
            LOGGER.info("TRAINING: Early stopping triggered.")
            # Save the last model state
            last_snapshot_path = os.path.join(cluster_dir, "last_snapshot.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "early_stop_counter": early_stop_counter,
                },
                last_snapshot_path,
            )
            LOGGER.info(f"TRAINING: Last model state saved to {last_snapshot_path}")
            break

        torch.cuda.empty_cache()

    LOGGER.info(f"TRAINING: Training complete! Best model saved as: {snapshot_path}")

    # Save losses data
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return
