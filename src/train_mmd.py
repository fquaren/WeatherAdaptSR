import torch
import os
import numpy as np
import time
from tqdm import tqdm
from itertools import cycle

from data.dataloader import get_single_cluster_dataloader, get_clusters_dataloader


def objective_mmd(trial, model, num_epochs, cluster, cluster_names, config, device, device_data):
    # Hyperparameter optimization
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(optimizer, **config["training"]["scheduler_params"])
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    # Get dataloaders for all clusters excluding the current one
    print(f"Training excluding cluster: {cluster}")
    source_cluster_dataloaders = get_clusters_dataloader(
        data_path=config["paths"]["data_path"],
        elev_dir=config["paths"]["elev_path"],
        excluded_cluster=cluster,
        cluster_names=cluster_names,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        use_theta_e=config["training"]["use_theta_e"],
        device=device_data,
    )
    target_cluster_dataloader = get_single_cluster_dataloader(
        data_path=config["paths"]["data_path"],
        elev_dir=config["paths"]["elev_path"],
        cluster=cluster,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        use_theta_e=config["training"]["use_theta_e"],
        device=device_data,
    )
    source_train_loader = source_cluster_dataloaders["train"]
    source_val_loader = source_cluster_dataloaders["val"]
    target_train_loader = target_cluster_dataloader["train"]

    for n in range(num_epochs):
        lambda_max = config["training"]["mmd"]["lambda_max"]
        lambda_mmd = lambda_max * (n / num_epochs) ** 2
        _, _, val_loss = _train_step_mmd(
            model=model,
            lambda_mmd=lambda_mmd,
            source_train_loader=source_train_loader,
            target_train_loader=target_train_loader,
            source_val_loader=source_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            regression_criterion=criterion,
            device=device,
        )

    return val_loss


# Train and val step
def _train_step_mmd(model, lambda_mmd, source_train_loader, source_val_loader, target_train_loader, optimizer, scheduler, regression_criterion, device):

    # Cycle through target data loader until the source data loader is exhausted
    target_iter = cycle(target_train_loader)

    # Training step
    train_loss = 0.0
    train_mmd_loss = 0.0

    for temperature, elevation, target in tqdm(source_train_loader):

        temperature_t, elevation_t, target_t = next(target_iter)

        temperature = temperature.to(device)
        elevation = elevation.to(device)
        target = target.to(device)
        temperature_t = temperature_t.to(device)
        elevation_t = elevation_t.to(device)
        target_t = target_t.to(device)

        optimizer.zero_grad()

        output, mmd_loss = model(temperature, elevation, target_variable=temperature_t, target_elevation=elevation_t)
        regression_loss = regression_criterion(output, target)

        loss = regression_loss + lambda_mmd * mmd_loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mmd_loss += mmd_loss.item()
        del output, loss, mmd_loss  # Free memory
    
    train_loss /= len(source_train_loader)
    train_mmd_loss /= len(source_train_loader)
    
    # Validation Step (regression only on source data)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for temperature, elevation, target in source_val_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output, _ = model(temperature, elevation)
            val_loss += regression_criterion(output, target).item()
            del output

    val_loss /= len(source_val_loader)
    
    scheduler.step(val_loss)

    torch.cuda.empty_cache()

    return train_loss, train_mmd_loss, val_loss


# Train loop
def train_model_mmd(model, excluding_cluster, num_epochs, source_train_loader, target_train_loader, source_val_loader, config, device, save_path):
    
    optimizer = getattr(torch.optim, config["training"]["optimizer"])(model.parameters(), **config["domain_specific"][excluding_cluster]["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(optimizer, **config["training"]["scheduler_params"])
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    patience = config["training"]["early_stopping_params"]["patience"]
    lambda_max = config["training"]["mmd"]["lambda_max"]

    train_losses, train_mmd_losses, val_losses = [], [], []
    early_stop_counter = 0

    cluster_dir = os.path.join(save_path, excluding_cluster)
    if os.path.exists(os.path.join(cluster_dir, "best_snapshot.pth")):
        print(f"Loading model checkpoint from {cluster_dir} ...")
        checkpoint = torch.load(os.path.join(cluster_dir, "best_snapshot.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        lambda_mmd = checkpoint["lambda_mmd"]
        train_losses = list(np.load(os.path.join(cluster_dir, "train_losses.npy")))
        train_mmd_losses = list(np.load(os.path.join(cluster_dir, "train_mmd_losses.npy")))
        val_losses = list(np.load(os.path.join(cluster_dir, "val_losses.npy")))
        best_val_loss = checkpoint["val_loss"]
        print(f"Resuming training from epoch {start_epoch+1}")
    else:
        print("No checkpoint found, starting fresh training.")
        os.makedirs(os.path.join(save_path, excluding_cluster), exist_ok=True)
        start_epoch = 0
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
    for epoch in tqdm(range(num_epochs)):

        model.train()
        epoch_start_time = time.time()

        lambda_mmd = lambda_max * (epoch / num_epochs) ** 2

        train_loss, train_mmd_loss, val_loss = _train_step_mmd(
            model=model,
            lambda_mmd=lambda_mmd,
            source_train_loader=source_train_loader,
            target_train_loader=target_train_loader,
            source_val_loader=source_val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            regression_criterion=criterion,
            device=device,
        )
        train_losses.append(train_loss)
        train_mmd_losses.append(train_mmd_loss)
        val_losses.append(val_loss)

        # Save only the latest best model snapshot
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            snapshot_path = os.path.join(cluster_dir, f"best_snapshot.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_mmd_loss": train_mmd_loss,
                "lambda_mmd": lambda_mmd,
                "val_loss": val_loss
            }, snapshot_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch}, Train loss: {train_loss}, Train MMD loss: {train_mmd_loss}, Lambda MMD: {lambda_mmd}, Val loss: {val_loss}")

        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        # current_lr = scheduler.get_last_lr() # TODO: fix this
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_mmd_loss:.6f},{lambda_mmd:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

        # Early Stopping
        if config["training"]["early_stopping"] and early_stop_counter >= patience:
            print("Early stopping triggered.")
            # Save the last model state
            last_snapshot_path = os.path.join(cluster_dir, f"last_snapshot.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            }, last_snapshot_path)
            print(f"Last model state saved to {last_snapshot_path}")
            break

        torch.cuda.empty_cache()

    print("Training complete! Best model saved as:", snapshot_path)

    # Save losses data
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cluster_dir, "train_mmd_losses.npy"), np.array(train_mmd_losses))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return best_val_loss