import torch
import os
import numpy as np
import time
from tqdm import tqdm


def objective(trial, model, num_epochs, train_loader, val_loader, config, device):
    # Hyperparameter optimization
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    optimizer = getattr(torch.optim, config["training"]["optimizer"])(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(optimizer, **config["training"]["scheduler_params"])
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    train_batch_losses, val_batch_losses = [], []
    for n in range(num_epochs):
        _, val_loss, _, _ = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            train_batch_losses=train_batch_losses, 
            val_batch_losses=val_batch_losses
        )

    return val_loss


# Train and val step
def _train_step(model, train_loader, val_loader, optimizer, scheduler, criterion, device, train_batch_losses, val_batch_losses):

    # Training step
    train_loss = 0.0
    for temperature, elevation, target in tqdm(train_loader):
        temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(temperature, elevation)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batch_losses.append(loss.item())
    
    train_loss /= len(train_loader)
    
    # Validation Step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for temperature, elevation, target in val_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output = model(temperature, elevation)
            val_loss += criterion(output, target).item()
            val_batch_losses.append(criterion(output, target).item())
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    return train_loss, val_loss, train_batch_losses, val_batch_losses 


# Train loop
def train_model(model, excluding_cluster, num_epochs, train_loader, val_loader, config, device, save_path):
    
    optimizer = getattr(torch.optim, config["training"]["optimizer"])(model.parameters(), **config["domain_specific"][excluding_cluster]["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(optimizer, **config["training"]["scheduler_params"])
    criterion = getattr(torch.nn, config["training"]["criterion"])()

    patience = config["training"]["early_stopping_params"]["patience"]

    train_batch_losses, val_batch_losses = [], []
    train_losses, val_losses = [], []
    early_stop_counter = 0

    cluster_dir = os.path.join(save_path, excluding_cluster)
    if os.path.exists(os.path.join(cluster_dir, "best_snapshot.pth")):
        print(f"Loading model checkpoint from {cluster_dir} ...")
        checkpoint = torch.load(os.path.join(cluster_dir, "best_snapshot.pth"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_losses = list(np.load(os.path.join(cluster_dir, "train_losses.npy")))
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
        
        train_loss, val_loss, train_batch_losses, val_batch_losses = _train_step(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            train_batch_losses=train_batch_losses,
            val_batch_losses=val_batch_losses
        )

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
                "val_loss": val_loss
            }, snapshot_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")

        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        # current_lr = scheduler.get_last_lr() # TODO: fix this
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

        # Early Stopping
        if config["training"]["early_stopping"] and early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete! Best model saved as:", snapshot_path)

    # Save losses data
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cluster_dir, "train_batch_losses.npy"), np.array(train_batch_losses))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(cluster_dir, "val_batch_losses.npy"), np.array(val_batch_losses))

    return best_val_loss