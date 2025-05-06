import torch
import os
import numpy as np
import time
from tqdm import tqdm


# Train loop
def train_model(model, excluding_cluster, train_loader, val_loader, config, device, save_path):
    
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["early_stopping_params"]["patience"]

    best_val_loss = float("inf")
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
    
    model.to(device)
    
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        for temperature, elevation, target in tqdm(train_loader):
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(temperature, elevation)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for temperature, elevation, target in val_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                output = model(temperature, elevation)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
                
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
        if epoch == 0:
            with open(log_file, "w") as f:
                f.write("Epoch,Train Loss,Validation Loss,Learning Rate,Epoch Time\n")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

        # Early Stopping
        if config["early_stopping"] and early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete! Best model saved as:", snapshot_path)

    # Save losses data
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return