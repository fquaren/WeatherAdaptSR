import torch
import os
import numpy as np


# Simple train loop
def train_model(model, train_loader, val_loader, config, device, save_path):
        
    # Get optimizer, scheduler and criterion  
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["patience"]

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0
    
    model.to(device)

    best_model_path = os.path.join(save_path, "best_model.pth")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for temperature, elevation, target in train_loader:
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
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Save losses data
        np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_path, "val_losses.npy"), np.array(val_losses))
        
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
    print("Training complete! Best model saved as:", best_model_path)

    return best_model_path


 # Train loop following step 1 from Häfner et al. 2023
def train_model_step_1(model, train_loaders, val_loaders, config, device, save_path, model_path=None):
    """
    Train a PyTorch model on multiple data clusters: train on all and validate on the single ones.
    
    Args:
        model: PyTorch model instance.
        train_loaders (dict): Dictionary where keys are cluster names and values are train DataLoaders.
        val_loaders (dict): Dictionary where keys are cluster names and values are val DataLoaders.
        config (dict): Training configuration.
        device (str): Device ('cpu' or 'mps' or 'cuda').
        save_path (str): Directory to save the model.
        model_path (str, optional): Path to a saved model checkpoint for resuming training.
    
    Returns:
        str: Path to the best model.
    """
    
    # Get optimizer, scheduler, and criterion  
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["patience"]
    snapshot_interval = config["snapshot_interval"]

    model.to(device)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0

    best_model_path = os.path.join(save_path, "best_model.pth")
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    
    # Load checkpoint if model_path is provided**
    if model_path and os.path.exists(model_path):
        print(f"Loading model checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_losses = list(np.load(os.path.join(save_path, "train_losses.npy")))
        val_losses = list(np.load(os.path.join(save_path, "val_losses.npy")))
        best_val_loss = checkpoint["val_loss"]
        print(f"Resuming training from epoch {start_epoch+1}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Train on all clusters
        for cluster_name, train_loader in train_loaders.items():
            for temperature, elevation, target in train_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(temperature, elevation)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        train_loss /= sum(len(loader) for loader in train_loaders.values())  # Normalize loss
        train_losses.append(train_loss)
        
        # Validation Step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for cluster_name, val_loader in val_loaders.items():
                cluster_val_loss = 0.0
                for temperature, elevation, target in val_loader:
                    temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                    output = model(temperature, elevation)
                    cluster_val_loss += criterion(output, target).item()
                
                cluster_val_loss /= len(val_loader)
                total_val_loss += cluster_val_loss  # Sum across all clusters
        
        val_losses.append(total_val_loss)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Total Val Loss: {total_val_loss:.4f}")
        
        # Save the best model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save model snapshots periodically
        if (epoch + 1) % snapshot_interval == 0:
            snapshot_path = os.path.join(save_path, f"snapshot_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": total_val_loss
            }, snapshot_path)
            print(f"Saved model snapshot: {snapshot_path}")
        
        # Early Stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        # Save losses data
        np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_path, "val_losses.npy"), np.array(val_losses))
    
    print("Training complete! Best model saved as:", best_model_path)

    return best_model_path
