import torch
import os
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


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
    tensorboard_path = os.path.join(cluster_dir, "tensorboard_logs")
    writer = SummaryWriter(tensorboard_path)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
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
        scheduler.step(val_loss)
                
        # Save only the latest best model snapshot
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            snapshot_path = os.path.join(save_path, f"best_snapshot.pth")
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

        # Logging
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0:
            with open(log_file, "w") as f:
                f.write("Epoch,Train Loss,Validation Loss,Learning Rate,Epoch Time\n")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

        writer.add_scalar("Epoch Time", epoch_time, epoch + 1)
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Learning Rate", current_lr, epoch + 1)
        writer.add_scalar("Best Validation Loss", best_val_loss, epoch + 1)

        # Early Stopping
        if config["early_stopping"] and early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete! Best model saved as:", snapshot_path)
    writer.close()

    # Save losses data
    np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return snapshot_path


def freeze_UNet8x_encoder(model):
    for param in model.encoder1.parameters():
        param.requires_grad = False
    for param in model.encoder2.parameters():
        param.requires_grad = False
    for param in model.encoder3.parameters():
        param.requires_grad = False
    for param in model.downsample_elevation.parameters():
        param.requires_grad = False
    return model


# Finetuning train loop
def finetune_model(model, train_loader, val_loader, config, device, save_path, model_path=None, model_name=None, fine_tuning=False):
        
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["early_stopping_params"]["patience"]
    snapshot_interval = config["snapshot_interval"]

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0

    if model_path and os.path.exists(os.path.join(model_path, model_name)):
        print(f"Loading model checkpoint from {model_path}...")
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if fine_tuning:
        model = freeze_UNet8x_encoder(model)  # Note: this is model specific!
        optimizer = getattr(torch.optim, config["optimizer"])(filter(lambda p: p.requires_grad, model.parameters()), **config["optimizer_params"])
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    else:
        optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    
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
        scheduler.step(val_loss)
                
        # Save the best model
        if config["early_stopping"] and val_loss < best_val_loss:
            best_val_loss = val_loss
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
                "val_loss": val_loss
            }, snapshot_path)
            print(f"Saved model snapshot: {snapshot_path}")
        
        # Early Stopping
        if config["early_stopping"] and early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        # Save losses data
        np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_path, "val_losses.npy"), np.array(val_losses))
        
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
    print("Training complete! Best model saved as:", best_model_path)

    return best_model_path


 # Train loop following step 1 from Häfner et al. 2023
def train_model_step_1(model, train_loaders, val_loaders, config, device, save_path, model_path, model_name):
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
        model_name (str, optional): Name of saved model checkpoint for resuming training.
    
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

    train_losses, val_losses = [], []
    early_stop_counter = 0

    best_model_path = os.path.join(save_path, "best_model.pth")
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    
    # Load checkpoint if model_path is provided**
    if model_path and os.path.exists(os.path.join(model_path, model_name)):
        print(f"Loading model checkpoint from {model_path}...")
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_losses = list(np.load(os.path.join(model_path, "train_losses.npy")))
        val_losses = list(np.load(os.path.join(model_path, "val_losses.npy")))
        best_val_loss = checkpoint["val_loss"]
        print(f"Resuming training from epoch {start_epoch+1}")
    else:
        start_epoch = 0
        best_val_loss = float("inf")
    
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
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


 # Train loop following step 2 from Häfner et al. 2023
def train_model_step_2(model, train_loaders, val_loaders, config, device, save_path, model_path):

    save_path_step_2 = os.path.join(save_path, "finetuning")
    os.makedirs(save_path_step_2_cluster, exist_ok=True)

    # Get optimizer, scheduler, and criterion  
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["patience"]
    snapshot_interval = config["snapshot_interval"]

    train_losses, val_losses = {}, {}
    early_stop_counter = 0

    for (cluster_name, train_loader), (_, val_loader) in zip(train_loaders.items(), val_loaders.items()):

        save_path_step_2_cluster = os.path.join(save_path_step_2, cluster_name)
        best_model_path = os.path.join(save_path_step_2_cluster, "best_model.pth")
        os.makedirs(save_path_step_2_cluster, exist_ok=True)

        # Load best model from pretrain
        checkpoint = torch.load(os.path.join(model_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        model.to(device)
        model.train()
        
        train_loss = 0.0
        best_val_loss = float("inf")

        for epoch in range(0, num_epochs):
            for temperature, elevation, target in train_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(temperature, elevation)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= sum(len(loader) for loader in train_loaders.values())  # Normalize loss
            train_losses[cluster_name].append(train_loss)

            # Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for temperature, elevation, target in val_loader:
                    temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                    output = model(temperature, elevation)
                    val_loss += criterion(output, target).item()
            
            val_loss /= len(val_loader)
            val_losses[cluster_name].append(val_loss)
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Save model snapshots periodically
            if (epoch + 1) % snapshot_interval == 0:
                snapshot_path = os.path.join(save_path_step_2_cluster, cluster_name, f"snapshot_epoch_{cluster_name}_{epoch+1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, snapshot_path)
                print(f"Saved model snapshot: {snapshot_path}")
            
            # Early Stopping
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

            # Save losses data
            np.save(os.path.join(save_path_step_2_cluster, cluster_name, "train_losses.npy"), np.array(train_losses))
            np.save(os.path.join(save_path_step_2_cluster, cluster_name, "val_losses.npy"), np.array(val_losses))

    return 0