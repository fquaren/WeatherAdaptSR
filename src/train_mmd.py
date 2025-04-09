import torch
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


# Train loop
def train_model_mmd(model, dataloaders, config, device, save_path):
    
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    
    regression_criterion = getattr(torch.nn, config["criterion"])()
    lambda_max = config["domain_adaptation"]["lambda_max"]

    num_epochs = config["num_epochs"]
    patience = config["early_stopping_params"]["patience"]

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0

    for cluster_name, _ in dataloaders.items():
        target_cluster = cluster_name
        cluster_dir = os.path.join(save_path, target_cluster)
        log_file = os.path.join(cluster_dir, "training_log.csv")
        checkpoint_path = os.path(cluster_dir, "best_checkpoint.pth")
        
        # Check if you have to resume experiment
        if os.path.isdir(cluster_dir):
            print(f"Found {cluster_name} ...")
            if os.path.isfile(checkpoint_path) and os.path.isfile(log_file):                
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                start_epoch = checkpoint.get("epoch", None)
                if start_epoch < num_epochs:
                    print(f"Resuming training for {cluster_name} starting from {start_epoch}")
                    df = pd.read_csv(log_file)
                    train_losses_list = df[df["Epoch"] <= start_epoch]["Train Loss"].to_numpy()
                    val_losses_list = df[df["Epoch"] <= start_epoch]["Validation Loss"].to_numpy()
                    train_losses.extend(train_losses_list)
                    val_losses.extend(val_losses_list)
                    best_val_loss = val_losses_list[-1]
                    model.load_state_dict(checkpoint["model_state_dict"])
        else:
            os.makedirs(cluster_dir, exist_ok=True)
            start_epoch = 0

        # Separate source and target loaders
        train_source_loaders = []
        val_source_loaders = []
        for cluster_name, loaders in dataloaders.items():
            if cluster_name == target_cluster:
                train_target_loader = loaders["train"]
                val_target_loader = loaders["val"]
            else:
                train_source_loaders.append(loaders["train"])
                val_source_loaders.append(loaders["val"])

        print(f"Training with {target_cluster} as target domain...")
        for epoch in tqdm(range(start_epoch, num_epochs)):
            epoch_start_time = time.time()
            model.train()

            # Gradually increase lambda_mmd(t) during training  with Annealing
            lambda_mmd = lambda_max * (epoch / num_epochs) ** 2
            
            # Training Step
            train_losses_loaders = []
            for j, train_source_loader in enumerate(train_source_loaders):
                train_loss = 0.0
                # Note: zip() iterates over the shortest of the two data loaders
                for k, ((sx, selev, sy), (tx, telev, ty)) in enumerate(zip(train_source_loader, train_target_loader)):
                    temperature, elevation, target = sx.to(device), selev.to(device), sy.to(device)
                    temperature_t, elevation_t = tx.to(device), telev.to(device)

                    optimizer.zero_grad()

                    # Forward pass for source data
                    output, mmd_loss = model(temperature, elevation, target_variable=temperature_t, target_elevation=elevation_t)                    
                    regression_loss = regression_criterion(output, target)

                    loss = regression_loss + lambda_mmd * mmd_loss

                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    # if k == 1:
                    #     break

                train_loss /= len(train_source_loader)
                train_losses_loaders.append(train_loss)
                print(f"Iteration {j}/{len(train_source_loaders)}, Train loss: {train_loss}")
            # Track mean train loss on all domains
            mean_train_loss = np.mean(train_losses_loaders)
            train_losses.append(mean_train_loss)

            # Validation step (only regression on target domain)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for k, (tx, telev, ty) in enumerate(val_target_loader):
                    temperature, elevation, target = tx.to(device), telev.to(device), ty.to(device)

                    # Forward pass for target data
                    output, _ = model(temperature, elevation)
                    loss = regression_criterion(output, target)

                    val_loss += loss.item()

                    # if k == 1:
                    #     break
                
                val_loss /= len(val_target_loader)
                scheduler.step(val_loss)
                val_losses.append(val_loss)
                print(f"Validation, Val loss: {val_loss}")

            # Save only the latest best model snapshot
            # TODO: I might want to save also intermediate snapshots, this presumes that the best is the last
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": mean_train_loss,
                    "val_loss": val_loss
                }, checkpoint_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Logging
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']  # TODO: fix get_last_lr()
            if epoch == 0:
                with open(log_file, "w") as f:
                    f.write("Epoch,Train Loss,Validation Loss,Learning Rate,Epoch Time\n")
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{mean_train_loss:.6f},{val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

            # Early Stopping
            if config["early_stopping"] and early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
        
        print(f"Training with {target_cluster} as target domain complete, best model saved as:", checkpoint_path)

        # Save losses data
        np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return