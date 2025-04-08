import torch
import os
import numpy as np
import time
from tqdm import tqdm


# Train loop
def train_model_mdan(model, dataloaders, config, device, save_path):
    
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    
    regression_criterion = getattr(torch.nn, config["criterion"])()
    domain_criterion = getattr(torch.nn, config["domain_adaptation"]["mdan_criterion"])()
    mu = config["domain_adaptation"]["mu"]
    gamma = config["domain_adaptation"]["gamma"]

    num_epochs = config["num_epochs"]
    patience = config["early_stopping_params"]["patience"]

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0

    for cluster_name, _ in dataloaders.items():
        target_cluster = cluster_name
        cluster_dir = os.path.join(save_path, target_cluster)
        os.makedirs(cluster_dir, exist_ok=True)
        # Logging
        log_file = os.path.join(cluster_dir, "training_log.csv")
        
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
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.time()
            model.train()
            
            # Training Step
            train_losses = []
            for j, train_source_loader in enumerate(train_source_loaders):
                train_loss = 0.0
                # Note: zip() iterates over the shortest of the two data loaders
                for (sx, selev, sy), (tx, telev, ty) in zip(train_source_loader, train_target_loader):
                    temperature, elevation, target = sx.to(device), selev.to(device), sy.to(device)
                    temperature_t, elevation_t = tx.to(device), telev.to(device)

                    optimizer.zero_grad()

                    # Forward pass for source data
                    output, s_domain_pred = model(temperature, elevation, domain_idx=j)                    
                    regression_loss = regression_criterion(output, target)
                    source_labels = torch.ones(s_domain_pred.shape[0], dtype=torch.long, device=device)  # Class 1 for source
                    s_domain_loss = domain_criterion(s_domain_pred, source_labels)  # Source label = 1

                    # Forward pass for target data (only domain classifiers)
                    _, t_domain_pred = model(temperature_t, elevation_t, domain_idx=j)
                    target_labels = torch.zeros(t_domain_pred.shape[0], dtype=torch.long, device=device)  # Class 0 for target
                    t_domain_loss = domain_criterion(t_domain_pred, target_labels)  # Target label = 0

                    # Compute final loss
                    domain_loss = s_domain_loss + t_domain_loss
                    # Note: If gamma is large, the exponentiation \exp(\gamma x) can explode to a very large number, causing an inf (overflow).
                    # => use PyTorch’s built-in stable LSE function
                    loss = torch.logsumexp(gamma * (regression_loss + mu * domain_loss), dim=0) / gamma

                    train_loss += loss.item()
                    loss.backward()
                    # Clipping gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss /= len(train_source_loader)
                train_losses.append(train_loss)
                print(f"Iteration {j}/{len(train_source_loaders)}, Train loss: {train_loss}")

                # Validation step (only regression on target domain)
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for tx, telev, ty in val_target_loader:
                        temperature, elevation, target = tx.to(device), telev.to(device), ty.to(device)

                        # Forward pass for target data
                        output, _ = model(temperature, elevation, domain_idx=j)
                        val_loss += regression_criterion(output, target).item()
                    
                    val_loss /= len(val_target_loader)
                    scheduler.step(val_loss)
                    val_losses.append(val_loss)

                print(f"Validation, Val loss: {val_loss}")
            
            # Track mean train and val loss on all domains
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)

            # Save only the latest best model snapshot
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                snapshot_path = os.path.join(cluster_dir, f"best_snapshot.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": mean_train_loss,
                    "val_loss": mean_val_loss
                }, snapshot_path)
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
                f.write(f"{epoch+1},{mean_train_loss:.6f},{mean_val_loss:.6f},{current_lr:.6e},{epoch_time:.2f}\n")

            # Early Stopping
            if config["early_stopping"] and early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
        
        print(f"Training with {target_cluster} as target domain complete, best model saved as:", snapshot_path)

        # Save losses data
        np.save(os.path.join(cluster_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(cluster_dir, "val_losses.npy"), np.array(val_losses))

    return