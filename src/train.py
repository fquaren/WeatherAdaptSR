import torch
import os
import numpy as np



def train_model(model, train_loader, val_loader, config, device, save_path):
        
    # Get optimizer, scheduler and criterion  
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_params"])
    scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"])(optimizer, **config["scheduler_params"])
    criterion = getattr(torch.nn, config["criterion"])()

    num_epochs = config["num_epochs"]
    patience = config["patience"]

    model.to(device)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0

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
        
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
        print("Training complete! Best model saved as:", best_model_path)
    
        # Save losses data
        np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_path, "val_losses.npy"), np.array(val_losses))

        return best_model_path
    
    