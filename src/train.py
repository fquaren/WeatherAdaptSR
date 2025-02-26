import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from models.unet import UNet8xBaseline
from data.dataset import SingleVariableDataset
from data.dataloader import get_dataloaders

# Load Config
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Define Paths and Datasets
train_dataset = SingleVariableDataset() 
val_dataset = SingleVariableDataset(config["data"]["variable"], ...)
test_dataset = SingleVariableDataset(config["data"]["variable"], ...)
train_loader, val_loader, test_loader = get_dataloaders(
        config["data"]["variable"],
        config["data"]["input_dir"],
        config["data"]["target_dir"],
        config["data"]["elev_file"],
        config["training"]["batch_size"]
    )

val_dataset = SingleVariableDataset(config["data"]["variable"], ...)
val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

# Model
model = UNet8xBaseline(lr=config["training"]["lr"])

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Example loss function, change if needed
optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

# Initialize Lists to Track Losses
train_losses = []
val_losses = []

# Training Loop
num_epochs = config["training"]["num_epochs"]
log_every_n_steps = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Training phase
    for step, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log training loss every n steps
        if step % log_every_n_steps == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

# Save Model
model_path = os.path.join(config["training"]["model_dir"], config["training"]["model_name"])
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save losses to a file (optional)
losses_dict = {'train_losses': train_losses, 'val_losses': val_losses}
losses_path = os.path.join(config["training"]["model_dir"], 'losses.yaml')
with open(losses_path, 'w') as f:
    yaml.dump(losses_dict, f)
print(f"Losses saved to {losses_path}")