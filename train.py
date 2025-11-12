import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import tqdm
from data.dataset import ClimateSRDataset
from model import UNet


def main():
    # Get path current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config_path = os.path.join(current_dir, "configs/config_rainshift.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 1. Configuration ---
    DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config['DEVICE']}")

    # Define domains
    DATA_ROOT = config["DATA_ROOT"]

    # Load the list of all domains to train on
    DOMAIN_LIST = config["DOMAIN_LIST"]

    # --- 2. Main Training Loop ---
    for domain in DOMAIN_LIST:
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING FOR DOMAIN: {domain}")
        print(f"{'='*60}\n")

        current_cluster_path = os.path.join(DATA_ROOT, domain)
        if not os.path.exists(current_cluster_path):
            print(f"Warning: Path not found for domain {domain}. Skipping.")
            continue

        # --- Load Per-Domain Stats ---
        # Load the domain-specific normalization stats file
        stats_filename = f"stats_{domain}.json"
        stats_path = os.path.join(DATA_ROOT, stats_filename)
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            print(f"Loaded domain-specific stats from: {stats_path}")
        except FileNotFoundError:
            print(f"Error: Stats file not found at {stats_path}")
            print(f"Please run compute_stats.py first for all domains.")
            print(f"Skipping domain {domain}.")
            continue

        # --- 2a. DataLoaders (Instantiated per-domain) ---
        print(f"Loading data for {domain}...")
        try:
            train_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path,
                split="train",
                normalization_stats=stats,  # Pass the domain-specific stats
                validation_split_pct=0.2,
                seed=42,
            )
            val_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path,
                split="validation",
                normalization_stats=stats,  # Pass the domain-specific stats
                validation_split_pct=0.2,
                seed=42,
            )
        except Exception as e:
            print(f"Error loading dataset for {domain}: {e}")
            print(f"Skipping domain {domain}.")
            continue

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            num_workers=config.get("NUM_WORKERS", 0),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=config.get("NUM_WORKERS", 0),
            pin_memory=True,
        )
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # --- 2b. Model, Loss, and Optimizer (Instantiated per-domain) ---
        print("Initializing new model and optimizer...")
        # Assuming 11 input channels (9 dynamic + 2 static)
        # This must match the channels in your stats file
        model = UNet(
            in_channels=11,
            out_channels=1,
        )
        model.to(DEVICE)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

        # --- 2c. Training Loop (Per-domain) ---
        NUM_EPOCHS = config["NUM_EPOCHS"]
        best_val_loss = float("inf")

        print(f"Starting {NUM_EPOCHS} training epochs for {domain}...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_pbar = tqdm.tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", total=len(train_loader)
            )

            for batch_idx, (x, y) in train_pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)

                with torch.amp.autocast(device_type="cuda"):
                    predictions = model(x)
                    loss = loss_fn(predictions, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- 2d. Validation Loop (Per-domain) ---
            model.eval()
            total_val_loss = 0
            val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", total=len(val_loader))

            with torch.no_grad():
                for x_val, y_val in val_pbar:
                    x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                    with torch.amp.autocast(device_type="cuda"):
                        preds_val = model(x_val)
                        val_loss = loss_fn(preds_val, y_val)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"--- EPOCH {epoch+1} | DOMAIN {domain} | VALIDATION LOSS: {avg_val_loss:.6f} ---")

            # --- 2e. Save Model (Per-domain) ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_filename = f"unet_baseline_{domain}.pth"
                save_path = os.path.join(current_dir, "models", save_filename)  # Save to a 'models' subfolder
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(model.state_dict(), save_path)
                print(f"New best model for {domain} saved to {save_path} (Loss: {best_val_loss:.6f})")

        print(f"\n--- COMPLETED TRAINING FOR {domain} ---")

        # Clean up memory before next loop
        del model, optimizer, train_dataset, val_dataset, train_loader, val_loader, stats
        torch.cuda.empty_cache()

    print("\nAll domains trained successfully.")


if __name__ == "__main__":
    main()
