import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import tqdm
from data.dataset import ClimateSRDataset
from deterministic_model import EDSRModel


def main():

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config['DEVICE']}")

    DATA_ROOT = config["DATA_ROOT"]
    DOMAIN_LIST = config["DOMAIN_LIST"]

    for domain in DOMAIN_LIST:
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING FOR DOMAIN: {domain}")
        print(f"{'='*60}\n")

        current_cluster_path = os.path.join(DATA_ROOT, domain)
        if not os.path.exists(current_cluster_path):
            print(f"Warning: Path not found for domain {domain}. Skipping.")
            continue

        stats_filename = f"stats_{domain}.json"
        stats_path = os.path.join(DATA_ROOT, stats_filename)
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            print(f"Loaded domain-specific stats from: {stats_path}")
        except FileNotFoundError:
            print(f"Error: Stats file not found at {stats_path}")
            continue

        print(f"Loading data for {domain}...")
        try:
            train_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path,
                split="train",
                normalization_stats=stats,
                validation_split_pct=0.2,
                seed=42,
            )
            val_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path,
                split="validation",
                normalization_stats=stats,
                validation_split_pct=0.2,
                seed=42,
            )
        except Exception as e:
            print(f"Error loading dataset for {domain}: {e}")
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

        print("Initializing new model and optimizer...")
        model = EDSRModel(
            dynamic_in_channels=9,
            static_in_channels=2,
            out_channels=1,
        )
        model.to(DEVICE)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

        # --- Gradient Scaler for mixed precision ---
        scaler = torch.amp.GradScaler("cuda")

        NUM_EPOCHS = config["NUM_EPOCHS"]
        best_val_loss = float("inf")

        print(f"Starting {NUM_EPOCHS} training epochs for {domain}...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_pbar = tqdm.tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", total=len(train_loader)
            )

            # --- 3-tensor unpacking ---
            for batch_idx, (x_dyn, x_stat, y) in train_pbar:
                x_dyn = x_dyn.to(DEVICE)
                x_stat = x_stat.to(DEVICE)
                y = y.to(DEVICE)

                with torch.amp.autocast(device_type="cuda"):
                    # --- Dual input forward pass ---
                    predictions = model(x_dyn, x_stat)
                    loss = loss_fn(predictions, y)

                optimizer.zero_grad()

                # --- Scaled backward pass ---
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % 50 == 0:
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            total_val_loss = 0
            val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", total=len(val_loader))

            with torch.no_grad():
                # --- 3-tensor unpacking ---
                for x_dyn_val, x_stat_val, y_val in val_pbar:
                    x_dyn_val = x_dyn_val.to(DEVICE)
                    x_stat_val = x_stat_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)

                    with torch.amp.autocast(device_type="cuda"):
                        # --- Dual input forward pass ---
                        preds_val = model(x_dyn_val, x_stat_val)
                        val_loss = loss_fn(preds_val, y_val)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"--- EPOCH {epoch+1} | DOMAIN {domain} | VALIDATION LOSS: {avg_val_loss:.6f} ---")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # --- Accurate naming convention ---
                save_filename = f"edsr_baseline_{domain}.pth"
                save_path = os.path.join(config["EXP_DIR"], "models", "EDSR", save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(model.state_dict(), save_path)
                print(f"New best model for {domain} saved to {save_path} (Loss: {best_val_loss:.6f})")

        print(f"\n--- COMPLETED TRAINING FOR {domain} ---")

        del model, optimizer, scaler, train_dataset, val_dataset, train_loader, val_loader, stats
        torch.cuda.empty_cache()

    print("\nAll domains trained successfully.")


if __name__ == "__main__":
    main()
