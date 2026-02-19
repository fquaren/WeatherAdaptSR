import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import tqdm
import optuna
from data.dataset import ClimateSRDataset
from deterministic_model import EDSRModel

NUM_WORKERS = 48

# Enable TensorFloat-32 for standard fp32 matrix multiplications on hopper
torch.set_float32_matmul_precision('high')

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def run_training_loop(
    domain, config, train_loader, val_loader, stats, device, 
    learning_rate, weight_decay, num_epochs, is_hpo_trial=False, trial=None
):
    model = EDSRModel(dynamic_in_channels=9, static_in_channels=2, out_channels=1)
    model.to(device)
    
    # Just-In-Time Compilation to fuse kernels
    model = torch.compile(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    patience = config.get("PATIENCE", 5)
    min_delta = config.get("MIN_DELTA", 0.0)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float("inf")
    disable_pbar = is_hpo_trial

    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", disable=disable_pbar, leave=False)

        for x_dyn, x_stat, y in train_pbar:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)

            # BFloat16 Mixed Precision for stable optimization
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(x_dyn, x_stat)
                loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not disable_pbar:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", disable=disable_pbar, leave=False)

        with torch.no_grad():
            for x_dyn_val, x_stat_val, y_val in val_pbar:
                x_dyn_val, x_stat_val, y_val = x_dyn_val.to(device), x_stat_val.to(device), y_val.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds_val = model(x_dyn_val, x_stat_val)
                    val_loss = loss_fn(preds_val, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if not disable_pbar:
            print(f"--- EPOCH {epoch+1} | DOMAIN {domain} | VALIDATION LOSS: {avg_val_loss:.6f} ---")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not is_hpo_trial:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                save_filename = f"edsr_baseline_{domain}.pth"
                save_path = os.path.join(config.get("EXP_DIR", current_dir), "models", "EDSR", save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                if not disable_pbar:
                    print(f"New best model for {domain} saved to {save_path} (Loss: {best_val_loss:.6f})")

        if is_hpo_trial and trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopper(avg_val_loss):
            if not disable_pbar:
                print(f"Early stopping triggered for {domain} at epoch {epoch+1}.")
            break

    del model, optimizer, early_stopper
    torch.cuda.empty_cache()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="GH200 Optimized Training Script")
    parser.add_argument("--source", type=str, required=True, help="Target domain for training")
    parser.add_argument("--subset_size", type=int, default=None, help="Limit the number of samples for rapid testing")
    args = parser.parse_args()
    domain = args.source
    subset_size = args.subset_size

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE} on domain: {domain}")
    
    if subset_size:
        print(f"*** RUNNING IN SUBSET MODE: Limited to {subset_size} samples total ***")

    DATA_ROOT = config["DATA_ROOT"]
    current_cluster_path = os.path.join(DATA_ROOT, domain)
    
    if not os.path.exists(current_cluster_path):
        raise FileNotFoundError(f"Path not found for domain {domain}.")

    stats_path = os.path.join(DATA_ROOT, f"stats_{domain}.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)

    print(f"Loading data for {domain}...")
    # Pass the subset_size variable down to the dataset instances
    train_dataset = ClimateSRDataset(cluster_path=current_cluster_path, split="train", normalization_stats=stats, subset_size=subset_size)
    val_dataset = ClimateSRDataset(cluster_path=current_cluster_path, split="validation", normalization_stats=stats, subset_size=subset_size)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n[PHASE 1] Hyperparameter Optimization for {domain}...")
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hpo_epochs = min(config["NUM_EPOCHS"], 5) 
        return run_training_loop(domain, config, train_loader, val_loader, stats, DEVICE, lr, wd, hpo_epochs, is_hpo_trial=True, trial=trial)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=10)

    best_lr = study.best_params["learning_rate"]
    best_wd = study.best_params["weight_decay"]

    print(f"\n[PHASE 2] Final Full Training for {domain}...")
    run_training_loop(domain, config, train_loader, val_loader, stats, DEVICE, best_lr, best_wd, config["NUM_EPOCHS"], is_hpo_trial=False)

    print(f"\n--- COMPLETED PIPELINE FOR {domain} ---")

if __name__ == "__main__":
    main()