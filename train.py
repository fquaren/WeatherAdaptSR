import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import tqdm
import optuna
from optuna.trial import TrialState
import argparse

from data.dataset import ClimateSRDataset
from deterministic_model import EDSRModel



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
    """
    Abstracted training loop to be utilized by both the Optuna objective function 
    and the final deterministic training run.
    """
    model = EDSRModel(
        dynamic_in_channels=9,
        static_in_channels=2,
        out_channels=1,
    )
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    
    patience = config.get("PATIENCE", 5)
    min_delta = config.get("MIN_DELTA", 0.0)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    best_val_loss = float("inf")

    # Disable tqdm progress bars during hyperparameter optimization to keep terminal clean
    disable_pbar = is_hpo_trial

    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm.tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
            disable=disable_pbar,
            leave=False
        )

        for x_dyn, x_stat, y in train_pbar:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type="cuda"):
                predictions = model(x_dyn, x_stat)
                loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not disable_pbar:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm.tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
            disable=disable_pbar,
            leave=False
        )

        with torch.no_grad():
            for x_dyn_val, x_stat_val, y_val in val_pbar:
                x_dyn_val = x_dyn_val.to(device)
                x_stat_val = x_stat_val.to(device)
                y_val = y_val.to(device)

                with torch.amp.autocast(device_type="cuda"):
                    preds_val = model(x_dyn_val, x_stat_val)
                    val_loss = loss_fn(preds_val, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if not disable_pbar:
            print(f"--- EPOCH {epoch+1} | DOMAIN {domain} | VALIDATION LOSS: {avg_val_loss:.6f} ---")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Only save weights if this is the final training run, not during an HPO trial
            if not is_hpo_trial:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                save_filename = f"edsr_baseline_{domain}.pth"
                save_path = os.path.join(config.get("EXP_DIR", current_dir), "models", "EDSR", save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"New best model for {domain} saved to {save_path} (Loss: {best_val_loss:.6f})")

        # Optuna pruning logic
        if is_hpo_trial and trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopper(avg_val_loss):
            if not disable_pbar:
                print(f"Early stopping triggered for {domain} at epoch {epoch+1}.")
            break

    # Clean up local references
    del model, optimizer, scaler, early_stopper
    torch.cuda.empty_cache()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Training Script for Weather Adaptation Super-Resolution")
    parser.add_argument("--subset_size", type=int, default=None, help="Limit the number of samples for rapid testing")
    parser.add_argument("--source", type=str, required=True, help="Target domain for training")
    args = parser.parse_args()
    
    subset_size = args.subset_size
    if subset_size:
        print(f"*** RUNNING IN SUBSET MODE: Limited to {subset_size} samples total ***")

    source_domain = args.source

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config['DEVICE']}")

    DATA_ROOT = config["DATA_ROOT"]

    for domain in [source_domain]:
        print(f"\n{'='*60}")
        print(f"STARTING PIPELINE FOR DOMAIN: {domain}")
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
        except FileNotFoundError:
            print(f"Error: Stats file not found at {stats_path}")
            continue

        print(f"Loading data for {domain}...")
        try:
            train_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path, split="train", normalization_stats=stats, subset_size=subset_size
            )
            val_dataset = ClimateSRDataset(
                cluster_path=current_cluster_path, split="validation", normalization_stats=stats, subset_size=subset_size
            )
        except Exception as e:
            print(f"Error loading dataset for {domain}: {e}")
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, 
            num_workers=config.get("NUM_WORKERS", 0), pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, 
            num_workers=config.get("NUM_WORKERS", 0), pin_memory=True
        )

        # --- PHASE 1: Hyperparameter Optimization ---
        print(f"\n[PHASE 1] Running Hyperparameter Optimization for {domain}...")
        
        def objective(trial):
            # Define search space
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            
            # Run a shortened training cycle for evaluation (e.g., max 5 epochs)
            hpo_epochs = min(config["NUM_EPOCHS"], 5) 
            
            val_loss = run_training_loop(
                domain, config, train_loader, val_loader, stats, DEVICE, 
                learning_rate=lr, weight_decay=wd, num_epochs=hpo_epochs, 
                is_hpo_trial=True, trial=trial
            )
            return val_loss

        # Use median stopping rule to prune unpromising trials early
        study = optuna.create_study(
            direction="minimize", 
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )
        study.optimize(objective, n_trials=10) # Adjust n_trials based on compute budget

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        
        print(f"Study statistics: {len(pruned_trials)} pruned, {len(complete_trials)} complete.")
        print(f"Best trial parameters for {domain}: {study.best_params}")
        
        best_lr = study.best_params["learning_rate"]
        best_wd = study.best_params["weight_decay"]

        # --- PHASE 2: Final Full Training ---
        print(f"\n[PHASE 2] Starting Final Full Training for {domain} with optimized parameters...")
        final_val_loss = run_training_loop(
            domain, config, train_loader, val_loader, stats, DEVICE, 
            learning_rate=best_lr, weight_decay=best_wd, num_epochs=config["NUM_EPOCHS"], 
            is_hpo_trial=False
        )

        print(f"\n--- COMPLETED PIPELINE FOR {domain} | FINAL LOSS: {final_val_loss:.6f} ---")

        del train_dataset, val_dataset, train_loader, val_loader, stats
        torch.cuda.empty_cache()

    print("\nAll domains processed successfully.")

if __name__ == "__main__":
    main()