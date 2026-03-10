import os
import json
import copy
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import tqdm
import optuna

from data.dataset import ClimateSRDataset
from deterministic_unet import DeterministicDualEncoderUNet
from consistency_model import ConsistencyModel

torch.set_float32_matmul_precision("high")


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


def update_ema(target_model, online_model, decay):
    with torch.no_grad():
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)


def sample_timesteps(n_samples, min_t, max_t, device, rho=7.0):
    u = torch.rand(n_samples, device=device)
    t = (min_t ** (1 / rho) + u * (max_t ** (1 / rho) - min_t ** (1 / rho))) ** rho
    return t


def run_deterministic_loop(
    domain, config, train_loader, val_loader, device, lr, wd, num_epochs, is_hpo=False, trial=None
):
    model = DeterministicDualEncoderUNet(target_channels=1, dynamic_channels=9, static_channels=2, base_dim=64)
    model.to(device)
    model = torch.compile(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    early_stopper = EarlyStopper(patience=config.get("PATIENCE", 5), min_delta=config.get("MIN_DELTA", 0.0))

    best_val_loss = float("inf")
    disable_pbar = is_hpo

    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=disable_pbar, leave=False)

        for x_dyn, x_stat, y in train_pbar:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x_dyn, x_stat), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_dyn_val, x_stat_val, y_val in val_loader:
                x_dyn_val, x_stat_val, y_val = x_dyn_val.to(device), x_stat_val.to(device), y_val.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    total_val_loss += loss_fn(model(x_dyn_val, x_stat_val), y_val).item()

        avg_val_loss = total_val_loss / len(val_loader)
        if not disable_pbar:
            print(f"--- EPOCH {epoch+1} | VALIDATION LOSS: {avg_val_loss:.6f} ---")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not is_hpo:
                save_path = os.path.join(config.get("EXP_DIR", "."), "models", "unet", f"unet_baseline_{domain}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if is_hpo and trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopper(avg_val_loss):
            break

    del model, optimizer
    torch.cuda.empty_cache()
    return best_val_loss


def run_consistency_loop(
    domain, config, train_loader, val_loader, device, lr, wd, num_epochs, is_hpo=False, trial=None
):
    online_model = ConsistencyModel().to(device)
    target_model = copy.deepcopy(online_model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    online_model = torch.compile(online_model)
    target_model = torch.compile(target_model)

    optimizer = optim.Adam(online_model.parameters(), lr=lr, weight_decay=wd)
    early_stopper = EarlyStopper(patience=config.get("PATIENCE", 5), min_delta=config.get("MIN_DELTA", 0.0))

    min_t, max_t = 0.002, 80.0
    ema_initial, ema_final = 0.95, 0.999
    total_steps = len(train_loader) * num_epochs
    step = 0
    best_val_loss = float("inf")
    disable_pbar = is_hpo

    for epoch in range(num_epochs):
        online_model.train()
        epoch_loss = 0.0

        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=disable_pbar, leave=False)
        for x_dyn, x_stat, y in train_pbar:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            cond = torch.cat([x_dyn, x_stat], dim=1)
            b = y.shape[0]

            t2 = sample_timesteps(b, min_t, max_t, device)
            t1 = torch.clamp(t2 - (max_t - min_t) / 100.0, min=min_t)

            noise = torch.randn_like(y)
            x2 = y + noise * t2.view(-1, 1, 1, 1)
            x1 = y + noise * t1.view(-1, 1, 1, 1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_online = online_model(x2, t2, cond)
                with torch.no_grad():
                    pred_target = target_model(x1, t1, cond)
                loss = nn.functional.mse_loss(pred_online, pred_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_ema = ema_final - (ema_final - ema_initial) * math.exp(-step / (total_steps * 0.1))
            update_ema(target_model, online_model, current_ema)

            epoch_loss += loss.item()
            step += 1

        avg_val_loss = epoch_loss / len(train_loader)
        if not disable_pbar:
            print(f"--- EPOCH {epoch+1} | CONSISTENCY LOSS: {avg_val_loss:.6f} ---")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not is_hpo:
                save_path = os.path.join(
                    config.get("EXP_DIR", "."), "models", "consistency", f"consistency_model_{domain}.pth"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(online_model.state_dict(), save_path)

        if is_hpo and trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stopper(avg_val_loss):
            break

    del online_model, target_model, optimizer
    torch.cuda.empty_cache()
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, choices=["unet", "consistency"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    domain = args.source

    current_cluster_path = os.path.join(config["DATA_ROOT"], domain)
    with open(os.path.join(config["DATA_ROOT"], f"stats_{domain}.json"), "r") as f:
        stats = json.load(f)

    train_dataset = ClimateSRDataset(
        cluster_path=current_cluster_path, split="train", normalization_stats=stats, subset_size=args.subset_size
    )
    val_dataset = ClimateSRDataset(
        cluster_path=current_cluster_path, split="validation", normalization_stats=stats, subset_size=args.subset_size
    )

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, pin_memory=True)

    print(f"\n[PHASE 1] Hyperparameter Optimization ({args.architecture}) for {domain}...")

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hpo_epochs = min(config["NUM_EPOCHS"], 5)

        if args.architecture == "unet":
            return run_deterministic_loop(
                domain, config, train_loader, val_loader, device, lr, wd, hpo_epochs, True, trial
            )
        else:
            return run_consistency_loop(
                domain, config, train_loader, val_loader, device, lr, wd, hpo_epochs, True, trial
            )

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=10)
    best_lr, best_wd = study.best_params["learning_rate"], study.best_params["weight_decay"]

    print(f"\n[PHASE 2] Final Full Training ({args.architecture}) for {domain}...")
    if args.architecture == "unet":
        run_deterministic_loop(
            domain, config, train_loader, val_loader, device, best_lr, best_wd, config["NUM_EPOCHS"], False
        )
    else:
        run_consistency_loop(
            domain, config, train_loader, val_loader, device, best_lr, best_wd, config["NUM_EPOCHS"], False
        )


if __name__ == "__main__":
    main()
