import os
import json
import math
import copy
import argparse
import csv
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
from uda import deep_coral_loss, spectral_density_loss, mmd_loss, sinkhorn_divergence, fourier_domain_adaptation

# --- GH200 OPTIMIZATIONS ---
slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 12))
NUM_WORKERS = min(slurm_cpus, 16)
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
    return (min_t ** (1 / rho) + u * (max_t ** (1 / rho) - min_t ** (1 / rho))) ** rho


def run_deterministic_loop(
    source_domain,
    config,
    train_loader_s,
    val_loader_s,
    target_loader_unlabeled,
    device,
    lr,
    wd,
    num_epochs,
    adaptation_method,
    is_hpo_trial=False,
    trial=None,
):
    model = DeterministicDualEncoderUNet(target_channels=1, dynamic_channels=9, static_channels=2, base_dim=64).to(
        device
    )
    compile_mode = "default" if is_hpo_trial else "max-autotune"
    model = torch.compile(model, mode=compile_mode)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopper(patience=config.get("PATIENCE", 5), min_delta=config.get("MIN_DELTA", 0.0))

    best_val_loss = float("inf")
    disable_pbar = is_hpo_trial

    lambda_align = config.get("LAMBDA_ALIGN", 0.1)
    lambda_spec = config.get("LAMBDA_SPEC", 0.01)

    is_pruned = False

    if not is_hpo_trial:
        log_dir = os.path.join(config.get("EXP_DIR", "."), "logs", "unet")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"metrics_{adaptation_method}_{source_domain}.csv")
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_total", "train_task", "train_align", "train_spec", "val_total"])

    epoch_batches = min(len(train_loader_s), len(target_loader_unlabeled))

    for epoch in range(num_epochs):
        model.train()

        epoch_loss_total, epoch_loss_task, epoch_loss_align, epoch_loss_spec = 0.0, 0.0, 0.0, 0.0

        train_pbar = tqdm.tqdm(
            zip(train_loader_s, target_loader_unlabeled),
            desc=f"Epoch {epoch+1}/{num_epochs} [UNET {adaptation_method.upper()}]",
            total=epoch_batches,
            disable=disable_pbar,
            leave=False,
        )

        for (x_dyn_s, x_stat_s, y_s), (x_dyn_t, x_stat_t, y_t_unl) in train_pbar:
            x_dyn_s, x_stat_s, y_s = x_dyn_s.to(device), x_stat_s.to(device), y_s.to(device)
            x_dyn_t, x_stat_t, y_t_unl = x_dyn_t.to(device), x_stat_t.to(device), y_t_unl.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                # Input-space adaptation
                if adaptation_method == "fourier":
                    x_dyn_s = fourier_domain_adaptation(x_dyn_s, x_dyn_t)

                extract = adaptation_method in ["coral", "mmd", "sinkhorn"]

                if extract:
                    pred_s, feat_s = model(x_dyn_s, x_stat_s, extract_features=True)
                else:
                    pred_s = model(x_dyn_s, x_stat_s)

                loss_task = loss_fn(pred_s, y_s)
                total_loss = loss_task

                if adaptation_method not in ["none", "fourier"]:
                    if extract:
                        pred_t, feat_t = model(x_dyn_t, x_stat_t, extract_features=True)
                    else:
                        pred_t = model(x_dyn_t, x_stat_t)

                    if adaptation_method == "coral":
                        loss_align = deep_coral_loss(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "mmd":
                        loss_align = mmd_loss(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "sinkhorn":
                        loss_align = sinkhorn_divergence(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "spectral":
                        loss_spec = spectral_density_loss(pred_t, y_t_unl)
                        total_loss = total_loss + lambda_spec * loss_spec
                        epoch_loss_spec += loss_spec.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss_total += total_loss.item()
            epoch_loss_task += loss_task.item()

        model.eval()
        total_val_loss = 0.0
        val_pbar = tqdm.tqdm(
            val_loader_s, desc=f"Epoch {epoch+1}/{num_epochs} [UNET Val]", disable=disable_pbar, leave=False
        )

        with torch.no_grad():
            for x_dyn_val, x_stat_val, y_val in val_pbar:
                x_dyn_val, x_stat_val, y_val = x_dyn_val.to(device), x_stat_val.to(device), y_val.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds_val = model(x_dyn_val, x_stat_val)
                    val_loss = loss_fn(preds_val, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader_s)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not is_hpo_trial:
                save_path = os.path.join(
                    config.get("EXP_DIR", "."), "models", "unet", f"unet_{adaptation_method}_{source_domain}.pth"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if not is_hpo_trial:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        current_lr,
                        epoch_loss_total / epoch_batches,
                        epoch_loss_task / epoch_batches,
                        epoch_loss_align / epoch_batches,
                        epoch_loss_spec / epoch_batches,
                        avg_val_loss,
                    ]
                )

        if early_stopper(avg_val_loss):
            break

    del model, optimizer
    torch.cuda.empty_cache()

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return best_val_loss


def run_consistency_loop(
    source_domain,
    config,
    train_loader_s,
    val_loader_s,
    target_loader_unlabeled,
    device,
    lr,
    wd,
    num_epochs,
    adaptation_method,
    is_hpo_trial=False,
    trial=None,
):
    online_model = ConsistencyModel(sigma_data=0.5, epsilon=0.002).to(device)
    target_model = copy.deepcopy(online_model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    compile_mode = "default" if is_hpo_trial else "max-autotune"
    online_model = torch.compile(online_model, mode=compile_mode)
    target_model = torch.compile(target_model, mode=compile_mode)

    optimizer = optim.Adam(online_model.parameters(), lr=lr, weight_decay=wd)
    early_stopper = EarlyStopper(patience=config.get("PATIENCE", 5), min_delta=config.get("MIN_DELTA", 0.0))

    min_t, max_t = 0.002, 80.0
    ema_initial, ema_final = 0.95, 0.999

    epoch_batches = min(len(train_loader_s), len(target_loader_unlabeled))
    total_steps = epoch_batches * num_epochs
    step = 0
    best_val_loss = float("inf")
    disable_pbar = is_hpo_trial

    lambda_align = config.get("LAMBDA_ALIGN", 0.1)
    lambda_spec = config.get("LAMBDA_SPEC", 0.01)

    is_pruned = False

    if not is_hpo_trial:
        log_dir = os.path.join(config.get("EXP_DIR", "."), "logs", "consistency")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"metrics_{adaptation_method}_{source_domain}.csv")
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_total", "train_task", "train_align", "train_spec", "val_total"])

    for epoch in range(num_epochs):
        online_model.train()

        epoch_loss_total, epoch_loss_task, epoch_loss_align, epoch_loss_spec = 0.0, 0.0, 0.0, 0.0

        train_pbar = tqdm.tqdm(
            zip(train_loader_s, target_loader_unlabeled),
            desc=f"Epoch {epoch+1}/{num_epochs} [CONSISTENCY {adaptation_method.upper()}]",
            total=epoch_batches,
            disable=disable_pbar,
            leave=False,
        )

        for (x_dyn_s, x_stat_s, y_s), (x_dyn_t, x_stat_t, y_t_unl) in train_pbar:
            x_dyn_s, x_stat_s, y_s = x_dyn_s.to(device), x_stat_s.to(device), y_s.to(device)
            x_dyn_t, x_stat_t, y_t_unl = x_dyn_t.to(device), x_stat_t.to(device), y_t_unl.to(device)

            if adaptation_method == "fourier":
                x_dyn_s = fourier_domain_adaptation(x_dyn_s, x_dyn_t)

            cond_s = torch.cat([x_dyn_s, x_stat_s], dim=1)
            cond_t = torch.cat([x_dyn_t, x_stat_t], dim=1)
            b = y_s.shape[0]

            t2_s = sample_timesteps(b, min_t, max_t, device)
            t1_s = torch.clamp(t2_s - (max_t - min_t) / 100.0, min=min_t)

            noise_s = torch.randn_like(y_s)
            x2_s = y_s + noise_s * t2_s.view(-1, 1, 1, 1)
            x1_s = y_s + noise_s * t1_s.view(-1, 1, 1, 1)

            t_max_tensor = torch.full((b,), max_t, device=device)
            x_noisy_t = torch.randn((b, 1, 200, 200), device=device) * max_t

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                extract = adaptation_method in ["coral", "mmd", "sinkhorn"]

                if extract:
                    pred_online_s, feat_s = online_model(x2_s, t2_s, cond_s, extract_features=True)
                else:
                    pred_online_s = online_model(x2_s, t2_s, cond_s)

                with torch.no_grad():
                    pred_target_s = target_model(x1_s, t1_s, cond_s)

                loss_task = nn.functional.mse_loss(pred_online_s, pred_target_s)
                total_loss = loss_task

                if adaptation_method not in ["none", "fourier"]:
                    if extract:
                        pred_t, feat_t = online_model(x_noisy_t, t_max_tensor, cond_t, extract_features=True)
                    else:
                        pred_t = online_model(x_noisy_t, t_max_tensor, cond_t)

                    if adaptation_method == "coral":
                        loss_align = deep_coral_loss(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "mmd":
                        loss_align = mmd_loss(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "sinkhorn":
                        loss_align = sinkhorn_divergence(feat_s, feat_t)
                        total_loss = total_loss + lambda_align * loss_align
                        epoch_loss_align += loss_align.item()
                    elif adaptation_method == "spectral":
                        loss_spec = spectral_density_loss(pred_t, y_t_unl)
                        total_loss = total_loss + lambda_spec * loss_spec
                        epoch_loss_spec += loss_spec.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            current_ema = ema_final - (ema_final - ema_initial) * math.exp(-step / (total_steps * 0.1))
            update_ema(target_model, online_model, current_ema)

            epoch_loss_total += total_loss.item()
            epoch_loss_task += loss_task.item()
            step += 1

        online_model.eval()
        target_model.eval()

        total_val_loss = 0.0
        val_pbar = tqdm.tqdm(
            val_loader_s, desc=f"Epoch {epoch+1}/{num_epochs} [CONSISTENCY Val]", disable=disable_pbar, leave=False
        )

        with torch.no_grad():
            for x_dyn_val, x_stat_val, y_val in val_pbar:
                x_dyn_val, x_stat_val, y_val = x_dyn_val.to(device), x_stat_val.to(device), y_val.to(device)
                cond_val = torch.cat([x_dyn_val, x_stat_val], dim=1)
                b_val = y_val.shape[0]

                t2_val = sample_timesteps(b_val, min_t, max_t, device)
                t1_val = torch.clamp(t2_val - (max_t - min_t) / 100.0, min=min_t)

                noise_val = torch.randn_like(y_val)
                x2_val = y_val + noise_val * t2_val.view(-1, 1, 1, 1)
                x1_val = y_val + noise_val * t1_val.view(-1, 1, 1, 1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_online_val = online_model(x2_val, t2_val, cond_val)
                    pred_target_val = target_model(x1_val, t1_val, cond_val)
                    val_loss = nn.functional.mse_loss(pred_online_val, pred_target_val)

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader_s)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not is_hpo_trial:
                save_path = os.path.join(
                    config.get("EXP_DIR", "."),
                    "models",
                    "consistency",
                    f"consistency_{adaptation_method}_{source_domain}.pth",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(online_model.state_dict(), save_path)

        if not is_hpo_trial:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        current_lr,
                        epoch_loss_total / epoch_batches,
                        epoch_loss_task / epoch_batches,
                        epoch_loss_align / epoch_batches,
                        epoch_loss_spec / epoch_batches,
                        avg_val_loss,
                    ]
                )

        if early_stopper(avg_val_loss):
            break

    del online_model, target_model, optimizer
    torch.cuda.empty_cache()

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="GH200 Optimized UDA Training Script")
    parser.add_argument("--architecture", type=str, choices=["unet", "consistency"], required=True)
    parser.add_argument(
        "--adaptation_method",
        type=str,
        choices=["none", "coral", "mmd", "sinkhorn", "spectral", "fourier"],
        default="none",
        help="Choose the UDA method to apply during training.",
    )
    parser.add_argument("--source", type=str, required=True, help="Labeled Source Domain")
    parser.add_argument("--target", type=str, required=True, help="Unlabeled Target Domain for UDA")
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    domain_s, domain_t, adaptation = args.source, args.target, args.adaptation_method

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    print(f"Transfer: {domain_s} -> {domain_t} | Arch: {args.architecture.upper()} | Method: {adaptation.upper()}")

    source_path = os.path.join(config["DATA_ROOT"], domain_s)
    with open(os.path.join(config["DATA_ROOT"], f"stats_{domain_s}.json"), "r") as f:
        stats_s = json.load(f)

    train_dataset_s = ClimateSRDataset(
        cluster_path=source_path, split="train", normalization_stats=stats_s, subset_size=args.subset_size
    )
    val_dataset_s = ClimateSRDataset(
        cluster_path=source_path, split="validation", normalization_stats=stats_s, subset_size=args.subset_size
    )

    train_loader_s = DataLoader(
        train_dataset_s,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader_s = DataLoader(
        val_dataset_s, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    target_path = os.path.join(config["DATA_ROOT"], domain_t)
    target_dataset_unl = ClimateSRDataset(
        cluster_path=target_path, split="train", normalization_stats=stats_s, subset_size=args.subset_size
    )
    target_loader_unl = DataLoader(
        target_dataset_unl,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hpo_epochs = min(config["NUM_EPOCHS"], 5)

        loop_fn = run_deterministic_loop if args.architecture == "unet" else run_consistency_loop
        return loop_fn(
            domain_s,
            config,
            train_loader_s,
            val_loader_s,
            target_loader_unl,
            DEVICE,
            lr,
            wd,
            hpo_epochs,
            adaptation,
            is_hpo_trial=True,
            trial=trial,
        )

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=10)

    loop_fn = run_deterministic_loop if args.architecture == "unet" else run_consistency_loop
    loop_fn(
        domain_s,
        config,
        train_loader_s,
        val_loader_s,
        target_loader_unl,
        DEVICE,
        study.best_params["learning_rate"],
        study.best_params["weight_decay"],
        config["NUM_EPOCHS"],
        adaptation,
        is_hpo_trial=False,
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
