import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import json
import warnings

from data.dataset import ClimateSRDataset
from deterministic_unet import DeterministicDualEncoderUNet
from consistency_model import ConsistencyModel
from uda import quantile_mapping

warnings.filterwarnings("ignore", category=RuntimeWarning)


def inverse_arcsinh(x):
    return np.sinh(x)


def evaluate(source_domain, target_domain, device, config, architecture, adaptation_method, apply_qm=False):

    if architecture == "unet":
        model_filename = f"unet_{adaptation_method}_{source_domain}.pth"
        model_path = os.path.join(config.get("EXP_DIR", "."), "models", "unet", model_filename)
        model = DeterministicDualEncoderUNet(target_channels=1, dynamic_channels=9, static_channels=2, base_dim=64)
    elif architecture == "consistency":
        model_filename = f"consistency_{adaptation_method}_{source_domain}.pth"
        model_path = os.path.join(config.get("EXP_DIR", "."), "models", "consistency", model_filename)
        model = ConsistencyModel(sigma_data=0.5, epsilon=0.002)
    else:
        raise ValueError("Invalid architecture specified.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod.") :]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    target_cluster_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/{target_domain}"
    stats_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/stats_{target_domain}.json"

    with open(stats_path, "r") as f:
        target_stats = json.load(f)

    subset_size = config.get("EVAL_SUBSET_SIZE", None)

    test_dataset = ClimateSRDataset(
        cluster_path=target_cluster_path, split="test", normalization_stats=target_stats, subset_size=subset_size
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    total_mse = 0.0
    total_samples = 0
    all_sample_mses, all_true_arrays, all_pred_arrays = [], [], []

    with torch.no_grad():
        for x_dyn, x_stat, y_true_transformed in test_loader:
            x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)
            y_true_target = y_true_transformed.to(
                device
            )  # Proxy for historical empirical cumulative distribution function

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if architecture == "unet":
                    y_pred_transformed = model(x_dyn, x_stat)
                elif architecture == "consistency":
                    b = x_dyn.shape[0]
                    cond = torch.cat([x_dyn, x_stat], dim=1)
                    t_max = 80.0
                    t = torch.full((b,), t_max, device=device)
                    x_noisy = torch.randn((b, 1, 200, 200), device=device) * t_max
                    y_pred_transformed = model(x_noisy, t, cond)

                if apply_qm:
                    y_pred_transformed = quantile_mapping(y_pred_transformed, y_true_target)

            y_pred_transformed = np.clip(y_pred_transformed.cpu().float().numpy(), a_min=0.0, a_max=10.0)
            y_true_transformed = y_true_transformed.numpy()

            y_pred_physical = inverse_arcsinh(y_pred_transformed).astype(np.float64)
            y_true_physical = inverse_arcsinh(y_true_transformed).astype(np.float64)

            batch_squared_error = (y_pred_physical - y_true_physical) ** 2
            total_mse += np.sum(batch_squared_error)
            total_samples += y_true_physical.size

            sample_mse = np.mean(batch_squared_error, axis=(1, 2, 3))
            all_sample_mses.extend(sample_mse.tolist())
            all_true_arrays.extend(y_true_physical[:, 0, :, :])
            all_pred_arrays.extend(y_pred_physical[:, 0, :, :])

    rmse_physical = np.sqrt(total_mse / total_samples)

    results_dir = os.path.join(config["EXP_DIR"], "results", architecture, adaptation_method)
    os.makedirs(results_dir, exist_ok=True)

    result_file = os.path.join(results_dir, "generalization_matrix.csv")
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            f.write("source,target,rmse_physical\n")

    with open(result_file, "a") as f:
        f.write(f"{source_domain},{target_domain},{rmse_physical}\n")

    sorted_indices = np.argsort(all_sample_mses)
    best_indices = sorted_indices[:5]
    worst_indices = sorted_indices[-5:]

    save_dict = {}
    for i, idx in enumerate(best_indices):
        save_dict[f"best_true_{i}"] = all_true_arrays[idx]
        save_dict[f"best_pred_{i}"] = all_pred_arrays[idx]
    for i, idx in enumerate(worst_indices):
        save_dict[f"worst_true_{i}"] = all_true_arrays[idx]
        save_dict[f"worst_pred_{i}"] = all_pred_arrays[idx]

    save_dict["dem"] = test_dataset.static_tensor[1].numpy()

    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    np.savez_compressed(os.path.join(samples_dir, f"{source_domain}_to_{target_domain}.npz"), **save_dict)
    print(
        f"[{architecture.upper()} | {adaptation_method.upper()} | QM: {apply_qm}] Evaluated {source_domain} -> {target_domain} | RMSE: {rmse_physical:.4f} mm"
    )


def compute_generalization_metrics(results_csv, w1_matrix_path, domain_list, var_idx=6, architecture="unet"):
    print(f"--- Computing Domain Generalization Metrics for {architecture.upper()} ---")
    df = pd.read_csv(results_csv).drop_duplicates(subset=["source", "target"], keep="last")
    rmse_matrix = pd.DataFrame(index=domain_list, columns=domain_list, dtype=float)

    for _, row in df.iterrows():
        rmse_matrix.loc[row["source"], row["target"]] = row["rmse_physical"]

    w1_full = np.load(w1_matrix_path)
    w1_precip = w1_full[var_idx, :, :]
    w1_matrix = pd.DataFrame(w1_precip, index=domain_list, columns=domain_list)

    w1_values = w1_matrix.values.flatten()
    epsilon = (np.min(w1_values[w1_values > 0]) * 0.1) if len(w1_values[w1_values > 0]) > 0 else 1e-5
    median_w1 = np.median(w1_values)
    gamma = 1.0 / median_w1 if median_w1 > 0 else 1.0

    metrics_list = []
    for source in domain_list:
        e_s = rmse_matrix.loc[source, source]
        target_errors, target_w1s, dda_sum = [], [], 0.0

        for target in domain_list:
            e_t, w1_st = rmse_matrix.loc[source, target], w1_matrix.loc[source, target]
            ngg = (e_t - e_s) / (w1_st + epsilon)

            if source != target:
                dda_sum += e_t * np.exp(-gamma * w1_st)
                target_errors.append(e_t)
                target_w1s.append(w1_st)

            metrics_list.append({"source": source, "target": target, "wasserstein_1d": w1_st, "rmse": e_t, "ngg": ngg})

        spearman_rho, _ = spearmanr(target_w1s, target_errors)
        print(f"\nSource Domain: {source} | DDA Score: {dda_sum:.4f} | Spearman ρ: {spearman_rho:.4f}")

    metrics_df = pd.DataFrame(metrics_list)
    out_csv = os.path.join(os.path.dirname(results_csv), "detailed_generalization_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=["evaluate", "compute_metrics"], required=True)
    parser.add_argument("--architecture", type=str, choices=["unet", "consistency"], required=True)
    parser.add_argument(
        "--adaptation_method",
        type=str,
        choices=["none", "coral", "mmd", "sinkhorn", "spectral", "fourier"],
        default="none",
        help="Specify the UDA method evaluated.",
    )
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--w1_path", type=str, default="covariate_shift_analysis/normalized/Wasserstein_1D_test.npy")
    parser.add_argument("--apply_qm", action="store_true", help="Apply marginal histogram matching.")
    args = parser.parse_args()

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.action == "evaluate":
        evaluate(
            args.source,
            args.target,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config,
            args.architecture,
            args.adaptation_method,
            args.apply_qm,
        )
    elif args.action == "compute_metrics":
        results_csv = os.path.join(
            config["EXP_DIR"], "results", args.architecture, args.adaptation_method, "generalization_matrix.csv"
        )
        compute_generalization_metrics(
            results_csv, args.w1_path, config["DOMAIN_LIST"], var_idx=6, architecture=args.architecture
        )
