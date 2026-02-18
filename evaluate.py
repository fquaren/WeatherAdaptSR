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
from deterministic_model import EDSRModel

warnings.filterwarnings("ignore", category=RuntimeWarning)


def inverse_arcsinh(x):
    """Converts transformed variables back to physical units (mm)."""
    return np.sinh(x)


def evaluate(source_domain, target_domain, device, config):
    """Computes the physical root mean squared error for a source-target domain pair."""

    # Load the model trained on the source domain
    model_path = os.path.join(config["EXP_DIR"], "models", "EDSR", f"edsr_baseline_{source_domain}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = EDSRModel(dynamic_in_channels=9, static_in_channels=2, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load the target domain test dataset
    target_cluster_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/{target_domain}"
    stats_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/rainshift/stats_{target_domain}.json"

    with open(stats_path, "r") as f:
        target_stats = json.load(f)

    test_dataset = ClimateSRDataset(
        cluster_path=target_cluster_path,
        split="test",
        normalization_stats=target_stats,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_dyn, x_stat, y_true_transformed in test_loader:
            x_dyn, x_stat = x_dyn.to(device), x_stat.to(device)

            # Forward pass
            y_pred_transformed = model(x_dyn, x_stat).cpu().numpy()
            y_true_transformed = y_true_transformed.numpy()

            # Transform back to physical space (mm of precipitation)
            y_pred_physical = inverse_arcsinh(y_pred_transformed)
            y_true_physical = inverse_arcsinh(y_true_transformed)

            # Compute physical mean squared error
            batch_mse = np.sum((y_pred_physical - y_true_physical) ** 2)
            total_mse += batch_mse
            total_samples += y_true_physical.size

    rmse_physical = np.sqrt(total_mse / total_samples)

    # Save the result to a central matrix file
    result_file = "results/generalization_matrix.csv"
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            f.write("source,target,rmse_physical\n")

    with open(result_file, "a") as f:
        f.write(f"{source_domain},{target_domain},{rmse_physical}\n")
    print(f"Evaluated Source: {source_domain} | Target: {target_domain} | RMSE: {rmse_physical:.4f} mm")


def compute_generalization_metrics(results_csv, w1_matrix_path, domain_list, var_idx=6):
    """
    Computes normalized generalization gap, divergence-discounted accuracy,
    and Spearman rank correlation from the populated RMSE matrix.

    Args:
        var_idx: Index of the precipitation variable in the saved Wasserstein matrix.
                 (Assumes 'tp' is index 6 based on your VAR_LIST).
    """
    print("--- Computing Domain Generalization Metrics ---")

    # 1. Load physical errors
    df = pd.read_csv(results_csv)

    # Create an empty 18x18 DataFrame for RMSE
    rmse_matrix = pd.DataFrame(index=domain_list, columns=domain_list, dtype=float)
    for _, row in df.iterrows():
        rmse_matrix.loc[row["source"], row["target"]] = row["rmse_physical"]

    if rmse_matrix.isna().any().any():
        print("Warning: The evaluation matrix is incomplete. Run all source-target pairs first.")
        return

    # 2. Load Wasserstein distances
    # Expected shape: (N_VARS, N_DOMAINS, N_DOMAINS)
    w1_full = np.load(w1_matrix_path)
    w1_precip = w1_full[var_idx, :, :]
    w1_matrix = pd.DataFrame(w1_precip, index=domain_list, columns=domain_list)

    # 3. Calculate global hyperparameters
    # Epsilon ensures numerical stability and should be set one order of magnitude smaller than the minimum non-zero Wasserstein distance[cite: 27].
    w1_values = w1_matrix.values.flatten()
    non_zero_w1 = w1_values[w1_values > 0]
    epsilon = (np.min(non_zero_w1) * 0.1) if len(non_zero_w1) > 0 else 1e-5

    # Gamma defines the length scale of domain similarity and is calibrated using the median heuristic[cite: 34].
    median_w1 = np.median(w1_values)
    gamma = 1.0 / median_w1 if median_w1 > 0 else 1.0

    metrics_list = []

    # 4. Compute metrics per source domain
    for s_idx, source in enumerate(domain_list):
        e_s = rmse_matrix.loc[source, source]  # Source error Es(f)

        target_errors = []
        target_w1s = []
        dda_sum = 0.0

        for t_idx, target in enumerate(domain_list):
            e_t = rmse_matrix.loc[source, target]
            w1_st = w1_matrix.loc[source, target]

            # Normalized generalization gap [cite: 22]
            # Isolates structural robustness from the inherent difficulty of the target climate[cite: 21].
            ngg = (e_t - e_s) / (w1_st + epsilon)

            # Divergence-discounted accuracy components [cite: 31]
            if source != target:
                dda_sum += e_t * np.exp(-gamma * w1_st)
                target_errors.append(e_t)
                target_w1s.append(w1_st)

            # Log pairwise NGG
            metrics_list.append({"source": source, "target": target, "wasserstein_1d": w1_st, "rmse": e_t, "ngg": ngg})

        # Spearman correlation [cite: 36]
        # Tests the theoretical validity of the domain adaptation bound[cite: 36].
        spearman_rho, _ = spearmanr(target_w1s, target_errors)

        # Log source-level aggregated metrics
        print(f"\nSource Domain: {source}")
        print(f"  DDA Score: {dda_sum:.4f}")
        print(f"  Spearman ρ: {spearman_rho:.4f}")

    # 5. Export results
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("results/detailed_generalization_metrics.csv", index=False)
    print("\nDetailed pairwise metrics saved to results/detailed_generalization_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Generalization Evaluation Framework")
    parser.add_argument(
        "--action",
        type=str,
        choices=["evaluate", "compute_metrics"],
        required=True,
        help="Action to perform: 'evaluate' a single pair, or 'compute_metrics' for the entire matrix.",
    )

    # Arguments for 'evaluate' action
    parser.add_argument("--source", type=str, help="Source domain model to load.")
    parser.add_argument("--target", type=str, help="Target domain dataset to evaluate on.")

    # Arguments for 'compute_metrics' action
    parser.add_argument(
        "--w1_path",
        type=str,
        default="covariate_shift_analysis/normalized/Wasserstein_1D_test.npy",
        help="Path to the pre-computed Wasserstein distance numpy array.",
    )

    args = parser.parse_args()

    config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Shared domain list to map array indices to domain names
    domain_list = config[
        "DOMAIN_LIST"
    ]  # Ensure this is defined in your config YAML and matches the order in the Wasserstein matrix

    if args.action == "evaluate":
        if not args.source or not args.target:
            raise ValueError("Both --source and --target must be provided for evaluation.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate(args.source, args.target, device, config)

    elif args.action == "compute_metrics":
        results_csv = "results/generalization_matrix.csv"
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Missing {results_csv}. Run the evaluation loop first.")
        if not os.path.exists(args.w1_path):
            raise FileNotFoundError(f"Missing {args.w1_path}. Provide the correct path to the Wasserstein distances.")

        compute_generalization_metrics(results_csv, args.w1_path, domain_list, var_idx=6)
