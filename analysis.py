import yaml
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
import xarray as xr
import warnings

# Suppress runtime warnings from divide-by-zero, etc.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 1. Metric Definitions ---


def get_pdfs(samples_A, samples_B, n_bins=100):
    """
    Calculates normalized histograms (PDFs) for two 1D sample arrays.
    """
    if samples_A.size == 0 or samples_B.size == 0:
        return np.ones(n_bins) / n_bins, np.ones(n_bins) / n_bins  # Return uniform dist for empty

    # Find a common range
    min_val = min(np.min(samples_A), np.min(samples_B))
    max_val = max(np.max(samples_A), np.max(samples_B))

    if min_val == max_val:  # Handle case where all data is identical
        min_val -= 0.5
        max_val += 0.5

    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms
    pdf_A, _ = np.histogram(samples_A, bins=bins, density=True)
    pdf_B, _ = np.histogram(samples_B, bins=bins, density=True)

    # Add a small epsilon to avoid log(0) or division by zero
    epsilon = 1e-10
    pdf_A = (pdf_A + epsilon) / (pdf_A.sum() + epsilon * n_bins)
    pdf_B = (pdf_B + epsilon) / (pdf_B.sum() + epsilon * n_bins)

    return pdf_A, pdf_B


def hellinger_distance(p, q):
    """Hellinger distance for two 1D PDFs."""
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def jensen_shannon_divergence(p, q):
    """Jensen-Shannon Divergence (JSD) for two 1D PDFs."""
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return np.sqrt(jsd)  # Return J-S Distance (sqrt of divergence) for a metric


def wasserstein1d(samples_A, samples_B):
    """1D Wasserstein distance (Earth-Mover's Distance)."""
    if samples_A.size == 0 or samples_B.size == 0:
        return np.nan
    return wasserstein_distance(samples_A, samples_B)


def mmd_rbf(X, Y, gamma=None):
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.
    Works on 1D or 2D sample arrays (N_samples, N_features).
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    if gamma is None:
        # Use the median heuristic for gamma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_dist = np.median(np.abs(X - np.median(X)))
        gamma = 1.0 / (median_dist + 1e-8)
        if not np.isfinite(gamma):
            gamma = 1.0

    # Reshape for sklearn
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    K_XX = rbf_kernel(X, X, gamma=gamma).mean()
    K_YY = rbf_kernel(Y, Y, gamma=gamma).mean()
    K_XY = rbf_kernel(X, Y, gamma=gamma).mean()

    # Ensure non-negative result (can be slightly negative due to precision)
    return np.maximum(0.0, K_XX + K_YY - 2 * K_XY)


# --- 2. Helper Functions ---


# --- New Dataset class for loading RAW data ---
class RawClimateDataset(Dataset):
    """
    Minimalist dataset to load RAW, UNNORMALIZED data from Zarr stores.
    It only performs the essential 'tp' unit correction.
    """

    def __init__(self, cluster_path, split, dynamic_vars, static_vars):

        # Use 'train' data for 'validation' split, as in ClimateSRDataset
        zarr_split = "train" if split == "validation" else split

        # --- Efficient Zarr loading ---
        # 1. Open Zarr store respecting its on-disk chunks
        _ds_in = xr.open_zarr(os.path.join(cluster_path, f"{zarr_split}_data_in.zarr"))
        # 2. Apply in-memory rechunking for our access pattern
        self.ds_in = _ds_in.chunk({"time": 1})

        self.ds_static = xr.open_dataset(
            os.path.join(cluster_path, "static_variables.nc")
        ).load()  # Static is small, load it

        self.dynamic_vars = dynamic_vars
        self.static_vars = static_vars
        self.num_samples = len(self.ds_in["time"])

        # This dataset needs to know about the train/val split
        # from the original dataset class to sample correctly
        all_indices = np.arange(self.num_samples)
        rng = np.random.RandomState(42)  # Use same seed
        rng.shuffle(all_indices)
        split_point = int(self.num_samples * (1.0 - 0.2))  # 0.2 = validation_split_pct

        if split == "train":
            self.indices = all_indices[:split_point]
        else:  # split == "validation"
            self.indices = all_indices[split_point:]

        self.split_num_samples = len(self.indices)

    def __len__(self):
        return self.split_num_samples

    def __getitem__(self, idx):
        # Map to the true index in the Zarr store
        true_idx = self.indices[idx]

        var_list = []

        # 1. Load dynamic variables
        for var in self.dynamic_vars:
            data = self.ds_in[var].isel(time=true_idx).values
            # --- Apply critical unit correction ---
            if var == "tp":
                data = data * 1000.0
            var_list.append(data)

        # 2. Load static variables
        for var in self.static_vars:
            data = self.ds_static[var].values
            var_list.append(data)

        # Stack all 11 channels
        x = np.stack(var_list, axis=0)

        # Return as a torch tensor for the DataLoader
        return torch.from_numpy(x).float()


# --- New loading function ---
def load_raw_samples(cluster_path, split, dynamic_vars, static_vars, num_samples_to_draw, n_workers=4):
    """
    Loads N random samples of RAW data from a given dataset split.
    """
    try:
        dataset = RawClimateDataset(
            cluster_path=cluster_path, split=split, dynamic_vars=dynamic_vars, static_vars=static_vars
        )
    except Exception as e:
        print(f"Error loading dataset for {cluster_path}: {e}. Returning empty array.")
        return np.array([])

    # Create a random sampler
    if num_samples_to_draw > len(dataset):
        num_samples_to_draw = len(dataset)

    if num_samples_to_draw == 0:
        return np.array([])

    sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:num_samples_to_draw])

    loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=False,
    )

    all_x = []
    # The new dataset only returns x
    for x_batch in tqdm(loader, desc=f"Loading {num_samples_to_draw} samples", leave=False):
        all_x.append(x_batch.numpy())

    if not all_x:
        return np.array([])

    return np.concatenate(all_x, axis=0)


def plot_matrix(matrix, title, labels, save_path):
    """
    Plots a heatmap of the distance matrix.
    """
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- 3. Main Script ---


def main():

    # --- Args ---
    import argparse

    parser = argparse.ArgumentParser(description="Covariate Shift Analysis across Domains")
    parser.add_argument(
        "--metrics_type",
        type=str,
        default="pdf",
        choices=["pdf", "sample"],
        help="Type of metrics to compute: 'pdf' for PDF-based, 'sample' for sample-based",
    )
    args = parser.parse_args()

    # --- Config ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "configs/config_rainshift.yaml"), "r") as f:
        config = yaml.safe_load(f)

    DATA_ROOT = config["DATA_ROOT"]
    DOMAINS = config["DOMAIN_LIST"]
    VAR_NAMES = config["VAR_LIST"]
    DYNAMIC_VARS = VAR_NAMES
    STATIC_VARS = []

    N_DOMAINS = len(DOMAINS)
    N_VARS = len(VAR_NAMES)
    N_SAMPLES = config["N_SAMPLES"]
    N_BINS = config["N_BINS"]
    N_WORKERS = config.get("NUM_WORKERS", 4)

    MMD_SUBSAMPLE_SIZE = 100000

    # Create base output directory
    output_dir = os.path.join(current_dir, "covariate_shift_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load and Cache Data (RAW) ---
    print(f"Loading {N_SAMPLES} RAW samples from all 18 domains...")
    data_cache = {"test": {}}

    for split in ["test"]:
        print(f"--- Loading {split} data ---")
        for domain in tqdm(DOMAINS, desc=f"Caching {split} domains"):
            cluster_path = os.path.join(DATA_ROOT, domain)
            data_cache[split][domain] = load_raw_samples(
                cluster_path, split, DYNAMIC_VARS, STATIC_VARS, N_SAMPLES, N_WORKERS
            )
    print("Data caching complete.")

    # --- 2. Select Metrics ---
    metrics_pdf = {}
    metrics_sample = {}
    if args.metrics_type == "pdf":
        print("Using PDF-based metrics for analysis.")
        metrics_pdf = {
            "Hellinger": hellinger_distance,
            "JSD": jensen_shannon_divergence,
        }
    else:
        print("Using Sample-based metrics for analysis.")
        metrics_sample = {
            "Wasserstein_1D": wasserstein1d,
            "MMD_RBF": mmd_rbf,
        }

    # --- 3. Compute Distance Matrices ---
    for analysis_type in ["raw", "normalized"]:
        print(f"\n{'='*70}")
        print(f"--- STARTING ANALYSIS: {analysis_type.upper()} DATA ---")
        print(f"{'='*70}\n")

        analysis_output_dir = os.path.join(output_dir, analysis_type)
        os.makedirs(analysis_output_dir, exist_ok=True)

        for split in ["test"]:
            print(f"\n--- Computing metrics for {split} split ---")

            # --- PDF-based metrics ---
            for metric_name, metric_func in metrics_pdf.items():
                print(f"Calculating {metric_name} distance...")
                dist_matrix = np.zeros((N_VARS, N_DOMAINS, N_DOMAINS))

                for k in tqdm(range(N_VARS), desc=f"Variables for {metric_name}"):
                    for i in range(N_DOMAINS):
                        for j in range(i + 1, N_DOMAINS):

                            # --- Add check for valid data ---
                            samples_A_all_vars = data_cache[split][DOMAINS[i]]
                            samples_B_all_vars = data_cache[split][DOMAINS[j]]

                            if samples_A_all_vars.ndim < 4 or samples_B_all_vars.ndim < 4:
                                # This pair has failed-to-load data
                                dist = np.nan
                                dist_matrix[k, i, j] = dist
                                dist_matrix[k, j, i] = dist
                                continue

                            samples_A = samples_A_all_vars[:, k].ravel()
                            samples_B = samples_B_all_vars[:, k].ravel()

                            # --- Conditional Normalization ---
                            if analysis_type == "normalized":
                                std_A = np.std(samples_A) + 1e-8
                                if std_A > 1e-6:  # Only normalize if not constant
                                    samples_A = (samples_A - np.mean(samples_A)) / std_A

                                std_B = np.std(samples_B) + 1e-8
                                if std_B > 1e-6:
                                    samples_B = (samples_B - np.mean(samples_B)) / std_B

                            pdf_A, pdf_B = get_pdfs(samples_A, samples_B, n_bins=N_BINS)
                            dist = metric_func(pdf_A, pdf_B)
                            dist_matrix[k, i, j] = dist
                            dist_matrix[k, j, i] = dist

                np.save(f"{analysis_output_dir}/{metric_name}_{split}.npy", dist_matrix)
                for k in range(N_VARS):
                    plot_matrix(
                        dist_matrix[k],
                        f"{metric_name} Distance - {VAR_NAMES[k]} ({split}, {analysis_type})",
                        DOMAINS,
                        f"{analysis_output_dir}/{metric_name}_{split}_{VAR_NAMES[k]}.png",
                    )

            # --- Sample-based metrics ---
            for metric_name, metric_func in metrics_sample.items():
                print(f"Calculating {metric_name} distance...")
                dist_matrix = np.zeros((N_VARS, N_DOMAINS, N_DOMAINS))

                for k in tqdm(range(N_VARS), desc=f"Variables for {metric_name}"):
                    for i in range(N_DOMAINS):
                        for j in range(i + 1, N_DOMAINS):

                            # --- Add check for valid data ---
                            samples_A_all_vars = data_cache[split][DOMAINS[i]]
                            samples_B_all_vars = data_cache[split][DOMAINS[j]]

                            if samples_A_all_vars.ndim < 4 or samples_B_all_vars.ndim < 4:
                                if i == 0 and j == 1 and k == 0:  # Print only once
                                    print(f"Warning: Data for {DOMAINS[i]} or {DOMAINS[j]} is invalid (ndim < 4).")
                                    print(
                                        f"Shapes: {DOMAINS[i]}={samples_A_all_vars.shape}, {DOMAINS[j]}={samples_B_all_vars.shape}"
                                    )
                                    print("Skipping comparison for this pair.")
                                # This pair has failed-to-load data
                                dist = np.nan
                                dist_matrix[k, i, j] = dist
                                dist_matrix[k, j, i] = dist
                                continue

                            samples_A = samples_A_all_vars[:, k].ravel()
                            samples_B = samples_B_all_vars[:, k].ravel()

                            # --- Conditional Normalization ---
                            if analysis_type == "normalized":
                                std_A = np.std(samples_A) + 1e-8
                                if std_A > 1e-6:  # Only normalize if not constant
                                    samples_A = (samples_A - np.mean(samples_A)) / std_A

                                std_B = np.std(samples_B) + 1e-8
                                if std_B > 1e-6:
                                    samples_B = (samples_B - np.mean(samples_B)) / std_B

                            # --- Subsampling for MMD ---
                            dist_A = samples_A
                            dist_B = samples_B
                            if metric_name == "MMD_RBF":
                                if len(dist_A) > MMD_SUBSAMPLE_SIZE:
                                    dist_A = np.random.choice(dist_A, MMD_SUBSAMPLE_SIZE, replace=False)
                                if len(dist_B) > MMD_SUBSAMPLE_SIZE:
                                    dist_B = np.random.choice(dist_B, MMD_SUBSAMPLE_SIZE, replace=False)

                            dist = metric_func(dist_A, dist_B)
                            dist_matrix[k, i, j] = dist
                            dist_matrix[k, j, i] = dist

                np.save(f"{analysis_output_dir}/{metric_name}_{split}.npy", dist_matrix)
                for k in range(N_VARS):
                    plot_matrix(
                        dist_matrix[k],
                        f"{metric_name} Distance - {VAR_NAMES[k]} ({split}, {analysis_type})",
                        DOMAINS,
                        f"{analysis_output_dir}/{metric_name}_{split}_{VAR_NAMES[k]}.png",
                    )

        print(f"\n--- {analysis_type.upper()} analysis for {split} split complete ---")

    print("\n--- All covariate shift analyses complete ---")


if __name__ == "__main__":
    # Set multiprocessing start method for safety with torch/numpy
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
