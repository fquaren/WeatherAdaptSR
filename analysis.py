import yaml
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Metric Definition ---


def wasserstein1d_gpu(samples_A, samples_B, device="mps"):
    """
    1d wasserstein distance calculated on GPU using empirical inverse CDF matching.
    """
    if samples_A.size == 0 or samples_B.size == 0:
        return np.nan

    t_A = torch.from_numpy(samples_A).to(device)
    t_B = torch.from_numpy(samples_B).to(device)

    # If shapes match exactly, direct sorting is the fastest operation
    if t_A.shape[0] == t_B.shape[0]:
        t_A, _ = torch.sort(t_A)
        t_B, _ = torch.sort(t_B)
        return torch.mean(torch.abs(t_A - t_B)).item()

    # If shapes differ, align them via a common quantile grid to approximate the continuous integral
    min_len = min(t_A.shape[0], t_B.shape[0])
    quantiles = torch.linspace(0, 1, steps=min_len, device=device)
    t_A = torch.quantile(t_A, quantiles)
    t_B = torch.quantile(t_B, quantiles)

    return torch.mean(torch.abs(t_A - t_B)).item()


# --- 2. Transformation Helper ---


def arcsinh_transform(x, c=1.0):
    """
    Inverse hyperbolic sine transformation for variance stabilization.
    Suitable for zero-inflated and negative-allowing atmospheric variables.
    """
    return np.arcsinh(x / c)


# --- 3. Helper Functions ---


class RawClimateDataset(Dataset):
    def __init__(self, cluster_path, split, dynamic_vars, static_vars):
        zarr_split = "train" if split == "validation" else split

        _ds_in = xr.open_zarr(os.path.join(cluster_path, f"{zarr_split}_data_in.zarr"))
        self.ds_in = _ds_in.chunk({"time": 1})

        self.ds_static = xr.open_dataset(os.path.join(cluster_path, "static_variables.nc")).load()

        self.dynamic_vars = dynamic_vars
        self.static_vars = static_vars
        self.num_samples = len(self.ds_in["time"])

        all_indices = np.arange(self.num_samples)
        rng = np.random.RandomState(42)
        rng.shuffle(all_indices)
        split_point = int(self.num_samples * (1.0 - 0.2))

        if split == "train":
            self.indices = all_indices[:split_point]
        else:
            self.indices = all_indices[split_point:]

        self.split_num_samples = len(self.indices)

    def __len__(self):
        return self.split_num_samples

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        var_list = []

        for var in self.dynamic_vars:
            data = self.ds_in[var].isel(time=true_idx).values
            if var == "tp":
                data = data * 1000.0
            var_list.append(data)

        for var in self.static_vars:
            data = self.ds_static[var].values
            var_list.append(data)

        x = np.stack(var_list, axis=0)
        return torch.from_numpy(x).float()


def load_raw_samples(cluster_path, split, dynamic_vars, static_vars, num_samples_to_draw, n_workers=4):
    try:
        dataset = RawClimateDataset(
            cluster_path=cluster_path, split=split, dynamic_vars=dynamic_vars, static_vars=static_vars
        )
    except Exception as e:
        print(f"Error loading dataset for {cluster_path}: {e}. Returning empty array.")
        return np.array([])

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
    for x_batch in tqdm(loader, desc=f"Loading {num_samples_to_draw} samples", leave=False):
        all_x.append(x_batch.numpy())

    if not all_x:
        return np.array([])

    return np.concatenate(all_x, axis=0)


def plot_matrix(matrix, title, labels, save_path):
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


# --- 4. Main Script ---


def main():
    # Configure hardware accelerator
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: Hardware acceleration not found, falling back to CPU.")

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
    N_WORKERS = config.get("NUM_WORKERS", 4)

    output_dir = os.path.join(current_dir, "covariate_shift_analysis")
    os.makedirs(output_dir, exist_ok=True)

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

    metric_name = "Wasserstein_1D"

    for analysis_type in ["raw", "normalized"]:
        print(f"\n{'='*70}")
        print(f"--- STARTING ANALYSIS: {analysis_type.upper()} DATA ---")
        print(f"{'='*70}\n")

        analysis_output_dir = os.path.join(output_dir, analysis_type)
        os.makedirs(analysis_output_dir, exist_ok=True)

        for split in ["test"]:
            print(f"\n--- Computing {metric_name} for {split} split on {device} ---")
            dist_matrix = np.zeros((N_VARS, N_DOMAINS, N_DOMAINS))

            for k in tqdm(range(N_VARS), desc="Variables"):
                current_var = VAR_NAMES[k]
                for i in range(N_DOMAINS):
                    for j in range(i + 1, N_DOMAINS):

                        samples_A_all_vars = data_cache[split][DOMAINS[i]]
                        samples_B_all_vars = data_cache[split][DOMAINS[j]]

                        if samples_A_all_vars.ndim < 4 or samples_B_all_vars.ndim < 4:
                            if i == 0 and j == 1 and k == 0:
                                print(f"Warning: Data for {DOMAINS[i]} or {DOMAINS[j]} is invalid.")
                            dist_matrix[k, i, j] = np.nan
                            dist_matrix[k, j, i] = np.nan
                            continue

                        samples_A = samples_A_all_vars[:, k].ravel()
                        samples_B = samples_B_all_vars[:, k].ravel()

                        if analysis_type == "normalized":
                            if current_var in ["tp", "cp", "lsp", "q", "total_precipitation"]:
                                samples_A = arcsinh_transform(samples_A)
                                samples_B = arcsinh_transform(samples_B)
                            else:
                                std_A = np.std(samples_A) + 1e-8
                                if std_A > 1e-6:
                                    samples_A = (samples_A - np.mean(samples_A)) / std_A

                                std_B = np.std(samples_B) + 1e-8
                                if std_B > 1e-6:
                                    samples_B = (samples_B - np.mean(samples_B)) / std_B

                        dist = wasserstein1d_gpu(samples_A, samples_B, device=device)
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
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
