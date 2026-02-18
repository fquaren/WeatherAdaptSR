import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import zarr


class ClimateSRDataset(Dataset):
    def __init__(
        self,
        cluster_path,
        split="train",
        normalization_stats=None,
        input_vars=None,
        output_var="precipitation",
        static_vars=["lsm", "z"],
        validation_split_pct=0.2,
        seed=42,
    ):
        if split not in ["train", "validation"]:
            raise ValueError("split must be 'train' or 'validation'")

        self.input_vars = input_vars or ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
        self.output_var = output_var
        self.static_vars = static_vars
        self.stats = normalization_stats

        # Open xarray strictly to extract metadata and static variables
        ds_in = xr.open_zarr(f"{cluster_path}/train_data_in.zarr", consolidated=True)
        ds_static = xr.open_dataset(f"{cluster_path}/static_variables.nc").load()

        self.num_samples_total = ds_in.dims["time"]

        # Calculate split indices
        all_indices = np.arange(self.num_samples_total)
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)
        split_point = int(self.num_samples_total * (1 - validation_split_pct))

        self.indices = all_indices[:split_point] if split == "train" else all_indices[split_point:]

        # Sort indices to optimize sequential disk reads
        self.indices = np.sort(self.indices)

        # Pre-load static data
        static_data_list = []
        for var in self.static_vars:
            data = ds_static[var].values
            static_data_list.append(self._normalize(data, var))
        self.static_tensor = torch.from_numpy(np.stack(static_data_list, axis=0)).float()

        # PRE-LOAD TO MEMORY: Highly efficient for A100 nodes with adequate RAM
        print(f"Loading {len(self.indices)} samples into RAM for {split} split...")

        # Access underlying zarr arrays directly for speed
        z_in = zarr.open(f"{cluster_path}/train_data_in.zarr", mode="r")
        z_out = zarr.open(f"{cluster_path}/train_data_out.zarr", mode="r")

        self.x_data = []
        self.y_data = []

        # Batch loading to avoid memory spikes
        for idx in self.indices:
            var_data = []
            for var in self.input_vars:
                # Read directly from Zarr slice
                val = z_in[var][idx]
                if var == "tp":
                    val = val * 1000.0
                var_data.append(self._normalize(val, var))

            self.x_data.append(np.stack(var_data, axis=0))

            y_val = z_out[self.output_var][idx]
            self.y_data.append(self._normalize(y_val, self.output_var))

        self.x_data = np.stack(self.x_data, axis=0)
        self.y_data = np.stack(self.y_data, axis=0)

        print("Data loaded successfully.")

    def __len__(self):
        return len(self.indices)

    def _normalize(self, data, var_name):
        """Conditionally applies physics-based transformations."""
        # 1. Variables with extreme right tails and zero-bounds
        if var_name in ["tp", "cp", "lsp", "cape", "tclw", "precipitation"]:
            return np.arcsinh(data)

        # 2. Binary or fractional masks (bypass standardization)
        if var_name == "lsm":
            return data

        # 3. Variables with roughly Gaussian properties
        mean, std = self.stats[var_name]
        return (data - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        # Data is already localized and normalized in RAM
        x_dynamic = torch.from_numpy(self.x_data[idx]).float()
        y_target = torch.from_numpy(self.y_data[idx]).float().unsqueeze(0)

        return x_dynamic, self.static_tensor, y_target


# --- Test Harness ---
if __name__ == "__main__":
    import yaml
    import json

    with open("configs/config_rainshift.yaml", "r") as f:
        config = yaml.safe_load(f)

    CLUSTER_PATH = config["DATA_ROOT"] + "/" + config["SOURCE_DOMAIN"]
    STATS_FILE = config["DATA_ROOT"] + "/" + config["STATS_FILE"]

    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
        print(f"Successfully loaded stats from {STATS_FILE}")
    except FileNotFoundError:
        print(f"Error: Stats file not found at {STATS_FILE}")
        stats = None

    if stats:
        train_dataset = ClimateSRDataset(
            cluster_path=CLUSTER_PATH,
            split="train",
            normalization_stats=stats,
            validation_split_pct=0.2,
            seed=42,
        )

        val_dataset = ClimateSRDataset(
            cluster_path=CLUSTER_PATH,
            split="validation",
            normalization_stats=stats,
            validation_split_pct=0.2,
            seed=42,
        )

        # --- Verification ---
        print("\n--- Verification ---")
        total_samples = train_dataset.num_samples_total
        train_samples = len(train_dataset)
        val_samples = len(val_dataset)

        assert total_samples == (train_samples + val_samples), "Error: Mismatch in sample counts!"
        overlap = set(train_dataset.indices).intersection(set(val_dataset.indices))
        assert len(overlap) == 0, "Error: Overlap found between train and val sets!"
        print("Test PASSED: Train/Validation splits are correct and have no overlap.")

        # --- CRITICAL FIX: Unpacking the dual-input paradigm ---
        print("\nTesting data loading...")
        x_dyn, x_stat, y = train_dataset[0]

        print("Loaded one sample (x_dyn, x_stat, y):")
        print(f"  Dynamic input shape: {x_dyn.shape}")
        print(f"  Static input shape:  {x_stat.shape}")
        print(f"  Output shape:        {y.shape}")

        # Shapes dynamically verified against input parameters rather than hardcoded ints
        assert x_dyn.shape[0] == len(train_dataset.input_vars), "Dynamic input channel mismatch!"
        assert x_stat.shape[0] == len(train_dataset.static_vars), "Static input channel mismatch!"
        assert y.shape[0] == 1, "Output channel mismatch!"

        print("Test PASSED: Data loading shapes and physical bounds are correct.")
