import torch
import xarray as xr
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class ClimateSRDataset(Dataset):
    """
    Custom PyTorch Dataset for lazily loading the Zarr climate data.
    This version handles the 2.5x super-resolution task by
    pre-upsampling low-resolution inputs to match the high-resolution output.

    This class handles splitting the 'train_data_...' files into
    a training and a validation set based on a random, seeded shuffle.
    """

    def __init__(
        self,
        cluster_path,
        split="train",
        normalization_stats=None,
        input_vars=None,
        output_var="precipitation",
        static_vars=["lsm", "z"],
        # New arguments for validation splitting
        validation_split_pct=0.2,
        seed=42,
    ):

        # Split now refers to 'train' or 'validation' subset
        if split not in ["train", "validation"]:
            raise ValueError("split must be 'train' or 'validation'")

        if input_vars is None:
            # All 9 dynamic input variables from train_data_in.zarr
            self.input_vars = [
                "cape",
                "cp",
                "sp",
                "tclw",
                "tcw",
                "tisr",
                "tp",
                "u",
                "v",
            ]
        else:
            self.input_vars = input_vars

        self.output_var = output_var
        self.static_vars = static_vars

        # 1. Open Zarr stores lazily
        zarr_in_path = f"{cluster_path}/train_data_in.zarr"
        zarr_out_path = f"{cluster_path}/train_data_out.zarr"

        try:
            # Open the dataset, respecting its native on-disk chunking
            _ds_in = xr.open_zarr(zarr_in_path)
            _ds_out = xr.open_zarr(zarr_out_path)

            # Now, tell Dask to re-chunk the data in memory to match our access pattern.
            # This is far more efficient than overriding the chunks at load time.
            self.ds_in = _ds_in.chunk({"time": 1})
            self.ds_out = _ds_out.chunk({"time": 1})
        except Exception as e:
            print(f"Error opening Zarr stores at: {cluster_path}")
            print("Please ensure the path is correct and Zarr files exist.")
            raise e

        # 2. Load static NetCDF (it's small)
        self.ds_static = xr.open_dataset(f"{cluster_path}/static_variables.nc").load()

        # 3. Get number of samples and target shape
        self.num_samples_total = len(self.ds_in["time"])
        self.target_shape = (len(self.ds_static["latitude"]), len(self.ds_static["longitude"]))

        # 4. Store normalization stats
        if normalization_stats is None:
            raise ValueError("normalization_stats must be provided.")
        self.stats = normalization_stats

        self.log_transform_epsilon = 1e-6

        # --- 5. Create deterministic train/validation split ---
        all_indices = np.arange(self.num_samples_total)

        # Use a numpy RandomState for reproducible shuffling
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)

        split_point = int(self.num_samples_total * (1 - validation_split_pct))

        if split == "train":
            self.indices = all_indices[:split_point]
            print(f"Loaded 'train' split: {len(self.indices)} samples.")
        else:  # split == "validation"
            self.indices = all_indices[split_point:]
            print(f"Loaded 'validation' split: {len(self.indices)} samples.")
        # --- End Modification ---

    def __len__(self):
        # Return the length of the subset of indices
        return len(self.indices)

    def _normalize(self, data, var_name):
        mean, std = self.stats[var_name]
        return (data - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        """
        Fetch a single sample (timestep) from the Zarr store.
        """

        # Map the relative index 'idx' to the true Zarr index
        zarr_index = self.indices[idx]

        # --- 1. Load Dynamic Inputs (Low-Res) ---
        input_data_list = []
        for var in self.input_vars:
            # Use zarr_index
            data = self.ds_in[var].isel(time=zarr_index).values

            # --- CRITICAL: Unit Conversion ---
            if var == "tp":
                data = data * 1000.0  # Convert (m) to (mm)
            # --- End Unit Conversion ---

            data_normalized = self._normalize(data, var)
            input_data_list.append(data_normalized)

        # Stack into (C_low_res, H_low, W_low)
        x_low_res = np.stack(input_data_list, axis=0)

        # Convert to tensor for upsampling
        x_low_res_tensor = torch.from_numpy(x_low_res).float()

        # --- 2. Upsample Inputs ---
        x_high_res_tensor = F.interpolate(
            x_low_res_tensor.unsqueeze(0),
            size=self.target_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # --- 3. Load Static Inputs (High-Res) ---
        static_data_list = []
        for var in self.static_vars:
            data = self.ds_static[var].values
            data_normalized = self._normalize(data, var)
            static_data_list.append(data_normalized)

        x_static_tensor = torch.from_numpy(np.stack(static_data_list, axis=0)).float()

        # --- 4. Concatenate Inputs ---
        # (9, 200, 200) + (2, 200, 200) -> (11, 200, 200)
        x_final = torch.cat([x_high_res_tensor, x_static_tensor], axis=0)

        # --- 5. Load Output (High-Res) ---
        # Use zarr_index
        y_raw = self.ds_out[self.output_var].isel(time=zarr_index).values

        # Apply log-transform and normalization
        y_log = np.log(y_raw + self.log_transform_epsilon)
        y_norm = self._normalize(y_log, self.output_var)

        # Add a channel dimension: (H, W) -> (1, H, W)
        y_final = torch.from_numpy(y_norm).float().unsqueeze(0)

        return x_final, y_final


# --- Test Harness ---
if __name__ == "__main__":
    """
    This block demonstrates how to use the modified Dataset class
    and verifies that the train/validation splits are correct.
    """

    # load config
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
        print("Please run compute_stats.py first and update the path.")
        stats = None
    except Exception as e:
        print(f"Error loading stats: {e}")
        stats = None

    if stats:
        print("\n--- Instantiating Training Set ---")
        train_dataset = ClimateSRDataset(
            cluster_path=CLUSTER_PATH,
            split="train",
            normalization_stats=stats,
            validation_split_pct=0.2,  # 80% for train
            seed=42,
        )

        print("\n--- Instantiating Validation Set ---")
        val_dataset = ClimateSRDataset(
            cluster_path=CLUSTER_PATH,
            split="validation",
            normalization_stats=stats,
            validation_split_pct=0.2,  # 20% for val
            seed=42,  # Using the SAME seed is crucial
        )

        # --- Verification ---
        print("\n--- Verification ---")
        total_samples = train_dataset.num_samples_total
        train_samples = len(train_dataset)
        val_samples = len(val_dataset)

        print(f"Total samples in source:   {total_samples}")
        print(f"Training subset samples:   {train_samples}")
        print(f"Validation subset samples: {val_samples}")
        print(f"Total in subsets:          {train_samples + val_samples}")

        assert total_samples == (train_samples + val_samples), "Error: Mismatch in sample counts!"

        # Check for overlap
        train_indices_set = set(train_dataset.indices)
        val_indices_set = set(val_dataset.indices)

        overlap = train_indices_set.intersection(val_indices_set)

        assert len(overlap) == 0, "Error: Overlap found between train and val sets!"

        print("Test PASSED: Train/Validation splits are correct and have no overlap.")

        # Test loading one item
        print("\nTesting data loading...")
        x, y = train_dataset[0]
        print("Loaded one sample (X, y):")
        print(f"  Input shape:  {x.shape}")  # Should be (11, 200, 200)
        print(f"  Output shape: {y.shape}")  # Should be (1, 200, 200)

        assert x.shape == (11, 200, 200), "Input shape mismatch!"
        assert y.shape == (1, 200, 200), "Output shape mismatch!"

        print("Test PASSED: Data loading shapes are correct.")
