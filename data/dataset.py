import torch
import torch.nn.functional as F
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
        subset_size=None,
    ):
        if split not in ["train", "validation", "test"]:
            raise ValueError("split must be 'train', 'validation', or 'test'")

        self.input_vars = input_vars or ["cape", "cp", "sp", "tclw", "tcw", "tisr", "tp", "u", "v"]
        self.output_var = output_var
        self.static_vars = static_vars
        self.stats = normalization_stats

        file_prefix = "test_data" if split == "test" else "train_data"

        ds_in = xr.open_zarr(f"{cluster_path}/{file_prefix}_in.zarr", consolidated=True)
        ds_static = xr.open_dataset(f"{cluster_path}/static_variables.nc").load()

        self.num_samples_total = ds_in.sizes["time"]
        all_indices = np.arange(self.num_samples_total)

        if subset_size is not None and subset_size < self.num_samples_total:
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)
            all_indices = all_indices[:subset_size]
            self.num_samples_total = subset_size
        elif split in ["train", "validation"]:
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)

        if split == "test":
            self.indices = all_indices
        else:
            split_point = int(self.num_samples_total * (1 - validation_split_pct))
            self.indices = all_indices[:split_point] if split == "train" else all_indices[split_point:]

        self.indices = np.sort(self.indices)

        static_data_list = []
        for var in self.static_vars:
            data = ds_static[var].values
            static_data_list.append(self._normalize(data, var))
        self.static_tensor = torch.from_numpy(np.stack(static_data_list, axis=0)).float()

        print(f"Loading {len(self.indices)} samples into RAM for {split} split...")

        z_in = zarr.open(f"{cluster_path}/{file_prefix}_in.zarr", mode="r")
        z_out = zarr.open(f"{cluster_path}/{file_prefix}_out.zarr", mode="r")

        self.x_data = []
        self.y_data = []

        for idx in self.indices:
            var_data = []
            for var in self.input_vars:
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
        if var_name in ["tp", "cp", "lsp", "cape", "tclw", "precipitation"]:
            return np.arcsinh(data)

        if var_name == "lsm":
            return data

        if var_name in self.stats:
            mean, std = self.stats[var_name]
        else:
            mean = np.mean(data)
            std = np.std(data)

        return (data - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        x_dynamic = torch.from_numpy(self.x_data[idx]).float()
        y_target = torch.from_numpy(self.y_data[idx]).float().unsqueeze(0)

        # --- CRITICAL FIX: Spatial Interpolation ---
        # Upsample the coarse 160x160 dynamic variables to match the 200x200 static/target grid
        # Requires unsqueeze to simulate batch dimension for F.interpolate, then squeeze back
        x_dynamic = x_dynamic.unsqueeze(0)
        x_dynamic = F.interpolate(x_dynamic, size=(200, 200), mode="bicubic", align_corners=False)
        x_dynamic = x_dynamic.squeeze(0)

        return x_dynamic, self.static_tensor, y_target