import os
import re
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


def compute_equivalent_potential_temperature(T, RH, P_surf):
    """Compute equivalent potential temperature θ_e (K) for GPU-accelerated tensors.
    
    Args:
        T (Tensor): Temperature in Kelvin (K)
        RH (Tensor): Relative humidity (0 to 1)
        P_surf (Tensor): Surface pressure in hPa
        
    Returns:
        Tensor: Equivalent potential temperature θ_e in Kelvin (K)
    """
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    cp = 1005    # Specific heat of dry air (J/kg/K)
    epsilon = 0.622  # Ratio of molecular weights of water vapor to dry air

    # Convert temperature from Kelvin to Celsius for the saturation vapor pressure formula
    T_Celsius = T - 273.15

    # Compute saturation vapor pressure (hPa) using Tetens' formula
    e_s = 6.112 * torch.exp((17.67 * T_Celsius) / (T_Celsius + 243.5))
    e = RH * e_s  # Actual vapor pressure (hPa)
    
    # Mixing ratio (kg/kg)
    w = epsilon * (e / (P_surf - e))
    
    # Compute potential temperature θ
    theta = T * (1000 / P_surf) ** 0.286
    
    # Compute equivalent potential temperature θ_e
    theta_e = theta * torch.exp((L_v * w) / (cp * T))
    
    return theta_e


# class SingleVariableDataset_v2(Dataset):
#     def __init__(self, variable, input_files, target_files, elev_dir, transform=None, normalization_stats=None):
#         self.variable = variable
#         self.input_files = input_files
#         self.target_files = target_files
#         self.elev_dir = elev_dir
#         self.elev_files = self._map_elevation_files()
#         self.transform = transform
#         self.normalization_stats = normalization_stats  # Dict containing means & stds

#     def _extract_numbers(self, filename):
#         match = re.match(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})", os.path.basename(filename))
#         if match:
#             A, B, year, month = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
#             return A, B, year, month
#         raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
#     def _extract_numbers_from_dem(self, filename):
#         match = re.match(r"(\d{1,2})_(\d{1,2})_", os.path.basename(filename))
#         if match:
#             A, B = int(match.group(1)), int(match.group(2))
#             return A, B
#         raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
#     def _map_elevation_files(self):
#         """Creates a mapping of (A, B) -> elevation file path, ensuring exact matches."""
#         elev_files = {}
#         for file in os.listdir(self.elev_dir):
#             if file.endswith(".nc"):
#                 try:
#                     A, B = self._extract_numbers_from_dem(file)
#                     elev_files[(A, B)] = os.path.join(self.elev_dir, file)
#                 except ValueError:
#                     continue  # Ignore files that don't match the pattern
#         return elev_files

#     def __len__(self):
#         return len(self.input_files)

#     def __getitem__(self, idx):
#         input_file = self.input_files[idx]
#         target_file = self.target_files[idx]
#         A, B, _, _ = self._extract_numbers(input_file)

#         # Ensure the correct elevation file exists
#         if (A, B) not in self.elev_files:
#             raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

#         elev_file = self.elev_files[(A, B)]

#         with xr.open_dataset(input_file) as ds:
#             input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

#         with xr.open_dataset(target_file) as ds:
#             target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

#         with xr.open_dataset(elev_file) as ds:
#             elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)

#         if self.transform == "theta_e":
#             # Load the necessary data
#             with xr.open_dataset(input_file) as ds:
#                 input_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
#                 input_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
#             # Apply transformation
#             input_data = compute_equivalent_potential_temperature(input_data, input_RH, input_P_surf)
#             with xr.open_dataset(target_file) as ds:
#                 target_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
#                 target_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
#             # Apply transformation
#             target_data = compute_equivalent_potential_temperature(target_data, target_RH, target_P_surf)

#         # Apply normalization if statistics are provided
#         if self.normalization_stats:
#             input_mean, input_std = self.normalization_stats["input"]
#             target_mean, target_std = self.normalization_stats["target"]
#             elev_mean, elev_std = self.normalization_stats["elevation"]

#             input_data = (input_data - input_mean) / input_std
#             target_data = (target_data - target_mean) / target_std
#             elevation_data = (elevation_data - elev_mean) / elev_std

#         return input_data, elevation_data, target_data


class SingleVariableDataset_v2(Dataset):
    def __init__(self, variable, input_files, target_files, elev_dir, transform=None):
        self.variable = variable
        self.input_files = input_files
        self.target_files = target_files
        self.elev_dir = elev_dir
        self.elev_files = self._map_elevation_files()
        self.transform = transform

    def _extract_numbers(self, filename):
        match = re.match(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})", os.path.basename(filename))
        if match:
            A, B, year, month = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return A, B, year, month
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
    def _extract_numbers_from_dem(self, filename):
        match = re.match(r"(\d{1,2})_(\d{1,2})_", os.path.basename(filename))
        if match:
            A, B = int(match.group(1)), int(match.group(2))
            return A, B
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
    def _map_elevation_files(self):
        """Creates a mapping of (A, B) -> elevation file path, ensuring exact matches."""
        elev_files = {}
        for file in os.listdir(self.elev_dir):
            if file.endswith(".nc"):
                try:
                    A, B = self._extract_numbers_from_dem(file)
                    elev_files[(A, B)] = os.path.join(self.elev_dir, file)
                except ValueError:
                    continue  # Ignore files that don't match the pattern
        return elev_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]
        A, B, _, _ = self._extract_numbers(input_file)
        # filename = os.path.basename(input_file)

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)

        if self.transform == "theta_e":
            # Load the necessary data
            with xr.open_dataset(input_file) as ds:
                input_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                input_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            input_data = compute_equivalent_potential_temperature(input_data, input_RH, input_P_surf)
            with xr.open_dataset(target_file) as ds:
                target_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                target_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            # Apply transfromation
            target_data = compute_equivalent_potential_temperature(target_data, target_RH, target_P_surf)
        else:
            with xr.open_dataset(input_file) as ds:
                input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            with xr.open_dataset(target_file) as ds:
                target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
        
        return input_data, elevation_data, target_data #, filename


class SingleVariableDataset_v3(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Directory containing the .npy files.
            split (str): Dataset split prefix, e.g., 'train'.
            transform (str): Optional transformation identifier, e.g., 'theta_e'.
        """
        self.transform = transform
        self.split = split
        self.data_dir = data_dir

        # Load input variables
        self.T_input = np.load(os.path.join(data_dir, f'{split}_T_2M_input.npy'))          # (N, H, W)
        self.P_input = np.load(os.path.join(data_dir, f'{split}_PS_input.npy'))            # (N, H, W)
        self.RH_input = np.load(os.path.join(data_dir, f'{split}_RELHUM_2M_input.npy'))    # (N, H, W)

        # Load target variables
        self.T_target = np.load(os.path.join(data_dir, f'{split}_T_2M_target.npy'))        # (N, H, W)
        self.P_target = np.load(os.path.join(data_dir, f'{split}_PS_target.npy'))          # (N, H, W)
        self.RH_target = np.load(os.path.join(data_dir, f'{split}_RELHUM_2M_target.npy'))  # (N, H, W)

        # Elevation
        self.elevation = np.load(os.path.join(data_dir, f'{split}_HSURF.npy'))             # (N, H, W)

        # Optional metadata
        self.locations = np.load(os.path.join(data_dir, f'{split}_LOCATION.npy'))
        self.filenames = np.load(os.path.join(data_dir, f'{split}_FILENAME.npy'))

        self.length = self.T_input.shape[0]
        for arr in [self.P_input, self.RH_input, self.T_target, self.P_target, self.RH_target, self.elevation]:
            if arr.shape[0] != self.length:
                raise ValueError("All variables must have the same number of samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        T_in = torch.tensor(self.T_input[idx], dtype=torch.float32)
        RH_in = torch.tensor(self.RH_input[idx], dtype=torch.float32)
        P_in = torch.tensor(self.P_input[idx], dtype=torch.float32)

        T_out = torch.tensor(self.T_target[idx], dtype=torch.float32)
        RH_out = torch.tensor(self.RH_target[idx], dtype=torch.float32)
        P_out = torch.tensor(self.P_target[idx], dtype=torch.float32)

        elev = torch.tensor(self.elevation[idx], dtype=torch.float32).unsqueeze(0)

        if self.transform == "theta_e":
            input_tensor = compute_equivalent_potential_temperature(T_in.unsqueeze(0), RH_in.unsqueeze(0), P_in.unsqueeze(0))
            target_tensor = compute_equivalent_potential_temperature(T_out.unsqueeze(0), RH_out.unsqueeze(0), P_out.unsqueeze(0))
        else:
            input_tensor = T_in.unsqueeze(0)
            target_tensor = T_out.unsqueeze(0)

        return input_tensor, elev, target_tensor


class GPUMemoryDataset(Dataset):
    def __init__(self, data_dir, split="train", use_theta_e=False, device="cuda"):
        """
        Args:
            data_dir (str): Path to directory with preprocessed .npy files.
            split (str): Dataset split, e.g., 'train', 'val', 'test'.
            use_theta_e (bool): Whether to use θₑ instead of T_2M.
            device (str): Device to load the data to ('cuda' or 'cpu').
        """
        suffix = "theta_e" if use_theta_e else "T_2M"

        self.input = torch.tensor(
            np.load(f"{data_dir}/{split}_{suffix}_input_standardized.npy"),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        self.elev = torch.tensor(
            np.load(os.path.join(data_dir, f'{split}_HSURF.npy')),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        self.target = torch.tensor(
            np.load(f"{data_dir}/{split}_{suffix}_target_standardized.npy"),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

    def __len__(self):
        return self.input.shape[0]
    
    def unload_from_gpu(self):
        self.input = self.input.to("cpu")
        self.target = self.target.to("cpu")
        self.elev = self.elev.to("cpu")

    def __getitem__(self, idx):
        return self.input[idx], self.elev[idx], self.target[idx]