import os
import re
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


def compute_equivalent_potential_temperature(T, RH, P_surf):
    """Compute equivalent potential temperature θ_e (K) for GPU-accelerated tensors."""
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    cp = 1005    # Specific heat of dry air (J/kg/K)
    epsilon = 0.622  # Ratio of molecular weights of water vapor to dry air

    # Compute saturation vapor pressure (hPa) using Tetens' formula
    e_s = 6.112 * torch.exp((17.67 * T) / (T + 243.5))
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

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        with xr.open_dataset(input_file) as ds:
            input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

        with xr.open_dataset(target_file) as ds:
            target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)

        if self.transform == "theta_e":
            # Load the necessary data
            with xr.open_dataset(input_file) as ds:
                input_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                input_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            # Apply transfromation
            input_data = compute_equivalent_potential_temperature(input_data, input_RH, input_P_surf)
            with xr.open_dataset(target_file) as ds:
                target_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                target_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            # Apply transfromation
            target_data = compute_equivalent_potential_temperature(target_data, target_RH, target_P_surf)
        
        return input_data, elevation_data, target_data