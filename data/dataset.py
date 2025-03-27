import os
import re
import torch
import xarray as xr
from torch.utils.data import Dataset


class SingleVariableDataset_v2(Dataset):
    def __init__(self, variable, input_files, target_files, elev_dir):
        self.variable = variable
        self.input_files = input_files
        self.target_files = target_files
        self.elev_dir = elev_dir
        self.elev_files = self._map_elevation_files()

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

        return input_data, elevation_data, target_data


class SingleVariableDataset(Dataset):
    def __init__(self, variable, input_files, elev_dir, target_files):
        """
        Args:
            input_files (list): Paths to low-resolution variable (temperature) NetCDF files.
            elev_dir (str): Directory containing elevation NetCDF files named as "A_B_something.nc".
            target_files (list): Paths to corresponding high-resolution target NetCDF files.
        """
        self.variable = variable
        self.input_files = input_files
        self.target_files = target_files
        self.elev_dir = elev_dir

        # Preload the mapping of elevation files for faster access
        self.elev_files = self._map_elevation_files()

    def _extract_numbers(self, filename):
        """Extracts A and B from a filename like '3_6_lffd20101208160000.nz'."""
        match = re.match(r"(\d{1,2})_(\d{1,2})_", os.path.basename(filename))  # Matches 0-11 correctly
        if match:
            A, B = int(match.group(1)), int(match.group(2))
            if 0 <= A <= 11 and 0 <= B <= 11:
                return A, B
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_*.nz")

    def _map_elevation_files(self):
        """Creates a mapping of (A, B) -> elevation file path, ensuring exact matches."""
        elev_files = {}
        for file in os.listdir(self.elev_dir):
            if file.endswith(".nc"):
                try:
                    A, B = self._extract_numbers(file)
                    elev_files[(A, B)] = os.path.join(self.elev_dir, file)
                except ValueError:
                    continue  # Ignore files that don't match the pattern
        return elev_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """Loads variable input, elevation, and corresponding high-resolution target"""
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        # Extract A and B from input filename
        A, B = self._extract_numbers(input_file)

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        # Load variable data
        with xr.open_dataset(input_file) as ds:
            input = torch.tensor(ds[self.variable].sel(time=ds[self.variable].time[0]).values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Load high-resolution target (same variable as input)
        with xr.open_dataset(target_file) as ds:
            target = torch.tensor(ds[self.variable].sel(time=ds[self.variable].time[0]).values, dtype=torch.float32).unsqueeze(0)  # [1, H_target, W_target]

        # Load the correct elevation data
        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return input, elevation_data, target