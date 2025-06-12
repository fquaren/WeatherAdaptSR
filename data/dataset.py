import os
import re
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import gc

class SingleVariableDataset_v6(Dataset):
    def __init__(self, data_dir, elev_dir, split="train", use_theta_e=False, device="cuda"):
        """
        Dataset loading all data directly onto GPU memory.

        Args:
            data_dir (str): Directory containing input/target .npy files.
            elev_dir (str): Directory with individual elevation .npy files.
            split (str): 'train', 'val', etc.
            use_theta_e (bool): If True, use theta_e files.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device
        self.suffix = "theta_e" if use_theta_e else "T_2M"

        # Load full input & target into GPU
        input_path = os.path.join(data_dir, f"{split}_{self.suffix}_input_normalized_interp8x_bicubic.npy")
        target_path = os.path.join(data_dir, f"{split}_{self.suffix}_target_normalized.npy")
        location_path = os.path.join(data_dir, f"{split}_LOCATION.npy")

        self.input_data = torch.tensor(np.load(input_path), dtype=torch.float32).unsqueeze(1).to(device)
        self.target_data = torch.tensor(np.load(target_path), dtype=torch.float32).unsqueeze(1).to(device)
        self.locations = np.load(location_path)

        self.elev_dir = elev_dir
        self.elev_cache = {}

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        location_name = tuple(self.locations[idx])
        if location_name not in self.elev_cache:
            elev_path = os.path.join(self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy")
            elev_array = np.load(elev_path)
            self.elev_cache[location_name] = torch.tensor(elev_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        elev_sample = self.elev_cache[location_name]
        
        return input_sample, elev_sample, target_sample
    
    def unload_from_gpu(self):
        del self.input_data
        del self.target_data

        for key in list(self.elev_cache.keys()):
            del self.elev_cache[key]
        self.elev_cache.clear()  # Just to be safe

        torch.cuda.empty_cache()
        gc.collect()


class SingleVariableDataset_v5(Dataset):
    def __init__(self, data_dir, elev_dir, split="train", use_theta_e=False):
        """
        Dataset using memory mapping + efficient elevation loading.

        Args:
            data_dir (str): Base dir with input/target .npy files.
            split (str): Dataset split ('train', 'val', etc).
            use_theta_e (bool): Use theta_e variable if True.
            elev_dir (str): Directory containing preconverted .npy elevation files.
            location_file (str): Path to e.g. train_LOCATION.npy.
        """
        self.suffix = "theta_e" if use_theta_e else "T_2M"

        # Memory-mapped input/target
        self.input_data = np.load(os.path.join(data_dir, f"{split}_{self.suffix}_input_bicubic.npy"), mmap_mode='r')
        self.target_data = np.load(os.path.join(data_dir, f"{split}_{self.suffix}_target_normalized.npy"), mmap_mode='r')
        self.location_file = os.path.join(data_dir, f"{split}_LOCATION.npy")
        self.elev_dir = elev_dir

        # Location metadata
        self.locations = np.load(self.location_file)  # shape (N, 2), dtype=int

        # Elevation cache: maps location name (e.g., "A_B") → torch.Tensor
        self.elev_cache = {}

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = torch.tensor(self.input_data[idx], dtype=torch.float32).unsqueeze(0)
        target_sample = torch.tensor(self.target_data[idx], dtype=torch.float32).unsqueeze(0)

        location_name = tuple(self.locations[idx])
        if location_name not in self.elev_cache:
            elev_path = os.path.join(self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy")
            elev_array = np.load(elev_path)
            self.elev_cache[location_name] = torch.tensor(elev_array, dtype=torch.float32).unsqueeze(0)

        elev_sample = self.elev_cache[location_name]

        return input_sample, elev_sample, target_sample


class SingleVariableDataset_v4(Dataset):
    def __init__(self, data_dir, split="train", use_theta_e=False):
        """
        Dataset that loads data lazily using memory mapping.

        Args:
            data_dir (str): Directory containing the .npy files.
            split (str): Dataset split prefix, e.g., 'train'.
            use_theta_e (bool): Whether to use equivalent potential temperature.
        """
        suffix = "theta_e" if use_theta_e else "T_2M"

        # File paths
        self.input_path = os.path.join(data_dir, f"{split}_{suffix}_input_bicubic.npy")
        self.elev_path = os.path.join(data_dir, f"{split}_HSURF.npy")
        self.target_path = os.path.join(data_dir, f"{split}_{suffix}_target_normalized.npy")

        # Use memory-mapped arrays to avoid loading all data into RAM
        self.input_data = np.load(self.input_path, mmap_mode='r')
        self.elev_data = np.load(self.elev_path, mmap_mode='r')
        self.target_data = np.load(self.target_path, mmap_mode='r')

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.elev_data[idx], self.target_data[idx]


class SingleVariableDataset_v3(Dataset):
    def __init__(self, data_dir, split="train", use_theta_e=False, device="cpu"):
        """
        Args:
            data_dir (str): Directory containing the .npy files.
            split (str): Dataset split prefix, e.g., 'train'.
            transform (str): Optional transformation identifier, e.g., 'theta_e'.
        """
        suffix = "theta_e" if use_theta_e else "T_2M"

        self.input = torch.tensor(
            np.load(f"{data_dir}/{split}_{suffix}_input_bicubic.npy"),
            dtype=torch.float32,
        ).unsqueeze(1).to(device)

        self.elev = torch.tensor(
            np.load(os.path.join(data_dir, f'{split}_HSURF.npy')),
            dtype=torch.float32
        ).unsqueeze(1).to(device)

        self.target = torch.tensor(
            np.load(f"{data_dir}/{split}_{suffix}_target_normalized.npy"),
            dtype=torch.float32,
        ).unsqueeze(1).to(device)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.elev[idx], self.target[idx]
    
    def unload_from_gpu(self):
        self.input = self.input.to("cpu")
        self.target = self.target.to("cpu")
        self.elev = self.elev.to("cpu")


