import os
import re
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


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


