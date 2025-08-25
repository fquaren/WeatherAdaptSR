import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import gc


class SingleVariableDataset_v8(Dataset):
    def __init__(
        self,
        data_dir,
        elev_dir,
        split="train",
        device="cuda",
        use_theta_e=False,
        vars=None,
        augment=False,
        elev_mean=None,
        elev_std=None,
    ):
        """
        Dataset loading all data directly onto GPU memory, with optional normalization and augmentation.

        Args:
            data_dir (str): Directory containing input/target .npy files.
            elev_dir (str): Directory with individual elevation .npy files.
            split (str): 'train', 'val', etc.
            use_theta_e (bool): If True, use theta_e files.
            device (str): 'cuda' or 'cpu'.
            augment (bool): Whether to apply data augmentation (only effective if split=="train").
            temp_mean (float or tensor): Mean value for temperature normalization.
            temp_std (float or tensor): Std deviation for temperature normalization.
            elev_mean (float or tensor): Mean value for elevation normalization.
            elev_std (float or tensor): Std deviation for elevation normalization.
        """
        self.device = device
        self.split = split
        self.augment = augment

        input_list = []
        target_list = []

        for var in vars:
            input_path = os.path.join(
                data_dir, f"{split}_{var}_input_normalized_interp8x_bilinear.npy"
            )
            target_path = os.path.join(data_dir, f"{split}_{var}_target_normalized.npy")

            input_data = np.load(input_path)
            target_data = np.load(target_path)

            input_list.append(torch.tensor(input_data, dtype=torch.float32))
            target_list.append(torch.tensor(target_data, dtype=torch.float32))

        # Stack along channel dimension (1) so shape becomes [samples, channels, H, W]
        self.input_data = torch.stack(input_list, dim=1).to(device)
        self.target_data = torch.stack(target_list, dim=1).to(device)

        location_path = os.path.join(data_dir, f"{split}_LOCATION.npy")
        self.locations = np.load(location_path)

        self.elev_dir = elev_dir
        self.elev_cache = {}

        # Normalization elevation
        self.elev_mean = elev_mean
        self.elev_std = elev_std

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]
        location_name = tuple(self.locations[idx])

        # Load and normalize elevation
        if location_name not in self.elev_cache:
            elev_path = os.path.join(
                self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy"
            )
            elev_array = np.load(elev_path)
            elev_tensor = torch.tensor(elev_array, dtype=torch.float32).unsqueeze(0)
            if self.elev_mean is not None and self.elev_std is not None:
                elev_tensor = (elev_tensor - self.elev_mean) / self.elev_std
                elev_tensor = np.log(
                    elev_tensor * 1000 + 1e5
                )  # As in Harder et al. 2025
            self.elev_cache[location_name] = elev_tensor.to(self.device)

        elev_sample = self.elev_cache[location_name]

        # Apply augmentation if in train split
        if self.augment:
            input_sample, elev_sample, target_sample = self.apply_augmentations(
                input_sample, elev_sample, target_sample
            )

        full_input_sample = torch.cat([input_sample, elev_sample], dim=0)

        return full_input_sample, target_sample

    def apply_augmentations(self, input_sample, elev_sample, target_sample):
        # Horizontal flip
        if random.random() < 0.5:
            input_sample = torch.flip(input_sample, dims=[2])
            elev_sample = torch.flip(elev_sample, dims=[2])
            target_sample = torch.flip(target_sample, dims=[2])

        # Vertical flip
        if random.random() < 0.5:
            input_sample = torch.flip(input_sample, dims=[1])
            elev_sample = torch.flip(elev_sample, dims=[1])
            target_sample = torch.flip(target_sample, dims=[1])

        # Rotation by 0°, 90°, 180°, 270°
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            input_sample = torch.rot90(input_sample, k, dims=[1, 2])
            elev_sample = torch.rot90(elev_sample, k, dims=[1, 2])
            target_sample = torch.rot90(target_sample, k, dims=[1, 2])

        # Multiplicative noise to input
        if random.random() < 0.5:
            factor = torch.randn_like(input_sample) * 0.01 + 1.0
            input_sample *= factor

        return input_sample, elev_sample, target_sample

    def unload_from_gpu(self):
        del self.input_data
        del self.target_data
        for key in list(self.elev_cache.keys()):
            del self.elev_cache[key]
        self.elev_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


class SingleVariableDataset_v7(Dataset):
    def __init__(
        self,
        data_dir,
        elev_dir,
        split="train",
        use_theta_e=False,
        device="cuda",
        temp_mean=None,
        temp_std=None,
        elev_mean=None,
        elev_std=None,
    ):
        """
        Dataset loading all data directly onto GPU memory with optional normalization.
        """
        self.device = device
        self.suffix = "theta_e" if use_theta_e else "T_2M"

        # Load full input & target into GPU
        input_path = os.path.join(
            data_dir, f"{split}_{self.suffix}_input_interp8x_bicubic.npy"
        )
        target_path = os.path.join(data_dir, f"{split}_{self.suffix}_target.npy")
        location_path = os.path.join(data_dir, f"{split}_LOCATION.npy")

        self.input_data = (
            torch.tensor(np.load(input_path), dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.target_data = (
            torch.tensor(np.load(target_path), dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.locations = np.load(location_path)

        self.elev_dir = elev_dir
        self.elev_cache = {}

        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.elev_mean = elev_mean
        self.elev_std = elev_std

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        # Normalize temperature
        if self.temp_mean is not None and self.temp_std is not None:
            input_sample = (input_sample - self.temp_mean) / self.temp_std

        location_name = tuple(self.locations[idx])
        if location_name not in self.elev_cache:
            elev_path = os.path.join(
                self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy"
            )
            elev_array = np.load(elev_path)
            elev_tensor = torch.tensor(elev_array, dtype=torch.float32).unsqueeze(0)

            # Normalize elevation
            if self.elev_mean is not None and self.elev_std is not None:
                elev_tensor = (elev_tensor - self.elev_mean) / self.elev_std

            self.elev_cache[location_name] = elev_tensor.to(self.device)

        elev_sample = self.elev_cache[location_name]
        return input_sample, elev_sample, target_sample

    def unload_from_gpu(self):
        del self.input_data
        del self.target_data
        for key in list(self.elev_cache.keys()):
            del self.elev_cache[key]
        self.elev_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()


class SingleVariableDataset_v6(Dataset):
    def __init__(
        self, data_dir, elev_dir, split="train", use_theta_e=False, device="cuda"
    ):
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
        input_path = os.path.join(
            data_dir, f"{split}_{self.suffix}_input_normalized_interp8x_bicubic.npy"
        )
        target_path = os.path.join(
            data_dir, f"{split}_{self.suffix}_target_normalized.npy"
        )
        location_path = os.path.join(data_dir, f"{split}_LOCATION.npy")

        self.input_data = (
            torch.tensor(np.load(input_path), dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        self.target_data = (
            torch.tensor(np.load(target_path), dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
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
            elev_path = os.path.join(
                self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy"
            )
            elev_array = np.load(elev_path)
            self.elev_cache[location_name] = (
                torch.tensor(elev_array, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

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
        self.input_data = np.load(
            os.path.join(data_dir, f"{split}_{self.suffix}_input_bicubic.npy"),
            mmap_mode="r",
        )
        self.target_data = np.load(
            os.path.join(data_dir, f"{split}_{self.suffix}_target_normalized.npy"),
            mmap_mode="r",
        )
        self.location_file = os.path.join(data_dir, f"{split}_LOCATION.npy")
        self.elev_dir = elev_dir

        # Location metadata
        self.locations = np.load(self.location_file)  # shape (N, 2), dtype=int

        # Elevation cache: maps location name (e.g., "A_B") → torch.Tensor
        self.elev_cache = {}

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input_sample = torch.tensor(
            self.input_data[idx], dtype=torch.float32
        ).unsqueeze(0)
        target_sample = torch.tensor(
            self.target_data[idx], dtype=torch.float32
        ).unsqueeze(0)

        location_name = tuple(self.locations[idx])
        if location_name not in self.elev_cache:
            elev_path = os.path.join(
                self.elev_dir, f"{location_name[0]}_{location_name[1]}_dem.npy"
            )
            elev_array = np.load(elev_path)
            self.elev_cache[location_name] = torch.tensor(
                elev_array, dtype=torch.float32
            ).unsqueeze(0)

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
        self.target_path = os.path.join(
            data_dir, f"{split}_{suffix}_target_normalized.npy"
        )

        # Use memory-mapped arrays to avoid loading all data into RAM
        self.input_data = np.load(self.input_path, mmap_mode="r")
        self.elev_data = np.load(self.elev_path, mmap_mode="r")
        self.target_data = np.load(self.target_path, mmap_mode="r")

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

        self.input = (
            torch.tensor(
                np.load(f"{data_dir}/{split}_{suffix}_input_bicubic.npy"),
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .to(device)
        )

        self.elev = (
            torch.tensor(
                np.load(os.path.join(data_dir, f"{split}_HSURF.npy")),
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .to(device)
        )

        self.target = (
            torch.tensor(
                np.load(f"{data_dir}/{split}_{suffix}_target_normalized.npy"),
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .to(device)
        )

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.elev[idx], self.target[idx]

    def unload_from_gpu(self):
        self.input = self.input.to("cpu")
        self.target = self.target.to("cpu")
        self.elev = self.elev.to("cpu")
