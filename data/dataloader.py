from data.dataset import SingleVariableDataset_v2
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
import json


# def compute_dataset_statistics(dataset, save_path):
#     """Computes the mean and std for input, target, and elevation across the dataset using NumPy."""
    
#     # Check if statistics already exist
#     if os.path.exists(save_path):
#         with open(save_path, "r") as f:
#             return json.load(f)  # Load existing statistics
    
#     input_list = []
#     target_list = []
#     elev_list = []

#     # Collect all data as NumPy arrays
#     for i in tqdm(range(len(dataset))):
#         input_data, elevation_data, target_data = dataset[i]
#         input_list.append(input_data.numpy().flatten())  # Convert to NumPy and flatten
#         target_list.append(target_data.numpy().flatten())
#         elev_list.append(elevation_data.numpy().flatten())

#     # Concatenate all arrays for vectorized operations
#     input_arr = np.concatenate(input_list)
#     target_arr = np.concatenate(target_list)
#     elev_arr = np.concatenate(elev_list)

#     # Compute mean and std in a single step
#     input_mean, input_std = np.mean(input_arr), np.std(input_arr)
#     target_mean, target_std = np.mean(target_arr), np.std(target_arr)
#     elev_mean, elev_std = np.mean(elev_arr), np.std(elev_arr)

#     return {
#         "input": (input_mean, input_std),
#         "target": (target_mean, target_std),
#         "elevation": (elev_mean, elev_std),
#     }


def get_file_splits(input_dir, target_dir, excluded_cluster):
    """
    Get file splits for training, validation, and testing datasets.
    Args:
        input_dir (str): Directory containing input files.
        target_dir (str): Directory containing target files.
        excluded_cluster (str): Cluster to be excluded from training.
    Returns:
        dict: Dictionary containing file splits for training, validation, and testing datasets.
    """
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    for cluster in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, cluster)
        target_path = os.path.join(target_dir, cluster)
        if not os.path.isdir(input_path) or not os.path.isdir(target_path):
            continue

        all_input_files = sorted([f for f in os.listdir(input_path) if f.endswith(".nz")])
        all_target_files = sorted([f for f in os.listdir(target_path) if f.endswith(".nz")])
        pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")
        
        for input_file, target_file in zip(all_input_files, all_target_files):
            assert pattern.search(input_file).group(3,4,5) == pattern.search(target_file).group(3,4,5), "Input and target files must match."
            try:
                _, _, year, month = SingleVariableDataset_v2._extract_numbers(None, input_file)
            except ValueError:
                continue

            input_file_path = os.path.join(input_path, input_file)
            target_file_path = os.path.join(target_path, target_file)

            if cluster == excluded_cluster:
                if year == 2017 and month in [3, 6, 9, 12]:
                    val_inputs.append(input_file_path)
                    val_targets.append(target_file_path)
            elif year == 2019 and month % 2 == 1:
                train_inputs.append(input_file_path)
                train_targets.append(target_file_path)
            elif year == 2015 and month % 2 == 1:
                test_inputs.append(input_file_path)
                test_targets.append(target_file_path)

    return {
        "train": (train_inputs, train_targets),
        "val": (val_inputs, val_targets),
        "test": (test_inputs, test_targets),
    }


def get_dataloaders(input_dir, target_dir, elev_dir, variable, batch_size=8, num_workers=1, transform=None):
    """
    Create dataloaders for training, validation, and testing datasets.
    Args:
        input_dir (str): Directory containing input files.
        target_dir (str): Directory containing target files.
        elev_dir (str): Directory containing elevation files.
        variable (str): Variable name to load from the dataset.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloaders.
        transform: Transformations to apply to the data.
    Returns:
        dict: Dictionary containing dataloaders for each cluster.
    """
    cluster_names = sorted([c for c in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, c))])
    dataloaders = {}

    for excluded_cluster in cluster_names:
        print(f"Excluding cluster: {excluded_cluster}")

        file_splits = get_file_splits(input_dir, target_dir, excluded_cluster)

        # train_stats = compute_dataset_statistics(SingleVariableDataset_v2(variable, *file_splits["train"], elev_dir, transform), input_dir)
        # val_stats = compute_dataset_statistics(SingleVariableDataset_v2(variable, *file_splits["val"], elev_dir, transform), input_dir)
        # test_stats = compute_dataset_statistics(SingleVariableDataset_v2(variable, *file_splits["test"], elev_dir, transform), input_dir)
        # print(train_stats)
        # print(val_stats)
        # print(test_stats)

        train_dataset = SingleVariableDataset_v2(variable, *file_splits["train"], elev_dir) #, normalization_stats=train_stats)
        val_dataset = SingleVariableDataset_v2(variable, *file_splits["val"], elev_dir) #, normalization_stats=val_stats)
        test_dataset = SingleVariableDataset_v2(variable, *file_splits["test"], elev_dir) #, normalization_stats=test_stats)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        dataloaders[excluded_cluster] = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders

