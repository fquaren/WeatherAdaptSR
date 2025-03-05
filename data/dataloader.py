from data.dataset import SingleVariableDataset
from torch.utils.data import DataLoader
import os
import re


def split_dataset(input_files, target_files):
    """
    Splits input and target files into training, validation, and test sets.
    """
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")

    for input_file, target_file in zip(input_files, target_files):
        # Check input file and target file contain the same date
        assert pattern.search(input_file).group(3,4,5) == pattern.search(target_file).group(3,4,5), "Input and target files must match."

        pattern.search(input_file).group()
        match = pattern.search(input_file)
        if match:
            day = int(match.group(5))
            if day <= 21:
                train_inputs.append(input_file)
                train_targets.append(target_file)
            elif 22 <= day < 25:
                test_inputs.append(input_file)
                test_targets.append(target_file)
            elif day >= 25:
                val_inputs.append(input_file)
                val_targets.append(target_file)

    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


def split_dataset_3h(input_files, target_files):
    """
    Splits input and target files into training, validation, and test sets.
    Only selects files where the hour is a multiple of 3 (every 3 hours).
    
    Args:
        input_files (list of str): List of input file paths.
        target_files (list of str): List of corresponding target file paths.
    
    Returns:
        tuple: (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)
    """
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})(\d{2})\d{4}")

    for input_file, target_file in zip(input_files, target_files):
        match_input = pattern.search(input_file)
        match_target = pattern.search(target_file)

        if match_input and match_target:
            # Extract day and hour
            day = int(match_input.group(5))   # Group 5 is the day (DD)
            hour = int(match_input.group(6))  # Group 6 is the hour (hh)

            # Ensure input and target files correspond to the same date
            assert match_input.group(3, 4, 5, 6) == match_target.group(3, 4, 5, 6), \
                "Input and target files must match."

            # Only select data points at 3-hour intervals
            if hour % 3 == 0:
                if day <= 21:
                    train_inputs.append(input_file)
                    train_targets.append(target_file)
                elif 22 <= day < 25:
                    test_inputs.append(input_file)
                    test_targets.append(target_file)
                elif day >= 25:
                    val_inputs.append(input_file)
                    val_targets.append(target_file)

    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


def get_dataloaders(variable, input_dir, target_dir, elev_file, batch_size=4):
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nz")])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".nz")])

    # Check input and target files contain the same number of samples
    assert len(input_files) == len(target_files), "Number of input and target files must match."

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_dataset_3h(input_files, target_files)

    train_dataset = SingleVariableDataset(variable, train_inputs, elev_file, train_targets)
    val_dataset = SingleVariableDataset(variable, val_inputs, elev_file, val_targets)
    test_dataset = SingleVariableDataset(variable, test_inputs, elev_file, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cluster_dataloaders(variable, input_path, target_path, dem_dir, batch_size=4):
    """
    Load train and validation DataLoaders for all clusters.

    Args:
        variable (str): Variable name for dataset.
        data_root (str): Root directory containing cluster subdirectories.
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loaders (dict): Dictionary of cluster train DataLoaders.
        val_loaders (dict): Dictionary of cluster validation DataLoaders.
        test_loaders (dict): Dictionary of cluster testing DataLoaders.
    """
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    print(input_path)
    for cluster_name in os.listdir(input_path):
        input_dir = os.path.join(input_path, cluster_name)
        target_dir = os.path.join(target_path, cluster_name)
        # Ensure directories exist
        if not (os.path.isdir(input_dir) and os.path.isdir(target_dir)):
            continue

        # Load train and validation data
        train_loader, val_loader, test_loader = get_dataloaders(variable, input_dir, target_dir, dem_dir, batch_size)

        train_loaders[cluster_name] = train_loader
        val_loaders[cluster_name] = val_loader
        test_loaders[cluster_name] = test_loader

    return train_loaders, val_loaders, test_loaders