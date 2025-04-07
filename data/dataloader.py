from data.dataset import SingleVariableDataset_v2
from torch.utils.data import DataLoader
import os
import re


def get_file_splits(input_dir, target_dir, excluded_cluster):
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


def get_dataloaders(input_dir, target_dir, elev_dir, variable, batch_size=8, num_workers=1):
    cluster_names = sorted([c for c in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, c))])
    dataloaders = {}

    for excluded_cluster in cluster_names:
        print(f"Excluding cluster: {excluded_cluster}")

        file_splits = get_file_splits(input_dir, target_dir, excluded_cluster)

        train_dataset = SingleVariableDataset_v2(variable, *file_splits["train"], elev_dir)
        val_dataset = SingleVariableDataset_v2(variable, *file_splits["val"], elev_dir)
        test_dataset = SingleVariableDataset_v2(variable, *file_splits["test"], elev_dir)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        dataloaders[excluded_cluster] = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders