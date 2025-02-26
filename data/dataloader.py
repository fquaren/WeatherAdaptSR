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


def get_dataloaders(variable, input_dir, target_dir, elev_file, batch_size=4):
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nz")])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".nz")])

    # Check input and target files contain the same number of samples
    assert len(input_files) == len(target_files), "Number of input and target files must match."

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_dataset(input_files, target_files)

    train_dataset = SingleVariableDataset(variable, train_inputs, elev_file, train_targets)
    val_dataset = SingleVariableDataset(variable, val_inputs, elev_file, val_targets)
    test_dataset = SingleVariableDataset(variable, test_inputs, elev_file, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader