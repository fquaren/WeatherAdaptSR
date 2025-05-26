import os
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import SingleVariableDataset_v3


def get_clusters_dataloader(data_path, excluded_cluster, batch_size=8, num_workers=1, use_theta_e=False, device="cpu"):
    """
    Create dataloaders for training and validation datasets.
    Args:
        data_path (str): Directory containing input files.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloaders.
        use_theta_e (bool): Whether to use theta_e in the dataset.
    Returns:
        dict: Dictionary containing dataloaders for each cluster.
    """

    # Check if the device is a GPU
    if device == "cuda":
        print(f"Loading data on GPU: {device}")
    else:
        print(f"Loading data on CPU: {device}")

    # Check if the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    
    print(f"Creating dataloader from all clusters except {excluded_cluster}...")

    train_clusters, val_clusters, test_clusters = [], [], []
    
    for cluster in sorted(os.listdir(data_path)):
        if cluster == excluded_cluster:
            continue
        else:
            data_dir = os.path.join(data_path, cluster)
            
            train_dataset = SingleVariableDataset_v3(data_dir, split='train', use_theta_e=use_theta_e, device=device)
            val_dataset = SingleVariableDataset_v3(data_dir, split='val', use_theta_e=use_theta_e, device=device)
            test_dataset = SingleVariableDataset_v3(data_dir, split='test', use_theta_e=use_theta_e)

            train_clusters.append(train_dataset)
            val_clusters.append(val_dataset)
            test_clusters.append(test_dataset)

            train_dataset = ConcatDataset(train_clusters)
            val_dataset = ConcatDataset(val_clusters)
            test_dataset = ConcatDataset(test_clusters)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Done.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}