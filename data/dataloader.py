import os
import json
import torch
from torch.utils.data import ConcatDataset, DataLoader
from data.dataset import SingleVariableDataset_v8
import logging


LOGGER = logging.getLogger("experiment")


def compute_temp_stats(datasets):
    all_temps = torch.cat([ds.input_data for ds in datasets], dim=0)
    return all_temps.mean().item(), all_temps.std().item()


def compute_elev_stats(datasets):
    elev_list = []
    for ds in datasets:
        for i in range(len(ds)):
            _, elev, _ = ds[i]
            elev_list.append(elev)
    all_elevs = torch.cat(elev_list, dim=0)
    return all_elevs.mean().item(), all_elevs.std().item()


def load_or_compute_stats(train_clusters, stats_path):

    if os.path.exists(stats_path):
        LOGGER.info(f"Loading normalization stats from {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        LOGGER.info("Computing normalization statistics...")
        LOGGER.info(
            "*** WARNING *** "
            "If you see this during evaluation there might be something wrong. "
            "Check that the statistics you are computing corresponds to the ones used during training! "
            "*** ------- *** "
        )
        temp_mean, temp_std = compute_temp_stats(train_clusters)
        elev_mean, elev_std = compute_elev_stats(train_clusters)
        stats = {
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "elev_mean": elev_mean,
            "elev_std": elev_std,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        LOGGER.info(f"Saved normalization stats to {stats_path}")
    return stats


def get_single_cluster_dataloader(
    data_path,
    elev_dir,
    cluster_name,
    vars,
    batch_size=8,
    num_workers=1,
    use_theta_e=False,
    device="cpu",
    stats_path="train_scaling_metadata.json",
    augment=False,
):
    """
    Load test dataloaders for a single cluster using provided normalization stats.

    Args:
        data_path: path to cluster data
        elev_dir: path to elevation data
        cluster_name: name of the cluster to load
        batch_size: batch size for evaluation
        num_workers: dataloader workers
        use_theta_e: whether to use theta_e
        device: CPU or GPU
        augment: ignored (no augmentation for eval)
    """
    LOGGER.info(f"DATALOADER: Loading single cluster dataloaders for '{cluster_name}'")

    statistics_path = os.path.join(data_path, cluster_name, stats_path)

    LOGGER.info(f"Loading normalization stats from {statistics_path}")
    with open(statistics_path, "r") as f:
        stats = json.load(f)
        LOGGER.info(f"DATALOADER: Normalization stats: {stats}")

    common_args = {
        "vars": vars,
        "elev_mean": stats["elevation"]["mean"],
        "elev_std": stats["elevation"]["std"],
        "use_theta_e": use_theta_e,
        "device": device,
    }

    # Load datasets
    train_dataset = SingleVariableDataset_v8(
        os.path.join(data_path, cluster_name),
        elev_dir,
        split="train",
        augment=augment,
        **common_args,
    )
    val_dataset = SingleVariableDataset_v8(
        os.path.join(data_path, cluster_name),
        elev_dir,
        split="val",
        augment=augment,
        **common_args,
    )
    test_dataset = SingleVariableDataset_v8(
        os.path.join(data_path, cluster_name),
        elev_dir,
        split="test",
        augment=augment,
        **common_args,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    LOGGER.info(
        f"DATALOADER: Done. \
            Train size: {len(train_dataset)} \
            Val size: {len(val_dataset)} \
            Test size: {len(test_dataset)}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# OLD


def get_clusters_dataloader(
    data_path,
    elev_dir,
    excluded_cluster,
    cluster_names,
    batch_size=8,
    num_workers=1,
    use_theta_e=False,
    device="cpu",
    stats_path="crossval_normalization_stats.json",
    augment=False,
):
    """
    Load dataloaders with global normalization and optional augmentation.
    """

    LOGGER.info(
        f"DATALOADER: Creating dataloaders from all clusters except '{excluded_cluster}'..."
    )

    # Preload raw datasets to compute statistics
    raw_train_clusters = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="train",
            use_theta_e=use_theta_e,
            device=device,
            augment=False,
        )
        for cluster in cluster_names
        if cluster != excluded_cluster
    ]

    LOGGER.info(
        f"DATALOADER: Computing normalization statistics for {len(raw_train_clusters)} clusters..."
    )
    stats = load_or_compute_stats(
        raw_train_clusters,
        stats_path=os.path.join(data_path, excluded_cluster, stats_path),
    )
    LOGGER.info(f"DATALOADER: Normalization stats: {stats}")

    common_args = {
        "temp_mean": stats["temp_mean"],
        "temp_std": stats["temp_std"],
        "elev_mean": stats["elev_mean"],
        "elev_std": stats["elev_std"],
        "use_theta_e": use_theta_e,
        "device": device,
    }

    # Load datasets
    train_clusters = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="train",
            augment=augment,
            **common_args,
        )
        for cluster in cluster_names
        if cluster != excluded_cluster
    ]

    val_clusters = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="val",
            augment=False,
            **common_args,
        )
        for cluster in cluster_names
        if cluster != excluded_cluster
    ]

    test_clusters = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="test",
            augment=False,
            **common_args,
        )
        for cluster in cluster_names
        if cluster != excluded_cluster
    ]

    # Wrap in dataloaders
    LOGGER.info("DATALOADER: Creating dataloaders...")
    train_dataset = ConcatDataset(train_clusters)
    val_dataset = ConcatDataset(val_clusters)
    test_dataset = ConcatDataset(test_clusters)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    LOGGER.info(
        f"DATALOADER: Done. \
            Train size: {len(train_dataset)} \
            Val size: {len(val_dataset)} \
            Test size: {len(test_dataset)}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_domain_adaptation_dataloaders(
    data_path,
    elev_dir,
    target_cluster,
    cluster_names,
    batch_size=8,
    num_workers=1,
    use_theta_e=False,
    device="cpu",
    stats_path="da_normalization_stats.json",
    augment=False,
):
    """
    Loads source and target dataloaders for domain adaptation.
    Normalization statistics are computed ONLY from source clusters.

    Returns:
        dict with 'source' and 'target' dataloaders
    """
    LOGGER.info(
        f"DATALOADER: Loading domain adaptation dataloaders (target: {target_cluster})"
    )

    # Prepare source clusters
    source_clusters = [c for c in cluster_names if c != target_cluster]

    # Raw source datasets for computing stats
    raw_source_train_clusters = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="train",
            use_theta_e=use_theta_e,
            device=device,
            augment=False,
        )
        for cluster in source_clusters
    ]

    LOGGER.info(
        f"DATALOADER: Computing normalization statistics from {len(raw_source_train_clusters)} source clusters..."
    )
    stats = load_or_compute_stats(
        raw_source_train_clusters,
        stats_path=os.path.join(data_path, target_cluster, stats_path),
    )
    LOGGER.info(f"DATALOADER: Normalization stats: {stats}")

    common_args = {
        "temp_mean": stats["temp_mean"],
        "temp_std": stats["temp_std"],
        "elev_mean": stats["elev_mean"],
        "elev_std": stats["elev_std"],
        "use_theta_e": use_theta_e,
        "device": device,
    }

    # Source datasets
    source_train = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="train",
            augment=augment,
            **common_args,
        )
        for cluster in source_clusters
    ]
    source_val = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="val",
            augment=False,
            **common_args,
        )
        for cluster in source_clusters
    ]
    source_test = [
        SingleVariableDataset_v8(
            os.path.join(data_path, cluster),
            elev_dir,
            split="test",
            augment=False,
            **common_args,
        )
        for cluster in source_clusters
    ]

    # Target datasets (use source normalization stats!)
    target_train = SingleVariableDataset_v8(
        os.path.join(data_path, target_cluster),
        elev_dir,
        split="train",
        augment=False,
        **common_args,
    )
    target_val = SingleVariableDataset_v8(
        os.path.join(data_path, target_cluster),
        elev_dir,
        split="val",
        augment=False,
        **common_args,
    )
    target_test = SingleVariableDataset_v8(
        os.path.join(data_path, target_cluster),
        elev_dir,
        split="test",
        augment=False,
        **common_args,
    )

    # Dataloaders
    LOGGER.info("DATALOADER: Creating dataloaders...")
    loaders = {
        "source": {
            "train": DataLoader(
                ConcatDataset(source_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                ConcatDataset(source_val),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            "test": DataLoader(
                ConcatDataset(source_test),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        },
        "target": {
            "train": DataLoader(
                target_train,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            "val": DataLoader(
                target_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            "test": DataLoader(
                target_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        },
        "stats": stats,
    }

    LOGGER.info(
        f"DATALOADER: Done. Loaded source ({len(source_train)} clusters) and target '{target_cluster}'"
    )
    return loaders
