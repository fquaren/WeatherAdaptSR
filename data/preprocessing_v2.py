import os
import re
import numpy as np
import xarray as xr
import json
from scipy.ndimage import zoom
import concurrent.futures


def _extract_location_time(filename):
    match = re.match(
        r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})", os.path.basename(filename)
    )
    if match:
        A, B, year, month = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        return A, B, year, month
    raise ValueError(
        f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nz"
    )


def get_file_splits_for_all_clusters(input_dir, target_dir):
    cluster_splits = {}

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")

    for cluster in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, cluster)
        target_path = os.path.join(target_dir, cluster)
        if not os.path.isdir(input_path) or not os.path.isdir(target_path):
            continue

        all_input_files = sorted(
            [f for f in os.listdir(input_path) if f.endswith(".nc")]
        )
        all_target_files = sorted(
            [f for f in os.listdir(target_path) if f.endswith(".nc")]
        )

        train_inputs, val_inputs, test_inputs = [], [], []
        train_targets, val_targets, test_targets = [], [], []

        for input_file, target_file in zip(all_input_files, all_target_files):
            if not pattern.search(input_file) or not pattern.search(target_file):
                continue

            try:
                A, B, year, month = _extract_location_time(input_file)
            except ValueError:
                continue

            input_file_path = os.path.join(input_path, input_file)
            target_file_path = os.path.join(target_path, target_file)

            if year == 2019:
                train_inputs.append(input_file_path)
                train_targets.append(target_file_path)
            elif year == 2017 and month in [3, 6, 9, 12]:
                val_inputs.append(input_file_path)
                val_targets.append(target_file_path)
            elif year == 2015:
                test_inputs.append(input_file_path)
                test_targets.append(target_file_path)

        cluster_splits[cluster] = {
            "train": (train_inputs, train_targets),
            "val": (val_inputs, val_targets),
            "test": (test_inputs, test_targets),
        }

    return cluster_splits


def load_nc_variables(file_path, var_names):
    with xr.open_dataset(file_path) as ds:
        return {var: ds[var].values.astype(np.float32) for var in var_names}


def process_files_pair(
    input_files, target_files, output_name_prefix, save_dir, excluded_cluster
):
    target_vars = ["T_2M", "TOT_PREC"]
    processed_inputs = {var: [] for var in target_vars}
    processed_targets = {var: [] for var in target_vars}
    location_list = []
    filename_list = []

    assert len(input_files) == len(target_files), "Mismatched input-target file count"

    print(f"Processing {len(input_files)} file pairs for {output_name_prefix}")

    for in_path, out_path in zip(input_files, target_files):
        try:
            A, B, _, _ = _extract_location_time(in_path)
            input_vars_data = load_nc_variables(in_path, target_vars)
            target_vars_data = load_nc_variables(out_path, target_vars)

            for var in target_vars:
                input_data = input_vars_data[var]
                target_data = target_vars_data[var]
                processed_inputs[var].append(input_data)
                processed_targets[var].append(target_data)

            location_list.append((A, B))
            filename_list.append(os.path.basename(in_path))
        except Exception as e:
            print(f"Error processing pair {in_path} and {out_path}: {e}")

    if not filename_list:
        print(f"No valid pairs found for {output_name_prefix}")
        return

    os.makedirs(os.path.join(save_dir, excluded_cluster), exist_ok=True)

    for var in target_vars:
        try:
            np.save(
                f"{save_dir}/{excluded_cluster}/{output_name_prefix}_{var}_input.npy",
                np.stack(processed_inputs[var]),
            )
            np.save(
                f"{save_dir}/{excluded_cluster}/{output_name_prefix}_{var}_target.npy",
                np.stack(processed_targets[var]),
            )
            print(
                f"Saved {var}_input and {var}_target with shape {np.shape(processed_inputs[var])}"
            )
        except Exception as e:
            print(f"Failed to save {var}: {e}")

    np.save(
        f"{save_dir}/{excluded_cluster}/{output_name_prefix}_LOCATION.npy",
        np.array(location_list),
    )
    np.save(
        f"{save_dir}/{excluded_cluster}/{output_name_prefix}_FILENAME.npy",
        np.array(filename_list),
    )
    print(f"Saved LOCATION, FILENAME with {len(location_list)} entries")


def compute_stats(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def normalize(x, stats):
    return (x - stats["mean"]) / stats["std"]


def interpolate_input_data(
    input_dir, split="train", var="T_2M", method="bilinear", scale_factor=8
):
    """
    Interpolates normalized input temperature data to a higher resolution.
    (Function body is unchanged)
    """

    input_fname = f"{split}_{var}_input_normalized.npy"
    input_fpath = os.path.join(input_dir, input_fname)

    if not os.path.exists(input_fpath):
        print(f"[!] File not found: {input_fpath}")
        return

    data = np.load(input_fpath)
    if method == "nn":
        order = 0
    elif method == "bilinear":
        order = 1
    elif method == "bicubic":
        order = 3
    else:
        raise ValueError("Invalid method. Use 'bilinear' or 'bicubic'.")

    # Assume input shape is (N, H, W)
    upscaled_data = np.stack(
        [
            zoom(sample, zoom=(scale_factor, scale_factor), order=order)
            for sample in data
        ]
    )

    out_fname = f"{split}_{var}_input_normalized_interp{scale_factor}x_{method}.npy"
    out_path = os.path.join(input_dir, out_fname)

    np.save(out_path, upscaled_data)
    print(f"[✓] Saved interpolated data to: {out_path}, shape: {upscaled_data.shape}")


def downsample_temperature_data(high_res_data, downscaling_factor, method="bilinear"):
    """
    Downsamples high-resolution temperature data to a lower resolution.
    """
    if not isinstance(high_res_data, np.ndarray):
        raise ValueError("Input 'high_res_data' must be a NumPy array.")
    if downscaling_factor < 0:
        raise ValueError("Downscaling factor must be a positive integer.")

    elif method == "nn":
        order = 0
    elif method == "bilinear":
        order = 1
    elif method == "bicubic":
        order = 3
    else:
        raise ValueError("Invalid method. Use 'bilinear' or 'bicubic'.")

    # The zoom factor for scipy.ndimage.zoom is 1/downscaling_factor
    zoom_factor = 1.0 / downscaling_factor

    # Handle both 2D (H, W) and 3D (N, H, W) arrays
    if high_res_data.ndim == 2:
        # For a single 2D image, no parallelization needed
        downscaled_data = zoom(
            high_res_data, zoom=(zoom_factor, zoom_factor), order=order
        )
    elif high_res_data.ndim == 3:
        # Parallelize processing for each sample in the batch (N, H, W)
        # Using ThreadPoolExecutor as scipy.ndimage.zoom releases the GIL
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the zoom function to each sample in the batch
            # zoom_args = (sample, (zoom_factor, zoom_factor), order)
            # The lambda function creates a callable for each sample with fixed zoom_factor and order
            downscaled_samples = list(
                executor.map(
                    lambda sample: zoom(
                        sample, zoom=(zoom_factor, zoom_factor), order=order
                    ),
                    high_res_data,
                )
            )
        downscaled_data = np.stack(downscaled_samples)
    else:
        raise ValueError("Input 'high_res_data' must be 2D (H, W) or 3D (N, H, W).")

    return downscaled_data


def calculate_pooled_training_stats(base_dir, cluster_names, var_names, elev_dir):
    """
    Calculates statistics across all clusters to derive pooled standard deviation
    for each variable in var_names and for elevation.

    Args:
        base_dir (str): Base directory for training .npy files.
        cluster_names (list): Names of clusters (directories inside base_dir).
        var_names (list): Variables to compute stats for.
        elev_dir (str): Directory containing elevation .npy files.

    Returns:
        dict: Dictionary with pooled std, mean, and cluster-specific stats for each variable.
    """
    print("\n--- Calculating Pooled Training Statistics ---")
    all_individual_stats = {}

    # 1. Gather individual stats for each cluster's training data
    for var in var_names:
        key = f"{var}_input"
        all_individual_stats[key] = {}
        for cluster_name in cluster_names:
            fpath = os.path.join(base_dir, cluster_name, f"train_{var}_input.npy")
            if not os.path.exists(fpath):
                print(f"Warning: File not found {fpath}, skipping for stats.")
                continue

            data = np.load(fpath)
            # Variance-stabilizing transform as in Harder et al. 2025
            if var == "TOT_PREC":
                data = np.log1p(data)
            stats = compute_stats(data)
            all_individual_stats[key][cluster_name] = stats
            print(f"[Stats] Loaded {key} for {cluster_name}")

    # 2. Gather elevation stats
    all_individual_stats["elevation"] = {}
    for cluster_name in cluster_names:
        loc_path = os.path.join(base_dir, cluster_name, "train_LOCATION.npy")
        if not os.path.exists(loc_path):
            print(f"Warning: LOCATION file not found {loc_path}, skipping elevation.")
            continue

        loc_array = np.load(loc_path)
        if loc_array.size == 0:
            print(
                f"Warning: LOCATION array empty for {cluster_name}, skipping elevation."
            )
            continue

        lat, lon = loc_array[0]  # first row
        elev_path = os.path.join(elev_dir, f"{lat}_{lon}_dem.npy")
        if not os.path.exists(elev_path):
            print(f"Warning: Elevation file not found {elev_path}, skipping.")
            continue

        elev_array = np.load(elev_path)
        stats = compute_stats(elev_array)
        all_individual_stats["elevation"][cluster_name] = stats
        print(f"[Stats] Loaded elevation for {cluster_name}")

    # 3. Calculate pooled std for each variable, including elevation
    final_stats = {}
    for key, cluster_stats_dict in all_individual_stats.items():
        if not cluster_stats_dict:
            continue

        means = [s["mean"] for s in cluster_stats_dict.values()]
        stds = [s["std"] for s in cluster_stats_dict.values()]

        # pooled std formula
        second_moments = [s**2 + m**2 for s, m in zip(stds, means)]
        mean_of_second_moments = np.mean(second_moments)
        mean_of_means = np.mean(means)

        pooled_std = np.sqrt(mean_of_second_moments - mean_of_means**2)

        final_stats[key] = {
            "pooled_std": float(pooled_std),
            "clusters": cluster_stats_dict,
        }
        print(f"[Pooled] {key}: Pooled Std = {pooled_std:.4f}")

    return final_stats


def preprocess_and_save(
    input_dir, output_dir, split, stats_to_use, var_names, precip_clip_percentile=99.9
):
    """
    Normalizes data using pre-computed stats and saves the result.
    Optionally clips precipitation outliers before log transform.
    """
    os.makedirs(output_dir, exist_ok=True)

    for var in var_names:
        for kind in ["input", "target"]:
            fname = f"{split}_{var}_{kind}.npy"
            fpath = os.path.join(input_dir, fname)

            if not os.path.exists(fpath):
                # This is expected if a split has no files
                continue

            if f"{var}_input" not in stats_to_use:
                print(
                    f"Warning: No stats found for {var}_input. Skipping normalization."
                )
                continue

            print(f"Normalizing {fname}...")
            raw = np.load(fpath)

            if var == "TOT_PREC":
                # --- Outlier removal ---
                clip_val = np.percentile(raw, precip_clip_percentile)
                raw = np.clip(raw, None, clip_val)

                # --- Variance-stabilizing transform ---
                raw = np.log1p(raw)

            # --- Normalization ---
            normalized = normalize(raw, stats_to_use[f"{var}_input"])

            np.save(
                os.path.join(output_dir, f"{split}_{var}_{kind}_normalized.npy"),
                normalized,
            )

    print(f"[✓] Normalization complete for '{split}' in '{output_dir}'.")


def main():
    old_data_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/domain_adaptation/"
    )
    input_dir = os.path.join(
        old_data_dir,
        "T2M_TOTPREC_cropped_gridded_clustered_threshold_8_blurred_x8",
    )
    target_dir = os.path.join(
        old_data_dir,
        "T2M_TOTPREC_cropped_gridded_clustered_threshold_8",
    )
    elev_dir = os.path.join(
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/dem_squares"
    )
    save_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/clusters_v5"
    var_names = ["T_2M", "TOT_PREC"]

    # --- Step 1: Process raw .nz files into .npy arrays for all clusters ---
    cluster_splits = get_file_splits_for_all_clusters(input_dir, target_dir)
    for cluster_name, splits in cluster_splits.items():
        print(f"\n=== Converting .nz files for cluster: {cluster_name} ===")
        for split in ["train", "val", "test"]:
            if not splits[split][0]:  # Skip if split is empty
                continue
            input_files, target_files = splits[split]
            process_files_pair(
                input_files=input_files,
                target_files=target_files,
                output_name_prefix=split,
                excluded_cluster=cluster_name,
                save_dir=save_dir,
            )

    # --- Step 2: Calculate pooled training statistics from all clusters ---
    cluster_names = list(cluster_splits.keys())
    pooled_training_stats = calculate_pooled_training_stats(
        save_dir, cluster_names, var_names, elev_dir
    )

    # Save the master statistics object for reference
    with open(os.path.join(save_dir, "master_pooled_stats.json"), "w") as f:
        json.dump(pooled_training_stats, f, indent=4)
    print("\n--- Master pooled statistics saved to master_pooled_stats.json ---")

    # --- Step 3: Apply normalization and save final data for each cluster ---
    for cluster_name in cluster_names:
        print(f"\n=== Normalizing data for cluster: {cluster_name} ===")
        cluster_raw_dir = os.path.join(save_dir, cluster_name)
        metadata_for_cluster_json = {}

        for split in ["train", "val", "test"]:
            stats_to_use_for_split = {}
            # For every variable, use its cluster-specific mean/min/max but the shared pooled_std
            for key, stats_data in pooled_training_stats.items():
                if cluster_name in stats_data["clusters"]:
                    # Get the specific mean, min, max for this cluster from the training data
                    cluster_specific_stats = stats_data["clusters"][cluster_name]
                    stats_to_use_for_split[key] = {
                        "mean": cluster_specific_stats["mean"],
                        "min": cluster_specific_stats["min"],
                        "max": cluster_specific_stats["max"],
                        "std": stats_data["pooled_std"],  # Use the pooled std
                    }

            if not stats_to_use_for_split:
                continue

            preprocess_and_save(
                input_dir=cluster_raw_dir,
                output_dir=cluster_raw_dir,
                split=split,
                stats_to_use=stats_to_use_for_split,
                var_names=var_names,
            )

            # Store the stats used for this cluster in its metadata file
            if split == "train":
                metadata_for_cluster_json = stats_to_use_for_split

        # Save the specific metadata for this cluster
        if metadata_for_cluster_json:
            train_stats_path = os.path.join(
                cluster_raw_dir, "train_scaling_metadata.json"
            )
            with open(train_stats_path, "w") as f:
                json.dump(metadata_for_cluster_json, f, indent=4)
            print(f"[✓] Saved training metadata for {cluster_name}")

    # --- Step 4: Perform interpolation ---
    print("\n--- Performing Interpolation ---")
    for cluster_name in cluster_names:
        print(f"\n=== Post-processing for cluster: {cluster_name} ===")
        cluster_raw_dir = os.path.join(save_dir, cluster_name)
        for split in ["train", "val", "test"]:
            for var in ["T_2M", "TOT_PREC"]:
                interpolate_input_data(
                    input_dir=cluster_raw_dir,
                    split=split,
                    var=var,
                    method="bilinear",
                    scale_factor=8,
                )


if __name__ == "__main__":
    main()
