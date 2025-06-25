import os
import re
import numpy as np
import xarray as xr
import json
from scipy.ndimage import zoom


def _extract_location_time(filename):
    match = re.match(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})", os.path.basename(filename))
    if match:
        A, B, year, month = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        return A, B, year, month
    raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nz")


def get_file_splits_for_all_clusters(input_dir, target_dir):
    cluster_splits = {}

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")

    for cluster in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, cluster)
        target_path = os.path.join(target_dir, cluster)
        if not os.path.isdir(input_path) or not os.path.isdir(target_path):
            continue

        all_input_files = sorted([f for f in os.listdir(input_path) if f.endswith(".nz")])
        all_target_files = sorted([f for f in os.listdir(target_path) if f.endswith(".nz")])

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
        return {
            var: ds[var].sel(time=ds[var].time[0]).values.astype(np.float32)
            for var in var_names
        }


def process_files_pair(input_files, target_files, output_name_prefix, save_dir, excluded_cluster):
    target_vars = ["T_2M"]
    processed_inputs = {var: [] for var in target_vars}
    processed_targets = {var: [] for var in target_vars}
    location_list = []
    filename_list = []

    assert len(input_files) == len(target_files), "Mismatched input-target file count"

    print(f"Processing {len(input_files)} file pairs for {output_name_prefix}")

    for in_path, out_path in zip(input_files, target_files):
        try:
            A, B, _, _ = _extract_location_time(in_path)
            input_vars = load_nc_variables(in_path, target_vars)
            target_vars_data = load_nc_variables(out_path, target_vars)

            for var in target_vars:
                processed_inputs[var].append(input_vars[var])
                processed_targets[var].append(target_vars_data[var])

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
            np.save(f"{save_dir}/{excluded_cluster}/{output_name_prefix}_{var}_input.npy", np.stack(processed_inputs[var]))
            np.save(f"{save_dir}/{excluded_cluster}/{output_name_prefix}_{var}_target.npy", np.stack(processed_targets[var]))
            print(f"Saved {var}_input and {var}_target with shape {np.shape(processed_inputs[var])}")
        except Exception as e:
            print(f"Failed to save {var}: {e}")

    np.save(f"{save_dir}/{excluded_cluster}/{output_name_prefix}_LOCATION.npy", np.array(location_list))
    np.save(f"{save_dir}/{excluded_cluster}/{output_name_prefix}_FILENAME.npy", np.array(filename_list))
    print(f"Saved LOCATION, FILENAME with {len(location_list)} entries")


def compute_stats(x):
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max())
    }


def normalize(x, stats):
    return (x - stats["min"]) / (stats["max"] - stats["min"])


def preprocess_and_save(input_dir, output_dir, split="train", train_stats=None):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    var_names = ["T_2M"]
    data = {}

    for var in var_names:
        for kind in ["input", "target"]:
            fname = f"{split}_{var}_{kind}.npy"
            fpath = os.path.join(input_dir, fname)
            if not os.path.exists(fpath):
                print(f"File {fname} does not exist in {input_dir}. Skipping.")
                continue
            print(f"Loading {fname} from {input_dir}")
            raw = np.load(fpath)
            data[f"{var}_{kind}"] = raw

    for key, arr in data.items():
        if train_stats is not None and key in train_stats:
            stats = train_stats[key]
        else:
            stats = compute_stats(arr)

        metadata[key] = stats
        normalized = normalize(arr, stats)
        np.save(os.path.join(output_dir, f"{split}_{key}_normalized.npy"), normalized)

    with open(os.path.join(output_dir, f"{split}_scaling_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Preprocessing complete for '{split}'. Files saved to '{output_dir}'.")


def interpolate_temperature_data(input_dir, split="train", var="T_2M", method="bilinear", scale_factor=8):
    """
    Interpolates normalized input temperature data to a higher resolution.

    Parameters:
        input_dir (str): Directory containing the normalized input data.
        split (str): Data split to process ("train", "val", or "test").
        var (str): Variable name (default "T_2M").
        method (str): Interpolation method ("bilinear" or "bicubic").
        scale_factor (int): Upscaling factor for each spatial dimension.

    Saves:
        A numpy file with interpolated data in the same directory.
    """

    # input_fname = f"{split}_{var}_input_normalized.npy"
    input_fname = f"{split}_{var}_input.npy"
    input_fpath = os.path.join(input_dir, input_fname)

    if not os.path.exists(input_fpath):
        print(f"[!] File not found: {input_fpath}")
        return

    data = np.load(input_fpath)

    if method == "bilinear":
        order = 1
    elif method == "bicubic":
        order = 3
    else:
        raise ValueError("Invalid method. Use 'bilinear' or 'bicubic'.")

    # Assume input shape is (N, H, W)
    upscaled_data = np.stack([
        zoom(sample, zoom=(scale_factor, scale_factor), order=order)
        for sample in data
    ])

    # out_fname = f"{split}_{var}_input_normalized_interp{scale_factor}x_{method}.npy"
    out_fname = f"{split}_{var}_input_interp{scale_factor}x_{method}.npy"
    out_path = os.path.join(input_dir, out_fname)

    np.save(out_path, upscaled_data)
    print(f"[✓] Saved interpolated data to: {out_path}, shape: {upscaled_data.shape}")


def main():
    input_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/old/DA/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12_blurred"
    target_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/old/DA/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"
    save_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/new_clusters"
    normalized_root = os.path.join(save_dir, "normalized_data")

    cluster_splits = get_file_splits_for_all_clusters(input_dir, target_dir)

    for cluster_name, splits in cluster_splits.items():
        print(f"\n=== Processing cluster: {cluster_name} ===")
        for split in ["train", "val", "test"]:
            input_files, target_files = splits[split]
            process_files_pair(
                input_files=input_files,
                target_files=target_files,
                output_name_prefix=split,
                excluded_cluster=cluster_name,
                save_dir=save_dir
            )

        cluster_raw_dir = os.path.join(save_dir, cluster_name)
        cluster_norm_dir = os.path.join(normalized_root, cluster_name)
        os.makedirs(cluster_norm_dir, exist_ok=True)

        # train_stats_path = os.path.join(cluster_norm_dir, "train_scaling_metadata.json")
        # preprocess_and_save(
        #     input_dir=cluster_raw_dir,
        #     output_dir=cluster_norm_dir,
        #     split="train"
        # )

        # with open(train_stats_path) as f:
        #     train_stats = json.load(f)

        # for split in ["val", "test"]:
        #     preprocess_and_save(
        #         input_dir=cluster_raw_dir,
        #         output_dir=cluster_norm_dir,
        #         split=split,
        #         train_stats=train_stats
        #     )

        # print(f"[✓] Done with {cluster_name}")

        # Interpolate after normalization
        for split in ["train", "val", "test"]:
            interpolate_temperature_data(
                input_dir=cluster_raw_dir,
                split=split,
                method="bicubic",  # or "bilinear"
                scale_factor=8
            )


if __name__ == "__main__":
    main()