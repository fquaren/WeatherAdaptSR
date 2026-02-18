import xarray as xr
import dask.array as da
import json
import os
import time
import yaml

# --- 1. CONFIGURATION ---
# Load configuration
config_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/configs/config_rainshift.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get paths and lists from config
DATA_ROOT = config["DATA_ROOT"]
DOMAIN_LIST = config["DOMAIN_LIST"]
LOG_EPSILON = 1e-6  # Small constant for log-transform
DYNAMIC_VARS = config["VAR_LIST"]
STATIC_VARS = config["STATIC_VAR_LIST"]
OUTPUT_VAR = config["OUTPUT_VAR"]
# -------------------------


def compute_stats_for_domain(domain_name, data_root):
    """
    Computes mean and standard deviation for all variables
    in a single domain's training set and saves them to a JSON file.
    """
    print(f"\n--- Starting statistics computation for cluster: {domain_name} ---")

    stats = {}
    cluster_path = os.path.join(data_root, domain_name)

    if not os.path.exists(cluster_path):
        print(f"Warning: Path not found {cluster_path}. Skipping.")
        return

    # --- 1. Dynamic Input Variables (from train_data_in.zarr) ---
    try:
        in_zarr_path = os.path.join(cluster_path, "train_data_in.zarr")
        ds_in = xr.open_zarr(in_zarr_path, chunks={"time": "auto"})
    except Exception as e:
        print(f"Error opening {in_zarr_path}. Skipping.")
        print(f"Error: {e}")
        return

    for var in DYNAMIC_VARS:
        if var not in ds_in.variables:
            print(f"Warning: Variable '{var}' not found in {in_zarr_path}. Skipping.")
            continue

        print(f"  Calculating stats for dynamic var: {var}...")
        t0 = time.time()

        data = ds_in[var]

        if var == "tp":
            print(f"    -> Converting '{var}' from (m) to (mm) before stats computation.")
            data = data * 1000.0

        mean = data.mean()
        std = data.std()

        var_mean = float(mean.compute())
        var_std = float(std.compute())

        stats[var] = (var_mean, var_std)
        print(f"    -> Mean: {var_mean:.4f}, Std: {var_std:.4f} (Time: {time.time()-t0:.2f}s)")

    ds_in.close()

    # --- 2. Static Input Variables (from static_variables.nc) ---
    static_path = os.path.join(cluster_path, "static_variables.nc")
    try:
        ds_static = xr.open_netcdf(static_path)
    except Exception as e:
        print(f"Error opening {static_path}. Skipping static vars.")
        print(f"Error: {e}")
        ds_static = None

    if ds_static:
        for var in STATIC_VARS:
            if var not in ds_static.variables:
                print(f"Warning: Variable '{var}' not found in {static_path}. Skipping.")
                continue

            print(f"  Calculating stats for static var: {var}...")
            t0 = time.time()

            data = ds_static[var]
            var_mean = float(data.mean().compute())
            var_std = float(data.std().compute())

            stats[var] = (var_mean, var_std)
            print(f"    -> Mean: {var_mean:.4f}, Std: {var_std:.4f} (Time: {time.time()-t0:.2f}s)")
        ds_static.close()

    # --- 3. Output Variable (from train_data_out.zarr) ---
    out_zarr_path = os.path.join(cluster_path, "train_data_out.zarr")
    try:
        ds_out = xr.open_zarr(out_zarr_path, chunks={"time": "auto"})
    except Exception as e:
        print(f"Error opening {out_zarr_path}. Skipping output var.")
        print(f"Error: {e}")
        ds_out = None

    if ds_out:
        print(f"  Calculating stats for log-transformed output: {OUTPUT_VAR}...")
        t0 = time.time()

        data_raw = ds_out[OUTPUT_VAR]
        data_log = da.log(data_raw + LOG_EPSILON)

        mean_log = data_log.mean()
        std_log = data_log.std()

        var_mean = float(mean_log.compute())
        var_std = float(std_log.compute())

        stats[OUTPUT_VAR] = (var_mean, var_std)
        print(f"    -> Log-Mean: {var_mean:.4f}, Log-Std: {var_std:.4f} (Time: {time.time()-t0:.2f}s)")

        ds_out.close()

    # --- 4. Save Statistics ---
    # Save file with domain name
    stats_filename = f"stats_{domain_name}.json"
    stats_path = os.path.join(DATA_ROOT, stats_filename)
    print(f"\nStats for {domain_name} computed. Saving to: {stats_path}")

    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print("Done.")
    except Exception as e:
        print(f"Error saving stats file: {e}")

    return stats


if __name__ == "__main__":
    # --- Main Loop ---
    # Loop over all domains in the config and compute stats for each
    print("Starting per-domain statistics computation...")
    for domain in DOMAIN_LIST:
        compute_stats_for_domain(domain, DATA_ROOT)
    print("\nAll statistics computed.")
