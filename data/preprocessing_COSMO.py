import os
import tqdm
import yaml
import glob
import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, LocalCluster


# --- Configuration ---
CONFIG_PATH = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/WeatherAdaptSR/data/COSMO_domains_config.yaml"

# Define the distinct directories for the temporal streams
RAW_DATA_DIR_5MIN = "/reference/FGSE/climate_simulations/cosmo6_2km_climate/5min_2D"
RAW_DATA_DIR_1H = "/reference/FGSE/climate_simulations/cosmo6_2km_climate/1h_2D"
OUTPUT_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/COSMO_zarr"

# Strict temporal splits from the experimental specification
SPLITS = {
    "train": ("2011-01-01", "2018-01-01"),
    "validation": ("2018-01-01", "2020-01-01"),
    "test": ("2020-01-01", "2022-01-01"),
}

POOLING_FACTOR = 10


def get_rotated_pole_indices(ds, wgs84_bounds):
    """Maps WGS84 coordinates to the native rotated pole array indices."""
    lat_mask = (ds.lat >= wgs84_bounds["lat_min"]) & (ds.lat <= wgs84_bounds["lat_max"])
    lon_mask = (ds.lon >= wgs84_bounds["lon_min"]) & (ds.lon <= wgs84_bounds["lon_max"])
    valid_cells = lat_mask & lon_mask

    rlat_idx, rlon_idx = np.where(valid_cells)

    return {
        "rlat_min": rlat_idx.min(),
        "rlat_max": rlat_idx.max(),
        "rlon_min": rlon_idx.min(),
        "rlon_max": rlon_idx.max(),
    }


def process_time_block(files_5min, files_1h, domain_name, wgs84_bounds):
    """Processes parallel 5-minute and 1-hour chunks, aggregates, and merges."""

    # ---------------------------------------------------------
    # 1. Process 5-Minute Data (Precipitation Aggregation)
    # ---------------------------------------------------------
    ds_5min = xr.open_mfdataset(
        files_5min, parallel=True, engine="netcdf4", chunks={"time": 12}  # 12 * 5min = 1 hour chunks
    )

    idx_5m = get_rotated_pole_indices(ds_5min, wgs84_bounds)
    ds_5m_cropped = ds_5min.isel(
        rlat=slice(idx_5m["rlat_min"], idx_5m["rlat_max"]), rlon=slice(idx_5m["rlon_min"], idx_5m["rlon_max"])
    )

    # Resample to hourly sums.
    # 'closed=right' and 'label=right' means the sum of [00:05, ..., 01:00] is labeled 01:00.
    # This aligns the accumulated precipitation with the instantaneous state at 01:00.
    tp_hourly = ds_5m_cropped["TOT_PREC"].resample(time="1h", label="right", closed="right").sum(dim="time")

    # Cast to dataset
    ds_tp = xr.Dataset({"tp": tp_hourly})

    # ---------------------------------------------------------
    # 2. Process 1-Hour Data (Predictors)
    # ---------------------------------------------------------
    ds_1h = xr.open_mfdataset(files_1h, parallel=True, engine="netcdf4", chunks={"time": 1})  # 1 hour chunks

    idx_1h = get_rotated_pole_indices(ds_1h, wgs84_bounds)
    ds_1h_cropped = ds_1h.isel(
        rlat=slice(idx_1h["rlat_min"], idx_1h["rlat_max"]), rlon=slice(idx_1h["rlon_min"], idx_1h["rlon_max"])
    )

    ds_vars = xr.Dataset()

    # Surface pressure and Instability
    # (Assuming PS is the COSMO variable for surface pressure)
    ds_vars["sp"] = ds_1h_cropped["PS"]
    ds_vars["cape"] = ds_1h_cropped["CAPE_MU"]

    # # Kinematic Advection (Extracting 700 hPa layer strictly)
    # if "pressure" in ds_1h_cropped.dims:
    #     ds_vars["u"] = ds_1h_cropped["U"].sel(pressure=700.0)
    #     ds_vars["v"] = ds_1h_cropped["V"].sel(pressure=700.0)
    # elif "level" in ds_1h_cropped.dims:
    #     # Note: If your data uses model levels instead of pressure levels,
    #     # you will need to index the specific level integer corresponding to 700 hPa here.
    #     ds_vars["u"] = ds_1h_cropped["U"].sel(level=700.0)
    #     ds_vars["v"] = ds_1h_cropped["V"].sel(level=700.0)

    # Mass conservation tracking (Solid + Liquid + Gas)
    ds_vars["twc"] = (
        ds_1h_cropped["TQV"]
        + ds_1h_cropped["TQC"]
        + ds_1h_cropped["TQI"]
        + ds_1h_cropped["TQR"]
        + ds_1h_cropped["TQG"]
        + ds_1h_cropped["TQS"]
    )
    ds_vars["tlwc"] = ds_1h_cropped["TQC"] + ds_1h_cropped["TQR"]

    # ---------------------------------------------------------
    # 3. Merge, Clean, and Coarsen
    # ---------------------------------------------------------
    # Inner join guarantees that if a 5-min file is missing and we don't have a full hour,
    # or a 1-hour file is missing, we drop that specific timestamp entirely.
    ds_hr = xr.merge([ds_tp, ds_vars], join="inner")

    # Predictor Coarsening (10x10)
    ds_lr = ds_hr.coarsen(rlat=POOLING_FACTOR, rlon=POOLING_FACTOR, boundary="trim").mean()

    # Isolate targets (High Res) and predictors (Low Res)
    ds_target = ds_hr[["tp"]]
    ds_predictors = ds_lr[["tp", "u", "v", "sp", "cape", "twc", "tlwc"]]

    return ds_target, ds_predictors


def build_zarr_pipeline():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Generate monthly blocks
    date_range = pd.date_range(start="2011-01-01", end="2021-12-31", freq="M")

    for domain_name, domain_data in config["domains"].items():
        bounds = domain_data["bounds_wgs84"]

        for split_name, (start_date, end_date) in tqdm.tqdm(SPLITS.items()):
            print(f"Processing {domain_name} - {split_name}")

            zarr_out_hr = os.path.join(OUTPUT_DIR, domain_name, f"{split_name}_data_out.zarr")
            zarr_out_lr = os.path.join(OUTPUT_DIR, domain_name, f"{split_name}_data_in.zarr")

            os.makedirs(os.path.join(OUTPUT_DIR, domain_name), exist_ok=True)

            for i in tqdm.tqdm(range(len(date_range) - 1)):
                m_start, _ = date_range[i], date_range[i + 1]

                if pd.Timestamp(start_date) <= m_start < pd.Timestamp(end_date):
                    month_str = m_start.strftime("%Y%m")

                    # Fetch dual data streams
                    files_5min = sorted(glob.glob(os.path.join(RAW_DATA_DIR_5MIN, f"lffd{month_str}*.nz")))
                    files_1h = sorted(glob.glob(os.path.join(RAW_DATA_DIR_1H, f"lffd{month_str}*.nz")))

                    if not files_5min or not files_1h:
                        print(f"  Missing data for {month_str}, skipping...")
                        continue

                    print(f"  Aggregating {month_str}...")
                    ds_hr, ds_lr = process_time_block(files_5min, files_1h, domain_name, bounds)

                    # Append strictly to the continuous arrays
                    append_mode = "a" if os.path.exists(zarr_out_hr) else "w"
                    append_dim = "time" if append_mode == "a" else None

                    ds_hr.to_zarr(zarr_out_hr, mode=append_mode, append_dim=append_dim)
                    ds_lr.to_zarr(zarr_out_lr, mode=append_mode, append_dim=append_dim)

                    ds_hr.close()
                    ds_lr.close()


if __name__ == "__main__":
    # Dynamically read the CPU allocation from your SLURM script
    # Fallback to 4 if running locally without SLURM
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    print(f"Initializing Dask cluster with {n_workers} independent processes...")

    # processes=True is the critical flag to bypass the HDF5 lock
    # threads_per_worker=1 ensures no thread contention within the processes
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit="40GB",  # Safeguard per worker (8 * 40GB = 320GB < 350GB requested)
    )

    # The client intercepts all xarray/dask parallel computations
    client = Client(cluster)

    print(f"Dask dashboard available at: {client.dashboard_link}")

    try:
        build_zarr_pipeline()
    finally:
        # Ensure the cluster shuts down cleanly
        client.close()
        cluster.close()
