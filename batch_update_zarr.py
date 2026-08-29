"""
AtmoPulse Zarr Daily Update Script (batch_update_zarr.py)

Small, fast, daily/scheduled companion to `batch_convert_netcdf_to_zarr.py`
(the full decades-long rewrite). This script only APPENDS whatever new
ERA5T days have landed in the current calendar year's master NetCDF file
since the Zarr store was last written/updated — it never re-reads or
re-writes historical years, so it stays cheap enough to run daily (e.g.
right after the ERA5T updater refreshes `era5_master_daily_<year>.nc`).

Logic
-----
1. Open the Zarr store read-only and find its current max `valid_time`.
2. Open the current year's `era5_master_daily_<year>.nc`, harmonized the
   same way as `backend_io.py` (time -> valid_time rename, ERA5T `expver`
   aux-coord drop, single-level pressure squeeze).
3. Keep only the days STRICTLY AFTER the Zarr store's max `valid_time`.
4. If there are none, print "No new data to append" and exit cleanly.
5. Otherwise, chunk the new slice like the rest of the store (small
   10x10 lat/lon tiles, one time chunk for the new slice) and append it
   with `.to_zarr(zarr_path, mode='a', append_dim='valid_time', compute=True)`.

Usage
-----
    python batch_update_zarr.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr

from backend_maps import drop_era5t_aux
from config import DATA_ROOT

MASTER_BATCHES_DIR = DATA_ROOT / "Master_Batches"
ZARR_STORE_PATH = DATA_ROOT / "Zarr_Archive" / "era5_master_time_series.zarr"

# Must match the lat/lon tiling used by batch_convert_netcdf_to_zarr.py so
# the appended slice's non-append-dim chunking lines up with the store's
# existing chunk boundaries. The append dim (valid_time) is written as one
# new chunk covering just the newly-appended days.
APPEND_CHUNKS = {"valid_time": -1, "latitude": 10, "longitude": 10}


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _harmonize(ds: xr.Dataset) -> xr.Dataset:
    """Same normalization as backend_io._harmonize_master_archive: rename
    time -> valid_time, drop ERA5T expver aux coords, squeeze a
    single-level pressure_level dim."""
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds = drop_era5t_aux(ds)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def update_zarr_with_new_days() -> None:
    _log("=" * 70)
    _log("AtmoPulse Zarr daily update starting")
    _log("=" * 70)

    if not ZARR_STORE_PATH.exists():
        _log(f"ERROR: Zarr store not found at {ZARR_STORE_PATH}.")
        _log("Run batch_convert_netcdf_to_zarr.py first to build the initial store.")
        sys.exit(1)

    _log(f"Opening existing Zarr store (read-only) to find its current max valid_time: {ZARR_STORE_PATH}")
    try:
        zarr_ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=True)
    except Exception:
        zarr_ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=False)

    if "valid_time" not in zarr_ds.dims or zarr_ds.sizes.get("valid_time", 0) == 0:
        _log("ERROR: Zarr store has no valid_time data — cannot determine append point.")
        sys.exit(1)

    zarr_max_time = pd.Timestamp(zarr_ds["valid_time"].max().values)
    _log(f"Zarr store currently holds data through: {zarr_max_time}")
    zarr_ds.close()

    this_year = pd.Timestamp.utcnow().year
    nc_path = MASTER_BATCHES_DIR / f"era5_master_daily_{this_year}.nc"
    _log(f"Opening current-year master file: {nc_path}")
    if not nc_path.exists():
        _log(f"ERROR: current-year master file not found: {nc_path}")
        sys.exit(1)

    try:
        nc_ds = xr.open_dataset(nc_path, engine="netcdf4", decode_timedelta=False).pipe(_harmonize)
    except Exception as exc:
        _log(f"ERROR: could not open {nc_path.name} — {exc!r}")
        sys.exit(1)

    if "valid_time" not in nc_ds.dims:
        _log(f"ERROR: {nc_path.name} has no valid_time dimension after harmonization.")
        nc_ds.close()
        sys.exit(1)

    _log("Selecting only days strictly after the Zarr store's max valid_time...")
    mask = nc_ds["valid_time"].values > np.datetime64(zarr_max_time)
    new_ds = nc_ds.isel(valid_time=mask)
    n_new = int(new_ds.sizes.get("valid_time", 0))

    if n_new == 0:
        _log("No new data to append")
        print("No new data to append")
        nc_ds.close()
        return

    new_ds = new_ds.load()
    nc_ds.close()

    new_min = pd.Timestamp(new_ds["valid_time"].min().values)
    new_max = pd.Timestamp(new_ds["valid_time"].max().values)
    _log(f"Found {n_new} new day(s) to append: {new_min} .. {new_max}")

    _log(f"Re-chunking new slice for append: {APPEND_CHUNKS}")
    new_ds = new_ds.sortby("valid_time").chunk(APPEND_CHUNKS)

    _log(f"Appending to Zarr store: {ZARR_STORE_PATH}")
    t0 = time.time()
    new_ds.to_zarr(ZARR_STORE_PATH, mode="a", append_dim="valid_time", compute=True)
    _log(f"Append complete in {time.time() - t0:.1f}s.")

    _log("Verifying store re-opens correctly after append...")
    check = xr.open_zarr(ZARR_STORE_PATH, consolidated=True)
    _log(f"Verification OK. New max valid_time: {pd.Timestamp(check['valid_time'].max().values)}; "
         f"total timesteps: {check.sizes.get('valid_time', '?')}")
    check.close()

    _log("=" * 70)
    _log(f"Daily update finished. Appended {n_new} new day(s).")
    _log("=" * 70)


if __name__ == "__main__":
    update_zarr_with_new_days()
