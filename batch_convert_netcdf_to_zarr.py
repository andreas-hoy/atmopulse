"""
AtmoPulse Zarr Migration Script (batch_convert_netcdf_to_zarr.py)

Offline, one-off (or periodically re-run/scheduled) migration that rewrites
the ERA5 master archive from spatially-chunked yearly NetCDF files
(`era5_master_daily_*.nc`) into a single Zarr store chunked for fast
point-time-series reads (Meteogram / Wavogram).

Layout
------
Each yearly NetCDF is opened, loaded, and written on its own, then released
before the next file is touched. Peak RAM is therefore one year (~0.5 GB
uncompressed), not the full 1940-present cube.

Zarr chunks: ``{'valid_time': -1, 'latitude': 10, 'longitude': 10}`` applied
*within each year*. After append, a point read touches one small 10x10 tile
per year (~87 tiny chunks) instead of decompressing huge spatial NetCDF
blocks. That is still millisecond-scale, without concatenating decades in RAM.

A failed year is skipped by name and can be retried on the next run.

Usage
-----
    python batch_convert_netcdf_to_zarr.py
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from pathlib import Path

# Must be set BEFORE any netCDF4/HDF5 file is opened. The current calendar
# year's master file is actively appended to by the ERA5T updater; Windows'
# default HDF5 file locking can make a concurrent reader raise "NetCDF: HDF
# error" even on a perfectly valid file.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import xarray as xr

from backend_maps import drop_era5t_aux
from config import DATA_ROOT

MASTER_BATCHES_DIR = DATA_ROOT / "Master_Batches"
ZARR_ARCHIVE_DIR = DATA_ROOT / "Zarr_Archive"
ZARR_STORE_PATH = ZARR_ARCHIVE_DIR / "era5_master_time_series.zarr"

# Per-year point-extraction chunking: the whole year in one time chunk,
# 10x10 lat/lon tiles. Appends keep one time-chunk per year.
YEAR_CHUNKS = {"valid_time": -1, "latitude": 10, "longitude": 10}


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _harmonize(ds: xr.Dataset) -> xr.Dataset:
    """Normalize time dim name and strip ERA5T aux coords so yearly files
    share one schema for append. See backend_maps.drop_era5t_aux."""
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds = drop_era5t_aux(ds)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def _load_one_year(path: Path) -> xr.Dataset:
    """Load a single yearly master file into memory. Tries netcdf4, then
    h5netcdf. Raises the last error if both fail."""
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
        return ds.pipe(_harmonize).load()
    except Exception as exc_netcdf4:
        _log(f"  netcdf4 engine failed ({exc_netcdf4!r}); retrying with h5netcdf...")
        ds = xr.open_dataset(path, engine="h5netcdf")
        return ds.pipe(_harmonize).load()


def _prepare_year(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" not in ds.dims:
        raise ValueError("yearly dataset has no valid_time dimension")
    ds = ds.sortby("valid_time")
    return ds.chunk(YEAR_CHUNKS)


def convert_netcdf_to_zarr() -> None:
    _log("=" * 70)
    _log("AtmoPulse ERA5 NetCDF -> Zarr migration (year-by-year append)")
    _log("=" * 70)

    if not MASTER_BATCHES_DIR.exists():
        _log(f"ERROR: source directory not found: {MASTER_BATCHES_DIR}")
        sys.exit(1)

    nc_files = sorted(MASTER_BATCHES_DIR.glob("era5_master_daily_*.nc"))
    if not nc_files:
        _log(f"ERROR: no era5_master_daily_*.nc files found under {MASTER_BATCHES_DIR}")
        sys.exit(1)

    _log(f"Found {len(nc_files)} yearly master file(s) under {MASTER_BATCHES_DIR}:")
    for f in nc_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        _log(f"  - {f.name} ({size_mb:,.1f} MB)")

    ZARR_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if ZARR_STORE_PATH.exists():
        _log(f"Existing Zarr store found at {ZARR_STORE_PATH} — removing before rewrite.")
        shutil.rmtree(ZARR_STORE_PATH)

    _log(f"Writing Zarr store year-by-year to: {ZARR_STORE_PATH}")
    _log(f"Per-year chunks: {YEAR_CHUNKS}")

    t0 = time.time()
    written = []
    skipped = []
    store_started = False

    import dask
    with dask.config.set(scheduler="synchronous"):
        for i, path in enumerate(nc_files, start=1):
            _log(f"[{i}/{len(nc_files)}] {path.name}")
            t_file = time.time()
            ds = None
            try:
                ds = _load_one_year(path)
                n_times = int(ds.sizes.get("valid_time", 0))
                _log(f"  loaded {n_times} timesteps in {time.time() - t_file:.1f}s; writing...")
                ds = _prepare_year(ds)
                if not store_started:
                    ds.to_zarr(ZARR_STORE_PATH, mode="w", compute=True, consolidated=True)
                    store_started = True
                else:
                    ds.to_zarr(
                        ZARR_STORE_PATH,
                        mode="a",
                        append_dim="valid_time",
                        compute=True,
                        consolidated=True,
                    )
                written.append(path.name)
                _log(f"  OK ({time.time() - t_file:.1f}s total).")
            except Exception as exc:
                _log(f"WARNING: skipping {path.name} — {exc!r}")
                skipped.append(path.name)
            finally:
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
                del ds
                gc.collect()

    if not store_started:
        _log("ERROR: no yearly files could be written to the Zarr store.")
        sys.exit(1)

    if skipped:
        _log(f"WARNING: {len(skipped)} file(s) skipped and are NOT in the store: "
             f"{', '.join(skipped)}")

    _log(f"Wrote {len(written)}/{len(nc_files)} year(s). Verifying store...")
    check = xr.open_zarr(ZARR_STORE_PATH, consolidated=True)
    _log(f"Verification OK. Dims: {dict(check.sizes)}; Variables: {list(check.data_vars)}")
    check.close()

    _log("=" * 70)
    _log(f"Migration finished in {time.time() - t0:.1f}s total.")
    _log(f"Zarr archive ready at: {ZARR_STORE_PATH}")
    _log("=" * 70)


if __name__ == "__main__":
    convert_netcdf_to_zarr()
