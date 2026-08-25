#!/usr/bin/env python3
"""Build Quantile Delta Mapping (QDM) transfer functions for IFS vs ERA5.

Computes a static 99-quantile bias lookup (IFS hindcast minus ERA5) for
TX, diurnal temperature range (DTR = TX − TN), and TG over the 2004–2023
calibration overlap. Each day-of-year uses a centred 31-day window on a
homogeneous ETCCDI 365-day calendar (29 February excised; leap-year DOY
shifted after 1 March). Output: ``qdm_transfer_functions.nc``.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ERA5_DIR = Path("ERA5_ClimateTool/Master_Batches")
IFS_HINDCAST_DIR = Path("ERA5_ClimateTool/IFS_Hindcasts")
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "qdm_transfer_functions.nc"

# Calibration period must be the ERA5 ∩ hindcast overlap.
CALIB_START = "2004-01-01"
CALIB_END = "2023-12-31"
QUANTILES = np.linspace(0.01, 0.99, 99)


def setup_logging() -> None:
    """Configure INFO logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_window_doys(target_doy: int, window_size: int = 31) -> list[int]:
    """Return a centred day-of-year window wrapped on the 365-day calendar."""
    half = window_size // 2
    window = []
    for offset in range(-half, half + 1):
        day = target_doy + offset
        if day < 1:
            day += 365
        elif day > 365:
            day -= 365
        window.append(day)
    return window


def calculate_empirical_quantiles(arr, doys, target_doy):
    """Extract the 31-day window and compute the 99 empirical quantiles."""
    win_doys = get_window_doys(target_doy, window_size=31)
    mask = np.isin(doys, win_doys)
    arr_win = arr[mask]

    if arr_win.shape[0] == 0:
        return np.full(
            (len(QUANTILES), arr.shape[1], arr.shape[2]),
            np.nan,
            dtype=np.float32,
        )

    return np.nanquantile(arr_win, QUANTILES * 100, axis=0).astype(np.float32)


def _etccdi_doys(ds: xr.Dataset) -> np.ndarray:
    """Map timestamps onto the ETCCDI 365-day day-of-year (1–365)."""
    doys = ds.time.dt.dayofyear.values
    is_leap = ds.time.dt.is_leap_year.values
    months = ds.time.dt.month.values
    return np.where(is_leap & (months >= 3), doys - 1, doys)


def build_qdm_matrices() -> None:
    """Compute and write the QDM bias cubes for TX, DTR, and TG."""
    start_time = time.time()
    setup_logging()
    logging.info("START QDM TRANSFER FUNCTION BUILDER (TX, DTR, TG)")

    logging.info("Loading ERA5 baseline...")
    era5_files = sorted(list(ERA5_DIR.glob("era5_master_daily_*.nc")))
    ds_era5 = xr.open_mfdataset(era5_files, combine="by_coords")
    ds_era5 = ds_era5.sel(time=slice(CALIB_START, CALIB_END))

    ds_era5 = ds_era5.sel(
        time=~((ds_era5.time.dt.month == 2) & (ds_era5.time.dt.day == 29))
    )
    era5_doys = _etccdi_doys(ds_era5)

    logging.info("Loading IFS hindcasts...")
    try:
        ifs_files = sorted(list(IFS_HINDCAST_DIR.glob("ifs_hindcast_*.nc")))
        if not ifs_files:
            raise FileNotFoundError("No IFS hindcast files found.")
        ds_ifs = xr.open_mfdataset(ifs_files, combine="by_coords")
        ds_ifs = ds_ifs.sel(time=slice(CALIB_START, CALIB_END))

        ds_ifs = ds_ifs.sel(
            time=~((ds_ifs.time.dt.month == 2) & (ds_ifs.time.dt.day == 29))
        )
        ifs_doys = _etccdi_doys(ds_ifs)
    except Exception as exc:
        logging.error("HINDCAST MISSING: %s", exc)
        logging.error(
            "Aborting. Download IFS hindcasts for 2004–2023 first."
        )
        return

    lats = ds_era5.latitude.values
    lons = ds_era5.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    logging.info("Initialising QDM NetCDF matrix...")

    def empty_4d():
        return (
            ("dayofyear", "quantile", "latitude", "longitude"),
            np.full(
                (365, len(QUANTILES), n_lats, n_lons),
                np.nan,
                dtype=np.float32,
            ),
        )

    ds_qdm = xr.Dataset(
        coords={
            "dayofyear": np.arange(1, 366),
            "quantile": QUANTILES,
            "latitude": lats,
            "longitude": lons,
        },
        data_vars={
            "tx_bias": empty_4d(),
            "dtr_bias": empty_4d(),
            "tg_bias": empty_4d(),
        },
    )

    variables = [
        ("tx", "tx", "tx_bias"),
        ("dtr", "dtr", "dtr_bias"),
        ("tg", "tg", "tg_bias"),
    ]

    for var_era5, var_ifs, out_var in variables:
        logging.info("Computing transfer functions for: %s...", out_var.upper())

        if var_era5 == "dtr":
            arr_era5 = (ds_era5["tx"] - ds_era5["tn"]).compute().values
            arr_ifs = (ds_ifs["tx"] - ds_ifs["tn"]).compute().values
        else:
            arr_era5 = ds_era5[var_era5].compute().values
            arr_ifs = ds_ifs[var_ifs].compute().values

        t0 = time.time()
        for target_doy in range(1, 366):
            idx = target_doy - 1

            q_era5 = calculate_empirical_quantiles(
                arr_era5, era5_doys, target_doy
            )
            q_ifs = calculate_empirical_quantiles(arr_ifs, ifs_doys, target_doy)

            # Bias = observation − model; added to the live forecast in
            # ifs_ingestion.py.
            bias_matrix = q_era5 - q_ifs

            ds_qdm[out_var].values[idx] = bias_matrix

            if target_doy % 30 == 0 or target_doy == 365:
                print(
                    f"   -> DOY {target_doy}/365 computed...",
                    end="\r",
                    flush=True,
                )

        logging.info(
            "%s finished in %.1fs", out_var.upper(), time.time() - t0
        )

        del arr_era5, arr_ifs
        gc.collect()

    logging.info("Writing QDM lookup table: %s", OUT_FILE.name)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_qdm.data_vars}
    ds_qdm.to_netcdf(OUT_FILE, encoding=encoding)

    ds_era5.close()
    ds_ifs.close()

    logging.info(
        "DONE. Total duration: %.1f minutes.",
        (time.time() - start_time) / 60,
    )


if __name__ == "__main__":
    build_qdm_matrices()
