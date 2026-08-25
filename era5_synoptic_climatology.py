#!/usr/bin/env python3
"""Build 5-day moving-window synoptic climatology for MSLP and Z500.

Computes day-of-year mean fields for epochs A (1961–1990) and B (1996–2025)
on a homogeneous ETCCDI 365-day calendar (29 February excised; leap-year DOY
shifted after 1 March). Each target DOY uses a centred 5-day window
(offset −2…+2). Writes ``climatology_synoptics.nc``.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import xarray as xr

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "climatology_synoptics.nc"

EPOCH_A = (1961, 1990)
EPOCH_B = (1996, 2025)
PARAMS = ["mslp", "z500"]
CUTOFF_DATE = "2026-07-31"


def preprocess_era5t(ds: xr.Dataset) -> xr.Dataset:
    """Resolve the ERA5T ``expver`` conflict (operational vs consolidated)."""
    if "expver" in ds.dims:
        if ds.sizes["expver"] > 1:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        else:
            ds = ds.squeeze("expver", drop=True)
    if "expver" in ds.coords:
        ds = ds.drop_vars("expver", errors="ignore")
    return ds


def get_window_doys(target_doy: int) -> list[int]:
    """Return the centred 5-day DOY window wrapped on the 365-day calendar."""
    window = []
    for offset in range(-2, 3):
        day = target_doy + offset
        if day < 1:
            day += 365
        elif day > 365:
            day -= 365
        window.append(day)
    return window


def build_synoptic_climatology() -> None:
    """Compute epoch A/B DOY means for MSLP and Z500 and write NetCDF."""
    total_start = time.time()
    print("START ATMOPULSE SYNOPTIC BUILDER (MSLP & Z500)")

    files = sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))
    ds_master = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        preprocess=preprocess_era5t,
    )
    ds_master = ds_master.sel(time=slice(None, CUTOFF_DATE))

    ds_master = ds_master.sel(
        time=~(
            (ds_master.time.dt.month == 2) & (ds_master.time.dt.day == 29)
        )
    )

    raw_doys = ds_master["time"].dt.dayofyear.values
    is_leap = ds_master["time"].dt.is_leap_year.values
    months = ds_master["time"].dt.month.values
    doys = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)

    years = ds_master["time"].dt.year.values
    lats = ds_master.latitude.values
    lons = ds_master.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    print("Initialising synoptic matrix...")
    data_vars = {}

    def empty_3d():
        return (
            ("dayofyear", "latitude", "longitude"),
            np.full((365, n_lats, n_lons), np.nan, dtype=np.float32),
        )

    for param in PARAMS:
        data_vars[f"{param}_mean_doy_A"] = empty_3d()
        data_vars[f"{param}_mean_doy_B"] = empty_3d()

    ds_syn = xr.Dataset(
        coords={
            "dayofyear": np.arange(1, 366),
            "latitude": lats,
            "longitude": lons,
        },
        data_vars=data_vars,
    )

    for param in PARAMS:
        param_start = time.time()
        print(f"\nLoading '{param}' into RAM...")
        if param not in ds_master:
            print(f"Parameter {param} not found. Skipping...")
            continue

        arr_full = ds_master[param].compute().values

        for target_doy in range(1, 366):
            idx = target_doy - 1
            win_doys = get_window_doys(target_doy)

            mask_win = np.isin(doys, win_doys)
            arr_win = arr_full[mask_win]
            years_win = years[mask_win]

            mask_A_win = (years_win >= EPOCH_A[0]) & (years_win <= EPOCH_A[1])
            mask_B_win = (years_win >= EPOCH_B[0]) & (years_win <= EPOCH_B[1])

            if np.any(mask_A_win):
                ds_syn[f"{param}_mean_doy_A"].values[idx] = np.nanmean(
                    arr_win[mask_A_win], axis=0
                )

            if np.any(mask_B_win):
                ds_syn[f"{param}_mean_doy_B"].values[idx] = np.nanmean(
                    arr_win[mask_B_win], axis=0
                )

            if target_doy % 60 == 0 or target_doy == 365:
                print(
                    f"   -> DOY {target_doy}/365 computed...",
                    end="\r",
                    flush=True,
                )

        print(
            f"   -> '{param}' complete in "
            f"{time.time() - param_start:.1f} seconds."
        )

        del arr_full
        gc.collect()

    ds_master.close()
    ds_syn.to_netcdf(OUT_FILE)
    print(
        f"\nSYNOPTIC BASELINE SAVED: {OUT_FILE.name} "
        f"(total time: {time.time() - total_start:.1f}s)"
    )


if __name__ == "__main__":
    build_synoptic_climatology()
