"""
AtmoPulse Backend: High-Performance Climatology Builder.

This module constructs the secular ERA5 climatological baselines required 
for real-time extreme weather analytics. It utilizes a single-variable 
RAM-load architecture to eliminate I/O bottlenecks and enforce a strictly 
homogeneous 365-day ETCCDI calendar.

Core functionalities:
- Pre-calculates 5-day moving window percentiles (P5 to P95) for historical 
  (1961-1990) and modern (1996-2025) epochs.
- Extracts absolute physical records and their exact chronological dates (YYYYMMDD).
- Computes seasonal thermal distributions (JJA, DJF).
- Manages memory rigorously via explicit garbage collection during multi-gigabyte 
  matrix operations.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import gc
import time
import warnings

warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning) 

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "climatology_reference.nc"
TEMP_FILE = OUT_DIR / "climatology_progress_temp.nc"

EPOCH_A = (1961, 1990)
EPOCH_B = (1996, 2025)
PARAMS = ['tx', 'tn', 'tg', 't850']
PCTS_DOY = [5, 10, 25, 75, 90, 95]
PCTS_DJF = [5, 10, 25]
PCTS_JJA = [75, 90, 95]
CUTOFF_DATE = '2026-07-31'


def preprocess_era5t(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess ERA5T data to resolve expver dimension conflicts."""
    if 'expver' in ds.dims:
        if ds.sizes['expver'] > 1:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        else:
            ds = ds.squeeze('expver', drop=True)
    if 'expver' in ds.coords:
        ds = ds.drop_vars('expver', errors='ignore')
    return ds


def get_window_doys(target_doy: int) -> list[int]:
    """Return a 5-day centered window of day-of-year indices, wrapping around year ends."""
    window = []
    for offset in range(-2, 3):
        d = target_doy + offset
        if d < 1: 
            d += 365
        elif d > 365: 
            d -= 365
        window.append(d)
    return window


def build_climatology():
    """Main execution block for building the AtmoPulse reference climatology."""
    total_start_time = time.time()
    print("🚀 STARTING ATMOPULSE CLIMATOLOGY BUILDER (RAM-Optimized I/O Mode)")
    
    files = sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))
    ds_master = xr.open_mfdataset(files, combine='by_coords', parallel=True, preprocess=preprocess_era5t)
    ds_master = ds_master.sel(time=slice(None, CUTOFF_DATE))
    
    # ETCCDI-STANDARD: Excision of February 29th for a homogeneous 365-day calendar
    ds_master = ds_master.sel(time=~((ds_master.time.dt.month == 2) & (ds_master.time.dt.day == 29)))
    print(f"📅 Data cutoff: {CUTOFF_DATE} (365-day calendar enforced)")
    
    # Generate standardized DOYs (1-365, correcting for leap years)
    raw_doys = ds_master['time'].dt.dayofyear.values
    is_leap = ds_master['time'].dt.is_leap_year.values
    months = ds_master['time'].dt.month.values
    doys = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    
    years = ds_master['time'].dt.year.values
    # Convert time axis to Integer YYYYMMDD for the frontend tooltip
    dates_int = ds_master.time.dt.strftime("%Y%m%d").values.astype(np.int32)
    
    lats = ds_master.latitude.values
    lons = ds_master.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    # Initialization of the target matrix
    print("🆕 Initializing new NetCDF matrix...")
    data_vars = {}
    
    empty_2d = lambda: (("latitude", "longitude"), np.full((n_lats, n_lons), np.nan, dtype=np.float32))
    empty_3d = lambda: (("dayofyear", "latitude", "longitude"), np.full((365, n_lats, n_lons), np.nan, dtype=np.float32))
    empty_3d_int = lambda: (("dayofyear", "latitude", "longitude"), np.full((365, n_lats, n_lons), -1, dtype=np.int32))

    for param in PARAMS:
        for p in PCTS_DOY:
            data_vars[f"{param}_p{p}_doy_A"] = empty_3d()
            data_vars[f"{param}_p{p}_doy_B"] = empty_3d()
        for p in PCTS_DJF:
            data_vars[f"{param}_djf_p{p}_A"] = empty_2d()
            data_vars[f"{param}_djf_p{p}_B"] = empty_2d()
        for p in PCTS_JJA:
            data_vars[f"{param}_jja_p{p}_A"] = empty_2d()
            data_vars[f"{param}_jja_p{p}_B"] = empty_2d()
        
        data_vars[f"{param}_max_val"] = empty_3d()
        data_vars[f"{param}_max_date"] = empty_3d_int()
        data_vars[f"{param}_min_val"] = empty_3d()
        data_vars[f"{param}_min_date"] = empty_3d_int()

    ds_clim = xr.Dataset(coords={"dayofyear": np.arange(1, 366), "latitude": lats, "longitude": lons}, data_vars=data_vars)

    print("\n=======================================================")
    print("⚡ CALCULATING PARAMETER BY PARAMETER (RAM-ISOLATED)")
    print("=======================================================")

    mask_A_yr = (years >= EPOCH_A[0]) & (years <= EPOCH_A[1])
    mask_B_yr = (years >= EPOCH_B[0]) & (years <= EPOCH_B[1])

    for param in PARAMS:
        param_start = time.time()
        print(f"\n📥 Loading '{param}' entirely into RAM (~2.5 GB)...")
        
        # Eliminate I/O bottleneck
        arr_full = ds_master[param].compute().values
        load_time = time.time() - param_start
        print(f"✅ Loaded in {load_time:.1f}s. Starting matrix calculations...")

        # --- PHASE 1: SEASONAL (JJA & DJF) ---
        t0 = time.time()
        for ep_name, mask_ep in [("A", mask_A_yr), ("B", mask_B_yr)]:
            # DJF
            mask_djf = mask_ep & np.isin(ds_master.time.dt.month.values, [12, 1, 2])
            if np.any(mask_djf):
                pcts = np.nanpercentile(arr_full[mask_djf], PCTS_DJF, axis=0)
                for i, p in enumerate(PCTS_DJF):
                    ds_clim[f"{param}_djf_p{p}_{ep_name}"].values = pcts[i]
            
            # JJA
            mask_jja = mask_ep & np.isin(ds_master.time.dt.month.values, [6, 7, 8])
            if np.any(mask_jja):
                pcts = np.nanpercentile(arr_full[mask_jja], PCTS_JJA, axis=0)
                for i, p in enumerate(PCTS_JJA):
                    ds_clim[f"{param}_jja_p{p}_{ep_name}"].values = pcts[i]
        
        print(f"   -> Seasonal percentiles (DJF/JJA) calculated in {time.time() - t0:.1f}s")

        # --- PHASE 2: DAILY 5-DAY WINDOWS & RECORDS ---
        t0 = time.time()
        for target_doy in range(1, 366):
            idx = target_doy - 1
            win_doys = get_window_doys(target_doy)
            
            # Sub-mask for the 5-day window
            mask_win = np.isin(doys, win_doys)
            arr_win = arr_full[mask_win]
            dates_win = dates_int[mask_win]
            years_win = years[mask_win]

            mask_A_win = (years_win >= EPOCH_A[0]) & (years_win <= EPOCH_A[1])
            mask_B_win = (years_win >= EPOCH_B[0]) & (years_win <= EPOCH_B[1])

            # Percentiles Epoch A & B
            if np.any(mask_A_win):
                pcts_A = np.nanpercentile(arr_win[mask_A_win], PCTS_DOY, axis=0)
                for i, p in enumerate(PCTS_DOY):
                    ds_clim[f"{param}_p{p}_doy_A"].values[idx] = pcts_A[i]
            
            if np.any(mask_B_win):
                pcts_B = np.nanpercentile(arr_win[mask_B_win], PCTS_DOY, axis=0)
                for i, p in enumerate(PCTS_DOY):
                    ds_clim[f"{param}_p{p}_doy_B"].values[idx] = pcts_B[i]

            # Absolute records & exact date extraction
            max_idx = np.nanargmax(arr_win, axis=0)
            ds_clim[f"{param}_max_val"].values[idx] = np.take_along_axis(arr_win, np.expand_dims(max_idx, axis=0), axis=0).squeeze()
            date_grid = np.broadcast_to(dates_win[:, None, None], arr_win.shape)
            ds_clim[f"{param}_max_date"].values[idx] = np.take_along_axis(date_grid, np.expand_dims(max_idx, axis=0), axis=0).squeeze()

            min_idx = np.nanargmin(arr_win, axis=0)
            ds_clim[f"{param}_min_val"].values[idx] = np.take_along_axis(arr_win, np.expand_dims(min_idx, axis=0), axis=0).squeeze()
            ds_clim[f"{param}_min_date"].values[idx] = np.take_along_axis(date_grid, np.expand_dims(min_idx, axis=0), axis=0).squeeze()
            
            if target_doy % 30 == 0 or target_doy == 365:
                print(f"   -> DOY {target_doy}/365 calculated...", end="\r", flush=True)

        print(f"   -> All 365 daily windows calculated in {time.time() - t0:.1f}s")
        
        # Rigorously clear RAM before processing the next parameter
        del arr_full
        gc.collect()

        # Checkpointing after each completed parameter
        print(f"💾 Saving intermediate state for '{param}'...")
        temp_write = OUT_DIR / "temp_step.nc"
        ds_clim.to_netcdf(temp_write)
        os.replace(temp_write, TEMP_FILE)
        print(f"🏁 Parameter '{param}' completed in {(time.time() - param_start)/60:.1f} min.")

    ds_master.close()
    os.replace(TEMP_FILE, OUT_FILE)
    print("\n=======================================================")
    print(f"🎉 ALL DONE! Total duration: {(time.time() - total_start_time)/60:.1f} minutes.")
    print(f"✅ Final climatology saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    build_climatology()