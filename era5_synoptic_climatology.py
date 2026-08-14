#!/usr/bin/env python3
"""
AtmoPulse Backend: Synoptic Baseline Builder (MSLP & Z500)
- Berechnet 5-Tage gleitende Mittelwerte für Zirkulationsanomalien.
- Epochen: 1961-1990 (A) und 1996-2025 (B).
- Hochgeschwindigkeits-Architektur über direkte NumPy-Means.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import os
import gc
import time

# --- PFADE ---
DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "climatology_synoptics.nc"

EPOCH_A = (1961, 1990)
EPOCH_B = (1996, 2025)
PARAMS = ['mslp', 'z500']
CUTOFF_DATE = '2026-07-31'

def preprocess_era5t(ds):
    """ERA5T expver-Konflikt auflösen."""
    if 'expver' in ds.dims:
        if ds.sizes['expver'] > 1:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        else:
            ds = ds.squeeze('expver', drop=True)
    if 'expver' in ds.coords:
        ds = ds.drop_vars('expver', errors='ignore')
    return ds

def get_window_doys(target_doy):
    window = []
    for offset in range(-2, 3):
        d = target_doy + offset
        if d < 1: d += 366
        elif d > 366: d -= 366
        window.append(d)
    return window

def build_synoptic_climatology():
    total_start = time.time()
    print("🚀 STARTE ATMOPULSE SYNOPTIC BUILDER (MSLP & Z500)")
    
    files = sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))
    ds_master = xr.open_mfdataset(files, combine='by_coords', parallel=True, preprocess=preprocess_era5t)
    ds_master = ds_master.sel(time=slice(None, CUTOFF_DATE))
    
    doys = ds_master['time'].dt.dayofyear.values
    years = ds_master['time'].dt.year.values
    lats = ds_master.latitude.values
    lons = ds_master.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    # Initialisierung
    print("🆕 Initialisiere Synoptik-Matrix...")
    data_vars = {}
    empty_3d = lambda: (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), np.nan, dtype=np.float32))

    for param in PARAMS:
        data_vars[f"{param}_mean_doy_A"] = empty_3d()
        data_vars[f"{param}_mean_doy_B"] = empty_3d()

    ds_syn = xr.Dataset(coords={"dayofyear": np.arange(1, 367), "latitude": lats, "longitude": lons}, data_vars=data_vars)

    mask_A_yr = (years >= EPOCH_A[0]) & (years <= EPOCH_A[1])
    mask_B_yr = (years >= EPOCH_B[0]) & (years <= EPOCH_B[1])

    for param in PARAMS:
        param_start = time.time()
        print(f"\n📥 Lade '{param}' in den RAM...")
        if param not in ds_master:
            print(f"⚠️ Parameter {param} nicht gefunden. Überspringe...")
            continue
            
        arr_full = ds_master[param].compute().values
        
        t0 = time.time()
        for target_doy in range(1, 367):
            idx = target_doy - 1
            win_doys = get_window_doys(target_doy)
            
            mask_win = np.isin(doys, win_doys)
            arr_win = arr_full[mask_win]
            years_win = years[mask_win]

            mask_A_win = (years_win >= EPOCH_A[0]) & (years_win <= EPOCH_A[1])
            mask_B_win = (years_win >= EPOCH_B[0]) & (years_win <= EPOCH_B[1])

            # Gleitender Mittelwert (Mean statt Percentile)
            if np.any(mask_A_win):
                ds_syn[f"{param}_mean_doy_A"].values[idx] = np.nanmean(arr_win[mask_A_win], axis=0)
            
            if np.any(mask_B_win):
                ds_syn[f"{param}_mean_doy_B"].values[idx] = np.nanmean(arr_win[mask_B_win], axis=0)
                
            if target_doy % 60 == 0 or target_doy == 366:
                print(f"   -> DOY {target_doy}/366 berechnet...", end="\r", flush=True)

        print(f"   -> '{param}' vollständig in {time.time() - param_start:.1f} Sekunden.")
        
        del arr_full
        gc.collect()

    ds_master.close()
    ds_syn.to_netcdf(OUT_FILE)
    print(f"\n🎉 SYNOPTIK-BASELINE GESICHERT: {OUT_FILE.name} (Gesamtzeit: {(time.time() - total_start):.1f}s)")

if __name__ == "__main__":
    build_synoptic_climatology()