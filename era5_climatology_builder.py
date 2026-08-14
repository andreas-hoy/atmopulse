#!/usr/bin/env python3
"""
AtmoPulse Backend: High-Performance Climatology Builder
- Single-Variable RAM-Load Architektur (Beseitigt I/O Flaschenhals)
- Speichert exaktes Datum (YYYYMMDD) für Rekorde
- Mit Live-Timer pro Verarbeitungsschritt
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

def preprocess_era5t(ds):
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

def build_climatology():
    total_start_time = time.time()
    print("🚀 STARTE ATMOPULSE CLIMATOLOGY BUILDER (RAM-Optimized I/O Mode)")
    
    files = sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))
    ds_master = xr.open_mfdataset(files, combine='by_coords', parallel=True, preprocess=preprocess_era5t)
    ds_master = ds_master.sel(time=slice(None, CUTOFF_DATE))
    print(f"📅 Daten-Cutoff: {CUTOFF_DATE}")
    
    doys = ds_master['time'].dt.dayofyear.values
    years = ds_master['time'].dt.year.values
    # Konvertiere Zeitachse in Integer YYYYMMDD für den Frontend-Tooltip
    dates_int = ds_master.time.dt.strftime("%Y%m%d").values.astype(np.int32)
    
    lats = ds_master.latitude.values
    lons = ds_master.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    # Initialisierung der Ziel-Matrix
    print("🆕 Initialisiere neue NetCDF-Matrix...")
    data_vars = {}
    empty_2d = lambda: (("latitude", "longitude"), np.full((n_lats, n_lons), np.nan, dtype=np.float32))
    empty_3d = lambda: (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), np.nan, dtype=np.float32))
    empty_3d_int = lambda: (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), -1, dtype=np.int32))

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
        
        # Geändert: max_date / min_date statt max_year
        data_vars[f"{param}_max_val"] = empty_3d()
        data_vars[f"{param}_max_date"] = empty_3d_int()
        data_vars[f"{param}_min_val"] = empty_3d()
        data_vars[f"{param}_min_date"] = empty_3d_int()

    ds_clim = xr.Dataset(coords={"dayofyear": np.arange(1, 367), "latitude": lats, "longitude": lons}, data_vars=data_vars)

    print("\n=======================================================")
    print("⚡ BERECHNUNG PARAMETER FÜR PARAMETER (RAM-ISOLIERT)")
    print("=======================================================")

    mask_A_yr = (years >= EPOCH_A[0]) & (years <= EPOCH_A[1])
    mask_B_yr = (years >= EPOCH_B[0]) & (years <= EPOCH_B[1])

    for param in PARAMS:
        param_start = time.time()
        print(f"\n📥 Lade '{param}' vollständig in den RAM (ca. 2.5 GB)...")
        # I/O Flaschenhals eliminieren: Lade die gesamte 86-Jahre Zeitreihe für diesen einen Parameter in den RAM
        arr_full = ds_master[param].compute().values
        load_time = time.time() - param_start
        print(f"✅ Geladen in {load_time:.1f}s. Starte Matrix-Berechnungen...")

        # --- PHASE 1: SAISONAL (JJA & DJF) ---
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
        
        print(f"   -> Saisonale Perzentile (DJF/JJA) berechnet in {time.time() - t0:.1f}s")

        # --- PHASE 2: TÄGLICHE 5-TAGE-FENSTER & REKORDE ---
        t0 = time.time()
        for target_doy in range(1, 367):
            idx = target_doy - 1
            win_doys = get_window_doys(target_doy)
            
            # Sub-Maske für das 5-Tage Fenster (über 86 Jahre)
            mask_win = np.isin(doys, win_doys)
            arr_win = arr_full[mask_win]
            dates_win = dates_int[mask_win]
            years_win = years[mask_win]

            mask_A_win = (years_win >= EPOCH_A[0]) & (years_win <= EPOCH_A[1])
            mask_B_win = (years_win >= EPOCH_B[0]) & (years_win <= EPOCH_B[1])

            # Perzentile Epoche A & B
            if np.any(mask_A_win):
                pcts_A = np.nanpercentile(arr_win[mask_A_win], PCTS_DOY, axis=0)
                for i, p in enumerate(PCTS_DOY):
                    ds_clim[f"{param}_p{p}_doy_A"].values[idx] = pcts_A[i]
            
            if np.any(mask_B_win):
                pcts_B = np.nanpercentile(arr_win[mask_B_win], PCTS_DOY, axis=0)
                for i, p in enumerate(PCTS_DOY):
                    ds_clim[f"{param}_p{p}_doy_B"].values[idx] = pcts_B[i]

            # Absolute Rekorde & Exaktes Datum
            max_idx = np.nanargmax(arr_win, axis=0)
            ds_clim[f"{param}_max_val"].values[idx] = np.take_along_axis(arr_win, np.expand_dims(max_idx, axis=0), axis=0).squeeze()
            date_grid = np.broadcast_to(dates_win[:, None, None], arr_win.shape)
            ds_clim[f"{param}_max_date"].values[idx] = np.take_along_axis(date_grid, np.expand_dims(max_idx, axis=0), axis=0).squeeze()

            min_idx = np.nanargmin(arr_win, axis=0)
            ds_clim[f"{param}_min_val"].values[idx] = np.take_along_axis(arr_win, np.expand_dims(min_idx, axis=0), axis=0).squeeze()
            ds_clim[f"{param}_min_date"].values[idx] = np.take_along_axis(date_grid, np.expand_dims(min_idx, axis=0), axis=0).squeeze()
            
            # Terminal Update (alle 30 Tage überschreiben für sauberes Log)
            if target_doy % 30 == 0 or target_doy == 366:
                print(f"   -> DOY {target_doy}/366 berechnet...", end="\r", flush=True)

        print(f"   -> Alle 366 Tagesfenster berechnet in {time.time() - t0:.1f}s")
        
        # RAM rigoros leeren vor dem nächsten Parameter
        del arr_full
        gc.collect()

        # Checkpointing nach jedem fertigen Parameter
        print(f"💾 Speichere Zwischenstand für '{param}'...")
        temp_write = OUT_DIR / "temp_step.nc"
        ds_clim.to_netcdf(temp_write)
        os.replace(temp_write, TEMP_FILE)
        print(f"🏁 Parameter '{param}' vollständig in {(time.time() - param_start)/60:.1f} Min.")

    ds_master.close()
    os.replace(TEMP_FILE, OUT_FILE)
    print("\n=======================================================")
    print(f"🎉 ALLES FERTIG! Gesamtdauer: {(time.time() - total_start_time)/60:.1f} Minuten.")
    print(f"✅ Finale Klimatologie gesichert in: {OUT_FILE.name}")

if __name__ == "__main__":
    build_climatology()