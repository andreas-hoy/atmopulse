#!/usr/bin/env python3
"""
AtmoPulse Backend: QDM Transfer Function Builder
Calculates the statistical bias between 20-year IFS Hindcasts and the ERA5 baseline.
Methodology: Quantile Delta Mapping (QDM) anchored on TX, DTR, and TG.
Generates a static lookup table (100 quantiles) for zero-latency live calibration.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import logging
import gc
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- PFADE (Platzhalter für deine spätere Ordnerstruktur) ---
ERA5_DIR = Path("ERA5_ClimateTool/Master_Batches")
IFS_HINDCAST_DIR = Path("ERA5_ClimateTool/IFS_Hindcasts") # Hier kommen später deine Hindcast-Daten rein
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "qdm_transfer_functions.nc"

# QDM Kalibrierungsperiode (Muss die Schnittmenge von ERA5 und Hindcast sein)
CALIB_START = '2004-01-01'
CALIB_END = '2023-12-31'
QUANTILES = np.linspace(0.01, 0.99, 99) # 99 Quantile für hochauflösendes Matching

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def get_window_doys(target_doy, window_size=31):
    """31-Tage Fenster für stabile CDFs bei nur 20 Jahren Hindcast-Daten."""
    half = window_size // 2
    window = []
    for offset in range(-half, half + 1):
        d = target_doy + offset
        if d < 1: d += 365
        elif d > 365: d -= 365
        window.append(d)
    return window

def calculate_empirical_quantiles(arr, doys, target_doy):
    """Extrahiert das 31-Tage-Fenster und berechnet die 99 Quantile."""
    win_doys = get_window_doys(target_doy, window_size=31)
    mask = np.isin(doys, win_doys)
    arr_win = arr[mask]
    
    if arr_win.shape[0] == 0:
        return np.full((len(QUANTILES), arr.shape[1], arr.shape[2]), np.nan, dtype=np.float32)
        
    return np.nanquantile(arr_win, QUANTILES * 100, axis=0).astype(np.float32)

def build_qdm_matrices():
    start_time = time.time()
    setup_logging()
    logging.info("🚀 STARTE QDM TRANSFER FUNCTION BUILDER (TX, DTR, TG)")

    # 1. ERA5 Baseline laden & vorbereiten
    logging.info("Lade ERA5 Baseline...")
    era5_files = sorted(list(ERA5_DIR.glob("era5_master_daily_*.nc")))
    ds_era5 = xr.open_mfdataset(era5_files, combine='by_coords')
    ds_era5 = ds_era5.sel(time=slice(CALIB_START, CALIB_END))
    
    # 365-Tage Kalender erzwingen
    ds_era5 = ds_era5.sel(time=~((ds_era5.time.dt.month == 2) & (ds_era5.time.dt.day == 29)))
    era5_doys = ds_era5.time.dt.dayofyear.values
    era5_is_leap = ds_era5.time.dt.is_leap_year.values
    era5_months = ds_era5.time.dt.month.values
    era5_doys = np.where(era5_is_leap & (era5_months >= 3), era5_doys - 1, era5_doys)

    # 2. IFS Hindcast laden & vorbereiten (Achtung: Diesen Block anpassen, sobald du die Daten hast)
    logging.info("Lade IFS Hindcasts...")
    try:
        ifs_files = sorted(list(IFS_HINDCAST_DIR.glob("ifs_hindcast_*.nc")))
        if not ifs_files:
            raise FileNotFoundError("Keine IFS Hindcast-Dateien gefunden.")
        ds_ifs = xr.open_mfdataset(ifs_files, combine='by_coords')
        ds_ifs = ds_ifs.sel(time=slice(CALIB_START, CALIB_END))
        
        ds_ifs = ds_ifs.sel(time=~((ds_ifs.time.dt.month == 2) & (ds_ifs.time.dt.day == 29)))
        ifs_doys = ds_ifs.time.dt.dayofyear.values
        ifs_is_leap = ds_ifs.time.dt.is_leap_year.values
        ifs_months = ds_ifs.time.dt.month.values
        ifs_doys = np.where(ifs_is_leap & (ifs_months >= 3), ifs_doys - 1, ifs_doys)
    except Exception as e:
        logging.error(f"⚠️ HINDCAST FEHLT: {e}")
        logging.error("Skript wird beendet. Bitte erst IFS Hindcasts herunterladen (2004-2023).")
        return

    # Geometrie aus ERA5 übernehmen
    lats = ds_era5.latitude.values
    lons = ds_era5.longitude.values
    n_lats, n_lons = len(lats), len(lons)

    # Output Dataset initialisieren
    logging.info("Initialisiere QDM NetCDF Matrix...")
    empty_4d = lambda: (("dayofyear", "quantile", "latitude", "longitude"), 
                        np.full((365, len(QUANTILES), n_lats, n_lons), np.nan, dtype=np.float32))
    
    ds_qdm = xr.Dataset(
        coords={
            "dayofyear": np.arange(1, 366), 
            "quantile": QUANTILES,
            "latitude": lats, 
            "longitude": lons
        },
        data_vars={
            "tx_bias": empty_4d(),
            "dtr_bias": empty_4d(),
            "tg_bias": empty_4d()
        }
    )

    # --- BERECHNUNG PRO PARAMETER ---
    variables = [
        ('tx', 'tx', 'tx_bias'),
        ('dtr', 'dtr', 'dtr_bias'),
        ('tg', 'tg', 'tg_bias')
    ]

    for var_era5, var_ifs, out_var in variables:
        logging.info(f"⚙️ Berechne Transferfunktionen für: {out_var.upper()}...")
        
        # DTR on the fly berechnen, falls es die Variable ist
        if var_era5 == 'dtr':
            arr_era5 = (ds_era5['tx'] - ds_era5['tn']).compute().values
            arr_ifs = (ds_ifs['tx'] - ds_ifs['tn']).compute().values
        else:
            arr_era5 = ds_era5[var_era5].compute().values
            arr_ifs = ds_ifs[var_ifs].compute().values

        t0 = time.time()
        for target_doy in range(1, 366):
            idx = target_doy - 1
            
            # Quantile für ERA5 (Beobachtung) und IFS (Modell) berechnen
            q_era5 = calculate_empirical_quantiles(arr_era5, era5_doys, target_doy)
            q_ifs = calculate_empirical_quantiles(arr_ifs, ifs_doys, target_doy)
            
            # Die eigentliche Bias-Matrix (Beobachtung minus Modell)
            # Im Live-Betrieb (ifs_ingestion.py) wird diese Matrix dann einfach auf den Forecast addiert
            bias_matrix = q_era5 - q_ifs
            
            ds_qdm[out_var].values[idx] = bias_matrix
            
            if target_doy % 30 == 0 or target_doy == 365:
                print(f"   -> DOY {target_doy}/365 berechnet...", end="\r", flush=True)
                
        logging.info(f"✅ {out_var.upper()} abgeschlossen in {time.time() - t0:.1f}s")
        
        del arr_era5, arr_ifs
        gc.collect()

    # Abspeichern
    logging.info(f"💾 Speichere finale QDM Lookup-Table: {OUT_FILE.name}")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_qdm.data_vars}
    ds_qdm.to_netcdf(OUT_FILE, encoding=encoding)
    
    ds_era5.close()
    ds_ifs.close()
    
    logging.info(f"🎉 ALLES FERTIG! Gesamtdauer: {(time.time() - start_time)/60:.1f} Minuten.")

if __name__ == "__main__":
    build_qdm_matrices()