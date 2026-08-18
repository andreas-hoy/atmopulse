#!/usr/bin/env python3
"""
AtmoPulse Backend: Operational IFS Ingestion
Downloads the stable IFS deterministic run, resolves xarray time-coordinate conflicts,
aggregates daily extremes (0-0Z) & 12Z synoptics, and applies CDO conservative regridding.
ETCCDI 365-day DOY mapping and fracarea coastal normalisation included.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import xarray as xr
import numpy as np
import scipy.sparse as sps
from ecmwf.opendata import Client
import warnings

warnings.filterwarnings("ignore", module="cfgrib")

BASE_DIR = Path.cwd() / "ERA5_ClimateTool"
TMP_DIR = BASE_DIR / ".tmp_ifs"
OUT_DIR = BASE_DIR / "Live_Forecasts"
REF_DIR = BASE_DIR / "Reference_Climatology"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERA5_GRID_REF = REF_DIR / "climatology_synoptics.nc"
WEIGHTS_FILE = REF_DIR / "regrid_weights_cdo.nc"

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

def download_ifs_gribs():
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    steps = list(range(0, 73, 6))
    target_sfc = str(TMP_DIR / "ifs_sfc.grib")
    target_pl = str(TMP_DIR / "ifs_pl.grib")
    
    now = datetime.now(timezone.utc)
    candidates = [(now - timedelta(hours=12*i), 12 if (now - timedelta(hours=12*i)).hour >= 12 else 0) for i in range(6)]
    
    for dt, hour in candidates:
        date_str = dt.strftime("%Y%m%d")
        logging.info(f"🔍 Prüfe Lauf {date_str} {hour:02d}Z...")
        try:
            client.retrieve(date=date_str, time=hour, type="fc", levtype="sfc", param=["2t", "msl"], step=steps, target=target_sfc)
            client.retrieve(date=date_str, time=hour, type="fc", levtype="pl", levelist=[300, 500, 850], param=["t", "z", "u", "v"], step=steps, target=target_pl)
            return target_sfc, target_pl, date_str, hour
        except Exception as e:
            logging.warning(f"⚠️ Lauf nicht verfügbar. Gehe zu vorherigem...")
            time.sleep(2)
    raise RuntimeError("❌ Kein Lauf gefunden.")

def apply_conservative_weights(ds_source, weights_file, ds_target_grid):
    logging.info("⚙️ Harmoniere Gitter und appliziere flächenkonservative CDO-Matrix (fracarea) via SciPy...")
    
    ds_source = ds_source.assign_coords(longitude=(((ds_source.longitude + 180) % 360) - 180)).sortby('longitude')
    
    ds_source_cropped = ds_source.sel(
        latitude=ds_target_grid.latitude, 
        longitude=ds_target_grid.longitude, 
        method='nearest'
    )
    
    with xr.open_dataset(weights_file) as ds_w:
        weights = sps.coo_matrix(
            (ds_w['remap_matrix'][:, 0].values, (ds_w['dst_address'].values - 1, ds_w['src_address'].values - 1)),
            shape=(ds_w.sizes['dst_grid_size'], ds_w.sizes['src_grid_size'])
        ).tocsr()
    
    shape_out = (ds_target_grid.sizes['latitude'], ds_target_grid.sizes['longitude'])
    ds_out = xr.Dataset(coords={
        'time': ds_source_cropped.time, 
        'latitude': ds_target_grid.latitude, 
        'longitude': ds_target_grid.longitude
    })
    
    for var in ds_source_cropped.data_vars:
        data_arrays = []
        for t_idx in range(ds_source_cropped.sizes['time']):
            flat_source = ds_source_cropped[var].isel(time=t_idx).values.flatten()
            
            # WISSENSCHAFTLICHER FIX: Fracarea Normalisierung
            valid_mask = ~np.isnan(flat_source)
            source_filled = np.where(valid_mask, flat_source, 0.0)
            
            y_num = weights.dot(source_filled)
            y_den = weights.dot(valid_mask.astype(np.float32))
            
            # Division durch Null abfangen für Ozean-Pixel ohne valide Nachbarn
            with np.errstate(divide='ignore', invalid='ignore'):
                y_corrected = np.where(y_den > 0, y_num / y_den, np.nan)
            
            data_arrays.append(y_corrected.reshape(shape_out))
            
        ds_out[var] = (("time", "latitude", "longitude"), np.array(data_arrays, dtype=np.float32))
        
    return ds_out

def process_and_align_ifs(sfc_file, pl_file):
    logging.info("Lade GRIBs und behebe xarray 'time' Konflikte...")
    ds_sfc = xr.open_dataset(sfc_file, engine='cfgrib')
    ds_pl = xr.open_dataset(pl_file, engine='cfgrib')
    
    ds_sfc = ds_sfc.swap_dims({'step': 'valid_time'})
    ds_pl = ds_pl.swap_dims({'step': 'valid_time'})
    
    if 'time' in ds_sfc.coords: ds_sfc = ds_sfc.drop_vars('time')
    if 'time' in ds_pl.coords: ds_pl = ds_pl.drop_vars('time')
    
    ds_sfc = ds_sfc.rename({'valid_time': 'time'})
    ds_pl = ds_pl.rename({'valid_time': 'time'})
    
    t2m_celsius = ds_sfc['t2m'] - 273.15
    mslp_hpa = ds_sfc['msl'] / 100.0  
    
    logging.info("Aggregiere 0-0Z für T-Extreme und 12Z für Synoptik...")
    tx = t2m_celsius.resample(time='1D').max()
    tn = t2m_celsius.resample(time='1D').min()
    tg = t2m_celsius.resample(time='1D').mean()
    
    ds_pl_12z = ds_pl.sel(time=ds_pl.time.dt.hour == 12)
    mslp_12z = mslp_hpa.sel(time=mslp_hpa.time.dt.hour == 12)
    
    mslp = mslp_12z.resample(time='1D').first()
    
    z500 = ds_pl_12z['z'].sel(isobaricInhPa=500).resample(time='1D').first().drop_vars('isobaricInhPa')
    t850 = (ds_pl_12z['t'].sel(isobaricInhPa=850) - 273.15).resample(time='1D').first().drop_vars('isobaricInhPa')
    u300 = ds_pl_12z['u'].sel(isobaricInhPa=300).resample(time='1D').first().drop_vars('isobaricInhPa')
    v300 = ds_pl_12z['v'].sel(isobaricInhPa=300).resample(time='1D').first().drop_vars('isobaricInhPa')
    
    ds_forecast = xr.Dataset({
        'tx': tx.astype('float32'), 'tn': tn.astype('float32'), 'tg': tg.astype('float32'),
        'mslp': mslp.astype('float32'), 'z500': z500.astype('float32'),
        't850': t850.astype('float32'), 'u300': u300.astype('float32'), 'v300': v300.astype('float32')
    })
    
    with xr.open_dataset(ERA5_GRID_REF) as ds_era5:
        ds_aligned = apply_conservative_weights(ds_forecast, WEIGHTS_FILE, ds_era5)
        
    # ETCCDI-STANDARD: 365-Tage Kalender Mapping für das Frontend und QDM
    raw_doys = ds_aligned.time.dt.dayofyear.values
    is_leap = ds_aligned.time.dt.is_leap_year.values
    months = ds_aligned.time.dt.month.values
    doy_365 = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    ds_aligned = ds_aligned.assign_coords(doy_365=("time", doy_365))
        
    return ds_aligned

def purge_old_forecasts(days_to_keep=10):
    logging.info(f"Garbage Collection: Lösche IFS-Rückstände älter als {days_to_keep} Tage...")
    now = time.time()
    for f in OUT_DIR.glob("ifs_daily_forecast_*.nc"):
        if os.stat(f).st_mtime < now - (days_to_keep * 86400):
            try:
                f.unlink()
                logging.info(f"Gelöscht: {f.name}")
            except Exception as e:
                logging.error(f"Löschen fehlgeschlagen für {f.name}: {e}")

def main():
    setup_logging()
    
    if not ERA5_GRID_REF.exists() or not WEIGHTS_FILE.exists():
        logging.error("Referenz- oder Gewichtsdatei fehlt!")
        sys.exit(1)
        
    try:
        sfc_file, pl_file, run_date, run_hour = download_ifs_gribs() 
        ds_aligned = process_and_align_ifs(sfc_file, pl_file)
        
        out_path = OUT_DIR / f"ifs_daily_forecast_{run_date}_{run_hour:02d}z.nc"
        
        encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_aligned.data_vars}
        ds_aligned.to_netcdf(out_path, encoding=encoding)
        logging.info(f"🎉 SUCCESS! Live Vorhersage gespeichert: {out_path.name}")
        
    except Exception as e:
        logging.error(f"IFS Pipeline fehlgeschlagen: {str(e)}")
    finally:
        if TMP_DIR.exists():
            import shutil
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        purge_old_forecasts(days_to_keep=10)

if __name__ == "__main__":
    main()