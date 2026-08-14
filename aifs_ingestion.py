#!/usr/bin/env python3
"""
AtmoPulse Backend: Operational AIFS Ingestion (Zero-Latency Forecast)
Downloads the latest AIFS deterministic run (00Z) via ECMWF Open Data,
aggregates 6-hourly steps into daily extremes (Tx, Tn, Tg) and synoptic means,
and aligns the grid perfectly with the ERA5 baseline.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
import xarray as xr
from ecmwf.opendata import Client

# Suppress cfgrib warnings for cleaner logs
import warnings
warnings.filterwarnings("ignore", module="cfgrib")

# --- PATH CONFIGURATION ---
ROOT_DIR = Path("ERA5_ClimateTool")
TMP_DIR = ROOT_DIR / ".tmp_aifs"
OUT_DIR = ROOT_DIR / "Live_Forecasts"
REF_DIR = ROOT_DIR / "Reference_Climatology"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use one of your static ERA5 baselines just to extract the exact lat/lon grid
ERA5_GRID_REF = REF_DIR / "climatology_synoptics.nc"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def download_aifs_gribs():
    """
    Downloads current AIFS surface and pressure-level GRIB2 files via ECMWF Open Data.
    Fetches steps from 0 to 72 hours (3 days) in 6-hour intervals.
    """
    client = Client(source="ecmwf", model="aifs")
    steps = list(range(0, 73, 6)) # [0, 6, 12, ..., 72]
    
    target_sfc = str(TMP_DIR / "aifs_sfc.grib")
    target_pl = str(TMP_DIR / "aifs_pl.grib")
    
    logging.info("Downloading AIFS Surface variables (2t, msl)...")
    client.retrieve(
        type="fc",
        levtype="sfc",
        param=["2t", "msl"],
        step=steps,
        target=target_sfc
    )
    
    logging.info("Downloading AIFS Pressure Levels (z, t, u, v)...")
    client.retrieve(
        type="fc",
        levtype="pl",
        levelist=[300, 500, 850],
        param=["t", "z", "u", "v"],
        step=steps,
        target=target_pl
    )
    return target_sfc, target_pl

def process_and_align_aifs(sfc_file, pl_file):
    """
    Reads the GRIB files, converts Kelvin to Celsius, aggregates 6-hourly 
    steps into daily (24h) values, and regrids to the ERA5 0.25 deg baseline.
    """
    logging.info("Loading GRIB files into RAM via cfgrib...")
    
    # cfgrib translates GRIB2 to xarray datasets
    ds_sfc = xr.open_dataset(sfc_file, engine='cfgrib')
    ds_pl = xr.open_dataset(pl_file, engine='cfgrib')
    
    logging.info("Aggregating 6-hourly steps to daily extremes/means...")
    
    # ECMWF Open Data time coordinate is usually 'valid_time' based on step + base time
    ds_sfc = ds_sfc.rename({'valid_time': 'time'})
    ds_pl = ds_pl.rename({'valid_time': 'time'})
    
    # Extract Surface Parameters (Convert Kelvin to Celsius)
    t2m_celsius = ds_sfc['t2m'] - 273.15
    mslp_hpa = ds_sfc['msl'] / 100.0  # Convert Pa to hPa
    
    # Calculate daily parameters using xarray resampling
    tx = t2m_celsius.resample(time='1D').max()
    tn = t2m_celsius.resample(time='1D').min()
    tg = t2m_celsius.resample(time='1D').mean()
    mslp = mslp_hpa.resample(time='1D').mean()
    
    # Extract Pressure Level Parameters
    z500 = ds_pl['z'].sel(isobaricInhPa=500).resample(time='1D').mean()
    t850 = (ds_pl['t'].sel(isobaricInhPa=850) - 273.15).resample(time='1D').mean()
    u300 = ds_pl['u'].sel(isobaricInhPa=300).resample(time='1D').mean()
    v300 = ds_pl['v'].sel(isobaricInhPa=300).resample(time='1D').mean()
    
    # Combine into a single daily forecast dataset
    ds_forecast = xr.Dataset({
        'tx': tx.astype('float32'),
        'tn': tn.astype('float32'),
        'tg': tg.astype('float32'),
        'mslp': mslp.astype('float32'),
        'z500': z500.astype('float32'),
        't850': t850.astype('float32'),
        'u300': u300.astype('float32'),
        'v300': v300.astype('float32')
    })
    
    logging.info("Regridding AIFS output to match ERA5 baseline exactly...")
    with xr.open_dataset(ERA5_GRID_REF) as ds_era5:
        # Strict interpolation to prevent coordinate floating point mismatches in Streamlit
        ds_aligned = ds_forecast.interp(
            latitude=ds_era5.latitude, 
            longitude=ds_era5.longitude, 
            method="nearest"
        )
    
    return ds_aligned

def main():
    setup_logging()
    start_time = time.time()
    
    if not ERA5_GRID_REF.exists():
        logging.error(f"ERA5 reference file missing at {ERA5_GRID_REF}. Cannot align grids.")
        sys.exit(1)
        
    try:
        sfc_file, pl_file = download_aifs_gribs()
        ds_aligned = process_and_align_aifs(sfc_file, pl_file)
        
        # Determine the forecast base date
        base_date_str = pd.to_datetime(ds_aligned.time.values[0]).strftime("%Y%m%d")
        out_path = OUT_DIR / f"aifs_daily_forecast_{base_date_str}.nc"
        
        # Save as optimized NetCDF
        encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_aligned.data_vars}
        ds_aligned.to_netcdf(out_path, encoding=encoding)
        
        logging.info(f"SUCCESS! Live AIFS forecast saved to: {out_path.name}")
        
    except Exception as e:
        logging.error(f"AIFS Pipeline Failed: {str(e)}")
        
    finally:
        # Cleanup temporary GRIB files to save disk space
        if TMP_DIR.exists():
            import shutil
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            
    logging.info(f"Total processing time: {(time.time() - start_time):.1f} seconds.")

if __name__ == "__main__":
    import pandas as pd # Ensure pandas is available for datetime extraction
    main()