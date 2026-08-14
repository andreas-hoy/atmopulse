#!/usr/bin/env python3
"""
AtmoPulse Backend: Initial-Download 2026 (Tx, Tn, Tg, MSLP, T850, Z500, Jets).
Entkoppelte Temperatur-Requests (Verhindert CDS-ZIP-Fallback).
"""

import logging
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import cdsapi
import xarray as xr

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]

def setup_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

def harmonize_time(ds: xr.Dataset) -> xr.Dataset:
    if 'valid_time' in ds.dims: ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords: ds = ds.rename({'valid_time': 'time'})
    return ds

def main():
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=50, sleep_max=120)
    
    target_year = 2026
    end_date = datetime.utcnow() - timedelta(days=5)
    
    start_date_str = f"{target_year}-01-01"
    end_date_str = end_date.strftime('%Y-%m-%d')
    date_string = f"{start_date_str}/{end_date_str}"
    
    master_file = root / f"era5_master_daily_{target_year}.nc"
    if master_file.exists():
        logging.error("❌ Masterdatei %s existiert bereits. Bitte löschen oder Update-Skript nutzen.", master_file.name)
        sys.exit(1)

    logging.info("🚀 STARTE INIT-DOWNLOAD FÜR %s | Zeitraum: %s", target_year, date_string)
    tmp_dir = root / ".tmp_init"
    tmp_dir.mkdir(exist_ok=True)

    try:
        # ==============================================================================
        # A) Synoptik
        # ==============================================================================
        logging.info("⏳ Fordere Synoptik (MSLP) an...")
        client.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "variable": "mean_sea_level_pressure", 
            "date": date_string, "time": "12:00", "area": DEFAULT_AREA, "format": "netcdf"
        }, str(tmp_dir / "syn_mslp.nc"))
        
        logging.info("⏳ Fordere Synoptik (Pressure Levels) an...")
        client.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "variable": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"], 
            "pressure_level": ["300", "500", "850"], "date": date_string, "time": "12:00", "area": DEFAULT_AREA, "format": "netcdf"
        }, str(tmp_dir / "syn_press.nc"))

        # ==============================================================================
        # B) Extreme (00-23 UTC) - ENTKOPPELT!
        # ==============================================================================
        base_req = {
            "product_type": "reanalysis", "date": date_string, 
            "time": [f"{h:02d}:00" for h in range(0, 24)], "area": DEFAULT_AREA, "format": "netcdf"
        }
        
        logging.info("⏳ Fordere Temperatur (TG) an...")
        req_tg = base_req.copy()
        req_tg["variable"] = "2m_temperature"
        client.retrieve("reanalysis-era5-single-levels", req_tg, str(tmp_dir / "t2m.nc"))

        logging.info("⏳ Fordere Temperatur (TX) an...")
        req_tx = base_req.copy()
        req_tx["variable"] = "maximum_2m_temperature_since_previous_post_processing"
        client.retrieve("reanalysis-era5-single-levels", req_tx, str(tmp_dir / "mx2t.nc"))

        logging.info("⏳ Fordere Temperatur (TN) an...")
        req_tn = base_req.copy()
        req_tn["variable"] = "minimum_2m_temperature_since_previous_post_processing"
        client.retrieve("reanalysis-era5-single-levels", req_tn, str(tmp_dir / "mn2t.nc"))

        # ==============================================================================
        # C) Lokales Aggregation & Merging
        # ==============================================================================
        logging.info("⚙️ Verarbeite und konvertiere Daten in den RAM...")
        with xr.open_dataset(tmp_dir / "syn_mslp.nc") as ds_mslp, \
             xr.open_dataset(tmp_dir / "syn_press.nc") as ds_press, \
             xr.open_dataset(tmp_dir / "t2m.nc") as ds_t2m, \
             xr.open_dataset(tmp_dir / "mx2t.nc") as ds_mx2t, \
             xr.open_dataset(tmp_dir / "mn2t.nc") as ds_mn2t:
            
            ds_mslp = harmonize_time(ds_mslp.load())
            ds_press = harmonize_time(ds_press.load())
            ds_t2m = harmonize_time(ds_t2m.load())
            ds_mx2t = harmonize_time(ds_mx2t.load())
            ds_mn2t = harmonize_time(ds_mn2t.load())

            ds_out = xr.Dataset({"mslp": ds_mslp["msl"].astype('float32')})
            ds_out["z500"] = ds_press["z"].sel(pressure_level=500, drop=True).astype('float32')
            ds_out["t850"] = (ds_press["t"].sel(pressure_level=850, drop=True) - 273.15).astype('float32')
            ds_out["u300"] = ds_press["u"].sel(pressure_level=300, drop=True).astype('float32')
            ds_out["v300"] = ds_press["v"].sel(pressure_level=300, drop=True).astype('float32')
            if "pressure_level" in ds_out.coords: ds_out = ds_out.drop_vars("pressure_level")

            ds_out["tx"] = (ds_mx2t["mx2t"].resample(time='1D').max() - 273.15).astype('float32')
            ds_out["tn"] = (ds_mn2t["mn2t"].resample(time='1D').min() - 273.15).astype('float32')
            ds_out["tg"] = (ds_t2m["t2m"].resample(time='1D').mean() - 273.15).astype('float32')
            
            ds_out['time'] = ds_out['time'].dt.floor('D')

            for var in ["tx", "tn", "tg", "t850"]: ds_out[var].attrs["units"] = "Celsius"
            
            encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_out.data_vars}
            ds_out.to_netcdf(master_file, encoding=encoding)
            logging.info("🎉 INITIALISIERUNG ABGESCHLOSSEN: %s", master_file.name)

    except Exception as e:
        logging.error("❌ FEHLER: %s", str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()