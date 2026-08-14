#!/usr/bin/env python3
"""
AtmoPulse Backend: Operativer ERA5 Daily Updater (90-Tage Rolling Window)
Entkoppelte Parameter, UTF-8-kompatibles Logging, stabiles Jahres-Splitting.
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
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(root / "era5_daily_update.log", encoding="utf-8")
        ]
    )

def harmonize_time(ds: xr.Dataset) -> xr.Dataset:
    if 'valid_time' in ds.dims: ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords: ds = ds.rename({'valid_time': 'time'})
    return ds

def main():
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=50, sleep_max=120)
    
    today = datetime.utcnow()
    end_date = today - timedelta(days=5)
    start_date = end_date - timedelta(days=90)
    
    date_string = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    logging.info("START ERA5 ROLLING UPDATE | Zeitraum: %s", date_string)

    tmp_dir = root / ".tmp_update"
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Synoptik
        logging.info("Fordere Synoptik (MSLP & Pressure Levels) an...")
        client.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "variable": "mean_sea_level_pressure", 
            "date": date_string, "time": "12:00", "area": DEFAULT_AREA, "format": "netcdf"
        }, str(tmp_dir / "syn_mslp.nc"))
        
        client.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "variable": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"], 
            "pressure_level": ["300", "500", "850"], "date": date_string, "time": "12:00", "area": DEFAULT_AREA, "format": "netcdf"
        }, str(tmp_dir / "syn_press.nc"))

        # 2. Extreme (Entkoppelt)
        base_req = {
            "product_type": "reanalysis", "date": date_string, 
            "time": [f"{h:02d}:00" for h in range(0, 24)], "area": DEFAULT_AREA, "format": "netcdf"
        }
        
        logging.info("Fordere TG, TX, TN an...")
        req_tg = base_req.copy()
        req_tg["variable"] = "2m_temperature"
        client.retrieve("reanalysis-era5-single-levels", req_tg, str(tmp_dir / "t2m.nc"))

        req_tx = base_req.copy()
        req_tx["variable"] = "maximum_2m_temperature_since_previous_post_processing"
        client.retrieve("reanalysis-era5-single-levels", req_tx, str(tmp_dir / "mx2t.nc"))

        req_tn = base_req.copy()
        req_tn["variable"] = "minimum_2m_temperature_since_previous_post_processing"
        client.retrieve("reanalysis-era5-single-levels", req_tn, str(tmp_dir / "mn2t.nc"))

        # 3. Aggregation
        logging.info("Verarbeite Daten und harmonisiere...")
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

            ds_update = xr.Dataset({"mslp": ds_mslp["msl"].astype('float32')})
            ds_update["z500"] = ds_press["z"].sel(pressure_level=500, drop=True).astype('float32')
            ds_update["t850"] = (ds_press["t"].sel(pressure_level=850, drop=True) - 273.15).astype('float32')
            ds_update["u300"] = ds_press["u"].sel(pressure_level=300, drop=True).astype('float32')
            ds_update["v300"] = ds_press["v"].sel(pressure_level=300, drop=True).astype('float32')
            if "pressure_level" in ds_update.coords: ds_update = ds_update.drop_vars("pressure_level")

            ds_update["tx"] = (ds_mx2t["mx2t"].resample(time='1D').max() - 273.15).astype('float32')
            ds_update["tn"] = (ds_mn2t["mn2t"].resample(time='1D').min() - 273.15).astype('float32')
            ds_update["tg"] = (ds_t2m["t2m"].resample(time='1D').mean() - 273.15).astype('float32')
            ds_update['time'] = ds_update['time'].dt.floor('D')

            for var in ["tx", "tn", "tg", "t850"]: ds_update[var].attrs["units"] = "Celsius"

        encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_update.data_vars}
        
        for year, ds_year in ds_update.groupby('time.year'):
            year_str = str(year)
            master_file = root / f"era5_master_daily_{year_str}.nc"
            logging.info("Aktualisiere Masterdatei fuer %s...", year_str)
            
            if not master_file.exists():
                ds_year.to_netcdf(master_file, encoding=encoding)
            else:
                with xr.open_dataset(master_file) as ds_master:
                    ds_master = ds_master.load()
                    overlap_start = ds_year.time.min().values
                    ds_master_clean = ds_master.where(ds_master.time < overlap_start, drop=True)
                    ds_final = xr.concat([ds_master_clean, ds_year], dim='time')
                    ds_final = ds_final.sortby('time')
                    temp_master = root / f".temp_master_{year_str}.nc"
                    ds_final.to_netcdf(temp_master, encoding=encoding)
                
                os.replace(temp_master, master_file)

        logging.info("UPDATE ERFOLGREICH BEENDET. Stand: %s", end_date.strftime('%Y-%m-%d'))
        
    except Exception as e:
        logging.error("UPDATE FEHLGESCHLAGEN: %s", str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()