import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
LIVE_DIR = Path("ERA5_ClimateTool/Live_Forecasts")

def get_synoptic_map_data(date_str):
    target_dt = pd.to_datetime(date_str)
    
    today = pd.Timestamp.now().normalize()
    if target_dt >= (today - pd.Timedelta(days=5)):
        mslp_files = sorted(list(LIVE_DIR.glob("live_forecast_mslp.nc")))
        z500_files = sorted(list(LIVE_DIR.glob("live_forecast_z500.nc")))
        txtn_files = sorted(list(LIVE_DIR.glob("live_forecast_txtn.nc")))
        
        if not mslp_files or not z500_files or not txtn_files:
            mslp_files = sorted(list(DATA_DIR.glob("era5_mslp_batch_*.nc")))
            z500_files = sorted(list(DATA_DIR.glob("era5_z500_batch_*.nc")))
            txtn_files = sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))
    else:
        mslp_files = sorted(list(DATA_DIR.glob("era5_mslp_batch_*.nc")))
        z500_files = sorted(list(DATA_DIR.glob("era5_z500_batch_*.nc")))
        txtn_files = sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))
    
    if not mslp_files or not z500_files or not txtn_files:
        raise FileNotFoundError(f"Files not found in {DATA_DIR} or {LIVE_DIR}!")

    ds_mslp = xr.open_mfdataset(mslp_files, combine='nested', concat_dim='valid_time').sortby('valid_time').drop_duplicates(dim='valid_time')
    ds_z500 = xr.open_mfdataset(z500_files, combine='nested', concat_dim='valid_time').sortby('valid_time').drop_duplicates(dim='valid_time')
    ds_txtn = xr.open_mfdataset(txtn_files, combine='nested', concat_dim='valid_time').sortby('valid_time').drop_duplicates(dim='valid_time')

    try:
        slice_mslp = ds_mslp.sel(valid_time=target_dt, method='nearest').load()
        slice_z500 = ds_z500.sel(valid_time=target_dt, method='nearest').load()
        slice_txtn = ds_txtn.sel(valid_time=target_dt, method='nearest').load()

        mslp_hPa = slice_mslp['msl'] / 100.0  
        z500_gpdm = slice_z500['z'] / 9.80665 / 10.0  
        tx_celsius = slice_txtn['mx2t'] - 273.15
        tn_celsius = slice_txtn['mn2t'] - 273.15

        clean_slices = {}
        for name, da in zip(["mslp", "z500", "tx", "tn"], [mslp_hPa, z500_gpdm, tx_celsius, tn_celsius]):
            if "expver" in da.dims:
                da = da.dropna(dim="expver", how="all").isel(expver=0)
                if "expver" in da.coords: 
                    da = da.drop_vars("expver")
            clean_slices[name] = da
            
        return clean_slices
    finally:
        ds_mslp.close()
        ds_z500.close()
        ds_txtn.close()