import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import gc

# --- PFADE ---
DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
OLD_CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")
NEW_CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")

print("1. Lade bestehende Klimatologie...")
with xr.open_dataset(OLD_CLIM_FILE) as old_ds:
    out_ds = old_ds.copy(deep=True)

print("2. Lade ERA5 Master-Batches (Rohdaten)...")
files = sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))

ds = xr.open_mfdataset(files, combine='nested', concat_dim='valid_time').sortby('valid_time')

# Zeitachse säubern
_, unique_index = np.unique(ds.valid_time.values, return_index=True)
ds = ds.isel(valid_time=unique_index)

ds_a = ds.sel(valid_time=slice('1961-01-01', '1990-12-31'))
ds_b = ds.sel(valid_time=slice('1996-01-01', '2025-12-31'))
lats, lons = ds.latitude.values, ds.longitude.values

print("3. Bereite Datenstruktur für fehlende Variablen vor...")
new_vars = ['tx_p5', 'tx_p10', 'tx_p25', 'tn_p75', 'tn_p90', 'tn_p95']
for v in new_vars:
    for epoch in ['A', 'B']:
        var_name = f'{v}_doy_{epoch}'
        if var_name not in out_ds:
            out_ds[var_name] = (('dayofyear', 'latitude', 'longitude'), np.zeros((366, len(lats), len(lons))))

if 'tx_min_val' not in out_ds:
    out_ds['tx_min_val'] = (('dayofyear', 'latitude', 'longitude'), np.zeros((366, len(lats), len(lons))))
    out_ds['tx_min_year'] = (('dayofyear', 'latitude', 'longitude'), np.zeros((366, len(lats), len(lons))))
if 'tn_max_val' not in out_ds:
    out_ds['tn_max_val'] = (('dayofyear', 'latitude', 'longitude'), np.zeros((366, len(lats), len(lons))))
    out_ds['tn_max_year'] = (('dayofyear', 'latitude', 'longitude'), np.zeros((366, len(lats), len(lons))))

print("4. Starte Berechnung der FEHLENDEN Anomalien (In-Memory Speed-Mode)...")

def get_window_doys(target_doy):
    window = []
    for offset in range(-2, 3):
        d = target_doy + offset
        if d < 1: d += 366
        elif d > 366: d -= 366
        window.append(d)
    return window

# ==========================================
# PHASE 1: EPOCH A (1961-1990)
# ==========================================
print("\n| -> Lade Epoch A in den RAM (Dask-Bypass)... ", end="", flush=True)
ds_a_mem = ds_a[['mx2t', 'mn2t']].load()
# Rohdaten liegen in Kelvin vor (ERA5-Konvention) - fuer Celsius-Klimatologie umrechnen,
# analog zu era5_climatology_builder.py. Ohne dies bleiben die neuen Perzentile in Kelvin
# und sind gegen die Celsius-Messwerte im Frontend nicht mehr vergleichbar.
ds_a_mem['mx2t'] = ds_a_mem['mx2t'] - 273.15
ds_a_mem['mn2t'] = ds_a_mem['mn2t'] - 273.15
doys_a = pd.to_datetime(ds_a_mem.valid_time.values).dayofyear.values
print("Erledigt. Berechne Quantile...")

for i in range(366):
    print(f"\r| -> Verarbeite Tag {i+1:03d}/366 (Epoch A)... ", end="", flush=True)
    mask_a = np.isin(doys_a, get_window_doys(i + 1))
    da_a = ds_a_mem.isel(valid_time=mask_a)
    
    qa_tx_cold = da_a['mx2t'].quantile([0.05, 0.10, 0.25], dim='valid_time')
    qa_tn_warm = da_a['mn2t'].quantile([0.75, 0.90, 0.95], dim='valid_time')
    
    out_ds['tx_p5_doy_A'][i], out_ds['tx_p10_doy_A'][i], out_ds['tx_p25_doy_A'][i] = qa_tx_cold.sel(quantile=0.05).values, qa_tx_cold.sel(quantile=0.10).values, qa_tx_cold.sel(quantile=0.25).values
    out_ds['tn_p75_doy_A'][i], out_ds['tn_p90_doy_A'][i], out_ds['tn_p95_doy_A'][i] = qa_tn_warm.sel(quantile=0.75).values, qa_tn_warm.sel(quantile=0.90).values, qa_tn_warm.sel(quantile=0.95).values
print("\n| -> Epoch A Quantile erfolgreich berechnet.")

del ds_a_mem
gc.collect()

# ==========================================
# PHASE 2: EPOCH B (1996-2025)
# ==========================================
print("| -> Lade Epoch B in den RAM... ", end="", flush=True)
ds_b_mem = ds_b[['mx2t', 'mn2t']].load()
# Siehe Kommentar bei Epoch A: Kelvin -> Celsius vor der Perzentilberechnung.
ds_b_mem['mx2t'] = ds_b_mem['mx2t'] - 273.15
ds_b_mem['mn2t'] = ds_b_mem['mn2t'] - 273.15
doys_b = pd.to_datetime(ds_b_mem.valid_time.values).dayofyear.values
print("Erledigt. Berechne Quantile...")

for i in range(366):
    print(f"\r| -> Verarbeite Tag {i+1:03d}/366 (Epoch B)... ", end="", flush=True)
    mask_b = np.isin(doys_b, get_window_doys(i + 1))
    da_b = ds_b_mem.isel(valid_time=mask_b)
    
    qb_tx_cold = da_b['mx2t'].quantile([0.05, 0.10, 0.25], dim='valid_time')
    qb_tn_warm = da_b['mn2t'].quantile([0.75, 0.90, 0.95], dim='valid_time')
    
    out_ds['tx_p5_doy_B'][i], out_ds['tx_p10_doy_B'][i], out_ds['tx_p25_doy_B'][i] = qb_tx_cold.sel(quantile=0.05).values, qb_tx_cold.sel(quantile=0.10).values, qb_tx_cold.sel(quantile=0.25).values
    out_ds['tn_p75_doy_B'][i], out_ds['tn_p90_doy_B'][i], out_ds['tn_p95_doy_B'][i] = qb_tn_warm.sel(quantile=0.75).values, qb_tn_warm.sel(quantile=0.90).values, qb_tn_warm.sel(quantile=0.95).values
print("\n| -> Epoch B Quantile erfolgreich berechnet.")

del ds_b_mem
gc.collect()

# ==========================================
# PHASE 3: ALL-TIME RECORDS (Dask-Modus für RAM-Sicherheit)
# ==========================================
print("| -> Berechne All-Time Rekorde (Dask-Modus für RAM-Sicherheit)...")

# KEIN .load() hier! Wir lassen Dask die Arbeit machen, um die 11.2 GB RAM zu sparen.
ds_all_tx = ds[['mx2t']]
ds_all_tn = ds[['mn2t']]
doys_all = pd.to_datetime(ds.valid_time.values).dayofyear.values
year_array = pd.to_datetime(ds.valid_time.values).year.values

for i in range(366):
    print(f"\r| -> Verarbeite Tag {i+1:03d}/366 (All-Time TX & TN)... ", end="", flush=True)
    mask_all = np.isin(doys_all, get_window_doys(i + 1))
    
    da_all_tx = ds_all_tx.isel(valid_time=mask_all)
    da_all_tn = ds_all_tn.isel(valid_time=mask_all)
    
    # .compute() zwingt Dask zur direkten Berechnung der Min/Max-Werte (ohne RAM-Kollaps)
    res_min_tx = da_all_tx['mx2t'].min(dim='valid_time').compute()
    idx_min_tx = da_all_tx['mx2t'].argmin(dim='valid_time').compute()
    
    res_max_tn = da_all_tn['mn2t'].max(dim='valid_time').compute()
    idx_max_tn = da_all_tn['mn2t'].argmax(dim='valid_time').compute()
    
    # Werte für TX (Kälterekorde) zuweisen
    out_ds['tx_min_val'][i] = res_min_tx.values - 273.15
    valid_min_tx = ~np.isnan(res_min_tx.values)
    
    if np.any(valid_min_tx):
        win_years = year_array[mask_all]
        temp_yr_tx = out_ds['tx_min_year'][i].values
        temp_yr_tx[valid_min_tx] = win_years[idx_min_tx.values[valid_min_tx].astype(int)]
        out_ds['tx_min_year'][i] = temp_yr_tx

    # Werte für TN (Wärmerekorde) zuweisen
    out_ds['tn_max_val'][i] = res_max_tn.values - 273.15
    valid_max_tn = ~np.isnan(res_max_tn.values)
    
    if np.any(valid_max_tn):
        win_years = year_array[mask_all]
        temp_yr_tn = out_ds['tn_max_year'][i].values
        temp_yr_tn[valid_max_tn] = win_years[idx_max_tn.values[valid_max_tn].astype(int)]
        out_ds['tn_max_year'][i] = temp_yr_tn

print("\n| -> All-Time Rekorde erfolgreich berechnet.")

print("\nSpeichere finale, erweiterte Klimatologie...")
out_ds.to_netcdf(NEW_CLIM_FILE)
print("Fertig! Die Datei wurde als 'climatology_reference_complete.nc' gespeichert.")