import xarray as xr
import numpy as np
import time
from pathlib import Path

def compute_complete_masks(tx_paths, tn_paths, output_path: str, mode: str = "both"):
    print(f"=== Starte Klimamasken-Berechnung (Modus: {mode}) ===")
    start_time = time.time()
    
    # open_mfdataset verarbeitet sowohl einzelne Pfade als auch Listen von Pfaden
    print("Lade Datensätze...")
    ds_tx = xr.open_mfdataset(tx_paths, combine='by_coords')
    ds_tn = xr.open_mfdataset(tn_paths, combine='by_coords')
    
    print("Bereinige Schalttage...")
    is_not_leap = ~((ds_tx['valid_time'].dt.month == 2) & (ds_tx['valid_time'].dt.day == 29))
    ds_tx = ds_tx.sel(valid_time=is_not_leap)
    
    is_not_leap_tn = ~((ds_tn['valid_time'].dt.month == 2) & (ds_tn['valid_time'].dt.day == 29))
    ds_tn = ds_tn.sel(valid_time=is_not_leap_tn)

    print("Konstruiere 5-Tage-Fenster...")
    roll_tx = ds_tx.rolling(valid_time=5, center=True).construct('window_dim')
    roll_tn = ds_tn.rolling(valid_time=5, center=True).construct('window_dim')
    
    grp_tx = roll_tx.groupby('valid_time.dayofyear')
    grp_tn = roll_tn.groupby('valid_time.dayofyear')
    
    data_vars = {}
    
    # P80, P90, P95 (Hitze) und P20, P10, P05 (Kälte) nur für Baselines
    if mode in ["baseline", "both"]:
        print("Berechne Perzentile (WMO-Standard)...")
        q_tx = grp_tx.quantile([0.80, 0.90, 0.95], dim=['valid_time', 'window_dim'])
        q_tn = grp_tn.quantile([0.20, 0.10, 0.05], dim=['valid_time', 'window_dim'])
        
        data_vars['tx_p80'] = q_tx.sel(quantile=0.80).drop_vars('quantile')['mx2t'].astype(np.float32)
        data_vars['tx_p90'] = q_tx.sel(quantile=0.90).drop_vars('quantile')['mx2t'].astype(np.float32)
        data_vars['tx_p95'] = q_tx.sel(quantile=0.95).drop_vars('quantile')['mx2t'].astype(np.float32)
        
        data_vars['tn_p20'] = q_tn.sel(quantile=0.20).drop_vars('quantile')['mn2t'].astype(np.float32)
        data_vars['tn_p10'] = q_tn.sel(quantile=0.10).drop_vars('quantile')['mn2t'].astype(np.float32)
        data_vars['tn_p05'] = q_tn.sel(quantile=0.05).drop_vars('quantile')['mn2t'].astype(np.float32)
        
    # MAX und MIN für den Allzeit-Rekord entkoppeln
    if mode in ["all_time", "both"]:
        print("Berechne absolute Rekorde (MAX/MIN seit 1940) und deren Datum...")
        max_tx = grp_tx.max(dim=['valid_time', 'window_dim'])
        min_tn = grp_tn.min(dim=['valid_time', 'window_dim'])
        
        # Hole das exakte Datum des Allzeitrekords für den jeweiligen Kalendertag
        date_tx = ds_tx['mx2t'].groupby('valid_time.dayofyear').idxmax(dim='valid_time')
        date_tn = ds_tn['mn2t'].groupby('valid_time.dayofyear').idxmin(dim='valid_time')
        
        data_vars['tx_max'] = max_tx['mx2t'].astype(np.float32)
        data_vars['tn_min'] = min_tn['mn2t'].astype(np.float32)
        
        # Datums-Variablen hinzufügen (werden nativ als datetime64 in NetCDF gespeichert)
        data_vars['tx_max_date'] = date_tx
        data_vars['tn_min_date'] = date_tn
        
    print("Baue finalen Datensatz auf...")
    reference_mask = xr.Dataset(data_vars)
    
    print(f"Speichere Maske unter: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    reference_mask.to_netcdf(output_path)
    
    print(f"=== FERTIG in {(time.time() - start_time)/60:.2f} Minuten ===")
    return reference_mask

if __name__ == "__main__":
    tx_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/TX_daily_known_1h_issue/era5_tx_daily_2026-01.nc"
    tn_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/TN_daily_known_1h_issue/era5_tn_daily_2026-01.nc"
    out_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Reference_Masks/klimamasken_januar_test.nc"
    
    compute_complete_masks(tx_file, tn_file, out_file, mode="both")