import xarray as xr
import pandas as pd
import time
from pathlib import Path

def stitch_era5_and_ifs(era5_file: str, ifs_file: str, output_file: str, var_name_era5: str, var_name_ifs: str):
    print(f"Verknüpfe Reanalyse und Vorhersage für Variable: {var_name_era5}...")
    
    # 1. Datensätze öffnen
    ds_era5 = xr.open_dataset(era5_file)
    ds_ifs = xr.open_dataset(ifs_file)
    
    # 2. Variablen isolieren und harmonisieren
    # IFS nutzt oft den Namen '2t', ERA5 nutzt 'mx2t'/'mn2t'. Wir passen das an.
    da_era5 = ds_era5[var_name_era5]
    da_ifs = ds_ifs[var_name_ifs].rename(var_name_era5)
    
    # 3. Zeitliche Überschneidungen bereinigen
    # Falls IFS-Datenhistorie Tage enthält, die ERA5 schon abdeckt, 
    # ist ERA5 als Goldstandard zu bevorzugen.
    last_era5_time = da_era5['valid_time'].max().values
    da_ifs_filtered = da_ifs.sel(valid_time=slice(last_era5_time + pd.Timedelta(days=1), None))
    
    # 4. Nahtlose Kombination entlang der Zeitachse
    combined_series = xr.concat([da_era5, da_ifs_filtered], dim='valid_time')
    
    # Sortieren, um absolute Chronologie zu garantieren
    combined_series = combined_series.sortby('valid_time')
    
    # In ein Dataset umwandeln und speichern
    ds_combined = combined_series.to_dataset(name=var_name_era5)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ds_combined.to_netcdf(output_file)
    print(f"Nahtlose Zeitreihe erfolgreich gespeichert: {output_file}")
    
    ds_era5.close()
    ds_ifs.close()

if __name__ == "__main__":
    # Test-Pfade für unseren Januar-Dummy
    base = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool"
    
    era5_tx = f"{base}/TX_daily_known_1h_issue/era5_tx_daily_2026-01.nc"
    ifs_tx = f"{base}/IFS_Forecast/ifs_tx_forecast_regridded.nc"
    out_tx = f"{base}/Stitched_Data/combined_tx_2026-01.nc"
    
    # Testlauf für TX
    stitch_era5_and_ifs(era5_tx, ifs_tx, out_tx, 'mx2t', 't2m')