import xarray as xr
import time
from pathlib import Path
import numpy as np

def detect_daily_anomalies(current_tx_file: str, current_tn_file: str, ref_mask_file: str, output_path: str):
    print("=== Starte Anomalie-Detektor ===")
    start_time = time.time()
    
    # 1. Daten laden
    print("Lade tagesaktuelle Daten und Referenzmasken...")
    ds_tx = xr.open_dataset(current_tx_file)
    ds_tn = xr.open_dataset(current_tn_file)
    ref = xr.open_dataset(ref_mask_file)
    
    # 2. Vergleichsfunktion definieren (nutzt xarray's groupby für dayofyear-Matching)
    def get_binary_mask(current_data, ref_data, condition='greater'):
        # Gruppiert die aktuellen Daten nach Tag-des-Jahres und vergleicht mit der Maske
        if condition == 'greater':
            binary = current_data.groupby('valid_time.dayofyear') > ref_data
        else:
            binary = current_data.groupby('valid_time.dayofyear') < ref_data
        
        # xarray ändert durch groupby manchmal die Koordinatenreihenfolge, wir sortieren das zurück
        # und wandeln True/False in 1/0 als int8 um (spart extrem viel Speicher)
        return binary.drop_vars('dayofyear').transpose('valid_time', 'latitude', 'longitude').astype(np.int8)

    # 3. Binärmasken berechnen (Hitze)
    print("Kalkuliere Hitze-Überschreitungen (TX)...")
    tx_var = ds_tx['mx2t']
    is_tx_p80 = get_binary_mask(tx_var, ref['tx_p80'], 'greater')
    is_tx_p90 = get_binary_mask(tx_var, ref['tx_p90'], 'greater')
    is_tx_p95 = get_binary_mask(tx_var, ref['tx_p95'], 'greater')
    is_tx_max = get_binary_mask(tx_var, ref['tx_max'], 'greater')

    # 4. Binärmasken berechnen (Kälte)
    print("Kalkuliere Kälte-Unterschreitungen (TN)...")
    tn_var = ds_tn['mn2t']
    is_tn_p20 = get_binary_mask(tn_var, ref['tn_p20'], 'less')
    is_tn_p10 = get_binary_mask(tn_var, ref['tn_p10'], 'less')
    is_tn_p05 = get_binary_mask(tn_var, ref['tn_p05'], 'less')
    is_tn_min = get_binary_mask(tn_var, ref['tn_min'], 'less')

    # 5. Zu neuem Datensatz zusammenfügen
    print("Baue Event-Datensatz auf...")
    events_ds = xr.Dataset({
        'event_tx_p80': is_tx_p80,
        'event_tx_p90': is_tx_p90,
        'event_tx_p95': is_tx_p95,
        'event_tx_max': is_tx_max,
        'event_tn_p20': is_tn_p20,
        'event_tn_p10': is_tn_p10,
        'event_tn_p05': is_tn_p05,
        'event_tn_min': is_tn_min
    })
    
    # 6. Speichern
    print(f"Speichere Binärmasken unter: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    events_ds.to_netcdf(output_path)
    
    print(f"=== FERTIG in {(time.time() - start_time):.2f} Sekunden ===")
    return events_ds

if __name__ == "__main__":
    # Pfade anpassen (wir testen mit deinem Januar 2026 gegen die frisch berechnete Januar-Referenz)
    tx_2026 = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/TX_daily_known_1h_issue/era5_tx_daily_2026-01.nc"
    tn_2026 = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/TN_daily_known_1h_issue/era5_tn_daily_2026-01.nc"
    ref_mask = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Reference_Masks/klimamasken_januar_test.nc"
    out_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Detected_Events/events_2026-01_test.nc"
    
    detect_daily_anomalies(tx_2026, tn_2026, ref_mask, out_file)