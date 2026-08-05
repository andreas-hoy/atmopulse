import xarray as xr
import numpy as np
import time
from pathlib import Path

def calculate_persistence_tracks(binary_events_file: str, output_path: str):
    print("=== Starte Persistenz-Tracker (Phase 2) ===")
    start_time = time.time()
    
    # 1. Daten laden
    ds = xr.open_dataset(binary_events_file)
    
    def get_persistence_and_spells(binary_array):
        """Berechnet tägliche Persistenz und WSDI/CSDI Spells ohne groupby (Vektor-Trick)"""
        # --- TRACK A: Tägliche Persistenz (1, 2, 3...) ---
        cumsum = binary_array.cumsum(dim='valid_time')
        reset = cumsum.where(binary_array == 0).ffill(dim='valid_time').fillna(0)
        persistence = (cumsum - reset)
        
        # --- TRACK B: Spell-Filterung (WSDI/CSDI-Standard >= 6 Tage) ---
        is_6 = binary_array.rolling(valid_time=6, min_periods=6).sum() == 6
        
        spell_mask = is_6.copy()
        for i in range(1, 6):
            spell_mask = spell_mask | is_6.shift(valid_time=-i, fill_value=False)
            
        spells = persistence.where(spell_mask, 0)
        
        return persistence, spells

    output_dict = {}
    
    # 2. Schleife über alle detektierten Extrem-Variablen (p80, p90, max, etc.)
    for var_name in ds.data_vars:
        print(f"Analysiere Zeitreihen-Persistenz für {var_name}...")
        
        # Schneidet das "event_" am Anfang ab für saubere Namen (z.B. tx_p90)
        prefix = var_name.replace('event_', '')
        pers, spell = get_persistence_and_spells(ds[var_name])
        
        output_dict[f'{prefix}_daily_persistence'] = pers
        output_dict[f'{prefix}_spell_duration'] = spell

    # 3. Datensatz zusammenbauen
    pers_ds = xr.Dataset(output_dict)
    
    # 4. DATENTYP-OPTIMIERUNG auf uint8 (0-255 Tage)
    print("Optimiere Speicherplatz für Streamlit Cloud (Konvertierung zu uint8)...")
    pers_ds = pers_ds.fillna(0).astype(np.uint8)
    
    # 5. Speichern
    print(f"Speichere Persistenzdaten unter: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pers_ds.to_netcdf(output_path)
    
    print(f"=== FERTIG in {(time.time() - start_time):.2f} Sekunden ===")
    ds.close()
    return pers_ds

if __name__ == "__main__":
    event_test_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Detected_Events/events_2026-01_test.nc"
    out_file = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool/Persistence_Data/persistence_2026-01_test.nc"
    calculate_persistence_tracks(event_test_file, out_file)