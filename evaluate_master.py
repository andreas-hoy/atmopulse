#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path

def evaluate_master_file(file_path):
    print(f"==================================================")
    print(f"🔍 EVALUIERUNG: {file_path.name}")
    print(f"==================================================")
    
    if not file_path.exists():
        print(f"❌ FEHLER: Datei nicht gefunden: {file_path}")
        return

    try:
        with xr.open_dataset(file_path) as ds:
            
            # --- 1. CF-CONVENTION CHECK ---
            print("\n1️⃣ DIMENSIONEN & METADATEN")
            dims = list(ds.dims)
            if 'valid_time' in dims or 'valid_time' in ds.coords:
                print("❌ KRITISCHER FEHLER: 'valid_time' ist noch vorhanden!")
            elif 'time' not in dims:
                print("❌ KRITISCHER FEHLER: Dimension 'time' fehlt!")
            else:
                print("✅ CF-Standard 'time' erfolgreich erzwungen.")
            
            # --- 2. VARIABLEN CHECK ---
            print("\n2️⃣ VARIABLEN-INTEGRITÄT")
            vars_in_file = list(ds.data_vars)
            print(f"Enthaltene Variablen: {vars_in_file}")
            
            has_tx_tn = 'tx' in vars_in_file and 'tn' in vars_in_file
            has_tg = 'tg' in vars_in_file
            
            # --- 3. THERMODYNAMIK-CHECK (Rohdaten in Kelvin) ---
            print("\n3️⃣ THERMODYNAMIK-CHECK (Gitterpunkt nahe Pärnu: 58.5°N, 24.5°E)")
            ds_parnu = ds.sel(latitude=58.5, longitude=24.5, method='nearest')
            
            if has_tx_tn:
                tx_raw = ds_parnu['tx'].values
                tn_raw = ds_parnu['tn'].values
                
                print(f"Absolute Jahreswerte (in Kelvin):")
                print(f"  -> Max TX:  {np.nanmax(tx_raw):.2f} K")
                print(f"  -> Min TN:  {np.nanmin(tn_raw):.2f} K")
                
                inconsistent_tx_tn = np.sum(tx_raw < tn_raw)
                if inconsistent_tx_tn > 0:
                    print(f"❌ PHYSIKALISCHER FEHLER: An {inconsistent_tx_tn} Tagen ist TN > TX!")
                else:
                    print("✅ Thermodynamik intakt: TX >= TN durchgehend erfüllt.")
            else:
                print("❌ FEHLER: tx oder tn fehlen für den Check!")

            if has_tg:
                tg_raw = ds_parnu['tg'].values
                print(f"  -> Mean TG: {np.nanmean(tg_raw):.2f} K")
                
                inconsistent_tg = np.sum((tg_raw > tx_raw) | (tg_raw < tn_raw))
                if inconsistent_tg > 0:
                    print(f"⚠️ WARNUNG: An {inconsistent_tg} Tagen liegt TG außerhalb von TX/TN!")
                else:
                    print("✅ Mittelwert-Logik intakt: TN <= TG <= TX durchgehend erfüllt.")
            else:
                print("ℹ️ HINWEIS: 'tg' ist noch nicht in der Datei (Pass 2 steht noch aus).")

    except Exception as e:
        print(f"Kritischer Fehler bei der Auswertung: {e}")

if __name__ == "__main__":
    file_path = Path("ERA5_ClimateTool/Master_Batches/era5_master_daily_2025.nc")
    evaluate_master_file(file_path)