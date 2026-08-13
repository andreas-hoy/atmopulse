#!/usr/bin/env python3
"""
Master-File Kelvin to Celsius Patcher
Autor: Dr. Andreas Hoy / SynEx Pipeline
Funktion: Iteriert über alle era5_master_daily_YYYY.nc Dateien.
Prüft für thermische Variablen (tx, tn, tg, t850), ob sie noch in Kelvin 
vorliegen (Mittelwert > 200). Falls ja, wird 273.15 abgezogen.
Sicherer Write-Prozess über temporäre Datei.
"""

import logging
import sys
from pathlib import Path
import xarray as xr
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def patch_kelvin_in_file(file_path: Path):
    temp_path = file_path.with_name(f".temp_patch_{file_path.name}")
    vars_to_check = ["tx", "tn", "tg", "t850", "utci_max", "utci_min"]
    patched_any = False
    
    try:
        with xr.open_dataset(file_path) as ds:
            ds.load()  # In den RAM laden für schnelle Bearbeitung
            
            for var in vars_to_check:
                if var in ds.data_vars:
                    # Sanity Check: Ist der Wert noch in Kelvin? 
                    # (Mittelwert über 200 K ist auf der Erde eindeutig)
                    mean_val = float(ds[var].mean().values)
                    
                    if mean_val > 200:
                        logging.info(f"  -> {var} ist in Kelvin (Mean: {mean_val:.1f} K). Patche zu Celsius...")
                        ds[var] = ds[var] - 273.15
                        patched_any = True
                    else:
                        logging.info(f"  -> {var} ist bereits in Celsius (Mean: {mean_val:.1f} °C). Skip.")
            
            if patched_any:
                logging.info(f"💾 Speichere gepatchte Datei: {temp_path.name}")
                encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
                ds.to_netcdf(temp_path, encoding=encoding)
                
    except Exception as e:
        logging.error(f"❌ Fehler bei Datei {file_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return

    # Wenn erfolgreich gepatcht wurde, überschreibe das Original
    if patched_any and temp_path.exists():
        temp_path.replace(file_path)
        logging.info(f"✅ {file_path.name} erfolgreich überschrieben.")
    else:
        logging.info(f"⏩ Keine Kelvin-Werte in {file_path.name} gefunden. Überspringe Speichern.")

def main():
    setup_logging()
    master_dir = Path("ERA5_ClimateTool/Master_Batches").resolve()
    
    if not master_dir.exists():
        logging.error(f"Verzeichnis {master_dir} nicht gefunden!")
        return

    master_files = sorted(list(master_dir.glob("era5_master_daily_*.nc")))
    
    logging.info(f"🛡️ START: Überprüfe {len(master_files)} Master-Dateien auf Kelvin-Werte...")
    
    for file_path in master_files:
        logging.info(f"🔍 Prüfe {file_path.name}...")
        patch_kelvin_in_file(file_path)
        
    logging.info("🏆 KELVIN-PATCHER VOLLSTÄNDIG ABGESCHLOSSEN!")

if __name__ == "__main__":
    main()