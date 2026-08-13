#!/usr/bin/env python3
"""
ERA5 Master Batch Repair & Optimization Tool
Zweck: 
1. Behebt den 12 UTC vs. 00 UTC Zeitachsen-Bug beim Mergen.
2. Konvertiert Temperaturvariablen (Kelvin -> Celsius).
3. Castet auf float32 für Speichereffizienz.
4. Führt am Ende einen Plausibilitäts-Check für Gröditz (Lat 51.5, Lon 13.5) durch.
"""

import xarray as xr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
master_dir = Path("ERA5_ClimateTool/Master_Batches")
synop_dir = master_dir / "Synoptics_Cache" 

# NUR ABGESCHLOSSENE Jahre eintragen! Keine Jahre, die der Phase2-Loader gerade bearbeitet.
years_to_repair = range(2010, 2026) 

vars_to_replace = ["mslp", "z500", "t850", "u300", "v300"]
temperature_vars = ["tx", "tn", "tg", "t850"]

# ==============================================================================
# REPAIR PIPELINE
# ==============================================================================
for year in years_to_repair:
    master_file = master_dir / f"era5_master_daily_{year}.nc"
    synop_file = synop_dir / f"synop_12utc_{year}.nc"
    temp_file = master_dir / f".temp_repair_{year}.nc"
    
    if not master_file.exists() or not synop_file.exists():
        logging.warning(f"[{year}] Übersprungen: Master- oder Synop-Datei fehlt.")
        continue

    try:
        logging.info(f"[{year}] Starte Reparatur, Celsius-Umrechnung & Float32-Optimierung...")
        
        with xr.open_dataset(master_file) as ds_master_disk, \
             xr.open_dataset(synop_file) as ds_synop_disk:
             
            ds_master = ds_master_disk.load()
            ds_synop = ds_synop_disk.load()
            
            # 1. Korrupte Variablen (mit NaNs) entfernen
            ds_master = ds_master.drop_vars(vars_to_replace, errors="ignore")
            
            # 2. Zeitachse hart überschreiben: 12 UTC wird zu 00 UTC
            ds_synop['time'] = ds_master['time']
            
            # 3. Saubere Höhenvariablen injizieren
            for var in vars_to_replace:
                ds_master[var] = ds_synop[var]
                
            # 4. Kelvin zu Celsius Umrechnung (Sicherheitscheck: nur wenn Mittelwert > 200)
            for temp_var in temperature_vars:
                if temp_var in ds_master and ds_master[temp_var].mean() > 200:
                    ds_master[temp_var] = ds_master[temp_var] - 273.15
                    ds_master[temp_var].attrs["units"] = "Celsius"
                    logging.info(f"    -> {temp_var} erfolgreich in Celsius konvertiert.")

            # 5. Datentyp für Effizienz auf float32 reduzieren
            ds_master = ds_master.astype('float32')
                
            # 6. Speichern mit zlib-Kompression
            encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_master.data_vars}
            # Die Zeitachse darf nicht als float32 komprimiert werden
            if 'time' in encoding:
                del encoding['time']
                
            ds_master.to_netcdf(temp_file, encoding=encoding)
            
        # Atomares Ersetzen der Originaldatei
        temp_file.replace(master_file)
        logging.info(f"[{year}] ✅ Reparatur abgeschlossen!")
        
    except Exception as e:
        logging.error(f"[{year}] ❌ Fehler bei der Reparatur: {e}")

# ==============================================================================
# PLAUSIBILITÄTS-CHECK
# ==============================================================================
print("\n" + "="*50)
print("🔍 PLAUSIBILITÄTS-CHECK (Grid-Punkt: 51.5°N, 13.5°E - Gröditz)")
print("="*50)

check_years = [2025, 2015, 2010] # Jahre, die repariert wurden
check_days = ['01-01', '04-01', '07-01', '10-01']
lat_check = 51.5
lon_check = 13.5

for y in check_years:
    file_path = master_dir / f"era5_master_daily_{y}.nc"
    if not file_path.exists():
        continue
        
    print(f"\n--- JAHR {y} ---")
    try:
        with xr.open_dataset(file_path) as ds:
            ds_point = ds.sel(latitude=lat_check, longitude=lon_check, method="nearest")
            
            for day in check_days:
                date_str = f"{y}-{day}"
                try:
                    val = ds_point.sel(time=date_str)
                    
                    tx = val['tx'].item()
                    tn = val['tn'].item()
                    tg = val['tg'].item()
                    t850 = val['t850'].item()
                    z500 = val['z500'].item()
                    
                    print(f"📅 {date_str}: Tx={tx:5.1f}°C | Tn={tn:5.1f}°C | Tg={tg:5.1f}°C || T850={t850:5.1f}°C | Z500={z500:5.0f}m")
                except KeyError:
                    print(f"📅 {date_str}: Datum nicht gefunden.")
    except Exception as e:
        print(f"❌ Fehler beim Plausibilitäts-Check: {e}")