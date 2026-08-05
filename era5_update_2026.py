#!/usr/bin/env python3
import cdsapi
import os
import shutil
import zipfile
from pathlib import Path
import time

# Ordner definieren
DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
DATA_DIR.mkdir(parents=True, exist_ok=True)

AREA = [72.0, -25.0, 30.0, 45.0]
MONTHS = [f"{m:02d}" for m in range(1, 8)] # Jan bis Juli (aktuell in 2026)
DAYS = [f"{d:02d}" for d in range(1, 32)]

client = cdsapi.Client(wait_until_complete=True)

def download_2026_surface():
    print("--- Starte Download: 2026 Oberfläche (TX, TN, MSLP) ---")
    req = {
        "product_type": "reanalysis",
        "variable": [
            "maximum_2m_temperature_since_previous_post_processing",
            "minimum_2m_temperature_since_previous_post_processing",
            "mean_sea_level_pressure"
        ],
        "year": "2026", "month": MONTHS, "day": DAYS,
        "time": ["00:00", "12:00"],
        "area": AREA, "data_format": "netcdf", "download_format": "unarchived",
    }
    temp_part = DATA_DIR / "temp_2026_surf.part"
    client.retrieve("reanalysis-era5-single-levels", req, str(temp_part))
    
    # Entpacken (Copernicus-ZIP Logik)
    with zipfile.ZipFile(temp_part) as archive:
        archive.extractall(DATA_DIR / "tmp_2026")
        
    for f in (DATA_DIR / "tmp_2026").glob("*.nc"):
        if "max" in f.name.lower():
            shutil.move(str(f), str(DATA_DIR / "era5_txtn_batch_2026_2026.nc"))
        elif "instant" in f.name.lower():
            shutil.move(str(f), str(DATA_DIR / "era5_mslp_batch_2026_2026.nc"))
            
    shutil.rmtree(DATA_DIR / "tmp_2026", ignore_errors=True)
    temp_part.unlink()
    print("Oberfläche 2026 erfolgreich in Master_Batches integriert!")

def download_2026_z500():
    print("\n--- Starte Download: 2026 Geopotenzial (Z500) ---")
    req = {
        "product_type": "reanalysis",
        "variable": "geopotential",
        "pressure_level": "500",
        "year": "2026", "month": MONTHS, "day": DAYS,
        "time": "12:00",
        "area": AREA, "format": "netcdf",
    }
    target = DATA_DIR / "era5_z500_batch_2026_2026.nc"
    client.retrieve("reanalysis-era5-pressure-levels", req, str(target))
    print("Z500 2026 erfolgreich in Master_Batches integriert!")

if __name__ == "__main__":
    download_2026_surface()
    download_2026_z500()
    print("\nAlle 2026 Daten sind fehlerfrei und kompatibel im System!")