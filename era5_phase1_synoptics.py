#!/usr/bin/env python3
"""
ERA5 Phase 1: Synoptics Downloader (V4 - Async 4-Worker Architecture)
Autor: Dr. Andreas Hoy / SynEx Pipeline
Architektur: 
- PASS 1: Lädt die synoptischen Treiber (12 UTC) herunter.
- Beinhaltet: MSLP (Single Levels) sowie Z500, T850, U300, V300 (Pressure Levels).
- Lokales Merging und saubere Selektion der Druckflächen.
- 4 Worker parallel für alle Jahre (2025-1940) zur maximalen Sättigung der CDS Queue.
"""

import argparse
import logging
import sys
import shutil
import time
import concurrent.futures
from pathlib import Path
import cdsapi
import xarray as xr

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# Synoptischer Standard: 12:00 UTC repräsentiert das Zirkulationsmuster des Tages
SYNOP_TIME = ["12:00"] 

# Alle Jahre in einer Liste, da wir mit 4 Workern durchgehend arbeiten
ALL_PERIODS = [y for y in range(2025, 1939, -1)]

def setup_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(root / "phase1_synoptics.log", encoding="utf-8")
        ]
    )

def safe_retrieve(client, name, request, target_file: Path, max_retries=3):
    """Sicherer Download mit Resume-Fähigkeit für bereits geladene Dateien."""
    if target_file.exists() and target_file.stat().st_size > 1_000_000:
        logging.info("⏩ RESUME | %s existiert bereits. Überspringe Download.", target_file.name)
        return True

    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(name, request, str(target_file))
            return True
        except Exception as e:
            logging.warning("⚠️ Versuch %d/%d fehlgeschlagen: %s", attempt, max_retries, e)
            if attempt == max_retries:
                logging.error("❌ Finaler Abbruch für diesen Request.")
                raise
            time.sleep(15)

def harmonize_time(ds: xr.Dataset) -> xr.Dataset:
    """Standardisiert die Zeitdimension von 'valid_time' auf 'time' (CF-Convention)."""
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

# ==============================================================================
# PASS 1: SYNOPTICS (12 UTC)
# ==============================================================================
def process_synoptics_worker(client: cdsapi.Client, year: int, area: list[float], root: Path):
    year_str = str(year)
    synop_dir = root / "Synoptics_Cache"
    synop_dir.mkdir(parents=True, exist_ok=True)
    out_file = synop_dir / f"synop_12utc_{year_str}.nc"
    
    # Check ob finales Synoptik-File bereits existiert
    if out_file.exists() and out_file.stat().st_size > 1_000_000:
        logging.info("⏩ SKIP SYNOP | %s existiert bereits im Cache.", out_file.name)
        return

    tmp_dir = synop_dir / f".tmp_synop_{year_str}"
    tmp_dir.mkdir(exist_ok=True)
    start_time = time.monotonic()
    
    tmp_mslp = tmp_dir / f"mslp_{year_str}.nc"
    tmp_press = tmp_dir / f"press_{year_str}.nc"

    logging.info("⏳ QUEUED [%s] | SYNOP: Fordere MSLP & Upper-Air an...", year_str)
    
    try:
        # 1. MSLP (Single Levels)
        req_mslp = {
            "product_type": "reanalysis", 
            "variable": ["mean_sea_level_pressure"], 
            "year": [year_str], 
            "month": ALL_MONTHS, 
            "day": ALL_DAYS, 
            "time": SYNOP_TIME, 
            "area": area, 
            "format": "netcdf"
        }
        safe_retrieve(client, "reanalysis-era5-single-levels", req_mslp, tmp_mslp)
        
        # 2. Upper-Air (Pressure Levels: Z500, T850, U300, V300)
        req_press = {
            "product_type": "reanalysis", 
            "variable": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"], 
            "pressure_level": ["300", "500", "850"], 
            "year": [year_str], 
            "month": ALL_MONTHS, 
            "day": ALL_DAYS, 
            "time": SYNOP_TIME, 
            "area": area, 
            "format": "netcdf"
        }
        safe_retrieve(client, "reanalysis-era5-pressure-levels", req_press, tmp_press)

        logging.info("⚙️ AGGREGATING [%s] | Extrahiere Druckflächen und merge Synoptik...", year_str)
        
        # 3. Lokale Verarbeitung und Merging
        with xr.open_dataset(tmp_mslp) as ds_mslp, xr.open_dataset(tmp_press) as ds_press:
            ds_mslp = harmonize_time(ds_mslp.load())
            ds_press = harmonize_time(ds_press.load())
            
            # Initialisiere Output-Dataset mit MSLP
            ds_out = xr.Dataset({"mslp": ds_mslp["msl"]})
            
            # Gezielte Selektion der Parameter auf den spezifischen Druckflächen inkl. Drop der Z-Koordinate
            ds_out["z500"] = ds_press["z"].sel(pressure_level=500, drop=True)
            ds_out["t850"] = ds_press["t"].sel(pressure_level=850, drop=True) - 273.15
            ds_out["u300"] = ds_press["u"].sel(pressure_level=300, drop=True)
            ds_out["v300"] = ds_press["v"].sel(pressure_level=300, drop=True)
            
            # Falls pressure_level als verwaiste Koordinate übrig bleibt, bereinigen
            if "pressure_level" in ds_out.coords:
                ds_out = ds_out.drop_vars("pressure_level")

            # Encoding für Speicheroptimierung
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
            ds_out.to_netcdf(out_file, encoding=encoding)
            
        logging.info("🎉 SYNOP DONE | %s erfolgreich gesichert in %.1f min.", year_str, (time.monotonic() - start_time) / 60)
        
        # Cleanup des temporären Ordners
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        logging.error("❌ SYNOP FAILED | %s hart fehlgeschlagen: %s", year_str, str(e))

# ==============================================================================
# PIPELINE KOORDINATOR
# ==============================================================================
def execute_synoptics(client, root, workers):
    logging.info("=========================================================")
    logging.info("🚀 SYNOPTICS (12 UTC) | %d WORKER PARALLEL (Alle Jahre)", workers)
    logging.info("=========================================================")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_synoptics_worker, client, y, DEFAULT_AREA, root): y for y in ALL_PERIODS}
        for future in concurrent.futures.as_completed(futures):
            future.result()

def main():
    root = Path("ERA5_ClimateTool").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)
    
    try:
        logging.info("🛡️ STARTING PHASE 1: MACRO-CIRCULATION DRIVERS (MSLP, Z500, T850, W300)")
        
        # 4 Worker parallel, exakt wie in Phase 3
        execute_synoptics(client, root, workers=4)
        
        logging.info("🏆 SYNOPTIK PIPELINE VOLLSTÄNDIG ABGESCHLOSSEN!")
        
    except KeyboardInterrupt:
        logging.warning("🛑 SKRIPT MANUELL ABGEBROCHEN! (Temp-Ordner bleiben für Resume erhalten)")
        sys.exit(0)

if __name__ == "__main__":
    main()