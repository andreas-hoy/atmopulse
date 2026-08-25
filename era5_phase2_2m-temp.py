#!/usr/bin/env python3
"""
ERA5 Phase 2: Two-Pass Decoupled Raw Data Downloader & Local Aggregator (V4 - Final & Robust)
Autor: Dr. Andreas Hoy / Atmopulse Pipeline
Architektur: 
- PASS 1 (TX/TN) & PASS 2 (TG) in strikter Entkopplung (Micro-Batching).
- Resume-Fähigkeit: Erkennt bereits teilweise geladene Temp-Dateien und spart Download-Zeit.
- Time-Harmonisierung: Zwingt alle Dateien (Synop & ERA5) on-the-fly auf den Standard 'time'.
- 2025-1960: 4 Worker parallel (Fast Disk)
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
ALL_HOURS = [f"{h:02d}:00" for h in range(0, 24)]

PERIODS_FAST = [y for y in range(2025, 1959, -1)]
PERIODS_TAPE = [y for y in range(1959, 1939, -1)]

def setup_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(root / "phase2_twopass_final.log", encoding="utf-8")
        ]
    )

def safe_retrieve(client, name, request, target_file: Path, max_retries=3):
    """Sicherer Download mit Resume-Fähigkeit für bereits geladene Dateien."""
    # Check ob Datei schon da ist (größer als 1MB als Sicherheitscheck für unvollständige Reste)
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
    # Falls valid_time noch als Koordinate übrig geblieben ist, ebenfalls umbenennen
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

# ==============================================================================
# PASS 1: TX & TN
# ==============================================================================
def process_pass1_worker(client: cdsapi.Client, year: int, area: list[float], root: Path):
    year_str = str(year)
    out_file = root / f"era5_master_daily_{year_str}.nc"
    synop_file = root / "Synoptics_Cache" / f"synop_12utc_{year_str}.nc"
    
    if out_file.exists():
        logging.info("⏩ SKIP PASS 1 | Masterdatei für %s existiert bereits.", year_str)
        return
        
    if not synop_file.exists():
        logging.error("❌ FEHLER | Synoptik-Datei %s fehlt im Cache!", year_str)
        return

    tmp_dir = root / f".tmp_pass1_{year_str}"
    tmp_dir.mkdir(exist_ok=True)
    start_time = time.monotonic()
    
    try:
        req_base = {
            "product_type": "reanalysis", "year": year_str, "month": ALL_MONTHS, 
            "day": ALL_DAYS, "time": ALL_HOURS, "area": area, "format": "netcdf"
        }
        
        # 1. Download mx2t (mit Resume-Check)
        req_mx2t = req_base.copy()
        req_mx2t["variable"] = "maximum_2m_temperature_since_previous_post_processing"
        logging.info("⏳ QUEUED [%s] | PASS 1: Fordere mx2t an...", year_str)
        safe_retrieve(client, "reanalysis-era5-single-levels", req_mx2t, tmp_dir / "mx2t.nc")

        # 2. Download mn2t (mit Resume-Check)
        req_mn2t = req_base.copy()
        req_mn2t["variable"] = "minimum_2m_temperature_since_previous_post_processing"
        logging.info("⏳ QUEUED [%s] | PASS 1: Fordere mn2t an...", year_str)
        safe_retrieve(client, "reanalysis-era5-single-levels", req_mn2t, tmp_dir / "mn2t.nc")

        logging.info("⚙️ AGGREGATING [%s] | Harmonisiere Zeiten, berechne TX/TN und merge...", year_str)
        
        # 3. Lokale Aggregation mit Time-Harmonisierung
        with xr.open_dataset(tmp_dir / "mx2t.nc") as ds_mx2t, \
             xr.open_dataset(tmp_dir / "mn2t.nc") as ds_mn2t, \
             xr.open_dataset(synop_file) as ds_synop:
            
            # Xarray in den RAM laden und Dimensionen auf 'time' zwingen
            ds_mx2t = harmonize_time(ds_mx2t.load())
            ds_mn2t = harmonize_time(ds_mn2t.load())
            ds_synop = harmonize_time(ds_synop.load())
            
            tx = ds_mx2t['mx2t'].resample(time='1D').max()
            tn = ds_mn2t['mn2t'].resample(time='1D').min()
            
            ds_out = xr.Dataset({"tx": tx, "tn": tn})
            
            # Synoptik dazu mergen
            for var in ["mslp", "z500", "t850", "u300", "v300"]:
                ds_out[var] = ds_synop[var]

            encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
            ds_out.to_netcdf(out_file, encoding=encoding)
            
        logging.info("🎉 PASS 1 DONE | %s (TX & TN) gesichert in %.1f min.", year_str, (time.monotonic() - start_time) / 60)
        
        # Physischer Hard-Delete erst nach erfolgreichem Schreiben des Masterfiles!
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        logging.error("❌ PASS 1 FAILED | %s hart fehlgeschlagen: %s", year_str, str(e))
        # Im Fehlerfall löschen wir das Temp-Verzeichnis NICHT, damit die Resume-Funktion 
        # beim nächsten Start die bisherigen Downloads wiederverwenden kann.

# ==============================================================================
# PASS 2: TG 
# ==============================================================================
def process_pass2_worker(client: cdsapi.Client, year: int, area: list[float], root: Path):
    year_str = str(year)
    master_file = root / f"era5_master_daily_{year_str}.nc"
    
    if not master_file.exists():
        logging.warning("⚠️ SKIP PASS 2 | Masterdatei %s fehlt. Überspringe TG.", year_str)
        return
        
    with xr.open_dataset(master_file) as ds:
        if "tg" in ds.data_vars:
            logging.info("⏩ SKIP PASS 2 | TG für %s ist bereits integriert.", year_str)
            return

    tmp_dir = root / f".tmp_pass2_{year_str}"
    tmp_dir.mkdir(exist_ok=True)
    start_time = time.monotonic()
    
    try:
        req_t2m = {
            "product_type": "reanalysis", "variable": "2m_temperature",
            "year": year_str, "month": ALL_MONTHS, "day": ALL_DAYS,
            "time": ALL_HOURS, "area": area, "format": "netcdf"
        }
        
        logging.info("⏳ QUEUED [%s] | PASS 2: Fordere t2m an...", year_str)
        safe_retrieve(client, "reanalysis-era5-single-levels", req_t2m, tmp_dir / "t2m.nc")
        
        logging.info("⚙️ UPDATE [%s] | Harmonisiere, berechne TG und füge es hinzu...", year_str)
        temp_master = root / f".temp_master_{year_str}.nc"
        
        with xr.open_dataset(tmp_dir / "t2m.nc") as ds_t2m, xr.open_dataset(master_file) as ds_master:
            ds_t2m = harmonize_time(ds_t2m.load())
            ds_master.load() 
            
            ds_master["tg"] = ds_t2m['t2m'].resample(time='1D').mean()
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds_master.data_vars}
            ds_master.to_netcdf(temp_master, encoding=encoding)
        
        temp_master.replace(master_file)
        logging.info("🎉 PASS 2 DONE | %s UPDATED (+TG) in %.1f min.", year_str, (time.monotonic() - start_time) / 60)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        logging.error("❌ PASS 2 FAILED | %s hart fehlgeschlagen: %s", year_str, str(e))

# ==============================================================================
# PIPELINE KOORDINATOR
# ==============================================================================
def execute_pass(client, root, pass_func, pass_name, periods, workers):
    logging.info("=========================================================")
    logging.info("🚀 %s | %d WORKER", pass_name, workers)
    logging.info("=========================================================")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(pass_func, client, y, DEFAULT_AREA, root): y for y in periods}
        for future in concurrent.futures.as_completed(futures):
            future.result()

def main():
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)
    
    try:
        logging.info("🛡️ STARTING PASS 1: KRITISCHE DATEN (TX & TN)")
        execute_pass(client, root, process_pass1_worker, "PASS 1 | FAST DISK", PERIODS_FAST, 4)
        execute_pass(client, root, process_pass1_worker, "PASS 1 | TAPE ARCHIVE", PERIODS_TAPE, 2)
        logging.info("✅ PASS 1 VOLLSTÄNDIG BEENDET.")

        logging.info("🎯 STARTING PASS 2: OPTIONALE DATEN (TG)")
        execute_pass(client, root, process_pass2_worker, "PASS 2 | FAST DISK", PERIODS_FAST, 4)
        execute_pass(client, root, process_pass2_worker, "PASS 2 | TAPE ARCHIVE", PERIODS_TAPE, 2)
        
        logging.info("🏆 PIPELINE VOLLSTÄNDIG ABGESCHLOSSEN!")
        
    except KeyboardInterrupt:
        logging.warning("🛑 SKRIPT MANUELL ABGEBROCHEN! (Temp-Ordner bleiben für Resume erhalten)")
        sys.exit(0)

if __name__ == "__main__":
    main()