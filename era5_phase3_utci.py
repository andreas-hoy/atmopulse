#!/usr/bin/env python3
"""
ERA5 Phase 3: UTCI Downloader & Local Aggregator
Autor: Dr. Andreas Hoy / SynEx Pipeline
Architektur: 
- PASS 3: Lädt den Universal Thermal Climate Index (UTCI) herunter (ZIP Workaround inkludiert).
- Resampling auf Daily Max und Daily Min.
- Integration direkt in die era5_master_daily_YYYY.nc Dateien.
- 4 Worker parallel für alle Jahre (1940-2025).
"""

import argparse
import logging
import sys
import shutil
import time
import zipfile
import concurrent.futures
from pathlib import Path
import cdsapi
import xarray as xr

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(0, 24)]

ALL_PERIODS = [y for y in range(2025, 1939, -1)]

def setup_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(root / "phase3_utci.log", encoding="utf-8")
        ]
    )

def safe_retrieve(client, name, request, target_file: Path, max_retries=3):
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
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

def process_pass3_worker(client: cdsapi.Client, year: int, area: list[float], root: Path):
    year_str = str(year)
    master_file = root / f"era5_master_daily_{year_str}.nc"
    
    if not master_file.exists():
        logging.warning("⚠️ SKIP PASS 3 | Masterdatei %s fehlt. Bitte erst PASS 1 durchführen.", year_str)
        return
        
    with xr.open_dataset(master_file) as ds:
        if "utci_max" in ds.data_vars and "utci_min" in ds.data_vars:
            logging.info("⏩ SKIP PASS 3 | UTCI für %s ist bereits im Masterfile integriert.", year_str)
            return

    tmp_dir = root / f".tmp_pass3_utci_{year_str}"
    tmp_dir.mkdir(exist_ok=True)
    start_time = time.monotonic()
    
    try:
        req_utci = {
            "version": "1_1",
            "product_type": "consolidated_dataset",
            "variable": "universal_thermal_climate_index",
            "year": year_str,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "area": area,
            "format": "zip"
        }
        
        logging.info("⏳ QUEUED [%s] | PASS 3: Fordere UTCI an...", year_str)
        utci_zip = tmp_dir / "utci.zip"
        safe_retrieve(client, "derived-utci-historical", req_utci, utci_zip)
        
        logging.info("⚙️ ENTPACKE [%s] | ZIP-Archiv wird extrahiert...", year_str)
        utci_ext_dir = tmp_dir / "utci_extracted"
        utci_ext_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(utci_zip, 'r') as zip_ref:
            zip_ref.extractall(utci_ext_dir)
        
        logging.info("⚙️ UPDATE [%s] | Harmonisiere, berechne UTCI Max/Min und füge hinzu...", year_str)
        temp_master = root / f".temp_master_utci_{year_str}.nc"
        
        with xr.open_mfdataset(str(utci_ext_dir / "*.nc"), combine='by_coords') as ds_utci, \
             xr.open_dataset(master_file) as ds_master:
            
            ds_utci = harmonize_time(ds_utci.load())
            ds_master.load() 
            
            ds_master["utci_max"] = (ds_utci['utci'].resample(time='1D').max() - 273.15).astype('float32')
            ds_master["utci_min"] = (ds_utci['utci'].resample(time='1D').min() - 273.15).astype('float32')
            
            ds_master["utci_max"].attrs["units"] = "Celsius"
            ds_master["utci_min"].attrs["units"] = "Celsius"
            
            encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_master.data_vars}
            ds_master.to_netcdf(temp_master, encoding=encoding)
        
        temp_master.replace(master_file)
        logging.info("🎉 PASS 3 DONE | %s UPDATED (+UTCI_MAX/MIN) in %.1f min.", year_str, (time.monotonic() - start_time) / 60)
        
        shutil.rmtree(tmp_dir, ignore_errors=True)
        
    except Exception as e:
        logging.error("❌ PASS 3 FAILED | %s hart fehlgeschlagen: %s", year_str, str(e))

def execute_pass3(client, root, workers):
    logging.info("=========================================================")
    logging.info("🚀 PASS 3 (UTCI) | %d WORKER PARALLEL (Alle Jahre)", workers)
    logging.info("=========================================================")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_pass3_worker, client, y, DEFAULT_AREA, root): y for y in ALL_PERIODS}
        for future in concurrent.futures.as_completed(futures):
            future.result()

def main():
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)
    
    try:
        logging.info("🛡️ STARTING PASS 3: BIOCLIMATOLOGY (UTCI MAX & MIN)")
        execute_pass3(client, root, workers=4)
        logging.info("🏆 UTCI PIPELINE VOLLSTÄNDIG ABGESCHLOSSEN!")
        
    except KeyboardInterrupt:
        logging.warning("🛑 SKRIPT MANUELL ABGEBROCHEN! (Temp-Ordner bleiben für Resume erhalten)")
        sys.exit(0)

if __name__ == "__main__":
    main()