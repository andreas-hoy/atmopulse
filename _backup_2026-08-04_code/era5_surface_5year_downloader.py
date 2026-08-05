#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cdsapi

SINGLE_DATASET = "reanalysis-era5-single-levels"
DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERA5 Surface Downloader: 5-Year Blocks with Auto-Unzip")
    p.add_argument("--output", type=Path, default=Path("ERA5_ClimateTool/Master_Batches"))
    p.add_argument("--area", nargs=4, type=float, default=DEFAULT_AREA)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(root / "surface_5year_download.log", encoding="utf-8"),
        ],
    )


def handle_hidden_zip(zip_path: Path, target_dir: Path, prefix: str) -> None:
    """Entpackt die versteckte ZIP-Datei und benennt die gesplitteten NC-Dateien sauber um."""
    temp_extract = target_dir / f".tmp_extract_{prefix}"
    shutil.rmtree(temp_extract, ignore_errors=True)
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_extract)
        
        extracted_files = list(temp_extract.glob("*.nc"))
        for f in extracted_files:
            if "max" in f.name.lower():
                shutil.move(str(f), str(target_dir / f"era5_txtn_batch_{prefix}.nc"))
            elif "instant" in f.name.lower():
                shutil.move(str(f), str(target_dir / f"era5_mslp_batch_{prefix}.nc"))
            else:
                shutil.move(str(f), str(target_dir / f"era5_extra_{f.name}_{prefix}.nc"))
                
        logging.info("ZIP-Kompression erkannt: Erfolgreich entpackt und aufgeteilt.")
    finally:
        shutil.rmtree(temp_extract, ignore_errors=True)
        zip_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.output.resolve()
    setup_logging(root)

    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)

    # 5-Jahres-Intervalle von 2019 rückwärts bis 1940
    intervals = [
        (2015, 2019), (2010, 2015),
        (2005, 2009), (2000, 2004),
        (1995, 1999), (1990, 1994),
        (1985, 1989), (1980, 1984),
        (1975, 1979), (1970, 1974),
        (1965, 1969), (1960, 1964),
        (1955, 1959), (1950, 1954),
        (1945, 1949), (1940, 1944)
    ]

    for start_year, end_year in intervals:
        prefix = f"{start_year}_{end_year}"
        txtn_target = root / f"era5_txtn_batch_{prefix}.nc"
        mslp_target = root / f"era5_mslp_batch_{prefix}.nc"
        
        if txtn_target.exists() and mslp_target.exists() and not args.overwrite:
            logging.info("SKIP | Block %s bereits komplett verarbeitet.", prefix)
            continue

        logging.info("START | Beantrage 5-Jahres-Oberflächen-Block: %s", prefix)
        years_list = [str(y) for y in range(start_year, end_year + 1)]
        
        req = {
            "product_type": ["reanalysis"],
            "variable": [
                "maximum_2m_temperature_since_previous_post_processing",
                "minimum_2m_temperature_since_previous_post_processing",
                "mean_sea_level_pressure"
            ],
            "year": years_list,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ["00:00", "12:00"],
            "area": args.area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        temp_download = root / f"temp_surface_{prefix}.part"
        temp_download.unlink(missing_ok=True)
        
        started = time.monotonic()
        try:
            client.retrieve(SINGLE_DATASET, req, str(temp_download))
            
            # Prüfen, ob es ein ZIP ist (Copernicus-Eigenart bei Bundles)
            with open(temp_download, "rb") as f:
                magic = f.read(4)
            
            if magic.startswith(b"PK\x03\x04"):
                handle_hidden_zip(temp_download, root, prefix)
            else:
                # Falls es doch mal direkt eine NC sein sollte (unwahrscheinlich bei dem Variablenmix)
                os.replace(temp_download, txtn_target)
                logging.warning("Direkte NC-Ausgabe erfolgt. Datei wurde als TXTN abgelegt.")
                
            logging.info("DONE | Block %s verarbeitet in %.1f min", prefix, (time.monotonic() - started) / 60)
            
        except Exception as exc:
            temp_download.unlink(missing_ok=True)
            logging.exception("FAILED | Block %s abgebrochen.", prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())