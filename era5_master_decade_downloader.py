#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cdsapi

SINGLE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]

# Statische Zeit-Strukturen für den Riesen-Request
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERA5 Master Downloader: 10-Year Parameter Bundling (1940-2025)"
    )
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
            logging.FileHandler(root / "master_batch_download.log", encoding="utf-8"),
        ],
    )


def retrieve_batch(client: cdsapi.Client, dataset: str, request: dict[str, Any], target: Path, overwrite: bool) -> bool:
    if target.exists() and target.stat().st_size > 1024**2 and not overwrite:
        logging.info("SKIP | %s bereits vorhanden.", target.name)
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)

    logging.info("START | Beantrage Batch: %s", target.name)
    started = time.monotonic()

    try:
        client.retrieve(dataset, request, str(part))
        os.replace(part, target)
        logging.info(
            "DONE | %s | %.1f MiB | %.1f min",
            target.name,
            target.stat().st_size / 1024**2,
            (time.monotonic() - started) / 60,
        )
        return True
    except Exception as exc:
        part.unlink(missing_ok=True)
        logging.exception("FAILED | %s", target.name)
        return False


def main() -> int:
    args = parse_args()
    root = args.output.resolve()
    setup_logging(root)

    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)

    # Wir definieren die 10-Jahres-Dekaden rückwärts
    decades = [
        (2020, 2025),
        (2010, 2019),
        (2000, 2009),
        (1990, 1999),
        (1980, 1989),
        (1970, 1979),
        (1960, 1969),
        (1950, 1959),
        (1940, 1949),
    ]

    logging.info("Starte gebündelten Dekaden-Download (1940-2025) für Pan-Europa")

    for start_year, end_year in decades:
        years_list = [str(y) for y in range(start_year, end_year + 1)]
        logging.info("DEKADE | Verarbeite Block %d bis %d", start_year, end_year)

        # JOB 1: BÜNDELUNG DER SURFACE-LEVELS (TX, TN, MSLP in einem Rutsch!)
        surface_req = {
            "product_type": ["reanalysis"],
            # Alle drei Variablen parallel im selben Stream!
            "variable": [
                "maximum_2m_temperature_since_previous_post_processing",
                "minimum_2m_temperature_since_previous_post_processing",
                "mean_sea_level_pressure"
            ],
            "year": years_list,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ["00:00", "12:00"], # Einheitlich für synoptischen Vergleich um 00/12 UTC
            "area": args.area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        surface_target = root / f"era5_surface_extreme_batch_{start_year}_{end_year}.nc"
        retrieve_batch(client, SINGLE_DATASET, surface_req, surface_target, args.overwrite)

        # JOB 2: HÖHEN-GEOPOTENZIAL (Z500)
        z500_req = {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "pressure_level": ["500"],
            "year": years_list,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ["00:00", "12:00"],
            "area": args.area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        z500_target = root / f"era5_z500_batch_{start_year}_{end_year}.nc"
        retrieve_batch(client, PRESSURE_DATASET, z500_req, z500_target, args.overwrite)

    logging.info("Master-Download beendet. Alle Dekaden-Pakete wurden verarbeitet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())