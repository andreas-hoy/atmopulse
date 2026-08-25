#!/usr/bin/env python3
"""ERA5 Phase 3: UTCI download, daily max/min aggregation, master-file merge.

Retrieves the Copernicus derived-UTCI historical product (ZIP workaround),
resamples to daily maximum and minimum (Kelvin→Celsius), and writes
``utci_max`` / ``utci_min`` into each existing ``era5_master_daily_YYYY.nc``.
Four parallel workers cover 1940–2025; completed years are skipped.
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{h:02d}:00" for h in range(0, 24)]

ALL_PERIODS = [y for y in range(2025, 1939, -1)]


def setup_logging(root: Path) -> None:
    """Configure INFO logging to stdout and ``phase3_utci.log``."""
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(root / "phase3_utci.log", encoding="utf-8"),
        ],
    )


def safe_retrieve(client, name, request, target_file: Path, max_retries=3):
    """Retrieve a CDS request, skipping if a large local file already exists."""
    if target_file.exists() and target_file.stat().st_size > 1_000_000:
        logging.info(
            "RESUME | %s already exists. Skipping download.", target_file.name
        )
        return True

    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(name, request, str(target_file))
            return True
        except Exception as exc:
            logging.warning(
                "Attempt %d/%d failed: %s", attempt, max_retries, exc
            )
            if attempt == max_retries:
                logging.error("Final abort for this request.")
                raise
            time.sleep(15)


def harmonize_time(ds: xr.Dataset) -> xr.Dataset:
    """Rename ``valid_time`` dimension/coordinate to CF ``time``."""
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def process_pass3_worker(
    client: cdsapi.Client, year: int, area: list[float], root: Path
) -> None:
    """Download, aggregate, and merge UTCI for a single year."""
    year_str = str(year)
    master_file = root / f"era5_master_daily_{year_str}.nc"

    if not master_file.exists():
        logging.warning(
            "SKIP PASS 3 | master file %s missing. Run Pass 1 first.",
            year_str,
        )
        return

    with xr.open_dataset(master_file) as ds:
        if "utci_max" in ds.data_vars and "utci_min" in ds.data_vars:
            logging.info(
                "SKIP PASS 3 | UTCI for %s is already in the master file.",
                year_str,
            )
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
            "format": "zip",
        }

        logging.info("QUEUED [%s] | PASS 3: requesting UTCI...", year_str)
        utci_zip = tmp_dir / "utci.zip"
        safe_retrieve(client, "derived-utci-historical", req_utci, utci_zip)

        logging.info("EXTRACT [%s] | unpacking ZIP archive...", year_str)
        utci_ext_dir = tmp_dir / "utci_extracted"
        utci_ext_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(utci_zip, "r") as zip_ref:
            zip_ref.extractall(utci_ext_dir)

        logging.info(
            "UPDATE [%s] | harmonise, compute UTCI max/min, merge...",
            year_str,
        )
        temp_master = root / f".temp_master_utci_{year_str}.nc"

        with xr.open_mfdataset(
            str(utci_ext_dir / "*.nc"), combine="by_coords"
        ) as ds_utci, xr.open_dataset(master_file) as ds_master:

            ds_utci = harmonize_time(ds_utci.load())
            ds_master.load()

            ds_master["utci_max"] = (
                ds_utci["utci"].resample(time="1D").max() - 273.15
            ).astype("float32")
            ds_master["utci_min"] = (
                ds_utci["utci"].resample(time="1D").min() - 273.15
            ).astype("float32")

            ds_master["utci_max"].attrs["units"] = "Celsius"
            ds_master["utci_min"].attrs["units"] = "Celsius"

            encoding = {
                v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_master.data_vars
            }
            ds_master.to_netcdf(temp_master, encoding=encoding)

        temp_master.replace(master_file)
        logging.info(
            "PASS 3 DONE | %s UPDATED (+UTCI_MAX/MIN) in %.1f min.",
            year_str,
            (time.monotonic() - start_time) / 60,
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as exc:
        logging.error("PASS 3 FAILED | %s hard-failed: %s", year_str, exc)


def execute_pass3(client, root, workers) -> None:
    """Run Pass 3 for all years with a thread pool."""
    logging.info("=" * 57)
    logging.info(
        "PASS 3 (UTCI) | %d WORKERS IN PARALLEL (all years)", workers
    )
    logging.info("=" * 57)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_pass3_worker, client, y, DEFAULT_AREA, root
            ): y
            for y in ALL_PERIODS
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    """Entry point: four-worker UTCI merge into master daily files."""
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(
        wait_until_complete=True, retry_max=500, sleep_max=120
    )

    try:
        logging.info("STARTING PASS 3: BIOCLIMATOLOGY (UTCI MAX & MIN)")
        execute_pass3(client, root, workers=4)
        logging.info("UTCI PIPELINE COMPLETE.")

    except KeyboardInterrupt:
        logging.warning(
            "SCRIPT INTERRUPTED. Temp folders are kept for resume."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
