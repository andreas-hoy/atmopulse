#!/usr/bin/env python3
"""
ERA5 prototype downloader: quick 2026 test, then full 1940–2026 archive.

Products:
  - TX daily from maximum_2m_temperature_since_previous_post_processing
  - TN daily from minimum_2m_temperature_since_previous_post_processing
  - Z500 at 00 and 12 UTC
  - MSLP at 00 and 12 UTC

The TX/TN source parameters are currently affected by the known +1-hour
time-assignment issue. They are intentionally used here for a prototype.
Replace them before beta/production.

Default Europe box:
  North 72°, West -25°, South 30°, East 45°

Typical use:
  1) Test 2026:
     py era5_prototype_then_full.py --mode test

  2) Full archive after successful test:
     py era5_prototype_then_full.py --mode full

The script downloads one month per file. It is restart-safe: valid existing
NetCDF files are skipped.
"""

from __future__ import annotations

import argparse
import calendar
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

DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
SINGLE_DATASET = "reanalysis-era5-single-levels"

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]
FIRST_YEAR = 1940
LAST_YEAR = 2026


def parse_args() -> argparse.Namespace:
    today = dt.date.today()
    default_2026_end = min(dt.date(2026, 12, 31), today - dt.timedelta(days=7))

    p = argparse.ArgumentParser(
        description="ERA5 prototype test downloader and full archive downloader."
    )
    p.add_argument(
        "--mode",
        choices=("test", "full"),
        default="test",
        help=(
            "test = 2026 only; full = 1940 through the available part of 2026. "
            "Default: test"
        ),
    )
    p.add_argument(
        "--test-start",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 1, 1),
        help="Start date for test mode. Default: 2026-01-01",
    )
    p.add_argument(
        "--end-date",
        type=dt.date.fromisoformat,
        default=default_2026_end,
        help=f"Last date for 2026. Default: {default_2026_end}",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ERA5_ClimateTool"),
        help="Output root folder.",
    )
    p.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=DEFAULT_AREA,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
    )
    p.add_argument(
        "--products",
        nargs="+",
        choices=("tx", "tn", "z500", "mslp"),
        default=("tx", "tn", "z500", "mslp"),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload existing valid files.",
    )
    return p.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(root / "download.log", encoding="utf-8"),
        ],
    )


def iter_months(start_date: dt.date, end_date: dt.date):
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def days_for_month(
    year: int,
    month: int,
    start_date: dt.date,
    end_date: dt.date,
) -> list[str]:
    first_day = 1
    last_day = calendar.monthrange(year, month)[1]

    if year == start_date.year and month == start_date.month:
        first_day = start_date.day
    if year == end_date.year and month == end_date.month:
        last_day = min(last_day, end_date.day)

    return [f"{day:02d}" for day in range(first_day, last_day + 1)]


def valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        magic = f.read(8)
    return magic.startswith(b"CDF") or magic.startswith(b"\x89HDF")


def normalize_download(part: Path, target: Path) -> None:
    with part.open("rb") as f:
        magic = f.read(8)

    if magic.startswith(b"CDF") or magic.startswith(b"\x89HDF"):
        os.replace(part, target)
        return

    if zipfile.is_zipfile(part):
        temp_dir = target.parent / f".extract_{target.stem}"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(part) as zf:
                members = [
                    name for name in zf.namelist()
                    if name.lower().endswith((".nc", ".nc4", ".netcdf"))
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected one NetCDF in ZIP, found {len(members)}: {members}"
                    )
                zf.extract(members[0], temp_dir)
                extracted = temp_dir / members[0]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extracted), str(target))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            part.unlink(missing_ok=True)
        return

    raise RuntimeError(f"Downloaded file is neither NetCDF nor ZIP: {part}")


def daily_request(
    variable: str,
    statistic: str,
    year: int,
    month: int,
    days: list[str],
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": "reanalysis",
        "variable": [variable],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "daily_statistic": statistic,
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def z500_request(
    year: int,
    month: int,
    days: list[str],
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": ["00:00", "12:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def mslp_request(
    year: int,
    month: int,
    days: list[str],
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["mean_sea_level_pressure"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": ["00:00", "12:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def retrieve(
    client: cdsapi.Client,
    dataset: str,
    request: dict[str, Any],
    target: Path,
    overwrite: bool,
) -> bool:
    if valid_netcdf(target) and not overwrite:
        logging.info("SKIP | %s", target)
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    failed = target.with_suffix(target.suffix + ".failed")
    part.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)

    logging.info("START | %s", target)
    started = time.monotonic()

    try:
        client.retrieve(dataset, request, str(part))
        normalize_download(part, target)

        if not valid_netcdf(target):
            raise RuntimeError(f"NetCDF validation failed: {target}")

        logging.info(
            "DONE  | %s | %.1f MiB | %.1f min",
            target,
            target.stat().st_size / 1024**2,
            (time.monotonic() - started) / 60,
        )
        return True

    except Exception as exc:
        part.unlink(missing_ok=True)
        failed.write_text(
            f"{dt.datetime.now().isoformat()}\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        logging.exception("FAILED | %s", target)
        return False


def build_jobs(
    root: Path,
    products: tuple[str, ...] | list[str],
    year: int,
    month: int,
    days: list[str],
    area: list[float],
) -> list[tuple[str, dict[str, Any], Path]]:
    ym = f"{year}-{month:02d}"
    jobs: list[tuple[str, dict[str, Any], Path]] = []

    if "tx" in products:
        jobs.append(
            (
                DAILY_DATASET,
                daily_request(
                    "maximum_2m_temperature_since_previous_post_processing",
                    "daily_maximum",
                    year,
                    month,
                    days,
                    area,
                ),
                root / "TX_daily_known_1h_issue" / f"era5_tx_daily_{ym}.nc",
            )
        )

    if "tn" in products:
        jobs.append(
            (
                DAILY_DATASET,
                daily_request(
                    "minimum_2m_temperature_since_previous_post_processing",
                    "daily_minimum",
                    year,
                    month,
                    days,
                    area,
                ),
                root / "TN_daily_known_1h_issue" / f"era5_tn_daily_{ym}.nc",
            )
        )

    if "z500" in products:
        jobs.append(
            (
                PRESSURE_DATASET,
                z500_request(year, month, days, area),
                root / "Z500_00_12UTC" / f"era5_z500_00_12utc_{ym}.nc",
            )
        )

    if "mslp" in products:
        jobs.append(
            (
                SINGLE_DATASET,
                mslp_request(year, month, days, area),
                root / "MSLP_00_12UTC" / f"era5_mslp_00_12utc_{ym}.nc",
            )
        )

    return jobs

def main() -> int:
    args = parse_args()
    root = args.output.resolve()
    setup_logging(root)

    if args.end_date.year != 2026:
        raise SystemExit("--end-date must be in 2026.")

    if args.mode == "test":
        start_date = args.test_start
        if start_date.year != 2026:
            raise SystemExit("--test-start must be in 2026.")
    else:
        start_date = dt.date(FIRST_YEAR, 1, 1)

    end_date = args.end_date
    if start_date > end_date:
        raise SystemExit("Start date is after end date.")

    notice = root / "PROTOTYPE_NOTICE.txt"
    notice.write_text(
        "TX/TN use the ERA5 parameters affected by the known +1-hour "
        "time-assignment issue. These files are for prototype use only and "
        "should be replaced before beta/production.\n",
        encoding="utf-8",
    )

    logging.info("Mode: %s", args.mode)
    logging.info("Period: %s to %s", start_date, end_date)
    logging.info("Products: %s", ", ".join(args.products))
    logging.info("Area [N,W,S,E]: %s", args.area)
    logging.info("Output: %s", root)

    client = cdsapi.Client(
        wait_until_complete=True,
        retry_max=500,
        sleep_max=120,
    )

    completed = 0
    failed = 0

    for year, month in iter_months(start_date, end_date):
        days = days_for_month(year, month, start_date, end_date)
        jobs = build_jobs(
            root=root,
            products=args.products,
            year=year,
            month=month,
            days=days,
            area=args.area,
        )

        for dataset, request, target in jobs:
            if retrieve(client, dataset, request, target, args.overwrite):
                completed += 1
            else:
                failed += 1

    logging.info("Finished | completed/skipped=%d | failed=%d", completed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())