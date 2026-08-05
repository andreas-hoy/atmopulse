#!/usr/bin/env python3
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
from typing import Any, Iterator

import cdsapi

DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
SINGLE_DATASET = "reanalysis-era5-single-levels"
DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ERA5 monthly files backwards from 2025 to 1940."
    )
    p.add_argument("--start-year", type=int, default=2025)
    p.add_argument("--end-year", type=int, default=1940)
    p.add_argument("--output", type=Path, default=Path("ERA5_ClimateTool"))
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
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                root / "download_monthly_backwards.log",
                encoding="utf-8",
            ),
        ],
    )


def iter_months_backwards(
    start_year: int,
    end_year: int,
) -> Iterator[tuple[int, int]]:
    for year in range(start_year, end_year - 1, -1):
        for month in range(12, 0, -1):
            yield year, month


def month_days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as file:
        magic = file.read(8)
    return magic.startswith(b"CDF") or magic.startswith(b"\x89HDF")


def normalize_download(part: Path, target: Path) -> None:
    with part.open("rb") as file:
        magic = file.read(8)

    if magic.startswith(b"CDF") or magic.startswith(b"\x89HDF"):
        os.replace(part, target)
        return

    if zipfile.is_zipfile(part):
        temp_dir = target.parent / f".extract_{target.stem}"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(part) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".nc", ".nc4", ".netcdf"))
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        f"Expected one NetCDF in ZIP, found {len(members)}: {members}"
                    )
                archive.extract(members[0], temp_dir)
                shutil.move(str(temp_dir / members[0]), str(target))
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
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": "reanalysis",
        "variable": [variable],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
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
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": ["00:00", "12:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def mslp_request(
    year: int,
    month: int,
    area: list[float],
) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["mean_sea_level_pressure"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": ["00:00", "12:00"],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_jobs(
    root: Path,
    products: list[str] | tuple[str, ...],
    year: int,
    month: int,
    area: list[float],
) -> list[tuple[str, dict[str, Any], Path]]:
    ym = f"{year}-{month:02d}"
    jobs: list[tuple[str, dict[str, Any], Path]] = []

    if "tx" in products:
        jobs.append((
            DAILY_DATASET,
            daily_request(
                "maximum_2m_temperature_since_previous_post_processing",
                "daily_maximum",
                year,
                month,
                area,
            ),
            root / "TX_daily_known_1h_issue" / f"era5_tx_daily_{ym}.nc",
        ))

    if "tn" in products:
        jobs.append((
            DAILY_DATASET,
            daily_request(
                "minimum_2m_temperature_since_previous_post_processing",
                "daily_minimum",
                year,
                month,
                area,
            ),
            root / "TN_daily_known_1h_issue" / f"era5_tn_daily_{ym}.nc",
        ))

    if "z500" in products:
        jobs.append((
            PRESSURE_DATASET,
            z500_request(year, month, area),
            root / "Z500_00_12UTC" / f"era5_z500_00_12utc_{ym}.nc",
        ))

    if "mslp" in products:
        jobs.append((
            SINGLE_DATASET,
            mslp_request(year, month, area),
            root / "MSLP_00_12UTC" / f"era5_mslp_00_12utc_{ym}.nc",
        ))

    return jobs


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
            "DONE | %s | %.1f MiB | %.1f min",
            target,
            target.stat().st_size / 1024**2,
            (time.monotonic() - started) / 60,
        )
        return True

    except KeyboardInterrupt:
        part.unlink(missing_ok=True)
        logging.warning("Interrupted by user while downloading %s", target)
        raise

    except Exception as exc:
        part.unlink(missing_ok=True)
        failed.write_text(
            f"{dt.datetime.now().isoformat()}\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        logging.exception("FAILED | %s", target)
        return False


def main() -> int:
    args = parse_args()

    if args.start_year > 2025:
        raise SystemExit("--start-year must be 2025 or earlier.")
    if args.end_year < 1940:
        raise SystemExit("--end-year must be 1940 or later.")
    if args.start_year < args.end_year:
        raise SystemExit(
            "--start-year must be newer than or equal to --end-year."
        )

    root = args.output.resolve()
    setup_logging(root)

    (root / "PROTOTYPE_NOTICE.txt").write_text(
        "TX/TN use ERA5 parameters affected by the known +1-hour "
        "time-assignment issue. Prototype use only; replace before "
        "beta/production.\n",
        encoding="utf-8",
    )

    logging.info(
        "Period: %d-12 backwards to %d-01",
        args.start_year,
        args.end_year,
    )
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

    try:
        for year, month in iter_months_backwards(
            args.start_year,
            args.end_year,
        ):
            logging.info("MONTH | %04d-%02d", year, month)
            jobs = build_jobs(
                root=root,
                products=args.products,
                year=year,
                month=month,
                area=args.area,
            )

            for dataset, request, target in jobs:
                if retrieve(
                    client,
                    dataset,
                    request,
                    target,
                    args.overwrite,
                ):
                    completed += 1
                else:
                    failed += 1

    except KeyboardInterrupt:
        logging.warning(
            "Stopped by user. Completed/skipped=%d | failed=%d",
            completed,
            failed,
        )
        return 130

    logging.info(
        "Finished | completed/skipped=%d | failed=%d",
        completed,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
