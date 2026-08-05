#!/usr/bin/env python3
"""
SynEx data updater: ERA5 Master_Batches + IFS Live_Forecasts bridge.

Usage (from project root, with cee_env or .venv active):
    python update_recent_data.py                  # ERA5 to today-5d + IFS
    python update_recent_data.py --era5-only
    python update_recent_data.py --ifs-only
    python update_recent_data.py --dry-run
"""
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
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "ERA5_ClimateTool" / "Master_Batches"
LIVE = ROOT / "ERA5_ClimateTool" / "Live_Forecasts"
AREA = [72.0, -25.0, 30.0, 45.0]
SINGLE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"

SURFACE_VARS = [
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "mean_sea_level_pressure",
]


def parse_args() -> argparse.Namespace:
    today = dt.date.today()
    default_end = today - dt.timedelta(days=5)
    p = argparse.ArgumentParser(description="Update ERA5 batches and IFS live forecasts for SynEx.")
    p.add_argument("--end-date", type=dt.date.fromisoformat, default=default_end,
                   help=f"Last ERA5 date to fetch (default: today-5 = {default_end})")
    p.add_argument("--era5-lag-days", type=int, default=5,
                   help="Used when --end-date omitted: today minus N days")
    p.add_argument("--era5-only", action="store_true")
    p.add_argument("--ifs-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite-era5", action="store_true",
                   help="Re-download even if local batch already covers end-date")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def latest_valid_time(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if "valid_time" not in ds:
            return None
        return pd.Timestamp(ds.valid_time.max().values).tz_localize(None).normalize()


def days_in_range(start: dt.date, end: dt.date) -> list[str]:
    days = []
    cur = start
    while cur <= end:
        days.append(f"{cur.day:02d}")
        cur += dt.timedelta(days=1)
    return days


def months_in_range(start: dt.date, end: dt.date) -> list[tuple[str, list[str]]]:
    """Return [(month, [days]), ...] for CDS requests."""
    chunks: dict[str, list[str]] = {}
    cur = start
    while cur <= end:
        key = f"{cur.month:02d}"
        chunks.setdefault(key, []).append(f"{cur.day:02d}")
        cur += dt.timedelta(days=1)
    return sorted(chunks.items())


def extract_surface_zip(zip_path: Path, tmp_dir: Path) -> tuple[xr.Dataset | None, xr.Dataset | None]:
    """Return (txtn_dataset with mx2t/mn2t, mslp_dataset)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    txtn_parts: list[xr.Dataset] = []
    mslp_ds: xr.Dataset | None = None

    for f in sorted(tmp_dir.glob("*.nc")):
        with xr.open_dataset(f, engine="netcdf4") as ds:
            vars_ = set(ds.data_vars)
            if {"mx2t", "mn2t"}.issubset(vars_):
                txtn_parts.append(ds.load())
            elif "mx2t" in vars_ or "mn2t" in vars_:
                txtn_parts.append(ds.load())
            elif "msl" in vars_:
                mslp_ds = ds.load()
            else:
                name = f.name.lower()
                if "max" in name or "min" in name:
                    txtn_parts.append(ds.load())
                elif "instant" in name or "msl" in name:
                    mslp_ds = ds.load()
                else:
                    logging.warning("Unknown surface file in ZIP: %s vars=%s", f.name, list(vars_))

    txtn_ds = None
    if txtn_parts:
        txtn_ds = xr.merge(txtn_parts, compat="override", join="outer")
    return txtn_ds, mslp_ds


def merge_tx_tn_parts(parts: list[Path]) -> xr.Dataset | None:
    datasets = [xr.open_dataset(p, engine="netcdf4") for p in parts]
    if not datasets:
        return None
    merged = xr.merge(datasets, compat="override", join="outer")
    for ds in datasets:
        ds.close()
    return merged


def dedupe_time(ds: xr.Dataset) -> xr.Dataset:
    times = pd.to_datetime(ds.valid_time.values)
    _, idx = np.unique(times, return_index=True)
    return ds.isel(valid_time=sorted(idx)).sortby("valid_time")


def append_batch(existing: Path, new_ds: xr.Dataset, dry_run: bool = False) -> None:
    new_ds = dedupe_time(new_ds)
    if existing.exists():
        with xr.open_dataset(existing, engine="netcdf4") as old:
            last = pd.Timestamp(old.valid_time.max().values)
            new_ds = new_ds.sel(valid_time=slice(last + pd.Timedelta(hours=1), None))
            if new_ds.sizes.get("valid_time", 0) == 0:
                logging.info("SKIP merge | %s already up to date", existing.name)
                return
            combined = dedupe_time(xr.concat([old, new_ds], dim="valid_time"))
    else:
        combined = new_ds

    if dry_run:
        logging.info("DRY-RUN | would write %s (%d steps)", existing.name, combined.sizes["valid_time"])
        return

    tmp = existing.with_suffix(".nc.tmp")
    combined.to_netcdf(tmp, engine="netcdf4")
    os.replace(tmp, existing)
    logging.info("UPDATED | %s -> %s", existing.name, str(combined.valid_time.max().values)[:10])


def cds_retrieve(client: cdsapi.Client, dataset: str, request: dict[str, Any], target: Path) -> Path:
    part = target.parent / (target.name + ".part")
    part.unlink(missing_ok=True)
    days = request.get("day")
    logging.info(
        "CDS request | %s | month=%s days=%s..%s",
        dataset, request.get("month"),
        days[0] if days else "?", days[-1] if days else "?",
    )
    client.retrieve(dataset, request, str(part))
    if zipfile.is_zipfile(part):
        return part
    os.replace(part, target)
    return target


def update_era5(end_date: dt.date, dry_run: bool = False, overwrite: bool = False) -> bool:
    MASTER.mkdir(parents=True, exist_ok=True)
    txtn_file = MASTER / "era5_txtn_batch_2026_2026.nc"
    mslp_file = MASTER / "era5_mslp_batch_2026_2026.nc"
    z500_file = MASTER / "era5_z500_batch_2026_2026.nc"

    latest = latest_valid_time(txtn_file)
    if latest is not None and not overwrite:
        start_date = (latest + pd.Timedelta(days=1)).date()
    else:
        start_date = dt.date(2026, 1, 1)

    if start_date > end_date:
        logging.info("ERA5 already current through %s (target %s)", latest.date() if latest else "?", end_date)
        return True

    logging.info("ERA5 gap | %s -> %s", start_date, end_date)
    if dry_run:
        for month, days in months_in_range(start_date, end_date):
            logging.info("DRY-RUN surface+z500 | 2026-%s days %s..%s", month, days[0], days[-1])
        return True

    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)
    tmp_root = MASTER / ".update_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    for month, days in months_in_range(start_date, end_date):
        surf_req = {
            "product_type": "reanalysis",
            "variable": SURFACE_VARS,
            "year": "2026",
            "month": month,
            "day": days,
            "time": ["00:00", "12:00"],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        z_req = {
            "product_type": "reanalysis",
            "variable": "geopotential",
            "pressure_level": "500",
            "year": "2026",
            "month": month,
            "day": days,
            "time": ["00:00", "12:00"],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        chunk_dir = tmp_root / f"2026_{month}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        surf_download = chunk_dir / "surface_download"
        surf_download.unlink(missing_ok=True)

        try:
            surf_path = cds_retrieve(client, SINGLE_DATASET, surf_req, surf_download)
            if zipfile.is_zipfile(surf_path):
                extract_dir = chunk_dir / "extract"
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                ds_txtn, ds_mslp = extract_surface_zip(surf_path, extract_dir)
                if ds_txtn is not None:
                    append_batch(txtn_file, ds_txtn, dry_run=False)
                    ds_txtn.close()
                else:
                    logging.error("No TX/TN data found in surface ZIP")
                if ds_mslp is not None:
                    append_batch(mslp_file, ds_mslp, dry_run=False)
                    ds_mslp.close()
                else:
                    logging.error("No MSLP data found in surface ZIP")
            else:
                logging.error("Surface download is not a ZIP: %s", surf_path)

            z_download = chunk_dir / "z500_download"
            z_download.unlink(missing_ok=True)
            z_path = cds_retrieve(client, PRESSURE_DATASET, z_req, z_download)
            if zipfile.is_zipfile(z_path):
                with zipfile.ZipFile(z_path) as zf:
                    zf.extractall(chunk_dir / "zextract")
                z_nc = next((chunk_dir / "zextract").glob("*.nc"))
                with xr.open_dataset(z_nc, engine="netcdf4") as ds_z:
                    append_batch(z500_file, ds_z, dry_run=False)
            elif z_path.with_suffix(z_path.suffix + ".part").exists():
                z_part = z_path.with_suffix(z_path.suffix + ".part")
                with xr.open_dataset(z_part, engine="netcdf4") as ds_z:
                    append_batch(z500_file, ds_z, dry_run=False)
            elif z_path.exists() and z_path.stat().st_size > 0:
                with xr.open_dataset(z_path, engine="netcdf4") as ds_z:
                    append_batch(z500_file, ds_z, dry_run=False)
        except Exception:
            logging.exception("FAILED | month 2026-%s", month)
            return False

    shutil.rmtree(tmp_root, ignore_errors=True)
    logging.info("ERA5 update complete.")
    return True


def _regrid_to_era5(ds: xr.Dataset, ref: xr.Dataset) -> xr.Dataset:
    return ds.interp(latitude=ref.latitude, longitude=ref.longitude, method="linear")


def _europe_slice(ds: xr.Dataset) -> xr.Dataset:
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Cannot find lat/lon in dataset coords: {list(ds.coords)}")
    if float(ds[lon_name].max()) > 180:
        ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180}).sortby(lon_name)
    return ds.sel({lat_name: slice(72.0, 30.0), lon_name: slice(-25.0, 45.0)})


def _daily_extrema_2t(ds_t: xr.Dataset, ref: xr.Dataset) -> xr.Dataset:
    """Convert 2t forecast steps to daily mx2t/mn2t on ERA5 grid."""
    var = "t2m" if "t2m" in ds_t else "2t"
    daily_max = ds_t.groupby("valid_time.date").max("step")
    daily_min = ds_t.groupby("valid_time.date").min("step")
    daily_max = daily_max.rename({"date": "valid_time"})
    daily_min = daily_min.rename({"date": "valid_time"})
    daily_max["valid_time"] = pd.to_datetime(daily_max["valid_time"].values)
    daily_min["valid_time"] = pd.to_datetime(daily_min["valid_time"].values)
    tx = _regrid_to_era5(daily_max, ref).rename({var: "mx2t"})
    tn = _regrid_to_era5(daily_min, ref).rename({var: "mn2t"})
    return dedupe_time(xr.merge([tx, tn]))


def _daily_msl_or_z(ds: xr.Dataset, ref: xr.Dataset, out_var: str, scale: float = 1.0) -> xr.Dataset:
    """Daily mean (12Z-ish) surface/level field regridded to ERA5."""
    src_var = next(iter(ds.data_vars))
    daily = ds.groupby("valid_time.date").mean("step")
    daily = daily.rename({"date": "valid_time"})
    daily["valid_time"] = pd.to_datetime(daily["valid_time"].values)
    out = _regrid_to_era5(daily, ref)
    out = out.rename({src_var: out_var})
    if scale != 1.0:
        out[out_var] = out[out_var] * scale
    return dedupe_time(out)


def update_ifs(dry_run: bool = False) -> bool:
    try:
        from ecmwf.opendata import Client
    except ImportError:
        logging.error("ecmwf-opendata not installed. Run: pip install ecmwf-opendata cfgrib")
        return False

    ref_file = MASTER / "era5_txtn_batch_2026_2026.nc"
    if not ref_file.exists():
        ref_file = next(iter(sorted(MASTER.glob("era5_txtn_batch_*.nc"))), None)
    if ref_file is None:
        logging.error("No ERA5 reference grid found in Master_Batches.")
        return False

    LIVE.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logging.info("DRY-RUN | would fetch IFS open data and write Live_Forecasts/live_forecast_*.nc")
        return True

    logging.info("IFS | downloading ECMWF open-data forecast (0-144h, 6-hourly)...")
    client = Client(source="ecmwf")
    steps = list(range(0, 145, 6))
    grib_t = LIVE / "ifs_2t.grib"
    grib_m = LIVE / "ifs_msl.grib"
    grib_z = LIVE / "ifs_gh500.grib"

    try:
        client.retrieve(time=0, step=steps, type="fc", param="2t", target=str(grib_t))
        client.retrieve(time=0, step=steps, type="fc", param="msl", target=str(grib_m))
        client.retrieve(time=0, step=steps, type="fc", param="gh", levelist=500, target=str(grib_z))
    except Exception:
        logging.exception("IFS download failed")
        return False

    try:
        import cfgrib  # noqa: F401
    except ImportError:
        logging.error("cfgrib not installed. Run: conda install -c conda-forge eccodes cfgrib")
        return False

    logging.info("IFS | processing GRIB -> NetCDF live forecasts...")
    with xr.open_dataset(ref_file, engine="netcdf4") as ref:
        ds_t = _europe_slice(xr.open_dataset(grib_t, engine="cfgrib"))
        ds_txtn = _daily_extrema_2t(ds_t, ref)
        ds_txtn.to_netcdf(LIVE / "live_forecast_txtn.nc", engine="netcdf4")
        ds_t.close()

        ds_m = _europe_slice(xr.open_dataset(grib_m, engine="cfgrib"))
        msl = _daily_msl_or_z(ds_m, ref, "msl")
        msl.to_netcdf(LIVE / "live_forecast_mslp.nc", engine="netcdf4")
        ds_m.close()

        ds_z = _europe_slice(xr.open_dataset(grib_z, engine="cfgrib"))
        # gh [m] -> ERA5 geopotential z [m2/s2]
        z = _daily_msl_or_z(ds_z, ref, "z", scale=9.80665)
        z.to_netcdf(LIVE / "live_forecast_z500.nc", engine="netcdf4")
        ds_z.close()

    for f in (grib_t, grib_m, grib_z, LIVE / "ifs_forecast_temp.grib"):
        f.unlink(missing_ok=True)
    logging.info("IFS live forecasts saved to %s", LIVE)
    return True


def main() -> int:
    setup_logging()
    args = parse_args()
    end_date = args.end_date
    if args.era5_lag_days and "--end-date" not in sys.argv:
        end_date = dt.date.today() - dt.timedelta(days=args.era5_lag_days)

    logging.info("SynEx update | target ERA5 end: %s", end_date)

    ok = True
    if not args.ifs_only:
        ok = update_era5(end_date, dry_run=args.dry_run, overwrite=args.overwrite_era5) and ok
    if not args.era5_only:
        ok = update_ifs(dry_run=args.dry_run) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
