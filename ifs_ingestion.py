#!/usr/bin/env python3
"""Ingest the latest ECMWF IFS deterministic run onto the ERA5 grid.

Downloads surface (mx2t3, mn2t3, 2t, msl) and pressure-level (T, Z, U, V at 300/500/850 hPa)
fields via ecmwf-opendata, resolves cfgrib time-coordinate conflicts,
aggregates daily TX/TN/TG (calendar-day 00–00 UTC resample) and 12 UTC
synoptics, then applies CDO conservative (fracarea-normalised) regridding.
Assigns an ETCCDI 365-day ``doy_365`` coordinate for the frontend and QDM.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import xarray as xr
from ecmwf.opendata import Client

warnings.filterwarnings("ignore", module="cfgrib")

BASE_DIR = Path.cwd() / "ERA5_ClimateTool"
TMP_DIR = BASE_DIR / ".tmp_ifs"
OUT_DIR = BASE_DIR / "Live_Forecasts"
REF_DIR = BASE_DIR / "Reference_Climatology"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERA5_GRID_REF = REF_DIR / "climatology_synoptics.nc"
WEIGHTS_FILE = REF_DIR / "regrid_weights_cdo.nc"


def setup_logging() -> None:
    """Configure INFO logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def download_ifs_gribs():
    """Retrieve the newest available IFS run from ECMWF open data."""
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # 3-hourly steps up to 96 hours.
    # CRITICAL: step=0 is explicitly omitted to prevent cfgrib duplication and data corruption
    # for accumulated fields (mx2t3, mn2t3).
    steps_3h = list(range(3, 120, 3))
    
    target_sfc = str(TMP_DIR / "ifs_sfc.grib")
    target_pl = str(TMP_DIR / "ifs_pl.grib")

    now = datetime.now(timezone.utc)
    candidates = [
        (
            now - timedelta(hours=12 * i),
            12 if (now - timedelta(hours=12 * i)).hour >= 12 else 0,
        )
        for i in range(6)
    ]

    for dt, hour in candidates:
        date_str = dt.strftime("%Y%m%d")
        logging.info("Checking run %s %02dZ...", date_str, hour)
        try:
            client.retrieve(
                date=date_str,
                time=hour,
                type="fc",
                stream="oper",
                levtype="sfc",
                param=["2t", "msl", "mx2t3", "mn2t3"],
                step=steps_3h,
                target=target_sfc,
            )
            client.retrieve(
                date=date_str,
                time=hour,
                type="fc",
                stream="oper",
                levtype="pl",
                levelist=[300, 500, 850],
                param=["t", "z", "u", "v"],
                step=steps_3h,
                target=target_pl,
            )
            return target_sfc, target_pl, date_str, hour
        except Exception:
            logging.warning("Run not available. Falling back to previous...")
            time.sleep(2)
    raise RuntimeError("No IFS run found.")


def apply_conservative_weights(ds_source, weights_file, ds_target_grid):
    """Regrid with a precomputed CDO matrix and fracarea normalisation."""
    logging.info(
        "Harmonising grid and applying conservative CDO matrix "
        "(fracarea) via SciPy..."
    )

    ds_source = ds_source.assign_coords(
        longitude=(((ds_source.longitude + 180) % 360) - 180)
    ).sortby("longitude")

    ds_source_cropped = ds_source.sel(
        latitude=ds_target_grid.latitude,
        longitude=ds_target_grid.longitude,
        method="nearest",
    )

    with xr.open_dataset(weights_file) as ds_w:
        weights = sps.coo_matrix(
            (
                ds_w["remap_matrix"][:, 0].values,
                (
                    ds_w["dst_address"].values - 1,
                    ds_w["src_address"].values - 1,
                ),
            ),
            shape=(ds_w.sizes["dst_grid_size"], ds_w.sizes["src_grid_size"]),
        ).tocsr()

    shape_out = (
        ds_target_grid.sizes["latitude"],
        ds_target_grid.sizes["longitude"],
    )
    ds_out = xr.Dataset(
        coords={
            "time": ds_source_cropped.time,
            "latitude": ds_target_grid.latitude,
            "longitude": ds_target_grid.longitude,
        }
    )

    for var in ds_source_cropped.data_vars:
        data_arrays = []
        for t_idx in range(ds_source_cropped.sizes["time"]):
            flat_source = (
                ds_source_cropped[var].isel(time=t_idx).values.flatten()
            )

            # Fracarea normalisation: ignore NaN source cells in the
            # weighted sum and divide by the weight of valid cells.
            valid_mask = ~np.isnan(flat_source)
            source_filled = np.where(valid_mask, flat_source, 0.0)

            y_num = weights.dot(source_filled)
            y_den = weights.dot(valid_mask.astype(np.float32))

            with np.errstate(divide="ignore", invalid="ignore"):
                y_corrected = np.where(y_den > 0, y_num / y_den, np.nan)

            # Hard copy: severs aliasing to the SciPy .dot() output buffer,
            # which can otherwise be reused/overwritten on later t_idx passes.
            data_arrays.append(np.array(y_corrected.reshape(shape_out), copy=True))

        ds_out[var] = (
            ("time", "latitude", "longitude"),
            np.stack(data_arrays).astype(np.float32),
        )

    return ds_out


def process_and_align_ifs(sfc_file, pl_file):
    """Aggregate daily 00–00 extremes / 12Z synoptics and regrid onto ERA5."""
    logging.info("Loading GRIBs and resolving xarray 'time' conflicts...")
    ds_sfc = xr.open_dataset(sfc_file, engine="cfgrib")
    ds_pl = xr.open_dataset(pl_file, engine="cfgrib")

    ds_sfc = ds_sfc.swap_dims({"step": "valid_time"})
    ds_pl = ds_pl.swap_dims({"step": "valid_time"})

    if "time" in ds_sfc.coords:
        ds_sfc = ds_sfc.drop_vars("time")
    if "time" in ds_pl.coords:
        ds_pl = ds_pl.drop_vars("time")

    ds_sfc = ds_sfc.rename({"valid_time": "time"})
    ds_pl = ds_pl.rename({"valid_time": "time"})

    tg_inst_c = ds_sfc["t2m"] - 273.15
    tx_c = ds_sfc["mx2t3"] - 273.15
    tn_c = ds_sfc["mn2t3"] - 273.15
    mslp_hpa = ds_sfc["msl"] / 100.0

    logging.info("Aggregating strict 00-00 UTC T-extremes and 12Z synoptics...")
    
    # Shift bin closure to the right to capture the trailing 00:00 UTC step
    daily_tx = tx_c.resample(time="1D", closed="right", label="left").max()
    daily_tn = tn_c.resample(time="1D", closed="right", label="left").min()
    daily_tg = tg_inst_c.resample(time="1D", closed="right", label="left").mean()
    
    # Enforce strict 0-0 UTC completeness (exactly 8 steps per calendar day)
    daily_count = tx_c.resample(time="1D", closed="right", label="left").count()
    
    tx = daily_tx.where(daily_count == 8, drop=True)
    tn = daily_tn.where(daily_count == 8, drop=True)
    tg = daily_tg.where(daily_count == 8, drop=True)

    # 12Z Synoptic Extraction
    ds_pl_12z = ds_pl.sel(time=ds_pl.time.dt.hour == 12)
    mslp_12z = mslp_hpa.sel(time=mslp_hpa.time.dt.hour == 12)

    mslp = mslp_12z.resample(time="1D").first().sel(time=tx.time)

    z500 = (
        ds_pl_12z["z"]
        .sel(isobaricInhPa=500)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tx.time)
    )
    t850 = (
        (ds_pl_12z["t"].sel(isobaricInhPa=850) - 273.15)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tx.time)
    )
    u300 = (
        ds_pl_12z["u"]
        .sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tx.time)
    )
    v300 = (
        ds_pl_12z["v"]
        .sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tx.time)
    )

    ds_forecast = xr.Dataset(
        {
            "tx": tx.astype("float32"),
            "tn": tn.astype("float32"),
            "tg": tg.astype("float32"),
            "mslp": mslp.astype("float32"),
            "z500": z500.astype("float32"),
            "t850": t850.astype("float32"),
            "u300": u300.astype("float32"),
            "v300": v300.astype("float32"),
        }
    )

    with xr.open_dataset(ERA5_GRID_REF) as ds_era5:
        ds_aligned = apply_conservative_weights(
            ds_forecast, WEIGHTS_FILE, ds_era5
        )

    raw_doys = ds_aligned.time.dt.dayofyear.values
    is_leap = ds_aligned.time.dt.is_leap_year.values
    months = ds_aligned.time.dt.month.values
    doy_365 = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    ds_aligned = ds_aligned.assign_coords(doy_365=("time", doy_365))

    return ds_aligned


def purge_old_forecasts(days_to_keep: int = 10) -> None:
    """Delete IFS forecast files older than days_to_keep."""
    logging.info(
        "Garbage collection: deleting IFS files older than %s days...",
        days_to_keep,
    )
    now = time.time()
    for f in OUT_DIR.glob("ifs_daily_forecast_*.nc"):
        if os.stat(f).st_mtime < now - (days_to_keep * 86400):
            try:
                f.unlink()
                logging.info("Deleted: %s", f.name)
            except Exception as exc:
                logging.error("Delete failed for %s: %s", f.name, exc)


def main() -> None:
    """Download, aggregate, regrid, and write the IFS daily forecast."""
    setup_logging()

    if not ERA5_GRID_REF.exists() or not WEIGHTS_FILE.exists():
        logging.error("Reference or weights file is missing!")
        sys.exit(1)

    try:
        sfc_file, pl_file, run_date, run_hour = download_ifs_gribs()
        ds_aligned = process_and_align_ifs(sfc_file, pl_file)

        out_path = OUT_DIR / f"ifs_daily_forecast_{run_date}_{run_hour:02d}z.nc"

        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in ds_aligned.data_vars
        }
        ds_aligned.to_netcdf(out_path, encoding=encoding)
        logging.info("SUCCESS. Live forecast saved: %s", out_path.name)

    except Exception as exc:
        logging.error("IFS pipeline failed: %s", exc)
    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        purge_old_forecasts(days_to_keep=10)


if __name__ == "__main__":
    main()
