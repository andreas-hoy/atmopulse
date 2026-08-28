#!/usr/bin/env python3
"""Ingest the latest ECMWF AIFS deterministic run onto the ERA5 grid.

Downloads surface (2t, msl) and pressure-level (T, Z, U, V at 300/500/850 hPa)
fields via ecmwf-opendata, aggregates daily TG (calendar-day resample)
and 12 UTC synoptics, then applies CDO conservative (fracarea-normalised)
regridding. Assigns an ETCCDI 365-day ``doy_365`` coordinate for the frontend.
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
TMP_DIR = BASE_DIR / ".tmp_aifs"
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


def download_aifs_gribs():
    """Retrieve the newest available AIFS run from the Azure open-data mirror."""
    sources = ["azure"]
    # 6-hourly steps up to 96 hours (4 days).
    steps = list(range(0, 120, 6))

    target_sfc = str(TMP_DIR / "aifs_sfc.grib")
    target_pl = str(TMP_DIR / "aifs_pl.grib")

    now = datetime.now(timezone.utc)
    candidates = [
        (now, 12),
        (now, 0),
        (now - timedelta(days=1), 12),
        (now - timedelta(days=1), 0),
    ]

    for src in sources:
        logging.info("Connecting to cloud mirror: %s...", src.upper())
        client = Client(source=src, model="aifs-single", resol="0p25")

        for dt, hour in candidates:
            date_str = dt.strftime("%Y%m%d")
            logging.info(
                "Checking %s for AIFS run: %s %02dZ...",
                src.upper(),
                date_str,
                hour,
            )

            try:
                client.retrieve(
                    date=date_str,
                    time=hour,
                    type="fc",
                    levtype="sfc",
                    param=["2t", "msl"],
                    step=steps,
                    target=target_sfc,
                )
                logging.info(
                    "Surface data (%s %02dZ) loaded via %s.",
                    date_str,
                    hour,
                    src.upper(),
                )

                client.retrieve(
                    date=date_str,
                    time=hour,
                    type="fc",
                    levtype="pl",
                    levelist=[300, 500, 850],
                    param=["t", "z", "u", "v"],
                    step=steps,
                    target=target_pl,
                )
                logging.info(
                    "Pressure-level data (%s %02dZ) loaded via %s.",
                    date_str,
                    hour,
                    src.upper(),
                )

                return target_sfc, target_pl, date_str, hour

            except Exception as exc:
                logging.warning(
                    "Run %s %02dZ not ready on %s: %s",
                    date_str,
                    hour,
                    src.upper(),
                    exc,
                )
                if Path(target_sfc).exists():
                    Path(target_sfc).unlink()
                if Path(target_pl).exists():
                    Path(target_pl).unlink()
                time.sleep(1)

    raise RuntimeError("ERROR: no AIFS run found on Azure.")


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


def process_and_align_aifs(sfc_file, pl_file):
    """Aggregate daily mean temperature / 12Z synoptics and regrid onto ERA5."""
    logging.info("Loading GRIB files via cfgrib into RAM...")
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

    t2m_celsius = ds_sfc["t2m"] - 273.15
    mslp_hpa = ds_sfc["msl"] / 100.0

    logging.info(
        "Aggregating strict 00-00 UTC daily means and 12Z synoptics..."
    )
    logging.warning(
        "AIFS utilizes discontinuous 6-hourly state jumps. True diurnal TX/TN "
        "extremes are absent from output tensors. Extreme extraction is OMITTED."
    )
    
    # Calculate daily mean (TG) with right closure to capture the 00:00 step correctly.
    daily_tg = t2m_celsius.resample(time="1D", closed="right", label="left").mean()
    
    # Enforce strict calendar day completeness (exactly 4 steps per day for 6-hourly data).
    daily_count = t2m_celsius.resample(time="1D", closed="right", label="left").count()
    tg = daily_tg.where(daily_count == 4, drop=True)

    ds_pl_12z = ds_pl.sel(time=ds_pl.time.dt.hour == 12)
    mslp_12z = mslp_hpa.sel(time=mslp_hpa.time.dt.hour == 12)

    mslp = mslp_12z.resample(time="1D").first().sel(time=tg.time)

    z500 = (
        ds_pl_12z["z"]
        .sel(isobaricInhPa=500)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tg.time)
    )
    t850 = (
        (ds_pl_12z["t"].sel(isobaricInhPa=850) - 273.15)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tg.time)
    )
    u300 = (
        ds_pl_12z["u"]
        .sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tg.time)
    )
    v300 = (
        ds_pl_12z["v"]
        .sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa")
        .sel(time=tg.time)
    )

    ds_forecast = xr.Dataset(
        {
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
    """Delete AIFS forecast files older than days_to_keep."""
    logging.info(
        "Garbage collection: deleting AIFS files older than %s days...",
        days_to_keep,
    )
    now = time.time()
    for f in OUT_DIR.glob("aifs_daily_forecast_*.nc"):
        if os.stat(f).st_mtime < now - (days_to_keep * 86400):
            try:
                f.unlink()
                logging.info("Deleted: %s", f.name)
            except Exception as exc:
                logging.error("Delete failed for %s: %s", f.name, exc)


def main() -> None:
    """Download, aggregate, regrid, and write the AIFS daily forecast."""
    setup_logging()
    start_time = time.time()

    if not ERA5_GRID_REF.exists():
        logging.error("ERA5 reference grid missing. Looked for: %s", ERA5_GRID_REF)
        sys.exit(1)

    if not WEIGHTS_FILE.exists():
        logging.error(
            "Interpolation matrix missing. Looked for: %s", WEIGHTS_FILE
        )
        sys.exit(1)

    try:
        sfc_file, pl_file, run_date, run_hour = download_aifs_gribs()
        ds_aligned = process_and_align_aifs(sfc_file, pl_file)

        out_path = OUT_DIR / f"aifs_daily_forecast_{run_date}_{run_hour:02d}z.nc"

        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in ds_aligned.data_vars
        }
        ds_aligned.to_netcdf(out_path, encoding=encoding)
        logging.info("SUCCESS. Live forecast saved: %s", out_path.name)

    except Exception as exc:
        logging.error("AIFS pipeline failed: %s", exc)

    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        purge_old_forecasts(days_to_keep=10)

    logging.info("Total duration: %.1f seconds.", time.time() - start_time)


if __name__ == "__main__":
    main()
