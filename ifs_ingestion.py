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
LOG_DIR = BASE_DIR / "Pipeline_Logs"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ERA5_GRID_REF = REF_DIR / "climatology_synoptics.nc"
WEIGHTS_FILE = REF_DIR / "regrid_weights_cdo.nc"

# Open-data 00Z/12Z files appear ~5h45 after base time (00Z ~05:45 UTC, 12Z ~17:45 UTC).
OPEN_DATA_LAG = timedelta(hours=5, minutes=45)
# Honour 120s only after every mirror 503s. Client must not sleep internally or
# AWS SlowDown on a missing 12Z index blocks Azure for minutes.
NATIVE_RETRY_AFTER_S = 120
CLIENT_MAX_RETRIES = 1
CLOUD_SOURCES = ("aws", "azure")
RETRIABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
# 3-hourly mx2t3/mn2t3 are not published beyond +72 h on the open-data stream.
ACCUM_STEP_MAX_H = 72
IFS_STEPS_PER_DAY = 8
CYCLE_ATTEMPTS = 6
FORECAST_GLOB = "ifs_daily_forecast_*.nc"


def setup_logging() -> None:
    """Log to stdout and to Pipeline_Logs (shared file when the batch sets it)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env_log = os.environ.get("ATMOPULSE_LOG_FILE")
    log_path = (
        Path(env_log)
        if env_log
        else LOG_DIR / f"ifs_ingestion_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    _purge_old_logs()


def _http_status(exc: BaseException):
    """Best-effort HTTP status from ecmwf-opendata / requests exceptions."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None) if resp is not None else None
    if code is not None:
        return int(code)
    for linked in (exc.__cause__, exc.__context__):
        if linked is not None and linked is not exc:
            nested = _http_status(linked)
            if nested is not None:
                return nested
    text = str(exc)
    for token in RETRIABLE_HTTP:
        if str(token) in text:
            return token
    return None


def _is_retriable(exc: BaseException) -> bool:
    status = _http_status(exc)
    if status in RETRIABLE_HTTP:
        return True
    name = type(exc).__name__.lower()
    return any(key in name for key in ("connection", "timeout", "chunked", "ssl"))


def _is_not_found(exc: BaseException) -> bool:
    if _http_status(exc) == 404:
        return True
    text = str(exc).lower()
    return "not found" in text or "cannot find index" in text


def _cycle_is_published(date_str: str, hour: int, now: datetime) -> bool:
    """Skip cycles that cannot yet exist on the ECMWF open-data rolling archive."""
    run_dt = datetime.strptime(date_str, "%Y%m%d").replace(
        hour=hour, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    ready_at = run_dt + OPEN_DATA_LAG
    if now < ready_at:
        logging.info(
            "Skipping %s %02dZ — open data expected after %s UTC (now %s UTC).",
            date_str,
            hour,
            ready_at.strftime("%H:%M"),
            now.strftime("%H:%M"),
        )
        return False
    return True


def _purge_old_logs(days_to_keep: int = 30) -> None:
    cutoff = time.time() - days_to_keep * 86400
    for f in LOG_DIR.glob("*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _exc_brief(exc: BaseException) -> str:
    return str(exc).split(" for url:")[0].split("?st=")[0][:180]


def _grib_ok(path) -> bool:
    try:
        p = Path(path)
        return p.is_file() and p.stat().st_size > 2048
    except OSError:
        return False


def _run_key(date_str: str, hour: int) -> tuple:
    return (str(date_str), int(hour))


def _newest_forecast_on_disk(pattern: str):
    newest = None
    newest_name = None
    for f in OUT_DIR.glob(pattern):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            key = (parts[-2], int(parts[-1].rstrip("zZ")))
        except ValueError:
            continue
        if newest is None or key > newest:
            newest = key
            newest_name = f.name
    return newest, newest_name


def _safe_unlink(*paths) -> None:
    """Drop partial GRIBs without racing a still-locked Windows handle."""
    for path in paths:
        p = Path(path)
        for attempt in range(4):
            try:
                if p.exists():
                    p.unlink()
                break
            except OSError as err:
                logging.warning(
                    "Partial file %s still locked (%s); waiting before retry",
                    p.name,
                    err,
                )
                time.sleep(1.0)
        else:
            logging.warning("Could not remove partial file %s; continuing", p)


def _make_opendata_client(source: str, model: str) -> Client:
    """Raise on the first 503 so the caller can fail over immediately."""
    return Client(
        source=source,
        model=model,
        resol="0p25",
        retry_after=NATIVE_RETRY_AFTER_S,
        use_server_retry_after=True,
        maximum_retries=CLIENT_MAX_RETRIES,
    )


def download_ifs_gribs():
    """Retrieve the newest published IFS run; never fall back past a confirmed cycle."""
    # 3-hourly steps; omit step=0 to prevent cfgrib duplication on accumulations.
    steps_3h = list(range(3, 120, 3))
    steps_accum = [s for s in steps_3h if s <= ACCUM_STEP_MAX_H]

    target_sfc = str(TMP_DIR / "ifs_sfc.grib")
    target_accum = str(TMP_DIR / "ifs_sfc_accum.grib")
    target_pl = str(TMP_DIR / "ifs_pl.grib")

    now = datetime.now(timezone.utc)
    on_disk, on_disk_name = _newest_forecast_on_disk(FORECAST_GLOB)
    if on_disk_name:
        logging.info("Newest IFS file already on disk: %s", on_disk_name)

    seen = set()
    candidates = []
    for i in range(6):
        dt = now - timedelta(hours=12 * i)
        hour = 12 if dt.hour >= 12 else 0
        key = (dt.strftime("%Y%m%d"), hour)
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    for date_str, hour in candidates:
        this_key = _run_key(date_str, hour)
        if on_disk and this_key < on_disk:
            logging.info(
                "Skipping %s %02dZ — older than live file %s.",
                date_str,
                hour,
                on_disk_name,
            )
            continue
        if not _cycle_is_published(date_str, hour, now):
            continue

        confirmed = False
        have_sfc = False
        have_accum = False
        have_pl = False
        for attempt in range(CYCLE_ATTEMPTS):
            for src in CLOUD_SOURCES:
                logging.info("Connecting to cloud mirror: %s...", src.upper())
                try:
                    client = _make_opendata_client(src, "ifs")
                except Exception as exc:
                    logging.warning(
                        "Failed to initialize Client for %s: %s", src.upper(), exc
                    )
                    continue

                logging.info("Checking run %s %02dZ on %s...", date_str, hour, src.upper())
                try:
                    if not have_sfc:
                        client.retrieve(
                            date=date_str,
                            time=hour,
                            type="fc",
                            stream="oper",
                            levtype="sfc",
                            param=["2t", "msl"],
                            step=steps_3h,
                            target=target_sfc,
                        )
                        have_sfc = True
                        confirmed = True
                    if not have_accum:
                        client.retrieve(
                            date=date_str,
                            time=hour,
                            type="fc",
                            stream="oper",
                            levtype="sfc",
                            param=["mx2t3", "mn2t3"],
                            step=steps_accum,
                            target=target_accum,
                        )
                        have_accum = True
                    if not have_pl:
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
                        have_pl = True
                    return target_sfc, target_accum, target_pl, date_str, hour
                except Exception as exc:
                    status = _http_status(exc)
                    if _is_not_found(exc) and not have_sfc:
                        _safe_unlink(target_sfc, target_accum, target_pl)
                        logging.warning(
                            "Run %s %02dZ not on %s (HTTP 404).",
                            date_str,
                            hour,
                            src.upper(),
                        )
                        continue
                    if have_sfc:
                        confirmed = True
                        logging.warning(
                            "Keeping completed GRIBs after %s on %s (%s); retrying remaining files.",
                            status or type(exc).__name__,
                            src.upper(),
                            _exc_brief(exc),
                        )
                        if not have_accum:
                            _safe_unlink(target_accum)
                        if not have_pl:
                            _safe_unlink(target_pl)
                    else:
                        _safe_unlink(target_sfc, target_accum, target_pl)
                    if _is_retriable(exc) or confirmed:
                        logging.warning(
                            "HTTP %s on %s for %s %02dZ — failing over immediately.",
                            status or type(exc).__name__,
                            src.upper(),
                            date_str,
                            hour,
                        )
                        continue
                    logging.warning(
                        "Run %s %02dZ failed on %s: %s",
                        date_str,
                        hour,
                        src.upper(),
                        _exc_brief(exc),
                    )

            if confirmed:
                logging.info(
                    "Cycle %s %02dZ is on the archive; waiting %ss then retrying remaining files (attempt %s/%s).",
                    date_str,
                    hour,
                    NATIVE_RETRY_AFTER_S,
                    attempt + 1,
                    CYCLE_ATTEMPTS,
                )
                time.sleep(NATIVE_RETRY_AFTER_S)
                continue
            break

        if confirmed:
            raise RuntimeError(
                f"IFS {date_str} {hour:02d}Z is on the archive but the download "
                "did not finish. Not falling back to an older cycle."
            )

    raise RuntimeError("No IFS run found across all configured mirrors.")


def apply_conservative_weights(ds_source, weights_file, ds_target_grid):
    """Regrid with a precomputed CDO matrix and fracarea normalisation.

    Sparse CSR is applied once per variable as a 2-D matmul
    ``(n_dst, n_src) @ (n_src, n_time)`` so the Python timestep loop and
    ``np.stack`` copies are avoided.
    """
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
        remap = np.asarray(ds_w["remap_matrix"].values[:, 0], dtype=np.float32)
        dst_addr = np.asarray(ds_w["dst_address"].values, dtype=np.int32) - 1
        src_addr = np.asarray(ds_w["src_address"].values, dtype=np.int32) - 1
        shape = (int(ds_w.sizes["dst_grid_size"]), int(ds_w.sizes["src_grid_size"]))

    weights = sps.csr_matrix(
        (remap, (dst_addr, src_addr)), shape=shape, dtype=np.float32
    )
    del remap, dst_addr, src_addr

    n_lat = int(ds_target_grid.sizes["latitude"])
    n_lon = int(ds_target_grid.sizes["longitude"])
    n_time = int(ds_source_cropped.sizes["time"])

    out_data = {}
    for var in ds_source_cropped.data_vars:
        # Own the buffer so in-place NaN fill cannot alias xarray/NetCDF memory.
        src = np.array(ds_source_cropped[var].values, dtype=np.float32, copy=True)
        flat = src.reshape(n_time, -1)
        valid = np.isfinite(flat)
        flat[~valid] = np.float32(0)

        # C-contiguous (n_time, n_src) → F-contiguous (n_src, n_time) view.
        y_num = weights @ flat.T
        y_den = weights @ valid.astype(np.float32).T
        del src, flat, valid

        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(y_num, y_den, out=y_num, where=y_den > 0)
        y_num[y_den <= 0] = np.nan
        del y_den

        out_data[var] = (
            ("time", "latitude", "longitude"),
            np.ascontiguousarray(
                y_num.T.reshape(n_time, n_lat, n_lon), dtype=np.float32
            ),
        )
        del y_num

    return xr.Dataset(
        out_data,
        coords={
            "time": ds_source_cropped.time,
            "latitude": ds_target_grid.latitude,
            "longitude": ds_target_grid.longitude,
        },
    )


def _swap_to_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.swap_dims({"step": "valid_time"})
    if "time" in ds.coords:
        ds = ds.drop_vars("time")
    return ds.rename({"valid_time": "time"})


def _load_grib_field(path, short_name: str) -> xr.DataArray:
    """Open one cfgrib hypercube, load into RAM, then release the file handle."""
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}, "indexpath": ""},
    )
    try:
        ds = _swap_to_time(ds)
        name = short_name if short_name in ds.data_vars else list(ds.data_vars)[0]
        return ds[name].load()
    finally:
        ds.close()


def _etccdi_complete_days(time_coord, steps_per_day: int) -> xr.DataArray:
    """Right-closed 00–00 UTC bins with exactly ``steps_per_day`` timestamps."""
    ones = xr.DataArray(
        np.ones(time_coord.size, dtype=np.int16),
        coords={"time": time_coord},
        dims="time",
    )
    counts = (
        ones.resample(time="1D", closed="right", label="left").sum().fillna(0)
    )
    complete = counts.where(counts == steps_per_day, drop=True)
    logging.info(
        "ETCCDI 00-00 UTC completeness: %s of %s days have exactly %s steps.",
        int(complete.size),
        int(counts.size),
        steps_per_day,
    )
    return complete.time


def process_and_align_ifs(sfc_file, accum_file, pl_file):
    """Aggregate daily 00–00 extremes / 12Z synoptics and regrid onto ERA5."""
    logging.info("Loading GRIBs and resolving xarray 'time' conflicts...")

    t2m = _load_grib_field(sfc_file, "2t")
    msl = _load_grib_field(sfc_file, "msl")
    t_pl = _load_grib_field(pl_file, "t")
    z_pl = _load_grib_field(pl_file, "z")
    u_pl = _load_grib_field(pl_file, "u")
    v_pl = _load_grib_field(pl_file, "v")

    try:
        mx2t3 = _load_grib_field(accum_file, "mx2t3")
        mn2t3 = _load_grib_field(accum_file, "mn2t3")
    except Exception as exc:
        logging.warning(
            "mx2t3/mn2t3 unavailable (%s); TX/TN will use instantaneous 2t.",
            exc,
        )
        mx2t3 = mn2t3 = None

    tg_inst_c = t2m - 273.15
    mslp_hpa = msl / 100.0

    logging.info("Aggregating strict 00-00 UTC T-extremes and 12Z synoptics...")

    resample_kw = dict(time="1D", closed="right", label="left")
    daily_tx_inst = tg_inst_c.resample(**resample_kw).max()
    daily_tn_inst = tg_inst_c.resample(**resample_kw).min()
    daily_tg = tg_inst_c.resample(**resample_kw).mean()

    # Splice: 3-hourly accumulations through +72 h, then instantaneous 2t.
    if mx2t3 is not None:
        daily_tx_accum = (mx2t3 - 273.15).resample(**resample_kw).max()
        daily_tn_accum = (mn2t3 - 273.15).resample(**resample_kw).min()
        tx_combined = daily_tx_accum.combine_first(daily_tx_inst)
        tn_combined = daily_tn_accum.combine_first(daily_tn_inst)
    else:
        tx_combined = daily_tx_inst
        tn_combined = daily_tn_inst

    complete_times = _etccdi_complete_days(tg_inst_c.time, IFS_STEPS_PER_DAY)
    tx = tx_combined.sel(time=complete_times)
    tn = tn_combined.sel(time=complete_times)
    tg = daily_tg.sel(time=complete_times)

    ds_pl_12z_t = t_pl.sel(time=t_pl.time.dt.hour == 12)
    ds_pl_12z_z = z_pl.sel(time=z_pl.time.dt.hour == 12)
    ds_pl_12z_u = u_pl.sel(time=u_pl.time.dt.hour == 12)
    ds_pl_12z_v = v_pl.sel(time=v_pl.time.dt.hour == 12)
    mslp_12z = mslp_hpa.sel(time=mslp_hpa.time.dt.hour == 12)

    mslp = mslp_12z.resample(time="1D").first().sel(time=tx.time)
    z500 = (
        ds_pl_12z_z.sel(isobaricInhPa=500)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa", errors="ignore")
        .sel(time=tx.time)
    )
    t850 = (
        (ds_pl_12z_t.sel(isobaricInhPa=850) - 273.15)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa", errors="ignore")
        .sel(time=tx.time)
    )
    u300 = (
        ds_pl_12z_u.sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa", errors="ignore")
        .sel(time=tx.time)
    )
    v300 = (
        ds_pl_12z_v.sel(isobaricInhPa=300)
        .resample(time="1D")
        .first()
        .drop_vars("isobaricInhPa", errors="ignore")
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

    exit_code = 0
    try:
        sfc_file, accum_file, pl_file, run_date, run_hour = download_ifs_gribs()
        existing, existing_name = _newest_forecast_on_disk(FORECAST_GLOB)
        new_key = _run_key(run_date, run_hour)
        if existing and new_key < existing:
            logging.warning(
                "Refusing to write ifs_daily_forecast_%s_%02dz.nc; newer file %s is already live.",
                run_date,
                run_hour,
                existing_name,
            )
            return

        ds_aligned = process_and_align_ifs(sfc_file, accum_file, pl_file)

        out_path = OUT_DIR / f"ifs_daily_forecast_{run_date}_{run_hour:02d}z.nc"

        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in ds_aligned.data_vars
        }
        ds_aligned.to_netcdf(out_path, encoding=encoding)
        logging.info("SUCCESS. Live forecast saved: %s", out_path.name)

    except Exception as exc:
        logging.error("IFS pipeline failed: %s", exc)
        exit_code = 1
    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        purge_old_forecasts(days_to_keep=10)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
