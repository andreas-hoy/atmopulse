"""
AtmoPulse Wave Status Batch Pre-computation (batch_precompute_waves.py)

Offline batch script that builds/updates the Map Tracker's "Wave tracking"
raster: for every ERA5 0.25 degree cell, the Kyselý heatwave/coldwave
intensity accumulated from that cell's event start THROUGH each calendar
day, for TX/TN/TG/T850, both directions (heat/cold), both severity levels
(Strong/Extreme), and both reference epochs (A = 1961-1990, B = 1996-2025).

This is a spatial analog to era5_climatology_builder.py / the anomaly
climatology stores, NOT a write into era5_master_time_series.zarr (that
Zarr is chunked for point time-series reads -- time=-1, 10x10 lat/lon
tiles -- useless for a Europe-wide map) and NOT a new observable mixed
into era5_master_daily_YYYY.nc.

Detection: `backend_waves.detect_kysely_waves_grid` -- the grid-vectorized
twin of the Point Wavogram's `_detect_kysely_waves`, verified bit-exact
against it (see test_wave_grid_detection*.py). One Python loop over
~365 days per season-year; every step inside it is a full-grid numpy
vector op, NEVER a per-cell Python loop. Thresholds come from the 2D
JJA/DJF percentile fields already written by era5_climatology_builder.py
(`{var}_jja_p{75,90,95}_{A|B}`, `{var}_djf_p{5,10,25}_{A|B}`) -- never the
DOY percentile fields, and never recomputed from a point series.

Season windows (must match the point algorithm exactly):
  - Heat: calendar year (1 Jan - 31 Dec).
  - Cold: "winter year" (1 Jul - 30 Jun), then SPLICED back into calendar-
    year dates before writing -- a calendar year's cold fields are built
    from the second half of winter_year (Y-1) [Jan-Jun] and the first half
    of winter_year Y [Jul-Dec]. This also naturally keeps heat events that
    cross 1 Jan SPLIT and cold events that cross 1 Jan intact as ONE event
    (the splice is display/storage bookkeeping; detection itself never sees
    the calendar-year boundary for cold).

Output store: ERA5_ClimateTool/Wave_Status/wave_status.zarr (config.
WAVE_STATUS_ZARR), dims (epoch: 2, valid_time, latitude, longitude), one
data variable per (var, direction, level) via config.wave_map_field(), e.g.
"tx_heat_strong_int" (float32 K.days, running intensity-to-date; 0 = not in
a wave) and "tx_heat_strong_dur" (int16, running duration-to-date in days;
0 = not in a wave). Chunks: config.WAVE_MAP_CHUNKS -- one (epoch, day) is
one chunk, so a map load touches exactly one chunk per field.

Usage
-----
    python batch_precompute_waves.py                 # incremental: last N years (default 3)
    python batch_precompute_waves.py --full           # full historical rebuild (1940 -> latest)
    python batch_precompute_waves.py --years 10       # last 10 calendar years
    python batch_precompute_waves.py --start-year 2015 --end-year 2024
    python batch_precompute_waves.py --full --batch-years 5 --max-retries 3

Robustness for long/unattended (e.g. overnight, full 1940->latest) runs:
  - RAM-bounded batching: all 4 variables' cubes for only `--batch-years`
    calendar years (default 5) are held in RAM at once, never the whole
    requested range -- a full 1940-latest x 4-variable run would otherwise
    need ~20-25 GB of raw cubes simultaneously, which does not fit on a
    16 GB machine.
  - Checkpoint/resume: after each calendar year is successfully written,
    its year number is flushed to a checkpoint JSON file (default next to
    the zarr store, `_precompute_checkpoint.json`; override with
    `--checkpoint`). Re-running the SAME command later (same variables,
    same/superset date range) skips already-completed years instead of
    reprocessing them -- a crash, OOM, reboot, or manual Ctrl+C loses at
    most the one calendar year that was in progress.
  - Per-year retry: a failure processing one calendar year is retried
    (`--max-retries`, default 3, with a short backoff) before the whole
    run aborts; on abort, the checkpoint reflects everything completed so
    far so the next invocation resumes right after it.
  - The expensive whole-store `_finalize_epoch_dim()` rewrite runs exactly
    ONCE, only after every requested year (across all batches) is written
    -- not once per batch -- and is itself idempotent/resumable (a no-op
    if the store is already finalized).

IMPORTANT -- always run with the FULL variable set for a given date range:
every requested variable's fields for a calendar year are combined into
ONE Dataset and written with ONE `to_zarr` call for that year (see
`_write_calendar_year_multi`), specifically so `to_zarr(mode='a',
append_dim='valid_time')` -- which unconditionally grows the store's
shared valid_time axis -- is only ever invoked once per calendar year,
ever. Running this script twice for the SAME date range with two
DIFFERENT, non-overlapping variable subsets (e.g. TX+TN today, TG+T850
next month, both for 2023-2026) would silently duplicate that range's
valid_time axis on the second run, corrupting the store. If a variable
needs to be added retroactively to an already-covered date range, rebuild
that whole range with the full variable set (`--start-year`/`--end-year`
covering it), not a bolt-on run for just the new variable.

Live/forecast (current season, last ~6 days + forecast) is NOT written to
this store by this script -- the Map Tracker merges that on-the-fly at
request time, the same way IFS/AIFS overlays live temperature maps (see
backend_wave_map.py). This script only ever writes the settled archive.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr

from backend_waves import (
    _param_var,
    _wave_season_end,
    _wave_season_origin,
    detect_kysely_waves_grid,
)
from config import (
    DATA_ROOT,
    MAP_WAVE_VARS,
    WAVE_MAP_CHUNKS,
    WAVE_STATUS_ZARR,
    wave_map_field,
)

MASTER_BATCHES_DIR = DATA_ROOT / "Master_Batches"
CLIM_FILE = DATA_ROOT / "Reference_Climatology" / "climatology_reference_complete.nc"
if not CLIM_FILE.exists():
    CLIM_FILE = DATA_ROOT / "Reference_Climatology" / "climatology_reference.nc"

EPOCHS = ("A", "B")
LEVELS = ("Strong", "Extreme")
DEFAULT_RECENT_YEARS = 3
EARLIEST_ARCHIVE_YEAR = 1940
DEFAULT_BATCH_YEARS = 5           # years of all-variable cubes held in RAM at once
DEFAULT_MAX_RETRIES_PER_YEAR = 3
CHECKPOINT_PATH = WAVE_STATUS_ZARR.parent / "_precompute_checkpoint.json"


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _available_years() -> list[int]:
    years = []
    for path in sorted(MASTER_BATCHES_DIR.glob("era5_master_daily_*.nc")):
        try:
            years.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(years)


def _harmonize(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    if "expver" in ds.dims:
        ds = ds.dropna(dim="expver", how="all")
        if "expver" in ds.dims:
            ds = ds.isel(expver=0, drop=True)
    for name in ("expver", "number", "heightAboveGround", "meanSea", "isobaricInhPa"):
        if name in ds.variables or name in ds.coords:
            ds = ds.drop_vars(name, errors="ignore")
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        ds = ds.rename({"mx2t": "tx"})
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        ds = ds.rename({"mn2t": "tn"})
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def _load_variable_cube(var_key: str, years: list[int]):
    """Load one variable (tx/tn/tg/t850), all requested years, fully into
    RAM as (values, time_index, lats, lons) -- same "one variable at a
    time" RAM strategy as era5_climatology_builder.py, so peak memory is
    one variable's worth of the requested year range, not the whole
    4-variable, 85-year archive at once."""
    paths = [MASTER_BATCHES_DIR / f"era5_master_daily_{y}.nc" for y in years]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    frames_vals, frames_time = [], []
    lats = lons = None
    for path in paths:
        ds = None
        try:
            ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False).pipe(_harmonize)
            if var_key not in ds.data_vars:
                continue
            da = ds[var_key]
            if lats is None:
                lats = np.asarray(ds["latitude"].values)
                lons = np.asarray(ds["longitude"].values)
            frames_vals.append(np.asarray(da.values, dtype=np.float32))
            frames_time.append(pd.to_datetime(ds["valid_time"].values))
        except Exception as exc:
            _log(f"  WARNING: could not load {var_key} from {path.name}: {exc!r}")
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
    if not frames_vals:
        return None
    values = np.concatenate(frames_vals, axis=0)
    times = pd.DatetimeIndex(np.concatenate([np.asarray(t) for t in frames_time]))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    times = times.normalize()
    order = np.argsort(times.values)
    times = times[order]
    values = values[order]
    # Auto-detect Kelvin (matches backend_maps._to_celsius).
    finite = values[np.isfinite(values)]
    if finite.size and float(np.mean(finite)) > 100:
        values = values - 273.15
    return values, times, lats, lons


def _season_slice(values, times, lats, lons, start, end):
    """(n_days, nlat, nlon) float array for [start, end] inclusive, real
    calendar days, reindexed onto the full date range so missing/absent
    days are NaN (never filled) -- matches the point pipeline exactly."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    full_dates = pd.date_range(start, end, freq="D")
    out = np.full((len(full_dates), len(lats), len(lons)), np.nan, dtype=np.float32)
    if values is not None:
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            sub_times = times[mask]
            sub_vals = values[mask]
            pos = full_dates.get_indexer(sub_times)
            keep = pos >= 0
            out[pos[keep]] = sub_vals[keep]
    return out, full_dates


def _load_climatology() -> xr.Dataset | None:
    if not CLIM_FILE.exists():
        return None
    return xr.open_dataset(CLIM_FILE, engine="netcdf4")


def _threshold_fields(clim, var_key: str, epoch: str, is_warm: bool, level: str, lats, lons):
    """(p_thresh, p_drop) 2D arrays on the master grid, from the JJA/DJF
    percentile fields (never DOY percentiles)."""
    season = "jja" if is_warm else "djf"
    if is_warm:
        p_t_ext, p_d_ext = f"{var_key}_{season}_p95_{epoch}", f"{var_key}_{season}_p90_{epoch}"
        p_t_str, p_d_str = f"{var_key}_{season}_p90_{epoch}", f"{var_key}_{season}_p75_{epoch}"
    else:
        p_t_ext, p_d_ext = f"{var_key}_{season}_p5_{epoch}", f"{var_key}_{season}_p10_{epoch}"
        p_t_str, p_d_str = f"{var_key}_{season}_p10_{epoch}", f"{var_key}_{season}_p25_{epoch}"
    key_t, key_d = (p_t_ext, p_d_ext) if level == "Extreme" else (p_t_str, p_d_str)
    if key_t not in clim.variables or key_d not in clim.variables:
        return None, None
    reg = clim[[key_t, key_d]].reindex(latitude=lats, longitude=lons, method="nearest")
    return (
        np.asarray(reg[key_t].values, dtype=np.float64),
        np.asarray(reg[key_d].values, dtype=np.float64),
    )


def _detect_for_window(values, times, lats, lons, start, end, clim, var_key, is_warm):
    """All (epoch, level) results for one season window, keyed
    (epoch, level) -> (intensity[n_days,nlat,nlon], duration[n_days,nlat,nlon])."""
    temps, dates = _season_slice(values, times, lats, lons, start, end)
    results = {}
    for epoch in EPOCHS:
        for level in LEVELS:
            p_thresh, p_drop = _threshold_fields(clim, var_key, epoch, is_warm, level, lats, lons)
            if p_thresh is None:
                continue
            intensity, duration = detect_kysely_waves_grid(temps, p_thresh, p_drop, is_warm)
            results[(epoch, level)] = (intensity, duration)
    return results, dates


def _slice_by_dates(intensity, duration, dates, want_start, want_end):
    want_start, want_end = pd.Timestamp(want_start).normalize(), pd.Timestamp(want_end).normalize()
    mask = (dates >= want_start) & (dates <= want_end)
    return intensity[mask], duration[mask], dates[mask]


def _write_calendar_year_multi(
    calendar_year: int, per_var_results: dict, lats, lons, dates, store_started: bool,
) -> bool:
    """Merge ALL requested variables' heat + spliced-cold results for ONE
    calendar year into a SINGLE Dataset (every var x direction x level x
    epoch field) and write it with exactly ONE `to_zarr` call for that
    year.

    This must stay year-outer / variable-inner (one combined write per
    year), NOT variable-outer / year-inner: `to_zarr(mode='a',
    append_dim='valid_time')` unconditionally GROWS the valid_time
    dimension for the whole store, regardless of which variables are in
    the dataset being appended. Writing TX for 2023-2026 and THEN writing
    TN for 2023-2026 as two separate variable-outer passes -- each with
    its own "mode='w' for my first year, else mode='a'" decision -- would
    make TN's very first calendar year (2023) append onto a store whose
    valid_time already covers 2023-2026 from TX, silently DUPLICATING the
    time axis (2023 written twice) instead of adding TN's fields at the
    SAME, already-existing 2023 timestamps. Combining every variable's
    fields for a given calendar year into one Dataset and writing it once
    avoids this class of bug entirely: every calendar year is written to
    the store exactly once, ever, regardless of how many variables are
    requested together or across separate incremental runs.

    `epoch` is folded into the variable name (e.g. "tx_heat_strong_int__A")
    on write and unstacked back into a real `epoch` dimension by
    `_finalize_epoch_dim` once, after all years are written -- appending
    along two dims (valid_time AND epoch) chunk-by-chunk with
    `to_zarr(mode='a')` is not supported.
    """
    data_vars = {}
    for var_key, (heat_by_epoch_level, cold_by_epoch_level) in per_var_results.items():
        for level in LEVELS:
            for epoch in EPOCHS:
                for direction, results in (("heat", heat_by_epoch_level), ("cold", cold_by_epoch_level)):
                    is_warm = direction == "heat"
                    int_field = wave_map_field(var_key.upper(), is_warm, level, kind="int")
                    dur_field = wave_map_field(var_key.upper(), is_warm, level, kind="dur")
                    pair = results.get((epoch, level))
                    if pair is None:
                        intensity = np.zeros((len(dates), len(lats), len(lons)), dtype=np.float32)
                        duration = np.zeros((len(dates), len(lats), len(lons)), dtype=np.int16)
                    else:
                        intensity, duration = pair
                    data_vars[f"{int_field}__{epoch}"] = (("valid_time", "latitude", "longitude"), intensity)
                    data_vars[f"{dur_field}__{epoch}"] = (("valid_time", "latitude", "longitude"), duration)

    ds_year = xr.Dataset(data_vars, coords={"valid_time": dates, "latitude": lats, "longitude": lons})
    chunk_spec = {
        "valid_time": 1,
        "latitude": WAVE_MAP_CHUNKS.get("latitude", -1),
        "longitude": WAVE_MAP_CHUNKS.get("longitude", -1),
    }
    ds_year = ds_year.chunk(chunk_spec)

    WAVE_STATUS_ZARR.parent.mkdir(parents=True, exist_ok=True)
    if not store_started:
        ds_year.to_zarr(WAVE_STATUS_ZARR, mode="w", compute=True, consolidated=True)
    else:
        ds_year.to_zarr(WAVE_STATUS_ZARR, mode="a", append_dim="valid_time", compute=True, consolidated=True)
    return True


def _load_all_variable_cubes(var_codes: list[str], years: list[int]) -> dict:
    """{var_code: (var_key, values, times, lats, lons) or None}, one
    variable fully into RAM at a time (never all of them at once), padded
    by 1 year on each side for the winter-year cold splice."""
    cubes = {}
    pad_years = list(range(years[0] - 1, years[-1] + 2))
    for var_code in var_codes:
        var_key = _param_var(var_code)
        _log(f"Loading '{var_key}' ({var_code}) for {years[0]}-{years[-1]} into RAM...")
        loaded = _load_variable_cube(var_key, pad_years)
        if loaded is None:
            _log(f"  SKIPPED {var_code}: no data on disk for this range.")
            cubes[var_code] = None
            continue
        values, times, lats, lons = loaded
        _log(f"  loaded {values.shape[0]} day(s), grid {values.shape[1]}x{values.shape[2]}.")
        cubes[var_code] = (var_key, values, times, lats, lons)
    return cubes


def _detect_year_for_var(var_key, values, times, lats, lons, calendar_year, clim, winter_cache: dict):
    """One calendar year's heat_by_epoch_level / cold_by_epoch_level /
    heat_dates for one variable, using (and updating) `winter_cache`
    (keyed by var_key) to compute each winter-year's cold detection
    exactly once and reuse it as both calendar year Y's Jul-Dec half and
    calendar year Y+1's Jan-Jun half."""
    heat_start, heat_end = _wave_season_origin(calendar_year, True), _wave_season_end(calendar_year, True)
    heat_results, heat_dates = _detect_for_window(values, times, lats, lons, heat_start, heat_end, clim, var_key, True)

    winter_start, winter_end = _wave_season_origin(calendar_year, False), _wave_season_end(calendar_year, False)
    this_winter_results, this_winter_dates = _detect_for_window(
        values, times, lats, lons, winter_start, winter_end, clim, var_key, False,
    )

    prev_winter_results, prev_winter_dates = winter_cache.get(var_key, (None, None))
    if prev_winter_results is None:
        prev_start, prev_end = _wave_season_origin(calendar_year - 1, False), _wave_season_end(calendar_year - 1, False)
        prev_winter_results, prev_winter_dates = _detect_for_window(
            values, times, lats, lons, prev_start, prev_end, clim, var_key, False,
        )

    cold_by_epoch_level = {}
    jan1, jun30 = pd.Timestamp(year=calendar_year, month=1, day=1), pd.Timestamp(year=calendar_year, month=6, day=30)
    jul1, dec31 = pd.Timestamp(year=calendar_year, month=7, day=1), pd.Timestamp(year=calendar_year, month=12, day=31)
    for key in set(prev_winter_results) | set(this_winter_results):
        if key in prev_winter_results:
            i1, d1, _ = _slice_by_dates(*prev_winter_results[key], prev_winter_dates, jan1, jun30)
        else:
            i1 = d1 = None
        if key in this_winter_results:
            i2, d2, _ = _slice_by_dates(*this_winter_results[key], this_winter_dates, jul1, dec31)
        else:
            i2 = d2 = None
        if i1 is None and i2 is None:
            continue
        cold_by_epoch_level[key] = (
            np.concatenate([a for a in (i1, i2) if a is not None], axis=0),
            np.concatenate([a for a in (d1, d2) if a is not None], axis=0),
        )

    winter_cache[var_key] = (this_winter_results, this_winter_dates)
    return heat_results, cold_by_epoch_level, heat_dates


def _finalize_epoch_dim() -> None:
    """Reshape the "__A"/"__B" variable-name suffixes used during the
    incremental per-year build into a real `epoch` coordinate dimension,
    consolidating metadata. Safe to run repeatedly (no-op once already
    finalized -- checks for the suffix before doing anything)."""
    ds = xr.open_zarr(WAVE_STATUS_ZARR, consolidated=True)
    suffixed = [v for v in ds.data_vars if v.endswith("__A") or v.endswith("__B")]
    if not suffixed:
        ds.close()
        return
    _log(f"Finalizing epoch dimension for {len(suffixed)} variable(s)...")
    base_names = sorted({v.rsplit("__", 1)[0] for v in suffixed})
    data_vars = {}
    for base in base_names:
        a = ds[f"{base}__A"] if f"{base}__A" in ds.data_vars else None
        b = ds[f"{base}__B"] if f"{base}__B" in ds.data_vars else None
        pieces = [x for x in (a, b) if x is not None]
        combined = xr.concat(pieces, dim=pd.Index(["A", "B"][: len(pieces)], name="epoch"))
        data_vars[base] = combined
    out = xr.Dataset(data_vars, coords=ds.coords)
    out = out.chunk({
        "epoch": WAVE_MAP_CHUNKS.get("epoch", 1),
        "valid_time": 1,
        "latitude": WAVE_MAP_CHUNKS.get("latitude", -1),
        "longitude": WAVE_MAP_CHUNKS.get("longitude", -1),
    })
    ds.close()
    del ds
    gc.collect()

    tmp_path = WAVE_STATUS_ZARR.parent / (WAVE_STATUS_ZARR.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    out.to_zarr(tmp_path, mode="w", compute=True, consolidated=True)
    del out
    gc.collect()

    # Swap tmp -> live via a rename-away-then-in dance, NOT delete-then-rename:
    # on Windows, `shutil.rmtree` of a directory that dask/zarr just had file
    # handles into can report success while the directory is still marked
    # "pending delete" for a moment, which makes an immediately-following
    # `rename(tmp, WAVE_STATUS_ZARR)` fail with WinError 5 (Access is denied)
    # even though the target genuinely no longer exists moments later. Instead,
    # rename the (now-superseded) live store out of the way first, rename tmp
    # into its place, then delete the old backup at leisure with retries.
    backup_path = WAVE_STATUS_ZARR.parent / (WAVE_STATUS_ZARR.name + ".bak")
    if backup_path.exists():
        _rmtree_with_retry(backup_path)
    if WAVE_STATUS_ZARR.exists():
        _rename_with_retry(WAVE_STATUS_ZARR, backup_path)
    _rename_with_retry(tmp_path, WAVE_STATUS_ZARR)
    if backup_path.exists():
        _rmtree_with_retry(backup_path)
    _log("Epoch dimension finalized.")


def _rename_with_retry(src: Path, dst: Path, attempts: int = 8, delay_s: float = 0.5) -> None:
    last_err = None
    for i in range(attempts):
        try:
            src.rename(dst)
            return
        except OSError as exc:
            last_err = exc
            gc.collect()
            time.sleep(delay_s * (i + 1))
    raise last_err


def _rmtree_with_retry(path: Path, attempts: int = 8, delay_s: float = 0.5) -> None:
    last_err = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_err = exc
            gc.collect()
            time.sleep(delay_s * (i + 1))
    raise last_err


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        _log(f"WARNING: could not read checkpoint {path} ({exc!r}); starting fresh.")
        return {}


def _save_checkpoint(path: Path, data: dict) -> None:
    """Atomic write (tmp + replace) so a crash mid-write never leaves a
    half-written / corrupt checkpoint file that a resumed run can't parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def run(
    *,
    full: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
    recent_years: int = DEFAULT_RECENT_YEARS,
    batch_years: int = DEFAULT_BATCH_YEARS,
    checkpoint_path: Path | None = None,
    max_retries_per_year: int = DEFAULT_MAX_RETRIES_PER_YEAR,
) -> None:
    _log("=" * 70)
    _log("AtmoPulse Wave Status Batch Pre-computation")
    _log("=" * 70)

    available = _available_years()
    if not available:
        _log(f"ERROR: no era5_master_daily_*.nc files found under {MASTER_BATCHES_DIR}")
        sys.exit(1)

    if start_year is not None or end_year is not None:
        y0 = start_year or available[0]
        y1 = end_year or available[-1]
    elif full:
        y0, y1 = available[0], available[-1]
    else:
        y1 = available[-1]
        y0 = max(available[0], y1 - recent_years + 1)

    years = [y for y in range(y0, y1 + 1) if y in set(available)]
    if not years:
        _log(f"ERROR: no requested years ({y0}-{y1}) found on disk.")
        sys.exit(1)
    _log(f"Processing calendar years: {years[0]}-{years[-1]} ({len(years)} year(s)) for {MAP_WAVE_VARS}")

    clim = _load_climatology()
    if clim is None:
        _log("ABORTED: reference climatology not found. Run era5_climatology_builder.py first.")
        sys.exit(1)

    variables = list(MAP_WAVE_VARS)
    ckpt_path = checkpoint_path or CHECKPOINT_PATH
    ckpt = _load_checkpoint(ckpt_path)

    ckpt_vars = ckpt.get("variables")
    if ckpt_vars and set(ckpt_vars) != set(variables):
        _log(
            f"WARNING: checkpoint at {ckpt_path} was tracking variables {ckpt_vars}, "
            f"this run requests {variables}. Resetting the checkpoint's completed-years "
            f"tracking for a fresh start (existing data already on disk in the zarr store "
            f"is NOT deleted -- but per the module docstring, do not mix variable subsets "
            f"across runs for OVERLAPPING date ranges; rebuild that range with the full "
            f"variable set instead)."
        )
        ckpt = {}

    completed_years = set(ckpt.get("completed_years", []))
    store_started = ckpt.get("store_started", WAVE_STATUS_ZARR.exists())
    remaining_years = [y for y in years if y not in completed_years]

    if not remaining_years:
        _log(f"All {len(years)} requested year(s) already completed per checkpoint {ckpt_path}.")
    else:
        _log(
            f"Remaining years to process: {remaining_years[0]}-{remaining_years[-1]} "
            f"({len(remaining_years)} year(s)), batch size {batch_years} "
            f"({len(completed_years)} year(s) already done per checkpoint)."
        )

    winter_cache: dict = {}  # var_key -> (this_winter_results, this_winter_dates), carried across years/batches

    for batch_start in range(0, len(remaining_years), batch_years):
        batch = remaining_years[batch_start:batch_start + batch_years]
        _log(f"--- Loading batch {batch[0]}-{batch[-1]} ({len(batch)} year(s)) ---")
        cubes = _load_all_variable_cubes(variables, batch)
        loaded_vars = [vc for vc, c in cubes.items() if c is not None]
        if not loaded_vars:
            _log(f"  SKIPPING batch {batch[0]}-{batch[-1]}: no data on disk for any requested variable.")
            del cubes
            gc.collect()
            continue

        for calendar_year in batch:
            attempt = 0
            while True:
                attempt += 1
                try:
                    t0 = time.time()
                    per_var_results = {}
                    heat_dates = None
                    lats = lons = None
                    for var_code in loaded_vars:
                        var_key, values, times, v_lats, v_lons = cubes[var_code]
                        lats, lons = v_lats, v_lons  # identical master grid for every variable
                        heat_results, cold_by_epoch_level, heat_dates = _detect_year_for_var(
                            var_key, values, times, v_lats, v_lons, calendar_year, clim, winter_cache,
                        )
                        per_var_results[var_key] = (heat_results, cold_by_epoch_level)

                    store_started = _write_calendar_year_multi(
                        calendar_year, per_var_results, lats, lons, heat_dates, store_started,
                    )
                    _log(
                        f"  {calendar_year}: wrote {'+'.join(loaded_vars)} heat+cold, "
                        f"{len(heat_dates)} day(s) ({time.time() - t0:.1f}s)."
                    )
                    completed_years.add(calendar_year)
                    _save_checkpoint(ckpt_path, {
                        "variables": variables,
                        "completed_years": sorted(completed_years),
                        "finalized": False,
                        "store_started": store_started,
                    })
                    break
                except Exception:
                    _log(
                        f"  ERROR processing calendar year {calendar_year} "
                        f"(attempt {attempt}/{max_retries_per_year}):\n{traceback.format_exc()}"
                    )
                    if attempt >= max_retries_per_year:
                        last_done = max(completed_years) if completed_years else "none"
                        _log(
                            f"ABORTING: {calendar_year} failed {attempt} times. Progress through "
                            f"{last_done} is saved in {ckpt_path} -- re-run the SAME command "
                            f"(same variables/date range) to resume from there; already-completed "
                            f"years will be skipped."
                        )
                        del cubes
                        gc.collect()
                        sys.exit(1)
                    time.sleep(5 * attempt)
                    gc.collect()

        del cubes
        gc.collect()

    _finalize_epoch_dim()
    ckpt = _load_checkpoint(ckpt_path)
    ckpt["finalized"] = True
    _save_checkpoint(ckpt_path, ckpt)

    _log("=" * 70)
    _log(f"Batch pre-computation complete. Store: {WAVE_STATUS_ZARR}")
    _log("=" * 70)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", help="Rebuild the entire historical record (1940 -> latest).")
    p.add_argument("--years", type=int, default=DEFAULT_RECENT_YEARS, help="Incremental mode: number of most recent calendar years to (re)process.")
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument(
        "--batch-years", type=int, default=DEFAULT_BATCH_YEARS,
        help=(
            "Years' worth of all-variable cubes to hold in RAM at once "
            f"(default {DEFAULT_BATCH_YEARS}). Lower this if you hit MemoryError; "
            "raise it (fewer, bigger batches) if you have plenty of RAM and want "
            "slightly less repeated disk I/O from the +/-1 year winter-splice padding."
        ),
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=f"Checkpoint file path (default: {CHECKPOINT_PATH}).",
    )
    p.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES_PER_YEAR,
        help="Retries for a single calendar year before aborting (default 3).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(
            full=args.full,
            start_year=args.start_year,
            end_year=args.end_year,
            recent_years=args.years,
            batch_years=args.batch_years,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            max_retries_per_year=args.max_retries,
        )
    except SystemExit:
        raise
    except Exception:
        # Last-resort safety net: any exception NOT already handled by the
        # per-year retry loop (e.g. a crash during setup, climatology
        # loading, or _finalize_epoch_dim itself) is logged in full instead
        # of dumping a raw traceback with no context -- checkpoint progress
        # made so far is already safely on disk (see _save_checkpoint),
        # so simply re-running the same command resumes from there.
        _log(f"FATAL, unhandled exception:\n{traceback.format_exc()}")
        _log("Progress made so far (if any) is saved in the checkpoint file; re-run the same command to resume.")
        sys.exit(1)
