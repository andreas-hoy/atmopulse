import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
LIVE_DIR = Path("ERA5_ClimateTool/Live_Forecasts")

# Physically plausible bounds for the rendered fields. Used only to mask out
# isolated fill-value/corrupt cells (e.g. leftover Kelvin values, sentinel
# fill numbers at land-sea mask edges) before they reach the colorbin logic -
# NOT to clip legitimate extremes.
_PHYS_BOUNDS = {
    "tx": (-90.0, 60.0),
    "tn": (-90.0, 60.0),
    "tg": (-90.0, 60.0),
    "mslp": (850.0, 1100.0),
    "z500": (400.0, 650.0),
    "t850": (-80.0, 45.0),
    "u850": (-150.0, 150.0),
    "v850": (-150.0, 150.0),
    "u300": (-150.0, 150.0),
    "v300": (-150.0, 150.0),
}

# New unified master batches (era5_master_daily_YYYY.nc, one file per year,
# produced by era5_master_decade_downloader.py). Each file bundles the
# true 24h daily statistics (tx/tn/tg from "derived-era5-single-levels-daily-
# statistics") alongside the 12:00 UTC synoptic fields (mslp, z500, t850,
# u850, v850, u300, v300) that used to live in three separate batch families
# (era5_mslp_batch_*.nc / era5_z500_batch_*.nc / era5_txtn_batch_*.nc).
MASTER_GLOB = "era5_master_daily_*.nc"

# Legacy live-forecast bridge files still ship the old GRIB-derived variable
# names (msl, z, mx2t, mn2t) — rename them on load so they merge cleanly with
# the new master-batch naming (mslp, z500, tx, tn) without touching the IFS
# update pipeline itself.
_LIVE_VAR_RENAME = {"msl": "mslp", "z": "z500", "mx2t": "tx", "mn2t": "tn"}


def _normalize_longitude(da):
    """Force -180..180 longitude convention so grids from different sources
    (ERA5 archive vs IFS live forecast, which may ship 0..360) line up
    pixel-for-pixel with the climatology reference instead of producing
    coastal boundary artifacts from a half-grid (or full wrap) misalignment."""
    if "longitude" not in da.coords:
        return da
    lon = da["longitude"].values
    if np.nanmax(lon) > 180.0:
        da = da.assign_coords(longitude=(((da["longitude"] + 180) % 360) - 180))
        da = da.sortby("longitude")
    return da


def _mask_implausible(da, bounds):
    """Replace isolated non-physical values (fill sentinels, stray corrupt
    cells) with NaN using an explicit skipna-safe comparison, rather than
    letting them fall through to the colorbin classification as spurious
    'record' extremes."""
    lo, hi = bounds
    return da.where((da >= lo) & (da <= hi) | da.isnull())


def _harmonize_master_batch(ds):
    """Preprocess hook for xr.open_mfdataset over era5_master_daily_*.nc.

    Normalizes the time dimension name to 'valid_time' (the daily-statistics
    product and the single/pressure-level products have historically not
    always agreed on 'time' vs 'valid_time'), and reconciles ERA5T's
    'expver' ensemble-version axis and any leftover singleton
    'pressure_level' dim from the upper-air retrieval.
    """
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    if "expver" in ds.dims:
        ds = ds.dropna(dim="expver", how="all").isel(expver=0)
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def _harmonize_live_forecast(ds):
    """Rename legacy IFS live-forecast variable/coord names onto the new
    master-batch schema so live files can be concatenated onto the archive
    dataset without a separate code path."""
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    for old, new in _LIVE_VAR_RENAME.items():
        if old in ds.variables and new not in ds.variables:
            ds = ds.rename({old: new})
    drop_coords = [c for c in ("expver", "heightAboveGround", "meanSea", "isobaricInhPa") if c in ds.coords]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


@st.cache_resource(show_spinner=False)
def load_global_datasets():
    """
    SINGLETON CACHE: Loads the unified master archive (one era5_master_daily_
    YYYY.nc file per year, all variables bundled) plus the IFS live-forecast
    bridge files into a single RAM-resident xarray handle.

    Replaces the previous three-way split (separate MSLP/Z500/TX-TN batch
    families, each opened with its own xr.open_mfdataset call) with one
    central open_mfdataset call, since every variable now lives in the same
    per-year file.
    """
    master_files = sorted(DATA_DIR.glob(MASTER_GLOB))
    if not master_files:
        raise FileNotFoundError(f"No {MASTER_GLOB} files found in {DATA_DIR}!")

    ds = xr.open_mfdataset(
        master_files, combine="nested", concat_dim="valid_time",
        engine="netcdf4", parallel=False, preprocess=_harmonize_master_batch,
    ).sortby("valid_time").drop_duplicates(dim="valid_time")

    live_files = [
        LIVE_DIR / "live_forecast_mslp.nc",
        LIVE_DIR / "live_forecast_z500.nc",
        LIVE_DIR / "live_forecast_txtn.nc",
    ]
    live_files = [f for f in live_files if f.exists()]
    if live_files:
        live_ds = xr.merge(
            [xr.open_dataset(f, engine="netcdf4").pipe(_harmonize_live_forecast) for f in live_files],
            compat="override", join="outer",
        )
        ds = xr.concat([ds, live_ds], dim="valid_time", combine_attrs="override", join="outer")
        ds = ds.sortby("valid_time").drop_duplicates(dim="valid_time")

    return ds


@st.cache_resource(show_spinner=False)
def load_windowed_synoptic_arrays(anchor_date_str, pad_past=6, pad_future=6):
    """
    SINGLETON CACHE (recomputed only when anchor_date_str changes, i.e. once
    per calendar day): the on-disk archive batches are internally chunked in
    ~1-year blocks (native NetCDF chunksizes span a full year x full lat/lon),
    so a lazy .sel(single day).load() still forces dask to decompress that
    ENTIRE multi-hundred-MB block from disk - regardless of how small the
    requested selection is. Eagerly decompressing the full slider-reachable
    range in one pass turns every subsequent Prev/Next Day click into pure
    in-memory NumPy slicing instead of a fresh multi-hundred-MB decompression
    per click.
    """
    anchor = pd.to_datetime(anchor_date_str)
    start = anchor - pd.Timedelta(days=pad_past)
    end = anchor + pd.Timedelta(days=pad_future)
    ds = load_global_datasets()
    return ds.sel(valid_time=slice(start, end)).load()


# Set by app.py before each slider day-slice so get_synoptic_map_data() can stay
# callable with a single date_str arg (avoids stale-module TypeError on reload).
_synoptic_anchor = None
_synoptic_pad_past = 6
_synoptic_pad_future = 6


def set_synoptic_anchor(anchor_date_str, pad_past=6, pad_future=6):
    """Pre-warm the slider-reachable window around anchor_date_str (once per calendar day)."""
    global _synoptic_anchor, _synoptic_pad_past, _synoptic_pad_future
    _synoptic_anchor = anchor_date_str
    _synoptic_pad_past = pad_past
    _synoptic_pad_future = pad_future
    if anchor_date_str is not None:
        load_windowed_synoptic_arrays(anchor_date_str, pad_past + 3, pad_future + 3)


def _select_calendar_day(ds, target_dt):
    """Return index into valid_time for target_dt's calendar day, or None if missing.

    Avoids method='nearest' snapping future slider days onto the last available
    forecast step (e.g. offset +5 and +6 both showing identical fields).
    """
    target_day = pd.Timestamp(target_dt).normalize()
    times = pd.DatetimeIndex(pd.to_datetime(ds.valid_time.values))
    day_ix = np.flatnonzero(times.normalize() == target_day)
    if len(day_ix) == 0:
        return None, None
    if len(day_ix) == 1:
        return int(day_ix[0]), times[int(day_ix[0])]
    noon = target_day + pd.Timedelta(hours=12)
    best = int(day_ix[np.argmin(np.abs(times[day_ix] - noon))])
    return best, times[best]


def _empty_field_like(ds, var_name):
    """NaN field with correct spatial dims when the requested day is unavailable."""
    template = ds.isel(valid_time=0)[var_name]
    return xr.full_like(template, np.nan)


def _to_celsius(da):
    """Auto-detect and normalize Kelvin to Celsius."""
    if float(da.mean(skipna=True)) > 100:
        return da - 273.15
    return da


def get_synoptic_map_data(date_str, anchor_date_str=None, pad_past=None, pad_future=None):
    target_dt = pd.to_datetime(date_str)
    if anchor_date_str is None:
        anchor_date_str = _synoptic_anchor
    if pad_past is None:
        pad_past = _synoptic_pad_past
    if pad_future is None:
        pad_future = _synoptic_pad_future

    ds = None
    if anchor_date_str is not None:
        try:
            # +3 day safety margin so a data gap sitting right at the slider's
            # edge still resolves to the same result the full (unwindowed)
            # dataset would have picked.
            ds = load_windowed_synoptic_arrays(anchor_date_str, pad_past + 3, pad_future + 3)
        except Exception:
            ds = None
    if ds is None:
        ds = load_global_datasets()

    t_idx, actual_time = _select_calendar_day(ds, target_dt)
    meta = {
        "requested": target_dt.normalize(),
        "available": actual_time is not None,
        "actual_time": pd.Timestamp(actual_time).normalize() if actual_time is not None else None,
    }

    field_names = ["mslp", "z500", "tx", "tn", "tg", "t850", "u850", "v850", "u300", "v300"]
    if t_idx is None:
        slice_ds = xr.Dataset({v: _empty_field_like(ds, v) for v in field_names if v in ds.data_vars})
    else:
        slice_ds = ds.isel(valid_time=t_idx).load()

    clean_slices = {"_meta": meta}
    for name in field_names:
        if name not in slice_ds.data_vars:
            continue
        da = slice_ds[name]
        if "expver" in da.dims:
            da = da.dropna(dim="expver", how="all").isel(expver=0)
            if "expver" in da.coords:
                da = da.drop_vars("expver")

        if name == "mslp":
            da = da / 100.0
        elif name == "z500":
            da = da / 9.80665 / 10.0
        elif name in ("tx", "tn", "tg", "t850"):
            da = _to_celsius(da)
        # u850/v850/u300/v300 are already in m/s, no conversion needed.

        da = _normalize_longitude(da)
        bounds = _PHYS_BOUNDS.get(name)
        if bounds is not None:
            da = _mask_implausible(da, bounds)
        clean_slices[name] = da

    return clean_slices
