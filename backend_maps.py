"""
AtmoPulse Geospatial Data Orchestration & Harmonization (backend_maps.py)

This module handles the ingestion, harmonization, and physical masking of
massive multi-dimensional atmospheric datasets. It dynamically merges secular
ERA5 reanalysis master batches with live ECMWF/IFS forecast bridges to enable
real-time tracking of synoptic extremes.

Core functionalities:
- Manages dask-backed, singleton caching of trailing temporal windows to ensure
  low-latency array slicing for interactive Streamlit frontends.
- Normalizes grid longitude conventions to prevent coastal boundary artifacts 
  during ERA5/IFS harmonization.
- Applies strict physical bounding to mask out interpolation artifacts or 
  fill values before they reach the extreme-value classification logic.
"""

from __future__ import annotations

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
LIVE_DIR = Path("ERA5_ClimateTool/Live_Forecasts")

# Physically plausible bounds for the rendered fields. Used solely to mask out
# isolated fill-value/corrupt cells (e.g., leftover Kelvin values, sentinel
# fill numbers at land-sea mask edges) before they reach the colorbin logic.
# These are NOT used to clip legitimate atmospheric extremes.
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

# New unified master batches (era5_master_daily_YYYY.nc, one file per year).
# Each file bundles the true 24h daily statistics (TX/TN/TG) alongside the 
# 12:00 UTC synoptic fields (MSLP, Z500, T850, U850, V850, U300, V300).
MASTER_GLOB = "era5_master_daily_*.nc"

# Legacy live-forecast bridge files currently ship with old GRIB-derived 
# variable names. These are renamed on load to merge cleanly with the new 
# master-batch naming schema without altering the operational IFS pipeline.
_LIVE_VAR_RENAME = {"msl": "mslp", "z": "z500", "mx2t": "tx", "mn2t": "tn"}


def _normalize_longitude(da: xr.DataArray) -> xr.DataArray:
    """
    Forces a -180..180 longitude convention. Ensures grids from different sources
    (ERA5 archive vs. IFS live forecast) align pixel-for-pixel with the reference 
    climatology, preventing coastal boundary artifacts from grid misalignment.
    """
    if "longitude" not in da.coords:
        return da
    lon = da["longitude"].values
    if np.nanmax(lon) > 180.0:
        da = da.assign_coords(longitude=(((da["longitude"] + 180) % 360) - 180))
        da = da.sortby("longitude")
    return da


def _mask_implausible(da: xr.DataArray, bounds: tuple) -> xr.DataArray:
    """
    Replaces isolated non-physical values (fill sentinels, stray corrupt cells) 
    with NaN using an explicit skipna-safe comparison to prevent them from 
    triggering spurious 'record' extreme classifications.
    """
    lo, hi = bounds
    return da.where((da >= lo) & (da <= hi) | da.isnull())


def drop_era5t_aux(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove ERA5T ``expver`` (dimension or coordinate) and other auxiliary
    coords so yearly master files concatenate. ``isel(expver=0)`` without
    ``drop=True`` left a scalar coord on some files and none on others,
    which raised: coordinate 'expver' not present in all datasets.
    """
    if "expver" in ds.dims:
        ds = ds.dropna(dim="expver", how="all")
        if "expver" in ds.dims:
            ds = ds.isel(expver=0, drop=True)
    for name in ("expver", "number", "heightAboveGround", "meanSea", "isobaricInhPa"):
        if name in ds.variables or name in ds.coords:
            ds = ds.drop_vars(name, errors="ignore")
    return ds


def _harmonize_master_batch(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess hook for xr.open_mfdataset over era5_master_daily_*.nc.
    Normalizes the time dimension name to 'valid_time' and reconciles 
    ERA5T's 'expver' ensemble-version axis and singleton pressure levels.
    """
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds = drop_era5t_aux(ds)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def _harmonize_live_forecast(ds: xr.Dataset) -> xr.Dataset:
    """
    Renames legacy IFS live-forecast variable/coordinate names to match the 
    master-batch schema, allowing seamless concatenation onto the archive dataset.
    """
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
    # IFS ingestion stores MSLP in hPa; ERA5 master batches store Pa.
    if "mslp" in ds.data_vars:
        time_dim = "valid_time" if "valid_time" in ds["mslp"].dims else ds["mslp"].dims[0]
        sample = float(ds["mslp"].isel({time_dim: 0}).mean(skipna=True))
        if np.isfinite(sample) and 0 < sample < 2000:
            ds = ds.assign(mslp=ds["mslp"] * 100.0)
    return ds


@st.cache_resource(show_spinner=False)
def load_global_datasets(_harmonize_version=6) -> xr.Dataset:
    """Nearby-year master batches plus the latest IFS daily forecast."""
    today = pd.Timestamp.now().normalize()
    return _open_synoptic_range(today - pd.Timedelta(days=400), today + pd.Timedelta(days=10))


def _master_files_covering(start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    files = []
    for year in range(int(start.year), int(end.year) + 1):
        path = DATA_DIR / f"era5_master_daily_{year}.nc"
        if path.exists():
            files.append(path)
    return files


def _latest_ifs_forecast_path() -> Path | None:
    files = sorted(LIVE_DIR.glob("ifs_daily_forecast_*.nc"))
    return files[-1] if files else None


def _drop_truncated_last_day(ds: xr.Dataset) -> xr.Dataset:
    """Drop a trailing calendar day that is only a single forecast step.

    Open-data IFS/AIFS runs are 0–72 h. The last daily bin often contains one
    00 UTC snapshot, so TX==TN==TG and 12 UTC MSLP is missing. That must not
    be shown as a daily extreme field.
    """
    tdim = "valid_time" if "valid_time" in ds.dims else ("time" if "time" in ds.dims else None)
    if tdim is None or ds.sizes.get(tdim, 0) < 2:
        return ds
    last = ds.isel({tdim: -1})
    truncated = False
    if "tx" in last and "tn" in last:
        tx = np.asarray(last["tx"].values, dtype=float)
        tn = np.asarray(last["tn"].values, dtype=float)
        finite = np.isfinite(tx) & np.isfinite(tn)
        if finite.any() and float(np.nanmax(np.abs(tx[finite] - tn[finite]))) < 1e-3:
            truncated = True
    if "mslp" in last:
        mslp = np.asarray(last["mslp"].values, dtype=float)
        if not np.isfinite(mslp).any():
            truncated = True
    if truncated:
        return ds.isel({tdim: slice(None, -1)})
    return ds


def _open_forecast_glob(pattern: str) -> xr.Dataset | None:
    """Open every matching live-forecast file; newer runs override older ones."""
    files = sorted(LIVE_DIR.glob(pattern))
    if not files:
        return None
    opened = [
        xr.open_dataset(path, engine="netcdf4")
        .pipe(_harmonize_live_forecast)
        .pipe(drop_era5t_aux)
        .pipe(_drop_truncated_last_day)
        for path in files
    ]
    if len(opened) == 1:
        return opened[0]
    shared = [v for v in opened[0].data_vars if all(v in ds.data_vars for ds in opened)]
    if not shared:
        return opened[-1]
    cleaned = [_strip_aux_coords(ds[shared]) for ds in opened]
    try:
        out = xr.concat(
            cleaned, dim="valid_time",
            data_vars="minimal", coords="minimal", join="outer",
        )
        return drop_era5t_aux(_drop_duplicates_time(out, keep="last"))
    except Exception:
        return opened[-1]


def _open_live_forecast_ds() -> xr.Dataset | None:
    """Prefer IFS daily files; AIFS fills calendar days IFS does not cover."""
    ifs = _open_forecast_glob("ifs_daily_forecast_*.nc")
    aifs = _open_forecast_glob("aifs_daily_forecast_*.nc")
    if ifs is not None and aifs is not None:
        shared = [v for v in ifs.data_vars if v in aifs.data_vars]
        if shared:
            try:
                ifs_c = _strip_aux_coords(ifs[shared])
                aifs_c = _strip_aux_coords(aifs[shared])
                if "latitude" in ifs_c.coords and "latitude" in aifs_c.coords:
                    if ifs_c.latitude.shape == aifs_c.latitude.shape:
                        aifs_c = aifs_c.assign_coords(
                            latitude=ifs_c.latitude, longitude=ifs_c.longitude,
                        )
                return drop_era5t_aux(ifs_c.combine_first(aifs_c))
            except Exception:
                return ifs
        return ifs
    if ifs is not None:
        return ifs
    if aifs is not None:
        return aifs

    legacy = [
        LIVE_DIR / "live_forecast_mslp.nc",
        LIVE_DIR / "live_forecast_z500.nc",
        LIVE_DIR / "live_forecast_txtn.nc",
    ]
    legacy = [f for f in legacy if f.exists()]
    if not legacy:
        return None
    live_ds = xr.merge(
        [xr.open_dataset(f, engine="netcdf4").pipe(_harmonize_live_forecast) for f in legacy],
        compat="override", join="outer",
    )
    return drop_era5t_aux(live_ds)


def _strip_aux_coords(ds: xr.Dataset) -> xr.Dataset:
    keep = {"valid_time", "time", "latitude", "longitude", "lat", "lon"}
    extra = [c for c in ds.coords if c not in keep and c not in ds.dims]
    return ds.reset_coords(extra, drop=True) if extra else ds


def _drop_duplicates_time(ds: xr.Dataset, keep: str = "last") -> xr.Dataset:
    ds = ds.sortby("valid_time")
    try:
        return ds.drop_duplicates(dim="valid_time", keep=keep)
    except TypeError:
        _, unique_idx = np.unique(ds.valid_time.values, return_index=True)
        if keep == "last":
            _, unique_idx = np.unique(ds.valid_time.values[::-1], return_index=True)
            unique_idx = ds.sizes["valid_time"] - 1 - unique_idx
        return ds.isel(valid_time=np.sort(unique_idx))


def _concat_master_and_live(master: xr.Dataset, live: xr.Dataset | None) -> xr.Dataset:
    ds = drop_era5t_aux(master)
    if live is None:
        return ds
    live = drop_era5t_aux(live)
    shared = [v for v in ds.data_vars if v in live.data_vars]
    if not shared:
        return ds
    ds = _strip_aux_coords(ds[shared])
    live = _strip_aux_coords(live[shared])
    if "latitude" in ds.coords and "latitude" in live.coords:
        if ds.latitude.shape == live.latitude.shape:
            live = live.assign_coords(latitude=ds.latitude, longitude=ds.longitude)
    try:
        # Prefer live (IFS/AIFS) on overlap; archive fills dates the forecast
        # does not cover. combine_first also backfills NaN cells in live.
        out = live.combine_first(ds)
        return drop_era5t_aux(_drop_duplicates_time(out, keep="first"))
    except Exception:
        try:
            out = xr.concat(
                [ds, live], dim="valid_time",
                data_vars="minimal", coords="minimal", join="outer",
            )
            return drop_era5t_aux(_drop_duplicates_time(out, keep="last"))
        except Exception:
            live_max = pd.to_datetime(live.valid_time.max().values)
            ds_max = pd.to_datetime(ds.valid_time.max().values)
            return live if live_max > ds_max else ds


def _open_synoptic_range(start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    files = _master_files_covering(start, end)
    if not files:
        live = _open_live_forecast_ds()
        if live is not None:
            return live
        raise FileNotFoundError(f"No master or IFS files covering {start.date()}–{end.date()}")
    if len(files) == 1:
        ds = xr.open_dataset(files[0], engine="netcdf4").pipe(_harmonize_master_batch)
    else:
        ds = xr.open_mfdataset(
            files, combine="nested", concat_dim="valid_time",
            engine="netcdf4", parallel=False, preprocess=_harmonize_master_batch,
            coords="minimal", compat="override", join="override",
        )
    ds = ds.sortby("valid_time").drop_duplicates(dim="valid_time")
    ds = ds.sel(valid_time=slice(start, end))
    live = _open_live_forecast_ds()
    if live is not None:
        live = live.sel(valid_time=slice(start, end))
    return _concat_master_and_live(ds, live)


@st.cache_resource(show_spinner=False)
def load_windowed_synoptic_arrays(anchor_date_str: str, pad_past: int = 6, pad_future: int = 6, _loader_version=6) -> xr.Dataset:
    """Lazy handle for the slider window (nearby year file(s) + latest IFS daily)."""
    anchor = pd.to_datetime(anchor_date_str)
    start = anchor - pd.Timedelta(days=pad_past)
    end = anchor + pd.Timedelta(days=pad_future)
    ds = _open_synoptic_range(start, end)
    return ds.sel(valid_time=slice(start, end))


# Global tracking variables for slider temporal windowing
_synoptic_anchor = None
_synoptic_pad_past = 6
_synoptic_pad_future = 6


def set_synoptic_anchor(anchor_date_str: str, pad_past: int = 6, pad_future: int = 6):
    """Pre-warms the slider-reachable temporal window around the anchor date."""
    global _synoptic_anchor, _synoptic_pad_past, _synoptic_pad_future
    _synoptic_anchor = anchor_date_str
    _synoptic_pad_past = pad_past
    _synoptic_pad_future = pad_future
    if anchor_date_str is not None:
        load_windowed_synoptic_arrays(anchor_date_str, pad_past + 3, pad_future + 3)


def _select_calendar_day(ds: xr.Dataset, target_dt: pd.Timestamp):
    """
    Returns the index into valid_time for the target's calendar day.
    Avoids nearest-neighbor method from snapping future slider days onto 
    the last available forecast step.
    """
    target_day = pd.Timestamp(target_dt).normalize()
    times = pd.to_datetime(ds.valid_time.values)
    times = pd.DatetimeIndex(times)
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    day_ix = np.flatnonzero(times.normalize() == target_day)
    
    if len(day_ix) == 0:
        return None, None
    if len(day_ix) == 1:
        return int(day_ix[0]), times[int(day_ix[0])]
        
    noon = target_day + pd.Timedelta(hours=12)
    best = int(day_ix[np.argmin(np.abs(times[day_ix] - noon))])
    return best, times[best]


def _empty_field_like(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Returns a NaN field with correct spatial dimensions if data is unavailable."""
    template = ds.isel(valid_time=0)[var_name]
    return xr.full_like(template, np.nan)


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Auto-detects Kelvin matrices and normalizes them to Celsius."""
    if float(da.mean(skipna=True)) > 100:
        return da - 273.15
    return da


def get_synoptic_map_data(date_str: str, anchor_date_str: str = None, pad_past: int = None, pad_future: int = None) -> dict:
    """
    Retrieves, standardizes, and applies physical masking to synoptic arrays 
    for a given date, seamlessly transitioning between ERA5 and IFS data.
    """
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
            # +3 day safety margin for data gaps sitting at the slider's edge
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
            da = drop_era5t_aux(da.to_dataset(name=name))[name]

        # Atmospheric physical unit conversions
        if name == "mslp":
            sample = float(da.mean(skipna=True))
            if np.isfinite(sample) and sample > 2000:
                da = da / 100.0
        elif name == "z500":
            da = da / 9.80665 / 10.0
        elif name in ("tx", "tn", "tg", "t850"):
            da = _to_celsius(da)
        # u850/v850/u300/v300 are already in m/s

        da = _normalize_longitude(da)
        bounds = _PHYS_BOUNDS.get(name)
        if bounds is not None:
            da = _mask_implausible(da, bounds)
        clean_slices[name] = da

    return clean_slices