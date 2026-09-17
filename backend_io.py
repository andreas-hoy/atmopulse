"""
AtmoPulse Data Access Layer (backend_io.py)

All NetCDF/HDF5 loading, singleton dataset caching, point-series extraction,
and subprocess-isolated file reads extracted from app.py. This is the only
layer that touches disk/xarray for the reference climatology, the ERA5
master archive, live IFS/AIFS forecasts, and the QDM bias cube.

Design note: none of the functions here import `app` (Streamlit runs app.py
as the entrypoint script, not as an importable module named "app" — a
`from app import ...` inside a function would re-execute the whole script
from scratch on every call and crash on already-instantiated widgets, as
happened with an earlier revision of frontend_plots.py/backend_analytics.py).
Anything previously read from an app.py module global (`ref_clim`) is
instead obtained by calling the local `@st.cache_resource`-decorated loader
directly — cheap after the first call, and guaranteed to return the exact
same singleton object.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

from backend_maps import (
    LIVE_OVERLAY_PAST_DAYS,
    _open_synoptic_range,
    _synoptic_array,
    drop_era5t_aux,
    etccdi_doy_365,
    get_synoptic_map_data,
    live_forecast_dates,
    set_synoptic_anchor,
)
from config import (
    DATA_ROOT,
    FORECAST_MODEL_IFS,
    PERSISTENCE_LOOKBACK_PAD,
    PERSISTENCE_MAX_DAYS,
    SLIDER_PAD_FUTURE,
    SLIDER_PAD_PAST,
    ZARR_MASTER_TIME_SERIES,
    meteo_var_code,
)

LIVE_TXTN = DATA_ROOT / "Live_Forecasts/live_forecast_txtn.nc"
QDM_TRANSFER_FILE = DATA_ROOT / "Reference_Climatology/qdm_transfer_functions.nc"


# --- REFERENCE CLIMATOLOGY & INVARIANTS ---
@st.cache_resource(show_spinner=False)
def load_reference_climatology():
    clim_path = DATA_ROOT / "Reference_Climatology/climatology_reference_complete.nc"
    if not clim_path.exists(): 
        clim_path = DATA_ROOT / "Reference_Climatology/climatology_reference.nc"
    return xr.open_dataset(clim_path) if clim_path.exists() else None


@st.cache_resource(show_spinner=False)
def load_synoptic_climatology():
    """Epoch A/B DOY-mean MSLP and Z500 from ``climatology_synoptics.nc``.

    Stored units match the ERA5 master batches (MSLP in Pa, Z500 as
    geopotential). Convert to map display units (hPa, dam) at extract time
    via ``synoptic_clim_mean_display``.
    """
    clim_path = DATA_ROOT / "Reference_Climatology/climatology_synoptics.nc"
    return xr.open_dataset(clim_path) if clim_path.exists() else None


def synoptic_clim_mean_display(syn_clim, param: str, epoch: str, doy, lats, lons):
    """DOY-mean synoptic field on the live map grid, in map display units.

    MSLP → hPa, Z500 → dam (geopotential / g / 10), matching
    ``backend_maps.get_synoptic_map_data``. Returns None if the climatology
    or variable is missing.
    """
    if syn_clim is None:
        return None
    key = f"{param}_mean_doy_{epoch}"
    if key not in syn_clim.variables:
        return None
    da = syn_clim[key].sel(dayofyear=int(doy)).reindex(
        latitude=lats, longitude=lons, method="nearest",
    )
    arr = np.squeeze(np.asarray(da.values, dtype=float))
    return _synoptic_clim_to_display(param, arr)


def synoptic_clim_point_doy(syn_clim, param: str, epoch: str, lat, lon):
    """365-length DOY-mean series at the nearest grid point, in display units."""
    if syn_clim is None:
        return None
    key = f"{param}_mean_doy_{epoch}"
    if key not in syn_clim.variables:
        return None
    da = syn_clim[key].sel(latitude=lat, longitude=lon, method="nearest")
    arr = np.squeeze(np.asarray(da.values, dtype=float))
    return _synoptic_clim_to_display(param, arr)


def _synoptic_clim_to_display(param: str, arr: np.ndarray) -> np.ndarray:
    """MSLP Pa→hPa, Z500 geopotential→dam — same conversions as the map fields."""
    sample = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
    if param == "mslp" and np.isfinite(sample) and sample > 2000:
        return arr / 100.0
    if param == "z500" and np.isfinite(sample) and sample > 10000:
        return arr / 9.80665 / 10.0
    return arr


@st.cache_resource(show_spinner=False)
def load_invariant_fields():
    """ERA5 time-invariant physiography fields (land-sea mask, orography,
    sub-grid orography variance) used to describe the physical footprint of
    a 0.25deg grid cell in the point-based tabs."""
    inv_path = DATA_ROOT / "Reference_Climatology/era5_invariants.nc"
    if not inv_path.exists():
        return None
    return xr.open_dataset(inv_path, engine="netcdf4")


@st.cache_data(show_spinner=False)
def _create_gridcell_map(target_lat, target_lon):
    """Renders the macro-scale ERA5 0.25deg grid cell footprint (satellite
    imagery + bounding rectangle) around the target point."""
    if not _FOLIUM_AVAILABLE:
        return None
    half_res = 0.125
    lat_south, lat_north = target_lat - half_res, target_lat + half_res
    lon_west, lon_east = target_lon - half_res, target_lon + half_res

    m = folium.Map(location=[target_lat, target_lon], zoom_start=9, tiles=None)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri World Imagery', overlay=False, control=True,
    ).add_to(m)
    folium.Rectangle(
        bounds=[[lat_south, lon_west], [lat_north, lon_east]],
        color="#ff7800", weight=2, fill_opacity=0.15,
    ).add_to(m)
    folium.CircleMarker(
        location=[target_lat, target_lon], radius=4, color="red",
    ).add_to(m)
    return m


# --- ERA5 MASTER ARCHIVE ---
@st.cache_resource(show_spinner=False)
def get_master_files():
    DATA_DIR = DATA_ROOT / "Master_Batches"
    return sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))


def _harmonize_master_archive(ds):
    """Normalize time-dim naming + expver/pressure_level across the unified
    era5_master_daily_*.nc batches, same as backend_maps.py's loader."""
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds = drop_era5t_aux(ds)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def _open_master_year_file(path):
    """Maps-style open: netcdf4, no dask chunks. `chunks={}` on the in-progress
    current-year file is what raised NetCDF: HDF error while maps still rendered."""
    return xr.open_dataset(path, engine="netcdf4").pipe(_harmonize_master_archive)


@st.cache_resource(show_spinner=False)
def get_master_archive_ds(_harmonize_version=6):
    """
    SINGLETON POINTER: unified handle for the ERA5 master archive.
    Historical years are opened as a multi-file dataset; the current calendar
    year is opened the same way Map Tracker does (single netcdf4 handle, no
    dask chunks), falling back to the already-cached maps window if Windows
    HDF locking refuses a second open of that file.
    """
    files = get_master_files()
    if not files:
        return None
    this_year = str(pd.Timestamp.utcnow().year)
    hist_files = [f for f in files if not f.stem.endswith(this_year)]
    cur_files = [f for f in files if f.stem.endswith(this_year)]

    opened = []
    if hist_files:
        if len(hist_files) == 1:
            opened.append(_open_master_year_file(hist_files[0]))
        else:
            opened.append(xr.open_mfdataset(
                hist_files, combine='nested', concat_dim='valid_time', engine='netcdf4',
                parallel=False, preprocess=_harmonize_master_archive,
                coords="minimal", compat="override", join="override",
            ))
    if cur_files:
        try:
            opened.append(_open_master_year_file(cur_files[0]))
        except Exception:
            pass
    if not opened:
        return None
    ds = opened[0] if len(opened) == 1 else xr.concat(
        opened, dim='valid_time', coords="minimal", compat="override", join="override",
    )
    ds = drop_era5t_aux(ds)
    ds = ds.sortby('valid_time')
    # Keep the LAST occurrence of any duplicated calendar day, not np.unique's
    # default first-occurrence. When two master batch files overlap on the
    # same day (e.g. an older, possibly NaN-placeholder "current year" file
    # re-downloaded/corrected later under a new batch file), sortby's stable
    # mergesort preserves original file-list order for ties, so "first" would
    # silently keep the STALE row. "Last" always keeps the most-recently
    # concatenated (i.e. most recently written) file's value for that day.
    times = ds.valid_time.values
    _, first_idx_of_reversed = np.unique(times[::-1], return_index=True)
    keep_idx = np.sort(len(times) - 1 - first_idx_of_reversed)
    return ds.isel(valid_time=keep_idx)


@st.cache_resource(show_spinner=False)
def get_live_txtn_ds(forecast_model=FORECAST_MODEL_IFS, _loader_version=8):
    """Latest selected-model daily forecast (tx/tn), falling back to the legacy txtn bridge."""
    from backend_maps import _open_live_forecast_ds
    live = _open_live_forecast_ds(forecast_model)
    if live is not None:
        return live
    if "AIFS" in str(forecast_model):
        return None
    if not LIVE_TXTN.exists():
        return None
    return xr.open_dataset(LIVE_TXTN, engine='netcdf4')


@st.cache_resource(show_spinner=False)
def _load_persistence_window_source(
    start_date_str, end_date_str,
    forecast_model=FORECAST_MODEL_IFS, _loader_version=10,
):
    """
    SINGLETON CACHE: the requested persistence window from covering
    era5_master_daily_YYYY.nc files, with IFS/AIFS only from today-6d
    through the forecast (see LIVE_OVERLAY_PAST_DAYS).
    """
    start = pd.to_datetime(start_date_str).normalize()
    end = pd.to_datetime(end_date_str).normalize()
    try:
        ds = _open_synoptic_range(start, end, forecast_model=forecast_model)
        return ds.sel(valid_time=slice(start, end))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_persistence_daily_series(
    start_date_str, end_date_str, anchor_date_str=None, forecast_model=FORECAST_MODEL_IFS,
    _series_version=4,
):
    """Build a daily TX/TN/TG/T850 cube from ERA5 masters; IFS/AIFS only in the last 6 days + forecast.

    Each variable is stored independently. Missing TX/TN is never filled from TG.
    """
    start_date = pd.to_datetime(start_date_str).normalize()
    end_date = pd.to_datetime(end_date_str).normalize()
    by_date = {}
    overlay_cut = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=LIVE_OVERLAY_PAST_DAYS)

    ds = _load_persistence_window_source(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
        forecast_model=forecast_model,
    )
    if ds is not None:
        with st.session_state.nc_lock:
            max_arch = pd.to_datetime(ds.valid_time.max().values).normalize()
            arch_end = min(end_date, max_arch)
            if arch_end >= start_date:
                sub = ds.sel(valid_time=slice(start_date, arch_end)).compute()
                # tx/tn are already true 24h daily statistics (one value per
                # calendar day) from era5_master_daily_*.nc; groupby/agg here
                # is a harmless idempotent no-op that also collapses any
                # leftover duplicate timestamps.
                # 29 Feb is kept as a real day here too — persistence streaks
                # are a live/actual-data view, not the 365-day baseline array.
                tx_by, tn_by, t850_by, tg_by = {}, {}, {}, {}
                if "tx" in sub.data_vars:
                    tx_d = sub["tx"].groupby("valid_time.date").max()
                    for i, d in enumerate(tx_d["date"].values):
                        tx_by[pd.Timestamp(d).normalize()] = tx_d.values[i]
                if "tn" in sub.data_vars:
                    tn_d = sub["tn"].groupby("valid_time.date").min()
                    for i, d in enumerate(tn_d["date"].values):
                        tn_by[pd.Timestamp(d).normalize()] = tn_d.values[i]
                if "t850" in sub.data_vars:
                    t850_d = sub["t850"].groupby("valid_time.date").mean()
                    for i, d in enumerate(t850_d["date"].values):
                        t850_by[pd.Timestamp(d).normalize()] = t850_d.values[i]
                if "tg" in sub.data_vars:
                    tg_d = sub["tg"].groupby("valid_time.date").mean()
                    for i, d in enumerate(tg_d["date"].values):
                        tg_by[pd.Timestamp(d).normalize()] = tg_d.values[i]
                for day in set(tx_by) | set(tn_by) | set(t850_by) | set(tg_by):
                    by_date[day] = (tx_by.get(day), tn_by.get(day), t850_by.get(day), tg_by.get(day))

    archive_max = max((d for d in by_date if d < overlay_cut), default=None)

    eligible = sorted(d for d in by_date if start_date <= d <= end_date)
    if not eligible:
        return None, {"archive_max": archive_max, "effective_end": None, "uses_ifs": False, "has_gap": False}

    eligible = eligible[-PERSISTENCE_MAX_DAYS:]

    def _stack_field(idx):
        sample = next((by_date[d][idx] for d in eligible if by_date[d][idx] is not None), None)
        if sample is None:
            other = next(
                (by_date[d][j] for d in eligible for j in range(4) if by_date[d][j] is not None),
                None,
            )
            if other is None:
                return None
            return np.full((len(eligible),) + np.shape(other), np.nan)
        nan_grid = np.full_like(sample, np.nan)
        return np.stack([by_date[d][idx] if by_date[d][idx] is not None else nan_grid for d in eligible])

    tx_vals = _stack_field(0)
    tn_vals = _stack_field(1)
    t850_vals = _stack_field(2)
    tg_vals = _stack_field(3)
    if tx_vals is None and tn_vals is None and t850_vals is None and tg_vals is None:
        return None, {"archive_max": archive_max, "effective_end": None, "uses_ifs": False, "has_gap": False}

    if (
        "AIFS" not in str(forecast_model)
        and ds is not None
        and "latitude" in ds.coords
        and "longitude" in ds.coords
    ):
        live_days = live_forecast_dates(forecast_model)
        if live_days:
            eligible_ts = [pd.Timestamp(d).normalize() for d in eligible]
            mask = np.array([d in live_days for d in eligible_ts], dtype=bool)
            if mask.any():
                lats = np.asarray(ds.latitude.values)
                lons = np.asarray(ds.longitude.values)
                ifs_dates = [d for d, keep in zip(eligible_ts, mask) if keep]
                tx_s = tx_vals[mask] if tx_vals is not None else None
                tn_s = tn_vals[mask] if tn_vals is not None else None
                tg_s = tg_vals[mask] if tg_vals is not None else None
                tx_s, tn_s, tg_s = _apply_ifs_mean_qdm(tx_s, tn_s, tg_s, lats, lons, ifs_dates)
                if tx_vals is not None and tx_s is not None:
                    tx_vals = np.array(tx_vals, copy=True)
                    tx_vals[mask] = tx_s
                if tn_vals is not None and tn_s is not None:
                    tn_vals = np.array(tn_vals, copy=True)
                    tn_vals[mask] = tn_s
                if tg_vals is not None and tg_s is not None:
                    tg_vals = np.array(tg_vals, copy=True)
                    tg_vals[mask] = tg_s

    ifs_used = any(d >= overlay_cut for d in eligible)
    has_gap = False
    if archive_max and ifs_used:
        ifs_days = [d for d in eligible if d >= overlay_cut]
        if ifs_days:
            has_gap = (min(ifs_days) - archive_max).days > 1

    meta = {
        "archive_max": archive_max,
        "effective_end": eligible[-1],
        "uses_ifs": ifs_used,
        "has_gap": has_gap,
    }
    return (np.array(eligible), tx_vals, tn_vals, t850_vals, tg_vals), meta


# --- QDM BIAS CORRECTION ---
@st.cache_resource(show_spinner=False)
def _load_qdm_bias_ds():
    """
    Optional IFS-vs-ERA5 QDM cube (see calculate_qdm_bias.py). True QDM
    needs ``*_ifs_q`` plus ``*_bias``; older files with only bias fall back
    to the mean-quantile shift. Returns None (zero-bias passthrough) until
    the hindcast builder has been run. T850 is never in this cube.
    """
    if not QDM_TRANSFER_FILE.exists():
        return None
    try:
        return xr.open_dataset(QDM_TRANSFER_FILE, engine='netcdf4')
    except Exception:
        return None


_QDM_IFS_Q_VAR = {
    "tx_bias": "tx_ifs_q",
    "dtr_bias": "dtr_ifs_q",
    "tg_bias": "tg_ifs_q",
}


def _qdm_apply_field(x, q_model, bias):
    """True QDM on a spatial field: x' = x + Δ(F_IFS(x)).

    ``q_model`` and ``bias`` are (nq, nlat, nlon). Values outside the stored
    quantile range keep the endpoint Δ (Cannon). NaN in ``x`` is unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q_model, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if x.shape != q.shape[1:] or b.shape != q.shape:
        return x
    nq = q.shape[0]
    if nq < 2:
        return x
    q_mono = np.maximum.accumulate(np.where(np.isfinite(q), q, -np.inf), axis=0)
    q0, qn = q_mono[0], q_mono[-1]
    below = x <= q0
    above = x >= qn
    ge = q_mono >= x
    has = np.any(ge, axis=0)
    idx_hi = np.where(has, np.argmax(ge, axis=0), nq - 1)
    idx_lo = np.maximum(idx_hi - 1, 0)

    def _take(arr, idx):
        return np.take_along_axis(arr, idx[np.newaxis], axis=0)[0]

    ql, qh = _take(q_mono, idx_lo), _take(q_mono, idx_hi)
    bl, bh = _take(b, idx_lo), _take(b, idx_hi)
    denom = qh - ql
    w = np.divide(x - ql, denom, out=np.zeros_like(x), where=np.abs(denom) > 1e-8)
    w = np.clip(w, 0.0, 1.0)
    delta = bl + w * (bh - bl)
    delta = np.where(below, b[0], delta)
    delta = np.where(above, b[-1], delta)
    delta = np.nan_to_num(delta, nan=0.0)
    return np.where(np.isfinite(x), x + delta, x)


def _qdm_apply_scalar(x, q_model, bias):
    """True QDM for one value against 1-D quantile curves."""
    if not np.isfinite(x):
        return x
    q = np.asarray(q_model, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    ok = np.isfinite(q) & np.isfinite(b)
    if int(ok.sum()) < 2:
        return x
    q, b = q[ok], b[ok]
    order = np.argsort(q, kind="mergesort")
    q, b = q[order], b[order]
    q, uniq = np.unique(q, return_index=True)
    b = b[uniq]
    if q.size < 2:
        return x + float(b[0] if b.size else 0.0)
    return x + float(np.interp(x, q, b))


def _qdm_has_ifs_q(ds_qdm, bias_var) -> bool:
    q_var = _QDM_IFS_Q_VAR.get(bias_var)
    return bool(ds_qdm is not None and q_var and q_var in ds_qdm.data_vars)


def _qdm_mean_bias_grid(lats, lons, doys_1_365, bias_var):
    """Mean-quantile bias on a lat/lon grid: shape (n_time, n_lat, n_lon).

    Fallback when the cube has no ``*_ifs_q`` (older files). Zeros when the
    cube or variable is missing.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    doys = np.atleast_1d(np.asarray(doys_1_365, dtype=int))
    shape = (doys.size, lats.size, lons.size)
    ds_qdm = _load_qdm_bias_ds()
    if ds_qdm is None or bias_var not in ds_qdm.data_vars:
        return np.zeros(shape, dtype=np.float64)
    try:
        da = ds_qdm[bias_var].mean(dim="quantile")
        da = da.reindex(latitude=lats, longitude=lons, method="nearest")
    except Exception:
        return np.zeros(shape, dtype=np.float64)
    n_doy = int(da.sizes.get("dayofyear", 365))
    idx = np.clip(doys - 1, 0, n_doy - 1)
    return np.nan_to_num(np.asarray(da.values, dtype=np.float64)[idx], nan=0.0)


def _qdm_correct_grid(field, lats, lons, doys_1_365, bias_var):
    """Correct a 2-D or 3-D IFS field. True QDM if ``*_ifs_q`` exists."""
    if field is None:
        return None
    arr = np.asarray(field, dtype=np.float64)
    doys = np.atleast_1d(np.asarray(doys_1_365, dtype=int))
    ds_qdm = _load_qdm_bias_ds()
    if ds_qdm is None or bias_var not in ds_qdm.data_vars:
        return arr
    if not _qdm_has_ifs_q(ds_qdm, bias_var):
        bias = _qdm_mean_bias_grid(lats, lons, doys, bias_var)
        if arr.ndim == 2:
            return arr + bias[0]
        return arr + bias
    q_var = _QDM_IFS_Q_VAR[bias_var]
    try:
        da_q = ds_qdm[q_var].reindex(latitude=lats, longitude=lons, method="nearest")
        da_b = ds_qdm[bias_var].reindex(latitude=lats, longitude=lons, method="nearest")
    except Exception:
        return arr
    n_doy = int(da_q.sizes.get("dayofyear", 365))
    out = np.array(arr, copy=True)
    stacked = out[np.newaxis, ...] if out.ndim == 2 else out
    n_t = stacked.shape[0]
    for t, doy in enumerate(doys[:n_t]):
        di = int(np.clip(doy - 1, 0, n_doy - 1))
        q_sl = np.asarray(da_q.isel(dayofyear=di).values, dtype=np.float64)
        b_sl = np.asarray(da_b.isel(dayofyear=di).values, dtype=np.float64)
        stacked[t] = _qdm_apply_field(stacked[t], q_sl, b_sl)
    return stacked[0] if out.ndim == 2 else stacked


def _apply_ifs_mean_qdm(tx, tn, tg, lats, lons, dates):
    """IFS-only QDM on grids (same TX/DTR/TG rules as the meteogram).

    True Cannon QDM when ``*_ifs_q`` is in the cube; otherwise the mean
    quantile bias. T850 is never corrected. ``tx``/``tn``/``tg`` are Celsius
    ndarrays, 2D or 3D, or None.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(dates)))
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    dates = dates.normalize()
    doys = etccdi_doy_365(dates)

    tx_out = _qdm_correct_grid(tx, lats, lons, doys, "tx_bias")
    tg_out = _qdm_correct_grid(tg, lats, lons, doys, "tg_bias")
    tn_out = None if tn is None else np.asarray(tn, dtype=np.float64)
    if tn_out is not None and tx is not None:
        tx_raw = np.asarray(tx, dtype=np.float64)
        dtr_raw = tx_raw - tn_out
        dtr_corr = _qdm_correct_grid(dtr_raw, lats, lons, doys, "dtr_bias")
        tn_out = None if tx_out is None or dtr_corr is None else tx_out - dtr_corr
    return tx_out, tn_out, tg_out


def _qdm_mean_bias(lat, lon, doys_1_365, bias_var):
    """Fallback: mean over the stored quantile axis (older cubes without Q_IFS)."""
    ds_qdm = _load_qdm_bias_ds()
    doys = np.atleast_1d(np.asarray(doys_1_365, dtype=int))
    if ds_qdm is None or bias_var not in ds_qdm.data_vars:
        return np.zeros(len(doys), dtype=np.float64)
    pt = ds_qdm[bias_var].sel(latitude=lat, longitude=lon, method='nearest')
    by_doy = pt.mean(dim='quantile').values
    idx = np.clip(doys - 1, 0, len(by_doy) - 1)
    return np.nan_to_num(by_doy[idx], nan=0.0)


def _qdm_correct_point(values, lat, lon, doys_1_365, bias_var):
    """Point QDM: true mapping when ``*_ifs_q`` exists, else mean-quantile bias."""
    values = np.asarray(values, dtype=np.float64)
    doys = np.atleast_1d(np.asarray(doys_1_365, dtype=int))
    ds_qdm = _load_qdm_bias_ds()
    if ds_qdm is None or bias_var not in ds_qdm.data_vars:
        return values
    if not _qdm_has_ifs_q(ds_qdm, bias_var):
        return values + _qdm_mean_bias(lat, lon, doys, bias_var)
    q_var = _QDM_IFS_Q_VAR[bias_var]
    pt_q = ds_qdm[q_var].sel(latitude=lat, longitude=lon, method="nearest")
    pt_b = ds_qdm[bias_var].sel(latitude=lat, longitude=lon, method="nearest")
    n_doy = int(pt_q.sizes.get("dayofyear", 365))
    out = np.array(values, dtype=np.float64, copy=True)
    flat = np.atleast_1d(out)
    for i, val in enumerate(flat):
        di = int(np.clip(doys[i] - 1, 0, n_doy - 1))
        q = np.asarray(pt_q.isel(dayofyear=di).values, dtype=np.float64)
        b = np.asarray(pt_b.isel(dayofyear=di).values, dtype=np.float64)
        flat[i] = _qdm_apply_scalar(val, q, b)
    return out


def _squeeze_celsius(values):
    arr = np.squeeze(np.asarray(values, dtype=np.float64))
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.mean(finite)) > 100:
        arr = arr - 273.15
    return arr


def _z500_to_dam(values):
    """Geopotential (m²/s²) → dam, matching ``get_synoptic_map_data``."""
    arr = np.squeeze(np.asarray(values, dtype=np.float64))
    finite = arr[np.isfinite(arr)]
    if finite.size:
        sample = float(np.mean(finite))
        if sample > 10000:
            arr = arr / 9.80665 / 10.0
        elif sample > 2000:
            arr = arr / 10.0
    return arr


def _point_frame_from_master_ds(ds, lat, lon, start, end):
    """1D TX/TN/TG/T850/Z500 at (lat, lon) for [start, end], then drop the rest of the cube."""
    ds = _harmonize_master_archive(ds)
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        ds = ds.rename({"mx2t": "tx"})
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        ds = ds.rename({"mn2t": "tn"})
    keep = [v for v in ("tx", "tn", "tg", "t850", "z500") if v in ds.data_vars]
    if not keep:
        return pd.DataFrame()
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    t_name = "valid_time" if "valid_time" in ds.dims else "time"
    pt = ds[keep].sel({lat_name: lat, lon_name: lon}, method="nearest")
    pt = pt.sel({t_name: slice(start, end)})
    times = pd.to_datetime(pt[t_name].values)
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    days = pd.DatetimeIndex(times).normalize()
    n = len(days)
    tx = _squeeze_celsius(pt["tx"].values) if "tx" in pt else np.full(n, np.nan)
    tn = _squeeze_celsius(pt["tn"].values) if "tn" in pt else np.full(n, np.nan)
    tg = _squeeze_celsius(pt["tg"].values) if "tg" in pt else np.full(n, np.nan)
    t850 = _squeeze_celsius(pt["t850"].values) if "t850" in pt else np.full(n, np.nan)
    z500 = _z500_to_dam(pt["z500"].values) if "z500" in pt else np.full(n, np.nan)
    return pd.DataFrame({"Date": days, "TX": tx, "TN": tn, "TG": tg, "T850": t850, "Z500": z500})


_POINT_EXTRACT_SCRIPT = r"""
import json, sys
import numpy as np, pandas as pd, xarray as xr
path, lat, lon, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
start, end = pd.Timestamp(t0), pd.Timestamp(t1)
ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
try:
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        ds = ds.rename({"mx2t": "tx"})
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        ds = ds.rename({"mn2t": "tn"})
    keep = [v for v in ("tx", "tn", "tg", "t850", "z500") if v in ds.data_vars]
    if not keep:
        raise SystemExit(2)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    t_name = "valid_time" if "valid_time" in ds.dims else "time"
    pt = ds[keep].sel({lat_name: lat, lon_name: lon}, method="nearest").sel({t_name: slice(start, end)})
    times = pd.to_datetime(pt[t_name].values)
    days = pd.DatetimeIndex(times).tz_localize(None).normalize() if getattr(times, "tz", None) else pd.DatetimeIndex(times).normalize()
    def _c(v):
        if v not in pt:
            return [None] * len(days)
        a = np.squeeze(np.asarray(pt[v].values, dtype=float))
        a = np.atleast_1d(a)
        finite = a[np.isfinite(a)]
        if finite.size and float(np.mean(finite)) > 100:
            a = a - 273.15
        return [None if not np.isfinite(x) else float(x) for x in a]
    def _z(v):
        if v not in pt:
            return [None] * len(days)
        a = np.squeeze(np.asarray(pt[v].values, dtype=float))
        a = np.atleast_1d(a)
        finite = a[np.isfinite(a)]
        if finite.size:
            sample = float(np.mean(finite))
            if sample > 10000:
                a = a / 9.80665 / 10.0
            elif sample > 2000:
                a = a / 10.0
        return [None if not np.isfinite(x) else float(x) for x in a]
    json.dump({"Date": [d.strftime("%Y-%m-%d") for d in days], "TX": _c("tx"), "TN": _c("tn"), "TG": _c("tg"), "T850": _c("t850"), "Z500": _z("z500")}, sys.stdout)
finally:
    ds.close()
"""


def _point_frame_from_master_file(path, lat, lon, start, end, isolate=False):
    """Read one yearly master file at a single grid point.

    Current-year files are often mid-write (ERA5T updater). HDF5 can abort the
    whole process on a bad global-heap checksum — that cannot be caught in
    Python — so the current year is read in a child process.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if isolate:
        env = os.environ.copy()
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _POINT_EXTRACT_SCRIPT, str(path), str(lat), str(lon),
                 pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat()],
                capture_output=True, text=True, timeout=90, env=env, creationflags=flags,
            )
        except (subprocess.TimeoutExpired, OSError):
            return pd.DataFrame()
        if proc.returncode != 0 or not proc.stdout.strip():
            return pd.DataFrame()
        payload = json.loads(proc.stdout)
        df = pd.DataFrame(payload)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    ds = None
    try:
        ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
        return _point_frame_from_master_ds(ds, lat, lon, start, end)
    except Exception:
        return pd.DataFrame()
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def _last_finite(series):
    """Latest non-null in a Date-grouped column (forecast overlay must not wipe archive T850)."""
    arr = series.to_numpy()
    ok = pd.notna(arr)
    if not ok.any():
        return np.nan
    return arr[ok][-1]


@st.cache_data(show_spinner=False)
def get_live_point_series(lat, lon, forecast_model=FORECAST_MODEL_IFS, _series_version=12):
    """
    Point daily TX/TN/TG/T850/Z500 for the Point Meteogram. Does NOT open the maps
    spatial cube (load_global_datasets): that concatenates 2025+2026 and then
    .compute()s every field at the point, which aborted Streamlit on a
    corrupted HDF5 heap in era5_master_daily_2026.nc.

    Each covering year is opened alone, only the point fields are
    read, and the current calendar year is isolated in a subprocess.
    """
    end = pd.Timestamp.utcnow().tz_localize(None).floor("D") + pd.Timedelta(days=10)
    start = end - pd.Timedelta(days=375)
    this_year = int(end.year)
    frames = []
    for year in range(int(start.year), int(end.year) + 1):
        path = DATA_ROOT / "Master_Batches" / f"era5_master_daily_{year}.nc"
        frames.append(_point_frame_from_master_file(
            path, lat, lon, start, end, isolate=(year == this_year),
        ))

    lf = get_live_txtn_ds(forecast_model=forecast_model)
    if lf is not None:
        try:
            with st.session_state.nc_lock:
                pt_lf = lf.sel(latitude=lat, longitude=lon, method="nearest")
                pt_lf = pt_lf.sel(valid_time=slice(start, end)) if "valid_time" in pt_lf.dims else pt_lf
            f_times = pd.to_datetime(pt_lf.valid_time.values)
            if getattr(f_times, "tz", None) is not None:
                f_times = f_times.tz_convert("UTC").tz_localize(None)
            f_days = pd.DatetimeIndex(f_times).normalize()
            cut = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=LIVE_OVERLAY_PAST_DAYS)
            keep_fcst = np.asarray(f_days >= cut)
            if not keep_fcst.any():
                raise RuntimeError("live forecast has no days in the overlay window")
            if "valid_time" in pt_lf.dims:
                pt_lf = pt_lf.isel(valid_time=np.flatnonzero(keep_fcst))
            f_times = f_times[keep_fcst]
            f_days = f_days[keep_fcst]
            f_doys = etccdi_doy_365(f_days)
            # IFS-vs-ERA5 transfer only. AIFS is trained on ERA5 — do not apply.
            # T850 is free atmosphere and is never QDM-corrected.
            apply_qdm = "AIFS" not in str(forecast_model)
            def _corr(vals, var):
                if not apply_qdm:
                    return vals
                return _qdm_correct_point(vals, lat, lon, f_doys, var)
            tx_name = "tx" if "tx" in pt_lf.data_vars else "mx2t"
            tn_name = "tn" if "tn" in pt_lf.data_vars else "mn2t"
            t850_vals = (
                _squeeze_celsius(pt_lf["t850"].values)
                if "t850" in pt_lf.data_vars else np.full(len(f_days), np.nan)
            )
            z500_vals = (
                _z500_to_dam(pt_lf["z500"].values)
                if "z500" in pt_lf.data_vars else np.full(len(f_days), np.nan)
            )
            if tx_name in pt_lf.data_vars and tn_name in pt_lf.data_vars:
                tx_raw = _squeeze_celsius(pt_lf[tx_name].values)
                tn_raw = _squeeze_celsius(pt_lf[tn_name].values)
                tx_corr = _corr(tx_raw, "tx_bias")
                dtr_corr = _corr(tx_raw - tn_raw, "dtr_bias")
                if "tg" in pt_lf.data_vars:
                    tg_corr = _corr(_squeeze_celsius(pt_lf["tg"].values), "tg_bias")
                else:
                    tg_corr = np.full(len(f_days), np.nan)
                frames.append(pd.DataFrame({
                    "Date": f_days, "TX": tx_corr, "TN": tx_corr - dtr_corr, "TG": tg_corr,
                    "T850": t850_vals, "Z500": z500_vals,
                }))
            elif "tg" in pt_lf.data_vars:
                tg_corr = _corr(_squeeze_celsius(pt_lf["tg"].values), "tg_bias")
                frames.append(pd.DataFrame({
                    "Date": f_days, "TX": np.nan, "TN": np.nan, "TG": tg_corr,
                    "T850": t850_vals, "Z500": z500_vals,
                }))
            elif "t850" in pt_lf.data_vars or "z500" in pt_lf.data_vars:
                frames.append(pd.DataFrame({
                    "Date": f_days, "TX": np.nan, "TN": np.nan, "TG": np.nan,
                    "T850": t850_vals, "Z500": z500_vals,
                }))
        except Exception:
            pass

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date")
    value_cols = [c for c in ("TX", "TN", "TG", "T850", "Z500") if c in df.columns]
    df = df.groupby("Date", as_index=False)[value_cols].agg(_last_finite)
    # 29 Feb stays VISIBLE on the live chart (real ERA5/IFS value plotted on
    # its real calendar date) — only the 365-day BASELINE array excises it.
    # Reindex on the full Gregorian calendar so true ERA5 holes stay NaN.
    # Never interpolate — AtmoPulse does not invent temperature peaks.
    full_index = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    return df.set_index("Date").reindex(full_index).rename_axis("Date").reset_index()


# --- ARCHIVE YEAR (Meteogram: closed calendar-year ERA5, no forecast) ---
def _archive_year_file(year: int) -> Path:
    return DATA_ROOT / "Master_Batches" / f"era5_master_daily_{year}.nc"


def _archive_year_is_complete(year: int) -> bool:
    """True iff era5_master_daily_{year}.nc exists and its last covered day
    reaches 31 Dec of `year` with real (non-placeholder) ERA5/ERA5T data —
    used to keep an in-progress year out of the *closed*-year ceiling of
    the Meteogram Archive Year selector (the current year is appended
    separately when the file exists)."""
    path = _archive_year_file(year)
    if not path.exists():
        return False
    ds = None
    try:
        ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
        ds = _harmonize_master_archive(ds)
        t_name = "valid_time" if "valid_time" in ds.dims else "time"
        if t_name not in ds.coords:
            return False
        times = pd.to_datetime(ds[t_name].values)
        if getattr(times, "tz", None) is not None:
            times = times.tz_convert("UTC").tz_localize(None)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        if pd.DatetimeIndex(times).max().normalize() < year_end:
            return False
        var = next((v for v in ("tg", "tx", "mx2t") if v in ds.data_vars), None)
        if var is None:
            return False
        last_day = ds[var].sel({t_name: slice(year_end, year_end)})
        return bool(np.isfinite(np.asarray(last_day.values, dtype=np.float64)).any())
    except Exception:
        return False
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def _archive_year_mtime(year: int) -> float:
    path = _archive_year_file(year)
    try:
        return float(path.stat().st_mtime) if path.exists() else 0.0
    except OSError:
        return 0.0


def _last_finite_temp_day(df) -> pd.Timestamp | None:
    """Latest calendar day with a real TX/TN/TG value (not an ERA5T placeholder)."""
    if df is None or df.empty or "Date" not in df.columns:
        return None
    temp_cols = [c for c in ("TX", "TN", "TG") if c in df.columns]
    if not temp_cols:
        return None
    dates = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    finite = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce") for c in temp_cols},
    ).notna().any(axis=1)
    if not bool(finite.any()):
        return None
    return pd.Timestamp(dates[finite].max()).normalize()


def get_archive_year_options(today_str: str):
    """Meteogram Archive Year dropdown: complete ERA5 years plus the
    in-progress current year when its master file exists.

    ``today_str`` and the current-year file mtime are cache-key nonces so a
    new day, the 10-Jan cutoff, or an ERA5T append actually refreshes the list.
    """
    today = pd.Timestamp(today_str)
    return _cached_archive_year_options(
        str(today_str), _archive_year_mtime(int(today.year)),
    )


@st.cache_data(show_spinner=False)
def _cached_archive_year_options(today_str: str, current_mtime: float, _version=2):
    del current_mtime  # cache-key nonce for ERA5T appends of the current year
    """Complete years 1940..last closed year, then any later year whose
    ``era5_master_daily_{year}.nc`` exists (typically the current year).

    Year-release rule for the *closed* ceiling: `today.year - 1` is only a
    candidate once the calendar day is >= 10 Jan (UTC); before that the
    ceiling is `today.year - 2` (the prior year's ERA5T tail is not considered
    settled yet). The ceiling then steps back until a year is complete through
    31 Dec. The current (or otherwise still-open) year is appended separately
    if the file is on disk — its series is truncated to the last finite ERA5
    day, never filled with IFS.
    """
    today = pd.Timestamp(today_str)
    candidate = today.year - 1 if (today.month > 1 or today.day >= 10) else today.year - 2
    last_complete = None
    for year in range(candidate, 1939, -1):
        if _archive_year_is_complete(year):
            last_complete = year
            break
    years = list(range(1940, last_complete + 1)) if last_complete is not None else []
    for year in range((last_complete or 1939) + 1, int(today.year) + 1):
        if _archive_year_file(year).exists():
            years.append(year)
    return years


def get_archive_year_point_series(lat, lon, year, _series_version=2):
    """ERA5/ERA5T point series for one calendar year: no IFS/AIFS overlay.

    Closed years are 1 Jan–31 Dec. The in-progress current year is read in a
    child process (HDF5 abort safety) and truncated at the last finite ERA5
    day. ``source_mtime`` is in the inner cache key so ERA5T appends refresh.
    """
    return _cached_archive_year_point_series(
        float(lat), float(lon), int(year), _archive_year_mtime(int(year)), _series_version,
    )


@st.cache_data(show_spinner=False)
def _cached_archive_year_point_series(lat, lon, year, source_mtime, _series_version=2):
    del source_mtime  # cache-key nonce; ERA5T appends change the file mtime
    path = _archive_year_file(year)
    start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)
    this_year = int(pd.Timestamp.utcnow().tz_localize(None).year)
    isolate = int(year) == this_year
    df = _point_frame_from_master_file(path, lat, lon, start, year_end, isolate=isolate)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    last = _last_finite_temp_day(df)
    if last is None:
        return pd.DataFrame()
    end = min(year_end, last)
    df = df[df["Date"] <= end]
    full_index = pd.date_range(start, end, freq="D")
    return df.set_index("Date").reindex(full_index).rename_axis("Date").reset_index()


def get_archive_window_point_series(lat, lon, start, end):
    """ERA5/ERA5T point series covering ``start``..``end`` (inclusive).

    Concatenates cached whole-year frames so a Live-aligned seasonal window
    that crosses 1 January does not require a new NetCDF path. Years without
    a master file contribute nothing; a fully missing window is an empty
    DataFrame.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        start, end = end, start
    frames = []
    for year in range(int(start.year), int(end.year) + 1):
        df = get_archive_year_point_series(lat, lon, year)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None).dt.normalize()
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date")
    return out.loc[(out["Date"] >= start) & (out["Date"] <= end)].reset_index(drop=True)


def _array_has_finite(val) -> bool:
    if val is None:
        return False
    arr = np.asarray(getattr(val, "values", val))
    return bool(np.isfinite(arr).any())


def synoptic_source_mtime(date_str, forecast_model=FORECAST_MODEL_IFS) -> float:
    """Latest mtime among the on-disk files that actually feed the Map
    Tracker synoptic fields for `date_str`: the newest matching live-forecast
    file (ifs_daily_forecast_*.nc / aifs_daily_forecast_*.nc, or the legacy
    live_forecast_*.nc bridge files for IFS) plus era5_master_daily_{year}.nc
    for the target year.

    Schritt C: used as an explicit, hashable @st.cache_resource /
    @st.cache_data key component (`source_mtime=`) so a freshly-downloaded
    forecast run — same filename, replaced on disk, so its own mtime is the
    only thing that changes — actually invalidates fetch_cached_synoptic_data
    and the map-tracker analytics that key off it, instead of silently
    serving a stale in-memory result for the same date_str/toggles. Returns
    0.0 (never crashes, never blocks) when nothing is found — callers treat
    that as "no fresher file known", not as an error.
    """
    live_dir = DATA_ROOT / "Live_Forecasts"
    is_aifs = "AIFS" in str(forecast_model)
    pattern = "aifs_daily_forecast_*.nc" if is_aifs else "ifs_daily_forecast_*.nc"
    mtimes = [p.stat().st_mtime for p in live_dir.glob(pattern) if p.exists()]
    if not is_aifs:
        # Legacy IFS bridge files (see backend_maps._open_live_forecast_ds).
        for name in ("live_forecast_mslp.nc", "live_forecast_z500.nc", "live_forecast_txtn.nc"):
            p = live_dir / name
            if p.exists():
                mtimes.append(p.stat().st_mtime)
    try:
        year = pd.to_datetime(date_str).year
    except Exception:
        year = None
    if year is not None:
        master_path = DATA_ROOT / "Master_Batches" / f"era5_master_daily_{year}.nc"
        if master_path.exists():
            mtimes.append(master_path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


@st.cache_resource(show_spinner=False, max_entries=10)
def fetch_cached_synoptic_data(
    date_str, anchor_date_str=None, forecast_model=FORECAST_MODEL_IFS,
    needed_vars=None, source_mtime=0.0, _loader_version=15,
):
    """
    `source_mtime` (Schritt C): hashable cache-key component, see
    `synoptic_source_mtime()`. @st.cache_resource is kept (not swapped to
    cache_data) precisely BECAUSE mtime is now in the key: a new forecast
    file gets its own cache entry instead of the old one going stale, so the
    "no defensive copy on every read" performance property cache_resource
    gives this large dict-of-ndarrays return value is preserved. `max_entries
    =10` bounds memory the same way it always did; entries keyed to a
    superseded mtime simply age out via the existing LRU eviction.
    """
    if needed_vars is not None:
        needed_vars = tuple(needed_vars)
    with st.session_state.nc_lock:
        if anchor_date_str is not None:
            set_synoptic_anchor(anchor_date_str, SLIDER_PAD_PAST, SLIDER_PAD_FUTURE, forecast_model=forecast_model)
        data = get_synoptic_map_data(
            date_str, forecast_model=forecast_model, needed_vars=needed_vars,
        )
        meta = data.pop("_meta", {})
        packed = {}
        sample = next((data[k] for k in ("mslp", "tg", "tx", "tn", "z500", "t850") if k in data), None)
        if sample is None:
            sample = next((v for v in data.values() if hasattr(v, "longitude")), None)
        if sample is not None and hasattr(sample, "longitude"):
            packed["_lons"] = np.asarray(sample.longitude.values)
            packed["_lats"] = np.asarray(sample.latitude.values)
        for key, val in data.items():
            packed[key] = _synoptic_array(val)
        if meta.get("ifs_live") and "_lats" in packed and "_lons" in packed:
            day = meta.get("actual_time") or pd.to_datetime(date_str)
            packed["tx"], packed["tn"], packed["tg"] = _apply_ifs_mean_qdm(
                packed.get("tx"), packed.get("tn"), packed.get("tg"),
                packed["_lats"], packed["_lons"], day,
            )
        meta["temps_available"] = any(
            _array_has_finite(packed.get(name)) for name in ("tx", "tn", "tg", "t850")
        )
        return packed, meta


@st.cache_data(show_spinner=False)
def get_persistence_arrays(
    target_date_str, baseline_type, map_var="TG", anchor_date_str=None,
    forecast_model=FORECAST_MODEL_IFS, _persist_version=4,
):
    ref_clim = load_reference_climatology()
    if ref_clim is None: 
        return None
    if "AIFS" in str(forecast_model) and map_var in ("TX", "TN"):
        return None
    end_date = pd.to_datetime(target_date_str)
    start_date = end_date - pd.Timedelta(days=PERSISTENCE_MAX_DAYS + PERSISTENCE_LOOKBACK_PAD)
    loaded = _load_persistence_daily_series(
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        anchor_date_str, forecast_model=forecast_model,
    )
    if loaded is None or loaded[0] is None: 
        return None
    payload, _meta = loaded
    daily_dates, tx_vals, tn_vals = payload[0], payload[1], payload[2]
    t850_vals = payload[3] if len(payload) > 3 else None
    tg_vals = payload[4] if len(payload) > 4 else None
    if tx_vals is None and tn_vals is None and t850_vals is None and tg_vals is None:
        return None

    def _to_celsius(arr):
        if arr is None:
            return None
        hist = arr.astype(np.float64)
        finite_mean = np.nanmean(hist)
        if np.isfinite(finite_mean) and finite_mean > 100:
            hist -= 273.15
        return hist

    tx_hist, tn_hist, t850_hist = _to_celsius(tx_vals), _to_celsius(tn_vals), _to_celsius(t850_vals)
    tg_hist = _to_celsius(tg_vals)
    dates_dt = pd.to_datetime(daily_dates)
    doys = etccdi_doy_365(dates_dt)
    suffix = "A" if baseline_type == "A" else "B"
    shape_src = next(a for a in (tx_hist, tn_hist, t850_hist, tg_hist) if a is not None)
    n_days, n_lats, n_lons = shape_src.shape
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in ref_clim.variables: 
            return ref_clim[var_key].values
        return np.full((365, n_lats, n_lons), fallback)

    if map_var == "TX":
        if tx_hist is None:
            return None
        v_h, v_p95, v_p90, v_p75 = tx_hist, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tx_max_val'), safe_get('tx_min_val')
    elif map_var == "TN":
        if tn_hist is None:
            return None
        v_h, v_p95, v_p90, v_p75 = tn_hist, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tn_max_val'), safe_get('tn_min_val')
    elif map_var == "T850":
        if t850_hist is None or not np.isfinite(t850_hist).any():
            return None
        v_h = t850_hist
        v_p95, v_p90, v_p75 = safe_get(f't850_p95_doy_{suffix}'), safe_get(f't850_p90_doy_{suffix}'), safe_get(f't850_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f't850_p25_doy_{suffix}'), safe_get(f't850_p10_doy_{suffix}'), safe_get(f't850_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('t850_max_val'), safe_get('t850_min_val')
    else:
        if tg_hist is None:
            return None
        v_h = tg_hist
        v_p95, v_p90, v_p75 = safe_get(f'tg_p95_doy_{suffix}'), safe_get(f'tg_p90_doy_{suffix}'), safe_get(f'tg_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tg_p25_doy_{suffix}'), safe_get(f'tg_p10_doy_{suffix}'), safe_get(f'tg_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tg_max_val'), safe_get('tg_min_val')

    streaks = np.zeros((8, n_lats, n_lons), dtype=int)
    exc = np.zeros((8, n_days, n_lats, n_lons), dtype=bool)
    
    for i, d in enumerate(doys):
        d_idx = d - 1
        exc[0, i], exc[1, i], exc[2, i], exc[3, i] = v_h[i] >= v_p75[d_idx], v_h[i] >= v_p90[d_idx], v_h[i] >= v_p95[d_idx], v_h[i] >= v_r_w[d_idx]
        exc[4, i], exc[5, i], exc[6, i], exc[7, i] = v_h[i] <= v_p25[d_idx], v_h[i] <= v_p10[d_idx], v_h[i] <= v_p5[d_idx], v_h[i] <= v_r_c[d_idx]
        
    for lvl in range(8): 
        streaks[lvl] = np.sum(np.cumprod(exc[lvl][::-1, :, :], axis=0), axis=0)
    return streaks


def _record_window_doys(target_doy: int) -> list[int]:
    window = []
    for offset in range(-2, 3):
        d = target_doy + offset
        if d < 1:
            d += 365
        elif d > 365:
            d -= 365
        window.append(d)
    return window


def _extreme_with_year(vals: np.ndarray, yrs, reducer) -> tuple[np.ndarray, np.ndarray]:
    """Grid-wise extreme value and the year it occurred (NaN-safe)."""
    yrs = np.asarray(yrs)
    val = reducer(vals, axis=0)
    idx = np.nanargmax(vals, axis=0) if reducer is np.nanmax else np.nanargmin(vals, axis=0)
    yr_grid = np.broadcast_to(yrs[:, None, None], vals.shape)
    yr = np.take_along_axis(yr_grid, np.expand_dims(idx, axis=0), axis=0).squeeze()
    val = np.where(np.isfinite(val), val, np.nan)
    yr = np.where(np.isfinite(val), yr, np.nan)
    return val, yr


@st.cache_resource(show_spinner=False)
def get_map_historical_records_bundle(target_doys: tuple, cutoff_year: int):
    """
    SINGLETON CACHE (recomputed only when the slider's reachable day-of-year
    set or the cutoff year changes, i.e. effectively once per calendar day):
    all-time warm/cold grids from the ERA5 archive, strictly before
    cutoff_year, for EVERY day-of-year reachable via the slider, computed in a
    single archive pass. The archive's on-disk chunking means even one day's
    lazy .load() must decompress a multi-hundred-MB block; batching every
    slider-reachable day-of-year into one shared scan turns up to 13
    full-chunk decompression passes per Prev/Next Day click into exactly one.
    """
    ds = get_master_archive_ds()
    if ds is None:
        return None
    window_by_doy = {d: _record_window_doys(d) for d in target_doys}
    union_doys = sorted({w for ws in window_by_doy.values() for w in ws})
    with st.session_state.nc_lock:
        vt = pd.DatetimeIndex(pd.to_datetime(ds.valid_time.values))
        # 29 Feb is a real candidate record day too (mapped into 1 March's
        # ETCCDI window) — it must not be excluded from actual-data scans.
        etccdi = etccdi_doy_365(vt)
        mask = np.isin(etccdi, union_doys) & (vt.year < cutoff_year)
        if not mask.any():
            return None
        sub = ds.isel(valid_time=mask).load()

    tx_all = sub["tx"].values.astype(np.float64) - 273.15
    tn_all = sub["tn"].values.astype(np.float64) - 273.15
    tg_all = (
        sub["tg"].values.astype(np.float64) - 273.15
        if "tg" in sub.data_vars
        else np.full_like(tx_all, np.nan)
    )
    doy_all = etccdi_doy_365(pd.to_datetime(sub.valid_time.values))
    yr_all = pd.to_datetime(sub.valid_time.values).year

    bundle = {}
    for d, window in window_by_doy.items():
        m = np.isin(doy_all, window)
        if not m.any():
            continue
        yrs = yr_all[m]
        bundle[d] = {
            "TX": (*_extreme_with_year(tx_all[m], yrs, np.nanmax), *_extreme_with_year(tx_all[m], yrs, np.nanmin)),
            "TN": (*_extreme_with_year(tn_all[m], yrs, np.nanmax), *_extreme_with_year(tn_all[m], yrs, np.nanmin)),
            "TG": (*_extreme_with_year(tg_all[m], yrs, np.nanmax), *_extreme_with_year(tg_all[m], yrs, np.nanmin)),
        }
    return bundle


# --- METEOGRAM CORE DATA ---
def _clim_doy_arr(pt_clim, key, doys):
    doys = np.asarray(doys, dtype=np.int64)
    if key not in pt_clim.variables:
        return np.full(doys.shape, np.nan, dtype=np.float64)
    return np.asarray(pt_clim[key].values, dtype=np.float64)[doys]


def point_clim_ladder(pt_clim, doys, meteo_var, epoch):
    """Warm/cold percentile ladder + reference value for one meteogram variable.

    Same definitions as ``compute_point_thresholds`` (the scalar form used by
    the narrative), so text, hover, and colour fills cannot disagree:

    * TX — TX percentiles / records
    * TN — TN percentiles / records
    * T850 — 850 hPa temperature percentiles / records
    * TG — native TG percentiles / records (`tg_p*_doy_*`, `tg_max_val` /
      `tg_min_val`). Never the mean of the TX and TN ladders.

    ``c_base`` is the midpoint of that variable's Moderate-warm and
    Moderate-cold bounds (the "average" the Above/Below-avg fills sit on).
    """
    doys = np.asarray(doys, dtype=np.int64)
    ep = epoch

    def a(key):
        return _clim_doy_arr(pt_clim, key, doys)

    code = meteo_var_code(meteo_var)
    if code == "TX":
        p75, p90, p95 = a(f"tx_p75_doy_{ep}"), a(f"tx_p90_doy_{ep}"), a(f"tx_p95_doy_{ep}")
        rec_w = a("tx_max_val")
        p25, p10, p5 = a(f"tx_p25_doy_{ep}"), a(f"tx_p10_doy_{ep}"), a(f"tx_p5_doy_{ep}")
        rec_c = a("tx_min_val")
    elif code == "TN":
        p75, p90, p95 = a(f"tn_p75_doy_{ep}"), a(f"tn_p90_doy_{ep}"), a(f"tn_p95_doy_{ep}")
        rec_w = a("tn_max_val")
        p25, p10, p5 = a(f"tn_p25_doy_{ep}"), a(f"tn_p10_doy_{ep}"), a(f"tn_p5_doy_{ep}")
        rec_c = a("tn_min_val")
    elif code == "T850":
        p75, p90, p95 = a(f"t850_p75_doy_{ep}"), a(f"t850_p90_doy_{ep}"), a(f"t850_p95_doy_{ep}")
        rec_w = a("t850_max_val")
        p25, p10, p5 = a(f"t850_p25_doy_{ep}"), a(f"t850_p10_doy_{ep}"), a(f"t850_p5_doy_{ep}")
        rec_c = a("t850_min_val")
    else:
        p75, p90, p95 = a(f"tg_p75_doy_{ep}"), a(f"tg_p90_doy_{ep}"), a(f"tg_p95_doy_{ep}")
        rec_w = a("tg_max_val")
        p25, p10, p5 = a(f"tg_p25_doy_{ep}"), a(f"tg_p10_doy_{ep}"), a(f"tg_p5_doy_{ep}")
        rec_c = a("tg_min_val")
    c_base = (p75 + p25) / 2.0
    return c_base, p75, p90, p95, rec_w, p25, p10, p5, rec_c


def compute_point_thresholds(ref_clim, lat, lon, target_date, meteo_var, epoch):
    """
    ETCCDI percentile + all-time-record thresholds for one coordinate/day,
    shaped as (p_warm, p_cold) for backend_narrative.classify_point_severity().
    Uses the same ladder as frontend_plots.get_meteogram_traces() so the
    Point Meteogram narrative always matches the chart colour fills.
    """
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method="nearest")
    doys = np.atleast_1d(etccdi_doy_365(pd.Timestamp(target_date)) - 1)
    _c_base, p75, p90, p95, rec_w, p25, p10, p5, rec_c = point_clim_ladder(
        pt_clim, doys, meteo_var, epoch
    )
    p_warm = {"p75": float(p75[0]), "p90": float(p90[0]), "p95": float(p95[0]), "rec": float(rec_w[0])}
    p_cold = {"p25": float(p25[0]), "p10": float(p10[0]), "p5": float(p5[0]), "rec": float(rec_c[0])}
    return p_warm, p_cold


def _series_frame_from_point(pt_series):
    """Shared tail-end: DataArray point series -> the 'time'/'val'/'year'/'doy'
    DataFrame contract both the Zarr and legacy NetCDF paths must return."""
    raw = np.asarray(pt_series.values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size > 0 and np.nanmean(finite) > 100:
        raw = raw - 273.15

    df = pd.DataFrame({'time': pt_series.valid_time.values, 'val': raw}).drop_duplicates(subset=['time'])
    dates = pd.to_datetime(df['time'])
    df['year'] = dates.dt.year

    # ETCCDI 365-day mapping for historical extremes — fully vectorized.
    # 29 Feb keeps its own row/value (it can still set an actual "Record" or
    # count into the yearly bars); it is only ever excised from the 365-day
    # BASELINE percentile array, never from this real-data table.
    df['doy'] = etccdi_doy_365(dates) - 1  # 0-based for array indexing
    return df


def _archive_var_candidates(var_code: str) -> tuple[str, ...]:
    code = meteo_var_code(var_code) if var_code not in ("TX", "TN", "TG", "T850") else str(var_code)
    if code == "T850":
        return ("t850",)
    if code == "TN":
        return ("tn", "mn2t")
    if code == "TG":
        return ("tg",)
    return ("tx", "mx2t")


def _load_point_archive_series_from_zarr(lat, lon, var_code):
    """Millisecond-scale point read from the temporally-chunked Zarr mirror
    (see batch_convert_netcdf_to_zarr.py / ZARR_MASTER_TIME_SERIES). Returns
    None on ANY problem (missing store, missing variable, corrupt/partial
    write, consolidated-metadata mismatch, etc.) so the caller falls back to
    the legacy NetCDF path unconditionally."""
    if not ZARR_MASTER_TIME_SERIES.exists():
        return None

    candidates = _archive_var_candidates(var_code)
    try:
        zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=True)
    except Exception:
        try:
            zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=False)
        except Exception:
            return None

    var_name = None
    for candidate in candidates:
        if candidate in zds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        return None

    with st.session_state.nc_lock:
        try:
            pt_series = zds[var_name].sel(latitude=lat, longitude=lon, method='nearest').compute()
        except Exception:
            return None
    return _series_frame_from_point(pt_series)


# --- DATETIME64 CRASH BUGFIX ---
# PERFORMANCE: the master archive point extraction (ds.sel(...).compute()) is
# the expensive step (multi-minute NetCDF/dask read for a fresh point) and is
# completely EPOCH-INDEPENDENT — only the climatology percentile lookup below
# depends on epoch "A" vs "B". Previously this ran TWICE per point (once per
# epoch, since epoch was baked into build_yearly_extremes_chart's own cache
# key), doubling the wait. Splitting it into its own @st.cache_data step means
# the raw series is read from disk once per (lat, lon, var_code) and reused for
# both epoch A and epoch B charts.
#
# STORAGE ENGINE: this now tries the point-extraction-optimal Zarr mirror
# first (millisecond reads — see batch_convert_netcdf_to_zarr.py) and only
# falls back to the legacy spatially-chunked NetCDF archive if that store
# hasn't been built yet, or the Zarr read fails for any reason. The fallback
# path below is intentionally IDENTICAL to the previous implementation, and
# both paths return the exact same 'time'/'val'/'year'/'doy' DataFrame shape.
@st.cache_data(show_spinner=False)
def _load_point_archive_series(lat, lon, var_code, _archive_version=6):
    zarr_df = _load_point_archive_series_from_zarr(lat, lon, var_code)
    if zarr_df is not None:
        return zarr_df

    ds = get_master_archive_ds()
    if ds is None:
        return None

    # Resolve the variable name from the (lazy, uncomputed) Dataset schema
    # FIRST — this costs nothing, no I/O — so we can subset to that ONE
    # variable before touching .sel()/.compute(). Selecting the point on the
    # full multi-variable Dataset first would force every other archived
    # field (at that point) to be read/materialized for nothing.
    var_name = None
    candidates = _archive_var_candidates(var_code)
    for candidate in candidates:
        if candidate in ds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        return None

    # Pushdown: (variable, then point) selection on the still-lazy DataArray
    # — only the single (lat, lon) time series for `var_name` is ever pulled
    # off disk/decompressed, never the full spatial grid or unrelated fields.
    with st.session_state.nc_lock:
        try:
            pt_series = ds[var_name].sel(latitude=lat, longitude=lon, method='nearest').compute()
        except Exception:
            return None

    return _series_frame_from_point(pt_series)


