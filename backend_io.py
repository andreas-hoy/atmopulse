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

from backend_analytics import _synoptic_array
from backend_maps import (
    LIVE_OVERLAY_PAST_DAYS,
    _open_synoptic_range,
    drop_era5t_aux,
    etccdi_doy_365,
    get_synoptic_map_data,
    set_synoptic_anchor,
)
from backend_waves import get_kiesely_waves_figs
from config import DATA_ROOT, FORECAST_MODEL_IFS, SLIDER_PAD_FUTURE, SLIDER_PAD_PAST

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
    forecast_model=FORECAST_MODEL_IFS, _loader_version=9,
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
def _load_persistence_daily_series(start_date_str, end_date_str, anchor_date_str=None, forecast_model=FORECAST_MODEL_IFS):
    """Build a daily TX/TN cube from ERA5 masters; IFS/AIFS only in the last 6 days + forecast."""
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
                if "tx" in sub.data_vars and "tn" in sub.data_vars:
                    tx_d = sub['tx'].groupby('valid_time.date').max()
                    tn_d = sub['tn'].groupby('valid_time.date').min()
                    for i, d in enumerate(tx_d['date'].values):
                        day = pd.Timestamp(d).normalize()
                        by_date[day] = (tx_d.values[i], tn_d.values[i])
                elif "tg" in sub.data_vars:
                    tg_d = sub['tg'].groupby('valid_time.date').mean()
                    for i, d in enumerate(tg_d['date'].values):
                        day = pd.Timestamp(d).normalize()
                        by_date[day] = (tg_d.values[i], tg_d.values[i])

    archive_max = max((d for d in by_date if d < overlay_cut), default=None)

    eligible = sorted(d for d in by_date if start_date <= d <= end_date)
    if not eligible:
        return None, {"archive_max": archive_max, "effective_end": None, "uses_ifs": False, "has_gap": False}

    eligible = eligible[-60:]
    tx_vals = np.stack([by_date[d][0] for d in eligible])
    tn_vals = np.stack([by_date[d][1] for d in eligible])
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
    return (np.array(eligible), tx_vals, tn_vals), meta


# --- QDM BIAS CORRECTION ---
@st.cache_resource(show_spinner=False)
def _load_qdm_bias_ds():
    """
    Optional IFS-vs-ERA5 QDM bias cube (see calculate_qdm_bias.py). Returns
    None — a documented zero-bias passthrough — until that build script has
    been run (it requires archived IFS_Hindcasts/*.nc, which are not part of
    this deployment yet).
    """
    if not QDM_TRANSFER_FILE.exists():
        return None
    try:
        return xr.open_dataset(QDM_TRANSFER_FILE, engine='netcdf4')
    except Exception:
        return None


def _qdm_mean_bias(lat, lon, doys_1_365, bias_var):
    """
    Per-day-of-year QDM bias for one point, averaged across the stored
    quantile axis. The transfer cube persists only the BIAS at each
    empirical quantile (q_era5 - q_ifs), not the raw IFS quantile VALUES
    needed to rank a brand-new forecast value into a quantile bin — so a
    full per-value quantile-mapping isn't reconstructible from this artifact
    alone. Averaging over quantiles yields the mean systematic bias for that
    calendar day, a documented simplification of true QDM. Returns zeros
    (no-op) when the cube isn't available.
    """
    ds_qdm = _load_qdm_bias_ds()
    if ds_qdm is None or bias_var not in ds_qdm.data_vars:
        return np.zeros(len(doys_1_365), dtype=np.float64)
    pt = ds_qdm[bias_var].sel(latitude=lat, longitude=lon, method='nearest')
    by_doy = pt.mean(dim='quantile').values  # shape (365,)
    idx = np.clip(np.asarray(doys_1_365) - 1, 0, len(by_doy) - 1)
    return np.nan_to_num(by_doy[idx], nan=0.0)


def _squeeze_celsius(values):
    arr = np.squeeze(np.asarray(values, dtype=np.float64))
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.mean(finite)) > 100:
        arr = arr - 273.15
    return arr


def _point_frame_from_master_ds(ds, lat, lon, start, end):
    """1D TX/TN/TG at (lat, lon) for [start, end], then drop the rest of the cube."""
    ds = _harmonize_master_archive(ds)
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        ds = ds.rename({"mx2t": "tx"})
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        ds = ds.rename({"mn2t": "tn"})
    keep = [v for v in ("tx", "tn", "tg") if v in ds.data_vars]
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
    tx = _squeeze_celsius(pt["tx"].values) if "tx" in pt else np.full(len(days), np.nan)
    tn = _squeeze_celsius(pt["tn"].values) if "tn" in pt else np.full(len(days), np.nan)
    tg = _squeeze_celsius(pt["tg"].values) if "tg" in pt else (tx + tn) / 2.0
    return pd.DataFrame({"Date": days, "TX": tx, "TN": tn, "TG": tg})


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
    keep = [v for v in ("tx", "tn", "tg") if v in ds.data_vars]
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
    json.dump({"Date": [d.strftime("%Y-%m-%d") for d in days], "TX": _c("tx"), "TN": _c("tn"), "TG": _c("tg")}, sys.stdout)
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


@st.cache_data(show_spinner=False)
def get_live_point_series(lat, lon, forecast_model=FORECAST_MODEL_IFS, _series_version=7):
    """
    Point daily TX/TN/TG for the Point Meteogram. Does NOT open the maps
    spatial cube (load_global_datasets): that concatenates 2025+2026 and then
    .compute()s every field at the point, which aborted Streamlit on a
    corrupted HDF5 heap in era5_master_daily_2026.nc.

    Each covering year is opened alone, only tx/tn/tg are read, and the
    current calendar year is isolated in a subprocess.
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
            tx_name = "tx" if "tx" in pt_lf.data_vars else "mx2t"
            tn_name = "tn" if "tn" in pt_lf.data_vars else "mn2t"
            if tx_name in pt_lf.data_vars and tn_name in pt_lf.data_vars:
                tx_raw = _squeeze_celsius(pt_lf[tx_name].values)
                tn_raw = _squeeze_celsius(pt_lf[tn_name].values)
                tx_corr = tx_raw + _qdm_mean_bias(lat, lon, f_doys, "tx_bias")
                dtr_corr = (tx_raw - tn_raw) + _qdm_mean_bias(lat, lon, f_doys, "dtr_bias")
                tg_corr = (tx_raw + tn_raw) / 2.0 + _qdm_mean_bias(lat, lon, f_doys, "tg_bias")
                frames.append(pd.DataFrame({
                    "Date": f_days, "TX": tx_corr, "TN": tx_corr - dtr_corr, "TG": tg_corr,
                }))
            elif "tg" in pt_lf.data_vars:
                tg_corr = _squeeze_celsius(pt_lf["tg"].values) + _qdm_mean_bias(lat, lon, f_doys, "tg_bias")
                frames.append(pd.DataFrame({
                    "Date": f_days, "TX": np.nan, "TN": np.nan, "TG": tg_corr,
                }))
        except Exception:
            pass

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    # 29 Feb stays VISIBLE on the live chart (real ERA5/IFS value plotted on
    # its real calendar date) — only the 365-day BASELINE array excises it.
    # Reindex on the full Gregorian calendar so true ERA5 holes stay NaN.
    # Never interpolate — AtmoPulse does not invent temperature peaks.
    full_index = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    return df.set_index("Date").reindex(full_index).rename_axis("Date").reset_index()


def _array_has_finite(val) -> bool:
    if val is None:
        return False
    arr = np.asarray(getattr(val, "values", val))
    return bool(np.isfinite(arr).any())


@st.cache_resource(show_spinner=False, max_entries=10)
def fetch_cached_synoptic_data(date_str, anchor_date_str=None, forecast_model=FORECAST_MODEL_IFS, _loader_version=9):
    with st.session_state.nc_lock:
        if anchor_date_str is not None:
            set_synoptic_anchor(anchor_date_str, SLIDER_PAD_PAST, SLIDER_PAD_FUTURE, forecast_model=forecast_model)
        data = get_synoptic_map_data(date_str, forecast_model=forecast_model)
        meta = data.pop("_meta", {})
        packed = {}
        sample = next((data[k] for k in ("mslp", "tg", "tx") if k in data), None)
        if sample is not None and hasattr(sample, "longitude"):
            packed["_lons"] = np.asarray(sample.longitude.values)
            packed["_lats"] = np.asarray(sample.latitude.values)
        for key, val in data.items():
            packed[key] = _synoptic_array(val)
        meta["temps_available"] = any(
            _array_has_finite(packed.get(name)) for name in ("tx", "tn", "tg")
        )
        return packed, meta


@st.cache_data(show_spinner=False)
def get_persistence_arrays(target_date_str, baseline_type, map_var="TG", anchor_date_str=None, forecast_model=FORECAST_MODEL_IFS):
    ref_clim = load_reference_climatology()
    if ref_clim is None: 
        return None
    if "AIFS" in str(forecast_model) and map_var in ("TX", "TN"):
        return None
    end_date = pd.to_datetime(target_date_str)
    start_date = end_date - pd.Timedelta(days=65)
    loaded = _load_persistence_daily_series(
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        anchor_date_str, forecast_model=forecast_model,
    )
    if loaded is None or loaded[0] is None: 
        return None
    (daily_dates, tx_vals, tn_vals), _meta = loaded

    tx_hist, tn_hist = tx_vals.astype(np.float64), tn_vals.astype(np.float64)
    if np.nanmean(tx_hist) > 100:
        tx_hist -= 273.15
        tn_hist -= 273.15
    dates_dt = pd.to_datetime(daily_dates)
    doys = etccdi_doy_365(dates_dt)
    suffix = "A" if baseline_type == "A" else "B"
    n_days, n_lats, n_lons = tx_hist.shape
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in ref_clim.variables: 
            return ref_clim[var_key].values
        return np.full((365, n_lats, n_lons), fallback)

    if map_var == "TX":
        v_h, v_p95, v_p90, v_p75 = tx_hist, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tx_max_val'), safe_get('tx_min_val')
    elif map_var == "TN":
        v_h, v_p95, v_p90, v_p75 = tn_hist, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tn_max_val'), safe_get('tn_min_val')
    else: 
        v_h = (tx_hist + tn_hist) / 2.0
        v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
        v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
        v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
        v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
        v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
        v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2
        v_r_w = (safe_get('tx_max_val') + safe_get('tn_max_val')) / 2
        v_r_c = (safe_get('tx_min_val') + safe_get('tn_min_val')) / 2

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
    tg_all = sub["tg"].values.astype(np.float64) - 273.15 if "tg" in sub.data_vars else (tx_all + tn_all) / 2.0
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
def compute_point_thresholds(ref_clim, lat, lon, target_date, meteo_var, epoch):
    """
    ETCCDI percentile + all-time-record thresholds for one coordinate/day,
    shaped as (p_warm, p_cold) for backend_narrative.classify_point_severity().
    Uses the same 365-day ETCCDI calendar mapping as
    frontend_plots.get_meteogram_traces() so the Point Meteogram narrative
    always matches the chart's climate boundaries envelope.
    """
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    ts = pd.Timestamp(target_date)
    doy = etccdi_doy_365(ts) - 1

    def g(key):
        return float(pt_clim[key].values[doy]) if key in pt_clim.variables else np.nan

    if meteo_var == "Max Temp (TX)":
        p_warm = {"p75": g(f'tx_p75_doy_{epoch}'), "p90": g(f'tx_p90_doy_{epoch}'), "p95": g(f'tx_p95_doy_{epoch}'), "rec": g('tx_max_val')}
        p_cold = {"p25": g(f'tx_p25_doy_{epoch}'), "p10": g(f'tx_p10_doy_{epoch}'), "p5": g(f'tx_p5_doy_{epoch}'), "rec": g('tx_min_val')}
    elif meteo_var == "Min Temp (TN)":
        p_warm = {"p75": g(f'tn_p75_doy_{epoch}'), "p90": g(f'tn_p90_doy_{epoch}'), "p95": g(f'tn_p95_doy_{epoch}'), "rec": g('tn_max_val')}
        p_cold = {"p25": g(f'tn_p25_doy_{epoch}'), "p10": g(f'tn_p10_doy_{epoch}'), "p5": g(f'tn_p5_doy_{epoch}'), "rec": g('tn_min_val')}
    else:
        p_warm = {
            "p75": (g(f'tx_p75_doy_{epoch}') + g(f'tn_p75_doy_{epoch}')) / 2,
            "p90": (g(f'tx_p90_doy_{epoch}') + g(f'tn_p90_doy_{epoch}')) / 2,
            "p95": (g(f'tx_p95_doy_{epoch}') + g(f'tn_p95_doy_{epoch}')) / 2,
            "rec": (g('tx_max_val') + g('tn_max_val')) / 2,
        }
        p_cold = {
            "p25": (g(f'tx_p25_doy_{epoch}') + g(f'tn_p25_doy_{epoch}')) / 2,
            "p10": (g(f'tx_p10_doy_{epoch}') + g(f'tn_p10_doy_{epoch}')) / 2,
            "p5": (g(f'tx_p5_doy_{epoch}') + g(f'tn_p5_doy_{epoch}')) / 2,
            "rec": (g('tx_min_val') + g('tn_min_val')) / 2,
        }
    return p_warm, p_cold


# --- DATETIME64 CRASH BUGFIX ---
# PERFORMANCE: the master archive point extraction (ds.sel(...).compute()) is
# the expensive step (multi-minute NetCDF/dask read for a fresh point) and is
# completely EPOCH-INDEPENDENT — only the climatology percentile lookup below
# depends on epoch "A" vs "B". Previously this ran TWICE per point (once per
# epoch, since epoch was baked into build_yearly_extremes_chart's own cache
# key), doubling the wait. Splitting it into its own @st.cache_data step means
# the raw series is read from disk once per (lat, lon, is_warm) and reused for
# both epoch A and epoch B charts.
@st.cache_data(show_spinner=False)
def _load_point_archive_series(lat, lon, is_warm, _archive_version=4):
    ds = get_master_archive_ds()
    if ds is None:
        return None

    # Resolve the variable name from the (lazy, uncomputed) Dataset schema
    # FIRST — this costs nothing, no I/O — so we can subset to that ONE
    # variable before touching .sel()/.compute(). Selecting the point on the
    # full multi-variable Dataset first would force every other archived
    # field (at that point) to be read/materialized for nothing.
    var_name = None
    candidates = ("tx", "mx2t") if is_warm else ("tn", "mn2t")
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


# Threshold-occurrence diagram ("Days exceeding thresholds") below the main
# meteogram: returns a complex Plotly Figure that is never mutated by its
# callers after return, so @st.cache_resource avoids the deep-copy cost
# @st.cache_data would otherwise pay on every rerun, while still keying on
# (lat, lon, epoch, is_warm). The heavy I/O now lives in
# `_load_point_archive_series` above, so this function only re-runs the cheap
# epoch-specific percentile classification + figure build on a cache miss.
# --- WAVE CACHE WRAPPER ---
@st.cache_data(show_spinner=False)
def fetch_wave_figs(lat_target, lon_target, param_code, selected_epoch, wave_thresh, wave_stat_metric, _axis_version=4):
    return get_kiesely_waves_figs(lat_target, lon_target, parameter=param_code, selected_epoch=selected_epoch, threshold_level=wave_thresh, stat_metric=wave_stat_metric)
