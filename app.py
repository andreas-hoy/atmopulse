"""
AtmoPulse Central UI Orchestration & Visualization Module (app.py)

This module serves as the primary Streamlit frontend for the AtmoPulse application.
It orchestrates the dynamic tracking of pan-European synoptic extremes by bridging 
secular ERA5 climate baselines with operational ECMWF/IFS forecasts.

Core functionalities:
- Manages the interactive UI state, navigation, and user toggles.
- Orchestrates high-performance data loading and singleton caching for large NetCDF 
  archives (lazy dask-backed handles) and live IFS datasets.
- Handles geospatial rendering via Plotly, including synchronized side-by-side 
  comparisons and single-map flickers for temperature anomalies (TG, TX, TN).
- Computes real-time spatial impact rankings (Top 10 affected countries) using 
  area-weighted masking.
- Generates localized point meteograms and wavograms, dynamically applying 
  ETCCDI-compliant 365-day leap-year adjustments.
"""

import base64
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import threading
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime

import backend_maps as _backend_maps_mod
from backend_map_locations import build_location_label_grid, build_country_weight_grid, EUROPE_BBOX
from backend_time_series import get_live_timeseries
import backend_waves as _backend_waves_mod

# DEV SAFETY NET: Streamlit's autoreload only re-executes THIS script on
# save; it does NOT automatically re-import already-loaded local modules
# like backend_waves.py. Without this, edits to backend_waves.py (bugfixes,
# debug instrumentation, etc.) silently keep running the STALE in-memory
# version for the lifetime of the server process, making fixes look like
# they "didn't work" even though the file on disk is correct. Force a fresh
# reload on every script run so code changes always take effect immediately.
import atmopulse_theme as _atmopulse_theme_mod
import importlib
importlib.reload(_backend_maps_mod)
importlib.reload(_backend_waves_mod)
importlib.reload(_atmopulse_theme_mod)
from backend_maps import drop_era5t_aux, get_synoptic_map_data, set_synoptic_anchor, _open_synoptic_range
from backend_waves import get_kiesely_waves_figs
from labels import HELP
from atmopulse_theme import (
    ATMOPULSE_BRAND,
    ATMOPULSE_COLD,
    ATMOPULSE_FONTS,
    ATMOPULSE_OVERLAY,
    ATMOPULSE_WARM,
    cold_rgba,
    diverging_persistence_colorscale,
    legend_badge_style,
    map_contour_label_font,
    map_extremes_colorscale,
    plotly_typography,
    atmopulse_streamlit_css,
    atmopulse_wordmark_html,
    LOGO_SVG,
    warm_rgba,
)

# --- UI & CSS: TOP NAVIGATION BAR ---
st.set_page_config(page_title="AtmoPulse", layout="wide", page_icon="assets/favicon.svg", initial_sidebar_state="expanded")
st.markdown(f"<style>{atmopulse_streamlit_css(ATMOPULSE_BRAND)}</style>", unsafe_allow_html=True)

geolocator = Nominatim(user_agent="atmopulse_extremes_tracker_2026")

if "nc_lock" not in st.session_state: st.session_state.nc_lock = threading.Lock()
if "search_history" not in st.session_state: st.session_state.search_history = ["Berlin", "Tallinn", "Budapest"]
if "toggles_warm" not in st.session_state: st.session_state.toggles_warm = {"p75": True, "p90": True, "p95": True, "rec": True}
if "toggles_cold" not in st.session_state: st.session_state.toggles_cold = {"p25": True, "p10": True, "p5": True, "rec": True}
if "offset_slider" not in st.session_state:
    st.session_state.offset_slider = 0
else:
    st.session_state.offset_slider = int(np.clip(st.session_state.offset_slider, -7, 3))

# Audience mode (Standard vs Expert) is independent of map *view*
# (daily snapshot vs persistence duration).
UI_MODE_STANDARD = "standard"
UI_MODE_EXPERT = "expert"
UI_MODE_LABELS = ("Standard", "Expert")
MAP_VIEW_DAILY = "Daily snapshot"
MAP_VIEW_PERSISTENCE = "Persistence duration"
LAYOUT_SIDE_BY_SIDE = "Side-by-Side Compare"
LAYOUT_FLICKER = "Single Map Flicker"

NAV_WELCOME = "Welcome"
NAV_MAP = "Map Tracker"
NAV_METEO = "Point Meteogram"
NAV_WAVE = "Point Wavogram"
NAV_METHODS = "Methods & Resources"
NAV_LEGAL = "Legal & Terms"
NAV_ITEMS = (NAV_WELCOME, NAV_MAP, NAV_METEO, NAV_WAVE, NAV_METHODS, NAV_LEGAL)
NAV_ANALYTICS = (NAV_MAP, NAV_METEO, NAV_WAVE)

ASSETS_DIR = Path("assets")
DOCUMENTS_DIR = Path("Documents")
LEGAL_MD = ASSETS_DIR / "legal.md"
METHODS_MD = ASSETS_DIR / "methods.md"

# Expert-only controls. Standard uses the defaults in STANDARD_DEFAULTS.
# t850 / jet / utci are catalogued here and wired in a later step.
EXPERT_FEATURES = frozenset({
    "map_tx_tn",
    "persistence_view",
    "percentile_layer_toggles",
    "z500",
    "t850",
    "jet",
    "utci",
    "apparent_temp",
    "meteo_tx_tn",
    "meteo_envelope",
    "wave_stat_metric",
    "flicker_layout",
})

STANDARD_DEFAULTS = {
    "map_var": "Mean Temperature (TG)",
    "map_view": MAP_VIEW_DAILY,
    "persist_metric": "Strong",
    "hatching": True,
    "mslp": True,
    "z500": False,
    "meteo_var": "Mean Temp (TG)",
    "meteo_env": "Strong",
    "show_air_temp": True,
    "show_app_temp": False,
    "wave_thresh": "Strong",
    "wave_stat_metric": "Cumulative Annual Wave Intensity",
    "map_layout": LAYOUT_SIDE_BY_SIDE,
}


def _query_ui_mode():
    raw = st.query_params.get("mode", None)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    return raw if raw in (UI_MODE_STANDARD, UI_MODE_EXPERT) else None


def _init_ui_mode():
    """Seed audience mode once per session. Query ?mode=expert|standard wins on first load."""
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = _query_ui_mode() or UI_MODE_STANDARD
    if "atmopulse_ui_mode" not in st.session_state:
        st.session_state.atmopulse_ui_mode = (
            "Expert" if st.session_state.ui_mode == UI_MODE_EXPERT else "Standard"
        )


def _on_ui_mode_change():
    st.session_state.ui_mode = str(st.session_state.atmopulse_ui_mode).lower()
    st.query_params["mode"] = st.session_state.ui_mode


def is_expert_mode() -> bool:
    return st.session_state.get("ui_mode") == UI_MODE_EXPERT


_EXTREME_LAYER_LABELS = {
    ("warm", "p75"): ("Warm: Moderate", " (> P75)"),
    ("warm", "p90"): ("Warm: Strong", " (> P90)"),
    ("warm", "p95"): ("Warm: Extreme", " (> P95)"),
    ("warm", "rec"): ("Warm: All-Time Record", ""),
    ("cold", "p25"): ("Cold: Moderate", " (< P25)"),
    ("cold", "p10"): ("Cold: Strong", " (< P10)"),
    ("cold", "p5"): ("Cold: Extreme", " (< P5)"),
    ("cold", "rec"): ("Cold: All-Time Record", ""),
}


def extreme_layer_label(kind: str, key: str) -> str:
    """Checkbox copy: impact names in Standard, percentiles appended in Expert."""
    base, expert_suffix = _EXTREME_LAYER_LABELS[(kind, key)]
    if is_expert_mode() and expert_suffix:
        return f"{base}{expert_suffix}"
    return base


def show_expert(feature: str) -> bool:
    """True when an expert-only control should be shown and honoured."""
    return is_expert_mode() and feature in EXPERT_FEATURES


def is_daily_map_view(view_mode: str) -> bool:
    return view_mode != MAP_VIEW_PERSISTENCE


_init_ui_mode()

FORECAST_OFFSET_MIN = -7
FORECAST_OFFSET_MAX = 3
SLIDER_PAD_PAST = 7    # matches abs(FORECAST_OFFSET_MIN)
SLIDER_PAD_FUTURE = 3  # matches FORECAST_OFFSET_MAX

def _load_markdown_page(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _documents_image(*filenames):
    for name in filenames:
        p = DOCUMENTS_DIR / name
        if p.exists():
            return str(p)
    return None


def add_day():
    if st.session_state.offset_slider < FORECAST_OFFSET_MAX:
        st.session_state.offset_slider += 1

def sub_day():
    if st.session_state.offset_slider > FORECAST_OFFSET_MIN:
        st.session_state.offset_slider -= 1

def toggle_warm_state():
    current = any(st.session_state.toggles_warm.values())
    for k in st.session_state.toggles_warm: 
        st.session_state.toggles_warm[k] = not current

def toggle_cold_state():
    current = any(st.session_state.toggles_cold.values())
    for k in st.session_state.toggles_cold: 
        st.session_state.toggles_cold[k] = not current

# --- DATA LOADERS ---
@st.cache_resource(show_spinner=False)
def load_reference_climatology():
    clim_path = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
    if not clim_path.exists(): 
        clim_path = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")
    return xr.open_dataset(clim_path) if clim_path.exists() else None

ref_clim = load_reference_climatology()

@st.cache_resource(show_spinner=False)
def get_master_files():
    DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
    return sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))

LIVE_TXTN = Path("ERA5_ClimateTool/Live_Forecasts/live_forecast_txtn.nc")

def _harmonize_master_archive(ds):
    """Normalize time-dim naming + expver/pressure_level across the unified
    era5_master_daily_*.nc batches, same as backend_maps.py's loader."""
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    ds = drop_era5t_aux(ds)
    if "pressure_level" in ds.dims and ds.sizes.get("pressure_level", 0) == 1:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds

@st.cache_resource(show_spinner=False)
def get_master_archive_ds(_harmonize_version=2):
    """
    SINGLETON POINTER: Initializes a unified, dask-backed handle for the ERA5 master archive.
    Prevents redundant NetCDF I/O operations during interactive UI state changes, ensuring minimal-latency array slicing for downstream extreme value extraction.
    """
    files = get_master_files()
    if not files:
        return None
    ds = xr.open_mfdataset(
        files, combine='nested', concat_dim='valid_time', engine='netcdf4',
        parallel=False, preprocess=_harmonize_master_archive,
        coords="minimal", compat="override", join="override",
    )
    ds = drop_era5t_aux(ds)
    ds = ds.sortby('valid_time')
    _, unique_idx = np.unique(ds.valid_time.values, return_index=True)
    return ds.isel(valid_time=unique_idx)

@st.cache_resource(show_spinner=False)
def get_live_txtn_ds(_loader_version=6):
    """Latest IFS daily forecast (tx/tn), falling back to the legacy txtn bridge."""
    from backend_maps import _open_live_forecast_ds
    live = _open_live_forecast_ds()
    if live is not None:
        return live
    if not LIVE_TXTN.exists():
        return None
    return xr.open_dataset(LIVE_TXTN, engine='netcdf4')

@st.cache_resource(show_spinner=False)
def _load_persistence_window_source(anchor_date_str, pad_past=SLIDER_PAD_PAST, pad_future=SLIDER_PAD_FUTURE, lookback_days=65, _loader_version=6):
    """
    SINGLETON CACHE: localized 65-day trailing window from the covering
    yearly master file(s) plus the latest IFS daily forecast — not the
    full 1940–present archive.
    """
    anchor = pd.to_datetime(anchor_date_str).normalize()
    start = anchor - pd.Timedelta(days=pad_past + lookback_days)
    end = anchor + pd.Timedelta(days=pad_future)
    ds = _open_synoptic_range(start, end)
    return ds.sel(valid_time=slice(start, end))

@st.cache_data(show_spinner=False)
def _load_persistence_daily_series(start_date_str, end_date_str, anchor_date_str=None):
    """Build a daily TX/TN series from ERA5 archive + IFS live bridge for persistence."""
    start_date = pd.to_datetime(start_date_str).normalize()
    end_date = pd.to_datetime(end_date_str).normalize()
    by_date = {}

    ds_window = _load_persistence_window_source(anchor_date_str) if anchor_date_str else None
    ds = ds_window if ds_window is not None else get_master_archive_ds()
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
                tx_d = sub['tx'].groupby('valid_time.date').max()
                tn_d = sub['tn'].groupby('valid_time.date').min()
                for i, d in enumerate(tx_d['date'].values):
                    day = pd.Timestamp(d).normalize()
                    by_date[day] = (tx_d.values[i], tn_d.values[i])

    archive_max = max(by_date) if by_date else None

    lf = get_live_txtn_ds()
    if lf is not None:
        with st.session_state.nc_lock:
            lf_sub = lf.sel(valid_time=slice(start_date, end_date))
            tx_name = "tx" if "tx" in lf_sub.data_vars else "mx2t"
            tn_name = "tn" if "tn" in lf_sub.data_vars else "mn2t"
            if tx_name in lf_sub.data_vars and tn_name in lf_sub.data_vars:
                for i, t in enumerate(lf_sub.valid_time.values):
                    day = pd.to_datetime(t).normalize()
                    if day not in by_date:
                        by_date[day] = (lf_sub[tx_name].values[i], lf_sub[tn_name].values[i])

    eligible = sorted(d for d in by_date if start_date <= d <= end_date)
    if not eligible:
        return None, {"archive_max": archive_max, "effective_end": None, "uses_ifs": False, "has_gap": False}

    eligible = eligible[-60:]
    tx_vals = np.stack([by_date[d][0] for d in eligible])
    tn_vals = np.stack([by_date[d][1] for d in eligible])
    ifs_used = archive_max is not None and any(d > archive_max for d in eligible)
    has_gap = False
    if archive_max and ifs_used:
        ifs_start = min(d for d in eligible if d > archive_max)
        has_gap = (ifs_start - archive_max).days > 1

    meta = {
        "archive_max": archive_max,
        "effective_end": eligible[-1],
        "uses_ifs": ifs_used,
        "has_gap": has_gap,
    }
    return (np.array(eligible), tx_vals, tn_vals), meta

@st.cache_resource(show_spinner=False)
def get_europe_borders_trace():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
    try:
        data = requests.get(url, timeout=10).json()
        x, y = [], []
        for feature in data['features']:
            geom = feature.get('geometry')
            if not geom: continue
            if geom['type'] == 'Polygon':
                for poly in geom['coordinates']:
                    for p in poly: x.append(p[0]); y.append(p[1])
                    x.append(None); y.append(None)
            elif geom['type'] == 'MultiPolygon':
                for multi in geom['coordinates']:
                    for poly in multi:
                        for p in poly: x.append(p[0]); y.append(p[1])
                        x.append(None); y.append(None)
        return go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=1.0), hoverinfo='skip', showlegend=False)
    except: 
        return None

border_trace = get_europe_borders_trace()

TOP10_GRID_VERSION = 5  # bump when country-filter or top10 mask rules change (invalidates st.cache_data)
TOP10_MASK_VERSION = 1

@st.cache_data(show_spinner=False)
def get_map_location_labels(lons_tuple, lats_tuple):
    return build_location_label_grid(np.array(lons_tuple), np.array(lats_tuple))

@st.cache_data(show_spinner=False)
def get_country_weight_grid(lons_tuple, lats_tuple, _version=TOP10_GRID_VERSION):
    return build_country_weight_grid(np.array(lons_tuple), np.array(lats_tuple))

def _synoptic_array(field):
    if field is None:
        return None
    if isinstance(field, np.ndarray):
        return field
    return np.asarray(getattr(field, "values", field))


def _synoptic_lonlat(map_phys_data):
    if map_phys_data is None:
        return None, None
    if "_lons" in map_phys_data and "_lats" in map_phys_data:
        return np.asarray(map_phys_data["_lons"]), np.asarray(map_phys_data["_lats"])
    sample = map_phys_data.get("mslp", map_phys_data.get("tg", map_phys_data.get("tx")))
    return np.asarray(sample.longitude.values), np.asarray(sample.latitude.values)


def _array_has_finite(val) -> bool:
    if val is None:
        return False
    arr = np.asarray(getattr(val, "values", val))
    return bool(np.isfinite(arr).any())


@st.cache_resource(show_spinner=False, max_entries=10)
def fetch_cached_synoptic_data(date_str, anchor_date_str=None, _loader_version=6):
    with st.session_state.nc_lock:
        if anchor_date_str is not None:
            set_synoptic_anchor(anchor_date_str, SLIDER_PAD_PAST, SLIDER_PAD_FUTURE)
        data = get_synoptic_map_data(date_str)
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
def get_persistence_arrays(target_date_str, baseline_type, map_var="TG", anchor_date_str=None):
    if ref_clim is None: 
        return None
    end_date = pd.to_datetime(target_date_str)
    start_date = end_date - pd.Timedelta(days=65)
    loaded = _load_persistence_daily_series(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), anchor_date_str)
    if loaded is None or loaded[0] is None: 
        return None
    (daily_dates, tx_vals, tn_vals), _meta = loaded

    tx_hist, tn_hist = tx_vals.astype(np.float64), tn_vals.astype(np.float64)
    if np.nanmean(tx_hist) > 100:
        tx_hist -= 273.15
        tn_hist -= 273.15
    dates_dt = pd.to_datetime(daily_dates)
    raw_doys = np.asarray(dates_dt.dayofyear)
    is_leap = np.asarray(dates_dt.is_leap_year)
    months = np.asarray(dates_dt.month)
    
    # ETCCDI 365-day mapping: shift Feb 29 (DOY 60) and all subsequent days in leap years by -1
    doys = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    suffix = "A" if baseline_type == "A" else "B"
    n_days, n_lats, n_lons = tx_hist.shape
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in ref_clim.variables: 
            return ref_clim[var_key].values
        return np.full((366, n_lats, n_lons), fallback)

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

MAP_VAR_LABELS = {"TG": "Mean Temperature", "TX": "Maximum Temperature", "TN": "Minimum Temperature"}
MAP_VIEW_LON = (EUROPE_BBOX[0], EUROPE_BBOX[2])
MAP_VIEW_LAT = (EUROPE_BBOX[1], EUROPE_BBOX[3])
MAP_CONTOUR_LINE_WIDTH = 1.5  # was 2.5 — MSLP / Z500 isolines
SYNOPTIC_MAP_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["autoScale2d", "select2d", "lasso2d"],
    "scrollZoom": True,
}
TOP10_MIN_PCT = 0.5


def _top10_header_html(title: str) -> str:
    return f'<span class="atmopulse-subsection-label" title="{HELP["top10_table"]}"><b>{title}</b> ℹ️</span>'

# Map z-bin indices (must match build_baseline_map mask assignment order).
_TOP10_COLD_BINS = {
    "Moderate": {4},
    "Strong": {3, 2, 1},
    "Extreme": {2, 1},
    "All-Time Record": {1},
}
_TOP10_WARM_BINS = {
    "Moderate": {5},
    "Strong": {6, 7, 8},
    "Extreme": {7, 8},
    "All-Time Record": {8},
}

def _top10_analysis_key(top10_threshold: str) -> str:
    if "All-Time" in top10_threshold:
        return "All-Time Record"
    if "Extreme" in top10_threshold:
        return "Extreme"
    if "Strong" in top10_threshold:
        return "Strong"
    return "Moderate"


def _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold):
    """Replicate the map's discrete extreme classification (same overwrite order)."""
    valid = np.isfinite(v_curr)
    mask = np.full(v_curr.shape, np.nan)
    if t_cold["p25"]: mask = np.where(valid & np.isfinite(v_p25) & (v_curr <= v_p25), 4, mask)
    if t_warm["p75"]: mask = np.where(valid & np.isfinite(v_p75) & (v_curr >= v_p75), 5, mask)
    if t_cold["p10"]: mask = np.where(valid & np.isfinite(v_p10) & (v_curr <= v_p10), 3, mask)
    if t_warm["p90"]: mask = np.where(valid & np.isfinite(v_p90) & (v_curr >= v_p90), 6, mask)
    if t_cold["p5"]:  mask = np.where(valid & np.isfinite(v_p5) & (v_curr <= v_p5), 2, mask)
    if t_warm["p95"]: mask = np.where(valid & np.isfinite(v_p95) & (v_curr >= v_p95), 7, mask)
    if t_cold["rec"]: mask = np.where(valid & np.isfinite(v_rec_c) & (v_curr <= v_rec_c), 1, mask)
    if t_warm["rec"]: mask = np.where(valid & np.isfinite(v_rec_w) & (v_curr >= v_rec_w), 8, mask)
    return mask

def _fmt_map_year(yr_val) -> str:
    try:
        y = int(float(yr_val))
        return str(y) if y > 0 else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def _record_window_doys(target_doy: int) -> list[int]:
    window = []
    for offset in range(-2, 3):
        d = target_doy + offset
        if d < 1:
            d += 366
        elif d > 366:
            d -= 366
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

def _slider_window_doys(anchor_date, pad_past=SLIDER_PAD_PAST, pad_future=SLIDER_PAD_FUTURE):
    """All calendar day-of-year values reachable via the Forecast Offset slider
    for a given anchor ('today') date."""
    dates = [anchor_date + pd.Timedelta(days=o) for o in range(-pad_past, pad_future + 1)]
    return tuple(sorted({d.dayofyear for d in dates}))

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
        vt = pd.to_datetime(ds.valid_time.values)
        mask = np.isin(vt.dayofyear, union_doys) & (vt.year < cutoff_year)
        if not mask.any():
            return None
        sub = ds.isel(valid_time=mask).load()

    tx_all = sub["tx"].values.astype(np.float64) - 273.15
    tn_all = sub["tn"].values.astype(np.float64) - 273.15
    tg_all = sub["tg"].values.astype(np.float64) - 273.15 if "tg" in sub.data_vars else (tx_all + tn_all) / 2.0
    doy_all = pd.to_datetime(sub.valid_time.values).dayofyear
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

def _yyyymmdd_year_grid(grid) -> np.ndarray:
    """YYYYMMDD int grids from the climatology → calendar year as float."""
    arr = np.asarray(grid, dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > 10_000_000), np.floor(arr / 10_000.0), np.nan)


def _map_historical_records(doy: int, target_date, map_var: str, shape, anchor_date=None):
    """All-time warm/cold values for map display, from the reference climatology."""
    nan = (np.full(shape, np.nan),) * 4
    if ref_clim is None:
        return nan
    try:
        daily_ref = ref_clim.sel(dayofyear=int(doy))
    except Exception:
        return nan
    if map_var == "TX":
        wkey, ckey, wd, cd = "tx_max_val", "tx_min_val", "tx_max_date", "tx_min_date"
    elif map_var == "TN":
        wkey, ckey, wd, cd = "tn_max_val", "tn_min_val", "tn_max_date", "tn_min_date"
    else:
        wkey, ckey, wd, cd = "tg_max_val", "tg_min_val", "tg_max_date", "tg_min_date"
    if wkey not in daily_ref or ckey not in daily_ref:
        return nan
    rec_w = np.asarray(daily_ref[wkey].values, dtype=np.float64)
    rec_c = np.asarray(daily_ref[ckey].values, dtype=np.float64)
    yr_w = _yyyymmdd_year_grid(daily_ref[wd].values) if wd in daily_ref else np.full(shape, np.nan)
    yr_c = _yyyymmdd_year_grid(daily_ref[cd].values) if cd in daily_ref else np.full(shape, np.nan)
    return rec_w, rec_c, yr_w, yr_c

def _fmt_hover_num(v) -> str:
    return f"{float(v):.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_diff(v) -> str:
    return f"{float(v):+.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_year(v) -> str:
    return str(int(float(v))) if np.isfinite(v) and float(v) > 0 else "N/A"

def _build_standard_hovertext(labels, lat2d, lon2d, v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, var_label):
    fmt1 = np.vectorize(_fmt_hover_num)
    fmtd = np.vectorize(_fmt_hover_diff)
    fyr = np.vectorize(_fmt_hover_year)
    loc = labels.astype(str)
    return (
        "<b>" + loc + "</b><br>"
        "Latitude: " + fmt1(lat2d) + ", Longitude: " + fmt1(lon2d) + "<br><br>"
        + var_label + ": " + fmt1(v_curr) + " °C<br>"
        "All-Time Warm: " + fmt1(v_rec_w) + " °C (Year " + fyr(yr_w) + "; " + fmtd(diff_w) + " °C diff)<br>"
        "All-Time Cold: " + fmt1(v_rec_c) + " °C (Year " + fyr(yr_c) + "; " + fmtd(diff_c) + " °C diff)"
    )

def _map_xaxis_kwargs(**extra):
    # constrain="domain" on X (not Y): if the box is a pixel off the 70:42
    # geographic ratio, leftover width letterboxes left/right instead of
    # cropping southern Europe off the latitude range.
    return dict(
        range=list(MAP_VIEW_LON), autorange=False, showgrid=False, zeroline=False,
        visible=False, constrain="domain", constraintoward="center", **extra,
    )

def _map_yaxis_kwargs(**extra):
    return dict(
        range=list(MAP_VIEW_LAT), autorange=False, showgrid=False, zeroline=False,
        scaleanchor="x", scaleratio=1, visible=False, **extra,
    )

def _add_map_source_label(fig, *, row=None, col=None):
    """Anchor source tag to the map axes domain (not full figure paper)."""
    ann = dict(
        text="Data: ERA5/IFS",
        xref="x domain", yref="y domain",
        x=0.99, y=0.03,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color=ATMOPULSE_BRAND["text_on_light"], family=ATMOPULSE_FONTS["sora_css"]),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(200,200,200,0.55)",
        borderwidth=1,
        borderpad=4,
    )
    if row is None and col is None:
        fig.add_annotation(**ann)
    else:
        fig.add_annotation(**ann, row=row, col=col)

def _render_synoptic_map(fig, title: str, key: str) -> None:
    """Render one synoptic map: Streamlit title above a CSS 70:42 frame.

    Titles stay outside Plotly so the plot area can match EUROPE_BBOX
    exactly. The keyed container is sized by CSS aspect-ratio; Plotly
    fills that box instead of using a fixed pixel height.
    """
    st.markdown(f"<p class='atmopulse-map-title'>{title}</p>", unsafe_allow_html=True)
    fig.update_layout(
        **plotly_typography(),
        uirevision="map_sync_state",
        autosize=True,
        height=None,
        title=None,
        margin=dict(t=0, l=0, r=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    with st.container(key=key):
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=SYNOPTIC_MAP_CONFIG,
            key=f"plotly_{key}",
        )

def _add_map_contour(fig, lons, lats, z, color, start, end, step):
    fig.add_trace(go.Contour(
        x=lons, y=lats, z=z,
        colorscale=[[0, color], [1, color]],
        contours=dict(start=start, end=end, size=step, showlabels=True, labelfont=map_contour_label_font()),
        contours_coloring='lines', showscale=False, line_width=MAP_CONTOUR_LINE_WIDTH, opacity=0.8, hoverinfo="skip",
    ))

def build_baseline_map(ref_data, map_phys_data, target_date, t_warm, t_cold, toggles, view_mode, persist_metric, top10_threshold, baseline_type="A", map_var="TG", anchor_date=None, *, full_width=False):
    if ref_data is None or map_phys_data is None: 
        return go.Figure()
        
    suffix, doy = ("A" if baseline_type == "A" else "B"), target_date.dayofyear
    tx_curr, tn_curr = _synoptic_array(map_phys_data["tx"]), _synoptic_array(map_phys_data["tn"])
    lons, lats = _synoptic_lonlat(map_phys_data)
    
    # Align the climatology grid to the live/archive field's actual lat/lon
    # coordinates (nearest-neighbor) instead of assuming positional array
    # equality. A silent grid mismatch here (e.g. different longitude
    # convention or half-cell offset between climatology and live sources)
    # is what produces isolated coastal boundary artifacts.
    daily_ref = ref_data.sel(dayofyear=doy).reindex(
        latitude=lats, longitude=lons, method="nearest"
    )
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in daily_ref.variables: 
            return daily_ref[var_key].values
        return np.full(tx_curr.shape, fallback)

    if map_var == "TX":
        v_curr, v_p95, v_p90, v_p75 = tx_curr, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
    elif map_var == "TN":
        v_curr, v_p95, v_p90, v_p75 = tn_curr, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
    else:
        v_curr = (tx_curr + tn_curr) / 2.0
        v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
        v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
        v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
        v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
        v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
        v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2

    # All-time records: archive only, strictly before the viewed year (keeps the
    # previous record visible when the current year breaks it).
    v_rec_w, v_rec_c, yr_w, yr_c = _map_historical_records(doy, target_date, map_var, tx_curr.shape, anchor_date)

    fig = go.Figure()
    loc_labels = get_map_location_labels(tuple(lons), tuple(lats))
    
    # Pure NumPy math without string loops (100x faster, minimal RAM footprint)
    diff_w = v_curr - v_rec_w
    diff_c = v_curr - v_rec_c
    lon2d, lat2d = np.meshgrid(lons, lats)
    var_label = MAP_VAR_LABELS.get(map_var, map_var)

    if is_daily_map_view(view_mode):
        # Guard against NaN/Inf/sentinel values on either side of the
        # comparison reaching the colorbin classification: a corrupt or
        # masked cell in v_curr or in any threshold array must never be
        # classified as an extreme, it must stay uncolored (NaN) instead of
        # rendering as an isolated implausible "record" pixel.
        valid = np.isfinite(v_curr)
        mask = _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold)

        colorscale = map_extremes_colorscale()
        
        hovertext = _build_standard_hovertext(
            loc_labels, lat2d, lon2d, v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, var_label,
        )

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=mask, hovertext=hovertext, colorscale=colorscale, showscale=False,
            opacity=0.85, zmin=1, zmax=8, zsmooth='best',
            hovertemplate="%{hovertext}<extra></extra>",
        ))
        
        if toggles.get("hatching", False):
            anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
            try:
                streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str)
            except Exception:
                streaks = None
            if streaks is not None:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                if "All-Time" in top10_threshold: 
                    h_idx, c_idx = 3, 7
                elif "Extreme" in top10_threshold: 
                    h_idx, c_idx = 2, 6
                elif "Strong" in top10_threshold: 
                    h_idx, c_idx = 1, 5 
                else: 
                    h_idx, c_idx = 0, 4
                
                hatch_mask = (streaks[h_idx] >= 6) | (streaks[c_idx] >= 6)
                if np.any(hatch_mask):
                    h_lons, h_lats = lon_grid[hatch_mask][::2], lat_grid[hatch_mask][::2]
                    fig.add_trace(go.Scatter(x=h_lons, y=h_lats, mode='markers', marker=dict(symbol='x', color='rgba(0,0,0,0.15)', size=3), hoverinfo='skip', showlegend=False))
                    
    else:
        anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
        streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str)
        if streaks is not None:
            mapping = {
                "Moderate": (0, 4, 60),
                "Strong": (1, 5, 30),
                "Extreme": (2, 6, 20),
                "All-Time Record": (3, 7, 15),
            }
            w_idx, c_idx, max_days = mapping.get(persist_metric, (1, 5, 30))
            warm = streaks[w_idx].astype(float)
            cold = streaks[c_idx].astype(float)
            warm_only = (warm > 0) & (cold == 0)
            cold_only = (cold > 0) & (warm == 0)
            both = (warm > 0) & (cold > 0)
            
            z = np.full(warm.shape, np.nan)
            z[warm_only] = warm[warm_only]
            z[cold_only] = -cold[cold_only]
            z[both] = np.where(warm[both] >= cold[both], warm[both], -cold[both])

            persist_colorbar = dict(
                title="Days", len=0.6, y=0.5, thickness=15,
                tickvals=[-max_days, -max_days // 2, 0, max_days // 2, max_days],
                ticktext=[str(max_days), str(max_days // 2), "0", str(max_days // 2), str(max_days)],
            )
            fig.add_trace(go.Heatmap(
                x=lons, y=lats, z=z,
                hovertext=(
                    "<b>" + loc_labels.astype(str) + "</b><br>"
                    "Latitude: " + np.vectorize(_fmt_hover_num)(lat2d) + ", Longitude: " + np.vectorize(_fmt_hover_num)(lon2d) + "<br><br>"
                    "Persistence: Warm: " + np.vectorize(_fmt_hover_num)(warm) + " days<br>"
                    "Persistence: Cold: " + np.vectorize(_fmt_hover_num)(cold) + " days"
                ),
                zmin=-max_days, zmax=max_days,
                colorscale=diverging_persistence_colorscale(), showscale=True, opacity=0.9, zsmooth='best',
                colorbar=persist_colorbar,
                hovertemplate="%{hovertext}<extra></extra>",
            ))
            
    if border_trace is not None: 
        fig.add_trace(border_trace)
    if toggles.get("mslp", False) and "mslp" in map_phys_data:
        _add_map_contour(fig, lons, lats, np.squeeze(_synoptic_array(map_phys_data["mslp"])), ATMOPULSE_OVERLAY['mslp_contour'], 980, 1040, 5)
    if toggles.get("z500", False) and "z500" in map_phys_data:
        _add_map_contour(fig, lons, lats, np.squeeze(_synoptic_array(map_phys_data["z500"])), ATMOPULSE_OVERLAY['z500_contour'], 500, 600, 8)

    _add_map_source_label(fig)
    fig.update_layout(
        **plotly_typography(),
        uirevision='map_sync_state',
        autosize=True,
        # Height is left unset; the keyed Streamlit frame is locked to
        # EUROPE_BBOX (70° × 42°) via CSS so Plotly fills that box at any zoom.
        height=None,
        xaxis=_map_xaxis_kwargs(),
        yaxis=_map_yaxis_kwargs(),
        margin=dict(t=0, l=0, r=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# --- TOP 10 COUNTRY IMPACT (Heat & Cold Extremes) ---
@st.cache_data(show_spinner=False)
def calculate_top10(_ref_data, _map_phys_data, target_date, t_warm, t_cold, view_mode, persist_metric, top10_threshold, baseline_type="A", map_var="TG", anchor_date=None, _mask_version=TOP10_MASK_VERSION):
    if _ref_data is None or _map_phys_data is None: 
        return pd.DataFrame(), pd.DataFrame()
        
    suffix, doy = ("A" if baseline_type == "A" else "B"), target_date.dayofyear
    lons, lats = _synoptic_lonlat(_map_phys_data)
    tx, tn = _synoptic_array(_map_phys_data["tx"]), _synoptic_array(_map_phys_data["tn"])
    heat_mask, cold_mask = np.zeros(tx.shape, dtype=bool), np.zeros(tx.shape, dtype=bool)

    if is_daily_map_view(view_mode):
        daily_ref = _ref_data.sel(dayofyear=doy).reindex(
            latitude=lats, longitude=lons, method="nearest"
        )
        def safe_get(var_key, fallback=np.nan):
            if var_key in daily_ref.variables: 
                return daily_ref[var_key].values
            return np.full(tx.shape, fallback)

        if map_var == "TX":
            v_curr = tx
            v_p95, v_p90, v_p75 = safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
            v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        elif map_var == "TN":
            v_curr = tn
            v_p95, v_p90, v_p75 = safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
            v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        else:
            v_curr = (tx + tn) / 2.0
            v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
            v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
            v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
            v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
            v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
            v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2
            
        v_rec_w, v_rec_c, _, _ = _map_historical_records(doy, target_date, map_var, tx.shape, anchor_date)

        display_mask = _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold)
        level = _top10_analysis_key(top10_threshold)
        heat_mask = np.isin(display_mask, list(_TOP10_WARM_BINS[level]))
        cold_mask = np.isin(display_mask, list(_TOP10_COLD_BINS[level]))
    else:
        anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
        streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str)
        if streaks is not None:
            mapping = {"Moderate": (0, 4), "Strong": (1, 5), "Extreme": (2, 6), "All-Time Record": (3, 7)}
            h_idx, c_idx = mapping.get(persist_metric, (1, 5))
            heat_mask, cold_mask = streaks[h_idx] >= 6, streaks[c_idx] >= 6

    weights, sizes = get_country_weight_grid(tuple(lons), tuple(lats))
    res_h, res_c = [], []
    for name, w in weights.items():
        tot = float(w.sum())
        if tot <= 0: continue
        fh, fc = float((heat_mask * w).sum() / tot * 100), float((cold_mask * w).sum() / tot * 100)
        if fh >= TOP10_MIN_PCT: res_h.append({"Country": name, "Warm Impact (%)": fh, "_size": sizes[name]})
        if fc >= TOP10_MIN_PCT: res_c.append({"Country": name, "Cold Impact (%)": fc, "_size": sizes[name]})

    def _rank(rows, col):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values(by=[col, "_size"], ascending=[False, False]).head(10)
        return df[["Country", col]].reset_index(drop=True)

    return _rank(res_h, "Warm Impact (%)"), _rank(res_c, "Cold Impact (%)")

# --- METEOGRAM CORE TRACES (For Subplots) ---
def get_meteogram_traces(df_live, ref_clim, lat, lon, target_date, epoch, show_air, show_app, meteo_env, meteo_var="TG"):
    traces = []
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)
    
    # ETCCDI 365-day mapping for the Point Meteogram
    raw_doys = df_live['Date'].dt.dayofyear.values
    is_leap = df_live['Date'].dt.is_leap_year.values
    months = df_live['Date'].dt.month.values
    doy_365 = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    doys = doy_365 - 1  # 0-based for array indexing
    
    dates = df_live['Date']
    
    if meteo_var == "Mean Temp (TG)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
    elif meteo_var == "Max Temp (TX)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tx_p25_doy_{epoch}'].values[doys]) / 2.0
    else:
        c_base = (pt_clim[f'tn_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
        
    env_map = {"Moderate": ("p75", "p25"), "Strong": ("p90", "p10"), "Extreme": ("p95", "p5"), "All-Time": ("max_val", "min_val")}
    el_up, el_dn = env_map.get(meteo_env, ("p90", "p10"))
    p_up_key = f'tx_{el_up}_doy_{epoch}' if el_up != "max_val" else 'tx_max_val'
    p_dn_key = f'tn_{el_dn}_doy_{epoch}' if el_dn != "min_val" else 'tn_min_val'
    
    env_upper = pt_clim[p_up_key].values[doys] if p_up_key in pt_clim.variables else np.full(len(doys), np.nan)
    env_lower = pt_clim[p_dn_key].values[doys] if p_dn_key in pt_clim.variables else np.full(len(doys), np.nan)
    
    traces.append(go.Scatter(x=dates, y=env_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=env_lower, mode='lines', fill='tonexty', fillcolor='rgba(220,220,220,0.5)', line=dict(width=0), name='Climate Boundaries Envelope', legendgroup='env', hoverinfo='skip'))

    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values if col_target in df_live.columns else ((df_live.loc[dates <= tgt_dt_norm, 'TX'].values + df_live.loc[dates <= tgt_dt_norm, 'TN'].values) / 2.0)
    
    d_hist = dates[dates <= tgt_dt_norm]
    c_hist = c_base[dates <= tgt_dt_norm]

    # Warm Anomalies
    p75 = pt_clim[f'tx_p75_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p75_doy_{epoch}' in pt_clim else c_hist
    p90 = pt_clim[f'tx_p90_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p90_doy_{epoch}' in pt_clim else c_hist
    p95 = pt_clim[f'tx_p95_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p95_doy_{epoch}' in pt_clim else c_hist
    y_w1 = np.where(t_hist > c_hist, np.minimum(t_hist, p75), c_hist)
    y_w2 = np.where(t_hist > p75, np.minimum(t_hist, p90), y_w1)
    y_w3 = np.where(t_hist > p90, np.minimum(t_hist, p95), y_w2)
    y_w4 = np.where(t_hist > p95, t_hist, y_w3)
    
    # Cold Anomalies
    p25 = pt_clim[f'tn_p25_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p25_doy_{epoch}' in pt_clim else c_hist
    p10 = pt_clim[f'tn_p10_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p10_doy_{epoch}' in pt_clim else c_hist
    p5  = pt_clim[f'tn_p5_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p5_doy_{epoch}' in pt_clim else c_hist
    y_c1 = np.where(t_hist < c_hist, np.maximum(t_hist, p25), c_hist)
    y_c2 = np.where(t_hist < p25, np.maximum(t_hist, p10), y_c1)
    y_c3 = np.where(t_hist < p10, np.maximum(t_hist, p5), y_c2)
    y_c4 = np.where(t_hist < p5, t_hist, y_c3)

    sh = True if epoch == "A" else False  # Draw legend only once

    traces.append(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w1, mode='lines', fill='tonexty', fillcolor=warm_rgba('moderate'), line=dict(width=0), name='Warm Moderate', legendgroup='wm', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w2, mode='lines', fill='tonexty', fillcolor=warm_rgba('strong'), line=dict(width=0), name='Warm Strong', legendgroup='ws', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w3, mode='lines', fill='tonexty', fillcolor=warm_rgba('extreme'), line=dict(width=0), name='Warm Extreme', legendgroup='we', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w4, mode='lines', fill='tonexty', fillcolor=warm_rgba('record'), line=dict(width=0), name='Warm Record', legendgroup='wr', showlegend=sh, hoverinfo='skip'))

    traces.append(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c1, mode='lines', fill='tonexty', fillcolor=cold_rgba('moderate'), line=dict(width=0), name='Cold Moderate', legendgroup='cm', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c2, mode='lines', fill='tonexty', fillcolor=cold_rgba('strong'), line=dict(width=0), name='Cold Strong', legendgroup='cs', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c3, mode='lines', fill='tonexty', fillcolor=cold_rgba('extreme'), line=dict(width=0), name='Cold Extreme', legendgroup='ce', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c4, mode='lines', fill='tonexty', fillcolor=cold_rgba('record'), line=dict(width=0), name='Cold Record', legendgroup='cr', showlegend=sh, hoverinfo='skip'))

    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(color='black', width=2), name='Climatology Base', legendgroup='base', showlegend=sh))

    fcst_mask = dates >= tgt_dt_norm
    if show_app:
        col_app = 'AT_Max' if meteo_var == "Max Temp (TX)" else ('AT_Min' if meteo_var == "Min Temp (TN)" else 'AT_Mean')
        if col_app in df_live.columns:
            traces.append(go.Scatter(x=d_hist, y=df_live.loc[dates <= tgt_dt_norm, col_app], mode='lines', name='Apparent Temperature', legendgroup='app', showlegend=sh, line=dict(color=ATMOPULSE_OVERLAY['apparent_temp'], width=1.5), hovertemplate="Apparent Temperature: %{y:.1f}°C<extra></extra>"))
            traces.append(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_app], mode='lines', line=dict(color=ATMOPULSE_OVERLAY['apparent_temp'], width=1.5, dash='dot'), legendgroup='app', showlegend=False, hovertemplate="Apparent Temperature (Fcst): %{y:.1f}°C<extra></extra>"))
    
    if show_air:
        c_ref_w = pt_clim['tx_max_val'].values[doys][dates <= tgt_dt_norm] if 'tx_max_val' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_c = pt_clim['tn_min_val'].values[doys][dates <= tgt_dt_norm] if 'tn_min_val' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_yw = pt_clim['tx_max_year'].values[doys][dates <= tgt_dt_norm] if 'tx_max_year' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_yc = pt_clim['tn_min_year'].values[doys][dates <= tgt_dt_norm] if 'tn_min_year' in pt_clim else np.full(len(d_hist), np.nan)

        c_data = np.empty((len(d_hist), 5), dtype=object)
        c_data[:, 0] = np.round(c_hist, 1)
        c_data[:, 1] = np.round(c_ref_w, 1)
        c_data[:, 2] = np.round(c_ref_c, 1)
        c_data[:, 3] = c_ref_yw
        c_data[:, 4] = c_ref_yc

        traces.append(go.Scatter(x=d_hist, y=t_hist, mode='lines', customdata=c_data, name='Air Temperature', legendgroup='air', showlegend=sh, line=dict(color='rgba(0,0,0,0.7)', width=1.5), 
            hovertemplate="Air Temperature: %{y:.1f}°C<br>Reference Value: %{customdata[0]:.1f}°C<br>Hist. Max: %{customdata[1]:.1f}°C (%{customdata[3]:.0f})<br>Hist. Min: %{customdata[2]:.1f}°C (%{customdata[4]:.0f})<extra></extra>"))
        if col_target in df_live.columns:
            traces.append(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_target], mode='lines', name='Air Temp Forecast', legendgroup='air', showlegend=False, line=dict(color='gray', width=2.5, dash='dot'), hovertemplate="Air Temperature (Fcst): %{y:.1f}°C<extra></extra>"))
        
    return traces

def build_top10_table(df_live, meteo_var):
    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    if col_target not in df_live.columns: 
        return pd.DataFrame()
    df_sorted = df_live[['Date', col_target]].dropna().sort_values(by=col_target, ascending=(meteo_var == "Min Temp (TN)"))
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date']).dt.strftime('%Y-%m-%d')
    df_sorted.rename(columns={col_target: f"{col_target} (°C)"}, inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1
    return df_sorted.head(10)

# --- DATETIME64 CRASH BUGFIX ---
@st.cache_data(show_spinner=False)
def build_yearly_extremes_chart(lat, lon, epoch, is_warm):
    ds = get_master_archive_ds()
    if ds is None or ref_clim is None: 
        return go.Figure()
    with st.session_state.nc_lock:
        try:
            pt_data = ds.sel(latitude=lat, longitude=lon, method='nearest').compute()
        except: 
            return go.Figure()
        
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    var_name = None
    if is_warm:
        for candidate in ("tx", "mx2t"):
            if candidate in pt_data:
                var_name = candidate
                break
    else:
        for candidate in ("tn", "mn2t"):
            if candidate in pt_data:
                var_name = candidate
                break
    if var_name is None:
        return go.Figure().add_annotation(text="Data Missing.", showarrow=False)

    raw = np.asarray(pt_data[var_name].values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size > 0 and np.nanmean(finite) > 100:
        raw = raw - 273.15
    val = raw
    
    df = pd.DataFrame({'time': pt_data.valid_time.values, 'val': val}).drop_duplicates(subset=['time'])
    dates = pd.to_datetime(df['time'])
    df['year'] = dates.dt.year
    
    # ETCCDI 365-day mapping for historical extremes
    raw_doys = dates.dt.dayofyear.values
    is_leap = dates.dt.is_leap_year.values
    months = dates.dt.month.values
    doy_365 = np.where(is_leap & (months >= 3), raw_doys - 1, raw_doys)
    doys = doy_365 - 1  # 0-based for array indexing
    
    v = df['val'].values
    
    if is_warm and f'tx_p95_doy_{epoch}' in pt_clim.variables:
        c_75, c_90, c_95, c_rec = pt_clim[f'tx_p75_doy_{epoch}'].values[doys], pt_clim[f'tx_p90_doy_{epoch}'].values[doys], pt_clim[f'tx_p95_doy_{epoch}'].values[doys], pt_clim['tx_max_val'].values[doys]
        df['p75'], df['p90'], df['p95'], df['rec'] = (v >= c_75) & (v < c_90), (v >= c_90) & (v < c_95), (v >= c_95) & (v < c_rec), v >= c_rec
    elif not is_warm and f'tn_p5_doy_{epoch}' in pt_clim.variables:
        c_25, c_10, c_5, c_rec = pt_clim[f'tn_p25_doy_{epoch}'].values[doys], pt_clim[f'tn_p10_doy_{epoch}'].values[doys], pt_clim[f'tn_p5_doy_{epoch}'].values[doys], pt_clim['tn_min_val'].values[doys]
        df['p25'], df['p10'], df['p5'], df['rec'] = (v <= c_25) & (v > c_10), (v <= c_10) & (v > c_5), (v <= c_5) & (v > c_rec), v <= c_rec
    else: 
        return go.Figure().add_annotation(text="Data Missing.", showarrow=False)

    cols_to_sum = ['year', 'p75', 'p90', 'p95', 'rec'] if is_warm else ['year', 'p25', 'p10', 'p5', 'rec']
    res = df[cols_to_sum].groupby('year').sum()
    
    fig = go.Figure()
    if is_warm:
        fig.add_trace(go.Bar(x=res.index, y=res['p75'], name='Moderate', marker_color=ATMOPULSE_WARM['p75']))
        fig.add_trace(go.Bar(x=res.index, y=res['p90'], name='Strong', marker_color=ATMOPULSE_WARM['p90']))
        fig.add_trace(go.Bar(x=res.index, y=res['p95'], name='Extreme', marker_color=ATMOPULSE_WARM['p95']))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_WARM['rec']))
    else:
        fig.add_trace(go.Bar(x=res.index, y=res['p25'], name='Moderate', marker_color=ATMOPULSE_COLD['p25']))
        fig.add_trace(go.Bar(x=res.index, y=res['p10'], name='Strong', marker_color=ATMOPULSE_COLD['p10']))
        fig.add_trace(go.Bar(x=res.index, y=res['p5'],  name='Extreme', marker_color=ATMOPULSE_COLD['p5']))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_COLD['rec']))

    fig.update_layout(**plotly_typography(), barmode='stack', title=f"Days exceeding thresholds | {'1961–1990' if epoch=='A' else '1996–2025'}", height=300, margin=dict(t=30, b=10), template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), yaxis=dict(rangemode="tozero"))
    return fig

# --- WAVE CACHE WRAPPER ---
@st.cache_data(show_spinner=False)
def fetch_wave_figs(lat_target, lon_target, param_code, selected_epoch, wave_thresh, wave_stat_metric, _axis_version=2):
    return get_kiesely_waves_figs(lat_target, lon_target, parameter=param_code, selected_epoch=selected_epoch, threshold_level=wave_thresh, stat_metric=wave_stat_metric)

# --- UI LAYOUT: TOP NAVIGATION BAR ---
with st.container(vertical_alignment="center", horizontal=True, horizontal_alignment="left", gap="small", key="atmopulse_nav_bar", border=False):
    nav_selection = st.radio(
        "Navigation",
        list(NAV_ITEMS),
        horizontal=True,
        label_visibility="collapsed",
        key="atmopulse_top_nav",
    )
st.divider()

default_date = pd.Timestamp.now().floor('D')
target_month = default_date.month
is_warm_season = False
if 4 < target_month < 10: 
    is_warm_season = True
elif target_month == 4 and default_date.day >= 16: 
    is_warm_season = True
elif target_month == 10 and default_date.day <= 15: 
    is_warm_season = True

default_wave_idx = 0 if is_warm_season else 1

with st.sidebar:
    st.markdown(
        f"<div class='atmopulse-sidebar-logo'>{atmopulse_wordmark_html()}{LOGO_SVG}</div>",
        unsafe_allow_html=True,
    )
    st.radio(
        "UI mode",
        list(UI_MODE_LABELS),
        horizontal=True,
        key="atmopulse_ui_mode",
        on_change=_on_ui_mode_change,
        label_visibility="collapsed",
        help=HELP["ui_mode"],
    )
    if nav_selection in NAV_ANALYTICS:
        st.header("Control Panel")
        st.markdown(
            f"<p style='font-size: 12px; color: #555; margin-top: -10px;'>📡 Data: ERA5 Archive (~ 5 days ago) | "
            f"IFS Forecast ({default_date.strftime('%d.%m.%Y')} 12 UTC).</p>",
            unsafe_allow_html=True,
            help=HELP["data_vintage"],
        )

        st.slider("Forecast Offset (Days):", FORECAST_OFFSET_MIN, FORECAST_OFFSET_MAX, key="offset_slider", help=HELP["forecast_offset"])
        # Native -7 / +3 tick labels are forced permanently visible via CSS
        # (.st-key-offset_slider in atmopulse_theme.py); we just add the
        # missing midpoint here, styled identically (0 sits at 70% of -7..3).
        _zero_pct = (0 - FORECAST_OFFSET_MIN) / (FORECAST_OFFSET_MAX - FORECAST_OFFSET_MIN) * 100
        st.markdown(
            f"""
            <div style='position: relative; width: 100%; height: 16px; margin-top: -20px; margin-bottom: 6px;'>
                <span class='atmopulse-slider-zero' style='position: absolute; left: {_zero_pct}%; transform: translateX(-50%);'>0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1: 
            st.button("← Prev Day", on_click=sub_day, use_container_width=True)
        with btn_col2: 
            st.button("Next Day →", on_click=add_day, use_container_width=True)
        
        target_date = default_date + pd.Timedelta(days=st.session_state.offset_slider)
        st.info(f"Target Date: **{target_date.strftime('%d.%m.%Y')}**")
        
        toggles = {}
        
        if nav_selection == NAV_MAP:
            st.markdown("---")
            if show_expert("map_tx_tn"):
                map_var = st.radio(
                    "**Mapped Variable:**",
                    ("Mean Temperature (TG)", "Maximum Temperature (TX)", "Minimum Temperature (TN)"),
                    index=0,
                    help=HELP["map_variable"],
                )
            else:
                map_var = STANDARD_DEFAULTS["map_var"]
            map_var_code = map_var.split('(')[1].strip(')')
            
            st.markdown("---")
            persist_metric = STANDARD_DEFAULTS["persist_metric"]
            if show_expert("persistence_view"):
                view_mode = st.radio(
                    "**Map view:**",
                    (MAP_VIEW_DAILY, MAP_VIEW_PERSISTENCE),
                    help=HELP["map_view_mode"],
                )
            else:
                view_mode = STANDARD_DEFAULTS["map_view"]
            st.markdown("---")
            top10_threshold = st.radio("**Analysis Level**", ("Moderate", "Strong", "Extreme", "All-Time Record"), index=1, help=HELP["map_analysis_level"])
            
            if is_daily_map_view(view_mode):
                st.markdown("---")
                st.markdown("**Map Extremes**", help=HELP["map_extremes"])
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    warm_active = any(st.session_state.toggles_warm.values())
                    if st.button("Warm", use_container_width=True, type="primary" if warm_active else "secondary", help=HELP["warm_toggle"]): 
                        toggle_warm_state()
                        st.rerun()
                with m_col2:
                    cold_active = any(st.session_state.toggles_cold.values())
                    if st.button("Cold", use_container_width=True, type="primary" if cold_active else "secondary", help=HELP["cold_toggle"]): 
                        toggle_cold_state()
                        st.rerun()
                if show_expert("percentile_layer_toggles"):
                    st.markdown("<hr style='margin-top:5px; margin-bottom:15px; border-top: 1px dashed gray;'>", unsafe_allow_html=True)
                    st.session_state.toggles_warm["p75"] = st.checkbox(
                        extreme_layer_label("warm", "p75"), value=st.session_state.toggles_warm["p75"],
                    )
                    st.session_state.toggles_warm["p90"] = st.checkbox(
                        extreme_layer_label("warm", "p90"), value=st.session_state.toggles_warm["p90"],
                    )
                    st.session_state.toggles_warm["p95"] = st.checkbox(
                        extreme_layer_label("warm", "p95"), value=st.session_state.toggles_warm["p95"],
                    )
                    st.session_state.toggles_warm["rec"] = st.checkbox(
                        extreme_layer_label("warm", "rec"), value=st.session_state.toggles_warm["rec"],
                    )
                    st.session_state.toggles_cold["p25"] = st.checkbox(
                        extreme_layer_label("cold", "p25"), value=st.session_state.toggles_cold["p25"],
                    )
                    st.session_state.toggles_cold["p10"] = st.checkbox(
                        extreme_layer_label("cold", "p10"), value=st.session_state.toggles_cold["p10"],
                    )
                    st.session_state.toggles_cold["p5"] = st.checkbox(
                        extreme_layer_label("cold", "p5"), value=st.session_state.toggles_cold["p5"],
                    )
                    st.session_state.toggles_cold["rec"] = st.checkbox(
                        extreme_layer_label("cold", "rec"), value=st.session_state.toggles_cold["rec"],
                    )
                else:
                    if any(st.session_state.toggles_warm.values()):
                        for k in st.session_state.toggles_warm:
                            st.session_state.toggles_warm[k] = True
                    if any(st.session_state.toggles_cold.values()):
                        for k in st.session_state.toggles_cold:
                            st.session_state.toggles_cold[k] = True
                st.markdown("---")
                toggles["hatching"] = st.checkbox(
                    "Show 6-Day WSDI/CSDI Overlay",
                    value=STANDARD_DEFAULTS["hatching"],
                    help=HELP["wsdi_csdi_overlay"],
                )
            else:
                st.markdown("---")
                st.markdown("**Persistence Visualization**")
                persist_metric = st.radio("Select intensity level:", ("Moderate", "Strong", "Extreme", "All-Time Record"), index=1, help=HELP["persistence_intensity"])
                
            st.markdown("---")
            toggles["mslp"] = st.checkbox("Show MSLP Contours", value=STANDARD_DEFAULTS["mslp"], help=HELP["mslp_contours"])
            if show_expert("z500"):
                toggles["z500"] = st.checkbox("Show Z500 Contours", value=STANDARD_DEFAULTS["z500"], help=HELP["z500_contours"])
            else:
                toggles["z500"] = STANDARD_DEFAULTS["z500"]
            
        elif nav_selection in (NAV_METEO, NAV_WAVE):
            st.markdown("---")
            st.markdown("**Location Settings**")
            
            if nav_selection == NAV_METEO:
                if show_expert("meteo_tx_tn"):
                    meteo_var = st.radio("Variable:", ["Mean Temp (TG)", "Max Temp (TX)", "Min Temp (TN)"])
                else:
                    meteo_var = STANDARD_DEFAULTS["meteo_var"]
                if show_expert("meteo_envelope"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    meteo_env = st.selectbox("Background Envelope:", ["Moderate", "Strong", "Extreme", "All-Time"], index=1, help=HELP["meteogram_envelope"])
                else:
                    meteo_env = STANDARD_DEFAULTS["meteo_env"]
                st.markdown("<br>", unsafe_allow_html=True)
                show_air_temp = st.checkbox("Show Air Temperature Colors", value=STANDARD_DEFAULTS["show_air_temp"], help=HELP["meteogram_air_temp_colors"])
                if show_expert("apparent_temp"):
                    show_app_temp = st.checkbox("Show Apparent Temperature", value=STANDARD_DEFAULTS["show_app_temp"], help=HELP["meteogram_apparent_temp"])
                else:
                    show_app_temp = STANDARD_DEFAULTS["show_app_temp"]
            
            if nav_selection == NAV_WAVE:
                wave_focus = st.radio("Wave Event Type:", ("Heatwaves", "Coldwaves"), index=default_wave_idx, help=HELP["wave_event_type"])
                wave_thresh = st.radio("Wave Intensity Threshold:", ("Strong", "Extreme"), help=HELP["wave_intensity_threshold"])
                if show_expert("wave_stat_metric"):
                    st.markdown("---")
                    wave_stat_metric = st.radio(
                        "Wave Statistic Metric:", 
                        ("Cumulative Annual Wave Intensity", "Maximum Annual Wave Intensity", "Cumulative Heat/Cold Intensity", "Annual Cycle Frequency"),
                        help=HELP["wave_stat_metric"],
                    )
                else:
                    wave_stat_metric = STANDARD_DEFAULTS["wave_stat_metric"]

if ref_clim is None: 
    st.error("Reference Climatology missing or corrupted! Please rebuild.")
    st.stop()

if nav_selection == NAV_WELCOME:
    st.markdown(f"### Welcome to {atmopulse_wordmark_html()}", unsafe_allow_html=True)
    st.markdown(f"""
    **{atmopulse_wordmark_html()}** merges real-time extreme weather tracking with shifting climate baselines. It provides interactive, synoptic-scale mapping and deep-dive local profiles. Currently focused on extreme temperatures, {atmopulse_wordmark_html()} aims to integrate further atmospheric variables in the future.
    <br><br>
    #### Understanding Percentiles
    {atmopulse_wordmark_html()} relies heavily on percentiles to contextualize current weather against historical norms. In our maps and meteograms, percentiles are calculated using a **centered 5-day moving window** across the reference periods (1961–1990 and 1996–2025). 
    For instance, the 90th percentile (P90) is a threshold exceeded only 10% of the time during the historical baseline. We track **Moderate** (P75/P25), **Strong** (P90/P10), and **Extreme** (P95/P5) thresholds to dynamically classify the severity of synoptic events.
    <br><br>
    #### The Importance of Event Duration
    The impact of extreme temperatures on sectors like human health, agriculture and infrastructure scales drastically with duration. A single hot day is a weather event; a prolonged sequence becomes a systemic hazard. 
    In the **Map Tracker** tab, you can visualize this through the **Cumulative Persistence** layer, showing how many days an extreme event has lasted. By default, the maps also display an overlay for **WSDI and CSDI** conditions.
    <br><br>
    #### Local Wave Definitions
    In the **Point Wavogram** tab, {atmopulse_wordmark_html()} uses a sophisticated definition (adapted from Kyselý) to track seasonally-bound heatwaves and coldwaves:
    * **Heatwaves:** Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) threshold for at least 3 consecutive days. 
    * **Coldwaves:** Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) threshold for at least 3 consecutive days.
    """, unsafe_allow_html=True)
    
    img_col1, img_col2 = st.columns(2)
    _warm_img = _documents_image("Warm.JPG", "Warm.jpg")
    _kalt_img = _documents_image("Kalt.JPG", "Kalt.jpg")
    with img_col1:
        if _warm_img:
            st.image(_warm_img, use_container_width=True, caption="Erfassung von Hitzewellen")
        else:
            st.caption("Warm example image not found in Documents/.")
    with img_col2:
        if _kalt_img:
            st.image(_kalt_img, use_container_width=True, caption="Erfassung von Kältewellen")
        else:
            st.caption("Cold example image not found in Documents/.")

elif nav_selection == NAV_MAP:
    if show_expert("flicker_layout"):
        map_layout = st.radio("Map Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True)
    else:
        map_layout = STANDARD_DEFAULTS["map_layout"]
    if map_layout == LAYOUT_FLICKER:
        flicker_epoch = st.radio("Select Reference Period:", ("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), horizontal=True, index=1)

    if is_daily_map_view(view_mode):
        if "Extreme" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme", highlight=True)
            s_rec_h = legend_badge_style("warm", "record", highlight=True)
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme", highlight=True)
            s_rec_c = legend_badge_style("cold", "record", highlight=True)
        elif "Strong" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong", highlight=True)
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record")
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong", highlight=True)
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record")
        elif "Moderate" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate", highlight=True)
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record")
            s_p25 = legend_badge_style("cold", "moderate", highlight=True)
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record")
        else:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record", highlight=True)
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record", highlight=True)
        st.markdown(
            f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
            f"style='margin-bottom: 6px; white-space: nowrap;'>"
            f"<b>Legend.</b> "
            f"Warm: <span style='{s_p75}'>Moderate</span> <span style='{s_p90}'>Strong</span> "
            f"<span style='{s_p95}'>Extreme</span> <span style='{s_rec_h}'>Record</span>"
            f"<span style='padding-left: 12px;'>Cold:</span> "
            f"<span style='{s_p25}'>Moderate</span> <span style='{s_p10}'>Strong</span> "
            f"<span style='{s_p5}'>Extreme</span> <span style='{s_rec_c}'>Record</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        _, pers_meta = _load_persistence_daily_series(
            (target_date - pd.Timedelta(days=65)).strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            default_date.strftime('%Y-%m-%d'),
        ) or (None, {})
        eff_end = pers_meta.get("effective_end")
        if pers_meta.get("uses_ifs") and eff_end is not None:
            gap_note = " Gaps between archive and IFS can interrupt streaks." if pers_meta.get("has_gap") else ""
            st.info(
                f"**Persistence Mode Active:** Hybrid ERA5 archive + IFS forecast. "
                f"Showing consecutive days with target percentiles, ending on {eff_end.strftime('%d.%m.%Y')} "
                f"(requested: {target_date.strftime('%d.%m.%Y')}). Recent days use IFS HRES.{gap_note}"
            )
        elif eff_end is not None and eff_end < target_date.normalize():
            st.info(
                f"**Persistence Mode Active:** Showing consecutive days with target percentiles, "
                f"ending on {eff_end.strftime('%d.%m.%Y')} (last available data; requested {target_date.strftime('%d.%m.%Y')})."
            )
        else:
            st.info(f"**Persistence Mode Active:** Showing number of consecutive days with target percentiles, ending on {target_date.strftime('%d.%m.%Y')}.")

    def render_top10_period(df_h, df_c, period_label=None):
        if period_label:
            st.markdown(f"**{period_label}**")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(_top10_header_html("Top 10 Countries – Warm Impact"), unsafe_allow_html=True)
            if not df_h.empty:
                st.dataframe(
                    df_h,
                    column_config={
                        "Warm Impact (%)": st.column_config.ProgressColumn(
                            "Warm Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")
        with t_col2:
            st.markdown(_top10_header_html("Top 10 Countries – Cold Impact"), unsafe_allow_html=True)
            if not df_c.empty:
                st.dataframe(
                    df_c,
                    column_config={
                        "Cold Impact (%)": st.column_config.ProgressColumn(
                            "Cold Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")

    def render_top10_tables(df_h, df_c):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(_top10_header_html("Top 10 Countries – Warm Impact"), unsafe_allow_html=True)
            if not df_h.empty:
                st.dataframe(
                    df_h,
                    column_config={
                        "Warm Impact (%)": st.column_config.ProgressColumn(
                            "Warm Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")
        with t_col2:
            st.markdown(_top10_header_html("Top 10 Countries – Cold Impact"), unsafe_allow_html=True)
            if not df_c.empty:
                st.dataframe(
                    df_c,
                    column_config={
                        "Cold Impact (%)": st.column_config.ProgressColumn(
                            "Cold Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")

    try:
        with st.spinner("Loading synoptic fields..."):
            map_phys_data, map_time_meta = fetch_cached_synoptic_data(
                target_date.strftime('%Y-%m-%d'), default_date.strftime('%Y-%m-%d')
            )
        if not map_time_meta.get("available"):
            st.warning(
                f"No synoptic data for **{target_date.strftime('%d.%m.%Y')}**. "
                "The IFS HRES forecast may not yet cover this date — try a lower Forecast Offset."
            )
        else:
            if not map_time_meta.get("temps_available", True):
                st.warning(
                    f"Temperature extremes (TX/TN/TG) are missing for **{target_date.strftime('%d.%m.%Y')}** "
                    "in the ERA5 archive, so the colour overlay cannot be drawn. "
                    "Synoptic contours are shown where available."
                )
            if map_layout == LAYOUT_SIDE_BY_SIDE:
                fig_a = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date)
                fig_b = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date)
                with st.container(key="atmopulse_map_columns"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        _render_synoptic_map(fig_a, "Historical Baseline (1961-1990)", "map_a")
                    with mc2:
                        _render_synoptic_map(fig_b, "Recent Baseline (1996-2025)", "map_b")
                map_col1, map_col2 = st.columns(2)
                with map_col1:
                    df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date)
                    render_top10_period(df_h_a, df_c_a)
                with map_col2:
                    df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date)
                    render_top10_period(df_h_b, df_c_b)
            else:
                ep_sel = "A" if "A" in flicker_epoch else "B"
                flicker_title = "Historical Baseline (1961-1990)" if ep_sel == "A" else "Recent Baseline (1996-2025)"
                _render_synoptic_map(
                    build_baseline_map(
                        ref_clim, map_phys_data, target_date,
                        st.session_state.toggles_warm, st.session_state.toggles_cold,
                        toggles, view_mode, persist_metric, top10_threshold,
                        ep_sel, map_var_code, anchor_date=default_date,
                        full_width=True,
                    ),
                    flicker_title,
                    "map_flicker",
                )
                _, table_col, _ = st.columns([1, 2, 1])
                with table_col:
                    df_h, df_c = calculate_top10(
                        ref_clim, map_phys_data, target_date,
                        st.session_state.toggles_warm, st.session_state.toggles_cold,
                        view_mode, persist_metric, top10_threshold,
                        ep_sel, map_var_code, anchor_date=default_date,
                    )
                    render_top10_tables(df_h, df_c)
    except Exception as e: 
        st.error(f"Error loading maps: {e}")

elif nav_selection in (NAV_METEO, NAV_WAVE):
    st.subheader("🏙️ Target Location")
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1: 
        loc_history_sel = st.selectbox("Select recent location:", ["Select..."] + st.session_state.search_history)
    with search_col2: 
        new_loc_input = st.text_input("Or select new location (Press Enter to see options):")
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    location = None
    if new_loc_input:
        if new_loc_input != st.session_state.get("last_query"):
            st.session_state.last_query = new_loc_input
            with st.spinner("Searching..."):
                results = geolocator.geocode(new_loc_input, exactly_one=False, limit=5)
                st.session_state.geocode_results = results
        results = st.session_state.get("geocode_results")
        if results:
            opts = {f"{r.address} (Lat: {r.latitude:.2f}, Lon: {r.longitude:.2f})": r for r in results}
            chosen = st.selectbox("Multiple matches found. Select exact location:", list(opts.keys()))
            location = opts[chosen]
            short_name = location.address.split(",")[0].strip()
            if short_name not in st.session_state.search_history:
                st.session_state.search_history.insert(0, short_name)
                if len(st.session_state.search_history) > 10: 
                    st.session_state.search_history.pop()
        else: 
            st.warning("No results found.")
    elif loc_history_sel != "Select...":
        location = geolocator.geocode(loc_history_sel, timeout=10)

    lat_target, lon_target = 52.52, 13.40 
    if location:
        lat_target, lon_target = round(location.latitude, 2), round(location.longitude, 2)
        if not (-25 <= lon_target <= 45 and 30 <= lat_target <= 72):
            st.warning(f"📍 Location {location.address} is outside the Europe domain.")
            location = None
        else: 
            st.success(f"📍 **Location Matrix Active:** {location.address} | **{lat_target}°N, {lon_target}°E**")

    if location:
        if nav_selection == NAV_METEO:
            if show_expert("flicker_layout"):
                map_layout = st.radio("Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True, key="met_layout")
            else:
                map_layout = STANDARD_DEFAULTS["map_layout"]
            with st.spinner("Fetching Meteogram data..."): 
                df_live = get_live_timeseries(lat_target, lon_target)
            if not df_live.empty:
                col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
                t_arr = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)
                global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
                tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)

                traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "A", show_air_temp, show_app_temp, meteo_env, meteo_var)
                traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "B", show_air_temp, show_app_temp, meteo_env, meteo_var)
                
                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), shared_yaxes=True)
                    for trace in traces_a: 
                        fig.add_trace(trace, row=1, col=1)
                    for trace in traces_b: 
                        fig.add_trace(trace, row=1, col=2)
                        
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                    fig.update_yaxes(range=[global_min, global_max])
                    fig.update_layout(**plotly_typography(), hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A", meteo_var != "Min Temp (TN)"), use_container_width=True)
                    with c2: 
                        st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "B", meteo_var != "Min Temp (TN)"), use_container_width=True)
                else:
                    flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="met_ep", index=1)
                    fig = go.Figure(data=traces_a if "A" in flicker_epoch else traces_b)
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8)
                    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                    fig.update_yaxes(range=[global_min, global_max])
                    fig.update_layout(**plotly_typography(), title=f"Reference Period {'1961–1990' if 'A' in flicker_epoch else '1996–2025'}", hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A" if "A" in flicker_epoch else "B", meteo_var != "Min Temp (TN)"), use_container_width=True)

        elif nav_selection == NAV_WAVE:
            if show_expert("flicker_layout"):
                map_layout = st.radio("Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True, key="wave_layout")
            else:
                map_layout = STANDARD_DEFAULTS["map_layout"]
            with st.spinner("Generating Historical Waves..."):
                param_code = "TX" if "Heatwaves" in wave_focus else "TN"
                fig_m_a, fig_s_a = fetch_wave_figs(lat_target, lon_target, param_code, "A", wave_thresh, wave_stat_metric)
                fig_m_b, fig_s_b = fetch_wave_figs(lat_target, lon_target, param_code, "B", wave_thresh, wave_stat_metric)

                def get_safe_max(fig, fallback=100.0):
                    """Type-safe max across all trace y-values.

                    Plotly (esp. after Streamlit caching round-trips figures
                    through JSON) may hand back t.y in several shapes:
                    - a tuple of numpy scalars / plain floats / None
                    - Plotly's compact "typed array" encoding, a dict like
                      {'dtype': 'f8', 'bdata': '<base64>'}, which Plotly.js
                      itself understands but np.asarray(..., dtype=float)
                      chokes on ("float() argument must be a string or a
                      real number, not 'dict'").
                    This decodes both shapes and always returns a plain
                    float, so downstream `g_max = ... * 1.1` never breaks.
                    """
                    def _to_array(y):
                        if y is None:
                            return np.array([], dtype=np.float64)
                        if isinstance(y, dict):
                            try:
                                raw = base64.b64decode(y["bdata"])
                                return np.frombuffer(raw, dtype=np.dtype(y["dtype"])).astype(np.float64)
                            except Exception:
                                return np.array([], dtype=np.float64)
                        try:
                            return np.asarray(y, dtype=np.float64).ravel()
                        except (TypeError, ValueError):
                            # Mixed list (e.g. stray dict/None entries mixed
                            # with numbers) — coerce element-by-element and
                            # silently drop anything non-numeric.
                            cleaned = []
                            for v in y:
                                try:
                                    cleaned.append(float(v))
                                except (TypeError, ValueError):
                                    continue
                            return np.asarray(cleaned, dtype=np.float64)

                    best = None
                    for t in fig.data:
                        arr = _to_array(getattr(t, "y", None))
                        if arr.size == 0:
                            continue
                        arr = arr[np.isfinite(arr)]
                        if arr.size == 0:
                            continue
                        local_max = float(arr.max())
                        if best is None or local_max > best:
                            best = local_max
                    return float(best) if best is not None else float(fallback)

                if fig_s_a.data and fig_s_b.data:
                    max_a = get_safe_max(fig_s_a)
                    max_b = get_safe_max(fig_s_b)
                    g_max = max(max_a, max_b) * 1.1
                    fig_s_a.update_yaxes(range=[0, g_max])
                    fig_s_b.update_yaxes(range=[0, g_max])

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    w_col1, w_col2 = st.columns(2)
                    with w_col1:
                        st.plotly_chart(fig_m_a, use_container_width=True)
                        st.plotly_chart(fig_s_a, use_container_width=True)
                    with w_col2:
                        st.plotly_chart(fig_m_b, use_container_width=True)
                        st.plotly_chart(fig_s_b, use_container_width=True)
                else:
                    flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="wave_ep", index=1)
                    st.plotly_chart(fig_m_a if "A" in flicker_epoch else fig_m_b, use_container_width=True)
                    st.plotly_chart(fig_s_a if "A" in flicker_epoch else fig_s_b, use_container_width=True)

elif nav_selection == NAV_METHODS:
    methods_md = _load_markdown_page(METHODS_MD)
    if methods_md:
        st.markdown(methods_md)
    else:
        st.markdown("### Methods & Resources")

elif nav_selection == NAV_LEGAL:
    legal_md = _load_markdown_page(LEGAL_MD)
    if legal_md:
        st.markdown(legal_md, unsafe_allow_html=True)
    else:
        st.error("Legal text not found. Expected `assets/legal.md`.")