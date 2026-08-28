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
import json
import math
import os
import subprocess
import sys
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
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

from backend_map_locations import build_location_label_grid, build_country_weight_grid, EUROPE_BBOX
from backend_maps import drop_era5t_aux, get_synoptic_map_data, set_synoptic_anchor, _open_synoptic_range, load_global_datasets, LIVE_OVERLAY_PAST_DAYS, etccdi_doy_365
from backend_waves import get_kiesely_waves_figs, get_wave_historical_rank
from backend_narrative import (
    EPOCH_LABELS,
    spatial_extreme_footprint,
    render_map_tracker_narrative,
    classify_point_severity,
    render_point_meteogram_narrative,
    render_point_wavogram_narrative,
)
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
from config import (
    DATA_ROOT,
    UI_MODE_STANDARD,
    UI_MODE_EXPERT,
    UI_MODE_LABELS,
    FORECAST_MODEL_IFS,
    FORECAST_MODEL_AIFS,
    FORECAST_MODEL_OPTIONS,
    MAP_VIEW_DAILY,
    MAP_VIEW_PERSISTENCE,
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    LAYOUT_OPACITY,
    AIFS_TXTN_WARNING,
    NAV_WELCOME,
    NAV_MAP,
    NAV_METEO,
    NAV_WAVE,
    NAV_METHODS,
    NAV_LEGAL,
    NAV_ITEMS,
    NAV_ANALYTICS,
    EXPERT_FEATURES,
    STANDARD_DEFAULTS,
    MAP_VAR_LABELS,
    FORECAST_OFFSET_MIN,
    FORECAST_OFFSET_MAX,
    SLIDER_PAD_PAST,
    SLIDER_PAD_FUTURE,
    TOP10_MIN_PCT,
    TOP10_GRID_VERSION,
    TOP10_MASK_VERSION,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    show_expert,
    is_daily_map_view,
)
from backend_analytics import _synoptic_array, compute_map_footprint, calculate_top10
from frontend_plots import (
    _render_synoptic_map,
    build_baseline_map,
    build_opacity_slider_map,
    get_meteogram_traces,
    build_yearly_extremes_chart,
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

ASSETS_DIR = Path("assets")
DOCUMENTS_DIR = Path("Documents")
LEGAL_MD = ASSETS_DIR / "legal.md"
METHODS_MD = ASSETS_DIR / "methods.md"


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


_init_ui_mode()
if "forecast_model" not in st.session_state:
    st.session_state.forecast_model = FORECAST_MODEL_IFS

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
    clim_path = DATA_ROOT / "Reference_Climatology/climatology_reference_complete.nc"
    if not clim_path.exists(): 
        clim_path = DATA_ROOT / "Reference_Climatology/climatology_reference.nc"
    return xr.open_dataset(clim_path) if clim_path.exists() else None

ref_clim = load_reference_climatology()

@st.cache_resource(show_spinner=False)
def load_invariant_fields():
    """ERA5 time-invariant physiography fields (land-sea mask, orography,
    sub-grid orography variance) used to describe the physical footprint of
    a 0.25deg grid cell in the point-based tabs."""
    inv_path = DATA_ROOT / "Reference_Climatology/era5_invariants.nc"
    if not inv_path.exists():
        return None
    return xr.open_dataset(inv_path, engine="netcdf4")

INVARIANT_VARS = ("lsm", "z", "sdor")

def _classify_roughness(sdor_m):
    if sdor_m < 20:
        return "flat"
    if sdor_m < 75:
        return "hilly"
    if sdor_m < 200:
        return "low mountains"
    return "high mountains"

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

def _on_loc_history_change():
    """Recent-location pick is the active source; drop leftover typed search."""
    if st.session_state.get("loc_history_sel", "Select...") == "Select...":
        return
    st.session_state.loc_source = "history"
    st.session_state.new_loc_input = ""
    st.session_state.last_query = None
    st.session_state.geocode_results = None

def _on_new_loc_input_change():
    """Typed search is the active source; clear the recent-location pick."""
    st.session_state.loc_source = "search"
    st.session_state.loc_history_sel = "Select..."

def render_grid_cell_profile(location_name, lat, lon):
    """Collapsible ERA5 grid-cell physical metadata + scientific disclaimer,
    used only in the Point Meteogram / Point Wavogram tabs."""
    with st.expander("ℹ️ ERA5 Grid Cell Profile & Spatial Limits", expanded=False):
        inv_ds = load_invariant_fields()
        if inv_ds is None or not all(v in inv_ds.variables for v in INVARIANT_VARS):
            st.caption("Grid cell physiography data is currently unavailable.")
            return

        pt = inv_ds[list(INVARIANT_VARS)].sel(latitude=lat, longitude=lon, method="nearest")

        lsm = float(pt["lsm"].values)
        elevation = float(pt["z"].values) / 9.80665
        sdor = float(pt["sdor"].values)

        ns_extent = 27.8
        ew_extent = 27.8 * math.cos(math.radians(lat))
        area = ns_extent * ew_extent
        roughness_class = _classify_roughness(sdor)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
**Grid Cell Dimensions:**
The meteorological data for [{location_name}; {lat}°N, {lon}°E] is calculated based on a macro-scale ERA5 grid cell covering a total area of **{area:.1f} km²** (North-South: 27.8 km | East-West: {ew_extent:.1f} km).

**Modeled Physical Profile:**
* **Surface Cover:** {lsm*100:.0f}% Land | {(1-lsm)*100:.0f}% Water
* **Topography:** Mean elevation {elevation:.0f} m a.s.l. (Terrain roughness: {roughness_class})

⚠️ **Scientific Disclaimer:**
*AtmoPulse tracks large-scale synoptic anomalies. The displayed values represent a spatial and thermodynamic average over this entire {area:.1f} km² grid cell. Local on-the-ground measurements—especially within urbanized areas, smaller islands and/or highly structured terrain—will deviate significantly from these macro-scale baselines. Note: The underlying 0.25° model physics do not explicitly resolve urban infrastructure.*
""")
        with col2:
            if _FOLIUM_AVAILABLE:
                st_folium(
                    _create_gridcell_map(lat, lon),
                    width="100%",
                    height=300,
                    returned_objects=[],
                    key=f"gridcell_map_{lat:.2f}_{lon:.2f}",
                )
            else:
                st.caption("Map preview unavailable: install `folium` and `streamlit-folium` to enable it.")

@st.cache_resource(show_spinner=False)
def get_master_files():
    DATA_DIR = DATA_ROOT / "Master_Batches"
    return sorted(list(DATA_DIR.glob("era5_master_daily_*.nc")))

LIVE_TXTN = DATA_ROOT / "Live_Forecasts/live_forecast_txtn.nc"

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

QDM_TRANSFER_FILE = DATA_ROOT / "Reference_Climatology/qdm_transfer_functions.nc"

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

@st.cache_data(show_spinner=False)
def get_map_location_labels(lons_tuple, lats_tuple):
    return build_location_label_grid(np.array(lons_tuple), np.array(lats_tuple))

@st.cache_data(show_spinner=False)
def get_country_weight_grid(lons_tuple, lats_tuple, _version=TOP10_GRID_VERSION):
    return build_country_weight_grid(np.array(lons_tuple), np.array(lats_tuple))

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

def _top10_header_html(title: str) -> str:
    return f'<span class="atmopulse-subsection-label" title="{HELP["top10_table"]}"><b>{title}</b> ℹ️</span>'


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

def _slider_window_doys(anchor_date, pad_past=SLIDER_PAD_PAST, pad_future=SLIDER_PAD_FUTURE):
    """All calendar day-of-year values reachable via the Forecast Offset slider
    for a given anchor ('today') date."""
    dates = [anchor_date + pd.Timedelta(days=o) for o in range(-pad_past, pad_future + 1)]
    return tuple(sorted({int(etccdi_doy_365(d)) for d in dates}))

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

# --- METEOGRAM CORE TRACES (For Subplots) ---
def compute_point_thresholds(ref_clim, lat, lon, target_date, meteo_var, epoch):
    """
    ETCCDI percentile + all-time-record thresholds for one coordinate/day,
    shaped as (p_warm, p_cold) for backend_narrative.classify_point_severity().
    Uses the same 365-day ETCCDI calendar mapping as get_meteogram_traces()
    so the Point Meteogram narrative always matches the chart's climate
    boundaries envelope.
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


@st.cache_data(show_spinner=False)
def build_top10_table(df_live, meteo_var):
    """Trailing 12-month (365-day) hottest/coldest-day table below the main meteogram.

    Pure data-processing (returns a DataFrame, not a Plotly figure) -> cached
    with @st.cache_data so re-sorting/re-slicing df_live doesn't re-run on
    every unrelated widget rerun (e.g. the Layout radio).
    """
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
        if show_expert("forecast_model"):
            st.radio(
                "Forecast Model",
                FORECAST_MODEL_OPTIONS,
                key="forecast_model",
                help=HELP["forecast_model"],
            )
        else:
            st.session_state.forecast_model = FORECAST_MODEL_IFS
        _fc_tag = "AIFS" if is_aifs_model() else "IFS"
        st.markdown(
            f"<p style='font-size: 12px; color: #555; margin-top: -10px;'>📡 Data: ERA5 Archive (~ 5 days ago) | "
            f"{_fc_tag} Forecast ({default_date.strftime('%d.%m.%Y')} 12 UTC).</p>",
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
    # Map Layout is a core viewing control (not an expert-only toggle) -> render
    # it globally for Standard and Expert users alike.
    map_layout = st.radio("Map Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER, LAYOUT_OPACITY), horizontal=True)
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
            forecast_model=selected_forecast_model(),
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

    aifs_txtn_blocked = is_aifs_model() and map_var_code in ("TX", "TN")
    aifs_hatch_blocked = is_aifs_model() and bool(toggles.get("hatching"))
    if aifs_txtn_blocked or aifs_hatch_blocked:
        st.warning(AIFS_TXTN_WARNING)
    if not aifs_txtn_blocked:
        if aifs_hatch_blocked:
            toggles = {**toggles, "hatching": False}
        try:
            with st.spinner("Loading synoptic fields..."):
                map_phys_data, map_time_meta = fetch_cached_synoptic_data(
                    target_date.strftime('%Y-%m-%d'), default_date.strftime('%Y-%m-%d'),
                    forecast_model=selected_forecast_model(),
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
                if is_daily_map_view(view_mode):
                    # Cumulative severity ladder: Moderate INCLUDES Strong/Extreme/Record,
                    # Strong INCLUDES Extreme/Record, Extreme INCLUDES Record, Record is exclusive.
                    footprint_tier_order = ("moderate", "strong", "extreme", "record")
                    footprint_tier_titles = {"moderate": "Moderate", "strong": "Strong", "extreme": "Extreme", "record": "Record"}
                    footprint_tier_label = {
                        "moderate": "moderate (P75/25)", "strong": "strong (P90/10)",
                        "extreme": "extreme (P95/5)", "record": "all-time record",
                    }
                    # Active analysis level (Moderate/Strong/Extreme/All-Time Record radio)
                    # drives which single tier the TEXT sentence narrates.
                    active_tier = {
                        "Moderate": "moderate", "Strong": "strong",
                        "Extreme": "extreme", "All-Time Record": "record",
                    }.get(top10_threshold, "strong")
                    layout_choice = map_layout  # the exact Map Layout radio state — "Side-by-Side Compare" | "Single Map Flicker" | "Opacity Slider Compare"
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    anchor_date_str = default_date.strftime('%Y-%m-%d')

                    if layout_choice == LAYOUT_FLICKER:
                        # --- SINGLE BASELINE STATE: one map on screen -> one set of cumulative percentages ---
                        active_epoch = "A" if "A" in flicker_epoch else "B"
                        active_baseline = EPOCH_LABELS[active_epoch]
                        footprint_single = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            active_epoch, map_var_code, anchor_date_str=anchor_date_str,
                        )
                        if footprint_single:
                            # TEXT SENTENCE — always rendered, Standard and Expert alike.
                            single_val = footprint_single[active_tier]["total_pct"]
                            st.markdown(
                                f"**Based on the {active_baseline} baseline**, **{single_val:.1f}%** of Europe is experiencing "
                                f"{footprint_tier_label[active_tier]} anomalies."
                            )
                            # DETAILED BREAKDOWN — Expert Mode only, placed directly below the text.
                            if is_expert_mode():
                                metric_cols = st.columns(4)
                                for m_col, tier in zip(metric_cols, footprint_tier_order):
                                    with m_col:
                                        st.metric(footprint_tier_titles[tier], f"{footprint_single[tier]['total_pct']:.1f}%")
                    else:
                        # --- COMPARE STATE (Side-by-Side or Opacity Compare): both maps on screen -> both baselines' cumulative percentages ---
                        footprint_a = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "A", map_var_code, anchor_date_str=anchor_date_str,
                        )
                        footprint_b = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "B", map_var_code, anchor_date_str=anchor_date_str,
                        )
                        if footprint_a and footprint_b:
                            # TEXT SENTENCE — always rendered, Standard and Expert alike.
                            hist_val = footprint_a[active_tier]["total_pct"]
                            rec_val = footprint_b[active_tier]["total_pct"]
                            trend_word = "amplified" if rec_val > hist_val else "reduced" if rec_val < hist_val else "unchanged"
                            st.markdown(
                                f"Relative to the historical **{EPOCH_LABELS['A']}** baseline, **{hist_val:.1f}%** of Europe is experiencing "
                                f"{footprint_tier_label[active_tier]} anomalies. Under the recent **{EPOCH_LABELS['B']}** climate state, "
                                f"this footprint is {trend_word} to **{rec_val:.1f}%**."
                            )
                            # DETAILED TABLE (1961-1990 vs 1996-2025 vs Delta) — Expert Mode only,
                            # placed directly below the text.
                            if is_expert_mode():
                                table_rows = "\n".join(
                                    f"| {footprint_tier_titles[t]} | {footprint_a[t]['total_pct']:.1f}% | "
                                    f"{footprint_b[t]['total_pct']:.1f}% | {footprint_b[t]['total_pct'] - footprint_a[t]['total_pct']:+.1f} pp |"
                                    for t in footprint_tier_order
                                )
                                st.markdown(
                                    f"| Severity (cumulative) | {EPOCH_LABELS['A']} | {EPOCH_LABELS['B']} | Δ |\n"
                                    f"|---|---|---|---|\n{table_rows}"
                                )

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    fig_a = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    fig_b = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    with st.container(key="atmopulse_map_columns"):
                        mc1, mc2 = st.columns(2, gap="small")
                        with mc1:
                            _render_synoptic_map(fig_a, "Historical Baseline (1961-1990)", "map_a")
                        with mc2:
                            _render_synoptic_map(fig_b, "Recent Baseline (1996-2025)", "map_b")
                    map_col1, map_col2 = st.columns(2)
                    with map_col1:
                        df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_a, df_c_a)
                    with map_col2:
                        df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_b, df_c_b)
                elif map_layout == LAYOUT_OPACITY:
                    fig_a = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, full_width=True, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    fig_b = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, full_width=True, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    _render_synoptic_map(
                        build_opacity_slider_map(fig_a, fig_b),
                        "Opacity Slider Compare: Historical Baseline (1961-1990) ↔ Recent Baseline (1996-2025)",
                        "map_opacity",
                        bottom_margin=60,
                    )
                    st.caption("Drag the slider under the map to cross-fade between the two reference periods. Zoom and tooltips stay interactive.")
                    map_col1, map_col2 = st.columns(2)
                    with map_col1:
                        df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_a, df_c_a, "Historical Baseline (1961–1990)")
                    with map_col2:
                        df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_b, df_c_b, "Recent Baseline (1996–2025)")
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
                            border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays,
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
                            _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid,
                        )
                        render_top10_tables(df_h, df_c)
        except Exception as e: 
            st.error(f"Error loading maps: {e}")

elif nav_selection in (NAV_METEO, NAV_WAVE):
    st.subheader("🏙️ Target Location")
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1: 
        loc_history_sel = st.selectbox(
            "Select recent location:",
            ["Select..."] + st.session_state.search_history,
            key="loc_history_sel",
            on_change=_on_loc_history_change,
        )
    with search_col2: 
        new_loc_input = st.text_input(
            "Or select new location (Press Enter to see options):",
            key="new_loc_input",
            on_change=_on_new_loc_input_change,
        )
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    location = None
    use_history = (
        loc_history_sel != "Select..."
        and st.session_state.get("loc_source") == "history"
    )
    if use_history:
        location = geolocator.geocode(loc_history_sel, timeout=10)
    elif new_loc_input:
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
        render_grid_cell_profile(location.address, lat_target, lon_target)
        if nav_selection == NAV_METEO:
            if show_expert("flicker_layout"):
                map_layout = st.radio("Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True, key="met_layout")
            else:
                map_layout = STANDARD_DEFAULTS["map_layout"]
            # Determine the active Flicker-mode reference period BEFORE the
            # narrative text is built below, so the sentence never "leaks" a
            # hardcoded baseline that doesn't match what the widget actually
            # shows under the chart. Re-rendered later at its original chart
            # position using the SAME key (Streamlit persists the selection
            # across the rerun, so reading it here is safe).
            if map_layout == LAYOUT_FLICKER:
                met_active_epoch = "A" if "A" in st.session_state.get("met_ep", "B (1996–2025)") else "B"
            if is_aifs_model() and meteo_var in ("Max Temp (TX)", "Min Temp (TN)"):
                st.warning(AIFS_TXTN_WARNING)
            else:
                with st.spinner("Fetching Meteogram data..."): 
                    df_live = get_live_point_series(lat_target, lon_target, selected_forecast_model())
                if not df_live.empty:
                    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')

                    # --- STRICT DATETIME INDEXING for "current conditions" ---
                    # Never .max()/.mean() over the series, and never a bare
                    # .iloc[0]/.iloc[-1] unless target_date genuinely falls
                    # outside the live window — the scalar MUST come from the
                    # exact calendar row matching the active target_date.
                    #
                    # `df_live` (get_live_point_series) is built entirely from calendar-day
                    # UTC aggregates — the ERA5 archive's daily valid_time and the IFS/AIFS
                    # forecast's own 00Z-00Z daily aggregation (ifs_ingestion.py) — so it is
                    # already on the same UTC calendar-day footing as `active_date` below.
                    df_indexed = df_live.copy()
                    df_indexed['Date'] = pd.to_datetime(df_indexed['Date']).dt.tz_localize(None).dt.normalize()
                    df_indexed = df_indexed.drop_duplicates(subset=['Date']).set_index('Date').sort_index()
                    active_date = (
                        pd.Timestamp.utcnow().tz_localize(None).floor('D')
                        + pd.Timedelta(days=st.session_state.offset_slider)
                    )

                    try:
                        current_row = df_indexed.loc[active_date]
                        current_row_date = active_date
                    except KeyError:
                        # Defensive-only safety net (e.g. offset_slider pushed past what
                        # the live series actually returned, or an upstream API gap) —
                        # NOT the primary alignment mechanism anymore. Nearest available
                        # calendar day, never the series' arbitrary last/forecast row.
                        nearest_pos = df_indexed.index.get_indexer([active_date], method='nearest')[0]
                        current_row = df_indexed.iloc[nearest_pos]
                        current_row_date = df_indexed.index[nearest_pos]

                    if col_target in df_indexed.columns:
                        value_now = float(current_row[col_target])
                    elif 'TX' in df_indexed.columns and 'TN' in df_indexed.columns:
                        value_now = float((current_row['TX'] + current_row['TN']) / 2.0)
                    else:
                        value_now = np.nan

                    # Thresholds must be evaluated for THAT specific day (current_row_date),
                    # not the slider's nominal target_date, so a forecast-fallback row never
                    # gets scored against the wrong calendar day's P75/90/95 climatology.
                    #
                    # Baseline isolation: each epoch gets its OWN, freshly-built p_warm/p_cold
                    # dict from compute_point_thresholds — "A" and "B" thresholds are never
                    # assigned into the same variable, so there is no possibility of one
                    # baseline's percentiles silently overwriting the other's.
                    # Both baselines' thresholds are computed unconditionally (cheap — just
                    # ref_clim.sel()+array-index lookups, no I/O) so the debug readout below
                    # can always show both, regardless of which layout is active.
                    percentiles_a = compute_point_thresholds(ref_clim, lat_target, lon_target, current_row_date, meteo_var, "A")
                    percentiles_b = compute_point_thresholds(ref_clim, lat_target, lon_target, current_row_date, meteo_var, "B")
                    cat_a, dir_a = classify_point_severity(value_now, *percentiles_a)
                    cat_b, dir_b = classify_point_severity(value_now, *percentiles_b)
                    condition_a = f"{cat_a} {dir_a}" if dir_a else "normal"
                    condition_b = f"{cat_b} {dir_b}" if dir_b else "normal"  # e.g. "extreme warm", "moderate cold", "normal"

                    if map_layout == LAYOUT_FLICKER:
                        # Single-baseline state: narrate strictly the ACTIVE epoch (the one the
                        # Flicker radio below is actually showing) — never a hardcoded baseline.
                        cat_x, dir_x = (cat_a, dir_a) if met_active_epoch == "A" else (cat_b, dir_b)
                        condition_x = condition_a if met_active_epoch == "A" else condition_b
                        x_txt = "within its normal range" if condition_x == "normal" else f"{condition_x} conditions"
                        condition_string = (
                            f"The area of {location.address} is currently experiencing {x_txt} relative to "
                            f"the {EPOCH_LABELS[met_active_epoch]} baseline."
                        )
                        condition_b = condition_x  # drives the styling block below
                    else:
                        # TASK 2: chronological order — historical (A, 1961-1990) baseline
                        # narrated FIRST, recent (B, 1996-2025) baseline SECOND, matching the
                        # left-to-right reading order of the UI.
                        a_txt = "within its normal range" if condition_a == "normal" else f"{condition_a} conditions"
                        b_txt = "within its normal range" if condition_b == "normal" else f"{condition_b} conditions"
                        condition_string = (
                            f"The area of {location.address} is currently experiencing {a_txt} relative to the "
                            f"historical {EPOCH_LABELS['A']} baseline. Compared to the recent {EPOCH_LABELS['B']} "
                            f"climate state, this equates to {b_txt}."
                        )

                    # --- Conditional styling: driven by the ACTIVE (baseline-B) severity tier,
                    # not a naive substring search of the sentence, so e.g. a compare-mode
                    # sentence mentioning both "warm" and "cold" can't pick the wrong branch.
                    if condition_b in ("record warm", "extreme warm", "strong warm"):
                        st.error(condition_string, icon="🔥")
                    elif condition_b == "moderate warm":
                        st.warning(condition_string)
                    elif condition_b in ("record cold", "extreme cold", "strong cold"):
                        st.info(condition_string, icon="❄️")
                    elif condition_b == "moderate cold":
                        st.markdown(
                            f"""<div style="background-color:{ATMOPULSE_COLD['p25']}; color:#003554;
                            padding:0.75rem 1rem; border-radius:0.5rem;">🧊 {condition_string}</div>""",
                            unsafe_allow_html=True,
                        )
                    else:  # "normal" — no significant anomaly
                        st.success(condition_string, icon="✅")

                    t_arr = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)
                    global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
                    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)

                    traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "A", show_air_temp, show_app_temp, meteo_env, meteo_var, current_condition=(cat_a, dir_a))
                    traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "B", show_air_temp, show_app_temp, meteo_env, meteo_var, current_condition=(cat_b, dir_b))

                    # TASK 2: compact HTML badge legend (Map Tracker style) for the 8
                    # warm/cold severity tiers, replacing Plotly's own cluttered legend
                    # for these fills. "Typical Range" (P25-P75) called out separately.
                    _s_normal = "background-color:rgba(180,180,180,0.4); color:#333; padding:1px 6px; border-radius:3px; font-family:" + ATMOPULSE_FONTS['outfit_css'] + "; font-size:12px; font-weight:" + str(ATMOPULSE_FONTS['ui_weight']) + ";"
                    st.markdown(
                        f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
                        f"style='margin-bottom: 6px; white-space: nowrap;'>"
                        f"<b>Legend.</b> "
                        f"<span style='{_s_normal}'>Typical Range</span>"
                        f"<span style='padding-left: 12px;'>Warm:</span> "
                        f"<span style='{legend_badge_style('warm', 'moderate')}'>Moderate</span> "
                        f"<span style='{legend_badge_style('warm', 'strong')}'>Strong</span> "
                        f"<span style='{legend_badge_style('warm', 'extreme')}'>Extreme</span> "
                        f"<span style='{legend_badge_style('warm', 'record')}'>Record</span>"
                        f"<span style='padding-left: 12px;'>Cold:</span> "
                        f"<span style='{legend_badge_style('cold', 'moderate')}'>Moderate</span> "
                        f"<span style='{legend_badge_style('cold', 'strong')}'>Strong</span> "
                        f"<span style='{legend_badge_style('cold', 'extreme')}'>Extreme</span> "
                        f"<span style='{legend_badge_style('cold', 'record')}'>Record</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    if map_layout == LAYOUT_SIDE_BY_SIDE:
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), shared_yaxes=True)
                        for trace in traces_a: 
                            fig.add_trace(trace, row=1, col=1)
                        for trace in traces_b: 
                            fig.add_trace(trace, row=1, col=2)
                            
                        fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                        fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                        fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                        fig.update_yaxes(range=[global_min, global_max])
                        fig.update_layout(**plotly_typography(), hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        c1, c2 = st.columns(2)
                        with c1: 
                            st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A", meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)
                        with c2: 
                            st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "B", meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)
                    else:
                        # Same widget key ("met_ep") whose value we already read into
                        # `met_active_epoch` above (before the narrative text was built) —
                        # re-rendering it here just places it at its usual spot below the chart.
                        flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="met_ep", index=1)
                        met_active_epoch = "A" if "A" in flicker_epoch else "B"
                        fig = go.Figure(data=traces_a if met_active_epoch == "A" else traces_b)
                        fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8)
                        fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                        fig.update_yaxes(range=[global_min, global_max])
                        fig.update_layout(**plotly_typography(), title=f"Reference Period {EPOCH_LABELS[met_active_epoch]}", hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, met_active_epoch, meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)

        elif nav_selection == NAV_WAVE:
            if show_expert("flicker_layout"):
                map_layout = st.radio("Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True, key="wave_layout")
            else:
                map_layout = STANDARD_DEFAULTS["map_layout"]
            if is_aifs_model():
                st.warning(AIFS_TXTN_WARNING)
            else:
                with st.spinner("Generating Historical Waves..."):
                    param_code = "TX" if "Heatwaves" in wave_focus else "TN"
                    fig_m_a, fig_s_a = fetch_wave_figs(lat_target, lon_target, param_code, "A", wave_thresh, wave_stat_metric)
                    fig_m_b, fig_s_b = fetch_wave_figs(lat_target, lon_target, param_code, "B", wave_thresh, wave_stat_metric)

                # STATIC vs. UI overlay state: always ranks against the full 1940-present
                # ERA5 record, independent of the Side-by-Side / Flicker layout selected above.
                wave_rank_info = get_wave_historical_rank(
                    lat_target, lon_target, parameter=param_code,
                    selected_epoch="B", threshold_level=wave_thresh,
                )
                if wave_rank_info is not None:
                    wave_rank_text = (
                        f"The area of {location.address} is currently experiencing its "
                        f"{wave_rank_info['rank_ordinal']} longest {wave_rank_info['severity']} "
                        f"{wave_rank_info['wave_type']} since the start of the ERA5 record in 1940."
                    )
                    st.markdown(f"**{wave_rank_text}**")

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