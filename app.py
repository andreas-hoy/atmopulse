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
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import threading
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime

from backend_map_locations import build_location_label_grid, build_country_weight_grid, EUROPE_BBOX
from backend_maps import etccdi_doy_365
from backend_waves import get_wave_historical_rank
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
from backend_io import (
    load_reference_climatology,
    load_invariant_fields,
    get_master_files,
    get_master_archive_ds,
    get_live_txtn_ds,
    get_live_point_series,
    fetch_cached_synoptic_data,
    get_persistence_arrays,
    get_map_historical_records_bundle,
    compute_point_thresholds,
    _load_persistence_daily_series,
    _load_point_archive_series,
    fetch_wave_figs,
)
from frontend_widgets import render_grid_cell_profile, build_top10_table, _top10_header_html

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
ref_clim = load_reference_climatology()

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

def _fmt_map_year(yr_val) -> str:
    try:
        y = int(float(yr_val))
        return str(y) if y > 0 else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def _slider_window_doys(anchor_date, pad_past=SLIDER_PAD_PAST, pad_future=SLIDER_PAD_FUTURE):
    """All calendar day-of-year values reachable via the Forecast Offset slider
    for a given anchor ('today') date."""
    dates = [anchor_date + pd.Timedelta(days=o) for o in range(-pad_past, pad_future + 1)]
    return tuple(sorted({int(etccdi_doy_365(d)) for d in dates}))

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