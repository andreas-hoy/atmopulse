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
import threading
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime

from backend_maps import etccdi_doy_365
from backend_waves import get_wave_historical_rank
from labels import HELP
from atmopulse_theme import (
    ATMOPULSE_BRAND,
    atmopulse_streamlit_css,
    atmopulse_wordmark_html,
    LOGO_SVG,
)
from config import (
    UI_MODE_STANDARD,
    UI_MODE_EXPERT,
    UI_MODE_LABELS,
    FORECAST_MODEL_IFS,
    FORECAST_MODEL_OPTIONS,
    MAP_VIEW_DAILY,
    MAP_VIEW_PERSISTENCE,
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    AIFS_TXTN_WARNING,
    NAV_WELCOME,
    NAV_MAP,
    NAV_METEO,
    NAV_WAVE,
    NAV_METHODS,
    NAV_LEGAL,
    NAV_ITEMS,
    NAV_ANALYTICS,
    STANDARD_DEFAULTS,
    FORECAST_OFFSET_MIN,
    FORECAST_OFFSET_MAX,
    SLIDER_PAD_PAST,
    SLIDER_PAD_FUTURE,
    is_expert_mode,
    is_aifs_model,
    show_expert,
    is_daily_map_view,
)
from backend_io import fetch_wave_figs
from frontend_widgets import render_grid_cell_profile
from page_map_tracker import render_map_tracker
from page_meteogram import render_meteogram

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
    render_map_tracker(map_var_code, view_mode, persist_metric, top10_threshold, toggles, target_date, default_date)

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
            render_meteogram(location, lat_target, lon_target, meteo_var, meteo_env, show_air_temp, show_app_temp, target_date)

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