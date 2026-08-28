"""
AtmoPulse Central Configuration (config.py)

Single source of truth for UI constants, navigation, feature flags, and the
on-disk data root. Extracted from app.py so frontend_plots.py,
backend_analytics.py, and app.py itself can all import the same values
without importing each other at module load time (avoids circular imports).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# --- Data root (single source of truth for the on-disk climate warehouse) ---
DATA_ROOT = Path("ERA5_ClimateTool")
MASTER_BATCHES_DIR = DATA_ROOT / "Master_Batches"
LIVE_FORECASTS_DIR = DATA_ROOT / "Live_Forecasts"
REFERENCE_CLIMATOLOGY_DIR = DATA_ROOT / "Reference_Climatology"

# --- Audience mode (Standard vs Expert) is independent of map *view* ---
# (daily snapshot vs persistence duration).
UI_MODE_STANDARD = "standard"
UI_MODE_EXPERT = "expert"
UI_MODE_LABELS = ("Standard", "Expert")

FORECAST_MODEL_IFS = "IFS (Physics-based)"
FORECAST_MODEL_AIFS = "AIFS (Machine Learning)"
FORECAST_MODEL_OPTIONS = (FORECAST_MODEL_IFS, FORECAST_MODEL_AIFS)

MAP_VIEW_DAILY = "Daily snapshot"
MAP_VIEW_PERSISTENCE = "Persistence duration"

LAYOUT_SIDE_BY_SIDE = "Side-by-Side Compare"
LAYOUT_FLICKER = "Single Map Flicker"
LAYOUT_OPACITY = "Opacity Slider Compare"

AIFS_TXTN_WARNING = (
    "Diurnal extreme analytics (TX/TN and associated Wave Tracking) are currently unavailable "
    "for the AIFS model. Due to the model's discontinuous 6-hourly autoregressive state jumps, "
    "true continuous boundary layer extremes cannot be natively resolved. Please switch to the "
    "IFS model for TX/TN analytics, or utilize Mean Temperature (TG) and T850 for AIFS."
)

NAV_WELCOME = "Welcome"
NAV_MAP = "Map Tracker"
NAV_METEO = "Point Meteogram"
NAV_WAVE = "Point Wavogram"
NAV_METHODS = "Methods & Resources"
NAV_LEGAL = "Legal & Terms"
NAV_ITEMS = (NAV_WELCOME, NAV_MAP, NAV_METEO, NAV_WAVE, NAV_METHODS, NAV_LEGAL)
NAV_ANALYTICS = (NAV_MAP, NAV_METEO, NAV_WAVE)

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
    "forecast_model",
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

MAP_VAR_LABELS = {"TG": "Mean Temperature", "TX": "Maximum Temperature", "TN": "Minimum Temperature"}

FORECAST_OFFSET_MIN = -7
FORECAST_OFFSET_MAX = 3
SLIDER_PAD_PAST = 7    # matches abs(FORECAST_OFFSET_MIN)
SLIDER_PAD_FUTURE = 3  # matches FORECAST_OFFSET_MAX

TOP10_MIN_PCT = 0.5
TOP10_GRID_VERSION = 5  # bump when country-filter or top10 mask rules change (invalidates st.cache_data)
TOP10_MASK_VERSION = 1


def is_expert_mode() -> bool:
    return st.session_state.get("ui_mode") == UI_MODE_EXPERT


def is_aifs_model() -> bool:
    return "AIFS" in str(st.session_state.get("forecast_model", FORECAST_MODEL_IFS))


def selected_forecast_model() -> str:
    return str(st.session_state.get("forecast_model", FORECAST_MODEL_IFS))


def show_expert(feature: str) -> bool:
    """True when an expert-only control should be shown and honoured."""
    return is_expert_mode() and feature in EXPERT_FEATURES


def is_daily_map_view(view_mode: str) -> bool:
    return view_mode != MAP_VIEW_PERSISTENCE
