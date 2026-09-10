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

# Point-extraction-optimal Zarr mirror of the master archive (see
# batch_convert_netcdf_to_zarr.py). Temporally-contiguous, small lat/lon
# tile chunking turns a 10-25s NetCDF point read into a millisecond-scale
# read. Built offline/on a schedule, not at request time. Lives in config.py
# (rather than backend_io.py) so backend_io and backend_waves can both import
# this path without importing each other.
ZARR_MASTER_TIME_SERIES = DATA_ROOT / "Zarr_Archive" / "era5_master_time_series.zarr"

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
LAYOUT_SWIPE = "Swipe Slider Compare"

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
    "map_analysis_level",
    "percentile_layer_toggles",
    "z500",
    "t850",
    "jet",
    "utci",
    "meteo_tx_tn",
    "meteo_envelope",
    "wave_stat_metric",
    "flicker_layout",
    "forecast_model",
    "meteo_wsdi_csdi",
})

SPELL_OFF = "Off"
SPELL_LABELS = (SPELL_OFF, "6 days", "15 days", "30 days")

METEO_COUNT_ALL = "All anomaly days"
METEO_COUNT_SPELL = "Spell days"
METEO_COUNT_OPTIONS = (METEO_COUNT_ALL, METEO_COUNT_SPELL)

# Consecutive-day lookback for persistence maps and the heat/cold-spell overlay.
# Count is unbroken from the map date backward; one sub-threshold day resets.
# 100 days covers long Mediterranean warm spells (2 m temperature, not SST).
PERSISTENCE_MAX_DAYS = 100
PERSISTENCE_LOOKBACK_PAD = 5  # extra days loaded so calendar gaps don't shrink the kept window
# Colour-bar range for the persistence map. Counts still run to PERSISTENCE_MAX_DAYS;
# values beyond this saturate so typical land spells remain readable.
PERSISTENCE_COLORBAR_DAYS = 30

STANDARD_DEFAULTS = {
    "map_var": "Mean Temperature (TG)",
    "map_view": MAP_VIEW_DAILY,
    "persist_metric": "Strong",
    "analysis_level": "Strong",
    "hatching": False,
    "spell_label": SPELL_OFF,
    "spell_days": 6,
    "mslp": True,
    "z500": False,
    "meteo_var": "Mean Temp (TG)",
    "meteo_env": "Strong",
    "meteo_count": METEO_COUNT_ALL,
    "wave_thresh": "Strong",
    "wave_stat_metric": "Cumulative Annual Wave Intensity",
    "map_layout": LAYOUT_SIDE_BY_SIDE,
}

MAP_VAR_LABELS = {
    "TG": "Mean Temperature",
    "TX": "Maximum Temperature",
    "TN": "Minimum Temperature",
    "T850": "850 hPa Temperature",
}
MAP_VAR_OPTIONS = (
    "Mean Temperature (TG)",
    "Maximum Temperature (TX)",
    "Minimum Temperature (TN)",
    "850 hPa Temperature (T850)",
)


def meteo_var_code(meteo_var: str) -> str:
    """Map a sidebar/map label to TX | TN | TG | T850."""
    raw = str(meteo_var)
    if "T850" in raw:
        return "T850"
    if "TX" in raw:
        return "TX"
    if "TN" in raw:
        return "TN"
    return "TG"


EPOCH_LABELS: dict[str, str] = {"A": "1961–1990", "B": "1996–2025"}
EPOCH_PERIOD_NAME: dict[str, str] = {
    "A": "Historical Reference Period",
    "B": "Recent Reference Period",
}


def epoch_period_label(epoch: str) -> str:
    """UI title shared by Map Tracker, Meteogram, and Wavogram."""
    key = "A" if str(epoch).upper().startswith("A") else "B"
    return f"{EPOCH_PERIOD_NAME[key]} ({EPOCH_LABELS[key]})"


def epoch_from_label(text: str) -> str:
    """Parse a radio/title string to epoch A or B (do not use `'A' in label`)."""
    raw = str(text)
    if "1961" in raw or "Historical" in raw:
        return "A"
    return "B"

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
