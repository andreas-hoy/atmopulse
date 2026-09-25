"""
AtmoPulse Central Configuration (config.py)

Single source of truth for UI constants, navigation, feature flags, and the
on-disk data root. Extracted from app.py so frontend_plots.py,
backend_analytics.py, and app.py itself can all import the same values
without importing each other at module load time (avoids circular imports).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
MAP_VIEW_WAVES = "Wave tracking"

# Map Tracker view radio, by audience mode. Standard omits Persistence;
# Wave tracking is offered in both, always after Persistence when present.
# Default view (STANDARD_DEFAULTS["map_view"]) stays MAP_VIEW_DAILY.
MAP_VIEW_OPTIONS_STANDARD = (MAP_VIEW_DAILY, MAP_VIEW_WAVES)
MAP_VIEW_OPTIONS_EXPERT = (MAP_VIEW_DAILY, MAP_VIEW_PERSISTENCE, MAP_VIEW_WAVES)

# Map Tracker offers all four. Meteogram and Wavogram offer the chart pair.
# Single Map / Single chart is the opening view (recent baseline 1996–2025).
LAYOUT_SINGLE_MAP = "Single Map"
LAYOUT_SINGLE_CHART = "Single chart"
LAYOUT_SIDE_BY_SIDE = "Side by side"
LAYOUT_SWIPE = "Swipe"
LAYOUT_OPACITY = "Opacity"

# Comparison *axis* (orthogonal to LAYOUT_*): climatology A vs B at one time,
# or two times against one climatology. Map Tracker uses COMPARE_DATES
# (Live or Archive left, Archive right). Meteogram uses COMPARE_YEARS
# (Live or a calendar year left, partner year right). Both radios read
# "Two dates".
COMPARE_EPOCHS = "Reference periods"
COMPARE_DATES = "Two dates"
COMPARE_YEARS = "Two dates"

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
# t850 / utci are catalogued here and wired in a later step. "jet" (300 hPa
# wind quiver overlay, Map Tracker only) is wired.
EXPERT_FEATURES = frozenset({
    "map_tx_tn",
    "persistence_view",
    "map_analysis_level",
    "percentile_layer_toggles",
    "z500",
    "synoptic_anomalies",
    "t850",
    "jet",
    "utci",
    "meteo_tx_tn",
    "meteo_envelope",
    "forecast_model",
    "meteo_wsdi_csdi",
    "wave_annual_cycle",
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
    "jet": False,
    "mslp_anom": False,
    "z500_anom": False,
    "meteo_var": "Mean Temp (TG)",
    "meteo_env": "Strong",
    "meteo_count": METEO_COUNT_ALL,
    "wave_thresh": "Strong",
    "wave_z500_outline": False,
    "map_wave_level": "Strong",
    "map_layout": LAYOUT_SINGLE_MAP,
    "chart_layout": LAYOUT_SINGLE_CHART,
    "map_compare": COMPARE_EPOCHS,
    "meteo_compare": COMPARE_EPOCHS,
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


def epoch_short_label(epoch: str) -> str:
    """Radio option: Historical (1961–1990) / Recent (1996–2025)."""
    key = "A" if str(epoch).upper().startswith("A") else "B"
    name = "Historical" if key == "A" else "Recent"
    return f"{name} ({EPOCH_LABELS[key]})"


def epoch_from_label(text: str) -> str:
    """Parse a radio/title string to epoch A or B (do not use `'A' in label`)."""
    raw = str(text)
    if "1961" in raw or "Historical" in raw:
        return "A"
    return "B"


def analog_calendar_date(live, *, years_back: int = 1, min_date=None, max_date=None):
    """Same month/day ``years_back`` earlier, clamped to an optional range.

    29 Feb maps to 28 Feb in a non-leap analog year (ETCCDI has no 29 Feb slot).
    """
    live = pd.Timestamp(live).normalize()
    year = int(live.year) - int(years_back)
    try:
        analog = live.replace(year=year)
    except ValueError:
        analog = live.replace(year=year, day=28)
    if min_date is not None:
        analog = max(analog, pd.Timestamp(min_date).normalize())
    if max_date is not None:
        analog = min(analog, pd.Timestamp(max_date).normalize())
    return analog


def analog_date_in_year(live, year: int):
    """Same month/day as ``live`` in ``year`` (29 Feb → 28 Feb if needed)."""
    live = pd.Timestamp(live).normalize()
    year = int(year)
    try:
        return live.replace(year=year)
    except ValueError:
        return live.replace(year=year, day=28)

FORECAST_OFFSET_MIN = -7
FORECAST_OFFSET_MAX = 3
SLIDER_PAD_PAST = 7    # matches abs(FORECAST_OFFSET_MIN)
SLIDER_PAD_FUTURE = 3  # matches FORECAST_OFFSET_MAX

TOP10_MIN_PCT = 0.5
TOP10_GRID_VERSION = 5  # bump when country-filter or top10 mask rules change (invalidates st.cache_data)
TOP10_MASK_VERSION = 1

# Kyselý wave intensity (K·days accumulated excess over the main threshold).
# Single source of truth for BOTH the Point Wavogram ridge colour ramp
# (frontend_wavogram.py) and the Map Tracker "Wave tracking" colorbar
# (frontend_maps.py): values at/above the cap saturate to the full warm/
# cold colour; hover always shows the true, uncapped number.
WAVE_INTENSITY_CAP_TX = 100.0    # heat (TX), K·days
WAVE_INTENSITY_CAP_TN = 200.0    # cold (TN), K·days

# --- Wave tracking (Map Tracker spatial Kyselý view) ---
# Precomputed daily wave-status raster: analog to the anomaly/climatology
# stores, NOT era5_master_time_series.zarr (that Zarr is chunked for point
# reads -- time=-1, lat/lon 10x10 tiles -- useless for a Europe-wide map).
# Built/incrementally updated by batch_precompute_waves.py.
WAVE_STATUS_DIR = DATA_ROOT / "Wave_Status"
WAVE_STATUS_ZARR = WAVE_STATUS_DIR / "wave_status.zarr"

# Map-optimal chunking: one (epoch, calendar day) slice is one chunk, so a
# map load touches exactly one chunk per data variable -- same order of
# magnitude as the Daily snapshot's own single-DOY climatology read.
WAVE_MAP_CHUNKS = {"epoch": 1, "valid_time": 1, "latitude": -1, "longitude": -1}

# Wave view's own Analysis / threshold control: Strong | Extreme only (no
# Moderate, no All-Time Record -- Kyselý has no such tiers).
MAP_WAVE_LEVELS = ("Strong", "Extreme")
MAP_WAVE_VARS = ("TX", "TN", "TG", "T850")          # Expert; Standard locks TX (heat) / TN (cold)
MAP_WAVE_DIRECTIONS = ("heat", "cold")


def wave_map_var_key(var_code: str) -> str:
    """Map-var code (TX/TN/TG/T850) -> on-disk variable key (tx/tn/tg/t850)."""
    return str(var_code).lower()


def wave_map_direction(is_warm: bool) -> str:
    return "heat" if is_warm else "cold"


def wave_map_tier(level: str) -> str:
    return "extreme" if "Extreme" in str(level) else "strong"


def wave_map_field(var_code: str, is_warm: bool, level: str, *, kind: str = "int") -> str:
    """Data-variable name inside WAVE_STATUS_ZARR, e.g. 'tx_heat_strong_int'
    (running Kyselý intensity-to-date, K·days, 0 = not in a wave) or
    'tx_heat_strong_dur' (packed int16 duration-to-date, days elapsed in
    the current wave through that map date, 0 = not in a wave)."""
    return f"{wave_map_var_key(var_code)}_{wave_map_direction(is_warm)}_{wave_map_tier(level)}_{kind}"


def wave_intensity_cap(is_warm: bool) -> float:
    """Same cap used by the Wavogram ridge colour and the Wave map colorbar."""
    return WAVE_INTENSITY_CAP_TX if is_warm else WAVE_INTENSITY_CAP_TN


def wave_locked_var_code(is_warm: bool) -> str:
    """Standard mode's locked Map Tracker Wave variable: TX for heat, TN for
    cold (Expert may instead pick any of MAP_WAVE_VARS via the normal
    Mapped Variable radio)."""
    return "TX" if is_warm else "TN"


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
    """True only for the Daily snapshot (DOY percentile) view.

    Was previously `view_mode != MAP_VIEW_PERSISTENCE`, which silently
    treated any future third view as Daily. Wave tracking is its own
    view — never Daily's percentile mask / spell hatching / footprint —
    so this must be an explicit equality, not "not Persistence".
    """
    return view_mode == MAP_VIEW_DAILY


def is_wave_map_view(view_mode: str) -> bool:
    """True only for the Wave tracking (Kyselý intensity-to-date) view."""
    return view_mode == MAP_VIEW_WAVES
