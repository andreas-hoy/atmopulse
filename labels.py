"""
AtmoPulse UI Copy (labels.py)

Central store for the tooltip/help text shown throughout the Streamlit UI
(app.py). Keeping this copy separate from layout and business logic means:

- One place to review or edit wording, instead of hunting through app.py.
- app.py stays focused on structure/state, not prose.
- Easy to swap in translations later (e.g. a HELP_DE dict) without touching
  any UI code.

This module is a plain Python module (not a static asset), so it lives at
the project root next to app.py / atmopulse_theme.py / backend_*.py — the
assets/ folder is reserved for genuinely static, non-code resources (SVGs,
the favicon, legal.md, etc.) that are read as files rather than imported.

Usage:
    from labels import HELP
    st.slider(..., help=HELP["forecast_offset"])
"""

from __future__ import annotations

HELP: dict[str, str] = {
    # --- Sidebar: global controls ---
    "ui_mode": (
        "Standard: core temperature extremes for media, education, and the interested public. "
        "Expert: additional parameters and synoptic overlays for meteorologists and climatologists."
    ),
    "data_vintage": (
        "Data Origin (Hybrid System Specifications):\n\n"
        "1. ERA5 Reanalysis: The primary climate reference dataset. Fully quality-assured "
        "data is typically available with a latency of 2 to 3 months behind real-time.\n\n"
        "2. ERA5T (Preliminary): Preliminary daily updates that seamlessly close the gap "
        "between the final ERA5 release and approximately 5 days prior to the present.\n\n"
        "3. ECMWF IFS (Analysis & HRES Forecast): Operative model runs that bridge the remaining "
        "5-day latency to real-time (using analysis data) and provide the short- to medium-range "
        "weather forecasts."
    ),
    "forecast_offset": (
        "Adjusts the target date. Negative values analyze the past (ERA5 reanalysis), "
        "positive values look into the future (IFS forecast)."
    ),

    # --- Map Tracker ---
    "map_variable": (
        "Mean Temperature: Daily Mean Temperature (TG). The best proxy for the total thermal energy of the day. \n\n"
        "Maximum Temperature: Daily Maximum Temperature (TX). Represents daytime warming (or lack thereof). \n\n"
        "Minimum Temperature: Daily Minimum Temperature (TN). Represents nighttime cooling (or lack thereof)."
    ),
    "map_view_mode": (
        "Daily snapshot colours each grid cell by how unusual that day is. "
        "Persistence duration shows how many consecutive days an extreme has lasted."
    ),
    "map_analysis_level": (
        "The following percentile-based levels can be selected: moderate (P75/25), "
        "strong (P90/10) and extreme (P95/5) levels, as well as all-time records."
    ),
    "map_extremes": "Deactivates warm/cold anomalies (default: all layers are active)",
    "warm_toggle": "Deactivates warm anomalies (default: all layers are active)",
    "cold_toggle": "Deactivates cold anomalies (default: all layers are active)",
    "wsdi_csdi_overlay": (
        "Hatched areas highlight regions experiencing at least 6 consecutive days above the "
        "90th percentile (WSDI) or below the 10th percentile (CSDI)."
    ),
    "persistence_intensity": "Maps warm persistence in red tones and cold persistence in blue tones on the same scale.",
    "mslp_contours": "Mean Sea Level Pressure (hPa)",
    "z500_contours": "500 hPa Geopotential Height (gpm) – indicates upper-level ridges and troughs.",
    "top10_table": "Excludes territories under 3000 km² and countries located completely outside of Europe",

    # --- Point Meteogram ---
    "meteogram_envelope": (
        "Displays the corresponding climate boundaries (percentile-based) behind the temperature curve: "
        "Uses the 75th (warm) and 25th (cold) percentile for moderate, 90th (warm) and 10th (cold) for strong "
        "and 95th (warm) and 5th (cold) for extreme conditions within the reference period. "
        "All-time records are given for the full period (starting 1940) prior to the current year."
    ),
    "meteogram_air_temp_colors": "Colors the space below the curve for cold anomalies and above the curve for warm anomalies.",
    "meteogram_apparent_temp": "Dotted Line: 'Feels-like' temperature, combining 2m air temperature, relative humidity and wind speed.",

    # --- Point Wavogram ---
    "wave_event_type": (
        "Heatwaves: Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) "
        "threshold for at least 3 consecutive days. The wave continues as long as the average TX remains above "
        "this threshold, and terminates immediately if a single day drops below a secondary, lower tolerance threshold.\n\n"
        "Coldwaves: Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) "
        "threshold for at least 3 consecutive days. It continues while the average TN remains below this threshold, "
        "and ends if a single day rises above the upper tolerance limit."
    ),
    "wave_intensity_threshold": (
        "Strong: Calculates waves using the 90th (heat) or 10th (cold) percentile as the main trigger.\n\n"
        "Extreme: Calculates waves using the stricter 95th (heat) or 5th (cold) percentile as the main trigger."
    ),
    "wave_stat_metric": (
        "Cumulative Annual Wave Intensity: Sum of Kyselý wave intensities (Σ TX−P90 per wave day) for all distinct "
        "May–Sep events in a year.\n\n"
        "Maximum Annual Wave Intensity: Intensity of the single strongest wave event of the year.\n\n"
        "Cumulative Heat/Cold Intensity: Σ excess above/below threshold for every day in the season, even without "
        "a 3-day wave (closest to literature \u201cTemperatursumme \u2265 P90\u201d).\n\n"
        "Annual Cycle Frequency: 5-day-smoothed relative frequency of threshold exceedance through the year."
    ),
}
