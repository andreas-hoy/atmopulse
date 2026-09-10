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
        "positive values look into the future (IFS or AIFS forecast)."
    ),
    "forecast_model": (
        "IFS (Physics-based): ECMWF HRES with native diurnal TX/TN extremes.\n\n"
        "AIFS (Machine Learning): ECMWF AIFS with TG, T850 and synoptic fields. "
        "TX/TN and associated wave tracking cannot be natively resolved."
    ),

    # --- Map Tracker ---
    "map_variable": (
        "Mean Temperature (TG): 24-hour mean (0–0 UTC) of air temperature at 2 m.\n\n"
        "Maximum Temperature (TX): Highest daily air temperature (0–0 UTC).\n\n"
        "Minimum Temperature (TN): Lowest daily air temperature (0–0 UTC).\n\n"
        "850 hPa Temperature (T850): Temperature at around 1.5 km height, mapping "
        "lower-tropospheric air masses."
    ),
    "map_view_mode": (
        "Daily snapshot colours each grid cell by how unusual that day is. "
        "Persistence duration shows how many consecutive days an extreme has lasted "
        "(unbroken run back from the map date, up to 100 days)."
    ),
    "map_analysis_level": (
        "How rare a value must be before it counts in the Europe-wide assessment, the country "
        "tables, and the legend highlight. The same threshold is used for persistence duration "
        "(consecutive days at this intensity) and for warm/cold-spell hatching. "
        "Moderate is P75/P25, Strong P90/P10, Extreme P95/P5; All-Time Record is the archive "
        "extreme for that day of year. Daily map colours still show every percentile; hide "
        "layers under Map Extremes."
    ),
    "map_extremes": "Deactivates warm/cold anomalies (default: all layers are active)",
    "warm_toggle": "Deactivates warm anomalies (default: all layers are active)",
    "cold_toggle": "Deactivates cold anomalies (default: all layers are active)",
    "wsdi_csdi_overlay": (
        "Persistence of warm or cold spells at the selected analysis level "
        "(counted back from the map date). Selectable are three durations: "
        "6 days (representing WSDI/CSDI-conditions), 15 and 30 days."
    ),
    "persistence_intensity": (
        "Maps how many consecutive days the selected intensity has lasted, counted "
        "unbroken back from the map date (one day below the threshold resets the "
        "count). Warm persistence in red, cold in blue. The colour scale is ±30 days "
        "so typical land spells stay readable; longer runs still show in the hover "
        "(up to 100 days)."
    ),
    "mslp_contours": (
        "Mean sea-level pressure as 5 hPa isolines. The pattern shows the surface highs and lows "
        "that set the low-level flow and, with it, warm or cold advection."
    ),
    "z500_contours": (
        "Height of the 500 hPa surface (about 5.5 km), as 8 dam isolines (80 geopotential metres — "
        "not hPa). The coarser step keeps the overlay readable on the temperature map; a dedicated "
        "500 hPa chart would typically use 4 dam. Ridges (high) and troughs (low) at this level "
        "steer the weather systems seen in the sea-level pressure field."
    ),
    "mslp_anomaly": (
        "Sea-level pressure minus the 5-day day-of-year mean of the map's selected reference "
        "period (1961–1990 or 1996–2025). Isolines every 2 hPa (tighter than the 5 hPa absolute "
        "field, which otherwise hides typical ±4…±15 hPa departures); "
        "solid = above that baseline, dashed = below. The two maps therefore differ even though "
        "the observed field is the same."
    ),
    "z500_anomaly": (
        "500 hPa height minus the 5-day day-of-year mean of the map's selected reference period. "
        "Isolines every 4 dam (not 8: typical European anomalies are ±8 to ±24 dam and would "
        "nearly vanish at the absolute-field step). Solid = ridge relative to that baseline, "
        "dashed = trough."
    ),
    "top10_table": "Excludes territories under 3000 km² and countries located completely outside of Europe",
    "europe_share_table": (
        "Share of Europe at each cumulative severity level. "
        "Change is in percentage points (1996–2025 minus 1961–1990), not a relative percent. "
        "Each level includes all stricter levels. The highlighted row is the selected analysis level. "
        "All-time records are not epoch-relative, so Change is omitted."
    ),

    # --- Point Meteogram ---
    "meteogram_envelope": (
        "Displays the corresponding climate boundaries (percentile-based) behind the temperature curve: "
        "Uses the 75th (warm) and 25th (cold) percentile for moderate, 90th (warm) and 10th (cold) for strong "
        "and 95th (warm) and 5th (cold) for extreme conditions within the reference period. "
        "All-time records are given for the full period (starting 1940) prior to the current year."
    ),
    "meteo_z500_panel": (
        "500 hPa height anomaly at this grid cell versus the 5-day day-of-year mean of the "
        "selected reference period (dam). Positive (purple fill) is a ridge; negative (blue fill) "
        "is a trough. Same baseline as the temperature envelope above, so the two panels can be "
        "read together as thermal response versus large-scale driver."
    ),
    "meteo_annual_count": (
        "Select between: \"all anomaly days\" (counts every day with a warm or cold anomaly) "
        "and \"spell days\" (counts only days that belong to a period of at least 6 consecutive "
        "warm or cold days)."
    ),

    # --- Point Wavogram ---
    "wave_event_type": (
        "Heatwaves: May–September events using the selected variable (default: daily maximum TX). "
        "Triggered when the value exceeds the local summer (June–August) threshold for at least 3 consecutive days. "
        "The wave continues as long as the average remains above this threshold, and terminates immediately if a "
        "single day drops below a secondary, lower tolerance threshold.\n\n"
        "Coldwaves: November–March events using the selected variable (default: daily minimum TN). "
        "Triggered when the value falls below the local winter (December–February) threshold for at least 3 consecutive days. "
        "It continues while the average remains below this threshold, and ends if a single day rises above the upper tolerance limit."
    ),
    "wave_variable": (
        "Temperature field used for Kyselý detection. Heatwaves default to daily maximum (TX); "
        "coldwaves to daily minimum (TN). Mean temperature (TG) and 850 hPa temperature (T850) "
        "keep the same season and heat/cold direction as the selected event type."
    ),
    "wave_intensity_threshold": (
        "Strong: Calculates waves using the 90th (heat) or 10th (cold) percentile as the main trigger.\n\n"
        "Extreme: Calculates waves using the stricter 95th (heat) or 5th (cold) percentile as the main trigger."
    ),
    "wave_annual_stack": (
        "Cumulative stack from the axis up: the strongest wave, then the remaining waves "
        "(together = all waves), then isolated threshold days (together = all days beyond "
        "the threshold). Intensity is in K; Days is duration."
    ),
    "wave_stack_metric": (
        "Intensity (default): Kyselý excess in K. Days: number of days. "
        "The stack is the same for heat and cold: strongest wave, all waves, then all days "
        "beyond the threshold."
    ),
    "wave_annual_cycle": (
        "5-day-smoothed frequency of days that fall inside a detected Strong or Extreme "
        "Kyselý wave — not isolated hot or cold days."
    ),
    "wave_z500_outline": (
        "Optional purple outline on events whose mean 500 hPa height anomaly over the wave days is at least "
        "+8 dam (heat, Ridge) or −8 dam (cold, Trough) versus the same reference period used for the "
        "temperature thresholds."
    ),
}
