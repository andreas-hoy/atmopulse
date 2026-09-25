"""
AtmoPulse Heavy Plot Rendering — thin re-export barrel (frontend_plots.py)

This module used to hold every Plotly figure builder used by the app
(~3050 lines). It has been split (Phase 0 of the Map Tracker "Wave
tracking" work) into four focused modules:

- frontend_export.py    — output credit, Kaleido, render_press_export,
                           st_plotly_press.
- frontend_maps.py      — synoptic maps, MSLP/H/L, jet, land/sea, swipe,
                           opacity, map hover helpers, get_cached_baseline_map.
- frontend_meteogram.py — get_meteogram_traces, Z500 anomaly traces,
                           build_yearly_extremes_chart,
                           align_yearly_extremes_yranges.
- frontend_wavogram.py  — WAVE_* ridge constants, build_kysely_wave_figs,
                           stack/freq, drill-down minis, union_wave_xrange.

This file exists ONLY so existing `from frontend_plots import …` call sites
in app.py / page_map_tracker.py / page_meteogram.py keep working without
edits beyond what Phase 0 already migrated. New code should import
directly from the specific module above instead of from this barrel.

No behaviour change, no visual change, no cache-key change versus the
pre-split module.
"""

from __future__ import annotations

# --- frontend_export.py ---
from frontend_export import (  # noqa: F401
    PLOTLY_UI_CONFIG,
    _output_credit_text,
    render_press_export,
    st_plotly_press,
)

# --- frontend_maps.py ---
from frontend_maps import (  # noqa: F401
    _MAP_OVERLAY_TOGGLES,
    _MSLP_HL_VERSION,
    _Z500_ANOM_Y_FLOOR,
    _render_synoptic_map,
    build_opacity_slider_map,
    get_cached_baseline_map,
    map_export_title,
    render_swipe_compare_map,
)

# --- frontend_meteogram.py ---
from frontend_meteogram import (  # noqa: F401
    align_yearly_extremes_yranges,
    build_yearly_extremes_chart,
    get_meteogram_traces,
    get_z500_anomaly_traces,
)

# --- frontend_wavogram.py ---
from frontend_wavogram import (  # noqa: F401
    align_wave_stats_yranges,
    build_kysely_wave_figs,
    build_wave_event_mini_fig,
    build_wave_event_z500_mini_fig,
    union_wave_xrange,
    wave_event_z500_window,
)
