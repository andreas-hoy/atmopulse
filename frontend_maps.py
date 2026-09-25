"""
AtmoPulse Synoptic Map Rendering (frontend_maps.py)

Plotly figure builders for the Map Tracker's synoptic baseline maps
(Daily snapshot + Persistence duration + Wave tracking colour fields),
their MSLP/Z500/jet overlays, H/L pressure-center labelling, land/sea
base layer, the opacity cross-fade and swipe-compare layouts, and the
Streamlit chrome around one rendered map (`_render_synoptic_map`).

Extracted from frontend_plots.py (Phase 0 split). Depends on
frontend_export.py for the generic Kaleido/press-export plumbing;
frontend_meteogram.py and frontend_wavogram.py depend on THIS module for
shared map constants (e.g. `_Z500_ANOM_Y_FLOOR`, `_Z500_CONTOUR_*`), never
the other way around, so there is no import cycle. This module never
imports app.py; the few app.py-owned callables a map builder needs
(`get_map_location_labels`, `ref_clim`, `border_trace`, ...) are imported
locally inside the functions that need them.
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from backend_map_locations import EUROPE_BBOX
from backend_analytics import (
    _build_display_mask,
    _map_historical_records,
    _map_var_threshold_arrays,
    _synoptic_lonlat,
    _synoptic_temp_pair,
    _temp_pair_missing,
)
from backend_maps import _synoptic_array, etccdi_doy_365
from backend_io import (
    load_invariant_fields,
    synoptic_clim_mean_display,
)
from config import (
    MAP_VAR_LABELS,
    PERSISTENCE_COLORBAR_DAYS,
    is_daily_map_view,
    is_wave_map_view,
    selected_forecast_model,
    wave_intensity_cap,
)
from atmopulse_theme import (
    ATMOPULSE_BRAND,
    ATMOPULSE_FONTS,
    ATMOPULSE_OVERLAY,
    cold_persistence_colorscale,
    diverging_persistence_colorscale,
    map_contour_label_font,
    map_extremes_colorscale,
    plotly_typography,
    warm_persistence_colorscale,
)
from frontend_export import render_press_export


MAP_VIEW_LON = (EUROPE_BBOX[0], EUROPE_BBOX[2])
MAP_VIEW_LAT = (EUROPE_BBOX[1], EUROPE_BBOX[3])
MAP_CONTOUR_LINE_WIDTH = 1.35
MAP_CONTOUR_LINE_SMOOTHING = 1.15
_MSLP_CONTOUR_SMOOTH_SIGMA = 2.8
_Z500_ANOM_SMOOTH_SIGMA = 1.4
# Isoline intervals. MSLP is hPa; Z500 is dam (decameters of geopotential
# height), never hPa. Absolute MSLP uses the WMO/synoptic 5 hPa step.
# Absolute Z500 uses 8 dam so the overlay stays readable on the percentile
# heatmap (a dedicated 500 hPa chart would typically be 4 dam).
# MSLP anomaly uses 3 hPa. 2 hPa packed the lines too tightly on the
# temperature map; 5 hPa still hides typical ±4…±15 hPa departures.
# Anomaly Z500 uses 4 dam because typical
# European departures (±8…±24 dam) would nearly vanish at 8 dam.
# Absolute MSLP is drawn every 5 hPa from the lowest value on the map
# to the highest. No fixed floor or ceiling.
_MSLP_CONTOUR_INTERVAL = 5.0
_Z500_CONTOUR_START = 500.0
_Z500_CONTOUR_END = 600.0
_Z500_CONTOUR_INTERVAL = 8.0
_MSLP_ANOM_INTERVAL = 3.0
_MSLP_ANOM_SPAN = 40.0
_Z500_ANOM_INTERVAL = 4.0
_Z500_ANOM_SPAN = 40.0
_Z500_ANOM_Y_FLOOR = 12.0
_MAP_OVERLAY_TOGGLES = ("mslp", "z500", "hatching", "mslp_anom", "z500_anom", "jet")

# --- 300 hPa jet overlay (Expert, Map Tracker only) ---
# Three channels, all gated by the single "jet" toggle, NO quiver carpet,
# NO animation, nothing below the jet-core threshold:
#   (A) a FILLED band of |V300| = hypot(u300, v300) above the threshold --
#       WHERE the core is.
#   (B) thin isotachs (40/50/60.../90 m/s) of the same smoothed |V300|
#       field, in the jet's own colour (never Z500 blue/MSLP green/the
#       anomaly purple) -- HOW FAST.
#   (C) a handful of short, fixed-length arrows inside the band only --
#       WHICH WAY. Deliberately not proportional to speed (that reading
#       already lives in the fill + isotachs) and deliberately short and
#       uniform (the previous long, speed-following streamlines read as
#       oversized "pipes" across the North Sea -- this replaces them).
# One colour family, unused elsewhere on this map (MSLP green, Z500 blue,
# both anomaly colours, and -- importantly -- the warm/cold extremes ramp,
# so the jet never reads as a heat signal).
_JET_COLOR = "#0F5C56"  # dark teal (isotachs + low end of the fill ramp)

# (A) Band fill. A light Gaussian smooth (reusing the MSLP helper) keeps
# the band from frayed single-cell noise without blurring the core into a
# wider, weaker feature than it actually is. Isotachs and arrows below
# reuse this exact smoothed field, so all three channels agree.
_JET_FILL_THRESHOLD_MS = 40.0  # ~80 kt; below this: fully transparent, no fill
_JET_FILL_MAX_MS = 90.0        # colour-ramp ceiling (fixed, not per-map dynamic)
_JET_FILL_SMOOTH_SIGMA = 1.3
_JET_FILL_OPACITY = 0.38       # extremes heatmap underneath must still show through
_JET_FILL_COLORSCALE = [
    [0.0, "#0F5C56"],   # dark teal at the threshold
    [0.55, "#2FA79B"],  # mid teal/cyan
    [1.0, "#D9F2A3"],   # pale cyan-yellow-green at the fastest core
]

# (B) Isotachs -- fixed m/s levels via the shared contour helper (same
# rendering family as MSLP/Z500, own colour/levels). Thinner than the
# Z500 line so it doesn't compete with it. `_add_map_contour` only draws
# the levels the field actually reaches, so a calm map with no 70+ m/s
# core simply doesn't get a 70/80/90 line -- nothing is forced.
_JET_ISOTACH_START_MS = _JET_FILL_THRESHOLD_MS  # 40
# Extends past the fill's colour ceiling (_JET_FILL_MAX_MS = 90) so an
# exceptionally strong core still gets labelled 100/110/... lines -- the
# shared contour helper only draws levels the field actually reaches, so
# this is just a bound, never a forced/fabricated line.
_JET_ISOTACH_END_MS = 150.0
_JET_ISOTACH_STEP_MS = 10.0
_JET_ISOTACH_LINE_WIDTH = 1.0  # < MAP_CONTOUR_LINE_WIDTH (Z500's 1.35)

# (C) Short direction arrows -- fixed length (NOT proportional to speed;
# that reading is the fill + isotachs), drawn only where the smoothed
# field is at/above the threshold. A coarse subsample of the core, sorted
# west->east and capped, gives a handful of arrows spread along the jet
# instead of one per grid cell.
_JET_ARROW_SEED_STEP_LAT = 4
_JET_ARROW_SEED_STEP_LON = 7
_JET_ARROW_TRACK_POSITIONS = 13  # along-track (west->east) sample positions
_JET_ARROW_WIDE_ROW_COUNT = 3    # sampled rows crossing the core at one position -> edges
_JET_ARROW_MID_SPAN_DEG = 10.0   # core wider than this also gets an arrow in the middle
_JET_ARROW_SHAFT_DEG = 1.5     # fixed shaft length -- the screenshot's arrows were ~15-25 deg
_JET_ARROWHEAD_LEN_DEG = 0.55
_JET_ARROWHEAD_ANGLE_DEG = 26.0
# White + thin dark halo so arrows read on the teal fill AND on the red/
# blue extremes heatmap outside a fill cell's opacity.
_JET_ARROW_COLOR = "#FFFFFF"
_JET_ARROW_WIDTH = 2.4
_JET_ARROW_HALO_COLOR = "#0B332F"
_JET_ARROW_HALO_WIDTH = 4.2

# Cache-key version for get_cached_baseline_map (see _MSLP_HL_VERSION):
# bump this whenever _add_jet_overlay's drawing changes, so an old cached
# figure (e.g. the previous long-streamline figure) is never served again
# for the "jet" toggle just because date/toggles/source_mtime didn't
# change.
_JET_OVERLAY_VERSION = 7
# Synoptic H/L. Centres are taken from the same smoothed field as the green
# isobars. A glyph needs a real bowl, not a kink: around the point the
# typical rise (a low) or fall (a high) is at least CLOSED_MEDIAN, and even
# the weakest side still clears CLOSED_DEPTH. A broad Iceland low on a later
# forecast day is only about 2 hPa deep on its open side, but several hPa
# deep on average; a contour wiggle over the Mediterranean is the reverse.
# Rings may be clipped by the map edge as long as most of the arc is on the grid.
# The drawn field is smoothed (sigma 2.8). The 1030 hPa high over Britain on
# 21 Sep 2026 is broad: inside 10° the weakest flank is still only about 1 hPa,
# so both IFS and AIFS (same smoothed field) failed the 1.35 hPa gate and the
# old 5 hPa median. A 14° ring with a 1 hPa flank and a 2 hPa median labels
# that closed high. A kink that is deep on one side only still fails the
# weakest-side test.
_MSLP_HL_NEIGHBORHOOD = 9
_MSLP_HL_CLOSED_DEPTH = 1.0
_MSLP_HL_CLOSED_MEDIAN = 2.0
_MSLP_HL_CLOSED_RADII_DEG = (2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0)
_MSLP_HL_CLOSED_SAMPLES = 24
_MSLP_HL_CLOSED_MIN_SAMPLES = 16
_MSLP_HL_MIN_SEP_SAME = 6.5
_MSLP_HL_MIN_SEP_CROSS = 4.0
_MSLP_HL_EDGE_DEG = 0.0
_MSLP_HL_INSET_DEG = 1.8
_MSLP_HL_EDGE_BAND = 5.0
_MSLP_HL_MAX_LABELS = 5
_MSLP_HL_VERSION = 15
MAP_EXTREMES_OPACITY = 0.75
# Spell-hatching (WSDI/CSDI overlay on the daily map): black "x" per cell.
# 0.40 / size 4.5 buried the colour field; keep a readable mark, not a second map.
MAP_HATCH_OPACITY = 0.20
MAP_HATCH_SIZE = 3.6
SYNOPTIC_MAP_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["toImage", "autoScale2d", "select2d", "lasso2d"],
    "scrollZoom": True,
}
# Every other Plotly chart keeps zoom and pan. The camera is the on-screen
# PNG; SVG/PDF under the figure are the press files.


def _fmt_hover_num(v) -> str:
    return f"{float(v):.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_diff(v) -> str:
    return f"{float(v):+.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_year(v) -> str:
    return str(int(float(v))) if np.isfinite(v) and float(v) > 0 else "N/A"

def _fmt_hover_days(v) -> str:
    return str(int(round(float(v)))) if np.isfinite(v) else "N/A"

_MASK_HOVER_CLASS = {
    1: "Record cold",
    2: "Extreme cold",
    3: "Strong cold",
    4: "Moderate cold",
    5: "Moderate warm",
    6: "Strong warm",
    7: "Extreme warm",
    8: "Record warm",
}


def _fmt_hover_class(v) -> str:
    """Legend class for a display-mask code; unclassed cells are Typical."""
    if not np.isfinite(v):
        return "Typical"
    return _MASK_HOVER_CLASS.get(int(v), "Typical")

# Vectorized once at module scope (Schritt B): reused by the customdata
# builders below instead of building a per-cell HTML string grid.
_vfmt_num = np.vectorize(_fmt_hover_num, otypes=[object])
_vfmt_diff = np.vectorize(_fmt_hover_diff, otypes=[object])
_vfmt_year = np.vectorize(_fmt_hover_year, otypes=[object])
_vfmt_days = np.vectorize(_fmt_hover_days, otypes=[object])
_vfmt_class = np.vectorize(_fmt_hover_class, otypes=[object])

def _build_map_customdata(v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, mask=None):
    """customdata for the Daily map heatmap, shape (nlat, nlon, 8).

    Channels: [0] v_curr, [1] v_rec_w, [2] yr_w, [3] diff_w, [4] v_rec_c,
    [5] yr_c, [6] diff_c, [7] legend class (Typical / Moderate warm / …).
    Diff channels stay in the payload for cache compatibility; the hover
    template no longer shows them (they are current minus all-time).
    """
    layers = [
        _vfmt_num(v_curr), _vfmt_num(v_rec_w), _vfmt_year(yr_w), _vfmt_diff(diff_w),
        _vfmt_num(v_rec_c), _vfmt_year(yr_c), _vfmt_diff(diff_c),
    ]
    if mask is None:
        layers.append(np.full(np.asarray(v_curr).shape, "Typical", dtype=object))
    else:
        layers.append(_vfmt_class(mask))
    return np.stack(layers, axis=-1)

def _build_persistence_customdata(warm, cold):
    """customdata for the Persistence map heatmap, shape (nlat, nlon, 2):
    [0] warm days, [1] cold days (integer day counts; spells have no fractions)."""
    return np.stack([_vfmt_days(warm), _vfmt_days(cold)], axis=-1)

def _map_xaxis_kwargs(**extra):
    # Both axes span the whole frame. A scale lock was leaving a white
    # margin inside the map whenever the pixel box was a step off 70:42.
    return dict(
        range=list(MAP_VIEW_LON), autorange=False, showgrid=False, zeroline=False,
        visible=False, constrain=None, domain=[0, 1], **extra,
    )

def _map_yaxis_kwargs(**extra):
    return dict(
        range=list(MAP_VIEW_LAT), autorange=False, showgrid=False, zeroline=False,
        visible=False, scaleanchor=None, scaleratio=None, constrain=None,
        domain=[0, 1], **extra,
    )


def map_export_title(
    map_var_code: str, view_mode: str, epoch_label: str, target_date,
    persist_metric: str | None = None,
) -> str:
    """Heading baked into map SVG/PDF (live maps keep titles outside Plotly)."""
    var = MAP_VAR_LABELS.get(map_var_code, map_var_code)
    daily = is_daily_map_view(view_mode)
    kind = "extremes" if daily else "persistence"
    metric = "" if daily or not persist_metric else f" · {persist_metric}"
    date_s = pd.Timestamp(target_date).strftime("%d.%m.%Y")
    return f"{var} ({map_var_code}) {kind}{metric} | {epoch_label} | {date_s}"


def _add_map_source_label(fig, *, row=None, col=None):
    """Kept for call-site compatibility; maps no longer burn credit onto the data."""
    del fig, row, col
    return



def _drop_plotly_sliders(fig) -> None:
    """Remove a layout slider. ``update_layout(sliders=[])`` leaves it drawn."""
    try:
        fig.layout.sliders = ()
    except Exception:
        pass
    props = getattr(fig.layout, "_props", None)
    if isinstance(props, dict):
        props.pop("sliders", None)


def _detach_opacity_slider(fig) -> str | None:
    """Move the cross-fade control out of the SVG so the map stays 70:42."""
    sliders = list(fig.layout.sliders or ())
    if not sliders:
        return None
    steps = []
    for step in sliders[0].steps:
        args = step["args"] if isinstance(step, dict) else step.args
        payload = args[0]
        if not isinstance(payload, dict):
            payload = dict(payload)
        steps.append({
            "opacity": [float(v) for v in payload.get("opacity", [])],
            "idx": [int(i) for i in args[1]],
        })
    if not steps:
        return None
    label_a, label_b = "Historical", "Recent"
    kept = []
    for ann in list(fig.layout.annotations or ()):
        try:
            xf = float(ann.x)
        except (TypeError, ValueError):
            xf = None
        if xf in (0.148, 0.852):
            text = str(getattr(ann, "text", "") or "")
            if xf < 0.5:
                label_a = text
            else:
                label_b = text
            continue
        kept.append(ann)
    fig.layout.annotations = tuple(kept)
    _drop_plotly_sliders(fig)
    data = json.dumps(steps)
    last = len(steps) - 1
    return (
        f'<div class="ap-opacity-ctrl">'
        f"<span>{html.escape(label_a)}</span>"
        f'<input id="ap-opacity-range" type="range" min="0" max="{last}" value="0" step="1">'
        f"<span>{html.escape(label_b)}</span></div>"
        f"<script>(function(){{"
        f"var steps={data};"
        f"function bind(){{"
        f'var gd=document.querySelector(".st-key-map_opacity .js-plotly-plot");'
        f'var input=document.getElementById("ap-opacity-range");'
        f"if(!gd||!input||!window.Plotly)return false;"
        f"if(input.__apBound)return true;"
        f"input.__apBound=true;"
        f'input.addEventListener("input",function(){{'
        f"var s=steps[Number(input.value)]||steps[0];"
        f"window.Plotly.restyle(gd,{{opacity:s.opacity}},s.idx);"
        f"}});return true;}}"
        f"if(!bind()){{var n=0;var t=setInterval(function(){{if(bind()||++n>25)clearInterval(t);}},200);}}"
        f"}})();</script>"
    )

def _render_synoptic_map(
    fig, title: str, key: str, *, bottom_margin: int = 0,
    export_title: str | None = None, help_text: str | None = None,
    legend_html: str | None = None, banner_html: str | None = None,
    show_chart: bool = True, show_footer: bool = True,
    show_credit: bool = True,
) -> None:
    """Render one synoptic map: Streamlit title above a CSS 70:42 frame.

    Titles stay outside Plotly so the plot area can match EUROPE_BBOX
    exactly. The keyed container is sized by CSS aspect-ratio; Plotly
    fills that box instead of using a fixed pixel height.

    `bottom_margin` reserves room below the map (e.g. for a Plotly
    layout slider) without affecting the default zero-margin callers.
    """
    if show_chart:
        slider_html = _detach_opacity_slider(fig) if key == "map_opacity" else None
        if slider_html:
            bottom_margin = 0
        title_html = f"<p class='atmopulse-map-title'>{title}</p>"
        if help_text:
            with st.container(key=f"{key}_title"):
                st.markdown(title_html, unsafe_allow_html=True, help=help_text, width="content")
        else:
            st.markdown(title_html, unsafe_allow_html=True)
        if banner_html:
            st.markdown(banner_html, unsafe_allow_html=True)
        fig.update_layout(
            **plotly_typography(),
            uirevision="map_sync_state",
            autosize=True,
            height=None,
            title=None,
            xaxis=_map_xaxis_kwargs(),
            yaxis=_map_yaxis_kwargs(),
            margin=dict(t=0, l=0, r=0, b=bottom_margin, pad=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        if slider_html:
            _drop_plotly_sliders(fig)
        with st.container(key=key):
            st.plotly_chart(
                fig,
                use_container_width=True,
                config=SYNOPTIC_MAP_CONFIG,
                key=f"plotly_{key}",
            )
        if slider_html:
            st.html(slider_html, unsafe_allow_javascript=True)
        del show_credit
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)
    if show_footer:
        render_press_export(
            fig, f"map_{key}", heavy=True,
            export_title=export_title or title,
        )

def _mslp_sep2(lon1, lat1, lon2, lat2) -> float:
    """Squared angular distance with a cosine correction for longitude."""
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * np.cos(np.radians(0.5 * (lat1 + lat2)))
    return dlat * dlat + dlon * dlon


def _mslp_inset(lon: float, lat: float) -> tuple[float, float]:
    """Pull a centre slightly inside EUROPE_BBOX so the glyph is not clipped."""
    lon = min(max(lon, EUROPE_BBOX[0] + _MSLP_HL_INSET_DEG), EUROPE_BBOX[2] - _MSLP_HL_INSET_DEG)
    lat = min(max(lat, EUROPE_BBOX[1] + _MSLP_HL_INSET_DEG), EUROPE_BBOX[3] - _MSLP_HL_INSET_DEG)
    return lon, lat


def _mslp_align_grid(lons, lats, z):
    """Return (field, lon, lat) as 2D (nlat, nlon) + 1D axes, or (None, None, None)."""
    field = np.squeeze(np.asarray(getattr(z, "values", z), dtype=float))
    lon = np.squeeze(np.asarray(lons, dtype=float))
    lat = np.squeeze(np.asarray(lats, dtype=float))
    while field.ndim > 2:
        field = field[0]
    if lon.ndim > 1:
        lon = lon[0] if lon.shape[0] <= lon.shape[-1] else lon[:, 0]
        lon = np.squeeze(lon)
    if lat.ndim > 1:
        lat = lat[:, 0] if lat.shape[0] >= lat.shape[-1] else lat[0]
        lat = np.squeeze(lat)
    if field.ndim != 2 or lon.ndim != 1 or lat.ndim != 1:
        return None, None, None
    if field.shape == (lat.size, lon.size):
        return field, lon, lat
    if field.shape == (lon.size, lat.size):
        return field.T, lon, lat
    return None, None, None


def _mslp_near_edge(lon: float, lat: float) -> bool:
    return (
        lon <= EUROPE_BBOX[0] + _MSLP_HL_EDGE_BAND
        or lon >= EUROPE_BBOX[2] - _MSLP_HL_EDGE_BAND
        or lat <= EUROPE_BBOX[1] + _MSLP_HL_EDGE_BAND
        or lat >= EUROPE_BBOX[3] - _MSLP_HL_EDGE_BAND
    )


def _mslp_edge_candidates(lon2, lat2, smooth, finite, radii) -> list[tuple]:
    """Systems whose centre sits on the map rim, if that rim point is still closed."""
    if not np.any(finite):
        return []
    out = []
    work = {
        "L": np.where(finite, smooth, np.inf),
        "H": np.where(finite, smooth, -np.inf),
    }
    for letter, arr in work.items():
        idx = np.unravel_index(np.argmin(arr) if letter == "L" else np.argmax(arr), smooth.shape)
        if not finite[idx]:
            continue
        lon, lat = float(lon2[idx]), float(lat2[idx])
        if not _mslp_near_edge(lon, lat):
            continue
        depth = _mslp_closure_depth(smooth, finite, idx[0], idx[1], letter, radii)
        if depth < _MSLP_HL_CLOSED_DEPTH:
            continue
        lon, lat = _mslp_inset(lon, lat)
        out.append((lon, lat, depth, letter))
    return out


def _mslp_nms(candidates: list[tuple]) -> list[tuple]:
    """Keep the deepest closed centres and drop a second glyph on the same system."""
    ranked = sorted(candidates, key=lambda rec: rec[2], reverse=True)
    kept: list[tuple] = []
    n_h = n_l = 0
    for lon, lat, prom, letter in ranked:
        if letter == "H" and n_h >= _MSLP_HL_MAX_LABELS:
            continue
        if letter == "L" and n_l >= _MSLP_HL_MAX_LABELS:
            continue
        if prom < _MSLP_HL_CLOSED_DEPTH:
            continue
        too_close = False
        for klon, klat, _kp, kletter in kept:
            sep2 = _mslp_sep2(lon, lat, klon, klat)
            limit = _MSLP_HL_MIN_SEP_SAME if letter == kletter else _MSLP_HL_MIN_SEP_CROSS
            if sep2 < limit * limit:
                too_close = True
                break
        if too_close:
            continue
        kept.append((lon, lat, prom, letter))
        if letter == "H":
            n_h += 1
        else:
            n_l += 1
    return kept


def _mslp_closure_depth(smooth, finite, r, c, letter, radii) -> float:
    """Weakest-side closure of the best qualifying ring, in hPa.

    A ring counts when most of it lies on the map, every sample is on the
    correct side of the centre, and the typical sample is a real bowl
    (median), not a one-sided kink. Returns 0 when no ring qualifies.
    """
    centre = float(smooth[r, c])
    nlat, nlon = smooth.shape
    best = 0.0
    for rad in radii:
        signed = []
        for ang in np.linspace(0.0, 2.0 * np.pi, _MSLP_HL_CLOSED_SAMPLES, endpoint=False):
            rr = int(round(r + rad * np.sin(ang)))
            cc = int(round(c + rad * np.cos(ang)))
            if rr < 0 or cc < 0 or rr >= nlat or cc >= nlon or not finite[rr, cc]:
                continue
            value = float(smooth[rr, cc])
            signed.append((value - centre) if letter == "L" else (centre - value))
        if len(signed) < _MSLP_HL_CLOSED_MIN_SAMPLES:
            continue
        if float(np.median(signed)) < _MSLP_HL_CLOSED_MEDIAN:
            continue
        depth = float(min(signed))
        if depth > best:
            best = depth
    return best


def _mslp_closed_records(lon2, lat2, smooth, finite, mask, letter, radii) -> list[tuple]:
    out = []
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        depth = _mslp_closure_depth(smooth, finite, r, c, letter, radii)
        if depth < _MSLP_HL_CLOSED_DEPTH:
            continue
        lon, lat = _mslp_inset(float(lon2[r, c]), float(lat2[r, c]))
        out.append((lon, lat, depth, letter))
    return out


def _mslp_pressure_centers(lons, lats, z) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Local MSLP maxima (H) and minima (L) with a closed-isobar footprint."""
    field, lon, lat = _mslp_align_grid(lons, lats, z)
    if field is None:
        return [], []

    finite = np.isfinite(field)
    if int(finite.sum()) < 50:
        return [], []

    fill = float(np.nanmean(field))
    filled = np.where(finite, field, fill)
    # Same smooth as the drawn isobars, so a glyph sits in a closed contour
    # the map actually shows, not on a kink that smoothing has removed.
    smooth = gaussian_filter(filled, sigma=_MSLP_CONTOUR_SMOOTH_SIGMA, mode="nearest")

    max_in = np.where(finite, smooth, -np.inf)
    min_in = np.where(finite, smooth, np.inf)
    is_max = finite & (smooth == maximum_filter(max_in, size=_MSLP_HL_NEIGHBORHOOD, mode="nearest"))
    is_min = finite & (smooth == minimum_filter(min_in, size=_MSLP_HL_NEIGHBORHOOD, mode="nearest"))

    lon2, lat2 = np.meshgrid(lon, lat)
    in_frame = (
        (lon2 >= EUROPE_BBOX[0] + _MSLP_HL_EDGE_DEG)
        & (lon2 <= EUROPE_BBOX[2] - _MSLP_HL_EDGE_DEG)
        & (lat2 >= EUROPE_BBOX[1] + _MSLP_HL_EDGE_DEG)
        & (lat2 <= EUROPE_BBOX[3] - _MSLP_HL_EDGE_DEG)
    )
    is_max &= in_frame
    is_min &= in_frame
    step = max(
        abs(float(lat[1] - lat[0])) if lat.size > 1 else 0.25,
        abs(float(lon[1] - lon[0])) if lon.size > 1 else 0.25,
        0.05,
    )
    radii = tuple(max(2, int(round(rd / step))) for rd in _MSLP_HL_CLOSED_RADII_DEG)
    candidates = (
        _mslp_closed_records(lon2, lat2, smooth, finite, is_max, "H", radii)
        + _mslp_closed_records(lon2, lat2, smooth, finite, is_min, "L", radii)
        + _mslp_edge_candidates(lon2, lat2, smooth, finite, radii)
    )
    kept = _mslp_nms(candidates)
    highs = [(lon, lat) for lon, lat, _p, letter in kept if letter == "H"]
    lows = [(lon, lat) for lon, lat, _p, letter in kept if letter == "L"]
    return highs, lows


def _add_mslp_hl_labels(fig, lons, lats, z) -> None:
    """Bold H / L on MSLP centres. One dark glyph, no halo (weather-chart style)."""
    highs, lows = _mslp_pressure_centers(lons, lats, z)
    xs = [p[0] for p in highs] + [p[0] for p in lows]
    ys = [p[1] for p in highs] + [p[1] for p in lows]
    texts = (["H"] * len(highs)) + (["L"] * len(lows))
    if not xs:
        return
    fig.add_trace(go.Scatter(
        x=xs, y=ys, text=texts,
        mode="text",
        textposition="middle center",
        cliponaxis=False,
        hoverinfo="skip",
        showlegend=False,
        textfont=dict(
            size=18,
            color=ATMOPULSE_OVERLAY["mslp_contour"],
            family=ATMOPULSE_FONTS["sora_css"],
            weight=700,
        ),
    ))


def _land_fraction_grid(lons, lats):
    """Land fraction 0..1 on the map grid. ERA5 LSM when present, else Natural Earth."""
    try:
        inv = load_invariant_fields()
    except Exception:
        inv = None
    if inv is not None:
        name = next((n for n in ("lsm", "land_sea_mask") if n in inv.variables), None)
        if name is not None:
            da = inv[name]
            for dim in list(da.dims):
                if dim not in ("latitude", "longitude", "lat", "lon") and da.sizes.get(dim, 1) == 1:
                    da = da.isel({dim: 0})
            try:
                da = da.reindex(latitude=lats, longitude=lons, method="nearest")
                z = np.squeeze(np.asarray(da.values, dtype=float))
                if z.shape == (len(lats), len(lons)):
                    return np.clip(z, 0.0, 1.0)
                if z.shape == (len(lons), len(lats)):
                    return np.clip(z.T, 0.0, 1.0)
            except Exception:
                pass
    from backend_map_locations import build_land_sea_grid
    return build_land_sea_grid(np.asarray(lons), np.asarray(lats))


@st.cache_data(show_spinner=False)
def _cached_land_fraction_grid(lons_tuple, lats_tuple):
    return _land_fraction_grid(np.asarray(lons_tuple), np.asarray(lats_tuple))


def _add_land_sea_base(fig, lons, lats) -> None:
    """Quiet land/sea wash under the extremes so uncoloured cells are not stark white."""
    try:
        z = _cached_land_fraction_grid(tuple(np.asarray(lons)), tuple(np.asarray(lats)))
    except Exception:
        return
    if z is None or np.asarray(z).size == 0:
        return
    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=z,
        colorscale=[
            [0.0, ATMOPULSE_OVERLAY["sea"]],
            [1.0, ATMOPULSE_OVERLAY["land"]],
        ],
        zmin=0, zmax=1, showscale=False,
        hoverinfo="skip", opacity=1.0, zsmooth=False,
    ))


def _mslp_contour_span(z) -> tuple[float, float]:
    """5 hPa start/end covering every value on the map, not a fixed window."""
    field = np.squeeze(np.asarray(getattr(z, "values", z), dtype=float))
    finite = field[np.isfinite(field)]
    step = _MSLP_CONTOUR_INTERVAL
    if finite.size == 0:
        return 980.0, 1040.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo > 2000.0:
        lo /= 100.0
        hi /= 100.0
    start = float(np.floor(lo / step) * step)
    end = float(np.ceil(hi / step) * step)
    if end < start:
        return 980.0, 1040.0
    return float(start), float(end)


def _mslp_smooth_field(z, sigma=_MSLP_CONTOUR_SMOOTH_SIGMA):
    """Nan-safe Gaussian smooth so isolines do not fray into tiny closed blobs."""
    field = np.squeeze(np.asarray(getattr(z, "values", z), dtype=float))
    while field.ndim > 2:
        field = field[0]
    finite = np.isfinite(field)
    if not np.any(finite):
        return field
    fill = float(np.nanmean(field))
    smooth = gaussian_filter(np.where(finite, field, fill), sigma=sigma, mode="nearest")
    return np.where(finite, smooth, np.nan)


def _add_map_contour(fig, lons, lats, z, color, start, end, step, *, dash=None, width=None):
    fig.add_trace(go.Contour(
        x=lons, y=lats, z=z,
        autocontour=False,
        colorscale=[[0, color], [1, color]],
        contours=dict(
            start=start, end=end, size=step, showlabels=True,
            labelfont=map_contour_label_font(color=color),
        ),
        contours_coloring="lines", showscale=False,
        line=dict(
            width=MAP_CONTOUR_LINE_WIDTH if width is None else width,
            color=color,
            dash=dash or "solid",
        ),
        line_smoothing=MAP_CONTOUR_LINE_SMOOTHING,
        opacity=1.0, hoverinfo="skip",
        connectgaps=True,
    ))


def _add_anomaly_contours(fig, lons, lats, z, color, interval, span, *, smooth_sigma=None):
    """Signed isolines of a departure field; zero contour omitted.

    Solid = above the selected reference-period DOY mean, dashed = below.
    Levels are exact multiples of ``interval`` through zero: -6, -3, 3, 6 …
    Anchoring the negative run at ``-span`` (for example -40) steps by 3 onto
    -4 and skips -3. The last multiple at or inside ``span`` is the outer end.
    """
    field = np.squeeze(np.asarray(z, dtype=float))
    if smooth_sigma:
        field = _mslp_smooth_field(field, sigma=smooth_sigma)
    if not np.isfinite(field).any():
        return
    step = float(interval)
    bound = float(int(np.floor(float(span) / step)) * step)
    # Plotly's end is exclusive on some builds; a hair past the last level
    # keeps -3 and +bound in the set.
    _add_map_contour(fig, lons, lats, field, color, step, bound + step * 0.01, step)
    _add_map_contour(
        fig, lons, lats, field, color, -bound, -step + step * 0.01, step, dash="dash",
    )

def _add_jet_band_fill(fig, lons, lats, field) -> None:
    """Filled jet band on the full grid from an already-smoothed |V300|
    field -- the main readout of the overlay. Values at or above the core
    threshold only; below it the cell is NaN, i.e. fully transparent, not
    a light-grey wash. One colour ramp (dark teal -> pale cyan/yellow-
    green), half-transparent, no second colorbar, so the temperature
    extremes heatmap underneath still shows through.
    """
    if not np.isfinite(field).any() or float(np.nanmax(field)) < _JET_FILL_THRESHOLD_MS:
        return  # weak-flow summer case: nothing to fill, not a bug
    z = np.where(field >= _JET_FILL_THRESHOLD_MS, field, np.nan)
    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=z,
        colorscale=_JET_FILL_COLORSCALE,
        zmin=_JET_FILL_THRESHOLD_MS, zmax=_JET_FILL_MAX_MS,
        showscale=False, opacity=_JET_FILL_OPACITY, zsmooth=False,
        hoverinfo="skip",
    ))


def _add_jet_isotachs(fig, lons, lats, field) -> None:
    """Thin isotachs of the same smoothed |V300| field used for the fill --
    HOW FAST the core is, as numbers, not just a colour. Fixed m/s levels
    via the shared contour helper (same rendering family as MSLP/Z500,
    own colour/levels, thinner line). `_add_map_contour` only draws the
    levels the field actually reaches, so a map without a 70+ m/s core
    simply has no 70/80/90 line -- nothing is forced or faked.
    """
    if not np.isfinite(field).any() or float(np.nanmax(field)) < _JET_ISOTACH_START_MS:
        return  # weak-flow summer case: nothing to contour, not a bug
    _add_map_contour(
        fig, lons, lats, field, _JET_COLOR,
        _JET_ISOTACH_START_MS, _JET_ISOTACH_END_MS, _JET_ISOTACH_STEP_MS,
        width=_JET_ISOTACH_LINE_WIDTH,
    )


def _select_jet_arrow_seeds(speed_smooth, lons, lats):
    """Along-track seed positions inside the jet core, spread WEST TO EAST.

    Picking a single row per sampled longitude (the previous approach)
    meant a MERIDIONALLY WIDE core only ever got one arrow, always at
    whichever row a flat top-N pick happened to land on -- typically
    leaving one whole edge (e.g. the northern half of a wide band) with no
    arrow at all. A narrow crossing still gets exactly one arrow (its
    middle row). A wider crossing gets one arrow near each edge. A core
    at least _JET_ARROW_MID_SPAN_DEG across also gets one in the middle,
    so a broad jet is not only marked on its flanks.

    Returns (iy, ix) index pairs -- arrows are drawn exactly ON these grid
    points using the field's own u/v there, no interpolation needed.
    """
    lat_idx = np.arange(0, len(lats), _JET_ARROW_SEED_STEP_LAT)
    lon_idx = np.arange(0, len(lons), _JET_ARROW_SEED_STEP_LON)
    if lat_idx.size == 0 or lon_idx.size == 0:
        return []
    sub = speed_smooth[np.ix_(lat_idx, lon_idx)]
    core = np.isfinite(sub) & (sub >= _JET_FILL_THRESHOLD_MS)

    cols_with_rows = [j for j in range(sub.shape[1]) if core[:, j].any()]
    if not cols_with_rows:
        return []

    # Downsample ALONG-TRACK POSITIONS west->east (lon_idx is ascending),
    # independent of how many rows qualify at each one -- so a very wide
    # stretch doesn't crowd out coverage of the rest of the track.
    if len(cols_with_rows) > _JET_ARROW_TRACK_POSITIONS:
        pick = np.linspace(0, len(cols_with_rows) - 1, _JET_ARROW_TRACK_POSITIONS)
        pick = sorted(set(int(round(p)) for p in pick))
        cols_with_rows = [cols_with_rows[p] for p in pick]

    lats_arr = np.asarray(lats, dtype=float)
    seeds = []
    for j in cols_with_rows:
        rows_j = np.flatnonzero(core[:, j])
        if rows_j.size == 0:
            continue
        lat_lo = float(lats_arr[lat_idx[rows_j[0]]])
        lat_hi = float(lats_arr[lat_idx[rows_j[-1]]])
        span = abs(lat_hi - lat_lo)
        if span >= _JET_ARROW_MID_SPAN_DEG:
            chosen = (rows_j[0], rows_j[rows_j.size // 2], rows_j[-1])
        elif rows_j.size >= _JET_ARROW_WIDE_ROW_COUNT:
            chosen = (rows_j[0], rows_j[-1])
        else:
            chosen = (rows_j[rows_j.size // 2],)
        seen = set()
        for i in chosen:
            ii = int(i)
            if ii in seen:
                continue
            seen.add(ii)
            seeds.append((int(lat_idx[ii]), int(lon_idx[j])))
    return seeds


def _add_jet_arrows(fig, lons, lats, u_full, v_full, speed_smooth) -> None:
    """A handful of short, FIXED-length direction arrows drawn only inside
    the jet band -- WHICH WAY, not how fast (that reading is the fill +
    isotachs). Deliberately not proportional to speed and deliberately
    short/uniform: the previous speed-following streamlines produced
    15-25 degree "pipes" across the North Sea; these are a constant
    ~1.5 degree shaft with a small head, same family as the old arrow
    quiver's head construction but capped to a handful of seeds and only
    where the smoothed field is at/above the threshold. White + thin dark
    halo so they read on the teal fill and on the red/blue extremes
    heatmap alike. Drawn last so they sit on top of the fill/isotachs.
    """
    seeds = _select_jet_arrow_seeds(speed_smooth, lons, lats)
    if not seeds:
        return  # weak-flow summer case: no core, no arrows -- not a bug
    lons_arr, lats_arr = np.asarray(lons), np.asarray(lats)
    head_rad = np.radians(_JET_ARROWHEAD_ANGLE_DEG)

    xs, ys = [], []
    for iy, ix in seeds:
        u, v = float(u_full[iy, ix]), float(v_full[iy, ix])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        speed = float(np.hypot(u, v))
        if speed <= 0:
            continue
        x0, y0 = float(lons_arr[ix]), float(lats_arr[iy])
        dx, dy = (u / speed) * _JET_ARROW_SHAFT_DEG, (v / speed) * _JET_ARROW_SHAFT_DEG
        xa, ya = x0 - dx / 2.0, y0 - dy / 2.0
        xb, yb = x0 + dx / 2.0, y0 + dy / 2.0
        angle = np.arctan2(dy, dx)
        hx1 = xb - _JET_ARROWHEAD_LEN_DEG * np.cos(angle - head_rad)
        hy1 = yb - _JET_ARROWHEAD_LEN_DEG * np.sin(angle - head_rad)
        hx2 = xb - _JET_ARROWHEAD_LEN_DEG * np.cos(angle + head_rad)
        hy2 = yb - _JET_ARROWHEAD_LEN_DEG * np.sin(angle + head_rad)
        xs.extend([xa, xb, np.nan, xb, hx1, np.nan, xb, hx2, np.nan])
        ys.extend([ya, yb, np.nan, yb, hy1, np.nan, yb, hy2, np.nan])
    if not xs:
        return

    # Dark halo first (wider, underneath), then the bright arrows on top.
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_JET_ARROW_HALO_COLOR, width=_JET_ARROW_HALO_WIDTH),
        hoverinfo="skip", showlegend=False, name="300 hPa jet (halo)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_JET_ARROW_COLOR, width=_JET_ARROW_WIDTH),
        hoverinfo="skip", showlegend=False, name="300 hPa jet",
    ))


def _add_jet_overlay(fig, lons, lats, map_phys_data) -> None:
    """300 hPa jet overlay, three channels sharing one smoothed |V300|
    field: a filled band above the jet-core threshold (WHERE), thin
    isotachs at fixed m/s levels (HOW FAST), and a handful of short,
    fixed-length direction arrows inside the band (WHICH WAY). No quiver
    carpet, no full wind field, no animation -- everywhere below the
    threshold is untouched (fully transparent). Missing u300/v300 or a
    grid mismatch skip the whole overlay silently, the same as any other
    optional synoptic layer on this map.
    """
    if "u300" not in map_phys_data or "v300" not in map_phys_data:
        return  # missing field: skip the overlay, never crash the map
    u_full = np.squeeze(_synoptic_array(map_phys_data["u300"]))
    v_full = np.squeeze(_synoptic_array(map_phys_data["v300"]))
    if u_full is None or v_full is None or u_full.shape != v_full.shape:
        return
    if u_full.shape != (len(lats), len(lons)):
        return  # grid mismatch with the drawn heatmap -- skip rather than guess

    u_full = np.asarray(u_full, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    speed_full = np.hypot(u_full, v_full)
    speed_smooth = _mslp_smooth_field(speed_full, sigma=_JET_FILL_SMOOTH_SIGMA)

    _add_jet_band_fill(fig, lons, lats, speed_smooth)
    _add_jet_isotachs(fig, lons, lats, speed_smooth)
    _add_jet_arrows(fig, lons, lats, u_full, v_full, speed_smooth)


def _add_wave_heatmap(fig, lons, lats, wave_intensity, is_warm, loc_labels) -> None:
    """Wave tracking's colour layer: Kyselý intensity-to-date, sequential
    warm/cold palette, fixed colorbar (0..cap; values above the cap
    saturate to the top colour, hover always shows the true number).
    Uncoloured cells (0 or NaN) are simply not inside a wave right now.

    `wave_intensity` is the caller's already-grid-aligned 2D array (K·days,
    0 = not in a wave) or None when the archive doesn't cover this date /
    the store hasn't been built yet — in which case nothing is drawn (an
    empty map with just the land/sea base + borders), never a crash.
    """
    if wave_intensity is None:
        return
    z = np.asarray(wave_intensity, dtype=np.float64)
    if z.shape != (len(lats), len(lons)):
        return  # grid mismatch -- skip rather than guess
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)  # 0/NaN cells stay uncoloured

    cap = wave_intensity_cap(is_warm)
    colorscale = warm_persistence_colorscale() if is_warm else cold_persistence_colorscale()
    unit = "K·days"
    fig.add_trace(go.Heatmap(
        x=lons, y=lats, z=z, text=loc_labels,
        colorscale=colorscale, showscale=True,
        zmin=0, zmax=cap, zsmooth=False,
        hoverongaps=False,
        colorbar=dict(title=unit, len=0.6, y=0.5, thickness=15),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Latitude: %{y:.2f}, Longitude: %{x:.2f}<br><br>"
            f"Kyselý intensity: %{{z:.1f}} {unit}"
            "<extra></extra>"
        ),
    ))


def _add_location_marker(fig, marker_lat, marker_lon, marker_name) -> None:
    """Small circle + name for the location last chosen on the Point
    Meteogram/Wavogram (persisted in a non-widget session key by app.py) —
    never a hardcoded fallback point; drawn only once a location has
    actually been picked."""
    if marker_lat is None or marker_lon is None:
        return
    fig.add_trace(go.Scatter(
        x=[float(marker_lon)], y=[float(marker_lat)], mode="markers+text",
        marker=dict(symbol="circle", size=9, color="rgba(0,0,0,0.85)", line=dict(color="white", width=1.5)),
        text=[str(marker_name or "")], textposition="top center",
        textfont=dict(size=11, color=ATMOPULSE_BRAND["text_on_light"]),
        hoverinfo="skip", showlegend=False, name="marker",
    ))


def build_baseline_map(
    ref_data, map_phys_data, target_date, t_warm, t_cold, toggles, view_mode, persist_metric, top10_threshold,
    baseline_type="A", map_var="TG", anchor_date=None, *, full_width=False,
    border_trace=None, get_map_location_labels=None, get_persistence_arrays=None,
    syn_clim=None, forecast_model=None,
    wave_is_warm=True, wave_intensity=None,
    marker_lat=None, marker_lon=None, marker_name=None,
):
    """
    `border_trace` / `get_map_location_labels` / `get_persistence_arrays` are
    app.py-owned (a module-level trace + cached loaders). They are passed in
    explicitly by the caller instead of imported here: Streamlit runs app.py
    as the entrypoint script (not as an importable module named "app"), so a
    `from app import ...` inside this function would re-execute the whole
    script from scratch on every call and crash on already-instantiated
    widgets.
    """
    if forecast_model is None:
        forecast_model = selected_forecast_model()
    if ref_data is None or map_phys_data is None: 
        return go.Figure()
        
    suffix, doy = ("A" if baseline_type == "A" else "B"), etccdi_doy_365(target_date)
    tx_curr, tn_curr = _synoptic_temp_pair(map_phys_data)
    lons, lats = _synoptic_lonlat(map_phys_data)
    if lons is None or lats is None:
        return go.Figure()
    if map_var != "T850" and _temp_pair_missing(map_var, tx_curr, tn_curr, map_phys_data):
        return go.Figure()
    
    # Align the climatology grid to the live/archive field's actual lat/lon
    # coordinates (nearest-neighbor) instead of assuming positional array
    # equality. A silent grid mismatch here (e.g. different longitude
    # convention or half-cell offset between climatology and live sources)
    # is what produces isolated coastal boundary artifacts.
    daily_ref = ref_data.sel(dayofyear=doy).reindex(
        latitude=lats, longitude=lons, method="nearest"
    )
    shape = tx_curr.shape if tx_curr is not None else (len(lats), len(lons))
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in daily_ref.variables: 
            return daily_ref[var_key].values
        return np.full(shape, fallback)

    v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5 = _map_var_threshold_arrays(
        map_var, map_phys_data, safe_get, suffix, tx_curr, tn_curr,
    )
    if v_curr is None:
        return go.Figure()

    # All-time records: archive only, strictly before the viewed year (keeps the
    # previous record visible when the current year breaks it).
    v_rec_w, v_rec_c, yr_w, yr_c = _map_historical_records(ref_data, doy, target_date, map_var, v_curr.shape, anchor_date)

    fig = go.Figure()
    _add_land_sea_base(fig, lons, lats)
    loc_labels = get_map_location_labels(tuple(lons), tuple(lats))
    
    # Pure NumPy math without string loops (100x faster, minimal RAM footprint)
    diff_w = v_curr - v_rec_w
    diff_c = v_curr - v_rec_c
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

        daily_customdata = _build_map_customdata(
            v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, mask=mask,
        )
        daily_hovertemplate = (
            "<b>%{text}</b><br>"
            "Latitude: %{y:.2f}, Longitude: %{x:.2f}<br><br>"
            + var_label + ": %{customdata[0]} °C<br>"
            "Class: %{customdata[7]}<br>"
            "All-Time Warm: %{customdata[1]} °C (Year %{customdata[2]})<br>"
            "All-Time Cold: %{customdata[4]} °C (Year %{customdata[5]})"
            "<extra></extra>"
        )

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=mask, text=loc_labels, customdata=daily_customdata,
            colorscale=colorscale, showscale=False,
            opacity=MAP_EXTREMES_OPACITY, zmin=1, zmax=8, zsmooth=False,
            hoverongaps=True,
            hovertemplate=daily_hovertemplate,
        ))
        
        if toggles.get("hatching", False):
            # Same model as this panel: live AIFS uses the AIFS series,
            # an archive day uses ERA5 even when the sidebar still says AIFS.
            anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
            try:
                streaks = get_persistence_arrays(
                    target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str,
                    forecast_model=forecast_model,
                )
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
                
                min_days = int(toggles.get("spell_days", 6) or 6)
                hatch_mask = (streaks[h_idx] >= min_days) | (streaks[c_idx] >= min_days)
                if np.any(hatch_mask):
                    h_lons, h_lats = lon_grid[hatch_mask], lat_grid[hatch_mask]
                    fig.add_trace(go.Scatter(
                        x=h_lons, y=h_lats, mode="markers",
                        marker=dict(
                            symbol="x",
                            color=f"rgba(0,0,0,{MAP_HATCH_OPACITY})",
                            size=MAP_HATCH_SIZE,
                        ),
                        hoverinfo="skip", showlegend=False,
                    ))
                    
    elif is_wave_map_view(view_mode):
        _add_wave_heatmap(fig, lons, lats, wave_intensity, wave_is_warm, loc_labels)

    else:
        anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
        streaks = get_persistence_arrays(
            target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str,
            forecast_model=forecast_model,
        )
        if streaks is not None:
            mapping = {
                "Moderate": (0, 4),
                "Strong": (1, 5),
                "Extreme": (2, 6),
                "All-Time Record": (3, 7),
            }
            w_idx, c_idx = mapping.get(persist_metric, (1, 5))
            max_days = PERSISTENCE_COLORBAR_DAYS
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
            persist_customdata = _build_persistence_customdata(warm, cold)
            persist_hovertemplate = (
                "<b>%{text}</b><br>"
                "Latitude: %{y:.2f}, Longitude: %{x:.2f}<br><br>"
                "Warm: %{customdata[0]} days<br>"
                "Cold: %{customdata[1]} days"
                "<extra></extra>"
            )
            fig.add_trace(go.Heatmap(
                x=lons, y=lats, z=z, text=loc_labels, customdata=persist_customdata,
                zmin=-max_days, zmax=max_days,
                colorscale=diverging_persistence_colorscale(), showscale=True, opacity=0.9, zsmooth=False,
                colorbar=persist_colorbar,
                hovertemplate=persist_hovertemplate,
            ))
            
    if border_trace is not None:
        traces = border_trace if isinstance(border_trace, (list, tuple)) else (border_trace,)
        for tr in traces:
            if tr is not None:
                fig.add_trace(tr)
    if toggles.get("mslp", False) and "mslp" in map_phys_data:
        mslp_z = np.squeeze(_synoptic_array(map_phys_data["mslp"]))
        mslp_draw = _mslp_smooth_field(mslp_z)
        _c0, _c1 = _mslp_contour_span(mslp_z)
        _add_map_contour(
            fig, lons, lats, mslp_draw, ATMOPULSE_OVERLAY['mslp_contour'],
            _c0, _c1, _MSLP_CONTOUR_INTERVAL,
        )
        _add_mslp_hl_labels(fig, lons, lats, mslp_z)
    if toggles.get("z500", False) and "z500" in map_phys_data:
        _add_map_contour(
            fig, lons, lats, np.squeeze(_synoptic_array(map_phys_data["z500"])),
            ATMOPULSE_OVERLAY['z500_contour'],
            _Z500_CONTOUR_START, _Z500_CONTOUR_END, _Z500_CONTOUR_INTERVAL,
        )

    want_mslp_anom = bool(toggles.get("mslp_anom")) and "mslp" in map_phys_data
    want_z500_anom = bool(toggles.get("z500_anom")) and "z500" in map_phys_data
    if syn_clim is not None and (want_mslp_anom or want_z500_anom):
        if want_mslp_anom:
            clim_mslp = synoptic_clim_mean_display(
                syn_clim, "mslp", suffix, doy, lats, lons,
            )
            live_mslp = np.squeeze(_synoptic_array(map_phys_data["mslp"]))
            if clim_mslp is not None and live_mslp.shape == clim_mslp.shape:
                _add_anomaly_contours(
                    fig, lons, lats, live_mslp - clim_mslp,
                    ATMOPULSE_OVERLAY["mslp_anom_contour"],
                    _MSLP_ANOM_INTERVAL, _MSLP_ANOM_SPAN,
                    smooth_sigma=_MSLP_CONTOUR_SMOOTH_SIGMA,
                )
        if want_z500_anom:
            clim_z500 = synoptic_clim_mean_display(
                syn_clim, "z500", suffix, doy, lats, lons,
            )
            live_z500 = np.squeeze(_synoptic_array(map_phys_data["z500"]))
            if clim_z500 is not None and live_z500.shape == clim_z500.shape:
                _add_anomaly_contours(
                    fig, lons, lats, live_z500 - clim_z500,
                    ATMOPULSE_OVERLAY["z500_anom_contour"],
                    _Z500_ANOM_INTERVAL, _Z500_ANOM_SPAN,
                    smooth_sigma=_Z500_ANOM_SMOOTH_SIGMA,
                )

    if toggles.get("jet", False):
        # Daily snapshot AND persistence view both allowed: wind is a daily
        # synoptic field, independent of the persistence-count heatmap.
        _add_jet_overlay(fig, lons, lats, map_phys_data)

    _add_location_marker(fig, marker_lat, marker_lon, marker_name)
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
        margin=dict(t=0, l=0, r=0, b=0, pad=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# Version args are hashed on purpose. A leading underscore would drop them
# from the Streamlit cache key and keep serving a figure built by older code.
@st.cache_data(show_spinner=False, max_entries=32)
def get_cached_baseline_map(
    date_str, baseline_type, map_var, view_mode, persist_metric, top10_threshold,
    t_warm_items, t_cold_items, active_toggles, source_mtime, forecast_model,
    full_width=False, anchor_date_str=None, spell_days=6,
    anom_mslp_hpa=_MSLP_ANOM_INTERVAL, hl_version=_MSLP_HL_VERSION,
    jet_version=_JET_OVERLAY_VERSION, credit_version=4, hatch_version=4,
    mslp_span_version=5, map_frame_version=2,
    wave_is_warm=True, wave_level="Strong", wave_store_mtime=0.0,
    marker_lat=None, marker_lon=None, marker_name=None,
    *, _ref_data, _map_phys_data, _syn_clim=None, _wave_intensity=None,
):
    """Schritt C: @st.cache_data front door for build_baseline_map.

    build_baseline_map itself is intentionally NOT decorated 1:1 — two of its
    args (`ref_data`: an xarray Dataset, `map_phys_data`: a dict of full-grid
    ndarrays) aren't cheap/reliable Streamlit cache keys, and the other two
    (`border_trace`: a go.Scatter, `get_map_location_labels`/
    `get_persistence_arrays`: callables) are flatly forbidden as cache keys
    (no functions, no Plotly traces in a cache key). This wrapper reduces the
    call to ONLY hashable primitives — `date_str`/`anchor_date_str` as ISO
    strings (not Timestamps), `t_warm`/`t_cold` as `tuple(sorted(d.items()))`,
    the mslp/z500/hatching/mslp_anom/z500_anom/jet toggle dict as a
    `frozenset` of the active names,
    `source_mtime` (see `backend_io.synoptic_source_mtime`) so a fresh
    forecast download busts this cache even for the same date/toggles, and
    `forecast_model` is both the cache key and the model passed into
    persistence. An archive panel stays on ERA5 when the sidebar is AIFS.

    `_ref_data`/`_map_phys_data`/`_syn_clim` are keyword-only with a leading
    underscore (Streamlit's convention for cache-key-EXCLUDED args) — identical
    role to every other `_ref_data`/`_map_phys_data` pair already used throughout
    this codebase (`compute_map_footprint`, `calculate_top10`). `_syn_clim` is
    the MSLP/Z500 DOY-mean climatology used only for expert anomaly isolines.

    The three callables (`border_trace` source + the two location/persistence
    loaders) are resolved with a LOCAL import of `page_map_tracker` inside
    this function body, never at module scope: `page_map_tracker.py` imports
    `build_baseline_map`/this wrapper from `frontend_plots.py`, so a
    module-level import here would be circular. They are page_map_tracker's
    own `@st.cache_resource`/`@st.cache_data`-decorated singletons — calling
    them again here is a cheap cache hit, not a rebuild, and (per the
    existing house rule) this is deliberately NOT `from app import ...`:
    Streamlit runs app.py as the entrypoint script, not as an importable
    module, so that would re-execute the whole script from scratch.
    """
    from page_map_tracker import get_europe_borders_trace, get_map_location_labels, get_persistence_arrays

    target_date = pd.Timestamp(date_str)
    anchor_date = pd.Timestamp(anchor_date_str) if anchor_date_str else None
    t_warm = dict(t_warm_items)
    t_cold = dict(t_cold_items)
    toggles = {name: (name in active_toggles) for name in _MAP_OVERLAY_TOGGLES}
    toggles["spell_days"] = int(spell_days)

    return build_baseline_map(
        _ref_data, _map_phys_data, target_date, t_warm, t_cold, toggles, view_mode,
        persist_metric, top10_threshold, baseline_type, map_var,
        anchor_date=anchor_date, full_width=full_width,
        border_trace=get_europe_borders_trace(),
        get_map_location_labels=get_map_location_labels,
        get_persistence_arrays=get_persistence_arrays,
        syn_clim=_syn_clim,
        forecast_model=forecast_model,
        wave_is_warm=wave_is_warm,
        wave_intensity=_wave_intensity,
        marker_lat=marker_lat, marker_lon=marker_lon, marker_name=marker_name,
    )


def _clone_map_trace(trace):
    data = trace.to_plotly_json()
    constructors = {"heatmap": go.Heatmap, "contour": go.Contour, "scatter": go.Scatter}
    return constructors.get(data.get("type", "scatter"), go.Scatter)(data)


def _plotly_compare_slider(*, active: int, steps: list) -> dict:
    # The bar sits between the two end labels, with no tick marks under it.
    return dict(
        active=active,
        x=0.16, y=0, len=0.68,
        pad=dict(t=12, b=0),
        currentvalue=dict(visible=False),
        ticklen=0,
        tickcolor="rgba(0,0,0,0)",
        steps=steps,
    )


def build_opacity_slider_map(
    fig_a, fig_b,
    label_a="Historical",
    label_b="Recent",
    n_steps=11,
):
    """Cross-fade two already-built maps with a Plotly layout slider.

    ``label_a`` and ``label_b`` sit directly against the two ends of the bar.
    The bar has no tick marks and no value line above it.
    ``method="restyle"`` runs entirely in the browser, so dragging the
    handle does not trigger a Streamlit rerun or rebuild the NetCDF layers.
    """
    if not fig_a.data and not fig_b.data:
        return go.Figure()

    fig = go.Figure(layout=(fig_a.layout if fig_a.data else fig_b.layout))
    traces_a = [_clone_map_trace(t) for t in fig_a.data]
    traces_b = [_clone_map_trace(t) for t in fig_b.data]
    for t in traces_b:
        if getattr(t, "showscale", None):
            t.showscale = False
    for t in traces_a:
        fig.add_trace(t)
    for t in traces_b:
        fig.add_trace(t)

    n_a = len(traces_a)
    idx_a, idx_b = list(range(0, n_a)), list(range(n_a, n_a + len(traces_b)))
    base_op_a = [1.0 if t.opacity is None else float(t.opacity) for t in traces_a]
    base_op_b = [1.0 if t.opacity is None else float(t.opacity) for t in traces_b]

    steps = []
    last = max(n_steps - 1, 1)
    for i in range(n_steps):
        frac = i / last
        opac_a = [round(o * (1.0 - frac), 4) for o in base_op_a]
        opac_b = [round(o * frac, 4) for o in base_op_b]
        steps.append(dict(
            method="restyle",
            args=[{"opacity": opac_a + opac_b}, idx_a + idx_b],
            label="",
        ))

    fig.update_layout(
        sliders=[_plotly_compare_slider(active=0, steps=steps)],
        margin=dict(t=0, l=0, r=0, b=39, pad=0),
    )
    end_font = dict(size=13, color="#31333F")
    fig.add_annotation(
        text=label_a, xref="paper", yref="paper",
        x=0.148, y=0, xanchor="right", yanchor="top", yshift=-12,
        showarrow=False, font=end_font,
    )
    fig.add_annotation(
        text=label_b, xref="paper", yref="paper",
        x=0.852, y=0, xanchor="left", yanchor="top", yshift=-12,
        showarrow=False, font=end_font,
    )
    for t, op in zip(fig.data[:n_a], base_op_a):
        t.opacity = op
    for t in fig.data[n_a:]:
        t.opacity = 0.0
    return fig


def render_swipe_compare_map(
    fig_a, fig_b, *, title: str, help_text: str,
    export_title_a: str | None = None, export_title_b: str | None = None,
    legend_html: str | None = None,
    banner_html: str | None = None,
) -> None:
    """One map, two Plotly.js instances drawn inside a single self-owned
    iframe, with the top layer clipped by a CSS custom property.

    This does NOT rely on Streamlit's outer DOM/class structure at all
    (nesting `st.container(key=...)` blocks proved unreliable for absolute
    overlay positioning across Streamlit versions). Both figures, the
    slider, and the drag handler live in one `components.html` document
    that we fully control, so the swipe updates a CSS variable only —
    no Streamlit rerun, no Plotly redraw, zero added latency.
    """
    fig_bottom = go.Figure(fig_a)
    fig_top = go.Figure(fig_b)
    fig_top.update_layout(annotations=[])
    for fig in (fig_bottom, fig_top):
        for t in fig.data:
            if getattr(t, "showscale", None):
                t.showscale = False
            if getattr(t, "type", None) == "heatmap":
                t.zsmooth = False

    common_layout = dict(
        **plotly_typography(),
        uirevision="map_sync_state",
        autosize=True,
        title=None,
        margin=dict(t=0, l=0, r=0, b=0, pad=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # Box-zoom drag would otherwise fight the swipe-divider drag on the
        # very same pointer gesture (Plotly reads the mousedown too and
        # draws a zoom rectangle), silently re-ranging the axes away from
        # EUROPE_BBOX. Modebar zoom buttons (fixed-step) still work.
        dragmode=False,
    )
    fig_bottom.update_layout(**common_layout, xaxis=_map_xaxis_kwargs(), yaxis=_map_yaxis_kwargs())
    fig_top.update_layout(**common_layout, xaxis=_map_xaxis_kwargs(), yaxis=_map_yaxis_kwargs())

    with st.container(key="swipe_map_title"):
        st.markdown(
            f"<p class='atmopulse-map-title'>{title}</p>",
            unsafe_allow_html=True,
            help=help_text,
            width="content",
        )
    if banner_html:
        st.markdown(banner_html, unsafe_allow_html=True)

    primary = ATMOPULSE_BRAND["primary"]
    json_a = fig_bottom.to_json()
    json_b = fig_top.to_json()

    html = f"""
<div id="swipe-stack">
  <div id="swipe-a"></div>
  <div id="swipe-b"></div>
  <div id="swipe-line"></div>
  <div id="swipe-tooltip"></div>
</div>
<div class="atmopulse-swipe-ctrl">
  <span>1961&ndash;1990</span>
  <input id="atmopulse-swipe-range" type="range" min="0" max="100" value="50" step="0.1">
  <span>1996&ndash;2025</span>
</div>
<style>
  html, body {{ margin: 0; overflow: hidden; background: transparent; }}
  #swipe-stack {{
    position: relative;
    width: 100%;
    aspect-ratio: 70 / 42;
    height: auto;
    overflow: hidden;
    --swipe: 50%;
  }}
  #swipe-a, #swipe-b {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  #swipe-b {{ z-index: 2; clip-path: inset(0 0 0 var(--swipe)); }}
  /* The native Plotly hover box lives inside the SAME clipped div as the
     map, so any tooltip triggered close to the swipe line gets sliced
     off by clip-path along with the hidden half of the data. It's hidden
     here and replaced by #swipe-tooltip below, a sibling that is never
     clipped and is JS-clamped to stay fully inside the visible box. */
  #swipe-a .hoverlayer, #swipe-b .hoverlayer {{ display: none !important; }}
  #swipe-line {{
    position: absolute; top: 0; bottom: 0; left: var(--swipe);
    width: 2px; background: {primary}; z-index: 3; pointer-events: none;
  }}
  #swipe-tooltip {{
    position: absolute; z-index: 4; pointer-events: none; display: none;
    max-width: 260px; padding: 6px 9px; border-radius: 6px;
    background: rgba(20, 24, 30, 0.92); color: #fff;
    font-family: {ATMOPULSE_FONTS["sora_css"]}; font-size: 12px; line-height: 1.35;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }}
  .atmopulse-swipe-ctrl {{
    display: flex; align-items: center; gap: 10px;
    font-family: {ATMOPULSE_FONTS["outfit_css"]};
    font-size: 13px; color: #000; padding: 6px 2px 0 2px;
  }}
  .atmopulse-swipe-ctrl span {{ white-space: nowrap; }}
  .atmopulse-swipe-ctrl input[type=range] {{ flex: 1; accent-color: {primary}; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.0.0/plotly.min.js"></script>
<script>
(function() {{
  var SLIDER_H = 46;

  var figA = {json_a};
  var figB = {json_b};
  var cfgA = {{displayModeBar: "hover", displaylogo: false, responsive: false,
               modeBarButtonsToRemove: ["toImage", "autoScale2d", "select2d", "lasso2d"], scrollZoom: false}};
  var cfgB = {{displayModeBar: false, responsive: false, scrollZoom: false}};

  var stack = document.getElementById("swipe-stack");
  var slider = document.getElementById("atmopulse-swipe-range");
  var plotted = false;

  function apply(pct) {{
    var v = Math.max(0, Math.min(100, Number(pct)));
    stack.style.setProperty("--swipe", v + "%");
    if (String(slider.value) !== String(v)) slider.value = v;
  }}

  // Size the stack to EUROPE_BBOX (70°×42°) from the iframe WIDTH, then
  // ask Streamlit to grow the iframe. A short default iframe + overflow
  // hidden previously cropped latitude (scaleanchor 1:1) — SVG/PDF export
  // never hit that path, so they looked complete.
  function layout() {{
    var w = window.innerWidth || stack.clientWidth || document.documentElement.clientWidth || 0;
    var mapH = w > 0 ? Math.round(w * 42 / 70) : 0;
    if (mapH > 0) {{
      stack.style.aspectRatio = "auto";
      stack.style.width = "100%";
      stack.style.height = mapH + "px";
    }}
    var ctrl = document.querySelector(".atmopulse-swipe-ctrl");
    var ctrlH = ctrl ? (ctrl.getBoundingClientRect().height || SLIDER_H) : SLIDER_H;
    // Trim only the empty band under the slider. A short first measure
    // must not be published — that collapsed the map to a thin strip.
    // The iframe starts at 900px, so the map is laid out full size first.
    if (mapH >= 280) {{
      var total = Math.ceil(mapH + ctrlH + 8);
      window.parent.postMessage({{
        isStreamlitMessage: true,
        type: "streamlit:setFrameHeight",
        height: total
      }}, "*");
    }}
    if (plotted && w > 0 && mapH > 0) {{
      Plotly.relayout("swipe-a", {{autosize: false, width: w, height: mapH}});
      Plotly.relayout("swipe-b", {{autosize: false, width: w, height: mapH}});
    }}
  }}

  layout();
  Promise.all([
    Plotly.newPlot("swipe-a", figA.data, figA.layout, cfgA),
    Plotly.newPlot("swipe-b", figB.data, figB.layout, cfgB),
  ]).then(function(gds) {{
    plotted = true;
    layout();
    setupTooltip(gds[0]);
    setupTooltip(gds[1]);
  }});

  // Custom hover readout: Plotly still fires "plotly_hover" even though
  // its own hover box is hidden via CSS above, so we render the same
  // hovertext ourselves into #swipe-tooltip — a sibling of swipe-a/b that
  // clip-path never touches — and clamp it inside the stack so it can
  // never poke out past the visible edge either.
  var tooltip = document.getElementById("swipe-tooltip");

  // Schritt B: the heatmaps no longer carry a pre-rendered HTML string in
  // `hovertext` (that was the ~10 MB-per-map hover grid). Hover content now
  // comes from each trace's own `hovertemplate` + `text`/`x`/`y`/`customdata`,
  // so the swipe overlay's custom tooltip (native Plotly hover box is CSS-
  // hidden here, see #swipe-a/#swipe-b .hoverlayer above) has to render that
  // template itself instead of just reading `pt.hovertext`. Only the small
  // set of placeholders actually used by this codebase's map hovertemplates
  // is supported (%{{text}}, %{{x:.2f}}, %{{y:.2f}}, %{{customdata[i]}},
  // <extra></extra>) — sufficient since customdata values here are already
  // pre-formatted short strings (no further numeric formatting needed).
  function renderHoverTemplate(pt) {{
    var tmpl = pt.data && pt.data.hovertemplate;
    if (!tmpl) return pt.hovertext != null ? pt.hovertext : pt.text;
    var out = tmpl.split("<extra></extra>").join("").split("<extra>%{{fullData.name}}</extra>").join("");
    out = out.split("%{{x:.2f}}").join(Number(pt.x).toFixed(2));
    out = out.split("%{{y:.2f}}").join(Number(pt.y).toFixed(2));
    out = out.split("%{{text}}").join(pt.text != null ? pt.text : "");
    if (Array.isArray(pt.customdata)) {{
      for (var i = 0; i < pt.customdata.length; i++) {{
        out = out.split("%{{customdata[" + i + "]}}").join(pt.customdata[i]);
      }}
    }}
    return out;
  }}

  function showTooltip(gd, ev) {{
    var pt = ev.points && ev.points[0];
    if (!pt) return;
    var text = renderHoverTemplate(pt);
    if (text == null) return;
    tooltip.innerHTML = String(text);
    tooltip.style.display = "block";
    var stackRect = stack.getBoundingClientRect();
    var mouseX = (ev.event ? ev.event.clientX : stackRect.left) - stackRect.left;
    var mouseY = (ev.event ? ev.event.clientY : stackRect.top) - stackRect.top;
    var tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    var x = mouseX + 14, y = mouseY + 14;
    if (x + tw > stackRect.width) x = mouseX - tw - 14;
    if (y + th > stackRect.height) y = mouseY - th - 14;
    x = Math.max(2, Math.min(x, stackRect.width - tw - 2));
    y = Math.max(2, Math.min(y, stackRect.height - th - 2));
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }}
  function setupTooltip(gd) {{
    if (!gd) return;
    gd.on("plotly_hover", function(ev) {{ showTooltip(gd, ev); }});
    gd.on("plotly_unhover", function() {{ tooltip.style.display = "none"; }});
  }}

  slider.addEventListener("input", function() {{ apply(slider.value); }});

  var dragging = false;
  function pctFromEvent(ev) {{
    var r = stack.getBoundingClientRect();
    return ((ev.clientX - r.left) / r.width) * 100;
  }}
  stack.addEventListener("pointerdown", function(ev) {{
    if (ev.target.closest(".modebar")) return;
    dragging = true;
    stack.setPointerCapture(ev.pointerId);
    apply(pctFromEvent(ev));
  }});
  stack.addEventListener("pointermove", function(ev) {{ if (dragging) apply(pctFromEvent(ev)); }});
  stack.addEventListener("pointerup", function() {{ dragging = false; }});
  stack.addEventListener("pointercancel", function() {{ dragging = false; }});

  window.addEventListener("resize", layout);
  if (window.ResizeObserver) {{
    new ResizeObserver(layout).observe(document.documentElement);
  }}
  [50, 150, 300, 600, 1000].forEach(function(t) {{ setTimeout(layout, t); }});
}})();
</script>
"""
    with st.container(key="swipe_map_frame"):
        components.html(html, height=900, scrolling=False)
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        render_press_export(
            fig_bottom, "swipe_historical", heavy=True,
            export_title=export_title_a,
        )
    with e2:
        render_press_export(
            fig_top, "swipe_recent", heavy=True,
            export_title=export_title_b,
        )

