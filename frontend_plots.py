"""
AtmoPulse Heavy Plot Rendering (frontend_plots.py)

Plotly figure builders extracted from app.py: the synoptic baseline maps
(daily snapshot + persistence view), the opacity-slider cross-fade, the
Point Meteogram trace stack, and the yearly extremes bar chart.

`get_map_location_labels`, `get_persistence_arrays`, `border_trace`,
`ref_clim`, and `_load_point_archive_series` remain owned by app.py (they
depend on the live/archive dataset loaders and the module-level climatology
handle defined there). They are imported locally, inside the functions that
need them, so this module never imports app.py at load time — avoiding a
circular import while app.py imports these plot builders from here.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter

from backend_map_locations import EUROPE_BBOX
from backend_analytics import (
    _build_display_mask,
    _map_historical_records,
    _map_var_threshold_arrays,
    _synoptic_lonlat,
    _synoptic_temp_pair,
    _yyyymmdd_dot_date_arr,
)
from backend_maps import _synoptic_array, etccdi_doy_365
from backend_io import (
    load_invariant_fields,
    point_clim_ladder,
    synoptic_clim_mean_display,
    synoptic_clim_point_doy,
)
from config import epoch_period_label
from atmopulse_theme import (
    ATMOPULSE_BRAND,
    ATMOPULSE_COLD,
    ATMOPULSE_FONTS,
    ATMOPULSE_OVERLAY,
    ATMOPULSE_WARM,
    cold_rgba,
    diverging_persistence_colorscale,
    map_contour_label_font,
    map_extremes_colorscale,
    plotly_title_font,
    plotly_typography,
    warm_rgba,
)
from config import MAP_VAR_LABELS, PERSISTENCE_COLORBAR_DAYS, is_aifs_model, is_daily_map_view, meteo_var_code, selected_forecast_model, show_expert

# --- Point Wavogram ridge-plot layout (tune wave shape / break aesthetics
# here — drawing-only, moved from backend_waves.py so that module stays
# compute-only). See `build_kysely_wave_figs` below. ---
WAVE_RIDGE_SPLINE_PTS = 100      # Smoothness of the ridge curve
WAVE_RIDGE_SKEW_FACTOR = 2.5     # Interior lean vs. intensity (0 = symmetric)
WAVE_RIDGE_HEIGHT_SCALE = 20.0   # Vertical extent in axis-year units (÷ intensity)
WAVE_BREAK_TAIL_LEN = 0.5        # X-axis length of the post-peak decay tail (days)
WAVE_BREAK_TAIL_STEPS = 30       # Number of points along the decay tail
WAVE_BREAK_CTRL_DX = -0.6        # Bezier ctrl offset in days; negative = left bulge under the peak
WAVE_BREAK_CTRL_Y = 0.6          # Bezier ctrl-y (× peak height); higher = rounder fall
WAVE_LINE_WIDTH = 1.0
WAVE_Z500_LINE_WIDTH = 1.7  # Expert outline when wave-mean Z500 anomaly supports the event
WAVE_FILL_ALPHA_BASE = 0.55      # Gradient fill opacity at ridge base
WAVE_FILL_ALPHA_PEAK = 0.88      # Gradient fill opacity at ridge peak
WAVE_LINE_ALPHA = 0.92
WAVE_INTENSITY_CAP_TX = 100.0    # Intensity (K·days) mapped to full warm colour
WAVE_INTENSITY_CAP_TN = 200.0    # Intensity (K·days) mapped to full cold colour


def _wave_hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _wave_lerp_hex(c0: str, c1: str, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    r0, g0, b0 = _wave_hex_to_rgb(c0)
    r1, g1, b1 = _wave_hex_to_rgb(c1)
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def _wave_ridge_colors(parameter: str, is_warm: bool, norm_val: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Wavogram fill: warm p75→p95 / cold p25→p5. Stops at Extreme so the
    peak stroke does not collide with the purple Z500 ridge/trough outline
    (map/meteogram Record pink and indigo are unchanged)."""
    if is_warm:
        base = _wave_hex_to_rgb(ATMOPULSE_WARM["p75"])
        peak = _wave_lerp_hex(ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["p95"], norm_val)
    else:
        base = _wave_hex_to_rgb(ATMOPULSE_COLD["p25"])
        peak = _wave_lerp_hex(ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["p5"], norm_val)
    return base, peak


def _wave_break_tail(x_end: float, y_peak: float) -> tuple[np.ndarray, np.ndarray]:
    """Visual-only post-peak closure (Bezier). `x_end` is the last event day.

    Control-x is an absolute day offset (not scaled by tail length), so a
    short tail still gets a leftward curl instead of a vertical drop.
    """
    if y_peak <= 0:
        return np.array([x_end]), np.array([0.0])

    t = np.linspace(0, 1, WAVE_BREAK_TAIL_STEPS)
    x_ctrl = x_end + WAVE_BREAK_CTRL_DX
    y_ctrl = y_peak * WAVE_BREAK_CTRL_Y
    x_out = x_end + WAVE_BREAK_TAIL_LEN

    x_break = (1 - t) ** 2 * x_end + 2 * (1 - t) * t * x_ctrl + t ** 2 * x_out
    y_break = (1 - t) ** 2 * y_peak + 2 * (1 - t) * t * y_ctrl
    return x_break, y_break


def _wave_ridge_x(x_true: np.ndarray, y_fine: np.ndarray) -> np.ndarray:
    """Lean the rising body right; keep the peak on the last real day."""
    x_true = np.asarray(x_true, dtype=float)
    y_fine = np.asarray(y_fine, dtype=float)
    if x_true.size < 2:
        return x_true.copy()
    y_max = float(np.max(y_fine)) if y_fine.size else 0.0
    span = x_true[-1] - x_true[0]
    if y_max <= 0 or span <= 0:
        return x_true.copy()
    t = np.clip((x_true - x_true[0]) / span, 0.0, 1.0)
    x_drawn = x_true + (y_fine / y_max) * WAVE_RIDGE_SKEW_FACTOR * (1.0 - t)
    x_drawn[-1] = x_true[-1]
    return x_drawn

MAP_VIEW_LON = (EUROPE_BBOX[0], EUROPE_BBOX[2])
MAP_VIEW_LAT = (EUROPE_BBOX[1], EUROPE_BBOX[3])
MAP_CONTOUR_LINE_WIDTH = 1.35
MAP_CONTOUR_LINE_SMOOTHING = 1.15
_MSLP_CONTOUR_SMOOTH_SIGMA = 2.8
_Z500_ANOM_SMOOTH_SIGMA = 1.4
# Isoline intervals. MSLP is hPa; Z500 is dam (decameters of geopotential
# height), never hPa. Absolute MSLP uses the WMO/synoptic 5 hPa step.
# Absolute Z500 uses 8 dam so the overlay stays readable on the percentile
# heatmap (a dedicated 500 hPa chart would typically be 4 dam). Anomaly
# MSLP uses 2 hPa (composite/reanalysis practice; 5 hPa hides typical
# ±4…±15 hPa departures). Anomaly Z500 uses 4 dam because typical
# European departures (±8…±24 dam) would nearly vanish at 8 dam.
_MSLP_CONTOUR_START = 980.0
_MSLP_CONTOUR_END = 1040.0
_MSLP_CONTOUR_INTERVAL = 5.0
_Z500_CONTOUR_START = 500.0
_Z500_CONTOUR_END = 600.0
_Z500_CONTOUR_INTERVAL = 8.0
_MSLP_ANOM_INTERVAL = 2.0
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
_JET_ISOTACH_END_MS = _JET_FILL_MAX_MS          # 90
_JET_ISOTACH_STEP_MS = 10.0
_JET_ISOTACH_LINE_WIDTH = 1.0  # < MAP_CONTOUR_LINE_WIDTH (Z500's 1.35)

# (C) Short direction arrows -- fixed length (NOT proportional to speed;
# that reading is the fill + isotachs), drawn only where the smoothed
# field is at/above the threshold. A coarse subsample of the core, sorted
# west->east and capped, gives a handful of arrows spread along the jet
# instead of one per grid cell.
_JET_ARROW_SEED_STEP_LAT = 6
_JET_ARROW_SEED_STEP_LON = 10
_JET_ARROW_MAX_COUNT = 10
_JET_ARROW_SHAFT_DEG = 1.5     # fixed shaft length -- the screenshot's arrows were ~15-25 deg
_JET_ARROWHEAD_LEN_DEG = 0.45
_JET_ARROWHEAD_ANGLE_DEG = 26.0
# White + thin dark halo so arrows read on the teal fill AND on the red/
# blue extremes heatmap outside a fill cell's opacity.
_JET_ARROW_COLOR = "#FFFFFF"
_JET_ARROW_WIDTH = 1.8
_JET_ARROW_HALO_COLOR = "#0B332F"
_JET_ARROW_HALO_WIDTH = 3.2

# Cache-key version for get_cached_baseline_map (see _MSLP_HL_VERSION):
# bump this whenever _add_jet_overlay's drawing changes, so an old cached
# figure (e.g. the previous long-streamline figure) is never served again
# for the "jet" toggle just because date/toggles/source_mtime didn't
# change.
_JET_OVERLAY_VERSION = 4
# Synoptic H/L: only label centres with a closed-system footprint
# (neighbourhood span of at least one 5 hPa isoline) and keep glyphs apart.
_MSLP_HL_SMOOTH_SIGMA = 2.5
_MSLP_HL_NEIGHBORHOOD = 21
_MSLP_HL_PROM_WINDOW = 45      # background mean; smaller window underestimates broad highs
_MSLP_HL_RANGE_WINDOW = 41     # ~10°: closed or strongly curved isolines nearby
_MSLP_HL_MIN_PROMINENCE = 2.0  # hPa vs regional mean
_MSLP_HL_MIN_RANGE = 4.0       # hPa span; drops flat saddles without isoline structure
_MSLP_HL_MIN_SEP_SAME = 14.0
_MSLP_HL_MIN_SEP_CROSS = 7.0
_MSLP_HL_EDGE_DEG = 0.0
_MSLP_HL_INSET_DEG = 1.8
_MSLP_HL_EDGE_BAND = 5.0
_MSLP_HL_EDGE_PROM = 4.0
_MSLP_HL_EDGE_RANGE = 8.0
_MSLP_HL_MAX_LABELS = 2
_MSLP_HL_STEER_FRAC = 0.60
_MSLP_HL_VERSION = 7
MAP_EXTREMES_OPACITY = 0.75
SYNOPTIC_MAP_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["autoScale2d", "select2d", "lasso2d"],
    "scrollZoom": True,
}


def _fmt_hover_num(v) -> str:
    return f"{float(v):.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_diff(v) -> str:
    return f"{float(v):+.1f}" if np.isfinite(v) else "N/A"

def _fmt_hover_year(v) -> str:
    return str(int(float(v))) if np.isfinite(v) and float(v) > 0 else "N/A"

def _fmt_hover_days(v) -> str:
    return str(int(round(float(v)))) if np.isfinite(v) else "N/A"

# Vectorized once at module scope (Schritt B): reused by the customdata
# builders below instead of building a per-cell HTML string grid.
_vfmt_num = np.vectorize(_fmt_hover_num, otypes=[object])
_vfmt_diff = np.vectorize(_fmt_hover_diff, otypes=[object])
_vfmt_year = np.vectorize(_fmt_hover_year, otypes=[object])
_vfmt_days = np.vectorize(_fmt_hover_days, otypes=[object])

def _build_standard_hovertext(labels, lat2d, lon2d, v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, var_label):
    """DEAD CODE as of Schritt B (kept for reference / potential rollback).

    This used to be handed to go.Heatmap as `hovertext=`, producing one full
    HTML string per grid cell (~47k cells -> ~10 MB of duplicated markup:
    "<b>", "Latitude:", city names, etc. repeated per cell). It is no longer
    called anywhere; `_build_map_customdata` / `_build_persistence_customdata`
    + a static `hovertemplate` replace it for both the Daily and Persistence
    map heatmaps.
    """
    fmt1 = np.vectorize(_fmt_hover_num)
    fmtd = np.vectorize(_fmt_hover_diff)
    fyr = np.vectorize(_fmt_hover_year)
    loc = labels.astype(str)
    return (
        "<b>" + loc + "</b><br>"
        "Latitude: " + fmt1(lat2d) + ", Longitude: " + fmt1(lon2d) + "<br><br>"
        + var_label + ": " + fmt1(v_curr) + " °C<br>"
        "All-Time Warm: " + fmt1(v_rec_w) + " °C (Year " + fyr(yr_w) + "; " + fmtd(diff_w) + " °C diff)<br>"
        "All-Time Cold: " + fmt1(v_rec_c) + " °C (Year " + fyr(yr_c) + "; " + fmtd(diff_c) + " °C diff)"
    )

def _build_map_customdata(v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c):
    """customdata for the Daily map heatmap, shape (nlat, nlon, 7).

    Channels: [0] v_curr, [1] v_rec_w, [2] yr_w, [3] diff_w, [4] v_rec_c,
    [5] yr_c, [6] diff_c.

    Values are pre-formatted short strings (object dtype), not raw
    float32/int16, for one hard reason: NaN cannot round-trip through
    Plotly's JSON payload as a *number* (`fig.to_json()` / the Streamlit
    component transport both need valid JSON, and bare `NaN`/`null` then
    format as "NaN"/"0.0" via `%{customdata[i]:.1f}`, not "N/A"). Formatting
    once here with the existing `_fmt_hover_*` helpers (same rules as the old
    hovertext path: 1 decimal, signed diff, integer year, "N/A" on non-finite)
    keeps the hover content byte-identical while cutting per-cell payload
    from a multi-line HTML block to 7 short tokens.
    """
    return np.stack([
        _vfmt_num(v_curr), _vfmt_num(v_rec_w), _vfmt_year(yr_w), _vfmt_diff(diff_w),
        _vfmt_num(v_rec_c), _vfmt_year(yr_c), _vfmt_diff(diff_c),
    ], axis=-1)

def _build_persistence_customdata(warm, cold):
    """customdata for the Persistence map heatmap, shape (nlat, nlon, 2):
    [0] warm days, [1] cold days (integer day counts; spells have no fractions)."""
    return np.stack([_vfmt_days(warm), _vfmt_days(cold)], axis=-1)

def _map_xaxis_kwargs(**extra):
    # constrain="domain" on X (not Y): if the box is a pixel off the 70:42
    # geographic ratio, leftover width letterboxes left/right instead of
    # cropping southern Europe off the latitude range.
    return dict(
        range=list(MAP_VIEW_LON), autorange=False, showgrid=False, zeroline=False,
        visible=False, constrain="domain", constraintoward="center", **extra,
    )

def _map_yaxis_kwargs(**extra):
    return dict(
        range=list(MAP_VIEW_LAT), autorange=False, showgrid=False, zeroline=False,
        scaleanchor="x", scaleratio=1, visible=False, **extra,
    )

def _add_map_source_label(fig, *, row=None, col=None):
    """Anchor source tag to the map axes domain (not full figure paper)."""
    ann = dict(
        text=f"Data: ERA5/{'AIFS' if is_aifs_model() else 'IFS'}",
        xref="x domain", yref="y domain",
        x=0.99, y=0.0,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color=ATMOPULSE_BRAND["text_on_light"], family=ATMOPULSE_FONTS["sora_css"]),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(200,200,200,0.55)",
        borderwidth=1,
        borderpad=3,
    )
    if row is None and col is None:
        fig.add_annotation(**ann)
    else:
        fig.add_annotation(**ann, row=row, col=col)


def _attach_press_csv(fig: go.Figure, csv_text: str | None) -> None:
    if not csv_text:
        return
    meta = fig.layout.meta
    payload = dict(meta) if isinstance(meta, dict) else {"press_csv": csv_text}
    if isinstance(meta, dict):
        payload["press_csv"] = csv_text
    fig.update_layout(meta=payload)


def _press_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "chart"


def _fig_fingerprint(fig: go.Figure) -> str:
    parts = [
        str(fig.layout.title.text if fig.layout.title else ""),
        str(fig.layout.height),
        str(len(fig.data)),
    ]
    for t in fig.data:
        parts.append(str(getattr(t, "type", "")))
        for attr in ("z", "y", "x"):
            val = getattr(t, attr, None)
            if val is None:
                continue
            arr = np.asarray(val)
            parts.append(attr + str(arr.shape))
            if arr.size:
                parts.append(str(arr.flat[0]))
                parts.append(str(arr.flat[-1]))
            break
    return hashlib.md5("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()


@st.cache_resource(show_spinner=False)
def _press_image_store() -> dict:
    return {}


def _kaleido_image(fig: go.Figure, fmt: str) -> bytes:
    store = _press_image_store()
    key = (_fig_fingerprint(fig), fmt)
    cached = store.get(key)
    if cached is not None:
        return cached
    height = int(fig.layout.height) if fig.layout.height else 840
    width = 1400 if fig.layout.height is None else 1200
    blob = fig.to_image(format=fmt, width=width, height=height)
    store[key] = blob
    return blob


def _press_vector_slot(fig: go.Figure, stem: str, fmt: str, label: str, mime: str) -> None:
    """One SVG or PDF control: click to render, then a download button.

    Kaleido only runs after the user asks for that format, and only for that
    format. A figure change (new fingerprint) clears the prepared state so
    date/toggle reruns do not silently re-export.
    """
    fp = _fig_fingerprint(fig)
    ready_key = f"press_ready_{fmt}_{stem}"
    fp_key = f"press_fp_{fmt}_{stem}"
    if st.session_state.get(fp_key) != fp:
        st.session_state[ready_key] = False
        st.session_state[fp_key] = fp
    if not st.session_state.get(ready_key):
        if st.button(label, key=f"press_go_{fmt}_{stem}"):
            st.session_state[ready_key] = True
            st.rerun()
        return
    try:
        with st.spinner(f"Rendering {label}…"):
            blob = _kaleido_image(fig, fmt)
    except Exception as exc:
        st.caption(f"{label} unavailable")
        st.caption(f"Vector export needs the kaleido package ({exc})")
        return
    st.download_button(
        label, data=blob,
        file_name=f"AtmoPulse_{stem}.{fmt}",
        mime=mime, key=f"press_{stem}_{fmt}",
    )


def render_press_export(
    fig: go.Figure, stem: str, csv_text: str | None = None, *, heavy: bool = False,
) -> None:
    """Compact SVG / PDF / CSV row. Vector files are built only on request.

    ``heavy`` is kept so existing map call-sites do not break; it is no longer
    a separate code path (maps used to bundle SVG+PDF behind one Prepare click).
    """
    del heavy
    if csv_text is None:
        meta = fig.layout.meta
        if isinstance(meta, dict):
            csv_text = meta.get("press_csv")
    stem = _press_stem(stem)
    n = 3 if csv_text else 2
    with st.container(key=f"press-row-{stem}"):
        cols = st.columns([1] * n + [10], gap="small")
        with cols[0]:
            _press_vector_slot(fig, stem, "svg", "SVG", "image/svg+xml")
        with cols[1]:
            _press_vector_slot(fig, stem, "pdf", "PDF", "application/pdf")
        if csv_text:
            with cols[2]:
                st.download_button(
                    "CSV", data=csv_text.encode("utf-8"),
                    file_name=f"AtmoPulse_{stem}.csv", mime="text/csv",
                    key=f"press_{stem}_csv",
                )


def st_plotly_press(
    fig: go.Figure, stem: str, csv_text: str | None = None,
    *, on_select: str | None = None, selection_mode=("points",), key: str | None = None,
    **chart_kw,
):
    """`st.plotly_chart` + the compact SVG/PDF/CSV row.

    `on_select`/`selection_mode`/`key` are optional and default to the old,
    unset behaviour (no selection wiring, no return value used) — existing
    call sites (maps, meteogram, wavogram) are unaffected. Pass
    `on_select="rerun"` to get the click-selection event back for a
    click-to-drill-down UI; the return value is Streamlit's selection dict
    (or None when selection isn't enabled/nothing is selected).
    """
    plot_kwargs = dict(use_container_width=True, **chart_kw)
    if on_select is not None:
        plot_kwargs["on_select"] = on_select
        plot_kwargs["selection_mode"] = selection_mode
    if key is not None:
        plot_kwargs["key"] = key
    event = st.plotly_chart(fig, **plot_kwargs)
    render_press_export(fig, stem, csv_text=csv_text)
    return event


def _render_synoptic_map(fig, title: str, key: str, *, bottom_margin: int = 0) -> None:
    """Render one synoptic map: Streamlit title above a CSS 70:42 frame.

    Titles stay outside Plotly so the plot area can match EUROPE_BBOX
    exactly. The keyed container is sized by CSS aspect-ratio; Plotly
    fills that box instead of using a fixed pixel height.

    `bottom_margin` reserves room below the map (e.g. for a Plotly
    layout slider) without affecting the default zero-margin callers.
    """
    st.markdown(f"<p class='atmopulse-map-title'>{title}</p>", unsafe_allow_html=True)
    fig.update_layout(
        **plotly_typography(),
        uirevision="map_sync_state",
        autosize=True,
        height=None,
        title=None,
        margin=dict(t=0, l=0, r=0, b=bottom_margin),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    with st.container(key=key):
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=SYNOPTIC_MAP_CONFIG,
            key=f"plotly_{key}",
        )
    render_press_export(fig, f"map_{key}", heavy=True)

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


def _mslp_collect_centres(lon2, lat2, smooth, local_mean, mask, letter: str) -> list[tuple]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return []
    out = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        lon, lat = _mslp_inset(float(lon2[r, c]), float(lat2[r, c]))
        prom = abs(float(smooth[r, c] - local_mean[r, c]))
        out.append((lon, lat, prom, letter))
    return out


def _mslp_near_edge(lon: float, lat: float) -> bool:
    return (
        lon <= EUROPE_BBOX[0] + _MSLP_HL_EDGE_BAND
        or lon >= EUROPE_BBOX[2] - _MSLP_HL_EDGE_BAND
        or lat <= EUROPE_BBOX[1] + _MSLP_HL_EDGE_BAND
        or lat >= EUROPE_BBOX[3] - _MSLP_HL_EDGE_BAND
    )


def _mslp_edge_candidates(lon2, lat2, smooth, local_mean, local_range, finite) -> list[tuple]:
    """Island-low / Azores-high style systems whose centre sits on the map rim."""
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
        prom = abs(float(smooth[idx] - local_mean[idx]))
        if prom < _MSLP_HL_EDGE_PROM or float(local_range[idx]) < _MSLP_HL_EDGE_RANGE:
            continue
        lon, lat = _mslp_inset(lon, lat)
        out.append((lon, lat, prom, letter))
    return out


def _mslp_nms(candidates: list[tuple]) -> list[tuple]:
    """Keep only steering-scale centres (strongest H/L, drop weak companions)."""
    ranked = sorted(candidates, key=lambda rec: rec[2], reverse=True)
    best = {"H": 0.0, "L": 0.0}
    for _lon, _lat, prom, letter in ranked:
        if prom > best[letter]:
            best[letter] = prom
    kept: list[tuple] = []
    n_h = n_l = 0
    for lon, lat, prom, letter in ranked:
        if letter == "H" and n_h >= _MSLP_HL_MAX_LABELS:
            continue
        if letter == "L" and n_l >= _MSLP_HL_MAX_LABELS:
            continue
        floor = max(_MSLP_HL_MIN_PROMINENCE, _MSLP_HL_STEER_FRAC * best[letter])
        if prom < floor:
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
    smooth = gaussian_filter(filled, sigma=_MSLP_HL_SMOOTH_SIGMA, mode="nearest")
    local_mean = uniform_filter(filled, size=_MSLP_HL_PROM_WINDOW, mode="nearest")
    local_range = (
        maximum_filter(smooth, size=_MSLP_HL_RANGE_WINDOW, mode="nearest")
        - minimum_filter(smooth, size=_MSLP_HL_RANGE_WINDOW, mode="nearest")
    )

    max_in = np.where(finite, smooth, -np.inf)
    min_in = np.where(finite, smooth, np.inf)
    is_max = finite & (smooth == maximum_filter(max_in, size=_MSLP_HL_NEIGHBORHOOD, mode="nearest"))
    is_min = finite & (smooth == minimum_filter(min_in, size=_MSLP_HL_NEIGHBORHOOD, mode="nearest"))
    is_max &= (smooth - local_mean) >= _MSLP_HL_MIN_PROMINENCE
    is_min &= (local_mean - smooth) >= _MSLP_HL_MIN_PROMINENCE
    is_max &= local_range >= _MSLP_HL_MIN_RANGE
    is_min &= local_range >= _MSLP_HL_MIN_RANGE

    lon2, lat2 = np.meshgrid(lon, lat)
    in_frame = (
        (lon2 >= EUROPE_BBOX[0] + _MSLP_HL_EDGE_DEG)
        & (lon2 <= EUROPE_BBOX[2] - _MSLP_HL_EDGE_DEG)
        & (lat2 >= EUROPE_BBOX[1] + _MSLP_HL_EDGE_DEG)
        & (lat2 <= EUROPE_BBOX[3] - _MSLP_HL_EDGE_DEG)
    )
    is_max &= in_frame
    is_min &= in_frame

    candidates = (
        _mslp_collect_centres(lon2, lat2, smooth, local_mean, is_max, "H")
        + _mslp_collect_centres(lon2, lat2, smooth, local_mean, is_min, "L")
        + _mslp_edge_candidates(lon2, lat2, smooth, local_mean, local_range, finite)
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
    """
    field = np.squeeze(np.asarray(z, dtype=float))
    if smooth_sigma:
        field = _mslp_smooth_field(field, sigma=smooth_sigma)
    if not np.isfinite(field).any():
        return
    _add_map_contour(fig, lons, lats, field, color, interval, span, interval)
    _add_map_contour(
        fig, lons, lats, field, color, -span, -interval, interval, dash="dash",
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
    """A handful of grid-index seeds inside the jet core, spread WEST TO
    EAST along the band (not whatever order a coarse-grid scan happens to
    return, which can bunch several picks into one neighbourhood if the
    core is a thin strip within a couple of subsampled rows). Returns
    (iy, ix) index pairs -- arrows are drawn exactly ON these grid points
    using the field's own u/v there, no interpolation needed.
    """
    lat_idx = np.arange(0, len(lats), _JET_ARROW_SEED_STEP_LAT)
    lon_idx = np.arange(0, len(lons), _JET_ARROW_SEED_STEP_LON)
    if lat_idx.size == 0 or lon_idx.size == 0:
        return []
    sub = speed_smooth[np.ix_(lat_idx, lon_idx)]
    core = np.argwhere(np.isfinite(sub) & (sub >= _JET_FILL_THRESHOLD_MS))
    if core.size == 0:
        return []
    lon_of = np.asarray(lons)[lon_idx[core[:, 1]]]
    core = core[np.argsort(lon_of)]  # west -> east, so picks spread along-track
    if len(core) > _JET_ARROW_MAX_COUNT:
        pick = np.linspace(0, len(core) - 1, _JET_ARROW_MAX_COUNT).round().astype(int)
        core = core[pick]
    return [(int(lat_idx[iy]), int(lon_idx[ix])) for iy, ix in core]


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


def build_baseline_map(
    ref_data, map_phys_data, target_date, t_warm, t_cold, toggles, view_mode, persist_metric, top10_threshold,
    baseline_type="A", map_var="TG", anchor_date=None, *, full_width=False,
    border_trace=None, get_map_location_labels=None, get_persistence_arrays=None,
    syn_clim=None,
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
    if ref_data is None or map_phys_data is None: 
        return go.Figure()
        
    suffix, doy = ("A" if baseline_type == "A" else "B"), etccdi_doy_365(target_date)
    tx_curr, tn_curr = _synoptic_temp_pair(map_phys_data)
    lons, lats = _synoptic_lonlat(map_phys_data)
    if lons is None or lats is None:
        return go.Figure()
    if map_var != "T850" and (tx_curr is None or tn_curr is None):
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

        daily_customdata = _build_map_customdata(v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c)
        daily_hovertemplate = (
            "<b>%{text}</b><br>"
            "Latitude: %{y:.2f}, Longitude: %{x:.2f}<br><br>"
            + var_label + ": %{customdata[0]} °C<br>"
            "All-Time Warm: %{customdata[1]} °C (Year %{customdata[2]}; %{customdata[3]} °C diff)<br>"
            "All-Time Cold: %{customdata[4]} °C (Year %{customdata[5]}; %{customdata[6]} °C diff)"
            "<extra></extra>"
        )

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=mask, text=loc_labels, customdata=daily_customdata,
            colorscale=colorscale, showscale=False,
            opacity=MAP_EXTREMES_OPACITY, zmin=1, zmax=8, zsmooth=False,
            hoverongaps=True,
            hovertemplate=daily_hovertemplate,
        ))
        
        if toggles.get("hatching", False) and not is_aifs_model():
            anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
            try:
                streaks = get_persistence_arrays(
                    target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str,
                    forecast_model=selected_forecast_model(),
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
                    h_lons, h_lats = lon_grid[hatch_mask][::2], lat_grid[hatch_mask][::2]
                    fig.add_trace(go.Scatter(x=h_lons, y=h_lats, mode='markers', marker=dict(symbol='x', color='rgba(0,0,0,0.15)', size=3), hoverinfo='skip', showlegend=False))
                    
    else:
        anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
        streaks = get_persistence_arrays(
            target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str,
            forecast_model=selected_forecast_model(),
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
        _add_map_contour(
            fig, lons, lats, mslp_draw, ATMOPULSE_OVERLAY['mslp_contour'],
            _MSLP_CONTOUR_START, _MSLP_CONTOUR_END, _MSLP_CONTOUR_INTERVAL,
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
        margin=dict(t=0, l=0, r=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

@st.cache_data(show_spinner=False, max_entries=32)
def get_cached_baseline_map(
    date_str, baseline_type, map_var, view_mode, persist_metric, top10_threshold,
    t_warm_items, t_cold_items, active_toggles, source_mtime, forecast_model,
    full_width=False, anchor_date_str=None, spell_days=6,
    anom_mslp_hpa=_MSLP_ANOM_INTERVAL, _hl_version=_MSLP_HL_VERSION,
    _jet_version=_JET_OVERLAY_VERSION,
    *, _ref_data, _map_phys_data, _syn_clim=None,
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
    `forecast_model` purely so the key differs per model even though
    build_baseline_map itself reads the active model from `config`'s global
    session state, not from an argument.

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
    )


def _clone_map_trace(trace):
    data = trace.to_plotly_json()
    constructors = {"heatmap": go.Heatmap, "contour": go.Contour, "scatter": go.Scatter}
    return constructors.get(data.get("type", "scatter"), go.Scatter)(data)


def _plotly_compare_slider(*, active: int, prefix: str, steps: list) -> dict:
    return dict(
        active=active,
        x=0.08, y=0.02, len=0.84,
        pad=dict(t=6, b=6),
        currentvalue=dict(
            prefix=prefix,
            visible=True,
            xanchor="center",
            font=dict(size=12),
        ),
        steps=steps,
    )


def build_opacity_slider_map(
    fig_a, fig_b,
    label_a=epoch_period_label("A"),
    label_b=epoch_period_label("B"),
    n_steps=21,
):
    """Cross-fade two already-built maps with a Plotly layout slider.

    `method="restyle"` runs entirely in the browser, so dragging the handle
    does not trigger a Streamlit rerun or rebuild the NetCDF layers.
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
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        opac_a = [round(o * (1.0 - frac), 4) for o in base_op_a]
        opac_b = [round(o * frac, 4) for o in base_op_b]
        steps.append(dict(
            method="restyle",
            args=[{"opacity": opac_a + opac_b}, idx_a + idx_b],
            label=f"{int(round(frac * 100))}% {label_b}",
        ))

    fig.update_layout(sliders=[_plotly_compare_slider(
        active=0,
        prefix=f"{label_a} → {label_b}: ",
        steps=steps,
    )])
    for t, op in zip(fig.data[:n_a], base_op_a):
        t.opacity = op
    for t in fig.data[n_a:]:
        t.opacity = 0.0
    return fig


def render_swipe_compare_map(fig_a, fig_b) -> None:
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
    for t in fig_top.data:
        if getattr(t, "showscale", None):
            t.showscale = False
    for fig in (fig_bottom, fig_top):
        for t in fig.data:
            # Since Schritt B, build_baseline_map already sets zsmooth=False
            # on every map heatmap it creates, so this is now a no-op belt-
            # and-suspenders line. Kept explicit: zsmooth='best' would
            # interpolate each cell against its OWN neighbours, and two
            # independently smoothed grids, hard-clipped together at the
            # swipe line, blend differently right at that seam — visible as
            # a jagged strip of "wrong" pixels that belong to neither
            # dataset. Flat per-cell colour on both sides makes the seam
            # land exactly on a real data boundary instead of an
            # interpolation artifact.
            if getattr(t, "type", None) == "heatmap":
                t.zsmooth = False

    common_layout = dict(
        **plotly_typography(),
        uirevision="map_sync_state",
        autosize=True,
        title=None,
        margin=dict(t=0, l=0, r=0, b=0),
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

    st.markdown(
        "<p class='atmopulse-map-title'>Swipe Compare: Historical (left) | Recent (right)</p>",
        unsafe_allow_html=True,
    )

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
  var cfgA = {{displayModeBar: true, displaylogo: false, responsive: true,
               modeBarButtonsToRemove: ["autoScale2d", "select2d", "lasso2d"], scrollZoom: false}};
  var cfgB = {{displayModeBar: false, responsive: true, scrollZoom: false}};

  var stack = document.getElementById("swipe-stack");
  var slider = document.getElementById("atmopulse-swipe-range");
  var plotted = false;

  function apply(pct) {{
    var v = Math.max(0, Math.min(100, Number(pct)));
    stack.style.setProperty("--swipe", v + "%");
    if (String(slider.value) !== String(v)) slider.value = v;
  }}

  // The stack uses CSS aspect-ratio 70/42 (EUROPE_BBOX), same as the
  // Streamlit map frames. Do not set height from full iframe width:
  // Plotly's colour bar then shrinks the plot area and scaleanchor
  // crops latitude (appears as a zoomed-in map).
  function layout() {{
    var ctrl = document.querySelector(".atmopulse-swipe-ctrl");
    var mapH = stack.getBoundingClientRect().height || 0;
    var ctrlH = ctrl ? (ctrl.getBoundingClientRect().height || SLIDER_H) : SLIDER_H;
    if (mapH > 0) {{
      window.parent.postMessage({{type: "streamlit:setFrameHeight", height: Math.ceil(mapH + ctrlH + 8)}}, "*");
    }}
    if (plotted) {{
      Plotly.Plots.resize("swipe-a");
      Plotly.Plots.resize("swipe-b");
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
    new ResizeObserver(layout).observe(stack);
  }}
  [50, 150, 300, 600, 1000].forEach(function(t) {{ setTimeout(layout, t); }});
}})();
</script>
"""
    components.html(html, height=520, scrolling=False)
    st.caption("Drag the map or the slider: left is 1961–1990, right is 1996–2025.")
    e1, e2 = st.columns(2)
    with e1:
        render_press_export(fig_bottom, "swipe_historical", heavy=True)
    with e2:
        render_press_export(fig_top, "swipe_recent", heavy=True)

# --- METEOGRAM CORE TRACES (For Subplots) ---
def _densify_at_level_crossings(x, y, *levels):
    """Insert vertices where ``y`` crosses any climatology level.

    Daily samples are piecewise-linear. Capped fill series (min(t, P90), …)
    interpolate those caps between days, so the colour polygons drift off
    the black temperature line on steep peaks. Extra crossing vertices keep
    every linear segment inside one severity tier, so the stacked fills
    clip exactly to the line.
    """
    x = pd.to_datetime(np.asarray(x)).to_numpy(dtype="datetime64[ns]")
    y = np.asarray(y, dtype=np.float64)
    lev = [np.asarray(L, dtype=np.float64) for L in levels]
    n = y.size
    if n < 2:
        return (x, y, *lev)

    x_ns = x.astype(np.int64)
    xs = [x_ns[0]]
    ys = [y[0]]
    ls = [[L[0] for L in lev]]

    for i in range(n - 1):
        y0, y1 = y[i], y[i + 1]
        x0, x1 = x_ns[i], x_ns[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            xs.append(x1)
            ys.append(y1)
            ls.append([L[i + 1] for L in lev])
            continue
        fracs = []
        for L in lev:
            L0, L1 = L[i], L[i + 1]
            if not (np.isfinite(L0) and np.isfinite(L1)):
                continue
            denom = (y1 - y0) - (L1 - L0)
            if abs(denom) < 1e-12:
                continue
            # Sign change of (y - L) on this segment.
            if (y0 - L0) * (y1 - L1) < 0:
                f = (L0 - y0) / denom
                if 0.0 < f < 1.0:
                    fracs.append(f)
        for f in sorted(set(np.round(fracs, 10))):
            xs.append(int(x0 + f * (x1 - x0)))
            y_ins = y0 + f * (y1 - y0)
            L_ins = [L[i] + f * (L[i + 1] - L[i]) for L in lev]
            for Lv in L_ins:
                if np.isfinite(Lv) and abs(y_ins - Lv) < 1e-9:
                    y_ins = Lv
                    break
            ys.append(y_ins)
            ls.append(L_ins)
        xs.append(x1)
        ys.append(y1)
        ls.append([L[i + 1] for L in lev])

    x_out = np.array(xs, dtype="datetime64[ns]")
    y_out = np.asarray(ys, dtype=np.float64)
    lev_out = [np.asarray(col, dtype=np.float64) for col in zip(*ls)]
    return (x_out, y_out, *lev_out)


def _meteogram_class_labels(t, c_base, p75, p90, p95, rec_w, p25, p10, p5, rec_c):
    """Per-day hover class matching the meteogram colour ladder."""
    t = np.asarray(t, dtype=np.float64)
    conds = [
        t >= rec_w,
        t >= p95,
        t >= p90,
        t > p75,
        t > c_base,
        t <= rec_c,
        t <= p5,
        t <= p10,
        t < p25,
        t < c_base,
    ]
    choices = [
        "warm record",
        "warm extreme",
        "warm strong",
        "warm moderate",
        "Above average",
        "cold record",
        "cold extreme",
        "cold strong",
        "cold moderate",
        "Below average",
    ]
    return np.select(conds, choices, default="normal")


def get_meteogram_traces(df_live, ref_clim, lat, lon, target_date, epoch, meteo_env, meteo_var="TG", current_condition=None):
    """
    `current_condition`: optional ("tier", "direction") from classify_point_severity
    for THIS epoch's active-day classification; currently unused by the chart
    (kept for call-site compatibility / future use). The median-to-P75 / P25
    band is the first warm/cold colour (above/below average), not a separate
    "Typical Range" fill.
    """
    traces = []
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)
    
    # ETCCDI 365-day mapping: 29 Feb (if present in df_live) borrows the
    # climatology slot of 1 March — it is NOT excised from the plotted line.
    doys = etccdi_doy_365(df_live['Date']) - 1
    
    dates = df_live['Date']
    d_hist = dates[dates <= tgt_dt_norm]

    col_target = meteo_var_code(meteo_var)
    if col_target in df_live.columns:
        t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values
        t_full = df_live[col_target].values
        y_all = t_full
    else:
        t_hist = (df_live.loc[dates <= tgt_dt_norm, 'TX'].values + df_live.loc[dates <= tgt_dt_norm, 'TN'].values) / 2.0
        t_full = (df_live['TX'].values + df_live['TN'].values) / 2.0
        y_all = t_full

    c_base, p75_daily, p90_daily, p95_daily, rec_w_daily, p25_daily, p10_daily, p5_daily, rec_c_daily = (
        point_clim_ladder(pt_clim, doys, meteo_var, epoch)
    )
        
    env_upper = {
        "Moderate": p75_daily,
        "Strong": p90_daily,
        "Extreme": p95_daily,
        "All-Time": rec_w_daily,
    }.get(meteo_env, p90_daily)
    env_lower = {
        "Moderate": p25_daily,
        "Strong": p10_daily,
        "Extreme": p5_daily,
        "All-Time": rec_c_daily,
    }.get(meteo_env, p10_daily)
    
    # TASK 4: upper boundary trace stays showlegend=False (it is only the
    # invisible fill anchor) so "Reference Value Envelope" appears exactly
    # once in the legend, from the lower/fill trace below.
    traces.append(go.Scatter(x=dates, y=env_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=env_lower, mode='lines', fill='tonexty', fillcolor='rgba(220,220,220,0.5)', line=dict(width=0), name='Reference Value Envelope', legendgroup='env', showlegend=False, hoverinfo='skip'))

    fill_x, t_fill, c_base_f, p75_full, p90_full, p95_full, rec_w_full, p25_full, p10_full, p5_full, rec_c_full = (
        _densify_at_level_crossings(
            dates, t_full,
            c_base, p75_daily, p90_daily, p95_daily, rec_w_daily,
            p25_daily, p10_daily, p5_daily, rec_c_daily,
        )
    )

    # Warm: above average (median→P75) + Moderate/Strong/Extreme/Record
    y_above = np.where(t_fill > c_base_f, np.minimum(t_fill, p75_full), c_base_f)
    y_w1 = np.where(t_fill > p75_full, np.minimum(t_fill, p90_full), p75_full)
    y_w2 = np.where(t_fill > p90_full, np.minimum(t_fill, p95_full), y_w1)
    y_w3 = np.where(t_fill > p95_full, np.minimum(t_fill, rec_w_full), y_w2)
    y_w4 = np.where(t_fill > rec_w_full, t_fill, y_w3)

    # Cold: below average (median→P25) + Moderate/Strong/Extreme/Record
    y_below = np.where(t_fill < c_base_f, np.maximum(t_fill, p25_full), c_base_f)
    y_c1 = np.where(t_fill < p25_full, np.maximum(t_fill, p10_full), p25_full)
    y_c2 = np.where(t_fill < p10_full, np.maximum(t_fill, p5_full), y_c1)
    y_c3 = np.where(t_fill < p5_full, np.maximum(t_fill, rec_c_full), y_c2)
    y_c4 = np.where(t_fill < rec_c_full, t_fill, y_c3)

    _fill_line = dict(width=0, shape="linear")
    # Five warm + five cold fills; HTML badge legend in page_meteogram.py.
    traces.append(go.Scatter(x=fill_x, y=c_base_f, mode='lines', line=_fill_line, showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_above, mode='lines', fill='tonexty', fillcolor=warm_rgba('above'), line=_fill_line, name='Warm Above average', legendgroup='wa', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=p75_full, mode='lines', line=_fill_line, showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_w1, mode='lines', fill='tonexty', fillcolor=warm_rgba('moderate'), line=_fill_line, name='Warm Moderate', legendgroup='wm', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_w2, mode='lines', fill='tonexty', fillcolor=warm_rgba('strong'), line=_fill_line, name='Warm Strong', legendgroup='ws', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_w3, mode='lines', fill='tonexty', fillcolor=warm_rgba('extreme'), line=_fill_line, name='Warm Extreme', legendgroup='we', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_w4, mode='lines', fill='tonexty', fillcolor=warm_rgba('record'), line=_fill_line, name='Warm Record', legendgroup='wr', showlegend=False, hoverinfo='skip'))

    traces.append(go.Scatter(x=fill_x, y=c_base_f, mode='lines', line=_fill_line, showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_below, mode='lines', fill='tonexty', fillcolor=cold_rgba('below'), line=_fill_line, name='Cold Below average', legendgroup='cb', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=p25_full, mode='lines', line=_fill_line, showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_c1, mode='lines', fill='tonexty', fillcolor=cold_rgba('moderate'), line=_fill_line, name='Cold Moderate', legendgroup='cm', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_c2, mode='lines', fill='tonexty', fillcolor=cold_rgba('strong'), line=_fill_line, name='Cold Strong', legendgroup='cs', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_c3, mode='lines', fill='tonexty', fillcolor=cold_rgba('extreme'), line=_fill_line, name='Cold Extreme', legendgroup='ce', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=fill_x, y=y_c4, mode='lines', fill='tonexty', fillcolor=cold_rgba('record'), line=_fill_line, name='Cold Record', legendgroup='cr', showlegend=False, hoverinfo='skip'))

    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(color='black', width=2, shape='linear'), name='Reference Value', legendgroup='base', showlegend=False, hoverinfo='skip'))

    # Visual split only: solid through the selected day, dotted after it (join day
    # included on the dotted line so the stroke is continuous). Hover is a SINGLE
    # full-series trace — splitting hover across hist/forecast lets Plotly's
    # unified hover (hoverdistance ~20px ≈ a week on a 365-day axis) pull in the
    # neighbouring day as a second "Current Value" block.
    fcst_line_mask = dates >= tgt_dt_norm
    if col_target in df_live.columns:
        y_all = df_live[col_target].values

    code = meteo_var_code(meteo_var)
    if code == "TX":
        rec_wd_key, rec_cd_key = "tx_max_date", "tx_min_date"
    elif code == "TN":
        rec_wd_key, rec_cd_key = "tn_max_date", "tn_min_date"
    elif code == "T850":
        rec_wd_key, rec_cd_key = "t850_max_date", "t850_min_date"
    else:
        rec_wd_key = "tg_max_date" if "tg_max_date" in pt_clim.variables else "tx_max_date"
        rec_cd_key = "tg_min_date" if "tg_min_date" in pt_clim.variables else "tn_min_date"

    rec_wd = pt_clim[rec_wd_key].values[doys] if rec_wd_key in pt_clim.variables else np.full(len(doys), np.nan)
    rec_cd = pt_clim[rec_cd_key].values[doys] if rec_cd_key in pt_clim.variables else np.full(len(doys), np.nan)

    class_labels = _meteogram_class_labels(
        y_all, c_base, p75_daily, p90_daily, p95_daily, rec_w_daily,
        p25_daily, p10_daily, p5_daily, rec_c_daily,
    )
    date_labels = pd.to_datetime(dates).dt.strftime("%d.%m.%Y")
    hover_head = np.array([f"{d}: {lab}" for d, lab in zip(date_labels, class_labels)], dtype=object)

    c_data_all = np.empty((len(dates), 6), dtype=object)
    c_data_all[:, 0] = np.round(c_base, 1)
    c_data_all[:, 1] = np.round(rec_w_daily, 1)
    c_data_all[:, 2] = np.round(rec_c_daily, 1)
    c_data_all[:, 3] = _yyyymmdd_dot_date_arr(rec_wd)
    c_data_all[:, 4] = _yyyymmdd_dot_date_arr(rec_cd)
    c_data_all[:, 5] = hover_head

    hover_current = (
        "<b>%{customdata[5]}</b><br>"
        "Current Value: %{y:.1f}°C<br>"
        "Reference Value: %{customdata[0]:.1f}°C<br>"
        "Maximum: %{customdata[1]:.1f}°C%{customdata[3]}<br>"
        "Minimum: %{customdata[2]:.1f}°C%{customdata[4]}"
        "<extra></extra>"
    )

    traces.append(go.Scatter(
        x=d_hist, y=t_hist, mode='lines',
        name='Current Value', legendgroup='air', showlegend=False,
        line=dict(color='rgba(0,0,0,0.7)', width=1.5, shape='linear'), hoverinfo='skip',
    ))
    traces.append(go.Scatter(
        x=dates[fcst_line_mask], y=y_all[fcst_line_mask.values],
        mode='lines', name='Current Value (Forecast)', legendgroup='air', showlegend=False,
        line=dict(color='gray', width=2.5, dash='dot'), hoverinfo='skip',
    ))
    traces.append(go.Scatter(
        x=dates, y=y_all, mode='lines',
        line=dict(width=0, color='rgba(0,0,0,0)'),
        customdata=c_data_all, name='Current Value',
        legendgroup='air', showlegend=False, hovertemplate=hover_current,
    ))

    return traces


def _z500_fill_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _wave_hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def get_z500_anomaly_traces(df_live, syn_clim, lat, lon, target_date, epoch):
    """Expert driver panel: point Z500 minus the epoch's 5-day DOY mean (dam).

    Returns ``(traces, y_range)``. ``y_range`` is symmetric about zero.
    Empty traces and ``None`` when Z500 or the synoptic climatology is missing.
    """
    if syn_clim is None or df_live is None or df_live.empty or "Z500" not in df_live.columns:
        return [], None
    clim = synoptic_clim_point_doy(syn_clim, "z500", epoch, lat, lon)
    if clim is None or clim.size < 365:
        return [], None

    dates = pd.to_datetime(df_live["Date"], utc=True).dt.tz_convert(None)
    tgt_dt_norm = pd.to_datetime(target_date, utc=True).tz_convert(None)
    doys = etccdi_doy_365(dates)
    z_live = np.asarray(df_live["Z500"].values, dtype=np.float64)
    z_clim = clim[np.clip(doys - 1, 0, len(clim) - 1)]
    anom = z_live - z_clim
    if not np.isfinite(anom).any():
        return [], None

    span = float(np.nanmax(np.abs(anom)))
    y_lim = max(_Z500_ANOM_Y_FLOOR, span * 1.15)
    y_range = (-y_lim, y_lim)

    ridge = _z500_fill_rgba(ATMOPULSE_OVERLAY["z500_anom_contour"], 0.32)
    trough = _z500_fill_rgba(ATMOPULSE_OVERLAY["z500_contour"], 0.28)
    line_col = ATMOPULSE_OVERLAY["z500_anom_contour"]

    y_pos = np.where(np.isfinite(anom) & (anom > 0), anom, 0.0)
    y_neg = np.where(np.isfinite(anom) & (anom < 0), anom, 0.0)
    traces = [
        go.Scatter(
            x=dates, y=np.zeros(len(dates)), mode="lines",
            line=dict(color="rgba(0,0,0,0.45)", width=1),
            name="Zero", showlegend=False, hoverinfo="skip",
        ),
        go.Scatter(
            x=dates, y=y_pos, mode="lines",
            line=dict(width=0), fill="tozeroy", fillcolor=ridge,
            name="Ridge", showlegend=False, hoverinfo="skip",
        ),
        go.Scatter(
            x=dates, y=y_neg, mode="lines",
            line=dict(width=0), fill="tozeroy", fillcolor=trough,
            name="Trough", showlegend=False, hoverinfo="skip",
        ),
    ]

    fcst_mask = dates >= tgt_dt_norm
    hist_mask = dates <= tgt_dt_norm
    traces.append(go.Scatter(
        x=dates[hist_mask], y=anom[hist_mask.values], mode="lines",
        line=dict(color=line_col, width=1.5, shape="linear"),
        name="Z500 anomaly", showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=dates[fcst_mask], y=anom[fcst_mask.values], mode="lines",
        line=dict(color=line_col, width=2.0, dash="dot"),
        name="Z500 anomaly (forecast)", showlegend=False, hoverinfo="skip",
    ))

    c_data = np.empty((len(dates), 3), dtype=object)
    c_data[:, 0] = np.round(z_live, 1)
    c_data[:, 1] = np.round(z_clim, 1)
    c_data[:, 2] = np.round(anom, 1)
    traces.append(go.Scatter(
        x=dates, y=anom, mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        customdata=c_data, name="Z500 anomaly",
        showlegend=False,
        hovertemplate=(
            "<b>Z500 anomaly</b><br>"
            "Anomaly: %{customdata[2]:+.1f} dam<br>"
            "Z500: %{customdata[0]:.1f} dam<br>"
            "DOY mean: %{customdata[1]:.1f} dam"
            "<extra></extra>"
        ),
    ))
    return traces, y_range


def _days_in_runs(flag, dates, min_len=6):
    """Days inside a spell of at least `min_len` consecutive calendar days.

    Spells continue across 1 January; each day is later counted in its own year.
    A missing calendar day breaks the run.
    """
    flag = np.asarray(flag, dtype=bool)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    ns = idx.normalize().asi8
    n = len(flag)
    out = np.zeros(n, dtype=bool)
    if n == 0 or not flag.any():
        return out
    one_day = np.int64(86_400_000_000_000)
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flag[j] and (ns[j] - ns[j - 1]) == one_day:
            j += 1
        if (j - i) >= min_len:
            out[i:j] = True
        i = j
    return out


@st.cache_resource(show_spinner=False)
def build_yearly_extremes_chart(
    lat, lon, epoch, var_code, panel="warm_cold_v8",
    wsdi=False, csdi=False,
    _ref_clim=None, _load_point_archive_series=None,
):
    """Warm (top) and cold (bottom) days beyond the selected parameter's
    percentile ladder (TG / TX / TN / T850), for one reference period.
    """
    if _ref_clim is None or _load_point_archive_series is None:
        return go.Figure()
    code = var_code if var_code in ("TX", "TN", "TG", "T850") else meteo_var_code(str(var_code))
    df = _load_point_archive_series(lat, lon, code)
    if df is None:
        return go.Figure().add_annotation(text="Data Missing.", showarrow=False)
    time_col = "time" if "time" in df.columns else ("Date" if "Date" in df.columns else None)
    if time_col is not None:
        df = df.sort_values(time_col)

    pt_clim = _ref_clim.sel(latitude=lat, longitude=lon, method="nearest")
    doys = df["doy"].values
    v = np.asarray(df["val"].values, dtype=np.float64)
    _c_base, p75, p90, p95, rec_w, p25, p10, p5, rec_c = point_clim_ladder(
        pt_clim, doys, code, epoch
    )

    df = df.copy()
    dates_arr = pd.to_datetime(df["time"] if "time" in df.columns else df.get("Date", df.index))
    warm_p75 = (v >= p75) & (v < p90)
    warm_p90 = (v >= p90) & (v < p95)
    warm_p95 = (v >= p95) & (v < rec_w)
    warm_rec = v >= rec_w
    cold_p25 = (v <= p25) & (v > p10)
    cold_p10 = (v <= p10) & (v > p5)
    cold_p5 = (v <= p5) & (v > rec_c)
    cold_rec = v <= rec_c
    if wsdi:
        in_w = _days_in_runs(np.isfinite(v) & (v >= p90), dates_arr, 6)
        warm_p75 = np.zeros_like(warm_p75)
        warm_p90 = in_w & warm_p90
        warm_p95 = in_w & warm_p95
        warm_rec = in_w & warm_rec
    if csdi:
        in_c = _days_in_runs(np.isfinite(v) & (v <= p10), dates_arr, 6)
        cold_p25 = np.zeros_like(cold_p25)
        cold_p10 = in_c & cold_p10
        cold_p5 = in_c & cold_p5
        cold_rec = in_c & cold_rec
    df["p75"], df["p90"], df["p95"], df["rec_w"] = warm_p75, warm_p90, warm_p95, warm_rec
    df["p25"], df["p10"], df["p5"], df["rec_c"] = cold_p25, cold_p10, cold_p5, cold_rec
    res = df.groupby("year")[
        ["p75", "p90", "p95", "rec_w", "p25", "p10", "p5", "rec_c"]
    ].sum()

    years = res.index
    bar_kw = dict(hoverinfo="skip", hovertemplate=None)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.26,
        subplot_titles=(
            "Warm · WSDI (6×P90)" if wsdi else "Warm",
            "Cold · CSDI (6×P10)" if csdi else "Cold",
        ),
    )

    def _hover(cols):
        hover_cd = np.column_stack([res[c].to_numpy(dtype=int) for c in cols])
        y_top = sum(res[c] for c in cols)
        return go.Scatter(
            x=years, y=y_top, mode="markers",
            marker=dict(size=1, opacity=0),
            customdata=hover_cd,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Moderate: %{customdata[0]}<br>"
                "Strong: %{customdata[1]}<br>"
                "Extreme: %{customdata[2]}<br>"
                "Records: %{customdata[3]}"
                "<extra></extra>"
            ),
            hoverlabel=dict(align="left"),
            showlegend=False, name="year-hover",
        )

    fig.add_trace(go.Bar(x=years, y=res["p75"], name="Moderate", marker_color=ATMOPULSE_WARM["p75"], legend="legend", **bar_kw), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=res["p90"], name="Strong", marker_color=ATMOPULSE_WARM["p90"], legend="legend", **bar_kw), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=res["p95"], name="Extreme", marker_color=ATMOPULSE_WARM["p95"], legend="legend", **bar_kw), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=res["rec_w"], name="Records", marker_color=ATMOPULSE_WARM["rec"], legend="legend", **bar_kw), row=1, col=1)
    fig.add_trace(_hover(("p75", "p90", "p95", "rec_w")), row=1, col=1)

    fig.add_trace(go.Bar(x=years, y=res["p25"], name="Moderate", marker_color=ATMOPULSE_COLD["p25"], legend="legend2", **bar_kw), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=res["p10"], name="Strong", marker_color=ATMOPULSE_COLD["p10"], legend="legend2", **bar_kw), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=res["p5"], name="Extreme", marker_color=ATMOPULSE_COLD["p5"], legend="legend2", **bar_kw), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=res["rec_c"], name="Records", marker_color=ATMOPULSE_COLD["rec"], legend="legend2", **bar_kw), row=2, col=1)
    fig.add_trace(_hover(("p25", "p10", "p5", "rec_c")), row=2, col=1)

    y_warm = float((res["p75"] + res["p90"] + res["p95"] + res["rec_w"]).max())
    y_cold = float((res["p25"] + res["p10"] + res["p5"] + res["rec_c"]).max())
    fig.update_layout(
        **plotly_typography(),
        barmode="stack",
        hovermode="x",
        title=f"Days exceeding thresholds | {epoch_period_label(epoch)}",
        height=580,
        margin=dict(t=56, b=56, l=50, r=20),
        template="plotly_white",
        legend=dict(
            orientation="h", y=0.63, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        legend2=dict(
            orientation="h", y=-0.06, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        meta={
            "y_warm_max": y_warm,
            "y_cold_max": y_cold,
            "press_csv": (
                res.reset_index()
                .rename(columns={
                    res.index.name or "index": "year",
                    "p75": "warm_moderate_days",
                    "p90": "warm_strong_days",
                    "p95": "warm_extreme_days",
                    "rec_w": "warm_record_days",
                    "p25": "cold_moderate_days",
                    "p10": "cold_strong_days",
                    "p5": "cold_extreme_days",
                    "rec_c": "cold_record_days",
                })
                .to_csv(index=False)
            ),
        },
    )
    grid = dict(
        showgrid=True,
        gridcolor=ATMOPULSE_OVERLAY["grid"],
        gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(title_text="WSDI days" if wsdi else "days ≥ P75", rangemode="tozero", **grid, row=1, col=1)
    fig.update_yaxes(title_text="CSDI days" if csdi else "days ≤ P25", rangemode="tozero", **grid, row=2, col=1)
    fig.update_xaxes(dtick=20, tick0=1960, automargin=True, **grid, row=1, col=1)
    fig.update_xaxes(dtick=20, tick0=1960, automargin=True, **grid, row=2, col=1)
    return fig


def align_yearly_extremes_yranges(fig_a, fig_b):
    """Same Warm/Cold y-scale on side-by-side Historical vs Recent charts."""
    meta_a = fig_a.layout.meta or {}
    meta_b = fig_b.layout.meta or {}
    yw = max(float(meta_a.get("y_warm_max") or 0), float(meta_b.get("y_warm_max") or 0))
    yc = max(float(meta_a.get("y_cold_max") or 0), float(meta_b.get("y_cold_max") or 0))
    if yw > 0:
        fig_a.update_yaxes(range=[0, yw * 1.08], row=1, col=1)
        fig_b.update_yaxes(range=[0, yw * 1.08], row=1, col=1)
    if yc > 0:
        fig_a.update_yaxes(range=[0, yc * 1.08], row=2, col=1)
        fig_b.update_yaxes(range=[0, yc * 1.08], row=2, col=1)


def _empty_wave_fig() -> go.Figure:
    fig = go.Figure().add_annotation(
        text="Data Missing or Processing.", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="red", family=ATMOPULSE_FONTS["sora_css"]),
    )
    fig.update_layout(**plotly_typography())
    return fig


def _zero_to_nan(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def _stack_y(arr) -> np.ndarray:
    """Keep zeros as zeros so stacked bars share a common baseline."""
    out = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(out), np.maximum(out, 0.0), 0.0)


def _wave_stack_from_payload(payload: dict | None, stack_metric: str = "Intensity") -> dict | None:
    if not payload or payload.get("empty") or "annual_stats" not in payload:
        return None
    stats = payload["annual_stats"]
    years = np.asarray(stats.index)
    use_days = str(stack_metric).lower().startswith("day")
    is_warm = bool(payload.get("is_warm", True))
    if use_days:
        strongest = _stack_y(stats.get("max_days", stats["max_int"]))
        all_waves = np.maximum(0.0, _stack_y(stats.get("sum_days", stats["sum_int"])) - strongest)
        isolated = _stack_y(stats.get("isolated_days", 0.0))
        if isolated.shape != strongest.shape:
            isolated = np.zeros_like(strongest)
        unit = "days"
    else:
        strongest = _stack_y(stats["max_int"])
        all_waves = np.maximum(0.0, _stack_y(stats["sum_int"]) - strongest)
        isolated = np.maximum(0.0, _stack_y(stats["total_heat"]) - _stack_y(stats["sum_int"]))
        unit = "K"
    stacked = strongest + all_waves + isolated
    y_max = float(np.nanmax(stacked)) if stacked.size and np.isfinite(stacked).any() else 0.0
    if is_warm:
        names = ("Strongest heatwave", "All heatwaves", "All days over threshold")
    else:
        names = ("Strongest coldwave", "All coldwaves", "All days under threshold")
    return {
        "years": years,
        "strongest": strongest,
        "all_waves": all_waves,
        "isolated": isolated,
        "y_max": y_max,
        "unit": unit,
        "use_days": use_days,
        "is_warm": is_warm,
        "names": names,
    }


def _wave_stack_hover(stack: dict) -> go.Scatter:
    n = len(stack["years"])
    unit = stack["unit"]
    n0, n1, n2 = stack["names"]
    fmt = "{:.0f}" if stack["use_days"] else "{:.1f}"
    cd = np.empty((n, 4), dtype=object)
    for i in range(n):
        s = float(stack["strongest"][i])
        a = s + float(stack["all_waves"][i])
        t = a + float(stack["isolated"][i])
        cd[i, 0] = fmt.format(s)
        cd[i, 1] = fmt.format(a)
        cd[i, 2] = fmt.format(t)
        cd[i, 3] = fmt.format(t)
    y_top = stack["strongest"] + stack["all_waves"] + stack["isolated"]
    return go.Scatter(
        x=stack["years"], y=y_top, mode="markers",
        marker=dict(size=1, opacity=0),
        customdata=cd,
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{n0}: %{{customdata[0]}} {unit}<br>"
            f"{n1}: %{{customdata[1]}} {unit}<br>"
            f"{n2}: %{{customdata[2]}} {unit}"
            "<extra></extra>"
        ),
        hoverlabel=dict(align="left"),
        showlegend=False, name="year-hover",
    )


def _add_wave_stack_traces(fig, stack: dict) -> None:
    years = stack["years"]
    n0, n1, n2 = stack["names"]
    bar_kw = dict(hoverinfo="skip", hovertemplate=None)
    if stack["is_warm"]:
        c_iso, c_all, c_top = ATMOPULSE_WARM["p75"], ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["p95"]
    else:
        c_iso, c_all, c_top = ATMOPULSE_COLD["p25"], ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["p5"]
    # Cumulative/overlay, not additive-stack: each bar's height is its own
    # absolute total (Strongest <= All waves <= All days over threshold),
    # so hiding one legend entry never changes the height of the others —
    # a true `barmode="stack"` re-bases the remaining layers to y=0 when a
    # layer below them is hidden, and "All days over threshold" on its own
    # (just the remainder above "All waves") has no meaningful standalone
    # value. Drawn largest-first/background, smallest-last/front so it still
    # reads exactly like a stack when all three are visible; `legendrank`
    # keeps the legend order (Strongest, All waves, All days) independent
    # of that draw order.
    cum_all = stack["strongest"] + stack["all_waves"]
    cum_total = cum_all + stack["isolated"]
    fig.add_trace(go.Bar(x=years, y=cum_total, name=n2, marker_color=c_iso, legendrank=3, **bar_kw))
    fig.add_trace(go.Bar(x=years, y=cum_all, name=n1, marker_color=c_all, legendrank=2, **bar_kw))
    fig.add_trace(go.Bar(x=years, y=stack["strongest"], name=n0, marker_color=c_top, legendrank=1, **bar_kw))
    fig.add_trace(_wave_stack_hover(stack))


def _build_kysely_wave_stack_fig(payload, stack_metric: str = "Intensity") -> go.Figure:
    stack = _wave_stack_from_payload(payload, stack_metric)
    fig = go.Figure()
    if stack is not None:
        _add_wave_stack_traces(fig, stack)
    y_int = stack["y_max"] if stack is not None else 0.0
    y_title = "Intensity [days]" if str(stack_metric).lower().startswith("day") else "Intensity [K]"
    fig.update_layout(
        **plotly_typography(),
        # Overlay, not stack: each bar already carries its own cumulative
        # total (see `_add_wave_stack_traces`), so layers don't shift when
        # one is hidden via the legend.
        barmode="overlay",
        hovermode="x",
        height=360,
        margin=dict(t=20, b=56, l=55, r=20),
        template="plotly_white",
        legend=dict(
            orientation="h", y=-0.12, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        bargap=0.15,
        meta={"y_int_max": y_int},
    )
    grid = dict(showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"], gridwidth=1, zeroline=False)
    fig.update_yaxes(title_text=y_title, rangemode="tozero", **grid)
    fig.update_xaxes(dtick=10, tick0=1940, **grid)
    if payload and not payload.get("empty") and "annual_stats" in payload:
        df = payload["annual_stats"].reset_index().rename(columns={"index": "year"})
        _attach_press_csv(fig, df.to_csv(index=False))
    return fig


# Wavogram x-axis: plot_x is days from 1 Jan (heat) or 1 Jul (cold), non-leap
# month bounds. Core display is May–Sep / Nov–Mar; extra months appear only
# when a detected wave has days there (e.g. October heat, April cold).
_WARM_MONTHS = (
    (1, "JANUARY", 1, 16, 31),
    (2, "FEBRUARY", 32, 47, 59),
    (3, "MARCH", 60, 75, 90),
    (4, "APRIL", 91, 106, 120),
    (5, "MAY", 121, 136, 151),
    (6, "JUNE", 152, 167, 181),
    (7, "JULY", 182, 197, 212),
    (8, "AUGUST", 213, 228, 243),
    (9, "SEPTEMBER", 244, 259, 273),
    (10, "OCTOBER", 274, 289, 304),
    (11, "NOVEMBER", 305, 320, 334),
    (12, "DECEMBER", 335, 350, 365),
)
_COLD_MONTHS = (
    (7, "JULY", 1, 16, 31),
    (8, "AUGUST", 32, 47, 62),
    (9, "SEPTEMBER", 63, 78, 92),
    (10, "OCTOBER", 93, 108, 123),
    (11, "NOVEMBER", 124, 139, 153),
    (12, "DECEMBER", 154, 169, 184),
    (1, "JANUARY", 185, 200, 215),
    (2, "FEBRUARY", 216, 231, 243),
    (3, "MARCH", 244, 259, 274),
    (4, "APRIL", 275, 290, 304),
    (5, "MAY", 305, 320, 335),
    (6, "JUNE", 336, 351, 365),
)
_WARM_CORE_MONTHS = frozenset({5, 6, 7, 8, 9})
_COLD_CORE_MONTHS = frozenset({11, 12, 1, 2, 3})
_WARM_DEFAULT_X = (121.0, 273.0)
_COLD_DEFAULT_X = (124.0, 274.0)


def _wave_xaxis(is_warm: bool, waves_data, x_range=None) -> dict:
    """Tick/grid/range spec. Default 5-month window; expand to shoulder months
    only when an event starts or ends there. `x_range` overrides after union
    across A/B baselines so side-by-side axes stay locked."""
    months = _WARM_MONTHS if is_warm else _COLD_MONTHS
    core = _WARM_CORE_MONTHS if is_warm else _COLD_CORE_MONTHS
    x0, x1 = (_WARM_DEFAULT_X if is_warm else _COLD_DEFAULT_X)
    extra_months: set[int] = set()
    xs_min, xs_max = x0, x1
    for w in waves_data or []:
        for ts in (w.get("start_date"), w.get("end_date")):
            if ts is None:
                continue
            month = int(pd.Timestamp(ts).month)
            if month not in core:
                extra_months.add(month)
        xs = w.get("xs") or []
        if xs:
            xs_min = min(xs_min, float(np.min(xs)))
            xs_max = max(xs_max, float(np.max(xs)))
    if extra_months:
        by_month = {row[0]: row for row in months}
        for month in extra_months:
            row = by_month.get(month)
            if row is None:
                continue
            x0 = min(x0, float(row[2]))
            x1 = max(x1, float(row[4]))
        x0 = min(x0, xs_min)
        x1 = max(x1, xs_max)
    else:
        # Leap-year plot_x can sit one day past the non-leap month end.
        x0 = min(x0, xs_min)
        x1 = max(x1, xs_max)
    if x_range is not None:
        x0, x1 = float(x_range[0]), float(x_range[1])
    visible = [row for row in months if not (row[4] < x0 or row[2] > x1)]
    tick_vals = [row[3] for row in visible]
    tick_text = [row[1] for row in visible]
    grid_lines = [row[2] for row in visible]
    if visible:
        last = visible[-1]
        next_rows = [row for row in months if row[2] > last[2]]
        grid_lines.append(next_rows[0][2] if next_rows else last[4] + 1)
    return {
        "x0": x0,
        "x1": x1,
        "tick_vals": tick_vals,
        "tick_text": tick_text,
        "grid_lines": grid_lines,
    }


def union_wave_xrange(payload_a, payload_b) -> tuple[float, float]:
    """Shared ridge/frequency x-window for side-by-side / flicker A vs B."""
    payloads = [p for p in (payload_a, payload_b) if p]
    is_warm = True
    waves = []
    for p in payloads:
        is_warm = bool(p.get("is_warm", is_warm))
        if not p.get("empty"):
            waves.extend(p.get("waves_data") or [])
    spec = _wave_xaxis(is_warm, waves)
    return spec["x0"], spec["x1"]


def _build_kysely_wave_freq_fig(payload, xaxis: dict | None = None) -> go.Figure:
    freq = None if not payload or payload.get("empty") else payload.get("freq_series")
    is_warm = bool((payload or {}).get("is_warm", True))
    fig = go.Figure()
    y_freq_max = 0.0
    if xaxis is None:
        waves = None if not payload or payload.get("empty") else payload.get("waves_data")
        xaxis = _wave_xaxis(is_warm, waves)
    if freq:
        f_str, f_ext = freq["f_str"], freq["f_ext"]
        if is_warm:
            c_str, c_ext = ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["p95"]
        else:
            c_str, c_ext = ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["p5"]
        y_str = _zero_to_nan(f_str.values)
        y_ext = _zero_to_nan(f_ext.values)
        fig.add_trace(go.Scatter(
            x=f_str.index, y=y_str, mode="lines",
            line=dict(color=c_str, width=2),
            name="Strong",
            connectgaps=False, hovertemplate="%{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=f_ext.index, y=y_ext, mode="lines",
            line=dict(color=c_ext, width=2),
            name="Extreme",
            connectgaps=False, hovertemplate="%{y:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(
            tickmode="array",
            tickvals=xaxis["tick_vals"],
            ticktext=xaxis["tick_text"],
            range=[xaxis["x0"], xaxis["x1"]],
            showgrid=False, zeroline=False,
        )
        finite = np.concatenate([
            y_str[np.isfinite(y_str)] if y_str.size else np.array([0.0]),
            y_ext[np.isfinite(y_ext)] if y_ext.size else np.array([0.0]),
        ])
        y_freq_max = float(np.nanmax(finite)) if finite.size else 0.0
        if not np.isfinite(y_freq_max):
            y_freq_max = 0.0
    fig.update_layout(
        **plotly_typography(),
        hovermode="x",
        height=320,
        margin=dict(t=20, b=56, l=55, r=20),
        template="plotly_white",
        legend=dict(
            orientation="h", y=-0.12, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
        ),
        meta={"y_freq_max": y_freq_max},
    )
    grid = dict(showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"], gridwidth=1, zeroline=False)
    fig.update_yaxes(title_text="Frequency [%]", rangemode="tozero", **grid)
    if freq:
        # Same CSV-export pattern as the intensity-stack figure above — was
        # missing here, which is why this chart only ever showed SVG/PDF.
        freq_df = pd.DataFrame({
            "plot_x": f_str.index,
            "strong_pct": f_str.values,
            "extreme_pct": f_ext.values,
        })
        _attach_press_csv(fig, freq_df.to_csv(index=False))
    for gl in xaxis.get("grid_lines") or []:
        fig.add_vline(
            x=gl, line_width=1, line_color=ATMOPULSE_OVERLAY["grid"], layer="below",
        )
    return fig


def align_wave_stats_yranges(stack_a, stack_b, freq_a, freq_b):
    """Same intensity/frequency y-scales on side-by-side wavogram stats."""
    yi = max(
        float((stack_a.layout.meta or {}).get("y_int_max") or 0),
        float((stack_b.layout.meta or {}).get("y_int_max") or 0),
    )
    yf = max(
        float((freq_a.layout.meta or {}).get("y_freq_max") or 0),
        float((freq_b.layout.meta or {}).get("y_freq_max") or 0),
    )
    if yi > 0:
        stack_a.update_yaxes(range=[0, yi * 1.08])
        stack_b.update_yaxes(range=[0, yi * 1.08])
    if yf > 0:
        freq_a.update_yaxes(range=[0, yf * 1.08])
        freq_b.update_yaxes(range=[0, yf * 1.08])


# --- Point Wavogram ---
def build_kysely_wave_figs(
    payload: dict, z500_outline: bool = False, stack_metric: str = "Intensity",
    x_range=None,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """
    Renders the Point Wavogram Plotly figures (ridge-plot, seasonal intensity
    stack, annual-cycle frequency) from the compute-only payload returned by
    `backend_waves.compute_kysely_waves_data`. All Plotly/theme concerns
    (colours, fonts, ridge-curve spline smoothing, break-tail Bezier closure)
    live here; `backend_waves.py` never imports Plotly or atmopulse_theme.
    `x_range` locks A/B panels to the same (possibly expanded) season window.
    """
    is_warm = bool(payload.get("is_warm", True))
    waves_data = None if payload.get("empty") else payload.get("waves_data")
    xaxis = _wave_xaxis(is_warm, waves_data, x_range=x_range)
    if payload.get("empty", True):
        return (
            _empty_wave_fig(),
            _build_kysely_wave_stack_fig(payload, stack_metric),
            _build_kysely_wave_freq_fig(payload, xaxis),
        )

    parameter = payload["parameter"]
    suffix = payload["epoch"]
    threshold_level = payload["threshold_level"]
    waves_data = payload["waves_data"]
    p_thresh = payload["p_thresh"]
    debug_info = payload["debug_info"]
    tick_vals, tick_text = xaxis["tick_vals"], xaxis["tick_text"]
    start_plot_x, end_plot_x = xaxis["x0"], xaxis["x1"]
    grid_lines = xaxis["grid_lines"]

    fig_main = go.Figure()
    start_year, end_year = 1940, 2026
    y_ticks_vals = list(range(start_year, end_year + 1))
    y_ticks_text = [
        str(y) if is_warm else f"{y}/{str(y + 1)[2:]}" for y in y_ticks_vals
    ]

    t_suff = "Heatwaves" if is_warm else "Coldwaves"
    lvl_text = "Extreme Level (P95/5)" if "Extreme" in threshold_level else "Strong Level (P90/10)"

    fig_main.update_layout(
        **plotly_typography(),
        title=dict(
            text=f"Duration and Intensity of Local {parameter} {t_suff} (1940–2026) | {lvl_text}<br><span style='font-size:11px;color:gray;'>{epoch_period_label(suffix)}</span>",
            font=plotly_title_font(size=13),
        ),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - (5.0 if is_warm else 15.0)], showgrid=False, zeroline=False, showline=False),
        height=750, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=55, r=20, t=50, b=40),
        meta=debug_info,
    )

    for gl in grid_lines:
        fig_main.add_vline(
            x=gl, line_width=1, line_color=ATMOPULSE_OVERLAY["grid"], layer="below",
        )
    for yr in range(start_year, end_year + 1, 5):
        fig_main.add_hline(
            y=yr, line_width=1, line_color=ATMOPULSE_OVERLAY["grid"], layer="below",
        )

    if waves_data:
        for w in waves_data:
            y_base, w_xs, w_ts = w['year'], np.array(w['xs']), np.array(w['temps'])
            cum_sum = np.cumsum(np.maximum(0, w_ts - p_thresh) if is_warm else np.maximum(0, p_thresh - w_ts))

            w_df = pd.DataFrame({'x': w_xs, 'y': cum_sum}).drop_duplicates(subset=['x']).sort_values('x')
            w_xs, cum_sum = w_df['x'].values, w_df['y'].values

            if len(w_xs) >= 3:
                x_fine = np.linspace(w_xs[0], w_xs[-1], WAVE_RIDGE_SPLINE_PTS)
                y_fine = np.clip(make_interp_spline(w_xs, cum_sum, k=2)(x_fine), 0, None)
                x_skewed = _wave_ridge_x(x_fine, y_fine)
            else:
                x_skewed, y_fine = np.asarray(w_xs, dtype=float), np.asarray(cum_sum, dtype=float)
                x_skewed = _wave_ridge_x(x_skewed, y_fine)

            break_x, y_break = _wave_break_tail(x_skewed[-1], y_fine[-1])

            x_full = np.concatenate(([x_skewed[0]], x_skewed, break_x, [break_x[-1]]))
            y_full = np.concatenate(([0.0], y_fine, y_break, [0.0]))
            y_coords = y_base - (y_full / WAVE_RIDGE_HEIGHT_SCALE)

            cap = WAVE_INTENSITY_CAP_TX if is_warm else WAVE_INTENSITY_CAP_TN
            norm_val = min(w['intensity'] / cap, 1.0)
            (r_b, g_b, b_b), (r, g, b) = _wave_ridge_colors(parameter, is_warm, norm_val)

            sd_str, ed_str = pd.to_datetime(w['start_date']).strftime('%d.%m.'), pd.to_datetime(w['end_date']).strftime('%d.%m.%Y')
            hover = (
                f"<b>Duration: {sd_str}–{ed_str}</b><br>"
                f"Length: {len(w_xs)} days<br>"
                f"Severity: {w['intensity']:.1f} K"
            )
            line_color = f"rgba({r},{g},{b},{WAVE_LINE_ALPHA})"
            line_width = WAVE_LINE_WIDTH
            if show_expert("z500"):
                z_mean = w.get("z500_anom_mean")
                ridge_dam = float(payload.get("z500_ridge_dam") or 8.0)
                if z_mean is not None and np.isfinite(z_mean):
                    tag = ""
                    if z_mean >= ridge_dam:
                        tag = " (Ridge)"
                    elif z_mean <= -ridge_dam:
                        tag = " (Trough)"
                    hover += f"<br>Z500-Mean: {z_mean:+.1f} dam{tag}"
                    z_ext = w.get("z500_anom_max") if is_warm else w.get("z500_anom_min")
                    ext_label = "Z500-Max" if is_warm else "Z500-Min"
                    if z_ext is not None and np.isfinite(z_ext):
                        hover += f"<br>{ext_label}: {z_ext:+.1f} dam"
                    supporting = (
                        (is_warm and z_mean >= ridge_dam)
                        or ((not is_warm) and z_mean <= -ridge_dam)
                    )
                    if z500_outline and supporting:
                        line_color = ATMOPULSE_OVERLAY["z500_anom_contour"]
                        line_width = WAVE_Z500_LINE_WIDTH

            event_id = w.get('event_id') or (
                f"{pd.Timestamp(w['start_date']).date().isoformat()}_"
                f"{pd.Timestamp(w['end_date']).date().isoformat()}"
            )
            fig_main.add_trace(go.Scatter(
                x=x_full, y=y_coords, mode='lines',
                line=dict(color=line_color, width=line_width, shape='spline'),
                fill='toself',
                fillgradient=dict(type='vertical', colorscale=[
                    [0, f"rgba({r_b},{g_b},{b_b},{WAVE_FILL_ALPHA_BASE})"],
                    [1, f"rgba({r},{g},{b},{WAVE_FILL_ALPHA_PEAK})"],
                ]),
                hoverinfo='text',
                text=hover,
                showlegend=False,
                # Click-to-drill-down identity (Point Wavogram event-drilldown):
                # stable per event, unlike the trace index (which A/B epochs
                # don't share). Purely additive — doesn't affect rendering.
                customdata=[event_id] * len(x_full),
                meta=event_id,
            ))
    else:
        fig_main.add_annotation(text="No wave events detected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="gray", family=ATMOPULSE_FONTS["sora_css"]))

    wave_csv = None
    if waves_data:
        wave_csv = pd.DataFrame([
            {
                "year": w.get("year"),
                "start_date": w.get("start_date"),
                "end_date": w.get("end_date"),
                "duration_days": w.get("duration_days"),
                "intensity": w.get("intensity"),
                **(
                    {
                        "z500_anom_mean": w.get("z500_anom_mean"),
                        "z500_anom_max": w.get("z500_anom_max"),
                        "z500_anom_min": w.get("z500_anom_min"),
                    }
                    if "z500_anom_mean" in w else {}
                ),
            }
            for w in waves_data
        ]).to_csv(index=False)
        _attach_press_csv(fig_main, wave_csv)

    return fig_main, _build_kysely_wave_stack_fig(payload, stack_metric), _build_kysely_wave_freq_fig(payload, xaxis)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_wave_event_mini_fig(payload: dict, wave: dict, n_total: int, y_range=None) -> go.Figure:
    """One small Point Wavogram drill-down chart: the raw ERA5 daily series
    (epoch-independent) over [start-3d, end+3d] display padding, with the two
    canonical/active percentile lines (main trigger = solid + bold, drop
    tolerance = dotted + faded, same hue) and the detected event window
    shaded (ending at the last in-wave day, not the day after — that day
    already failed the Kyselý trigger/drop check). Does not touch Kyselý
    detection or the ridge geometry — this is a companion read-only view of
    `waves_data`.
    """
    is_warm = bool(payload.get("is_warm", True))
    p_thresh = payload.get("p_thresh")
    p_drop = payload.get("p_drop")
    daily = payload.get("daily_series")
    threshold_level = str(payload.get("threshold_level", ""))
    is_extreme = "Extreme" in threshold_level
    if is_warm:
        thresh_pct, drop_pct = ("P95", "P90") if is_extreme else ("P90", "P75")
    else:
        thresh_pct, drop_pct = ("P5", "P10") if is_extreme else ("P10", "P25")

    start = pd.Timestamp(wave["start_date"]).normalize()
    end = pd.Timestamp(wave["end_date"]).normalize()
    pad_start, pad_end = start - pd.Timedelta(days=3), end + pd.Timedelta(days=3)
    line_color = ATMOPULSE_WARM["p90"] if is_warm else ATMOPULSE_COLD["p10"]
    # Both threshold lines share one bold hue per severity family — same as
    # the main data line — the main trigger (P90/P10 or P95/P5) is solid and
    # full-strength, the drop tolerance (P75/P25 or P90/P10) is a fainter
    # dotted line in that same colour, not a separate/paler swatch. Warm and
    # cold use the same HSL saturation/lightness (100%/50%) so red and blue
    # read as equally intense/vivid, not one bold and the other muted.
    base_color = "#FF6600" if is_warm else "#0080FF"
    thresh_color = base_color
    drop_color = _hex_to_rgba(base_color, 0.55)
    # Label placement points each annotation away from the *other* line, so
    # they never sit in the (often narrow) gap between the two thresholds.
    # Warm: p_thresh (P90/P95) is the higher line -> label above it; p_drop
    # (P75/P90) is lower -> label below it. Cold is inverted: p_thresh
    # (P10/P5) is the *lower*, colder line, p_drop (P25/P10) sits above it.
    thresh_position = "top right" if is_warm else "bottom right"
    drop_position = "bottom right" if is_warm else "top right"

    fig = go.Figure()
    if daily is not None and len(daily):
        window = daily[(daily.index >= pad_start) & (daily.index <= pad_end)]
        if len(window):
            fig.add_trace(go.Scatter(
                x=window.index, y=window.values, mode="lines+markers",
                line=dict(color=line_color, width=2), marker=dict(size=4),
                showlegend=False, connectgaps=False,
                hovertemplate="%{y:.1f}°C<extra></extra>",
            ))
    if p_thresh is not None and np.isfinite(p_thresh):
        fig.add_hline(
            y=float(p_thresh), line_width=2.25, line_color=thresh_color,
            annotation_text=thresh_pct, annotation_position=thresh_position,
            annotation_font=dict(size=9, color=thresh_color),
        )
    if p_drop is not None and np.isfinite(p_drop):
        fig.add_hline(
            y=float(p_drop), line_width=1.25, line_dash="dot", line_color=drop_color,
            annotation_text=drop_pct, annotation_position=drop_position,
            annotation_font=dict(size=9, color=drop_color),
        )
    fig.add_vrect(
        # Ends at the last in-wave day, not the day after: that next day is
        # exactly the one the Kyselý loop popped back off (temp below the
        # drop tolerance, or the running mean falling back below p_thresh),
        # so it must render outside the shaded window, not inside it.
        x0=start, x1=end,
        fillcolor="rgba(255,153,51,0.12)" if is_warm else "rgba(51,153,255,0.12)",
        line_width=0, layer="below",
    )

    rank = wave.get("rank")
    rank_txt = f"#{rank}/{n_total} · " if rank else ""
    sd_str, ed_str = start.strftime("%d.%m.%Y"), end.strftime("%d.%m.%Y")
    dur, inten = wave.get("duration_days", 0), float(wave.get("intensity", 0.0))
    fig.update_layout(
        **plotly_typography(),
        title=dict(
            text=(
                f"{rank_txt}{sd_str}–{ed_str}<br>"
                f"<span style='font-size:10px;color:gray;'>{dur} d · {inten:.1f} K</span>"
            ),
            font=plotly_title_font(size=11),
        ),
        height=230, margin=dict(t=44, b=28, l=42, r=10),
        template="plotly_white", showlegend=False,
        hovermode="x",
    )
    fig.update_xaxes(tickformat="%d.%m", showgrid=False, zeroline=False)
    yaxis_kw = dict(showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"], zeroline=False)
    if y_range is not None:
        # Shared y-scale across all displayed slots (set by the caller from
        # the union of their data + thresholds) so the mini-charts are
        # directly comparable instead of each auto-scaling independently.
        yaxis_kw["range"] = list(y_range)
    fig.update_yaxes(**yaxis_kw)
    return fig
