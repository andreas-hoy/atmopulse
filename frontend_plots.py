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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.interpolate import make_interp_spline

from backend_map_locations import EUROPE_BBOX
from backend_analytics import (
    _build_display_mask,
    _map_historical_records,
    _synoptic_lonlat,
    _synoptic_temp_pair,
    _yyyymmdd_dot_date_arr,
)
from backend_maps import _synoptic_array, etccdi_doy_365
from backend_io import point_clim_ladder
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
from config import MAP_VAR_LABELS, is_aifs_model, is_daily_map_view, selected_forecast_model

# --- Point Wavogram ridge-plot layout (tune wave shape / break aesthetics
# here — drawing-only, moved from backend_waves.py so that module stays
# compute-only). See `build_kysely_wave_figs` below. ---
WAVE_RIDGE_SPLINE_PTS = 100      # Smoothness of the ridge curve
WAVE_RIDGE_SKEW_FACTOR = 2.5     # Horizontal bulge vs. intensity (0 = symmetric)
WAVE_RIDGE_HEIGHT_SCALE = 20.0   # Vertical extent in axis-year units (÷ intensity)
WAVE_BREAK_TAIL_LEN = 1.0        # X-axis length of the post-peak decay tail
WAVE_BREAK_TAIL_STEPS = 30       # Number of points along the decay tail
WAVE_BREAK_CTRL_X = -0.25        # Bezier ctrl-x (× tail_len); negative -> mid-fall bulges left
WAVE_BREAK_CTRL_Y = 0.42         # Bezier ctrl-y (× peak height); shapes the curl
WAVE_LINE_WIDTH = 1.0
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
    """AtmoPulse map palette: warm p75->rec / cold p25->rec by severity."""
    if is_warm:
        base = _wave_hex_to_rgb(ATMOPULSE_WARM["p75"])
        peak = _wave_lerp_hex(ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["rec"], norm_val)
    else:
        base = _wave_hex_to_rgb(ATMOPULSE_COLD["p25"])
        peak = _wave_lerp_hex(ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["rec"], norm_val)
    return base, peak


def _wave_break_tail(x_end: float, y_peak: float) -> tuple[np.ndarray, np.ndarray]:
    """Visual-only post-peak closure (Bezier); Kyselý data ends at the peak."""
    if y_peak <= 0:
        return np.array([x_end]), np.array([0.0])

    t = np.linspace(0, 1, WAVE_BREAK_TAIL_STEPS)
    x_ctrl = x_end + WAVE_BREAK_TAIL_LEN * WAVE_BREAK_CTRL_X
    y_ctrl = y_peak * WAVE_BREAK_CTRL_Y
    x_out = x_end + WAVE_BREAK_TAIL_LEN

    x_break = (1 - t) ** 2 * x_end + 2 * (1 - t) * t * x_ctrl + t ** 2 * x_out
    y_break = (1 - t) ** 2 * y_peak + 2 * (1 - t) * t * y_ctrl
    return x_break, y_break

MAP_VIEW_LON = (EUROPE_BBOX[0], EUROPE_BBOX[2])
MAP_VIEW_LAT = (EUROPE_BBOX[1], EUROPE_BBOX[3])
MAP_CONTOUR_LINE_WIDTH = 1.5  # was 2.5 — MSLP / Z500 isolines
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

# Vectorized once at module scope (Schritt B): reused by the customdata
# builders below instead of building a per-cell HTML string grid.
_vfmt_num = np.vectorize(_fmt_hover_num, otypes=[object])
_vfmt_diff = np.vectorize(_fmt_hover_diff, otypes=[object])
_vfmt_year = np.vectorize(_fmt_hover_year, otypes=[object])

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
    [0] warm days, [1] cold days (same 1-decimal "N/A"-safe formatting)."""
    return np.stack([_vfmt_num(warm), _vfmt_num(cold)], axis=-1)

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
        x=0.99, y=0.03,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color=ATMOPULSE_BRAND["text_on_light"], family=ATMOPULSE_FONTS["sora_css"]),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="rgba(200,200,200,0.55)",
        borderwidth=1,
        borderpad=4,
    )
    if row is None and col is None:
        fig.add_annotation(**ann)
    else:
        fig.add_annotation(**ann, row=row, col=col)

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

def _add_map_contour(fig, lons, lats, z, color, start, end, step):
    fig.add_trace(go.Contour(
        x=lons, y=lats, z=z,
        colorscale=[[0, color], [1, color]],
        contours=dict(start=start, end=end, size=step, showlabels=True, labelfont=map_contour_label_font()),
        contours_coloring='lines', showscale=False, line_width=MAP_CONTOUR_LINE_WIDTH, opacity=0.8, hoverinfo="skip",
    ))

def build_baseline_map(
    ref_data, map_phys_data, target_date, t_warm, t_cold, toggles, view_mode, persist_metric, top10_threshold,
    baseline_type="A", map_var="TG", anchor_date=None, *, full_width=False,
    border_trace=None, get_map_location_labels=None, get_persistence_arrays=None,
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
    if tx_curr is None or tn_curr is None:
        return go.Figure()
    lons, lats = _synoptic_lonlat(map_phys_data)
    
    # Align the climatology grid to the live/archive field's actual lat/lon
    # coordinates (nearest-neighbor) instead of assuming positional array
    # equality. A silent grid mismatch here (e.g. different longitude
    # convention or half-cell offset between climatology and live sources)
    # is what produces isolated coastal boundary artifacts.
    daily_ref = ref_data.sel(dayofyear=doy).reindex(
        latitude=lats, longitude=lons, method="nearest"
    )
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in daily_ref.variables: 
            return daily_ref[var_key].values
        return np.full(tx_curr.shape, fallback)

    if map_var == "TX":
        v_curr, v_p95, v_p90, v_p75 = tx_curr, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
    elif map_var == "TN":
        v_curr, v_p95, v_p90, v_p75 = tn_curr, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
    else:
        tg_curr = map_phys_data.get("tg")
        v_curr = _synoptic_array(tg_curr) if tg_curr is not None else (tx_curr + tn_curr) / 2.0
        v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
        v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
        v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
        v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
        v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
        v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2

    # All-time records: archive only, strictly before the viewed year (keeps the
    # previous record visible when the current year breaks it).
    v_rec_w, v_rec_c, yr_w, yr_c = _map_historical_records(ref_data, doy, target_date, map_var, tx_curr.shape, anchor_date)

    fig = go.Figure()
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
            opacity=0.85, zmin=1, zmax=8, zsmooth=False,
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
                
                hatch_mask = (streaks[h_idx] >= 6) | (streaks[c_idx] >= 6)
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
                "Moderate": (0, 4, 60),
                "Strong": (1, 5, 30),
                "Extreme": (2, 6, 20),
                "All-Time Record": (3, 7, 15),
            }
            w_idx, c_idx, max_days = mapping.get(persist_metric, (1, 5, 30))
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
                "Persistence: Warm: %{customdata[0]} days<br>"
                "Persistence: Cold: %{customdata[1]} days"
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
        fig.add_trace(border_trace)
    if toggles.get("mslp", False) and "mslp" in map_phys_data:
        _add_map_contour(fig, lons, lats, np.squeeze(_synoptic_array(map_phys_data["mslp"])), ATMOPULSE_OVERLAY['mslp_contour'], 980, 1040, 5)
    if toggles.get("z500", False) and "z500" in map_phys_data:
        _add_map_contour(fig, lons, lats, np.squeeze(_synoptic_array(map_phys_data["z500"])), ATMOPULSE_OVERLAY['z500_contour'], 500, 600, 8)

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
    full_width=False, anchor_date_str=None, *, _ref_data, _map_phys_data,
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
    the mslp/z500/hatching toggle dict as a `frozenset` of the active names,
    `source_mtime` (see `backend_io.synoptic_source_mtime`) so a fresh
    forecast download busts this cache even for the same date/toggles, and
    `forecast_model` purely so the key differs per model even though
    build_baseline_map itself reads the active model from `config`'s global
    session state, not from an argument.

    `_ref_data`/`_map_phys_data` are keyword-only with a leading underscore
    (Streamlit's convention for cache-key-EXCLUDED args) — identical role to
    every other `_ref_data`/`_map_phys_data` pair already used throughout
    this codebase (`compute_map_footprint`, `calculate_top10`).

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
    toggles = {name: (name in active_toggles) for name in ("mslp", "z500", "hatching")}

    return build_baseline_map(
        _ref_data, _map_phys_data, target_date, t_warm, t_cold, toggles, view_mode,
        persist_metric, top10_threshold, baseline_type, map_var,
        anchor_date=anchor_date, full_width=full_width,
        border_trace=get_europe_borders_trace(),
        get_map_location_labels=get_map_location_labels,
        get_persistence_arrays=get_persistence_arrays,
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
    label_a="1961–1990",
    label_b="1996–2025",
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
    fig_bottom.update_layout(**common_layout)
    fig_top.update_layout(**common_layout)
    # Belt-and-suspenders: explicit ranges must win regardless of dragmode.
    # Critically, `constrain="domain"` is forced on BOTH axes here (the
    # shared `_map_yaxis_kwargs()` only sets it on X; Y falls back to
    # Plotly's default `constrain="range"` for a scaleanchor'd axis). In
    # every other map the CSS box is pixel-perfect to the 70:42 ratio, so
    # that default never bites. Here the box size comes from a JS-measured
    # iframe width and is only ever approximately 70:42 — with the old
    # X-only constrain, any sub-pixel mismatch made Plotly silently CROP
    # the latitude range (chopping off southern Europe) to preserve the
    # 1:1 scaleanchor ratio. Constraining both axes' domains instead means
    # any leftover mismatch just adds a thin blank margin — the data range
    # itself (and therefore the zoom level) can never be cropped.
    fig_bottom.update_xaxes(range=list(MAP_VIEW_LON), autorange=False, constrain="domain")
    fig_bottom.update_yaxes(range=list(MAP_VIEW_LAT), autorange=False, constrain="domain")
    fig_top.update_xaxes(range=list(MAP_VIEW_LON), autorange=False, constrain="domain")
    fig_top.update_yaxes(range=list(MAP_VIEW_LAT), autorange=False, constrain="domain")

    st.markdown(
        "<p class='atmopulse-map-title'>Swipe Compare: Historical (left) | Recent (right)</p>",
        unsafe_allow_html=True,
    )

    primary = ATMOPULSE_BRAND["primary"]
    json_a = fig_bottom.to_json()
    json_b = fig_top.to_json()

    lon_span, lat_span = (MAP_VIEW_LON[1] - MAP_VIEW_LON[0]), (MAP_VIEW_LAT[1] - MAP_VIEW_LAT[0])
    aspect = lat_span / lon_span  # height / width, e.g. 42/70

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
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
(function() {{
  // The EUROPE_BBOX aspect ratio (height/width), computed in Python from
  // MAP_VIEW_LON/MAP_VIEW_LAT so it always matches the other maps exactly.
  var ASPECT = {aspect!r};
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

  // Height is derived from WIDTH (known immediately, independent of the
  // iframe's own height) instead of measuring the box's rendered height
  // and feeding that back — that read-back loop is what kept producing a
  // wrong/cropped zoom, because it raced against Streamlit's own iframe
  // sizing. Width -> explicit pixel height is a one-way, race-free
  // calculation that always reproduces the exact EUROPE_BBOX ratio.
  function layout() {{
    var w = document.documentElement.clientWidth || document.body.clientWidth || stack.clientWidth;
    if (!w) return;
    var mapH = Math.round(w * ASPECT);
    stack.style.height = mapH + "px";
    window.parent.postMessage({{type: "streamlit:setFrameHeight", height: mapH + SLIDER_H}}, "*");
    if (plotted) {{
      // Plotly.relayout({{width, height}}) does NOT reliably recompute
      // scaleanchor/constrain="domain" margins on an existing graph — it
      // can leave the two independently-resized instances (A and B) with
      // subtly different domain math, which is exactly what showed up as
      // a mismatched/distorted map and a jagged seam where they meet.
      // Plotly.Plots.resize() re-measures the (already CSS-sized) div and
      // redoes the full autosize/constrain pass, so both instances always
      // resolve to the identical EUROPE_BBOX geometry.
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
    new ResizeObserver(layout).observe(document.body);
  }}
  [50, 150, 300, 600, 1000].forEach(function(t) {{ setTimeout(layout, t); }});
}})();
</script>
"""
    components.html(html, height=int(700 * aspect) + 46, scrolling=False)
    st.caption("Drag the map or the slider: left is 1961–1990, right is 1996–2025.")

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

    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values if col_target in df_live.columns else ((df_live.loc[dates <= tgt_dt_norm, 'TX'].values + df_live.loc[dates <= tgt_dt_norm, 'TN'].values) / 2.0)
    d_hist = dates[dates <= tgt_dt_norm]
    t_full = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)

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
    y_all = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)

    if meteo_var == "Max Temp (TX)":
        rec_wd_key, rec_cd_key = "tx_max_date", "tx_min_date"
    elif meteo_var == "Min Temp (TN)":
        rec_wd_key, rec_cd_key = "tn_max_date", "tn_min_date"
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


@st.cache_resource(show_spinner=False)
def build_yearly_extremes_chart(lat, lon, epoch, is_warm, _ref_clim=None, _load_point_archive_series=None):
    """
    `_ref_clim` / `_load_point_archive_series` are app.py-owned (module-level
    climatology handle + cached loader), passed in explicitly by the caller.
    Leading underscores exclude them from Streamlit's cache-key hash (same
    convention as the rest of this codebase), so the cache key stays exactly
    (lat, lon, epoch, is_warm) as before.
    """
    if _ref_clim is None or _load_point_archive_series is None:
        return go.Figure()
    df = _load_point_archive_series(lat, lon, is_warm)
    if df is None:
        return go.Figure()

    pt_clim = _ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    doys = df['doy'].values
    v = df['val'].values

    if is_warm and f'tx_p95_doy_{epoch}' in pt_clim.variables:
        c_75, c_90, c_95, c_rec = pt_clim[f'tx_p75_doy_{epoch}'].values[doys], pt_clim[f'tx_p90_doy_{epoch}'].values[doys], pt_clim[f'tx_p95_doy_{epoch}'].values[doys], pt_clim['tx_max_val'].values[doys]
        df['p75'], df['p90'], df['p95'], df['rec'] = (v >= c_75) & (v < c_90), (v >= c_90) & (v < c_95), (v >= c_95) & (v < c_rec), v >= c_rec
    elif not is_warm and f'tn_p5_doy_{epoch}' in pt_clim.variables:
        c_25, c_10, c_5, c_rec = pt_clim[f'tn_p25_doy_{epoch}'].values[doys], pt_clim[f'tn_p10_doy_{epoch}'].values[doys], pt_clim[f'tn_p5_doy_{epoch}'].values[doys], pt_clim['tn_min_val'].values[doys]
        df['p25'], df['p10'], df['p5'], df['rec'] = (v <= c_25) & (v > c_10), (v <= c_10) & (v > c_5), (v <= c_5) & (v > c_rec), v <= c_rec
    else: 
        return go.Figure().add_annotation(text="Data Missing.", showarrow=False)

    cols_to_sum = ['year', 'p75', 'p90', 'p95', 'rec'] if is_warm else ['year', 'p25', 'p10', 'p5', 'rec']
    res = df[cols_to_sum].groupby('year').sum()
    
    cols_mod, cols_str, cols_ext, cols_rec = (
        ("p75", "p90", "p95", "rec") if is_warm else ("p25", "p10", "p5", "rec")
    )
    hover_cd = np.column_stack([
        res[cols_mod].to_numpy(dtype=int),
        res[cols_str].to_numpy(dtype=int),
        res[cols_ext].to_numpy(dtype=int),
        res[cols_rec].to_numpy(dtype=int),
    ])
    hover_tmpl = (
        "<b>%{x}</b><br>"
        "Moderate: %{customdata[0]}<br>"
        "Strong: %{customdata[1]}<br>"
        "Extreme: %{customdata[2]}<br>"
        "Records: %{customdata[3]}"
        "<extra></extra>"
    )
    bar_kw = dict(hoverinfo="skip", hovertemplate=None)
    fig = go.Figure()
    if is_warm:
        fig.add_trace(go.Bar(x=res.index, y=res['p75'], name='Moderate', marker_color=ATMOPULSE_WARM['p75'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['p90'], name='Strong', marker_color=ATMOPULSE_WARM['p90'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['p95'], name='Extreme', marker_color=ATMOPULSE_WARM['p95'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_WARM['rec'], **bar_kw))
    else:
        fig.add_trace(go.Bar(x=res.index, y=res['p25'], name='Moderate', marker_color=ATMOPULSE_COLD['p25'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['p10'], name='Strong', marker_color=ATMOPULSE_COLD['p10'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['p5'],  name='Extreme', marker_color=ATMOPULSE_COLD['p5'], **bar_kw))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_COLD['rec'], **bar_kw))

    y_top = res[cols_mod] + res[cols_str] + res[cols_ext] + res[cols_rec]
    fig.add_trace(go.Scatter(
        x=res.index, y=y_top,
        mode="markers",
        marker=dict(size=1, opacity=0),
        customdata=hover_cd,
        hovertemplate=hover_tmpl,
        hoverlabel=dict(align="left"),
        showlegend=False,
        name="year-hover",
    ))

    fig.update_layout(
        **plotly_typography(),
        barmode="stack",
        hovermode="x",
        title=f"Days exceeding thresholds | {'1961–1990' if epoch=='A' else '1996–2025'}",
        height=340,
        margin=dict(t=40, b=88, l=50, r=20),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        yaxis=dict(rangemode="tozero"),
        xaxis=dict(automargin=True),
    )
    return fig


# --- Point Wavogram ---
def build_kysely_wave_figs(payload: dict) -> tuple[go.Figure, go.Figure]:
    """
    Renders the two Point Wavogram Plotly figures (ridge-plot `fig_main` +
    stats/frequency panel `fig_stats`) from the compute-only payload
    returned by `backend_waves.compute_kysely_waves_data`. All Plotly/theme
    concerns (colours, fonts, ridge-curve spline smoothing, break-tail
    Bezier closure) live here; `backend_waves.py` never imports Plotly or
    atmopulse_theme.
    """
    if payload.get("empty", True):
        empty_fig = go.Figure().add_annotation(text="Data Missing or Processing.", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="red", family=ATMOPULSE_FONTS["sora_css"]))
        empty_fig.update_layout(**plotly_typography())
        return empty_fig, empty_fig

    parameter = payload["parameter"]
    suffix = payload["epoch"]
    threshold_level = payload["threshold_level"]
    stat_metric = payload["stat_metric"]
    is_warm = payload["is_warm"]
    waves_data = payload["waves_data"]
    p_thresh = payload["p_thresh"]
    p_t_ext, p_d_ext = payload["p_t_ext"], payload["p_d_ext"]
    p_t_str, p_d_str = payload["p_t_str"], payload["p_d_str"]
    debug_info = payload["debug_info"]

    if is_warm:
        tick_vals, tick_text = [16, 46, 77, 107, 138], ["MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER"]
        start_plot_x, end_plot_x = 1, 153
        grid_lines = [1, 32, 62, 93, 124, 154]
    else:
        tick_vals, tick_text = [16, 46, 77, 107, 136], ["NOV", "DEC", "JAN", "FEB", "MAR"]
        start_plot_x, end_plot_x = 1, 152
        grid_lines = [1, 31, 62, 93, 121, 152]

    fig_main = go.Figure()
    start_year, end_year = 1940, 2026
    y_ticks_vals = list(range(start_year, end_year + 1))
    y_ticks_text = [str(y) if is_warm else f"{y-1}/{str(y)[2:]}" for y in y_ticks_vals]

    t_suff = "Heatwaves" if is_warm else "Coldwaves"
    lvl_text = "Extreme Level (P95/5)" if "Extreme" in threshold_level else "Strong Level (P90/10)"

    fig_main.update_layout(
        **plotly_typography(),
        title=dict(
            text=f"Duration and Intensity of Local {parameter} {t_suff} (1940–2026) | {lvl_text}<br><span style='font-size:11px;color:gray;'>Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}</span>",
            font=plotly_title_font(size=13),
        ),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - (5.0 if is_warm else 15.0)], showgrid=False, zeroline=False, showline=False),
        height=750, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=55, r=20, t=50, b=40),
        meta=debug_info,
    )

    for gl in grid_lines:
        fig_main.add_vline(x=gl, line_width=1.2, line_color="rgba(30,30,30,0.6)", layer="below")
    for yr in range(start_year, end_year + 1, 5):
        fig_main.add_hline(y=yr, line_width=0.7, line_color="rgba(100,100,100,0.4)", layer="below")

    if waves_data:
        for w in waves_data:
            y_base, w_xs, w_ts = w['year'], np.array(w['xs']), np.array(w['temps'])
            cum_sum = np.cumsum(np.maximum(0, w_ts - p_thresh) if is_warm else np.maximum(0, p_thresh - w_ts))

            w_df = pd.DataFrame({'x': w_xs, 'y': cum_sum}).drop_duplicates(subset=['x']).sort_values('x')
            w_xs, cum_sum = w_df['x'].values, w_df['y'].values

            if len(w_xs) >= 3:
                x_fine = np.linspace(w_xs[0], w_xs[-1], WAVE_RIDGE_SPLINE_PTS)
                y_fine = np.clip(make_interp_spline(w_xs, cum_sum, k=2)(x_fine), 0, None)
                x_skewed = x_fine + (y_fine / (max(y_fine) if max(y_fine) > 0 else 1)) * WAVE_RIDGE_SKEW_FACTOR
            else:
                x_skewed, y_fine = w_xs, cum_sum

            break_x, y_break = _wave_break_tail(x_skewed[-1], y_fine[-1])

            x_full = np.concatenate(([x_skewed[0]], x_skewed, break_x, [break_x[-1]]))
            y_full = np.concatenate(([0.0], y_fine, y_break, [0.0]))
            y_coords = y_base - (y_full / WAVE_RIDGE_HEIGHT_SCALE)

            cap = WAVE_INTENSITY_CAP_TX if is_warm else WAVE_INTENSITY_CAP_TN
            norm_val = min(w['intensity'] / cap, 1.0)
            (r_b, g_b, b_b), (r, g, b) = _wave_ridge_colors(parameter, is_warm, norm_val)

            sd_str, ed_str = pd.to_datetime(w['start_date']).strftime('%d.%m.'), pd.to_datetime(w['end_date']).strftime('%d.%m.%Y')

            fig_main.add_trace(go.Scatter(
                x=x_full, y=y_coords, mode='lines',
                line=dict(color=f"rgba({r},{g},{b},{WAVE_LINE_ALPHA})", width=WAVE_LINE_WIDTH, shape='spline'),
                fill='toself',
                fillgradient=dict(type='vertical', colorscale=[
                    [0, f"rgba({r_b},{g_b},{b_b},{WAVE_FILL_ALPHA_BASE})"],
                    [1, f"rgba({r},{g},{b},{WAVE_FILL_ALPHA_PEAK})"],
                ]),
                hoverinfo='text',
                text=f"<b>Duration: {sd_str}–{ed_str}</b><br>Length: {len(w_xs)} days<br>Severity: {w['intensity']:.1f} K",
                showlegend=False,
            ))
    else:
        fig_main.add_annotation(text="No wave events detected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="gray", family=ATMOPULSE_FONTS["sora_css"]))

    if stat_metric == "Annual Cycle Frequency":
        freq_series = payload["freq_series"]
        f_str, f_ext = freq_series["f_str"], freq_series["f_ext"]

        fig_stats = go.Figure()
        if is_warm:
            c_str, c_ext = ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["p95"]
        else:
            c_str, c_ext = ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["p5"]

        fig_stats.add_trace(go.Scatter(x=f_str.index, y=f_str.values, mode='lines', line=dict(color=c_str, width=2), name="Strong", hovertemplate='%{y:.1f}%<extra></extra>'))
        fig_stats.add_trace(go.Scatter(x=f_ext.index, y=f_ext.values, mode='lines', line=dict(color=c_ext, width=2), name="Extreme", hovertemplate='%{y:.1f}%<extra></extra>'))

        fig_stats.update_layout(
            **plotly_typography(),
            title=f"Annual Cycle Frequency (5-Day Smoothing) | Reference {'1961–1990' if suffix=='A' else '1996–2025'}",
            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, showgrid=True),
            yaxis_title="Relative Frequency (%)",
            height=350, template="plotly_white",
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        return fig_main, fig_stats

    stats = payload["annual_stats"]
    col_map = {"Cumulative Annual Wave Intensity": 'sum_int', "Maximum Annual Wave Intensity": 'max_int', "Cumulative Heat/Cold Intensity": 'total_heat'}
    sel_col = col_map.get(stat_metric, 'sum_int')
    y_titles = {
        'sum_int': 'Σ wave intensity (K·days)',
        'max_int': 'Max wave intensity (K·days)',
        'total_heat': f'Σ excess vs. threshold (K·days)',
    }

    fig_stats = go.Figure()
    if is_warm:
        bar_color, mean_color = "#E8A8A0", ATMOPULSE_WARM["p95"]
    else:
        bar_color, mean_color = "#9EC5E8", ATMOPULSE_COLD["p5"]

    fig_stats.add_trace(go.Bar(
        x=stats.index, y=stats[sel_col], marker_color=bar_color, name="Intensity",
        hovertemplate='Year: %{x}<br>Value: %{y:.1f} K<extra></extra>',
    ))
    fig_stats.add_trace(go.Scatter(
        x=stats.index, y=stats[sel_col].rolling(11, center=True).mean(), mode='lines',
        line=dict(color=mean_color, width=2.5), name="11-yr Mean",
        hovertemplate='Year: %{x}<br>11-year mean: %{y:.1f} K<extra></extra>',
    ))

    valid = stats[sel_col].dropna()
    if len(valid) > 2:
        z = np.polyfit(valid.index, valid.values, 1)
        fig_stats.add_trace(go.Scatter(
            x=valid.index, y=np.poly1d(z)(valid.index), mode='lines',
            line=dict(color=mean_color, width=1.5, dash='dot'), name="Trend", hoverinfo='skip',
        ))

    fig_stats.update_layout(
        **plotly_typography(),
        title=f"{stat_metric} | Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}",
        height=350, template="plotly_white",
        margin=dict(t=40, b=10, l=55, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(180,180,180,0.35)", gridwidth=1, dtick=10, zeroline=False),
        yaxis=dict(title=y_titles.get(sel_col, "Intensity (K·days)"), showgrid=True, gridcolor="rgba(180,180,180,0.35)", gridwidth=1, zeroline=False),
        bargap=0.15,
    )

    return fig_main, fig_stats
