"""
AtmoPulse Point Wavogram Ridge Plots (frontend_wavogram.py)

Plotly figure builders for the Point Wavogram: the Kyselý ridge-plot
drawing-only aesthetics (skew, break-tail Bezier, colour ramp — moved from
backend_waves.py so that module stays compute-only), the annual
intensity/duration stack + seasonal frequency charts
(`build_kysely_wave_figs`), and the Expert per-event drill-down mini
figures (temperature + Z500 anomaly).

Extracted from frontend_plots.py (Phase 0 split). Depends on
frontend_maps.py (map/synoptic constants) and frontend_meteogram.py (the
Z500-anomaly band/line helpers shared with the Meteogram's own Z500 panel)
at module load time; those two modules never import this one, so there is
no import cycle. `WAVE_INTENSITY_CAP_TX` / `WAVE_INTENSITY_CAP_TN` live in
config.py (shared with the future wave map colorbar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

from atmopulse_theme import (
    ATMOPULSE_COLD,
    ATMOPULSE_FONTS,
    ATMOPULSE_OVERLAY,
    ATMOPULSE_WARM,
    plotly_title_font,
    plotly_typography,
)
from backend_io import load_synoptic_climatology, synoptic_clim_point_doy
from backend_maps import etccdi_doy_365
from backend_waves import rank_waves_by_metric
from config import WAVE_INTENSITY_CAP_TN, WAVE_INTENSITY_CAP_TX, show_expert
from frontend_export import _attach_press_csv
from frontend_meteogram import (
    _Z500_LINE_WIDTH,
    _z500_anomaly_band_traces,
    _z500_anomaly_colors,
    _z500_anomaly_hover_trace,
    _z500_fill_rgba,
)

# --- Point Wavogram ridge-plot layout (tune wave shape / break aesthetics
# here — drawing-only, moved from backend_waves.py so that module stays
# compute-only). See `build_kysely_wave_figs` below. ---
WAVE_RIDGE_SPLINE_PTS = 100      # Smoothness of the ridge curve
WAVE_RIDGE_SKEW_FACTOR = 2.5     # Interior lean vs. intensity (0 = symmetric)
WAVE_RIDGE_SKEW_SPAN_FRAC = 0.30 # Max lean as a fraction of event length (days)
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


def _wave_break_tail(x_end: float, y_peak: float, span: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Visual-only post-peak closure (Bezier). `x_end` is the last event day.

    Control-x is an absolute day offset (not scaled by tail length), so a
    short tail still gets a leftward curl instead of a vertical drop.
    For short events the curl and tail are scaled down so the glyph does
    not sit a day late of its dates.
    """
    if y_peak <= 0:
        return np.array([x_end]), np.array([0.0])

    scale = 1.0
    if span is not None and span > 0:
        scale = float(np.clip(span / 6.0, 0.35, 1.0))
    t = np.linspace(0, 1, WAVE_BREAK_TAIL_STEPS)
    x_ctrl = x_end + WAVE_BREAK_CTRL_DX * scale
    y_ctrl = y_peak * WAVE_BREAK_CTRL_Y
    x_out = x_end + WAVE_BREAK_TAIL_LEN * scale

    x_break = (1 - t) ** 2 * x_end + 2 * (1 - t) * t * x_ctrl + t ** 2 * x_out
    y_break = (1 - t) ** 2 * y_peak + 2 * (1 - t) * t * y_ctrl
    return x_break, y_break


def _wave_ridge_x(x_true: np.ndarray, y_fine: np.ndarray) -> np.ndarray:
    """Lean the rising body right; keep start and peak on the real days.

    Short events cap the lean at ``WAVE_RIDGE_SKEW_SPAN_FRAC`` of their
    length so a 3-day wave is not shoved a full day later.
    """
    x_true = np.asarray(x_true, dtype=float)
    y_fine = np.asarray(y_fine, dtype=float)
    if x_true.size < 2:
        return x_true.copy()
    y_max = float(np.max(y_fine)) if y_fine.size else 0.0
    span = x_true[-1] - x_true[0]
    if y_max <= 0 or span <= 0:
        return x_true.copy()
    max_skew = min(WAVE_RIDGE_SKEW_FACTOR, WAVE_RIDGE_SKEW_SPAN_FRAC * span)
    t = np.clip((x_true - x_true[0]) / span, 0.0, 1.0)
    x_drawn = x_true + (y_fine / y_max) * max_skew * (1.0 - t)
    x_drawn[0] = x_true[0]
    x_drawn[-1] = x_true[-1]
    return x_drawn



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


# Wavogram x-axis: plot_x is ETCCDI 365-day from 1 Jan (heat) or 1 Jul (cold).
# Month bounds match that calendar (29 Feb shares 1 March). Core display is
# May–Sep / Nov–Mar; extra months appear only when an event has days there.
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
            text=f"Duration and Intensity of Local {parameter} {t_suff} (1940–2026) | {lvl_text}",
            font=plotly_title_font(size=13),
        ),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - (5.0 if is_warm else 15.0)], showgrid=False, zeroline=False, showline=False),
        height=750, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=55, r=20, t=36, b=40),
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
        ranked = rank_waves_by_metric(waves_data, stack_metric)
        n_total = len(ranked)
        rank_by_id = {w.get("event_id"): w["rank"] for w in ranked}
        rank_metric = "Days" if str(stack_metric).lower().startswith("day") else "Intensity"
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

            span = float(w_xs[-1] - w_xs[0]) if len(w_xs) else 0.0
            break_x, y_break = _wave_break_tail(x_skewed[-1], y_fine[-1], span=span)

            x_full = np.concatenate(([x_skewed[0]], x_skewed, break_x, [break_x[-1]]))
            y_full = np.concatenate(([0.0], y_fine, y_break, [0.0]))
            y_coords = y_base - (y_full / WAVE_RIDGE_HEIGHT_SCALE)

            cap = WAVE_INTENSITY_CAP_TX if is_warm else WAVE_INTENSITY_CAP_TN
            norm_val = min(w['intensity'] / cap, 1.0)
            (r_b, g_b, b_b), (r, g, b) = _wave_ridge_colors(parameter, is_warm, norm_val)

            sd_str, ed_str = pd.to_datetime(w['start_date']).strftime('%d.%m.'), pd.to_datetime(w['end_date']).strftime('%d.%m.%Y')
            event_id = w.get('event_id') or (
                f"{pd.Timestamp(w['start_date']).date().isoformat()}_"
                f"{pd.Timestamp(w['end_date']).date().isoformat()}"
            )
            rank = rank_by_id.get(event_id)
            hover = (
                f"<b>Duration: {sd_str}–{ed_str}</b><br>"
                f"Length: {len(w_xs)} days<br>"
                f"Severity: {w['intensity']:.1f} K"
            )
            if rank is not None:
                hover += f"<br>Rank: #{rank}/{n_total} ({rank_metric})"
            else:
                hover += "<br>Rank: —"
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
                    # Extra hover line; wavogram CSS shrinks the 5th tspan
                    # to ~half a line (Plotly otherwise treats every <br> as
                    # a full line and strips padding/font-size in hover HTML).
                    hover += f"<br>\u00a0<br>Z500-Mean: {z_mean:+.1f} dam{tag}"
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
    fig.update_xaxes(
        range=[pad_start, pad_end],
        tickformat="%d.%m", showgrid=False, zeroline=False,
    )
    yaxis_kw = dict(showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"], zeroline=False)
    if y_range is not None:
        # Shared y-scale across all displayed slots (set by the caller from
        # the union of their data + thresholds) so the mini-charts are
        # directly comparable instead of each auto-scaling independently.
        yaxis_kw["range"] = list(y_range)
    fig.update_yaxes(**yaxis_kw)
    return fig


def wave_event_z500_window(payload: dict, wave: dict):
    """Z500 anomaly (dam) for [start-3d, end+3d]. None if series/clim missing."""
    series = payload.get("z500_series")
    lat, lon = payload.get("lat"), payload.get("lon")
    if series is None or len(series) == 0 or lat is None or lon is None:
        return None
    epoch = "A" if str(payload.get("epoch", "B")).upper().startswith("A") else "B"
    start = pd.Timestamp(wave["start_date"]).normalize() - pd.Timedelta(days=3)
    end = pd.Timestamp(wave["end_date"]).normalize() + pd.Timedelta(days=3)
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.normalize()
    z = pd.Series(np.asarray(series.values, dtype=np.float64), index=idx)
    z = z[(z.index >= start) & (z.index <= end)]
    if z.empty or not np.isfinite(z.values).any():
        return None
    clim = synoptic_clim_point_doy(load_synoptic_climatology(), "z500", epoch, lat, lon)
    if clim is None or clim.size < 365:
        return None
    doys = etccdi_doy_365(z.index)
    z_clim = clim[np.clip(doys - 1, 0, len(clim) - 1)]
    anom = np.asarray(z.values, dtype=np.float64) - z_clim
    return z.index, anom, np.asarray(z.values, dtype=np.float64), z_clim


def build_wave_event_z500_mini_fig(payload: dict, wave: dict, y_range=None) -> go.Figure | None:
    """Expert drill-down: Z500 anomaly under one event, same ±3-day window.

    Fill, line colour/width and hover match ``get_z500_anomaly_traces``
    (Meteogram Z500 panel). The event window is the same vrect as the
    temperature mini (0.12 wash, last in-wave day, no extra outline).
    """
    packed = wave_event_z500_window(payload, wave)
    if packed is None:
        return None
    dates, anom, z_live, z_clim = packed
    is_warm = bool(payload.get("is_warm", True))
    start = pd.Timestamp(wave["start_date"]).normalize()
    end = pd.Timestamp(wave["end_date"]).normalize()
    line_col = _z500_anomaly_colors()["line"]
    fig = go.Figure()
    for tr in _z500_anomaly_band_traces(dates, anom):
        fig.add_trace(tr)
    fig.add_trace(go.Scatter(
        x=dates, y=anom, mode="lines",
        line=dict(color=line_col, width=_Z500_LINE_WIDTH, shape="linear"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(_z500_anomaly_hover_trace(dates, anom, z_live, z_clim))
    period_hex = (
        ATMOPULSE_OVERLAY["z500_anom_contour"] if is_warm
        else ATMOPULSE_OVERLAY["z500_contour"]
    )
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor=_z500_fill_rgba(period_hex, 0.12),
        line_width=0, layer="below",
    )
    fig.update_layout(
        **plotly_typography(),
        title=dict(text="Z500 anomaly", font=plotly_title_font(size=11)),
        height=170, margin=dict(t=28, b=24, l=42, r=10),
        template="plotly_white", showlegend=False, hovermode="x",
    )
    fig.update_xaxes(
        range=[start - pd.Timedelta(days=3), end + pd.Timedelta(days=3)],
        tickformat="%d.%m", showgrid=False, zeroline=False,
    )
    yaxis_kw = dict(
        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
        zeroline=True, zerolinecolor="rgba(0,0,0,0.45)",
    )
    if y_range is not None:
        yaxis_kw["range"] = list(y_range)
    fig.update_yaxes(**yaxis_kw)
    return fig
