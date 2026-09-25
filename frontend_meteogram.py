"""
AtmoPulse Point Meteogram Traces (frontend_meteogram.py)

Plotly trace/figure builders for the Point Meteogram: the stacked daily
temperature trace family (`get_meteogram_traces`, capped severity fills
clipped exactly to the black temperature line via
`_densify_at_level_crossings`), the Expert Z500 anomaly band
(`get_z500_anomaly_traces`), and the yearly-extremes bar chart
(`build_yearly_extremes_chart` / `align_yearly_extremes_yranges`).

Extracted from frontend_plots.py (Phase 0 split). Depends on
frontend_maps.py for shared map/synoptic constants (e.g.
`_Z500_ANOM_Y_FLOOR`, `_Z500_CONTOUR_START`) and on frontend_export.py for
generic export plumbing; never the other way around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from atmopulse_theme import (
    ATMOPULSE_COLD,
    ATMOPULSE_OVERLAY,
    ATMOPULSE_WARM,
    cold_rgba,
    plotly_typography,
    warm_rgba,
)
from backend_analytics import _yyyymmdd_dot_date_arr
from backend_io import point_clim_ladder, synoptic_clim_point_doy
from backend_maps import etccdi_doy_365
from config import meteo_var_code
from frontend_maps import _Z500_ANOM_Y_FLOOR


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
        t_full = df_live[col_target].values
        t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values
    else:
        t_full = np.full(len(df_live), np.nan)
        t_hist = np.full(int((dates <= tgt_dt_norm).sum()), np.nan)
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
        line=dict(color='rgba(0,0,0,0.5)', width=1.1, shape='linear'), hoverinfo='skip',
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
    # Same "#RRGGBB, alpha -> rgba()" conversion as atmopulse_theme._hex_to_rgba
    # and the Wavogram's _wave_hex_to_rgb; kept as its own tiny helper here so
    # this module doesn't have to import frontend_wavogram.py (which itself
    # imports this module for `_Z500_LINE_WIDTH` — importing it back would
    # be circular).
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Shared Z500-anomaly drawing (Meteogram panel and Wavogram event minis).
_Z500_RIDGE_FILL_ALPHA = 0.32   # positive: purple fill
_Z500_TROUGH_FILL_ALPHA = 0.28  # negative: blue fill
_Z500_LINE_WIDTH = 1.5
_Z500_FCST_LINE_WIDTH = 2.0
_Z500_ZERO_LINE = dict(color="rgba(0,0,0,0.45)", width=1)
_Z500_HOVERTEMPLATE = (
    "<b>Z500 anomaly</b><br>"
    "Anomaly: %{customdata[2]:+.1f} dam<br>"
    "Z500: %{customdata[0]:.1f} dam<br>"
    "DOY mean: %{customdata[1]:.1f} dam"
    "<extra></extra>"
)


def _z500_anomaly_colors() -> dict:
    overlay = ATMOPULSE_OVERLAY
    return {
        "ridge": _z500_fill_rgba(overlay["z500_anom_contour"], _Z500_RIDGE_FILL_ALPHA),
        "trough": _z500_fill_rgba(overlay["z500_contour"], _Z500_TROUGH_FILL_ALPHA),
        "line": overlay["z500_anom_contour"],
    }


def _z500_anomaly_band_traces(dates, anom) -> list:
    """Zero line + ridge/trough fills. Same traces the Meteogram Z500 panel uses."""
    colors = _z500_anomaly_colors()
    y_pos = np.where(np.isfinite(anom) & (anom > 0), anom, 0.0)
    y_neg = np.where(np.isfinite(anom) & (anom < 0), anom, 0.0)
    return [
        go.Scatter(
            x=dates, y=np.zeros(len(dates)), mode="lines",
            line=_Z500_ZERO_LINE,
            name="Zero", showlegend=False, hoverinfo="skip",
        ),
        go.Scatter(
            x=dates, y=y_pos, mode="lines",
            line=dict(width=0), fill="tozeroy", fillcolor=colors["ridge"],
            name="Ridge", showlegend=False, hoverinfo="skip",
        ),
        go.Scatter(
            x=dates, y=y_neg, mode="lines",
            line=dict(width=0), fill="tozeroy", fillcolor=colors["trough"],
            name="Trough", showlegend=False, hoverinfo="skip",
        ),
    ]


def _z500_anomaly_hover_trace(dates, anom, z_live, z_clim) -> go.Scatter:
    c_data = np.empty((len(dates), 3), dtype=object)
    c_data[:, 0] = np.round(z_live, 1)
    c_data[:, 1] = np.round(z_clim, 1)
    c_data[:, 2] = np.round(anom, 1)
    return go.Scatter(
        x=dates, y=anom, mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        customdata=c_data, name="Z500 anomaly",
        showlegend=False, hovertemplate=_Z500_HOVERTEMPLATE,
    )


def get_z500_anomaly_traces(df_live, syn_clim, lat, lon, target_date, epoch):
    """Expert driver panel: point Z500 minus the epoch's 5-day DOY mean (dam).

    Returns ``(traces, y_range, csv_text)``. ``y_range`` is symmetric about zero.
    Empty traces and ``None`` when Z500 or the synoptic climatology is missing.
    """
    if syn_clim is None or df_live is None or df_live.empty or "Z500" not in df_live.columns:
        return [], None, None
    clim = synoptic_clim_point_doy(syn_clim, "z500", epoch, lat, lon)
    if clim is None or clim.size < 365:
        return [], None, None

    dates = pd.to_datetime(df_live["Date"], utc=True).dt.tz_convert(None)
    tgt_dt_norm = pd.to_datetime(target_date, utc=True).tz_convert(None)
    doys = etccdi_doy_365(dates)
    z_live = np.asarray(df_live["Z500"].values, dtype=np.float64)
    z_clim = clim[np.clip(doys - 1, 0, len(clim) - 1)]
    anom = z_live - z_clim
    if not np.isfinite(anom).any():
        return [], None, None

    span = float(np.nanmax(np.abs(anom)))
    y_lim = max(_Z500_ANOM_Y_FLOOR, span * 1.15)
    y_range = (-y_lim, y_lim)

    line_col = _z500_anomaly_colors()["line"]
    traces = _z500_anomaly_band_traces(dates, anom)

    fcst_mask = dates >= tgt_dt_norm
    hist_mask = dates <= tgt_dt_norm
    traces.append(go.Scatter(
        x=dates[hist_mask], y=anom[hist_mask.values], mode="lines",
        line=dict(color=line_col, width=_Z500_LINE_WIDTH, shape="linear"),
        name="Z500 anomaly", showlegend=False, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=dates[fcst_mask], y=anom[fcst_mask.values], mode="lines",
        line=dict(color=line_col, width=_Z500_FCST_LINE_WIDTH, dash="dot"),
        name="Z500 anomaly (forecast)", showlegend=False, hoverinfo="skip",
    ))
    traces.append(_z500_anomaly_hover_trace(dates, anom, z_live, z_clim))
    csv_text = pd.DataFrame({
        "date": pd.DatetimeIndex(pd.to_datetime(dates)).strftime("%Y-%m-%d"),
        "z500_dam": np.round(z_live, 2),
        "doy_mean_dam": np.round(z_clim, 2),
        "z500_anom_dam": np.round(anom, 2),
    }).to_csv(index=False)
    return traces, y_range, csv_text


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
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.28,
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
        title="Days exceeding thresholds",
        height=620,
        margin=dict(t=56, b=80, l=50, r=20),
        template="plotly_white",
        legend=dict(
            orientation="h", y=0.50, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        legend2=dict(
            orientation="h", y=-0.10, yanchor="top",
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
    fig.update_xaxes(
        dtick=20, tick0=1960, automargin=True,
        showticklabels=True, ticks="outside", visible=True, **grid,
        row=1, col=1,
    )
    fig.update_xaxes(
        dtick=20, tick0=1960, automargin=True,
        showticklabels=True, ticks="outside", visible=True, **grid,
        row=2, col=1,
    )
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

