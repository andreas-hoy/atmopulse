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

from backend_map_locations import EUROPE_BBOX
from backend_analytics import (
    _build_display_mask,
    _map_historical_records,
    _synoptic_array,
    _synoptic_lonlat,
    _synoptic_temp_pair,
    _yyyymmdd_dot_date_arr,
)
from backend_maps import etccdi_doy_365
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
    plotly_typography,
    warm_rgba,
)
from config import MAP_VAR_LABELS, is_aifs_model, is_daily_map_view, selected_forecast_model

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

def _build_standard_hovertext(labels, lat2d, lon2d, v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, var_label):
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
    lon2d, lat2d = np.meshgrid(lons, lats)
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
        
        hovertext = _build_standard_hovertext(
            loc_labels, lat2d, lon2d, v_curr, v_rec_w, yr_w, diff_w, v_rec_c, yr_c, diff_c, var_label,
        )

        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=mask, hovertext=hovertext, colorscale=colorscale, showscale=False,
            opacity=0.85, zmin=1, zmax=8, zsmooth='best',
            hovertemplate="%{hovertext}<extra></extra>",
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
            fig.add_trace(go.Heatmap(
                x=lons, y=lats, z=z,
                hovertext=(
                    "<b>" + loc_labels.astype(str) + "</b><br>"
                    "Latitude: " + np.vectorize(_fmt_hover_num)(lat2d) + ", Longitude: " + np.vectorize(_fmt_hover_num)(lon2d) + "<br><br>"
                    "Persistence: Warm: " + np.vectorize(_fmt_hover_num)(warm) + " days<br>"
                    "Persistence: Cold: " + np.vectorize(_fmt_hover_num)(cold) + " days"
                ),
                zmin=-max_days, zmax=max_days,
                colorscale=diverging_persistence_colorscale(), showscale=True, opacity=0.9, zsmooth='best',
                colorbar=persist_colorbar,
                hovertemplate="%{hovertext}<extra></extra>",
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

def build_opacity_slider_map(fig_a, fig_b, label_a="Historical Baseline (1961–1990)", label_b="Recent Baseline (1996–2025)", n_steps=21):
    """Cross-fade two baseline map figures (e.g. 1961-1990 vs 1996-2025) into
    a single Plotly figure driven by a layout slider.

    The slider uses method="restyle": Plotly.js applies the opacity change
    entirely client-side in the browser, so scrubbing it does NOT trigger a
    Streamlit rerun. Zoom, pan, and hover/tooltip state are untouched, and
    every trace (including colorbars) keeps its own opacity/visibility so
    nothing from either source figure is lost.
    """
    if not fig_a.data and not fig_b.data:
        return go.Figure()

    fig = go.Figure(layout=(fig_a.layout if fig_a.data else fig_b.layout))

    traces_a = list(fig_a.data)
    traces_b = list(fig_b.data)
    base_op_a = [1.0 if t.opacity is None else t.opacity for t in traces_a]
    base_op_b = [1.0 if t.opacity is None else t.opacity for t in traces_b]

    # Avoid stacking two identical colorbars for the same value range on top
    # of one another; the "A" layer's colorbar stays fully visible/functional
    # and continues to describe both layers since they share one colorscale.
    for t in traces_b:
        if getattr(t, "showscale", None):
            t.showscale = False

    for t in traces_a:
        fig.add_trace(t)
    for t in traces_b:
        fig.add_trace(t)

    n_a = len(traces_a)
    idx_a, idx_b = list(range(0, n_a)), list(range(n_a, n_a + len(traces_b)))

    steps = []
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        opac_a = [round(o * (1.0 - frac), 4) for o in base_op_a]
        opac_b = [round(o * frac, 4) for o in base_op_b]
        steps.append(dict(
            method="restyle",
            args=[{"opacity": opac_a + opac_b}, idx_a + idx_b],
            label=f"{int(round(frac * 100))}%",
        ))

    fig.update_layout(
        sliders=[dict(
            active=0,
            x=0.08, y=0.02, len=0.84,
            pad=dict(t=6, b=6),
            currentvalue=dict(
                prefix=f"{label_a} \u2192 {label_b}: ",
                suffix="%",
                visible=True,
                xanchor="center",
                font=dict(size=12),
            ),
            steps=steps,
        )],
    )
    # Initial render must match slider step 0 (100% historical, 0% recent).
    for t, op in zip(fig.data[:n_a], base_op_a):
        t.opacity = op
    for t, op in zip(fig.data[n_a:], base_op_b):
        t.opacity = 0.0
    return fig

# --- METEOGRAM CORE TRACES (For Subplots) ---
def get_meteogram_traces(df_live, ref_clim, lat, lon, target_date, epoch, show_air, show_app, meteo_env, meteo_var="TG", current_condition=None):
    """
    `current_condition`: optional ("tier", "direction") from classify_point_severity
    for THIS epoch's active-day classification; currently unused by the chart
    (kept for call-site compatibility / future use) now that "Normal" is
    represented by the "Typical Range" fill (P25-P75) rather than a marker.
    """
    traces = []
    sh = True if epoch == "A" else False  # Draw each legend entry only once (epoch A pass)
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)
    
    # ETCCDI 365-day mapping: 29 Feb (if present in df_live) borrows the
    # climatology slot of 1 March — it is NOT excised from the plotted line.
    doys = etccdi_doy_365(df_live['Date']) - 1
    
    dates = df_live['Date']
    
    if meteo_var == "Mean Temp (TG)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
    elif meteo_var == "Max Temp (TX)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tx_p25_doy_{epoch}'].values[doys]) / 2.0
    else:
        c_base = (pt_clim[f'tn_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
        
    env_map = {"Moderate": ("p75", "p25"), "Strong": ("p90", "p10"), "Extreme": ("p95", "p5"), "All-Time": ("max_val", "min_val")}
    el_up, el_dn = env_map.get(meteo_env, ("p90", "p10"))
    p_up_key = f'tx_{el_up}_doy_{epoch}' if el_up != "max_val" else 'tx_max_val'
    p_dn_key = f'tn_{el_dn}_doy_{epoch}' if el_dn != "min_val" else 'tn_min_val'
    
    env_upper = pt_clim[p_up_key].values[doys] if p_up_key in pt_clim.variables else np.full(len(doys), np.nan)
    env_lower = pt_clim[p_dn_key].values[doys] if p_dn_key in pt_clim.variables else np.full(len(doys), np.nan)
    
    # TASK 4: upper boundary trace stays showlegend=False (it is only the
    # invisible fill anchor) so "Reference Value Envelope" appears exactly
    # once in the legend, from the lower/fill trace below.
    traces.append(go.Scatter(x=dates, y=env_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=env_lower, mode='lines', fill='tonexty', fillcolor='rgba(220,220,220,0.5)', line=dict(width=0), name='Reference Value Envelope', legendgroup='env', showlegend=False, hoverinfo='skip'))

    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values if col_target in df_live.columns else ((df_live.loc[dates <= tgt_dt_norm, 'TX'].values + df_live.loc[dates <= tgt_dt_norm, 'TN'].values) / 2.0)

    d_hist = dates[dates <= tgt_dt_norm]

    # TASK 2: colored anomaly bands (y_w1..y_w4 / y_c1..y_c4) now span the FULL
    # `dates` axis (history + forecast) instead of stopping at d_hist/t_hist —
    # otherwise the forecast tail rendered with no shading at all, visually
    # implying "normal" even for an extreme forecasted temperature.
    t_full = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)

    # Record thresholds resolved here (not just inside `if show_air:`) so the
    # "Record" fill tier below has a genuine all-time boundary instead of
    # being an uncapped catch-all merged with "Extreme".
    if meteo_var == "Max Temp (TX)":
        _rec_w_key, _rec_c_key = "tx_max_val", "tx_min_val"
    elif meteo_var == "Min Temp (TN)":
        _rec_w_key, _rec_c_key = "tn_max_val", "tn_min_val"
    else:
        _rec_w_key = "tg_max_val" if "tg_max_val" in pt_clim.variables else "tx_max_val"
        _rec_c_key = "tg_min_val" if "tg_min_val" in pt_clim.variables else "tn_min_val"
    rec_w_full = pt_clim[_rec_w_key].values[doys] if _rec_w_key in pt_clim.variables else np.full(len(doys), np.inf)
    rec_c_full = pt_clim[_rec_c_key].values[doys] if _rec_c_key in pt_clim.variables else np.full(len(doys), -np.inf)

    # Warm Anomalies
    p75_full = pt_clim[f'tx_p75_doy_{epoch}'].values[doys] if f'tx_p75_doy_{epoch}' in pt_clim else c_base
    p90_full = pt_clim[f'tx_p90_doy_{epoch}'].values[doys] if f'tx_p90_doy_{epoch}' in pt_clim else c_base
    p95_full = pt_clim[f'tx_p95_doy_{epoch}'].values[doys] if f'tx_p95_doy_{epoch}' in pt_clim else c_base
    # TASK 3: the median-to-P75 zone is climatologically "Normal" (see
    # classify_point_severity), not "Moderate" — Moderate now starts exactly
    # at P75, matching the text classification tier-for-tier.
    y_normal_warm = np.where(t_full > c_base, np.minimum(t_full, p75_full), c_base)
    y_w1 = np.where(t_full > p75_full, np.minimum(t_full, p90_full), p75_full)
    y_w2 = np.where(t_full > p90_full, np.minimum(t_full, p95_full), y_w1)
    y_w3 = np.where(t_full > p95_full, np.minimum(t_full, rec_w_full), y_w2)
    y_w4 = np.where(t_full > rec_w_full, t_full, y_w3)

    # Cold Anomalies
    p25_full = pt_clim[f'tn_p25_doy_{epoch}'].values[doys] if f'tn_p25_doy_{epoch}' in pt_clim else c_base
    p10_full = pt_clim[f'tn_p10_doy_{epoch}'].values[doys] if f'tn_p10_doy_{epoch}' in pt_clim else c_base
    p5_full  = pt_clim[f'tn_p5_doy_{epoch}'].values[doys] if f'tn_p5_doy_{epoch}' in pt_clim else c_base
    y_normal_cold = np.where(t_full < c_base, np.maximum(t_full, p25_full), c_base)
    y_c1 = np.where(t_full < p25_full, np.maximum(t_full, p10_full), p25_full)
    y_c2 = np.where(t_full < p10_full, np.maximum(t_full, p5_full), y_c1)
    y_c3 = np.where(t_full < p5_full, np.maximum(t_full, rec_c_full), y_c2)
    y_c4 = np.where(t_full < rec_c_full, t_full, y_c3)

    # TASK 2: the 8 warm/cold severity tiers are represented by the compact
    # HTML badge legend (rendered above the chart, Map-Tracker style) instead
    # of cluttering Plotly's own legend — so all 8 stay showlegend=False here.
    # Warm side: median -> Typical Range (grey) -> P75 -> Moderate/Strong/Extreme/Record
    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_normal_warm, mode='lines', fill='tonexty', fillcolor='rgba(180,180,180,0.4)', line=dict(width=0), name='Typical Range', legendgroup='normal', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=p75_full, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_w1, mode='lines', fill='tonexty', fillcolor=warm_rgba('moderate'), line=dict(width=0), name='Warm Moderate', legendgroup='wm', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_w2, mode='lines', fill='tonexty', fillcolor=warm_rgba('strong'), line=dict(width=0), name='Warm Strong', legendgroup='ws', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_w3, mode='lines', fill='tonexty', fillcolor=warm_rgba('extreme'), line=dict(width=0), name='Warm Extreme', legendgroup='we', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_w4, mode='lines', fill='tonexty', fillcolor=warm_rgba('record'), line=dict(width=0), name='Warm Record', legendgroup='wr', showlegend=False, hoverinfo='skip'))

    # Cold side: median -> Typical Range (grey, legend already shown on warm side) -> P25 -> Moderate/Strong/Extreme/Record
    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_normal_cold, mode='lines', fill='tonexty', fillcolor='rgba(180,180,180,0.4)', line=dict(width=0), name='Typical Range', legendgroup='normal', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=p25_full, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_c1, mode='lines', fill='tonexty', fillcolor=cold_rgba('moderate'), line=dict(width=0), name='Cold Moderate', legendgroup='cm', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_c2, mode='lines', fill='tonexty', fillcolor=cold_rgba('strong'), line=dict(width=0), name='Cold Strong', legendgroup='cs', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_c3, mode='lines', fill='tonexty', fillcolor=cold_rgba('extreme'), line=dict(width=0), name='Cold Extreme', legendgroup='ce', showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=y_c4, mode='lines', fill='tonexty', fillcolor=cold_rgba('record'), line=dict(width=0), name='Cold Record', legendgroup='cr', showlegend=False, hoverinfo='skip'))

    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(color='black', width=2), name='Reference Value', legendgroup='base', showlegend=False, hoverinfo='skip'))

    # Visual split only: solid through the selected day, dotted after it (join day
    # included on the dotted line so the stroke is continuous). Hover is a SINGLE
    # full-series trace — splitting hover across hist/forecast lets Plotly's
    # unified hover (hoverdistance ~20px ≈ a week on a 365-day axis) pull in the
    # neighbouring day as a second "Current Value" block.
    hist_mask = dates <= tgt_dt_norm
    fcst_line_mask = dates >= tgt_dt_norm
    y_all = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)

    if show_app:
        col_app = 'AT_Max' if meteo_var == "Max Temp (TX)" else ('AT_Min' if meteo_var == "Min Temp (TN)" else 'AT_Mean')
        if col_app in df_live.columns:
            traces.append(go.Scatter(x=d_hist, y=df_live.loc[hist_mask, col_app], mode='lines', name='Apparent Temperature', legendgroup='app', showlegend=sh, line=dict(color=ATMOPULSE_OVERLAY['apparent_temp'], width=1.5), hoverinfo='skip'))
            traces.append(go.Scatter(x=dates[fcst_line_mask], y=df_live.loc[fcst_line_mask, col_app], mode='lines', line=dict(color=ATMOPULSE_OVERLAY['apparent_temp'], width=1.5, dash='dot'), legendgroup='app', showlegend=False, hoverinfo='skip'))
            traces.append(go.Scatter(
                x=dates, y=df_live[col_app], mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'), legendgroup='app', showlegend=False,
                hovertemplate="Apparent Temperature: %{y:.1f}°C<extra></extra>",
            ))

    if show_air:
        if meteo_var == "Max Temp (TX)":
            rec_wd_key, rec_cd_key = "tx_max_date", "tx_min_date"
        elif meteo_var == "Min Temp (TN)":
            rec_wd_key, rec_cd_key = "tn_max_date", "tn_min_date"
        else:
            rec_wd_key = "tg_max_date" if "tg_max_date" in pt_clim.variables else "tx_max_date"
            rec_cd_key = "tg_min_date" if "tg_min_date" in pt_clim.variables else "tn_min_date"

        # Reuse the record thresholds already resolved above for the "Record"
        # fill tier, rather than re-reading them from pt_clim a second time.
        rec_wd = pt_clim[rec_wd_key].values[doys] if rec_wd_key in pt_clim.variables else np.full(len(doys), np.nan)
        rec_cd = pt_clim[rec_cd_key].values[doys] if rec_cd_key in pt_clim.variables else np.full(len(doys), np.nan)

        c_data_all = np.empty((len(dates), 5), dtype=object)
        c_data_all[:, 0] = np.round(c_base, 1)
        c_data_all[:, 1] = np.round(rec_w_full, 1)
        c_data_all[:, 2] = np.round(rec_c_full, 1)
        c_data_all[:, 3] = _yyyymmdd_dot_date_arr(rec_wd)
        c_data_all[:, 4] = _yyyymmdd_dot_date_arr(rec_cd)

        hover_current = (
            "Current Value: %{y:.1f}°C<br>"
            "Reference Value: %{customdata[0]:.1f}°C<br>"
            "Maximum: %{customdata[1]:.1f}°C%{customdata[3]}<br>"
            "Minimum: %{customdata[2]:.1f}°C%{customdata[4]}"
            "<extra></extra>"
        )

        # TASK 4: this is the trace that actually shows in the Plotly legend
        # for the temperature line (the hover-carrying trace below stays
        # showlegend=False) — renamed "Air Temperature" -> "Current Value".
        traces.append(go.Scatter(
            x=d_hist, y=t_hist, mode='lines',
            name='Current Value', legendgroup='air', showlegend=False,
            line=dict(color='rgba(0,0,0,0.7)', width=1.5), hoverinfo='skip',
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
    
    fig = go.Figure()
    if is_warm:
        fig.add_trace(go.Bar(x=res.index, y=res['p75'], name='Moderate', marker_color=ATMOPULSE_WARM['p75']))
        fig.add_trace(go.Bar(x=res.index, y=res['p90'], name='Strong', marker_color=ATMOPULSE_WARM['p90']))
        fig.add_trace(go.Bar(x=res.index, y=res['p95'], name='Extreme', marker_color=ATMOPULSE_WARM['p95']))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_WARM['rec']))
    else:
        fig.add_trace(go.Bar(x=res.index, y=res['p25'], name='Moderate', marker_color=ATMOPULSE_COLD['p25']))
        fig.add_trace(go.Bar(x=res.index, y=res['p10'], name='Strong', marker_color=ATMOPULSE_COLD['p10']))
        fig.add_trace(go.Bar(x=res.index, y=res['p5'],  name='Extreme', marker_color=ATMOPULSE_COLD['p5']))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color=ATMOPULSE_COLD['rec']))

    fig.update_layout(**plotly_typography(), barmode='stack', title=f"Days exceeding thresholds | {'1961–1990' if epoch=='A' else '1996–2025'}", height=300, margin=dict(t=30, b=10), template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), yaxis=dict(rangemode="tozero"))
    return fig
