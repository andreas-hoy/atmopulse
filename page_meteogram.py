"""
AtmoPulse Point Meteogram Page (page_meteogram.py)

Extracted from app.py so the Meteogram's blocking calls (reference
climatology NetCDF open, live point-series extraction) only ever run when a
user actually selects the Point Meteogram tab with a resolved location —
never on app cold-start, and never for Welcome/Legal/Methods/Map Tracker/
Point Wavogram.
"""

from __future__ import annotations

import html

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend_narrative import (
    classify_point_severity,
    point_condition_phrase,
    baseline_against_lead,
    epoch_period_label,
    epoch_from_label,
)
from labels import HELP
from atmopulse_theme import (
    ATMOPULSE_OVERLAY,
    legend_badge_style,
    plotly_typography,
)
from config import (
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    STANDARD_DEFAULTS,
    AIFS_TXTN_WARNING,
    is_aifs_model,
    selected_forecast_model,
    show_expert,
    meteo_var_code,
)
from backend_io import (
    load_reference_climatology,
    load_synoptic_climatology,
    get_live_point_series,
    compute_point_thresholds,
    _load_point_archive_series,
)
from frontend_plots import (
    get_meteogram_traces,
    get_z500_anomaly_traces,
    build_yearly_extremes_chart,
    align_yearly_extremes_yranges,
    st_plotly_press,
)


def _severity_phrase_html(cat: str, direction: str | None, phrase: str) -> str:
    """Wrap a narrative phrase in a CSS chip class (legend colours, !important)."""
    safe = html.escape(phrase)
    if not direction or cat == "normal":
        cls = "atmopulse-narrative-chip atmopulse-sev-normal"
    else:
        cls = f"atmopulse-narrative-chip atmopulse-sev-{direction}-{cat}"
    return f'<span class="{cls}">{safe}</span>'


def _render_html(body: str) -> None:
    """Prefer st.html so Streamlit does not strip chip markup."""
    html_fn = getattr(st, "html", None)
    if callable(html_fn):
        html_fn(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def render_meteogram(location, lat_target, lon_target, meteo_var, meteo_env, target_date, meteo_spell=False):
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        st.error("Reference Climatology missing or corrupted! Please rebuild.")
        st.stop()

    if show_expert("flicker_layout"):
        map_layout = st.radio("Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER), horizontal=True, key="met_layout")
    else:
        map_layout = STANDARD_DEFAULTS["map_layout"]
    # Determine the active Flicker-mode reference period BEFORE the
    # narrative text is built below, so the sentence never "leaks" a
    # hardcoded baseline that doesn't match what the widget actually
    # shows under the chart. Re-rendered later at its original chart
    # position using the SAME key (Streamlit persists the selection
    # across the rerun, so reading it here is safe).
    if map_layout == LAYOUT_FLICKER:
        met_active_epoch = epoch_from_label(st.session_state.get("met_ep", epoch_period_label("B")))
    if is_aifs_model() and meteo_var_code(meteo_var) in ("TX", "TN"):
        st.warning(AIFS_TXTN_WARNING)
    else:
        with st.spinner("Fetching Meteogram data..."): 
            df_live = get_live_point_series(lat_target, lon_target, selected_forecast_model())
        if not df_live.empty:
            col_target = meteo_var_code(meteo_var)

            # --- STRICT DATETIME INDEXING for "current conditions" ---
            # Never .max()/.mean() over the series, and never a bare
            # .iloc[0]/.iloc[-1] unless target_date genuinely falls
            # outside the live window — the scalar MUST come from the
            # exact calendar row matching the active target_date.
            #
            # `df_live` (get_live_point_series) is built entirely from calendar-day
            # UTC aggregates — the ERA5 archive's daily valid_time and the IFS/AIFS
            # forecast's own 00Z-00Z daily aggregation (ifs_ingestion.py) — so it is
            # already on the same UTC calendar-day footing as `active_date` below.
            df_indexed = df_live.copy()
            df_indexed['Date'] = pd.to_datetime(df_indexed['Date']).dt.tz_localize(None).dt.normalize()
            df_indexed = df_indexed.drop_duplicates(subset=['Date']).set_index('Date').sort_index()
            active_date = (
                pd.Timestamp.utcnow().tz_localize(None).floor('D')
                + pd.Timedelta(days=st.session_state.offset_slider)
            )

            try:
                current_row = df_indexed.loc[active_date]
                current_row_date = active_date
            except KeyError:
                # Defensive-only safety net (e.g. offset_slider pushed past what
                # the live series actually returned, or an upstream API gap) —
                # NOT the primary alignment mechanism anymore. Nearest available
                # calendar day, never the series' arbitrary last/forecast row.
                nearest_pos = df_indexed.index.get_indexer([active_date], method='nearest')[0]
                current_row = df_indexed.iloc[nearest_pos]
                current_row_date = df_indexed.index[nearest_pos]

            if col_target in df_indexed.columns:
                value_now = float(current_row[col_target])
            elif col_target != "T850" and 'TX' in df_indexed.columns and 'TN' in df_indexed.columns:
                value_now = float((current_row['TX'] + current_row['TN']) / 2.0)
            else:
                value_now = np.nan

            # Thresholds must be evaluated for THAT specific day (current_row_date),
            # not the slider's nominal target_date, so a forecast-fallback row never
            # gets scored against the wrong calendar day's P75/90/95 climatology.
            #
            # Baseline isolation: each epoch gets its OWN, freshly-built p_warm/p_cold
            # dict from compute_point_thresholds — "A" and "B" thresholds are never
            # assigned into the same variable, so there is no possibility of one
            # baseline's percentiles silently overwriting the other's.
            # Both baselines' thresholds are computed unconditionally (cheap — just
            # ref_clim.sel()+array-index lookups, no I/O) so the debug readout below
            # can always show both, regardless of which layout is active.
            percentiles_a = compute_point_thresholds(ref_clim, lat_target, lon_target, current_row_date, meteo_var, "A")
            percentiles_b = compute_point_thresholds(ref_clim, lat_target, lon_target, current_row_date, meteo_var, "B")
            cat_a, dir_a = classify_point_severity(value_now, *percentiles_a)
            cat_b, dir_b = classify_point_severity(value_now, *percentiles_b)
            addr = html.escape(location.address)

            if map_layout == LAYOUT_FLICKER:
                cat_x, dir_x = (cat_a, dir_a) if met_active_epoch == "A" else (cat_b, dir_b)
                chip = _severity_phrase_html(cat_x, dir_x, point_condition_phrase(cat_x, dir_x))
                lead = html.escape(baseline_against_lead(met_active_epoch))
                _render_html(
                    f"<div class='atmopulse-narrative-banner'>"
                    f"{lead}, the area of {addr} is currently{chip}."
                    f"</div>"
                )
            else:
                a_html = _severity_phrase_html(cat_a, dir_a, point_condition_phrase(cat_a, dir_a))
                b_html = _severity_phrase_html(cat_b, dir_b, point_condition_phrase(cat_b, dir_b))
                lead_a = html.escape(baseline_against_lead("A"))
                lead_b = html.escape(baseline_against_lead("B"))
                _render_html(
                    f"<div class='atmopulse-narrative-banner'>"
                    f"{lead_a}, the area of {addr} is currently{a_html}. "
                    f"{lead_b}, it is{b_html}."
                    f"</div>"
                )

            t_arr = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)
            global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
            tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)

            traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "A", meteo_env, meteo_var, current_condition=(cat_a, dir_a))
            traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "B", meteo_env, meteo_var, current_condition=(cat_b, dir_b))

            show_z500 = show_expert("z500")
            syn_clim = load_synoptic_climatology() if show_z500 else None
            z500_a, z500_rng_a = get_z500_anomaly_traces(
                df_live, syn_clim, lat_target, lon_target, target_date, "A",
            ) if show_z500 else ([], None)
            z500_b, z500_rng_b = get_z500_anomaly_traces(
                df_live, syn_clim, lat_target, lon_target, target_date, "B",
            ) if show_z500 else ([], None)
            use_z500 = bool(z500_a and z500_b)
            z500_ylim = None
            if use_z500 and z500_rng_a and z500_rng_b:
                z500_ylim = (
                    min(z500_rng_a[0], z500_rng_b[0]),
                    max(z500_rng_a[1], z500_rng_b[1]),
                )
            vline_x = tgt_dt_norm.timestamp() * 1000
            title_a = f"{epoch_period_label('A').replace(' (', '<br>(')}"
            title_b = f"{epoch_period_label('B').replace(' (', '<br>(')}"

            # Five warm + five cold chips. Moderate/Strong/Extreme/Record share
            # the Map Tracker palette; Above/Below average are the pale extras.
            st.markdown(
                f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
                f"style='margin-bottom: 6px; white-space: nowrap;'>"
                f"<b>Legend.</b> "
                f"<span style='padding-left: 4px;'>Warm:</span> "
                f"<span style='{legend_badge_style('warm', 'above')}'>Above average</span> "
                f"<span style='{legend_badge_style('warm', 'moderate')}'>Moderate</span> "
                f"<span style='{legend_badge_style('warm', 'strong')}'>Strong</span> "
                f"<span style='{legend_badge_style('warm', 'extreme')}'>Extreme</span> "
                f"<span style='{legend_badge_style('warm', 'record')}'>Record</span>"
                f"<span style='padding-left: 12px;'>Cold:</span> "
                f"<span style='{legend_badge_style('cold', 'below')}'>Below average</span> "
                f"<span style='{legend_badge_style('cold', 'moderate')}'>Moderate</span> "
                f"<span style='{legend_badge_style('cold', 'strong')}'>Strong</span> "
                f"<span style='{legend_badge_style('cold', 'extreme')}'>Extreme</span> "
                f"<span style='{legend_badge_style('cold', 'record')}'>Record</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            
            if map_layout == LAYOUT_SIDE_BY_SIDE:
                fig = make_subplots(
                    rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=(title_a, title_b),
                )
                for trace in traces_a:
                    fig.add_trace(trace, row=1, col=1)
                for trace in traces_b:
                    fig.add_trace(trace, row=1, col=2)
                fig.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                fig.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                fig.update_yaxes(range=[global_min, global_max], row=1, col=1)
                fig.update_yaxes(range=[global_min, global_max], row=1, col=2)
                fig.update_layout(
                    **plotly_typography(), hovermode="x", height=520,
                    template="plotly_white", margin=dict(t=56, b=10), showlegend=False,
                )
                fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                live_csv = df_live.to_csv(index=False)
                st_plotly_press(fig, "meteogram_compare", csv_text=live_csv)
                if use_z500:
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    fig_z = make_subplots(rows=1, cols=2, shared_yaxes=True)
                    for trace in z500_a:
                        fig_z.add_trace(trace, row=1, col=1)
                    for trace in z500_b:
                        fig_z.add_trace(trace, row=1, col=2)
                    fig_z.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                    fig_z.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                    fig_z.update_yaxes(
                        range=list(z500_ylim), title_text="Z500 anom. (dam)",
                        showticklabels=True, ticks="outside", automargin=True,
                        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
                        zeroline=True, zerolinecolor="rgba(0,0,0,0.45)",
                        row=1, col=1,
                    )
                    fig_z.update_yaxes(
                        range=list(z500_ylim),
                        showticklabels=True, ticks="outside",
                        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
                        zeroline=True, zerolinecolor="rgba(0,0,0,0.45)",
                        row=1, col=2,
                    )
                    fig_z.update_xaxes(
                        dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y",
                        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
                        ticks="outside", automargin=True,
                    )
                    fig_z.update_layout(
                        **plotly_typography(), hovermode="x", height=280,
                        template="plotly_white", margin=dict(t=10, b=40), showlegend=False,
                    )
                    st_plotly_press(fig_z, "meteogram_z500_compare")
                st.markdown("<div class='atmopulse-meteo-yearly-gap'></div>", unsafe_allow_html=True)

                fig_yr_a = build_yearly_extremes_chart(lat_target, lon_target, "A", col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series)
                fig_yr_b = build_yearly_extremes_chart(lat_target, lon_target, "B", col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series)
                align_yearly_extremes_yranges(fig_yr_a, fig_yr_b)
                c1, c2 = st.columns(2)
                with c1:
                    st_plotly_press(fig_yr_a, "yearly_historical")
                with c2:
                    st_plotly_press(fig_yr_b, "yearly_recent")
            else:
                # Same widget key ("met_ep") whose value we already read into
                # `met_active_epoch` above (before the narrative text was built) —
                # re-rendering it here just places it at its usual spot below the chart.
                flicker_epoch = st.radio(
                    "Select Reference Period:",
                    (epoch_period_label("A"), epoch_period_label("B")),
                    horizontal=True, key="met_ep", index=1,
                )
                met_active_epoch = epoch_from_label(flicker_epoch)
                traces = traces_a if met_active_epoch == "A" else traces_b
                z500_tr = z500_a if met_active_epoch == "A" else z500_b
                fig = go.Figure(data=traces)
                fig.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8)
                fig.update_yaxes(range=[global_min, global_max])
                fig.update_layout(
                    **plotly_typography(), title=epoch_period_label(met_active_epoch),
                    hovermode="x", height=500, template="plotly_white",
                    margin=dict(t=40, b=10), showlegend=False,
                )
                fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                st_plotly_press(fig, f"meteogram_{met_active_epoch}", csv_text=df_live.to_csv(index=False))
                if use_z500:
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    fig_z = go.Figure(data=z500_tr)
                    fig_z.add_vline(x=vline_x, line_dash="dash", line_color="gray", opacity=0.8)
                    fig_z.update_yaxes(
                        range=list(z500_ylim), title_text="Z500 anom. (dam)",
                        showticklabels=True, ticks="outside", automargin=True,
                        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
                        zeroline=True, zerolinecolor="rgba(0,0,0,0.45)",
                    )
                    fig_z.update_xaxes(
                        dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y",
                        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
                        ticks="outside", automargin=True,
                    )
                    fig_z.update_layout(
                        **plotly_typography(), hovermode="x", height=280,
                        template="plotly_white", margin=dict(t=10, b=40), showlegend=False,
                    )
                    st_plotly_press(fig_z, f"meteogram_z500_{met_active_epoch}")
                st.markdown("<div class='atmopulse-meteo-yearly-gap'></div>", unsafe_allow_html=True)
                st_plotly_press(build_yearly_extremes_chart(lat_target, lon_target, met_active_epoch, col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), f"yearly_{met_active_epoch}")
