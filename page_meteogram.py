"""
AtmoPulse Point Meteogram Page (page_meteogram.py)

Extracted from app.py so the Meteogram's blocking calls (reference
climatology NetCDF open, live point-series extraction) only ever run when a
user actually selects the Point Meteogram tab with a resolved location —
never on app cold-start, and never for Welcome/Legal/Methods/Map Tracker/
Point Wavogram.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend_narrative import EPOCH_LABELS, classify_point_severity
from atmopulse_theme import (
    ATMOPULSE_COLD,
    ATMOPULSE_FONTS,
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
)
from backend_io import (
    load_reference_climatology,
    get_live_point_series,
    compute_point_thresholds,
    _load_point_archive_series,
)
from frontend_plots import get_meteogram_traces, build_yearly_extremes_chart


def render_meteogram(location, lat_target, lon_target, meteo_var, meteo_env, show_air_temp, show_app_temp, target_date):
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
        met_active_epoch = "A" if "A" in st.session_state.get("met_ep", "B (1996–2025)") else "B"
    if is_aifs_model() and meteo_var in ("Max Temp (TX)", "Min Temp (TN)"):
        st.warning(AIFS_TXTN_WARNING)
    else:
        with st.spinner("Fetching Meteogram data..."): 
            df_live = get_live_point_series(lat_target, lon_target, selected_forecast_model())
        if not df_live.empty:
            col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')

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
            elif 'TX' in df_indexed.columns and 'TN' in df_indexed.columns:
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
            condition_a = f"{cat_a} {dir_a}" if dir_a else "normal"
            condition_b = f"{cat_b} {dir_b}" if dir_b else "normal"  # e.g. "extreme warm", "moderate cold", "normal"

            if map_layout == LAYOUT_FLICKER:
                # Single-baseline state: narrate strictly the ACTIVE epoch (the one the
                # Flicker radio below is actually showing) — never a hardcoded baseline.
                cat_x, dir_x = (cat_a, dir_a) if met_active_epoch == "A" else (cat_b, dir_b)
                condition_x = condition_a if met_active_epoch == "A" else condition_b
                x_txt = "within its normal range" if condition_x == "normal" else f"{condition_x} conditions"
                condition_string = (
                    f"The area of {location.address} is currently experiencing {x_txt} relative to "
                    f"the {EPOCH_LABELS[met_active_epoch]} baseline."
                )
                condition_b = condition_x  # drives the styling block below
            else:
                # TASK 2: chronological order — historical (A, 1961-1990) baseline
                # narrated FIRST, recent (B, 1996-2025) baseline SECOND, matching the
                # left-to-right reading order of the UI.
                a_txt = "within its normal range" if condition_a == "normal" else f"{condition_a} conditions"
                b_txt = "within its normal range" if condition_b == "normal" else f"{condition_b} conditions"
                condition_string = (
                    f"The area of {location.address} is currently experiencing {a_txt} relative to the "
                    f"historical {EPOCH_LABELS['A']} baseline. Compared to the recent {EPOCH_LABELS['B']} "
                    f"climate state, this equates to {b_txt}."
                )

            # --- Conditional styling: driven by the ACTIVE (baseline-B) severity tier,
            # not a naive substring search of the sentence, so e.g. a compare-mode
            # sentence mentioning both "warm" and "cold" can't pick the wrong branch.
            if condition_b in ("record warm", "extreme warm", "strong warm"):
                st.error(condition_string, icon="🔥")
            elif condition_b == "moderate warm":
                st.warning(condition_string)
            elif condition_b in ("record cold", "extreme cold", "strong cold"):
                st.info(condition_string, icon="❄️")
            elif condition_b == "moderate cold":
                st.markdown(
                    f"""<div style="background-color:{ATMOPULSE_COLD['p25']}; color:#003554;
                    padding:0.75rem 1rem; border-radius:0.5rem;">🧊 {condition_string}</div>""",
                    unsafe_allow_html=True,
                )
            else:  # "normal" — no significant anomaly
                st.success(condition_string, icon="✅")

            t_arr = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values) / 2.0)
            global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
            tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)

            traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "A", show_air_temp, show_app_temp, meteo_env, meteo_var, current_condition=(cat_a, dir_a))
            traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "B", show_air_temp, show_app_temp, meteo_env, meteo_var, current_condition=(cat_b, dir_b))

            # TASK 2: compact HTML badge legend (Map Tracker style) for the 8
            # warm/cold severity tiers, replacing Plotly's own cluttered legend
            # for these fills. "Typical Range" (P25-P75) called out separately.
            _s_normal = "background-color:rgba(180,180,180,0.4); color:#333; padding:1px 6px; border-radius:3px; font-family:" + ATMOPULSE_FONTS['outfit_css'] + "; font-size:12px; font-weight:" + str(ATMOPULSE_FONTS['ui_weight']) + ";"
            st.markdown(
                f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
                f"style='margin-bottom: 6px; white-space: nowrap;'>"
                f"<b>Legend.</b> "
                f"<span style='{_s_normal}'>Typical Range</span>"
                f"<span style='padding-left: 12px;'>Warm:</span> "
                f"<span style='{legend_badge_style('warm', 'moderate')}'>Moderate</span> "
                f"<span style='{legend_badge_style('warm', 'strong')}'>Strong</span> "
                f"<span style='{legend_badge_style('warm', 'extreme')}'>Extreme</span> "
                f"<span style='{legend_badge_style('warm', 'record')}'>Record</span>"
                f"<span style='padding-left: 12px;'>Cold:</span> "
                f"<span style='{legend_badge_style('cold', 'moderate')}'>Moderate</span> "
                f"<span style='{legend_badge_style('cold', 'strong')}'>Strong</span> "
                f"<span style='{legend_badge_style('cold', 'extreme')}'>Extreme</span> "
                f"<span style='{legend_badge_style('cold', 'record')}'>Record</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            
            if map_layout == LAYOUT_SIDE_BY_SIDE:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), shared_yaxes=True)
                for trace in traces_a: 
                    fig.add_trace(trace, row=1, col=1)
                for trace in traces_b: 
                    fig.add_trace(trace, row=1, col=2)
                    
                fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                fig.update_yaxes(range=[global_min, global_max])
                fig.update_layout(**plotly_typography(), hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1: 
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A", meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)
                with c2: 
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "B", meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)
            else:
                # Same widget key ("met_ep") whose value we already read into
                # `met_active_epoch` above (before the narrative text was built) —
                # re-rendering it here just places it at its usual spot below the chart.
                flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="met_ep", index=1)
                met_active_epoch = "A" if "A" in flicker_epoch else "B"
                fig = go.Figure(data=traces_a if met_active_epoch == "A" else traces_b)
                fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8)
                fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y", showgrid=True, gridcolor=ATMOPULSE_OVERLAY['grid'])
                fig.update_yaxes(range=[global_min, global_max])
                fig.update_layout(**plotly_typography(), title=f"Reference Period {EPOCH_LABELS[met_active_epoch]}", hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, met_active_epoch, meteo_var != "Min Temp (TN)", _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series), use_container_width=True)
