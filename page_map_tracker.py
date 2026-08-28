"""
AtmoPulse Map Tracker Page (page_map_tracker.py)

Extracted from app.py so the Map Tracker's own blocking calls (reference
climatology NetCDF open, Europe border GeoJSON fetch, synoptic field load)
only ever run when a user actually selects the Map Tracker tab — never on
app cold-start for Welcome/Legal/Methods, and never for Meteogram/Wavogram.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from backend_map_locations import build_location_label_grid, build_country_weight_grid
from backend_narrative import EPOCH_LABELS
from atmopulse_theme import legend_badge_style
from config import (
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    LAYOUT_OPACITY,
    AIFS_TXTN_WARNING,
    TOP10_GRID_VERSION,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    is_daily_map_view,
)
from backend_analytics import compute_map_footprint, calculate_top10
from frontend_plots import _render_synoptic_map, build_baseline_map, build_opacity_slider_map
from backend_io import (
    load_reference_climatology,
    fetch_cached_synoptic_data,
    get_persistence_arrays,
    _load_persistence_daily_series,
)
from frontend_widgets import _top10_header_html


@st.cache_resource(show_spinner=False)
def get_europe_borders_trace():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
    try:
        data = requests.get(url, timeout=10).json()
        x, y = [], []
        for feature in data['features']:
            geom = feature.get('geometry')
            if not geom: continue
            if geom['type'] == 'Polygon':
                for poly in geom['coordinates']:
                    for p in poly: x.append(p[0]); y.append(p[1])
                    x.append(None); y.append(None)
            elif geom['type'] == 'MultiPolygon':
                for multi in geom['coordinates']:
                    for poly in multi:
                        for p in poly: x.append(p[0]); y.append(p[1])
                        x.append(None); y.append(None)
        return go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=1.0), hoverinfo='skip', showlegend=False)
    except: 
        return None


@st.cache_data(show_spinner=False)
def get_map_location_labels(lons_tuple, lats_tuple):
    return build_location_label_grid(np.array(lons_tuple), np.array(lats_tuple))


@st.cache_data(show_spinner=False)
def get_country_weight_grid(lons_tuple, lats_tuple, _version=TOP10_GRID_VERSION):
    return build_country_weight_grid(np.array(lons_tuple), np.array(lats_tuple))


def render_map_tracker(map_var_code, view_mode, persist_metric, top10_threshold, toggles, target_date, default_date):
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        st.error("Reference Climatology missing or corrupted! Please rebuild.")
        st.stop()
    border_trace = get_europe_borders_trace()

    # Map Layout is a core viewing control (not an expert-only toggle) -> render
    # it globally for Standard and Expert users alike.
    map_layout = st.radio("Map Layout:", (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER, LAYOUT_OPACITY), horizontal=True)
    if map_layout == LAYOUT_FLICKER:
        flicker_epoch = st.radio("Select Reference Period:", ("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), horizontal=True, index=1)

    if is_daily_map_view(view_mode):
        if "Extreme" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme", highlight=True)
            s_rec_h = legend_badge_style("warm", "record", highlight=True)
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme", highlight=True)
            s_rec_c = legend_badge_style("cold", "record", highlight=True)
        elif "Strong" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong", highlight=True)
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record")
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong", highlight=True)
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record")
        elif "Moderate" in top10_threshold:
            s_p75 = legend_badge_style("warm", "moderate", highlight=True)
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record")
            s_p25 = legend_badge_style("cold", "moderate", highlight=True)
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record")
        else:
            s_p75 = legend_badge_style("warm", "moderate")
            s_p90 = legend_badge_style("warm", "strong")
            s_p95 = legend_badge_style("warm", "extreme")
            s_rec_h = legend_badge_style("warm", "record", highlight=True)
            s_p25 = legend_badge_style("cold", "moderate")
            s_p10 = legend_badge_style("cold", "strong")
            s_p5 = legend_badge_style("cold", "extreme")
            s_rec_c = legend_badge_style("cold", "record", highlight=True)
        st.markdown(
            f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
            f"style='margin-bottom: 6px; white-space: nowrap;'>"
            f"<b>Legend.</b> "
            f"Warm: <span style='{s_p75}'>Moderate</span> <span style='{s_p90}'>Strong</span> "
            f"<span style='{s_p95}'>Extreme</span> <span style='{s_rec_h}'>Record</span>"
            f"<span style='padding-left: 12px;'>Cold:</span> "
            f"<span style='{s_p25}'>Moderate</span> <span style='{s_p10}'>Strong</span> "
            f"<span style='{s_p5}'>Extreme</span> <span style='{s_rec_c}'>Record</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        _, pers_meta = _load_persistence_daily_series(
            (target_date - pd.Timedelta(days=65)).strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            default_date.strftime('%Y-%m-%d'),
            forecast_model=selected_forecast_model(),
        ) or (None, {})
        eff_end = pers_meta.get("effective_end")
        if pers_meta.get("uses_ifs") and eff_end is not None:
            gap_note = " Gaps between archive and IFS can interrupt streaks." if pers_meta.get("has_gap") else ""
            st.info(
                f"**Persistence Mode Active:** Hybrid ERA5 archive + IFS forecast. "
                f"Showing consecutive days with target percentiles, ending on {eff_end.strftime('%d.%m.%Y')} "
                f"(requested: {target_date.strftime('%d.%m.%Y')}). Recent days use IFS HRES.{gap_note}"
            )
        elif eff_end is not None and eff_end < target_date.normalize():
            st.info(
                f"**Persistence Mode Active:** Showing consecutive days with target percentiles, "
                f"ending on {eff_end.strftime('%d.%m.%Y')} (last available data; requested {target_date.strftime('%d.%m.%Y')})."
            )
        else:
            st.info(f"**Persistence Mode Active:** Showing number of consecutive days with target percentiles, ending on {target_date.strftime('%d.%m.%Y')}.")

    def render_top10_period(df_h, df_c, period_label=None):
        if period_label:
            st.markdown(f"**{period_label}**")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(_top10_header_html("Top 10 Countries – Warm Impact"), unsafe_allow_html=True)
            if not df_h.empty:
                st.dataframe(
                    df_h,
                    column_config={
                        "Warm Impact (%)": st.column_config.ProgressColumn(
                            "Warm Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")
        with t_col2:
            st.markdown(_top10_header_html("Top 10 Countries – Cold Impact"), unsafe_allow_html=True)
            if not df_c.empty:
                st.dataframe(
                    df_c,
                    column_config={
                        "Cold Impact (%)": st.column_config.ProgressColumn(
                            "Cold Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")

    def render_top10_tables(df_h, df_c):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(_top10_header_html("Top 10 Countries – Warm Impact"), unsafe_allow_html=True)
            if not df_h.empty:
                st.dataframe(
                    df_h,
                    column_config={
                        "Warm Impact (%)": st.column_config.ProgressColumn(
                            "Warm Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")
        with t_col2:
            st.markdown(_top10_header_html("Top 10 Countries – Cold Impact"), unsafe_allow_html=True)
            if not df_c.empty:
                st.dataframe(
                    df_c,
                    column_config={
                        "Cold Impact (%)": st.column_config.ProgressColumn(
                            "Cold Impact (%)", format="%.1f%%", min_value=0, max_value=100, width="medium"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No countries affected.")

    aifs_txtn_blocked = is_aifs_model() and map_var_code in ("TX", "TN")
    aifs_hatch_blocked = is_aifs_model() and bool(toggles.get("hatching"))
    if aifs_txtn_blocked or aifs_hatch_blocked:
        st.warning(AIFS_TXTN_WARNING)
    if not aifs_txtn_blocked:
        if aifs_hatch_blocked:
            toggles = {**toggles, "hatching": False}
        try:
            with st.spinner("Loading synoptic fields..."):
                map_phys_data, map_time_meta = fetch_cached_synoptic_data(
                    target_date.strftime('%Y-%m-%d'), default_date.strftime('%Y-%m-%d'),
                    forecast_model=selected_forecast_model(),
                )
            if not map_time_meta.get("available"):
                st.warning(
                    f"No synoptic data for **{target_date.strftime('%d.%m.%Y')}**. "
                    "The IFS HRES forecast may not yet cover this date — try a lower Forecast Offset."
                )
            else:
                if not map_time_meta.get("temps_available", True):
                    st.warning(
                        f"Temperature extremes (TX/TN/TG) are missing for **{target_date.strftime('%d.%m.%Y')}** "
                        "in the ERA5 archive, so the colour overlay cannot be drawn. "
                        "Synoptic contours are shown where available."
                    )
                if is_daily_map_view(view_mode):
                    # Cumulative severity ladder: Moderate INCLUDES Strong/Extreme/Record,
                    # Strong INCLUDES Extreme/Record, Extreme INCLUDES Record, Record is exclusive.
                    footprint_tier_order = ("moderate", "strong", "extreme", "record")
                    footprint_tier_titles = {"moderate": "Moderate", "strong": "Strong", "extreme": "Extreme", "record": "Record"}
                    footprint_tier_label = {
                        "moderate": "moderate (P75/25)", "strong": "strong (P90/10)",
                        "extreme": "extreme (P95/5)", "record": "all-time record",
                    }
                    # Active analysis level (Moderate/Strong/Extreme/All-Time Record radio)
                    # drives which single tier the TEXT sentence narrates.
                    active_tier = {
                        "Moderate": "moderate", "Strong": "strong",
                        "Extreme": "extreme", "All-Time Record": "record",
                    }.get(top10_threshold, "strong")
                    layout_choice = map_layout  # the exact Map Layout radio state — "Side-by-Side Compare" | "Single Map Flicker" | "Opacity Slider Compare"
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    anchor_date_str = default_date.strftime('%Y-%m-%d')

                    if layout_choice == LAYOUT_FLICKER:
                        # --- SINGLE BASELINE STATE: one map on screen -> one set of cumulative percentages ---
                        active_epoch = "A" if "A" in flicker_epoch else "B"
                        active_baseline = EPOCH_LABELS[active_epoch]
                        footprint_single = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            active_epoch, map_var_code, anchor_date_str=anchor_date_str,
                        )
                        if footprint_single:
                            # TEXT SENTENCE — always rendered, Standard and Expert alike.
                            single_val = footprint_single[active_tier]["total_pct"]
                            st.markdown(
                                f"**Based on the {active_baseline} baseline**, **{single_val:.1f}%** of Europe is experiencing "
                                f"{footprint_tier_label[active_tier]} anomalies."
                            )
                            # DETAILED BREAKDOWN — Expert Mode only, placed directly below the text.
                            if is_expert_mode():
                                metric_cols = st.columns(4)
                                for m_col, tier in zip(metric_cols, footprint_tier_order):
                                    with m_col:
                                        st.metric(footprint_tier_titles[tier], f"{footprint_single[tier]['total_pct']:.1f}%")
                    else:
                        # --- COMPARE STATE (Side-by-Side or Opacity Compare): both maps on screen -> both baselines' cumulative percentages ---
                        footprint_a = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "A", map_var_code, anchor_date_str=anchor_date_str,
                        )
                        footprint_b = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "B", map_var_code, anchor_date_str=anchor_date_str,
                        )
                        if footprint_a and footprint_b:
                            # TEXT SENTENCE — always rendered, Standard and Expert alike.
                            hist_val = footprint_a[active_tier]["total_pct"]
                            rec_val = footprint_b[active_tier]["total_pct"]
                            trend_word = "amplified" if rec_val > hist_val else "reduced" if rec_val < hist_val else "unchanged"
                            st.markdown(
                                f"Relative to the historical **{EPOCH_LABELS['A']}** baseline, **{hist_val:.1f}%** of Europe is experiencing "
                                f"{footprint_tier_label[active_tier]} anomalies. Under the recent **{EPOCH_LABELS['B']}** climate state, "
                                f"this footprint is {trend_word} to **{rec_val:.1f}%**."
                            )
                            # DETAILED TABLE (1961-1990 vs 1996-2025 vs Delta) — Expert Mode only,
                            # placed directly below the text.
                            if is_expert_mode():
                                table_rows = "\n".join(
                                    f"| {footprint_tier_titles[t]} | {footprint_a[t]['total_pct']:.1f}% | "
                                    f"{footprint_b[t]['total_pct']:.1f}% | {footprint_b[t]['total_pct'] - footprint_a[t]['total_pct']:+.1f} pp |"
                                    for t in footprint_tier_order
                                )
                                st.markdown(
                                    f"| Severity (cumulative) | {EPOCH_LABELS['A']} | {EPOCH_LABELS['B']} | Δ |\n"
                                    f"|---|---|---|---|\n{table_rows}"
                                )

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    fig_a = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    fig_b = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    with st.container(key="atmopulse_map_columns"):
                        mc1, mc2 = st.columns(2, gap="small")
                        with mc1:
                            _render_synoptic_map(fig_a, "Historical Baseline (1961-1990)", "map_a")
                        with mc2:
                            _render_synoptic_map(fig_b, "Recent Baseline (1996-2025)", "map_b")
                    map_col1, map_col2 = st.columns(2)
                    with map_col1:
                        df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_a, df_c_a)
                    with map_col2:
                        df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_b, df_c_b)
                elif map_layout == LAYOUT_OPACITY:
                    fig_a = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, full_width=True, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    fig_b = build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, full_width=True, border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays)
                    _render_synoptic_map(
                        build_opacity_slider_map(fig_a, fig_b),
                        "Opacity Slider Compare: Historical Baseline (1961-1990) ↔ Recent Baseline (1996-2025)",
                        "map_opacity",
                        bottom_margin=60,
                    )
                    st.caption("Drag the slider under the map to cross-fade between the two reference periods. Zoom and tooltips stay interactive.")
                    map_col1, map_col2 = st.columns(2)
                    with map_col1:
                        df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_a, df_c_a, "Historical Baseline (1961–1990)")
                    with map_col2:
                        df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid)
                        render_top10_period(df_h_b, df_c_b, "Recent Baseline (1996–2025)")
                else:
                    ep_sel = "A" if "A" in flicker_epoch else "B"
                    flicker_title = "Historical Baseline (1961-1990)" if ep_sel == "A" else "Recent Baseline (1996-2025)"
                    _render_synoptic_map(
                        build_baseline_map(
                            ref_clim, map_phys_data, target_date,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            toggles, view_mode, persist_metric, top10_threshold,
                            ep_sel, map_var_code, anchor_date=default_date,
                            full_width=True,
                            border_trace=border_trace, get_map_location_labels=get_map_location_labels, get_persistence_arrays=get_persistence_arrays,
                        ),
                        flicker_title,
                        "map_flicker",
                    )
                    _, table_col, _ = st.columns([1, 2, 1])
                    with table_col:
                        df_h, df_c = calculate_top10(
                            ref_clim, map_phys_data, target_date,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            view_mode, persist_metric, top10_threshold,
                            ep_sel, map_var_code, anchor_date=default_date,
                            _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid,
                        )
                        render_top10_tables(df_h, df_c)
        except Exception as e: 
            st.error(f"Error loading maps: {e}")
