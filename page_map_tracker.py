"""
AtmoPulse Map Tracker Page (page_map_tracker.py)

Extracted from app.py so the Map Tracker's own blocking calls (reference
climatology NetCDF open, Europe border GeoJSON fetch, synoptic field load)
only ever run when a user actually selects the Map Tracker tab — never on
app cold-start for Welcome/Legal/Methods, and never for Meteogram/Wavogram.
"""

import html
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from backend_map_locations import build_location_label_grid, build_country_weight_grid
from backend_narrative import EPOCH_LABELS
from atmopulse_theme import ATMOPULSE_BRAND, ATMOPULSE_FONTS, legend_badge_style
from config import (
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    LAYOUT_OPACITY,
    LAYOUT_SWIPE,
    AIFS_TXTN_WARNING,
    TOP10_GRID_VERSION,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    is_daily_map_view,
)
from backend_analytics import compute_map_footprint, calculate_top10
from frontend_plots import _render_synoptic_map, get_cached_baseline_map, build_opacity_slider_map, render_swipe_compare_map
from backend_io import (
    load_reference_climatology,
    fetch_cached_synoptic_data,
    get_persistence_arrays,
    synoptic_source_mtime,
    _load_persistence_daily_series,
)
from backend_maps import synoptic_vars_for_map
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


_FOOTPRINT_TIER_ORDER = ("moderate", "strong", "extreme", "record")
_FOOTPRINT_TIER_TITLES = {"moderate": "Moderate", "strong": "Strong", "extreme": "Extreme", "record": "Record"}
_WARM_TIER_HINT = {"moderate": "P75", "strong": "P90", "extreme": "P95", "record": "all-time"}
_COLD_TIER_HINT = {"moderate": "P25", "strong": "P10", "extreme": "P5", "record": "all-time"}


def _render_html(body: str) -> None:
    html_fn = getattr(st, "html", None)
    if callable(html_fn):
        html_fn(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def _chip_style(direction: str, tier: str) -> str:
    """Legend colours with inherited sentence font size (not the 12px badge size)."""
    return (
        legend_badge_style(direction, tier)
        .replace("font-size:12px;", "font-size:inherit;")
        .replace(f" font-weight:{ATMOPULSE_FONTS['ui_weight']};", " font-weight:inherit;")
    )


def _phrase_chip(direction: str, tier: str, text: str) -> str:
    return (
        f'<span class="atmopulse-narrative-chip atmopulse-sev-{direction}-{tier}" '
        f'style="{_chip_style(direction, tier)}">{html.escape(text)}</span>'
    )


def _trend_word(new: float, old: float) -> str:
    if new > old:
        return "amplified"
    if new < old:
        return "reduced"
    return "unchanged"


def _trend_clause(new: float, old: float) -> str:
    word = _trend_word(new, old)
    if word == "unchanged":
        return f"unchanged at {new:.1f}%"
    return f"{word} to {new:.1f}%"


def _daily_legend_html(top10_threshold: str) -> str:
    if "Extreme" in top10_threshold:
        hi_w, hi_c = "extreme", "extreme"
    elif "Strong" in top10_threshold:
        hi_w, hi_c = "strong", "strong"
    elif "Moderate" in top10_threshold:
        hi_w, hi_c = "moderate", "moderate"
    else:
        hi_w, hi_c = "record", "record"

    def s(side, level):
        return legend_badge_style(side, level, highlight=(level == (hi_w if side == "warm" else hi_c)))

    return (
        f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
        f"style='margin-bottom: 12px; white-space: nowrap;'>"
        f"<b>Legend.</b> "
        f"Warm: <span style='{s('warm', 'moderate')}'>Moderate</span> "
        f"<span style='{s('warm', 'strong')}'>Strong</span> "
        f"<span style='{s('warm', 'extreme')}'>Extreme</span> "
        f"<span style='{s('warm', 'record')}'>Record</span>"
        f"<span style='padding-left: 12px;'>Cold:</span> "
        f"<span style='{s('cold', 'moderate')}'>Moderate</span> "
        f"<span style='{s('cold', 'strong')}'>Strong</span> "
        f"<span style='{s('cold', 'extreme')}'>Extreme</span> "
        f"<span style='{s('cold', 'record')}'>Record</span>"
        f"</div>"
    )


def _direction_labels(tier: str) -> tuple[str, str]:
    if tier == "record":
        return "all-time warm record", "all-time cold record"
    return (
        f"{tier} warm ({_WARM_TIER_HINT[tier]})",
        f"{tier} cold ({_COLD_TIER_HINT[tier]})",
    )


def _banner_wrap(inner: str) -> str:
    bg = ATMOPULSE_BRAND["nav_bg"]
    fg = ATMOPULSE_BRAND["text_on_light"]
    font = ATMOPULSE_FONTS["outfit_css"]
    weight = ATMOPULSE_FONTS["ui_weight"]
    return (
        f"<div class='atmopulse-narrative-banner' style='"
        f"background-color:{bg}; color:{fg}; padding:0.75rem 1rem; "
        f"border-radius:0.5rem; font-family:{font}; font-weight:{weight}; "
        f"line-height:1.55; margin:0 0 0.5rem 0;'>"
        f"{inner}</div>"
    )


def _warm_clause(tier: str, pct: float) -> str:
    warm_lbl, _ = _direction_labels(tier)
    return _phrase_chip(
        "warm", tier,
        f"{pct:.1f}% of Europe is experiencing {warm_lbl} anomalies",
    )


def _cold_clause(tier: str, pct: float) -> str:
    _, cold_lbl = _direction_labels(tier)
    return _phrase_chip(
        "cold", tier,
        f"{pct:.1f}% is experiencing {cold_lbl} anomalies",
    )


def _single_footprint_banner(footprint: dict, active_tier: str, baseline_label: str) -> str:
    warm_pct = footprint[active_tier]["warm_pct"]
    cold_pct = footprint[active_tier]["cold_pct"]
    return _banner_wrap(
        f"Based on the {html.escape(baseline_label)} baseline, "
        f"{_warm_clause(active_tier, warm_pct)} and "
        f"{_cold_clause(active_tier, cold_pct)}."
    )


def _compare_footprint_banner(footprint_a: dict, footprint_b: dict, active_tier: str) -> str:
    wa = footprint_a[active_tier]["warm_pct"]
    ca = footprint_a[active_tier]["cold_pct"]
    wb = footprint_b[active_tier]["warm_pct"]
    cb = footprint_b[active_tier]["cold_pct"]
    epoch_a = html.escape(EPOCH_LABELS["A"])
    epoch_b = html.escape(EPOCH_LABELS["B"])
    return _banner_wrap(
        f"Relative to the historical {epoch_a} baseline, "
        f"{_warm_clause(active_tier, wa)} and {_cold_clause(active_tier, ca)}. "
        f"Under the recent {epoch_b} climate state, the warm footprint is "
        f"{_phrase_chip('warm', active_tier, _trend_clause(wb, wa))} "
        f"and the cold footprint is "
        f"{_phrase_chip('cold', active_tier, _trend_clause(cb, ca))}."
    )


def _delta_cell(new: float, old: float) -> str:
    return f"{new - old:+.1f}%"


def _baseline_expert_df(footprint: dict, footprint_a: dict | None = None, footprint_b: dict | None = None) -> pd.DataFrame:
    show_delta = footprint_a is not None and footprint_b is not None
    rows = []
    for t in _FOOTPRINT_TIER_ORDER:
        row = {
            "Severity": _FOOTPRINT_TIER_TITLES[t],
            "Warm": f"{footprint[t]['warm_pct']:.1f}%",
        }
        if show_delta:
            row["Δ Warm"] = (
                "—" if t == "record"
                else _delta_cell(footprint_b[t]["warm_pct"], footprint_a[t]["warm_pct"])
            )
        row["Cold"] = f"{footprint[t]['cold_pct']:.1f}%"
        if show_delta:
            row["Δ Cold"] = (
                "—" if t == "record"
                else _delta_cell(footprint_b[t]["cold_pct"], footprint_a[t]["cold_pct"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_expert_footprint(footprint: dict, footprint_a: dict | None = None, footprint_b: dict | None = None) -> None:
    df = _baseline_expert_df(footprint, footprint_a, footprint_b)
    column_config = {"Severity": st.column_config.TextColumn("Severity (cumulative)", width="small")}
    for col in df.columns:
        if col == "Severity":
            continue
        column_config[col] = st.column_config.TextColumn(col, width="small")
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )


def render_map_tracker(map_var_code, view_mode, persist_metric, top10_threshold, toggles, target_date, default_date):
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        st.error("Reference Climatology missing or corrupted! Please rebuild.")
        st.stop()
    # get_europe_borders_trace() is now resolved INSIDE get_cached_baseline_map
    # (Schritt C: a Plotly trace can't be a cache-data key), so it is no
    # longer fetched here directly.

    # Map Layout is a core viewing control (not an expert-only toggle) -> render
    # it globally for Standard and Expert users alike.
    map_layout = st.radio(
        "Map Layout:",
        (LAYOUT_SIDE_BY_SIDE, LAYOUT_FLICKER, LAYOUT_OPACITY, LAYOUT_SWIPE),
        horizontal=True,
    )
    if map_layout == LAYOUT_FLICKER:
        flicker_epoch = st.radio("Select Reference Period:", ("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), horizontal=True, index=1)

    if not is_daily_map_view(view_mode):
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

    def _render_impact_table(title: str, df, impact_col: str) -> None:
        st.markdown(_top10_header_html(title), unsafe_allow_html=True)
        if df.empty:
            st.caption("No countries affected.")
            return
        st.dataframe(
            df,
            column_config={
                "Country": st.column_config.TextColumn("Country", width="small"),
                impact_col: st.column_config.ProgressColumn(
                    impact_col, format="%.1f%%", min_value=0, max_value=100, width="small"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    def render_top10_period(df_h, df_c, period_label=None):
        if period_label:
            st.markdown(f"**{period_label}**")
        _render_impact_table("Top 10 Countries – Warm Impact", df_h, "Warm Impact (%)")
        _render_impact_table("Top 10 Countries – Cold Impact", df_c, "Cold Impact (%)")

    def render_top10_tables(df_h, df_c):
        render_top10_period(df_h, df_c)

    aifs_txtn_blocked = is_aifs_model() and map_var_code in ("TX", "TN")
    aifs_hatch_blocked = is_aifs_model() and bool(toggles.get("hatching"))
    if aifs_txtn_blocked or aifs_hatch_blocked:
        st.warning(AIFS_TXTN_WARNING)
    if not aifs_txtn_blocked:
        if aifs_hatch_blocked:
            toggles = {**toggles, "hatching": False}
        try:
            target_date_str = target_date.strftime('%Y-%m-%d')
            anchor_date_str = default_date.strftime('%Y-%m-%d')
            forecast_model = selected_forecast_model()
            # Schritt C: same mtime feeds fetch_cached_synoptic_data (the
            # data itself), compute_map_footprint/calculate_top10 (the
            # narrative/tables, which key on _map_phys_data being underscore-
            # excluded from their own cache), and get_cached_baseline_map
            # (the figure) — one fresh forecast download invalidates all
            # three consistently instead of only some of them.
            source_mtime = synoptic_source_mtime(target_date_str, forecast_model=forecast_model)
            with st.spinner("Loading synoptic fields..."):
                map_phys_data, map_time_meta = fetch_cached_synoptic_data(
                    target_date_str, anchor_date_str,
                    forecast_model=forecast_model,
                    needed_vars=synoptic_vars_for_map(map_var_code, toggles, view_mode),
                    source_mtime=source_mtime,
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
                footprint_a = footprint_b = footprint_single = None
                if is_daily_map_view(view_mode):
                    # Cumulative severity ladder: Moderate INCLUDES Strong/Extreme/Record,
                    # Strong INCLUDES Extreme/Record, Extreme INCLUDES Record, Record is exclusive.
                    # Active analysis level drives which single tier the sentence narrates.
                    active_tier = {
                        "Moderate": "moderate", "Strong": "strong",
                        "Extreme": "extreme", "All-Time Record": "record",
                    }.get(top10_threshold, "strong")

                    if map_layout == LAYOUT_FLICKER:
                        active_epoch = "A" if "A" in flicker_epoch else "B"
                        footprint_single = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            active_epoch, map_var_code, anchor_date_str=anchor_date_str,
                            source_mtime=source_mtime,
                        )
                        if footprint_single:
                            _render_html(_single_footprint_banner(
                                footprint_single, active_tier, EPOCH_LABELS[active_epoch],
                            ))
                    else:
                        footprint_a = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "A", map_var_code, anchor_date_str=anchor_date_str,
                            source_mtime=source_mtime,
                        )
                        footprint_b = compute_map_footprint(
                            ref_clim, map_phys_data, target_date_str,
                            st.session_state.toggles_warm, st.session_state.toggles_cold,
                            "B", map_var_code, anchor_date_str=anchor_date_str,
                            source_mtime=source_mtime,
                        )
                        if footprint_a and footprint_b:
                            _render_html(_compare_footprint_banner(
                                footprint_a, footprint_b, active_tier,
                            ))

                    st.markdown(_daily_legend_html(top10_threshold), unsafe_allow_html=True)

                # Schritt C: hashable-only front door for build_baseline_map.
                # Same date_str/toggles/source_mtime -> a Layout-radio rerun
                # (e.g. Side-by-Side <-> Opacity) hits this cache instead of
                # rebuilding the figure/hovertemplate from scratch; a new
                # forecast download (source_mtime changes) or an actual
                # toggle/date change still rebuilds it.
                def _cached_map(baseline_type, full_width=False):
                    return get_cached_baseline_map(
                        target_date_str, baseline_type, map_var_code, view_mode,
                        persist_metric, top10_threshold,
                        tuple(sorted(st.session_state.toggles_warm.items())),
                        tuple(sorted(st.session_state.toggles_cold.items())),
                        frozenset(name for name, active in toggles.items() if active),
                        source_mtime, forecast_model,
                        full_width=full_width, anchor_date_str=anchor_date_str,
                        _ref_data=ref_clim, _map_phys_data=map_phys_data,
                    )

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    fig_a = _cached_map("A")
                    fig_b = _cached_map("B")
                    with st.container(key="atmopulse_map_columns"):
                        mc1, mc2 = st.columns(2, gap="small")
                        with mc1:
                            _render_synoptic_map(fig_a, "Historical Baseline (1961-1990)", "map_a")
                            if is_expert_mode() and is_daily_map_view(view_mode) and footprint_a:
                                _render_expert_footprint(footprint_a, footprint_a, footprint_b)
                            df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_a, df_c_a)
                        with mc2:
                            _render_synoptic_map(fig_b, "Recent Baseline (1996-2025)", "map_b")
                            if is_expert_mode() and is_daily_map_view(view_mode) and footprint_b:
                                _render_expert_footprint(footprint_b, footprint_a, footprint_b)
                            df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_b, df_c_b)
                elif map_layout in (LAYOUT_OPACITY, LAYOUT_SWIPE):
                    fig_a = _cached_map("A", full_width=True)
                    fig_b = _cached_map("B", full_width=True)
                    if map_layout == LAYOUT_SWIPE:
                        render_swipe_compare_map(fig_a, fig_b)
                    else:
                        _render_synoptic_map(
                            build_opacity_slider_map(fig_a, fig_b),
                            "Opacity Slider Compare: Historical Baseline (1961-1990) ↔ Recent Baseline (1996-2025)",
                            "map_opacity",
                            bottom_margin=60,
                        )
                        st.caption("Drag the slider under the map to cross-fade between the two reference periods.")
                    with st.container(key="atmopulse_map_columns"):
                        map_col1, map_col2 = st.columns(2, gap="small")
                        with map_col1:
                            if is_expert_mode() and is_daily_map_view(view_mode) and footprint_a:
                                _render_expert_footprint(footprint_a, footprint_a, footprint_b)
                            df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_a, df_c_a, "Historical Baseline (1961–1990)")
                        with map_col2:
                            if is_expert_mode() and is_daily_map_view(view_mode) and footprint_b:
                                _render_expert_footprint(footprint_b, footprint_a, footprint_b)
                            df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=default_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_b, df_c_b, "Recent Baseline (1996–2025)")
                else:
                    ep_sel = "A" if "A" in flicker_epoch else "B"
                    flicker_title = "Historical Baseline (1961-1990)" if ep_sel == "A" else "Recent Baseline (1996-2025)"
                    _render_synoptic_map(
                        _cached_map(ep_sel, full_width=True),
                        flicker_title,
                        "map_flicker",
                    )
                    if is_expert_mode() and is_daily_map_view(view_mode) and footprint_single:
                        _render_expert_footprint(footprint_single)
                    df_h, df_c = calculate_top10(
                        ref_clim, map_phys_data, target_date,
                        st.session_state.toggles_warm, st.session_state.toggles_cold,
                        view_mode, persist_metric, top10_threshold,
                        ep_sel, map_var_code, anchor_date=default_date,
                        _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid,
                        source_mtime=source_mtime,
                    )
                    render_top10_tables(df_h, df_c)
        except Exception as e: 
            st.error(f"Error loading maps: {e}")
