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

from backend_map_locations import EUROPE_BBOX, build_location_label_grid, build_country_weight_grid
from backend_narrative import EPOCH_LABELS, epoch_period_label, epoch_from_label
from atmopulse_theme import ATMOPULSE_BRAND, ATMOPULSE_FONTS, ATMOPULSE_OVERLAY, legend_badge_style
from labels import HELP
from config import (
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_FLICKER,
    LAYOUT_OPACITY,
    LAYOUT_SWIPE,
    AIFS_TXTN_WARNING,
    FORECAST_MODEL_IFS,
    PERSISTENCE_LOOKBACK_PAD,
    PERSISTENCE_MAX_DAYS,
    TOP10_GRID_VERSION,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    is_daily_map_view,
)
from backend_analytics import compute_map_footprint, calculate_top10
from frontend_plots import (
    _MAP_OVERLAY_TOGGLES,
    _render_synoptic_map,
    get_cached_baseline_map,
    build_opacity_slider_map,
    render_swipe_compare_map,
)
from backend_io import (
    load_reference_climatology,
    load_synoptic_climatology,
    fetch_cached_synoptic_data,
    get_persistence_arrays,
    synoptic_source_mtime,
    _load_persistence_daily_series,
)
from backend_maps import synoptic_vars_for_map
from frontend_widgets import _top10_header_html


_NE_GEOJSON = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
_NE_COAST = f"{_NE_GEOJSON}/ne_50m_coastline.geojson"
_NE_INLAND = f"{_NE_GEOJSON}/ne_50m_admin_0_boundary_lines_land.geojson"


def _geojson_line_xy(geom):
    """Append lon/lat vertices of a GeoJSON line geometry, None-separated."""
    x, y = [], []
    if not geom:
        return x, y
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    if gtype == "LineString":
        parts = [coords]
    elif gtype == "MultiLineString":
        parts = coords
    else:
        return x, y
    pad = 3.0
    lon0, lat0, lon1, lat1 = (
        EUROPE_BBOX[0] - pad, EUROPE_BBOX[1] - pad,
        EUROPE_BBOX[2] + pad, EUROPE_BBOX[3] + pad,
    )
    for line in parts:
        if not line:
            continue
        if not any(lon0 <= p[0] <= lon1 and lat0 <= p[1] <= lat1 for p in line):
            continue
        for p in line:
            x.append(p[0])
            y.append(p[1])
        x.append(None)
        y.append(None)
    return x, y


def _line_scatter(x, y, color, width):
    return go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip", showlegend=False,
    )


@st.cache_resource(show_spinner=False)
def get_europe_borders_trace(_style_version=3):
    """Inland country borders plus a stronger coastline (continent/ocean)."""
    try:
        inland = requests.get(_NE_INLAND, timeout=10).json()
        coast = requests.get(_NE_COAST, timeout=10).json()
        ix, iy, cx, cy = [], [], [], []
        for feature in inland.get("features", []):
            x, y = _geojson_line_xy(feature.get("geometry"))
            ix.extend(x)
            iy.extend(y)
        for feature in coast.get("features", []):
            x, y = _geojson_line_xy(feature.get("geometry"))
            cx.extend(x)
            cy.extend(y)
        traces = []
        if ix:
            traces.append(_line_scatter(ix, iy, ATMOPULSE_OVERLAY["border"], ATMOPULSE_OVERLAY["border_width"]))
        if cx:
            traces.append(_line_scatter(cx, cy, ATMOPULSE_OVERLAY["coast"], ATMOPULSE_OVERLAY["coast_width"]))
        return tuple(traces) if traces else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_map_location_labels(lons_tuple, lats_tuple):
    return build_location_label_grid(np.array(lons_tuple), np.array(lats_tuple))


@st.cache_data(show_spinner=False)
def get_country_weight_grid(lons_tuple, lats_tuple, _version=TOP10_GRID_VERSION):
    return build_country_weight_grid(np.array(lons_tuple), np.array(lats_tuple))


_FOOTPRINT_TIER_ORDER = ("moderate", "strong", "extreme", "record")
_FOOTPRINT_TIER_TITLES = {"moderate": "Moderate", "strong": "Strong", "extreme": "Extreme", "record": "Record"}


def _render_html(body: str) -> None:
    st.markdown(body, unsafe_allow_html=True)


_BANNER_FONT_PX = 15

# Copernicus C3S public percentile language (Climate Bulletins): P90/P10 =
# "much warmer/colder than average"; milder thresholds as "warmer/colder
# than average"; records as warm/cold records. P95/P5 have no C3S label, so
# "extremely warm/cold" is the media-facing step above P90.
_WARM_PLAIN = {
    "moderate": "warmer than average (P75)",
    "strong": "much warmer than average (P90)",
    "extreme": "extremely warm (P95)",
    "record": "at an all-time warm record",
}
_COLD_PLAIN = {
    "moderate": "colder than average (P25)",
    "strong": "much colder than average (P10)",
    "extreme": "extremely cold (P5)",
    "record": "at an all-time cold record",
}


def _chip_style(direction: str, tier: str) -> str:
    """Same 15px sentence size as the banner body (legend badges stay 12px)."""
    return (
        legend_badge_style(direction, tier)
        .replace("font-size:12px;", f"font-size:{_BANNER_FONT_PX}px;")
        .replace(f" font-weight:{ATMOPULSE_FONTS['ui_weight']};", " font-weight:inherit;")
        .replace("padding: 1px 6px;", "padding: 0 6px;")
        + " vertical-align: baseline;"
    )


def _phrase_chip(direction: str, tier: str, text: str) -> str:
    return (
        f'<span class="atmopulse-narrative-chip atmopulse-sev-{direction}-{tier}" '
        f'style="{_chip_style(direction, tier)}">{html.escape(text)}</span>'
    )


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
        f"style='margin-top: 18px; margin-bottom: 12px; white-space: nowrap;'>"
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


def _banner_wrap(inner: str) -> str:
    bg = ATMOPULSE_BRAND["nav_bg"]
    fg = ATMOPULSE_BRAND["text_on_light"]
    font = ATMOPULSE_FONTS["outfit_css"]
    weight = ATMOPULSE_FONTS["ui_weight"]
    return (
        f"<div class='atmopulse-narrative-banner' style='"
        f"background-color:{bg}; color:{fg}; padding:0.75rem 1rem; "
        f"border-radius:0.5rem; font-family:{font}; font-weight:{weight}; "
        f"font-size:{_BANNER_FONT_PX}px; line-height:1.55; margin:0 0 0.5rem 0;'>"
        f"{inner}</div>"
    )


def _europe_clause(tier: str, warm_pct: float, cold_pct: float) -> str:
    warm_txt = f"{warm_pct:.1f}% of Europe is {_WARM_PLAIN[tier]}"
    cold_txt = f"{cold_pct:.1f}% is {_COLD_PLAIN[tier]}"
    return (
        f"{_phrase_chip('warm', tier, warm_txt)} and "
        f"{_phrase_chip('cold', tier, cold_txt)}"
    )


def _single_footprint_banner(footprint: dict, active_tier: str, baseline_label: str) -> str:
    warm_pct = footprint[active_tier]["warm_pct"]
    cold_pct = footprint[active_tier]["cold_pct"]
    colder = baseline_label.startswith("1961")
    lead = (
        f"Against the colder historical baseline ({html.escape(baseline_label)})"
        if colder
        else f"Against the warmer recent baseline ({html.escape(baseline_label)})"
    )
    return _banner_wrap(
        f"{lead}, {_europe_clause(active_tier, warm_pct, cold_pct)}."
    )


def _compare_footprint_banner(footprint_a: dict, footprint_b: dict, active_tier: str) -> str:
    wa = footprint_a[active_tier]["warm_pct"]
    ca = footprint_a[active_tier]["cold_pct"]
    wb = footprint_b[active_tier]["warm_pct"]
    cb = footprint_b[active_tier]["cold_pct"]
    epoch_a = html.escape(EPOCH_LABELS["A"])
    epoch_b = html.escape(EPOCH_LABELS["B"])
    return _banner_wrap(
        f"Against the colder historical baseline ({epoch_a}), "
        f"{_europe_clause(active_tier, wa, ca)}. "
        f"Against the warmer recent baseline ({epoch_b}), "
        f"{_europe_clause(active_tier, wb, cb)}."
    )


def _fmt_area_pct(pct: float) -> str:
    return f"{pct:.1f}"


def _fmt_change_cell(new: float, old: float, *, dash: bool = False) -> str:
    if dash:
        return "—"
    return f"{new - old:+.1f}"


def _severity_level_rows(values_fn, active_tier: str, *, with_change: bool) -> str:
    rows = []
    for t in _FOOTPRINT_TIER_ORDER:
        old, new = values_fn(t)
        active = " atmopulse-sev-row-active" if t == active_tier else ""
        rec = t == "record"
        cells = (
            f"<th scope='row'>{_FOOTPRINT_TIER_TITLES[t]}</th>"
            f"<td>{_fmt_area_pct(old)}</td>"
        )
        if with_change:
            cells += f"<td>{_fmt_area_pct(new)}</td><td>{_fmt_change_cell(new, old, dash=rec)}</td>"
        rows.append(f"<tr class='atmopulse-sev-level{active}'>{cells}</tr>")
    return "".join(rows)


def _severity_table_html(headers: tuple[str, ...], body: str) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return (
        f"<table class='atmopulse-sev-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _severity_pair_html(warm_table: str, cold_table: str) -> str:
    tip = html.escape(HELP["europe_share_table"])
    return (
        "<div class='atmopulse-sev-pair'>"
        "<div class='atmopulse-sev-block'>"
        "<div class='atmopulse-sev-heading atmopulse-sev-heading-warm'>Warm:</div>"
        f"{warm_table}</div>"
        "<div class='atmopulse-sev-block'>"
        "<div class='atmopulse-sev-heading atmopulse-sev-heading-cold'>Cold:</div>"
        f"{cold_table}</div>"
        f"<span class='atmopulse-sev-help' title='{tip}'>?</span>"
        "</div>"
    )


def _severity_compare_html(footprint_a: dict, footprint_b: dict, active_tier: str) -> str:
    headers = ("", "1961–1990", "1996–2025", "Change")

    def warm(t):
        return footprint_a[t]["warm_pct"], footprint_b[t]["warm_pct"]

    def cold(t):
        return footprint_a[t]["cold_pct"], footprint_b[t]["cold_pct"]

    return _severity_pair_html(
        _severity_table_html(headers, _severity_level_rows(warm, active_tier, with_change=True)),
        _severity_table_html(headers, _severity_level_rows(cold, active_tier, with_change=True)),
    )


def _severity_single_html(footprint: dict, active_tier: str) -> str:
    headers = ("", "Area")

    def warm(t):
        return footprint[t]["warm_pct"], footprint[t]["warm_pct"]

    def cold(t):
        return footprint[t]["cold_pct"], footprint[t]["cold_pct"]

    return _severity_pair_html(
        _severity_table_html(headers, _severity_level_rows(warm, active_tier, with_change=False)),
        _severity_table_html(headers, _severity_level_rows(cold, active_tier, with_change=False)),
    )


def _render_expert_severity(
    *,
    footprint_a: dict | None,
    footprint_b: dict | None,
    footprint_single: dict | None,
    active_tier: str,
    compare: bool,
) -> None:
    # Markdown so page CSS (severity table) applies; st.html is isolated.
    if compare and footprint_a is not None and footprint_b is not None:
        st.markdown(_severity_compare_html(footprint_a, footprint_b, active_tier), unsafe_allow_html=True)
    elif footprint_single is not None:
        st.markdown(_severity_single_html(footprint_single, active_tier), unsafe_allow_html=True)


def render_map_tracker(
    map_var_code, view_mode, persist_metric, top10_threshold, toggles, target_date, default_date,
    map_is_archive=False, map_anchor_date=None,
):
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        st.error("Reference Climatology missing or corrupted! Please rebuild.")
        st.stop()
    syn_clim = load_synoptic_climatology()

    # Archive Date (map_is_archive=True): anchor is the archive day itself
    # (short existing pad_past/pad_future window around it, not the live
    # ~today window), and the Forecast Model choice is ignored — a
    # historical day always reads era5_master_daily_{year}.nc only, the
    # same way page_meteogram.py's Archive Year ignores it. Live keeps the
    # exact previous behaviour (anchor = default_date/"today").
    anchor_date = map_anchor_date if map_anchor_date is not None else default_date
    forecast_model = FORECAST_MODEL_IFS if map_is_archive else selected_forecast_model()
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
        flicker_epoch = st.radio(
            "Select Reference Period:",
            (epoch_period_label("A"), epoch_period_label("B")),
            horizontal=True,
            index=1,
        )

    if not is_daily_map_view(view_mode):
        _, pers_meta = _load_persistence_daily_series(
            (target_date - pd.Timedelta(days=PERSISTENCE_MAX_DAYS + PERSISTENCE_LOOKBACK_PAD)).strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            anchor_date.strftime('%Y-%m-%d'),
            forecast_model=forecast_model,
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
            st.caption("No European countries affected.")
            return
        st.dataframe(
            df,
            column_config={
                "Country": st.column_config.TextColumn("Country", width="small"),
                impact_col: st.column_config.ProgressColumn(
                    "Area %", format="%.1f%%", min_value=0, max_value=100, width="small"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    def render_top10_period(df_h, df_c, period_label=None):
        if period_label:
            st.markdown(f"**{period_label}**")
        wcol, ccol = st.columns(2, gap="small")
        with wcol:
            _render_impact_table("Warm", df_h, "Warm Impact (%)")
        with ccol:
            _render_impact_table("Cold", df_c, "Cold Impact (%)")

    def render_top10_tables(df_h, df_c):
        render_top10_period(df_h, df_c)

    # AIFS restrictions only make sense for the live forecast path — an
    # Archive Date always reads plain ERA5 (native TX/TN, no forecast model
    # involved at all), so it is exempt here (forecast_model is already
    # forced to IFS above for map_is_archive).
    aifs_txtn_blocked = (not map_is_archive) and is_aifs_model() and map_var_code in ("TX", "TN")
    aifs_hatch_blocked = (not map_is_archive) and is_aifs_model() and bool(toggles.get("hatching"))
    if aifs_txtn_blocked or aifs_hatch_blocked:
        st.warning(AIFS_TXTN_WARNING)
    if not aifs_txtn_blocked:
        if aifs_hatch_blocked:
            toggles = {**toggles, "hatching": False}
        try:
            target_date_str = target_date.strftime('%Y-%m-%d')
            anchor_date_str = anchor_date.strftime('%Y-%m-%d')
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
                if map_is_archive:
                    st.warning(
                        f"No ERA5 archive data for **{target_date.strftime('%d.%m.%Y')}** "
                        f"in era5_master_daily_{target_date.year}.nc — try another Archive date."
                    )
                else:
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
                active_tier = "strong"
                if is_daily_map_view(view_mode):
                    # Cumulative severity ladder: Moderate INCLUDES Strong/Extreme/Record,
                    # Strong INCLUDES Extreme/Record, Extreme INCLUDES Record, Record is exclusive.
                    # Active analysis level drives which single tier the sentence narrates.
                    active_tier = {
                        "Moderate": "moderate", "Strong": "strong",
                        "Extreme": "extreme", "All-Time Record": "record",
                    }.get(top10_threshold, "strong")

                    if map_layout == LAYOUT_FLICKER:
                        active_epoch = epoch_from_label(flicker_epoch)
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

                    if is_expert_mode():
                        _render_expert_severity(
                            footprint_a=footprint_a, footprint_b=footprint_b,
                            footprint_single=footprint_single, active_tier=active_tier,
                            compare=map_layout != LAYOUT_FLICKER,
                        )
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
                        frozenset(
                            name for name, active in toggles.items()
                            if name in _MAP_OVERLAY_TOGGLES and active
                        ),
                        source_mtime, forecast_model,
                        full_width=full_width, anchor_date_str=anchor_date_str,
                        spell_days=int(toggles.get("spell_days", 6)),
                        _ref_data=ref_clim, _map_phys_data=map_phys_data,
                        _syn_clim=syn_clim,
                    )

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    fig_a = _cached_map("A")
                    fig_b = _cached_map("B")
                    with st.container(key="atmopulse_map_columns"):
                        mc1, mc2 = st.columns(2, gap="small")
                        with mc1:
                            _render_synoptic_map(fig_a, epoch_period_label("A"), "map_a")
                        with mc2:
                            _render_synoptic_map(fig_b, epoch_period_label("B"), "map_b")
                    with st.container(key="atmopulse_map_tables"):
                        mc1, mc2 = st.columns(2, gap="small")
                        with mc1:
                            df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=anchor_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_a, df_c_a)
                        with mc2:
                            df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=anchor_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_b, df_c_b)
                elif map_layout in (LAYOUT_OPACITY, LAYOUT_SWIPE):
                    fig_a = _cached_map("A", full_width=True)
                    fig_b = _cached_map("B", full_width=True)
                    if map_layout == LAYOUT_SWIPE:
                        render_swipe_compare_map(fig_a, fig_b)
                    else:
                        _render_synoptic_map(
                            build_opacity_slider_map(fig_a, fig_b),
                            f"Opacity Slider Compare: {epoch_period_label('A')} ↔ {epoch_period_label('B')}",
                            "map_opacity",
                            bottom_margin=60,
                        )
                        st.caption("Drag the slider under the map to cross-fade between the two reference periods.")
                    with st.container(key="atmopulse_map_tables"):
                        map_col1, map_col2 = st.columns(2, gap="small")
                        with map_col1:
                            df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code, anchor_date=anchor_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_a, df_c_a, epoch_period_label("A"))
                        with map_col2:
                            df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code, anchor_date=anchor_date, _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid, source_mtime=source_mtime)
                            render_top10_period(df_h_b, df_c_b, epoch_period_label("B"))
                else:
                    ep_sel = epoch_from_label(flicker_epoch)
                    flicker_title = epoch_period_label(ep_sel)
                    _render_synoptic_map(
                        _cached_map(ep_sel, full_width=True),
                        flicker_title,
                        "map_flicker",
                    )
                    df_h, df_c = calculate_top10(
                        ref_clim, map_phys_data, target_date,
                        st.session_state.toggles_warm, st.session_state.toggles_cold,
                        view_mode, persist_metric, top10_threshold,
                        ep_sel, map_var_code, anchor_date=anchor_date,
                        _get_persistence_arrays=get_persistence_arrays, _get_country_weight_grid=get_country_weight_grid,
                        source_mtime=source_mtime,
                    )
                    render_top10_tables(df_h, df_c)
        except Exception as e: 
            st.error(f"Error loading maps: {e}")
