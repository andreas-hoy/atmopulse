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
    LAYOUT_SINGLE_MAP,
    LAYOUT_SIDE_BY_SIDE,
    LAYOUT_SWIPE,
    LAYOUT_OPACITY,
    COMPARE_EPOCHS,
    COMPARE_DATES,
    AIFS_TXTN_WARNING,
    FORECAST_MODEL_IFS,
    PERSISTENCE_LOOKBACK_PAD,
    PERSISTENCE_MAX_DAYS,
    TOP10_GRID_VERSION,
    analog_calendar_date,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    is_daily_map_view,
    STANDARD_DEFAULTS,
)
from backend_analytics import compute_map_footprint, calculate_top10
from frontend_plots import (
    _MAP_OVERLAY_TOGGLES,
    _render_synoptic_map,
    get_cached_baseline_map,
    build_opacity_slider_map,
    map_export_title,
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
from backend_maps import synoptic_vars_for_map, latest_era5_archive_date
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
        f"font-size:{_BANNER_FONT_PX}px; line-height:1.55; margin:0 0 1.15rem 0;'>"
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


def _dates_footprint_banner(
    footprint_live: dict, footprint_arch: dict, active_tier: str,
    baseline_label: str, live_date, arch_date,
) -> str:
    colder = baseline_label.startswith("1961")
    lead = (
        f"Against the colder historical baseline ({html.escape(baseline_label)})"
        if colder
        else f"Against the warmer recent baseline ({html.escape(baseline_label)})"
    )
    live_s = pd.Timestamp(live_date).strftime("%d.%m.%Y")
    arch_s = pd.Timestamp(arch_date).strftime("%d.%m.%Y")
    wl = footprint_live[active_tier]["warm_pct"]
    cl = footprint_live[active_tier]["cold_pct"]
    wa = footprint_arch[active_tier]["warm_pct"]
    ca = footprint_arch[active_tier]["cold_pct"]
    return _banner_wrap(
        f"{lead}, on {html.escape(live_s)} {_europe_clause(active_tier, wl, cl)}. "
        f"On {html.escape(arch_s)} {_europe_clause(active_tier, wa, ca)}."
    )


def _shift_map_compare_date(days: int) -> None:
    current = st.session_state.get("map_compare_date")
    if current is None:
        return
    min_d = pd.Timestamp(1940, 1, 1).date()
    max_d = latest_era5_archive_date().date()
    new_d = (pd.Timestamp(current) + pd.Timedelta(days=int(days))).date()
    st.session_state.map_compare_date = min(max(new_d, min_d), max_d)


def map_compare_prev_day():
    _shift_map_compare_date(-1)


def map_compare_next_day():
    _shift_map_compare_date(1)


_MAP_DATE_MIN = pd.Timestamp(1940, 1, 1).date()


def _map_date_max():
    return latest_era5_archive_date().date()


def _shift_map_archive_date(days: int) -> None:
    current = st.session_state.get("map_archive_date")
    if current is None:
        return
    new_d = (pd.Timestamp(current) + pd.Timedelta(days=int(days))).date()
    st.session_state.map_archive_date = min(max(new_d, _MAP_DATE_MIN), _map_date_max())


def map_archive_prev_day():
    _shift_map_archive_date(-1)


def map_archive_next_day():
    _shift_map_archive_date(1)


def _render_day_buttons(prev_key, next_key, on_prev, on_next, current, max_d):
    c1, c2, _rest = st.columns([1, 1, 1.35], gap="small")
    with c1:
        st.button(
            "← Day before", key=prev_key, on_click=on_prev,
            use_container_width=True, disabled=current <= _MAP_DATE_MIN,
        )
    with c2:
        st.button(
            "Day after →", key=next_key, on_click=on_next,
            use_container_width=True, disabled=current >= max_d,
        )


def _render_archive_date_picker():
    """Archive calendar above the map it drives. Key matches the sidebar seed."""
    max_d = _map_date_max()
    st.date_input(
        "Archive date:",
        min_value=_MAP_DATE_MIN,
        max_value=max_d,
        key="map_archive_date",
        format="DD.MM.YYYY",
        help=HELP["map_archive_clock"],
    )
    st.session_state["_active_map_archive_date"] = st.session_state.map_archive_date
    _render_day_buttons(
        "map_archive_prev_btn", "map_archive_next_btn",
        map_archive_prev_day, map_archive_next_day,
        st.session_state.map_archive_date, max_d,
    )


def _render_compare_date_picker():
    max_d = _map_date_max()
    st.date_input(
        "Compare with:",
        min_value=_MAP_DATE_MIN,
        max_value=max_d,
        key="map_compare_date",
        format="DD.MM.YYYY",
        help=HELP["map_compare_date"],
    )
    _render_day_buttons(
        "map_compare_prev_btn", "map_compare_next_btn",
        map_compare_prev_day, map_compare_next_day,
        st.session_state.map_compare_date, max_d,
    )


def _render_map_date_controls(map_is_archive, compare_dates, map_layout, live_label, arch_label):
    """Date clocks above the figures they belong to.

    Two dates: one clock per side (left is Live or an archive day, right is
    always ERA5). Reference periods share a single archive day. Flicker,
    opacity, and swipe keep both clocks above the one map.
    """
    if compare_dates:
        left, right = st.columns(2, gap="small")
        with left:
            if map_is_archive:
                _render_archive_date_picker()
            elif live_label:
                st.markdown(f"**{live_label}**", help=HELP["map_live_clock"])
        with right:
            _render_compare_date_picker()
        if map_layout == LAYOUT_SINGLE_MAP and live_label and arch_label:
            if st.session_state.get("map_flicker_date") not in (live_label, arch_label):
                st.session_state.map_flicker_date = live_label
            st.radio(
                "Show date:",
                (live_label, arch_label),
                horizontal=True,
                key="map_flicker_date",
            )
        return
    if map_is_archive:
        if map_layout != LAYOUT_SINGLE_MAP:
            st.caption("Same day for both panels.")
        col, _rest = st.columns(2, gap="small")
        with col:
            _render_archive_date_picker()


def _fmt_area_pct(pct: float) -> str:
    return f"{pct:.1f}"


def _fmt_change_cell(new: float, old: float, *, dash: bool = False) -> str:
    if dash:
        return "—"
    return f"{new - old:+.1f}"


def _severity_level_rows(values_fn, active_tier: str, *, with_change: bool, show_right: bool | None = None) -> str:
    if show_right is None:
        show_right = with_change
    rows = []
    for t in _FOOTPRINT_TIER_ORDER:
        old, new = values_fn(t)
        active = " atmopulse-sev-row-active" if t == active_tier else ""
        rec = t == "record"
        cells = (
            f"<th scope='row'>{_FOOTPRINT_TIER_TITLES[t]}</th>"
            f"<td>{_fmt_area_pct(old)}</td>"
        )
        if show_right:
            cells += f"<td>{_fmt_area_pct(new)}</td>"
        if with_change:
            cells += f"<td>{_fmt_change_cell(new, old, dash=rec)}</td>"
        rows.append(f"<tr class='atmopulse-sev-level{active}'>{cells}</tr>")
    return "".join(rows)


def _severity_table_html(headers: tuple[str, ...], body: str) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return (
        f"<table class='atmopulse-sev-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _severity_pair_html(warm_table: str, cold_table: str) -> str:
    return (
        "<div class='atmopulse-sev-pair'>"
        "<div class='atmopulse-sev-block'>"
        "<div class='atmopulse-sev-heading atmopulse-sev-heading-warm'>Warm:</div>"
        f"{warm_table}</div>"
        "<div class='atmopulse-sev-block'>"
        "<div class='atmopulse-sev-heading atmopulse-sev-heading-cold'>Cold:</div>"
        f"{cold_table}</div>"
        "</div>"
    )


def _severity_compare_html(
    footprint_a: dict, footprint_b: dict, active_tier: str,
    headers: tuple[str, ...] | None = None, *, with_change: bool = True,
) -> str:
    headers = headers or ("", "1961–1990", "1996–2025", "Change")

    def warm(t):
        return footprint_a[t]["warm_pct"], footprint_b[t]["warm_pct"]

    def cold(t):
        return footprint_a[t]["cold_pct"], footprint_b[t]["cold_pct"]

    return _severity_pair_html(
        _severity_table_html(
            headers,
            _severity_level_rows(warm, active_tier, with_change=with_change, show_right=True),
        ),
        _severity_table_html(
            headers,
            _severity_level_rows(cold, active_tier, with_change=with_change, show_right=True),
        ),
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
    headers: tuple[str, ...] | None = None,
    with_change: bool = True,
    help_key: str = "europe_share_table",
) -> None:
    # Native Streamlit `help=` (same tooltip as Map view), not a HTML title= bubble.
    if compare and footprint_a is not None and footprint_b is not None:
        body = _severity_compare_html(
            footprint_a, footprint_b, active_tier,
            headers=headers, with_change=with_change,
        )
    elif footprint_single is not None:
        body = _severity_single_html(footprint_single, active_tier)
    else:
        return
    st.markdown("**Share of Europe**", help=HELP[help_key])
    st.markdown(body, unsafe_allow_html=True)


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

    # Compare axis is orthogonal to Map Layout: A/B climatology at one date,
    # or two dates at one climatology. Layout then applies to that pair.
    if "map_compare_axis" not in st.session_state:
        st.session_state.map_compare_axis = STANDARD_DEFAULTS["map_compare"]
    elif st.session_state.map_compare_axis not in (COMPARE_EPOCHS, COMPARE_DATES):
        # Session still holding the old "Dates (Live vs Archive)" label.
        st.session_state.map_compare_axis = COMPARE_DATES
    _map_layouts = (LAYOUT_SINGLE_MAP, LAYOUT_SIDE_BY_SIDE, LAYOUT_SWIPE, LAYOUT_OPACITY)
    if st.session_state.get("map_layout") not in _map_layouts:
        st.session_state.map_layout = LAYOUT_SINGLE_MAP
    top1, top2 = st.columns([1.05, 2.15])
    with top1:
        compare_axis = st.radio(
            "Compare:",
            (COMPARE_EPOCHS, COMPARE_DATES),
            horizontal=True,
            key="map_compare_axis",
            help=HELP["map_compare_axis"],
        )
    with top2:
        map_layout = st.radio(
            "Map Layout:",
            _map_layouts,
            horizontal=True,
            key="map_layout",
        )
    compare_dates = compare_axis == COMPARE_DATES
    date_epoch = "B"
    live_label = arch_label = None
    compare_date = None
    if compare_dates:
        _max_archive = latest_era5_archive_date()
        _min_archive = pd.Timestamp(1940, 1, 1)
        analog = analog_calendar_date(
            target_date, years_back=1, min_date=_min_archive, max_date=_max_archive,
        )
        if "map_compare_date" not in st.session_state:
            st.session_state.map_compare_date = analog.date()
        else:
            st.session_state.map_compare_date = min(
                st.session_state.map_compare_date, _max_archive.date(),
            )
        if is_expert_mode():
            if "map_date_epoch" not in st.session_state:
                st.session_state.map_date_epoch = epoch_period_label("B")
            date_epoch = epoch_from_label(st.radio(
                "Colour against:",
                (epoch_period_label("A"), epoch_period_label("B")),
                horizontal=True,
                key="map_date_epoch",
                help=HELP["map_date_epoch"],
            ))
        compare_date = pd.Timestamp(st.session_state.map_compare_date)
        left_when = pd.Timestamp(target_date).strftime("%d.%m.%Y")
        right_when = compare_date.strftime("%d.%m.%Y")
        # Map titles and the flicker switch show the calendar day only.
        # Identical days need a side marker so the radio options stay unique.
        if left_when == right_when:
            live_label = f"{left_when} (left)"
            arch_label = f"{right_when} (right)"
        else:
            live_label = left_when
            arch_label = right_when

    _render_map_date_controls(
        map_is_archive, compare_dates, map_layout, live_label, arch_label,
    )

    flicker_epoch = None
    flicker_date_choice = None
    if map_layout == LAYOUT_SINGLE_MAP:
        if compare_dates:
            flicker_date_choice = st.session_state.get("map_flicker_date", live_label)
        else:
            if "map_flicker_epoch" not in st.session_state:
                st.session_state.map_flicker_epoch = epoch_period_label("B")
            flicker_epoch = st.radio(
                "Select Reference Period:",
                (epoch_period_label("A"), epoch_period_label("B")),
                horizontal=True,
                key="map_flicker_epoch",
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
        if compare_dates and compare_date is not None:
            st.caption(
                f"Right panel persistence ends on {compare_date.strftime('%d.%m.%Y')} (ERA5 only)."
            )

    def _render_impact_table(title: str, df, impact_col: str) -> None:
        st.markdown(_top10_header_html(title), unsafe_allow_html=True)
        if df.empty:
            st.caption("No European countries affected.")
            return
        st.dataframe(
            df,
            column_config={
                "Country": st.column_config.TextColumn("Country", width=120),
                impact_col: st.column_config.ProgressColumn(
                    "Area %", format="%.1f%%", min_value=0, max_value=100, width=160
                ),
            },
            hide_index=True,
            use_container_width=False,
        )

    def render_top10_period(df_h, df_c, period_label=None):
        if period_label:
            st.markdown(f"**{period_label}**")
        wcol, ccol = st.columns(2, gap="medium")
        with wcol:
            _render_impact_table("Warm", df_h, "Warm Impact (%)")
        with ccol:
            _render_impact_table("Cold", df_c, "Cold Impact (%)")

    def render_top10_tables(df_h, df_c):
        # Same adjacency as one panel of the side-by-side compare: Warm and
        # Cold share the left half instead of sitting at opposite page edges.
        pair, _rest = st.columns([1, 1], gap="small")
        with pair:
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
            needed_vars = synoptic_vars_for_map(map_var_code, toggles, view_mode)

            def _load_fields(date, anchor, model):
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
                anchor_str = pd.Timestamp(anchor).strftime("%Y-%m-%d")
                mtime = synoptic_source_mtime(date_str, forecast_model=model)
                phys, meta = fetch_cached_synoptic_data(
                    date_str, anchor_str,
                    forecast_model=model,
                    needed_vars=needed_vars,
                    source_mtime=mtime,
                )
                return date_str, anchor_str, mtime, phys, meta

            def _unavailable_warning(date, *, archive: bool) -> None:
                if archive:
                    st.warning(
                        f"Archive data is not yet available for **{pd.Timestamp(date).strftime('%d.%m.%Y')}**."
                    )
                else:
                    st.warning(
                        f"No synoptic data for **{pd.Timestamp(date).strftime('%d.%m.%Y')}**. "
                        "The IFS HRES forecast may not yet cover this date — try a lower Forecast Offset."
                    )

            def _temps_warning(date) -> None:
                st.warning(
                    f"Temperature extremes (TX/TN/TG) are missing for **{pd.Timestamp(date).strftime('%d.%m.%Y')}** "
                    "in the ERA5 archive, so the colour overlay cannot be drawn. "
                    "Synoptic contours are shown where available."
                )

            spinner_label = (
                "Loading synoptic fields (two dates)..."
                if compare_dates else "Loading synoptic fields..."
            )
            with st.spinner(spinner_label):
                target_date_str, anchor_date_str, source_mtime, map_phys_data, map_time_meta = _load_fields(
                    target_date, anchor_date, forecast_model,
                )
                arch_pack = None
                if compare_dates and compare_date is not None:
                    arch_pack = _load_fields(compare_date, compare_date, FORECAST_MODEL_IFS)

            live_ok = bool(map_time_meta.get("available"))
            arch_ok = bool(arch_pack and arch_pack[4].get("available")) if compare_dates else True
            if not live_ok:
                _unavailable_warning(target_date, archive=map_is_archive)
            if compare_dates and not arch_ok:
                _unavailable_warning(compare_date, archive=True)
            if not live_ok or (compare_dates and not arch_ok):
                return

            if not map_time_meta.get("temps_available", True):
                _temps_warning(target_date)
            if compare_dates and arch_pack is not None and not arch_pack[4].get("temps_available", True):
                _temps_warning(compare_date)

            if compare_dates:
                arch_date_str, arch_anchor_str, arch_mtime, arch_phys, _arch_meta = arch_pack
                left_title, right_title = live_label, arch_label
                left_epoch = right_epoch = date_epoch
                left_date, right_date = target_date, compare_date
                left_anchor, right_anchor = anchor_date, compare_date
                left_mtime, right_mtime = source_mtime, arch_mtime
                left_phys, right_phys = map_phys_data, arch_phys
                left_model, right_model = forecast_model, FORECAST_MODEL_IFS
            else:
                left_title, right_title = epoch_period_label("A"), epoch_period_label("B")
                left_epoch, right_epoch = "A", "B"
                left_date = right_date = target_date
                left_anchor = right_anchor = anchor_date
                left_mtime = right_mtime = source_mtime
                left_phys = right_phys = map_phys_data
                left_model = right_model = forecast_model
                arch_date_str = arch_anchor_str = None
                arch_mtime = None
                arch_phys = None

            footprint_a = footprint_b = footprint_single = None
            active_tier = "strong"
            if is_daily_map_view(view_mode):
                active_tier = {
                    "Moderate": "moderate", "Strong": "strong",
                    "Extreme": "extreme", "All-Time Record": "record",
                }.get(top10_threshold, "strong")

                def _fp(phys, date_str, epoch, anchor_str, mtime):
                    return compute_map_footprint(
                        ref_clim, phys, date_str,
                        st.session_state.toggles_warm, st.session_state.toggles_cold,
                        epoch, map_var_code, anchor_date_str=anchor_str,
                        source_mtime=mtime,
                    )

                if map_layout == LAYOUT_SINGLE_MAP:
                    if compare_dates:
                        use_live = flicker_date_choice == live_label
                        footprint_single = _fp(
                            map_phys_data if use_live else arch_phys,
                            target_date_str if use_live else arch_date_str,
                            date_epoch,
                            anchor_date_str if use_live else arch_anchor_str,
                            source_mtime if use_live else arch_mtime,
                        )
                        if footprint_single:
                            _render_html(_single_footprint_banner(
                                footprint_single, active_tier, EPOCH_LABELS[date_epoch],
                            ))
                    else:
                        active_epoch = epoch_from_label(flicker_epoch)
                        footprint_single = _fp(
                            map_phys_data, target_date_str, active_epoch,
                            anchor_date_str, source_mtime,
                        )
                        if footprint_single:
                            _render_html(_single_footprint_banner(
                                footprint_single, active_tier, EPOCH_LABELS[active_epoch],
                            ))
                else:
                    if compare_dates:
                        footprint_a = _fp(
                            map_phys_data, target_date_str, date_epoch,
                            anchor_date_str, source_mtime,
                        )
                        footprint_b = _fp(
                            arch_phys, arch_date_str, date_epoch,
                            arch_anchor_str, arch_mtime,
                        )
                        if footprint_a and footprint_b:
                            _render_html(_dates_footprint_banner(
                                footprint_a, footprint_b, active_tier,
                                EPOCH_LABELS[date_epoch], target_date, compare_date,
                            ))
                    else:
                        footprint_a = _fp(
                            map_phys_data, target_date_str, "A",
                            anchor_date_str, source_mtime,
                        )
                        footprint_b = _fp(
                            map_phys_data, target_date_str, "B",
                            anchor_date_str, source_mtime,
                        )
                        if footprint_a and footprint_b:
                            _render_html(_compare_footprint_banner(
                                footprint_a, footprint_b, active_tier,
                            ))

                if is_expert_mode():
                    sev_headers = None
                    sev_change = True
                    sev_help = "europe_share_table"
                    if compare_dates:
                        sev_headers = (
                            "",
                            pd.Timestamp(target_date).strftime("%d.%m.%Y"),
                            pd.Timestamp(compare_date).strftime("%d.%m.%Y"),
                        )
                        sev_change = False
                        sev_help = "europe_share_table_dates"
                    _render_expert_severity(
                        footprint_a=footprint_a, footprint_b=footprint_b,
                        footprint_single=footprint_single, active_tier=active_tier,
                        compare=map_layout != LAYOUT_SINGLE_MAP,
                        headers=sev_headers, with_change=sev_change,
                        help_key=sev_help,
                    )
                st.markdown(_daily_legend_html(top10_threshold), unsafe_allow_html=True)

            overlay_names = frozenset(
                name for name, active in toggles.items()
                if name in _MAP_OVERLAY_TOGGLES and active
            )
            warm_items = tuple(sorted(st.session_state.toggles_warm.items()))
            cold_items = tuple(sorted(st.session_state.toggles_cold.items()))
            spell_days = int(toggles.get("spell_days", 6))

            def _cached_map_for(date_str, epoch, mtime, model, phys, anchor_str, full_width=False):
                return get_cached_baseline_map(
                    date_str, epoch, map_var_code, view_mode,
                    persist_metric, top10_threshold,
                    warm_items, cold_items, overlay_names,
                    mtime, model,
                    full_width=full_width, anchor_date_str=anchor_str,
                    spell_days=spell_days,
                    _ref_data=ref_clim, _map_phys_data=phys,
                    _syn_clim=syn_clim,
                )

            def _title_for(panel, when):
                return map_export_title(
                    map_var_code, view_mode, panel, when, persist_metric,
                )

            def _top10(phys, date, epoch, anchor, mtime):
                return calculate_top10(
                    ref_clim, phys, date,
                    st.session_state.toggles_warm, st.session_state.toggles_cold,
                    view_mode, persist_metric, top10_threshold,
                    epoch, map_var_code, anchor_date=anchor,
                    _get_persistence_arrays=get_persistence_arrays,
                    _get_country_weight_grid=get_country_weight_grid,
                    source_mtime=mtime,
                )

            left_date_str = pd.Timestamp(left_date).strftime("%Y-%m-%d")
            right_date_str = pd.Timestamp(right_date).strftime("%Y-%m-%d")
            left_anchor_str = pd.Timestamp(left_anchor).strftime("%Y-%m-%d")
            right_anchor_str = pd.Timestamp(right_anchor).strftime("%Y-%m-%d")

            if map_layout == LAYOUT_SIDE_BY_SIDE:
                fig_a = _cached_map_for(
                    left_date_str, left_epoch, left_mtime, left_model, left_phys, left_anchor_str,
                )
                fig_b = _cached_map_for(
                    right_date_str, right_epoch, right_mtime, right_model, right_phys, right_anchor_str,
                )
                with st.container(key="atmopulse_split_map"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        _render_synoptic_map(
                            fig_a, left_title, "map_a",
                            export_title=_title_for(left_title, left_date),
                        )
                    with mc2:
                        _render_synoptic_map(
                            fig_b, right_title, "map_b",
                            export_title=_title_for(right_title, right_date),
                        )
                with st.container(key="atmopulse_split_map_tables"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        df_h_a, df_c_a = _top10(left_phys, left_date, left_epoch, left_anchor, left_mtime)
                        render_top10_period(df_h_a, df_c_a)
                    with mc2:
                        df_h_b, df_c_b = _top10(right_phys, right_date, right_epoch, right_anchor, right_mtime)
                        render_top10_period(df_h_b, df_c_b)
            elif map_layout in (LAYOUT_OPACITY, LAYOUT_SWIPE):
                fig_a = _cached_map_for(
                    left_date_str, left_epoch, left_mtime, left_model, left_phys, left_anchor_str,
                    full_width=True,
                )
                fig_b = _cached_map_for(
                    right_date_str, right_epoch, right_mtime, right_model, right_phys, right_anchor_str,
                    full_width=True,
                )
                if map_layout == LAYOUT_SWIPE:
                    if compare_dates:
                        swipe_title = (
                            f"Swipe: {left_title} (left) | {right_title} (right)"
                        )
                        swipe_help = "Drag the map or the slider to compare the two dates."
                    else:
                        swipe_title = (
                            "Swipe: Historical Reference Period (left) | "
                            "Recent Reference Period (right)"
                        )
                        swipe_help = (
                            "Drag the map or the slider: left is 1961–1990, right is 1996–2025."
                        )
                    render_swipe_compare_map(
                        fig_a, fig_b,
                        title=swipe_title,
                        help_text=swipe_help,
                        export_title_a=_title_for(left_title, left_date),
                        export_title_b=_title_for(right_title, right_date),
                    )
                else:
                    opacity_help = (
                        "Drag the slider under the map to cross-fade between the two dates."
                        if compare_dates else
                        "Drag the slider under the map to cross-fade between the two reference periods."
                    )
                    _render_synoptic_map(
                        build_opacity_slider_map(
                            fig_a, fig_b, label_a=left_title, label_b=right_title,
                        ),
                        f"Opacity: {left_title} ↔ {right_title}",
                        "map_opacity",
                        bottom_margin=60,
                        export_title=map_export_title(
                            map_var_code, view_mode,
                            f"{left_title} ↔ {right_title}",
                            left_date, persist_metric,
                        ),
                        help_text=opacity_help,
                    )
                with st.container(key="atmopulse_split_map_tables_overlay"):
                    map_col1, map_col2 = st.columns(2, gap="small")
                    with map_col1:
                        df_h_a, df_c_a = _top10(left_phys, left_date, left_epoch, left_anchor, left_mtime)
                        render_top10_period(df_h_a, df_c_a, left_title)
                    with map_col2:
                        df_h_b, df_c_b = _top10(right_phys, right_date, right_epoch, right_anchor, right_mtime)
                        render_top10_period(df_h_b, df_c_b, right_title)
            else:
                if compare_dates:
                    use_live = flicker_date_choice == live_label
                    flicker_title = live_label if use_live else arch_label
                    fig = _cached_map_for(
                        left_date_str if use_live else right_date_str,
                        date_epoch,
                        left_mtime if use_live else right_mtime,
                        left_model if use_live else right_model,
                        left_phys if use_live else right_phys,
                        left_anchor_str if use_live else right_anchor_str,
                        full_width=True,
                    )
                    _render_synoptic_map(
                        fig, flicker_title, "map_flicker",
                        export_title=_title_for(flicker_title, left_date if use_live else right_date),
                    )
                    df_h, df_c = _top10(
                        left_phys if use_live else right_phys,
                        left_date if use_live else right_date,
                        date_epoch,
                        left_anchor if use_live else right_anchor,
                        left_mtime if use_live else right_mtime,
                    )
                    render_top10_tables(df_h, df_c)
                else:
                    ep_sel = epoch_from_label(flicker_epoch)
                    flicker_title = epoch_period_label(ep_sel)
                    _render_synoptic_map(
                        _cached_map_for(
                            target_date_str, ep_sel, source_mtime, forecast_model,
                            map_phys_data, anchor_date_str, full_width=True,
                        ),
                        flicker_title,
                        "map_flicker",
                        export_title=_title_for(flicker_title, target_date),
                    )
                    df_h, df_c = _top10(
                        map_phys_data, target_date, ep_sel, anchor_date, source_mtime,
                    )
                    render_top10_tables(df_h, df_c)
        except Exception as e:
            st.error(f"Error loading maps: {e}")
