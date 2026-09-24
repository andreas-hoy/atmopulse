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
    LAYOUT_SINGLE_CHART,
    LAYOUT_SIDE_BY_SIDE,
    COMPARE_EPOCHS,
    COMPARE_YEARS,
    STANDARD_DEFAULTS,
    AIFS_TXTN_WARNING,
    analog_calendar_date,
    analog_date_in_year,
    epoch_short_label,
    is_aifs_model,
    is_expert_mode,
    selected_forecast_model,
    show_expert,
    meteo_var_code,
)
from backend_io import (
    load_reference_climatology,
    load_synoptic_climatology,
    get_live_point_series,
    get_archive_year_options,
    get_archive_year_point_series,
    get_archive_window_point_series,
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


def _is_normal_condition(cat, direction) -> bool:
    return (not direction) or cat == "normal"


def _point_note(
    addr: str, verb: str, lead: str, cat, direction, *,
    when: str | None = None, force_lead: bool = False,
) -> str:
    """Short while the place is in the normal range; the baseline lead returns outside it.

    Side-by-side passes ``force_lead`` so each half still names its baseline.
    """
    phrase = point_condition_phrase(cat, direction)
    chip = _severity_phrase_html(cat, direction, phrase)
    if _is_normal_condition(cat, direction) and not force_lead:
        body = (
            f"On {when} the area of {addr} {verb} {chip}."
            if when else
            f"The area of {addr} {verb} {chip}."
        )
    elif when:
        body = f"{lead}, on {when} the area of {addr} {verb} {chip}."
    else:
        body = f"{lead}, the area of {addr} {verb} {chip}."
    return f"<div class='atmopulse-narrative-banner is-under-figure'>{body}</div>"


def _meteo_export_title(heading: str, epoch: str, when, panel: str | None = None) -> str:
    date_s = pd.Timestamp(when).strftime("%d.%m.%Y")
    epoch_s = epoch_period_label(epoch)
    if panel:
        return f"{heading} | {panel} | {epoch_s} | {date_s}"
    return f"{heading} | {epoch_s} | {date_s}"


def _classify_on_date(df, when, ref_clim, lat, lon, meteo_var, col_target, epoch):
    """Exact-calendar-day severity; missing row → (None, None), never a neighbour."""
    if df is None or df.empty or col_target not in df.columns:
        return None, None
    indexed = df.copy()
    indexed["Date"] = pd.to_datetime(indexed["Date"]).dt.tz_localize(None).dt.normalize()
    indexed = indexed.drop_duplicates(subset=["Date"]).set_index("Date").sort_index()
    when = pd.Timestamp(when).tz_localize(None).normalize()
    if when not in indexed.index:
        return None, None
    value_now = float(indexed.loc[when, col_target])
    if not np.isfinite(value_now):
        return None, None
    percentiles = compute_point_thresholds(ref_clim, lat, lon, when, meteo_var, epoch)
    return classify_point_severity(value_now, *percentiles)


def _seasonal_archive_frame(df_live, year, lat, lon):
    """Archive series covering the same month/day span as the Live window.

    Year alignment is anchored on the Live window's *end* date: a window
    spanning 18.09.2025–27.09.2026 compared with 2003 maps to
    18.09.2002–27.09.2003, so a December day in the first Live year stays
    in the first analog year.
    """
    dates = pd.to_datetime(df_live["Date"]).dt.tz_localize(None).dt.normalize()
    live_start, live_end = dates.min(), dates.max()
    year_shift = int(live_end.year) - int(year)
    analog_end = analog_calendar_date(live_end, years_back=year_shift)
    analog_start = analog_calendar_date(live_start, years_back=year_shift)
    analog_start = max(analog_start, pd.Timestamp(1940, 1, 1))
    df = get_archive_window_point_series(lat, lon, analog_start, analog_end)
    return df, analog_start, analog_end, year_shift


def _mark_chart_date(fig, vline_x) -> None:
    """Vertical date marker. Stops in the x-axis month band, above the legend."""
    if vline_x is None:
        return
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=vline_x, x1=vline_x,
        y0=-0.045, y1=1,
        line=dict(color="rgba(49, 51, 63, 0.75)", width=1.2, dash="dash"),
        layer="above",
    )


def _build_meteogram_panel(traces, title: str, y_min: float, y_max: float, vline_x) -> go.Figure:
    """One reference-period meteogram. Never a shared-y subplot — the right
    panel must keep its own ticks and can carry its own license/export."""
    fig = go.Figure(data=list(traces))
    _mark_chart_date(fig, vline_x)
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text="°C",
        showticklabels=True, ticks="outside", automargin=True,
        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
        zeroline=False, visible=True, side="left",
        showspikes=False,
    )
    fig.update_xaxes(
        dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y",
        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
        ticks="outside", automargin=True, showspikes=False,
    )
    fig.update_layout(
        **plotly_typography(), title=title,
        hovermode="x", height=520, template="plotly_white",
        margin=dict(t=56, b=40, l=56, r=16), showlegend=False,
    )
    return fig


def _build_z500_panel(traces, y_lim, vline_x) -> go.Figure:
    fig = go.Figure(data=list(traces))
    _mark_chart_date(fig, vline_x)
    fig.update_yaxes(
        range=list(y_lim),
        title_text="Z500 anom. (dam)",
        showticklabels=True, ticks="outside", automargin=True,
        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
        zeroline=True, zerolinecolor="rgba(0,0,0,0.45)",
        visible=True, side="left",
        showspikes=False,
    )
    fig.update_xaxes(
        dtick="M2", tickformat="%b\n%Y", hoverformat="%d.%m.%Y",
        showgrid=True, gridcolor=ATMOPULSE_OVERLAY["grid"],
        ticks="outside", automargin=True, showspikes=False,
    )
    fig.update_layout(
        **plotly_typography(), hovermode="x", height=280,
        template="plotly_white", margin=dict(t=10, b=40, l=56, r=16),
        showlegend=False,
    )
    return fig


def _render_html(body: str) -> None:
    """Prefer st.html so Streamlit does not strip chip markup."""
    html_fn = getattr(st, "html", None)
    if callable(html_fn):
        html_fn(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def _live_window_bounds():
    """Same rolling window as ``get_live_point_series`` (today+10, 375 days back)."""
    end = pd.Timestamp.utcnow().tz_localize(None).floor("D") + pd.Timedelta(days=10)
    start = end - pd.Timedelta(days=375)
    return start, end


def _analog_live_window(year: int):
    live_start, live_end = _live_window_bounds()
    year_shift = int(live_end.year) - int(year)
    analog_end = analog_calendar_date(live_end, years_back=year_shift)
    analog_start = analog_calendar_date(live_start, years_back=year_shift)
    analog_start = max(analog_start, pd.Timestamp(1940, 1, 1))
    return analog_start, analog_end


def _naive_day(when) -> pd.Timestamp:
    t = pd.Timestamp(when)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def _archive_day_available(df, when) -> bool:
    """True when ``when`` is a real ERA5/ERA5T day in ``df`` (finite TX/TN/TG)."""
    if df is None or df.empty or when is None or "Date" not in df.columns:
        return False
    when = _naive_day(when)
    indexed = df.copy()
    dates = pd.to_datetime(indexed["Date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    indexed["Date"] = dates.dt.normalize()
    indexed = indexed.drop_duplicates(subset=["Date"]).set_index("Date")
    if when not in indexed.index:
        return False
    temp_cols = [c for c in ("TX", "TN", "TG") if c in indexed.columns]
    if not temp_cols:
        return True
    row = indexed.loc[when]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return any(pd.notna(row[c]) for c in temp_cols)


def _vline_ms(df, when):
    if not _archive_day_available(df, when):
        return None
    return _naive_day(when).timestamp() * 1000


def _archive_plot_target(df, year):
    if df is not None and not df.empty and "Date" in df.columns:
        return _naive_day(pd.to_datetime(df["Date"]).max())
    return pd.Timestamp(year=int(year), month=12, day=31)


def _default_compare_year(left_choice, year_labels):
    """Partner year: one year before the left series when possible.

    Live vs Year defaults to the last *closed* year, not the in-progress
    current year (that would nearly duplicate the Live window without IFS).
    """
    if not year_labels:
        return None
    this_year = str(pd.Timestamp.utcnow().year)
    closed = [y for y in year_labels if y != this_year] or list(year_labels)
    if left_choice and left_choice != "Live" and left_choice in year_labels:
        i = year_labels.index(left_choice)
        if i > 0:
            return year_labels[i - 1]
        if i + 1 < len(year_labels):
            return year_labels[i + 1]
    return closed[-1]


def _meteo_compare_help(
    compare_years: bool, date_epoch: str, analog_start, analog_end,
    *, left_is_archive: bool = False, left_year=None, right_year=None,
) -> str:
    base = HELP["meteo_compare_axis"]
    if not compare_years:
        return base
    bits = [base, f"Both panels coloured against {epoch_period_label(date_epoch)}."]
    if left_is_archive and left_year is not None and right_year is not None:
        bits.append(
            f"Left is calendar year {left_year}; right is {right_year}. "
            "ERA5/ERA5T only — the current year stops at the last archive day. "
            "No forecast overlay."
        )
    elif analog_start is not None and analog_end is not None:
        bits.append(
            f"Archive window is the same season as Live "
            f"({analog_start.strftime('%d.%m.%Y')}–{analog_end.strftime('%d.%m.%Y')})."
        )
    return " ".join(bits)


def _temp_legend_html() -> str:
    """Colour key for the temperature chart. Sits under the chart, above the credit."""
    return (
        f"<div class='atmopulse-map-legend atmopulse-subsection-label' "
        f"style='margin-top: -2px; margin-bottom: 2px; white-space: nowrap;'>"
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
        f"</div>"
    )


def render_meteogram(location, lat_target, lon_target, meteo_var, meteo_env, target_date, meteo_spell=False):
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        st.error("Reference Climatology missing or corrupted! Please rebuild.")
        st.stop()

    # Archive Year: "Live" (default, index 0) keeps the rolling ~375-day
    # window + IFS/AIFS overlay exactly as before; any calendar year below
    # switches to a closed 1 Jan-31 Dec ERA5/ERA5T series at this point,
    # with no forecast involved. Session-state key is scoped to this page
    # only (met_archive_year) — the Offset slider / Jahresbalken / other
    # pages are untouched by this selection.
    archive_years = get_archive_year_options(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    year_labels = [str(y) for y in archive_years]
    if "pending_met_archive_year" in st.session_state:
        st.session_state["met_archive_year"] = st.session_state.pop("pending_met_archive_year")
        st.session_state.met_compare_axis = COMPARE_EPOCHS

    if "met_compare_axis" not in st.session_state:
        st.session_state.met_compare_axis = STANDARD_DEFAULTS["meteo_compare"]
    elif st.session_state.met_compare_axis not in (COMPARE_EPOCHS, COMPARE_YEARS):
        # Session still holding the old "Live vs Year" label.
        st.session_state.met_compare_axis = COMPARE_YEARS
    compare_years_pending = st.session_state.met_compare_axis == COMPARE_YEARS
    if compare_years_pending:
        left_pending = st.session_state.get("met_archive_year") or st.session_state.get(
            "_active_met_archive_year", "Live",
        )
        if "met_compare_year" not in st.session_state:
            partner = _default_compare_year(left_pending, year_labels)
            if partner is not None:
                st.session_state.met_compare_year = partner
        elif year_labels and st.session_state.met_compare_year not in year_labels:
            st.session_state.met_compare_year = year_labels[-1]
        help_year = st.session_state.get("met_compare_year")
        help_epoch = epoch_from_label(
            st.session_state.get("met_date_epoch", epoch_period_label("B")),
        )
        left_is_arch = bool(left_pending and left_pending != "Live")
        analog_s = analog_e = None
        if not left_is_arch and help_year and str(help_year) in year_labels:
            analog_s, analog_e = _analog_live_window(int(help_year))
        compare_help = _meteo_compare_help(
            True, help_epoch, analog_s, analog_e,
            left_is_archive=left_is_arch,
            left_year=left_pending if left_is_arch else None,
            right_year=help_year,
        )
    else:
        compare_help = HELP["meteo_compare_axis"]

    if "met_archive_year" not in st.session_state:
        st.session_state["met_archive_year"] = st.session_state.get(
            "_active_met_archive_year", "Live",
        )
    archive_choice = st.session_state["met_archive_year"]
    _chart_layouts = (LAYOUT_SINGLE_CHART, LAYOUT_SIDE_BY_SIDE)
    if st.session_state.get("met_layout") not in _chart_layouts:
        st.session_state.met_layout = LAYOUT_SINGLE_CHART
    _period_choice = (epoch_short_label("A"), epoch_short_label("B"))
    show_period = (
        st.session_state.get("met_compare_axis", COMPARE_EPOCHS) != COMPARE_YEARS
        and st.session_state.get("met_layout", LAYOUT_SINGLE_CHART) == LAYOUT_SINGLE_CHART
    )
    if show_period and st.session_state.get("met_ep") not in _period_choice:
        raw = st.session_state.get("met_ep", "")
        st.session_state.met_ep = (
            _period_choice[0] if epoch_from_label(raw or "B") == "A" else _period_choice[1]
        )
    show_archive_top = not compare_years_pending
    show_date_period = compare_years_pending
    if show_date_period:
        _date_period = (epoch_short_label("A"), epoch_short_label("B"))
        _date_now = st.session_state.get("met_date_epoch")
        if _date_now not in _date_period:
            st.session_state.met_date_epoch = (
                _date_period[0] if epoch_from_label(_date_now or "B") == "A" else _date_period[1]
            )
    n_top = 2 + int(show_period) + int(show_archive_top) + int(show_date_period)
    with st.container(key="met_top_controls"):
        top_cols = st.columns(n_top)
        with top_cols[0]:
            compare_axis = st.radio(
                "Compare:",
                (COMPARE_EPOCHS, COMPARE_YEARS),
                horizontal=True,
                key="met_compare_axis",
                help=compare_help,
            )
        series_opts = None
        if compare_years_pending and year_labels and st.session_state.get(
            "met_layout", LAYOUT_SINGLE_CHART,
        ) == LAYOUT_SINGLE_CHART:
            _y1 = st.session_state.get("met_archive_year") or st.session_state.get(
                "_active_met_archive_year", "Live",
            )
            _y2 = st.session_state.get("met_compare_year") or year_labels[-1]
            _left = f"Archive {int(_y1)}" if str(_y1) != "Live" else "Live"
            _right = f"Archive {int(_y2)}"
            if _left == _right:
                _left = f"{_left} (left)"
                _right = f"{_right} (right)"
            series_opts = (_left, _right)
            if st.session_state.get("met_year_pick") not in series_opts:
                st.session_state.met_year_pick = series_opts[0]
        with top_cols[1]:
            map_layout = st.radio(
                "Layout:",
                _chart_layouts,
                horizontal=True,
                key="met_layout",
            )
            if series_opts:
                st.radio(
                    "Show series:",
                    series_opts,
                    horizontal=True,
                    key="met_year_pick",
                )
        next_col = 2
        if show_period:
            with top_cols[next_col]:
                st.radio(
                    "Select Reference Period:",
                    _period_choice,
                    horizontal=True,
                    key="met_ep",
                )
            next_col += 1
        if show_date_period:
            with top_cols[next_col]:
                st.radio(
                    "Reference Period:",
                    _date_period,
                    horizontal=True,
                    key="met_date_epoch",
                    help=HELP["meteo_date_epoch"],
                )
            next_col += 1
        if show_archive_top:
            with top_cols[next_col]:
                archive_choice = st.selectbox(
                    "Archive Year:",
                    ["Live"] + year_labels,
                    key="met_archive_year",
                    help=HELP["meteo_archive_year"],
                )
    compare_years = compare_axis == COMPARE_YEARS

    is_archive = False
    archive_year = None
    compare_year = None
    date_epoch = "B"
    if compare_years:
        if not year_labels:
            st.warning("No complete ERA5 archive years are available to compare against.")
            return
        if "met_archive_year" not in st.session_state:
            st.session_state["met_archive_year"] = st.session_state.get(
                "_active_met_archive_year", "Live",
            )
        date_epoch = epoch_from_label(
            st.session_state.get("met_date_epoch", epoch_short_label("B"))
        )
        with st.container(key="met_year_row"):
            y1, y2, _year_pad = st.columns([1.15, 1.15, 2.7])
            with y1:
                archive_choice = st.selectbox(
                    "Year 1",
                    ["Live"] + year_labels,
                    key="met_archive_year",
                    help=HELP["meteo_archive_year"],
                )
            with y2:
                compare_year = int(st.selectbox(
                    "Year 2",
                    year_labels,
                    key="met_compare_year",
                    help=HELP["meteo_compare_year"],
                ))
            st.session_state["_active_met_archive_year"] = archive_choice
            is_archive = archive_choice != "Live"
            archive_year = int(archive_choice) if is_archive else None
            live_panel = f"Archive {archive_year}" if is_archive else "Live"
            year_panel = f"Archive {compare_year}"
            if live_panel == year_panel:
                live_panel = f"{live_panel} (left)"
                year_panel = f"{year_panel} (right)"
            if st.session_state.get("met_year_pick") not in (live_panel, year_panel):
                st.session_state.met_year_pick = live_panel
            if map_layout == LAYOUT_SINGLE_CHART:
                met_year_pick = st.session_state.get("met_year_pick", live_panel)
            else:
                met_year_pick = live_panel
    else:
        st.session_state["_active_met_archive_year"] = archive_choice
        is_archive = archive_choice != "Live"
        archive_year = int(archive_choice) if is_archive else None
        if map_layout == LAYOUT_SINGLE_CHART:
            met_active_epoch = epoch_from_label(st.session_state.get("met_ep", epoch_period_label("B")))
    live_blocked = (not is_archive) and is_aifs_model() and meteo_var_code(meteo_var) in ("TX", "TN")
    if live_blocked:
        # ERA5 (Archive Year) natively has TX/TN, so the AIFS warning only
        # ever applies to the Live/forecast path — including Live vs Year.
        st.warning(AIFS_TXTN_WARNING)
    elif compare_years:
        col_target = meteo_var_code(meteo_var)
        analog_start = analog_end = None
        year_shift = 0
        if is_archive:
            with st.spinner("Fetching Meteogram data (two archive years)..."):
                df_live = get_archive_year_point_series(lat_target, lon_target, archive_year)
                df_arch = get_archive_year_point_series(lat_target, lon_target, compare_year)
            left_when = analog_date_in_year(target_date, archive_year)
            analog_target = analog_date_in_year(target_date, compare_year)
            plot_left = _archive_plot_target(df_live, archive_year)
            plot_right = _archive_plot_target(df_arch, compare_year)
            left_empty_msg = (
                f"No ERA5 archive data available for {archive_year} at this location."
            )
            right_empty_msg = (
                f"No ERA5 archive data available for {compare_year} at this location."
            )
        else:
            with st.spinner("Fetching Meteogram data (Live + archive year)..."):
                df_live = get_live_point_series(lat_target, lon_target, selected_forecast_model())
                df_arch, analog_start, analog_end, year_shift = (
                    _seasonal_archive_frame(df_live, compare_year, lat_target, lon_target)
                    if not df_live.empty else (pd.DataFrame(), None, None, 0)
                )
            left_when = pd.Timestamp(target_date)
            analog_target = analog_calendar_date(target_date, years_back=year_shift)
            plot_left = left_when
            plot_right = analog_end if analog_end is not None else analog_target
            left_empty_msg = "No Live point series available at this location."
            if analog_start is not None and analog_end is not None:
                right_empty_msg = (
                    f"No ERA5 archive data for the {compare_year} season window "
                    f"({analog_start.strftime('%d.%m.%Y')}–{analog_end.strftime('%d.%m.%Y')}) "
                    "at this location."
                )
            else:
                right_empty_msg = (
                    f"No ERA5 archive data for {compare_year} at this location."
                )
        if df_live.empty:
            st.error(left_empty_msg)
        elif df_arch.empty:
            st.error(right_empty_msg)
        else:
            cat_live, dir_live = _classify_on_date(
                df_live, left_when, ref_clim, lat_target, lon_target, meteo_var, col_target, date_epoch,
            )
            cat_arch, dir_arch = _classify_on_date(
                df_arch, analog_target, ref_clim, lat_target, lon_target, meteo_var, col_target, date_epoch,
            )
            addr = html.escape(location.address)
            lead = html.escape(baseline_against_lead(date_epoch))
            live_s = pd.Timestamp(left_when).strftime("%d.%m.%Y")
            arch_s = analog_target.strftime("%d.%m.%Y")
            left_verb = "was" if is_archive else "is"
            narrative_html = None
            narrative_left = None
            narrative_right = None
            if is_archive:
                pass
            elif map_layout == LAYOUT_SINGLE_CHART:
                show_live = met_year_pick != year_panel
                if show_live and cat_live is not None:
                    narrative_html = _point_note(
                        addr, left_verb, lead, cat_live, dir_live, when=html.escape(live_s),
                    )
                elif (not show_live) and cat_arch is not None:
                    narrative_html = _point_note(
                        addr, "was", lead, cat_arch, dir_arch, when=html.escape(arch_s),
                    )
            elif cat_live is not None or cat_arch is not None:
                if cat_live is not None:
                    narrative_left = _point_note(
                        addr, left_verb, lead, cat_live, dir_live,
                        when=html.escape(live_s), force_lead=True,
                    )
                if cat_arch is not None:
                    narrative_right = _point_note(
                        addr, "was", lead, cat_arch, dir_arch,
                        when=html.escape(arch_s), force_lead=True,
                    )

            t_live = df_live[col_target].values if col_target in df_live.columns else np.full(len(df_live), np.nan)
            t_arch = df_arch[col_target].values if col_target in df_arch.columns else np.full(len(df_arch), np.nan)
            global_min = float(np.nanmin([np.nanmin(t_live), np.nanmin(t_arch)])) - 3
            global_max = float(np.nanmax([np.nanmax(t_live), np.nanmax(t_arch)])) + 3

            traces_live = get_meteogram_traces(
                df_live, ref_clim, lat_target, lon_target, plot_left, date_epoch, meteo_env, meteo_var,
                current_condition=(cat_live, dir_live),
            )
            # Archive series is ERA5 only: plot target is the last day so nothing is dotted as forecast.
            traces_arch = get_meteogram_traces(
                df_arch, ref_clim, lat_target, lon_target, plot_right, date_epoch, meteo_env, meteo_var,
                current_condition=(cat_arch, dir_arch),
            )

            show_z500 = show_expert("z500")
            syn_clim = load_synoptic_climatology() if show_z500 else None
            z500_live, z500_rng_live, z500_csv_live = get_z500_anomaly_traces(
                df_live, syn_clim, lat_target, lon_target, plot_left, date_epoch,
            ) if show_z500 else ([], None, None)
            z500_arch, z500_rng_arch, z500_csv_arch = get_z500_anomaly_traces(
                df_arch, syn_clim, lat_target, lon_target, plot_right, date_epoch,
            ) if show_z500 else ([], None, None)
            use_z500 = bool(z500_live and z500_arch)
            z500_ylim = None
            if use_z500 and z500_rng_live and z500_rng_arch:
                z500_ylim = (
                    min(z500_rng_live[0], z500_rng_arch[0]),
                    max(z500_rng_live[1], z500_rng_arch[1]),
                )
            vline_live = None if is_archive else _vline_ms(df_live, left_when)
            vline_arch = None
            title_live = f"{live_panel}<br>({live_s})"
            title_arch = f"{year_panel}<br>({arch_s})"

            heading = str(meteo_var)
            if map_layout == LAYOUT_SIDE_BY_SIDE:
                if narrative_left or narrative_right:
                    note_l, note_r = st.columns(2, gap="small")
                    with note_l:
                        if narrative_left:
                            st.markdown(narrative_left, unsafe_allow_html=True)
                    with note_r:
                        if narrative_right:
                            st.markdown(narrative_right, unsafe_allow_html=True)
                fig_live = _build_meteogram_panel(traces_live, title_live, global_min, global_max, vline_live)
                fig_arch = _build_meteogram_panel(traces_arch, title_arch, global_min, global_max, vline_arch)
                with st.container(key="atmopulse_split_meteo"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        st_plotly_press(
                            fig_live, "meteogram_historical",
                            csv_text=df_live.to_csv(index=False),
                            export_title=_meteo_export_title(heading, date_epoch, left_when, panel=live_panel),
                            show_footer=False,
                        )
                    with mc2:
                        st_plotly_press(
                            fig_arch, "meteogram_recent",
                            csv_text=df_arch.to_csv(index=False),
                            export_title=_meteo_export_title(heading, date_epoch, analog_target, panel=year_panel),
                            show_footer=False,
                        )
                st.markdown(_temp_legend_html(), unsafe_allow_html=True)
                with st.container(key="atmopulse_split_meteo_foot"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        st_plotly_press(
                            fig_live, "meteogram_historical",
                            csv_text=df_live.to_csv(index=False),
                            export_title=_meteo_export_title(heading, date_epoch, left_when, panel=live_panel),
                            show_chart=False,
                        )
                    with mc2:
                        st_plotly_press(
                            fig_arch, "meteogram_recent",
                            csv_text=df_arch.to_csv(index=False),
                            export_title=_meteo_export_title(heading, date_epoch, analog_target, panel=year_panel),
                            show_chart=False,
                        )
                if use_z500:
                    st.markdown("<div class='atmopulse-z500-gap'></div>", unsafe_allow_html=True)
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    with st.container(key="atmopulse_split_meteo_z500"):
                        zc1, zc2 = st.columns(2, gap="small")
                        with zc1:
                            st_plotly_press(
                                _build_z500_panel(z500_live, z500_ylim, vline_live),
                                "meteogram_z500_historical",
                                csv_text=z500_csv_live,
                                export_title=_meteo_export_title("Z500 anomaly", date_epoch, left_when, panel=live_panel),
                            )
                        with zc2:
                            st_plotly_press(
                                _build_z500_panel(z500_arch, z500_ylim, vline_arch),
                                "meteogram_z500_recent",
                                csv_text=z500_csv_arch,
                                export_title=_meteo_export_title("Z500 anomaly", date_epoch, analog_target, panel=year_panel),
                            )
                st.markdown("<div class='atmopulse-meteo-yearly-gap'></div>", unsafe_allow_html=True)
                fig_yr_live = build_yearly_extremes_chart(
                    lat_target, lon_target, "A", col_target, wsdi=meteo_spell, csdi=meteo_spell,
                    _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series,
                )
                fig_yr_arch = build_yearly_extremes_chart(
                    lat_target, lon_target, "B", col_target, wsdi=meteo_spell, csdi=meteo_spell,
                    _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series,
                )
                align_yearly_extremes_yranges(fig_yr_live, fig_yr_arch)
                with st.container(key="atmopulse_split_meteo_yearly_dates"):
                    yc1, yc2 = st.columns(2, gap="small")
                    with yc1:
                        st_plotly_press(
                            fig_yr_live, "yearly_historical_dates",
                            export_title=_meteo_export_title("Days exceeding thresholds", "A", left_when),
                        )
                    with yc2:
                        st_plotly_press(
                            fig_yr_arch, "yearly_recent_dates",
                            export_title=_meteo_export_title("Days exceeding thresholds", "B", analog_target),
                        )
            else:
                flicker_choice = st.session_state.get("met_year_pick", live_panel)
                show_live = flicker_choice == live_panel
                traces = traces_live if show_live else traces_arch
                title = title_live if show_live else title_arch
                vline_x = vline_live if show_live else vline_arch
                df_csv = df_live if show_live else df_arch
                when = left_when if show_live else analog_target
                panel = live_panel if show_live else year_panel
                fig = _build_meteogram_panel(traces, title.replace("<br>", " "), global_min, global_max, vline_x)
                st_plotly_press(
                    fig, f"meteogram_{'live' if show_live else 'year'}",
                    csv_text=df_csv.to_csv(index=False),
                    export_title=_meteo_export_title(heading, date_epoch, when, panel=panel),
                    legend_html=_temp_legend_html(),
                    note_html=narrative_html,
                )
                if use_z500:
                    st.markdown("<div class='atmopulse-z500-gap'></div>", unsafe_allow_html=True)
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    z_tr = z500_live if show_live else z500_arch
                    z_csv = z500_csv_live if show_live else z500_csv_arch
                    st_plotly_press(
                        _build_z500_panel(z_tr, z500_ylim, vline_x),
                        f"meteogram_z500_{'live' if show_live else 'year'}",
                        csv_text=z_csv,
                        export_title=_meteo_export_title("Z500 anomaly", date_epoch, when, panel=panel),
                    )
    else:
        with st.spinner("Fetching Meteogram data..."):
            if is_archive:
                df_live = get_archive_year_point_series(lat_target, lon_target, archive_year)
            else:
                df_live = get_live_point_series(lat_target, lon_target, selected_forecast_model())
        if is_archive and df_live.empty:
            st.error(f"No ERA5 archive data available for {archive_year} at this location.")
        elif not df_live.empty:
            col_target = meteo_var_code(meteo_var)

            # Archive Year: the whole series is drawn solid, so the plotting
            # target is the last day of the frame (31 Dec for a closed year,
            # last finite ERA5/ERA5T day for the in-progress current year —
            # get_meteogram_traces only dots dates >= target_date). No IFS
            # overlay. The "currently ..." banner is skipped for Archive Year.
            if is_archive:
                plot_target_date = _archive_plot_target(df_live, archive_year)
                cat_a = dir_a = cat_b = dir_b = None
                narrative_html = None
                narrative_left = None
                narrative_right = None
            else:
                plot_target_date = target_date

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

                cat_a = dir_a = cat_b = dir_b = None
                narrative_html = None
                narrative_left = None
                narrative_right = None
                # Only the exact calendar day. A missing row is a missing day —
                # never snap to a neighbour and narrate the wrong date.
                if active_date in df_indexed.index and col_target in df_indexed.columns:
                    value_now = float(df_indexed.loc[active_date, col_target])
                    if np.isfinite(value_now):
                        percentiles_a = compute_point_thresholds(
                            ref_clim, lat_target, lon_target, active_date, meteo_var, "A",
                        )
                        percentiles_b = compute_point_thresholds(
                            ref_clim, lat_target, lon_target, active_date, meteo_var, "B",
                        )
                        cat_a, dir_a = classify_point_severity(value_now, *percentiles_a)
                        cat_b, dir_b = classify_point_severity(value_now, *percentiles_b)
                        addr = html.escape(location.address)
                        if map_layout == LAYOUT_SINGLE_CHART:
                            cat_x, dir_x = (cat_a, dir_a) if met_active_epoch == "A" else (cat_b, dir_b)
                            lead = html.escape(baseline_against_lead(met_active_epoch))
                            narrative_html = _point_note(addr, "is currently", lead, cat_x, dir_x)
                        else:
                            narrative_left = _point_note(
                                addr, "is currently", html.escape(baseline_against_lead("A")), cat_a, dir_a,
                                force_lead=True,
                            )
                            narrative_right = _point_note(
                                addr, "is currently", html.escape(baseline_against_lead("B")), cat_b, dir_b,
                                force_lead=True,
                            )

            t_arr = (
                df_live[col_target].values
                if col_target in df_live.columns
                else np.full(len(df_live), np.nan)
            )
            global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
            tgt_dt_norm = pd.to_datetime(plot_target_date).tz_localize(None)

            traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, plot_target_date, "A", meteo_env, meteo_var, current_condition=(cat_a, dir_a))
            traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, plot_target_date, "B", meteo_env, meteo_var, current_condition=(cat_b, dir_b))

            show_z500 = show_expert("z500")
            syn_clim = load_synoptic_climatology() if show_z500 else None
            z500_a, z500_rng_a, z500_csv_a = get_z500_anomaly_traces(
                df_live, syn_clim, lat_target, lon_target, plot_target_date, "A",
            ) if show_z500 else ([], None, None)
            z500_b, z500_rng_b, z500_csv_b = get_z500_anomaly_traces(
                df_live, syn_clim, lat_target, lon_target, plot_target_date, "B",
            ) if show_z500 else ([], None, None)
            use_z500 = bool(z500_a and z500_b)
            z500_ylim = None
            if use_z500 and z500_rng_a and z500_rng_b:
                z500_ylim = (
                    min(z500_rng_a[0], z500_rng_b[0]),
                    max(z500_rng_a[1], z500_rng_b[1]),
                )
            vline_x = tgt_dt_norm.timestamp() * 1000
            if is_archive and (
                map_layout == LAYOUT_SINGLE_CHART
                or not _archive_day_available(df_live, plot_target_date)
            ):
                vline_x = None
            title_a = epoch_period_label("A")
            title_b = epoch_period_label("B")

            if map_layout == LAYOUT_SIDE_BY_SIDE:
                live_csv = df_live.to_csv(index=False)
                fig_a = _build_meteogram_panel(
                    traces_a, title_a, global_min, global_max, vline_x,
                )
                fig_b = _build_meteogram_panel(
                    traces_b, title_b, global_min, global_max, vline_x,
                )
                heading = str(meteo_var)
                if narrative_left or narrative_right:
                    note_l, note_r = st.columns(2, gap="small")
                    with note_l:
                        if narrative_left:
                            st.markdown(narrative_left, unsafe_allow_html=True)
                    with note_r:
                        if narrative_right:
                            st.markdown(narrative_right, unsafe_allow_html=True)
                with st.container(key="atmopulse_split_meteo"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        st_plotly_press(
                            fig_a, "meteogram_historical", csv_text=live_csv,
                            export_title=_meteo_export_title(heading, "A", plot_target_date),
                            show_footer=False,
                        )
                    with mc2:
                        st_plotly_press(
                            fig_b, "meteogram_recent", csv_text=live_csv,
                            export_title=_meteo_export_title(heading, "B", plot_target_date),
                            show_footer=False,
                        )
                st.markdown(_temp_legend_html(), unsafe_allow_html=True)
                with st.container(key="atmopulse_split_meteo_foot"):
                    mc1, mc2 = st.columns(2, gap="small")
                    with mc1:
                        st_plotly_press(
                            fig_a, "meteogram_historical", csv_text=live_csv,
                            export_title=_meteo_export_title(heading, "A", plot_target_date),
                            show_chart=False,
                        )
                    with mc2:
                        st_plotly_press(
                            fig_b, "meteogram_recent", csv_text=live_csv,
                            export_title=_meteo_export_title(heading, "B", plot_target_date),
                            show_chart=False,
                        )
                if use_z500:
                    st.markdown("<div class='atmopulse-z500-gap'></div>", unsafe_allow_html=True)
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    fig_z_a = _build_z500_panel(z500_a, z500_ylim, vline_x)
                    fig_z_b = _build_z500_panel(z500_b, z500_ylim, vline_x)
                    with st.container(key="atmopulse_split_meteo_z500"):
                        zc1, zc2 = st.columns(2, gap="small")
                        with zc1:
                            st_plotly_press(
                                fig_z_a, "meteogram_z500_historical",
                                csv_text=z500_csv_a,
                                export_title=_meteo_export_title("Z500 anomaly", "A", plot_target_date),
                            )
                        with zc2:
                            st_plotly_press(
                                fig_z_b, "meteogram_z500_recent",
                                csv_text=z500_csv_b,
                                export_title=_meteo_export_title("Z500 anomaly", "B", plot_target_date),
                            )
                st.markdown("<div class='atmopulse-meteo-yearly-gap'></div>", unsafe_allow_html=True)

                fig_yr_a = build_yearly_extremes_chart(lat_target, lon_target, "A", col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series)
                fig_yr_b = build_yearly_extremes_chart(lat_target, lon_target, "B", col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series)
                align_yearly_extremes_yranges(fig_yr_a, fig_yr_b)
                with st.container(key="atmopulse_split_meteo_yearly"):
                    yc1, yc2 = st.columns(2, gap="small")
                    with yc1:
                        st_plotly_press(
                            fig_yr_a, "yearly_historical",
                            export_title=_meteo_export_title("Days exceeding thresholds", "A", plot_target_date),
                        )
                    with yc2:
                        st_plotly_press(
                            fig_yr_b, "yearly_recent",
                            export_title=_meteo_export_title("Days exceeding thresholds", "B", plot_target_date),
                        )
            else:
                traces = traces_a if met_active_epoch == "A" else traces_b
                z500_tr = z500_a if met_active_epoch == "A" else z500_b
                fig = _build_meteogram_panel(
                    traces, epoch_period_label(met_active_epoch),
                    global_min, global_max, vline_x,
                )
                st_plotly_press(
                    fig, f"meteogram_{met_active_epoch}",
                    csv_text=df_live.to_csv(index=False),
                    export_title=_meteo_export_title(str(meteo_var), met_active_epoch, plot_target_date),
                    legend_html=_temp_legend_html(),
                    note_html=narrative_html,
                )
                if use_z500:
                    st.markdown("<div class='atmopulse-z500-gap'></div>", unsafe_allow_html=True)
                    st.markdown("**Z500 anomaly**", help=HELP["meteo_z500_panel"])
                    fig_z = _build_z500_panel(z500_tr, z500_ylim, vline_x)
                    st_plotly_press(
                        fig_z, f"meteogram_z500_{met_active_epoch}",
                        csv_text=z500_csv_a if met_active_epoch == "A" else z500_csv_b,
                        export_title=_meteo_export_title("Z500 anomaly", met_active_epoch, plot_target_date),
                    )
                st.markdown("<div class='atmopulse-meteo-yearly-gap'></div>", unsafe_allow_html=True)
                st_plotly_press(
                    build_yearly_extremes_chart(lat_target, lon_target, met_active_epoch, col_target, wsdi=meteo_spell, csdi=meteo_spell, _ref_clim=ref_clim, _load_point_archive_series=_load_point_archive_series),
                    f"yearly_{met_active_epoch}",
                    export_title=_meteo_export_title("Days exceeding thresholds", met_active_epoch, plot_target_date),
                )
