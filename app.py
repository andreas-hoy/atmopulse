"""
AtmoPulse Central UI Orchestration & Visualization Module (app.py)

This module serves as the primary Streamlit frontend for the AtmoPulse application.
It orchestrates the dynamic tracking of pan-European synoptic extremes by bridging 
secular ERA5 climate baselines with operational ECMWF/IFS forecasts.

Core functionalities:
- Manages the interactive UI state, navigation, and user toggles.
- Orchestrates high-performance data loading and singleton caching for large NetCDF 
  archives (lazy dask-backed handles) and live IFS datasets.
- Handles geospatial rendering via Plotly, including synchronized side-by-side 
  comparisons and single-map flickers for temperature anomalies (TG, TX, TN).
- Computes real-time spatial impact rankings (Top 10 affected countries) using 
  area-weighted masking.
- Generates localized point meteograms and wavograms, dynamically applying 
  ETCCDI-compliant 365-day leap-year adjustments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import threading
from pathlib import Path
from datetime import datetime

from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

from backend_map_locations import EUROPE_BBOX
from backend_maps import etccdi_doy_365, latest_era5_archive_date, latest_forecast_cycle_label
from backend_io import get_archive_year_options
from backend_narrative import render_point_wavogram_narrative
from backend_waves import compute_kysely_waves_data, rank_waves_by_metric
from labels import HELP
from atmopulse_theme import (
    ATMOPULSE_BRAND,
    atmopulse_streamlit_css,
    atmopulse_wordmark_html,
    inject_monday_weekstart,
    LOGO_SVG,
)
from config import (
    UI_MODE_STANDARD,
    UI_MODE_EXPERT,
    UI_MODE_LABELS,
    FORECAST_MODEL_IFS,
    FORECAST_MODEL_OPTIONS,
    MAP_VIEW_WAVES,
    MAP_VIEW_OPTIONS_STANDARD,
    MAP_VIEW_OPTIONS_EXPERT,
    MAP_WAVE_LEVELS,
    LAYOUT_SINGLE_CHART,
    LAYOUT_SIDE_BY_SIDE,
    AIFS_TXTN_WARNING,
    NAV_WELCOME,
    NAV_MAP,
    NAV_METEO,
    NAV_WAVE,
    NAV_METHODS,
    NAV_LEGAL,
    NAV_ITEMS,
    NAV_ANALYTICS,
    STANDARD_DEFAULTS,
    SPELL_OFF,
    SPELL_LABELS,
    MAP_VAR_OPTIONS,
    MAP_VAR_LABELS,
    METEO_COUNT_OPTIONS,
    METEO_COUNT_SPELL,
    FORECAST_OFFSET_MIN,
    FORECAST_OFFSET_MAX,
    SLIDER_PAD_PAST,
    SLIDER_PAD_FUTURE,
    is_expert_mode,
    is_aifs_model,
    selected_forecast_model,
    show_expert,
    is_daily_map_view,
    is_wave_map_view,
    wave_locked_var_code,
    epoch_period_label,
    epoch_short_label,
    epoch_from_label,
    meteo_var_code,
)
from frontend_widgets import render_grid_cell_profile
from page_map_tracker import render_map_tracker
from page_meteogram import render_meteogram
from frontend_plots import (
    st_plotly_press,
    align_wave_stats_yranges,
    build_wave_event_mini_fig,
    build_wave_event_z500_mini_fig,
    render_press_export,
    _output_credit_text,
    PLOTLY_UI_CONFIG,
    wave_event_z500_window,
    _Z500_ANOM_Y_FLOOR,
)


# --- WAVE CACHE WRAPPER ---
# `compute_kysely_waves_data` (backend_waves.py, compute-only: pandas/numpy/
# xarray) is cached on the hashable (lat, lon, param_code, selected_epoch,
# wave_thresh, is_warm) inputs. `build_kysely_wave_figs`
# (frontend_plots.py) then turns that payload into the two Plotly Figures —
# kept OUT of the cached function so Streamlit never has to hash/pickle
# go.Figure objects. Owned by app.py (not backend_io.py) so backend_io never
# has to import backend_waves, and backend_waves never has to import Plotly.
@st.cache_data(show_spinner=False)
def _compute_wave_payload(lat_target, lon_target, param_code, selected_epoch, wave_thresh, is_warm=True, z500_ctx=5):
    return compute_kysely_waves_data(
        lat_target, lon_target, parameter=param_code, selected_epoch=selected_epoch,
        threshold_level=wave_thresh, is_warm=is_warm, z500_ctx=z500_ctx,
    )


def fetch_wave_figs(lat_target, lon_target, param_code, selected_epoch, wave_thresh, is_warm=True, z500_outline=False, stack_metric="Intensity", z500_ctx=5, x_range=None):
    from frontend_plots import build_kysely_wave_figs
    payload = _compute_wave_payload(
        lat_target, lon_target, param_code, selected_epoch, wave_thresh,
        is_warm=is_warm, z500_ctx=z500_ctx,
    )
    return build_kysely_wave_figs(
        payload, z500_outline=z500_outline, stack_metric=stack_metric, x_range=x_range,
    )


def fetch_aligned_wave_figs(
    lat_target, lon_target, param_code, wave_thresh, is_warm=True,
    z500_outline=False, stack_metric="Intensity", z500_ctx=5,
):
    """Build A/B wavograms on a shared x-window (core season, or expanded
    when either baseline has a shoulder-month event)."""
    from frontend_plots import build_kysely_wave_figs, union_wave_xrange
    payload_a = _compute_wave_payload(
        lat_target, lon_target, param_code, "A", wave_thresh,
        is_warm=is_warm, z500_ctx=z500_ctx,
    )
    payload_b = _compute_wave_payload(
        lat_target, lon_target, param_code, "B", wave_thresh,
        is_warm=is_warm, z500_ctx=z500_ctx,
    )
    x_range = union_wave_xrange(payload_a, payload_b)
    return (
        build_kysely_wave_figs(
            payload_a, z500_outline=z500_outline, stack_metric=stack_metric, x_range=x_range,
        ),
        build_kysely_wave_figs(
            payload_b, z500_outline=z500_outline, stack_metric=stack_metric, x_range=x_range,
        ),
    )


# --- Point Wavogram: Event Drill-down ---
# Five swappable mini-charts under the ridge plot, above the intensity
# stack. The swap dropdown's labels double as the event list (Rank, Start,
# End, Duration, Intensity) — no separate table. Ranking/selection state
# lives here (app.py); reads only the already-detected `waves_data` from
# the period chosen at the top of the wavogram (Side by side: recent
# period). No second period switch in this block.
def _wave_event_label(w: dict) -> str:
    # Rank, Start, End, Duration, Intensity — the compact event table used to
    # show these as columns; now folded into the dropdown label itself since
    # it duplicated the same information (user feedback).
    sd = pd.Timestamp(w["start_date"]).strftime("%d.%m.%Y")
    ed = pd.Timestamp(w["end_date"]).strftime("%d.%m.%Y")
    rank = w.get("rank")
    prefix = f"#{rank} \u00b7 " if rank else ""
    return f"{prefix}{sd}\u2013{ed} \u00b7 {int(w['duration_days'])} d \u00b7 {float(w['intensity']):.1f} K"


def _wave_section_spacer(px: int = 28) -> None:
    # Purely visual breathing room between the stacked wave sections
    # (ridge -> drill-down -> intensity -> frequency) — no divider line,
    # just vertical margin, so the sections read as distinct blocks.
    st.markdown(f"<div style='margin-top:{px}px'></div>", unsafe_allow_html=True)


def _year_ago_archive_date():
    """Calendar day one year before today, clamped to the archive picker."""
    max_d = latest_era5_archive_date().date()
    min_d = pd.Timestamp(1940, 1, 1).date()
    year_ago = (pd.Timestamp.now().normalize() - pd.DateOffset(years=1)).date()
    return min(max(year_ago, min_d), max_d)


def _on_map_date_mode_change():
    """Selecting Date jumps the archive calendar to the same day last year."""
    if st.session_state.get("map_archive_mode") != "Archive":
        return
    picked = _year_ago_archive_date()
    st.session_state.map_archive_date = picked
    st.session_state["_active_map_archive_date"] = picked


def _open_wave_event_on_map(start_date, *, is_warm=None, parameter=None, threshold_level=None, event_id=None):
    """Callback for the per-event 'Map this event' button: point the Map
    Tracker at this wave's start day (ERA5 Archive) on the Wave tracking
    view — same is_warm/variable/Strong-Extreme/event_id as the Wavogram
    event, so the map always lands on the SAME event, never on Daily
    snapshot or Persistence duration (LOCKED PRODUCT DECISIONS: "Map this
    event" ... "does NOT open Daily/Persistence/Point Meteogram"). Also
    pre-selects the Meteogram's Archive Year to match if that calendar
    year is in the Archive Year list (complete years plus the in-progress
    current year; else leave the Meteogram on Live), and jumps the top nav
    to Map Tracker. Never touches offset_slider — this is a one-way "open
    on map" action, not a general Map<->Meteogram coupling.

    No explicit st.rerun() here: Streamlit always reruns the script once
    after any on_click callback finishes, so calling st.rerun() inside it
    is a documented no-op (and raises a "Calling st.rerun() within a
    callback is a no-op" warning banner) rather than a second, real rerun.
    """
    start_ts = pd.Timestamp(start_date)
    min_d = pd.Timestamp(1940, 1, 1).date()
    max_d = latest_era5_archive_date().date()
    st.session_state.map_archive_mode = "Archive"
    st.session_state.map_archive_date = min(max(start_ts.date(), min_d), max_d)

    archive_years = get_archive_year_options(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    # Stashed in a plain (never-a-widget) key, not written straight into
    # met_archive_year: the Meteogram's selectbox won't be instantiated
    # again for at least this rerun (we're jumping to Map Tracker), and
    # Streamlit drops session_state for a widget key that isn't recreated
    # in a run. page_meteogram.render_meteogram() consumes this pending
    # value into met_archive_year right before building that selectbox,
    # whenever the user actually gets there — even several reruns later.
    st.session_state.pending_met_archive_year = (
        str(start_ts.year) if start_ts.year in archive_years else "Live"
    )

    # Retarget the Map Tracker itself: Wave tracking view, matching
    # direction/level, and (Expert only) matching variable — Standard mode
    # locks TX/TN to the direction anyway via wave_locked_var_code().
    # These are all plain widget keys created with key= below, so writing
    # them here before the sidebar widgets are (re)built is the documented
    # Streamlit pattern for programmatically setting a widget's value.
    st.session_state["map_view_mode"] = MAP_VIEW_WAVES
    if is_warm is not None:
        st.session_state["map_wave_direction"] = "Heat" if is_warm else "Cold"
    if threshold_level:
        st.session_state["map_wave_level"] = "Extreme" if "Extreme" in threshold_level else "Strong"
    if parameter:
        matching = next(
            (opt for opt in MAP_VAR_OPTIONS if f"({parameter})" in opt), None,
        )
        if matching:
            st.session_state["map_var_radio"] = matching
    # Non-widget key: not consumed by any rendering yet, but kept for a
    # future "highlight this specific event" affordance without needing
    # another retarget-signature change.
    st.session_state["map_wave_event_id"] = event_id

    st.session_state.atmopulse_top_nav = NAV_MAP


def _render_wave_drilldown(payload_a, payload_b, stack_metric, ctx_key, click_events=(), epoch: str | None = None):
    payloads = {"A": payload_a, "B": payload_b}
    avail_epochs = [
        ep for ep in ("A", "B")
        if payloads.get(ep) and not payloads[ep].get("empty") and payloads[ep].get("waves_data")
    ]
    if not avail_epochs:
        return  # no waves in either reference period: the existing empty-state elsewhere already covers this
    _wave_section_spacer()

    # The period is chosen once, at the top of the wavogram. This block only
    # names it. Side by side has no single choice, so it uses the recent period.
    if epoch not in avail_epochs:
        epoch = "B" if "B" in avail_epochs else avail_epochs[0]
    st.session_state["wave_drill_epoch"] = epoch
    st.markdown(
        f"**Event Drill-down | {epoch_period_label(epoch)}**",
        help=HELP["wave_drilldown"],
    )
    payload_b = payloads[epoch]  # reuse the name below — rest of the function is unchanged

    ranked = rank_waves_by_metric(payload_b["waves_data"], stack_metric)
    id_lookup = {w["event_id"]: w for w in ranked}
    n_total = len(ranked)
    top_ids = [w["event_id"] for w in ranked[:5]]

    ms_key = "wave_drill_ms"
    # The reference-period toggle is folded into the reset context, same as
    # location/warm-cold/threshold: switching it means a different set of
    # detected events, so manual slot picks from the other period don't
    # carry over as stale event_ids.
    full_ctx = (ctx_key, epoch)
    prev_ctx = st.session_state.get("wave_drill_ctx")
    prev_metric = st.session_state.get("wave_drill_metric")
    manual = bool(st.session_state.get("wave_drill_manual", False))

    def _set_selection(ids):
        st.session_state[ms_key] = [_wave_event_label(id_lookup[i]) for i in ids if i in id_lookup]

    # Location / Heat<->Cold / Strong<->Extreme / Reference period (baked
    # into full_ctx) always reset to Top 5. Intensity<->Days only resets
    # Top 5 if the user hasn't touched a slot yet; otherwise the picks are
    # kept, just re-ranked.
    if prev_ctx != full_ctx:
        manual = False
        _set_selection(top_ids)
    elif prev_metric != stack_metric and not manual:
        _set_selection(top_ids)

    # Best-effort click-to-swap from the ridge plot(s): replaces the
    # currently-weakest slot unless the clicked event is already selected.
    # Wrapped defensively — an empty/odd selection payload must never crash
    # the page; the dropdown below always works regardless.
    #
    # IMPORTANT: a Plotly `on_select="rerun"` selection is sticky — it stays
    # in session_state and gets returned again on every later, unrelated
    # rerun (e.g. a dropdown pick), not just the run where the user actually
    # clicked. Without the per-widget "last seen" signature below, that
    # stale click would be reprocessed on every rerun and silently overwrite
    # whatever the user had just picked in the dropdown.
    try:
        current_ids = [
            i for i in st.session_state.get("wave_drill_ids", top_ids) if i in id_lookup
        ] or top_ids
        for click_key, ev in click_events:
            pts = None
            if isinstance(ev, dict):
                pts = (ev.get("selection") or {}).get("points")
            elif ev is not None:
                pts = getattr(getattr(ev, "selection", None), "points", None)
            if not pts:
                continue
            clicked_id = (pts[0] or {}).get("customdata") if isinstance(pts[0], dict) else getattr(pts[0], "customdata", None)
            if isinstance(clicked_id, (list, tuple)):
                clicked_id = clicked_id[0] if clicked_id else None
            seen_key = f"_wave_click_seen_{click_key}"
            if clicked_id == st.session_state.get(seen_key):
                continue  # same click as last time we looked — already handled, not a new interaction
            st.session_state[seen_key] = clicked_id
            if not clicked_id or clicked_id not in id_lookup or clicked_id in current_ids:
                continue
            if len(current_ids) < 5:
                current_ids = current_ids + [clicked_id]
            else:
                weakest = max(current_ids, key=lambda i: id_lookup[i]["rank"])
                current_ids = [clicked_id if i == weakest else i for i in current_ids]
            _set_selection(current_ids)
            manual = True
            break
    except Exception:
        pass

    st.session_state.wave_drill_ctx = full_ctx
    st.session_state.wave_drill_metric = stack_metric

    all_labels = [_wave_event_label(w) for w in ranked]
    label_to_id = {_wave_event_label(w): w["event_id"] for w in ranked}
    selected_labels = st.multiselect(
        "Swap slots:", all_labels, max_selections=5,
        key=ms_key, help=HELP["wave_drilldown_slot"],
    )
    selected_ids = [label_to_id[lbl] for lbl in selected_labels if lbl in label_to_id]
    if not selected_ids:
        selected_ids = top_ids
    st.session_state.wave_drill_ids = selected_ids
    st.session_state.wave_drill_manual = manual or (set(selected_ids) != set(top_ids))

    selected_waves = sorted(
        (id_lookup[i] for i in selected_ids if i in id_lookup),
        key=lambda w: w["rank"],
    )
    if selected_waves:
        # Shared y-scale across the displayed mini-charts (union of their
        # padded daily windows + the two threshold lines), so severity is
        # visually comparable slot-to-slot instead of each auto-scaling.
        daily = payload_b.get("daily_series")
        y_vals = []
        if daily is not None and len(daily):
            for w in selected_waves:
                pad_start = pd.Timestamp(w["start_date"]).normalize() - pd.Timedelta(days=3)
                pad_end = pd.Timestamp(w["end_date"]).normalize() + pd.Timedelta(days=3)
                window = daily[(daily.index >= pad_start) & (daily.index <= pad_end)]
                if len(window):
                    finite = window.values[np.isfinite(window.values)]
                    if finite.size:
                        y_vals.extend([float(np.min(finite)), float(np.max(finite))])
        for v in (payload_b.get("p_thresh"), payload_b.get("p_drop")):
            if v is not None and np.isfinite(v):
                y_vals.append(float(v))
        y_range = None
        if y_vals:
            y_min, y_max = min(y_vals), max(y_vals)
            pad = max(1.0, (y_max - y_min) * 0.08)
            y_range = (y_min - pad, y_max + pad)

        mini_cols = st.columns(len(selected_waves))
        show_z500_minis = show_expert("z500")
        z500_range = None
        if show_z500_minis:
            z_abs = []
            for w in selected_waves:
                packed = wave_event_z500_window(payload_b, w)
                if packed is None:
                    continue
                anom = packed[1]
                finite = anom[np.isfinite(anom)]
                if finite.size:
                    z_abs.append(float(np.max(np.abs(finite))))
            if z_abs:
                z_lim = max(_Z500_ANOM_Y_FLOOR, max(z_abs) * 1.15)
                z500_range = (-z_lim, z_lim)
        for col, w in zip(mini_cols, selected_waves):
            with col:
                mini_fig = build_wave_event_mini_fig(payload_b, w, n_total, y_range=y_range)
                st.plotly_chart(
                    mini_fig, use_container_width=True, key=f"wave_mini_{w['event_id']}",
                    config=PLOTLY_UI_CONFIG,
                )
                # Same compact SVG/PDF/CSV row as every other AtmoPulse
                # figure, per event — not one combined export for all 5
                # (different date ranges per slot don't share one CSV/SVG).
                csv_text = None
                daily = payload_b.get("daily_series")
                if daily is not None and len(daily):
                    pad_start = pd.Timestamp(w["start_date"]).normalize() - pd.Timedelta(days=3)
                    pad_end = pd.Timestamp(w["end_date"]).normalize() + pd.Timedelta(days=3)
                    window = daily[(daily.index >= pad_start) & (daily.index <= pad_end)]
                    if len(window):
                        csv_text = pd.DataFrame({
                            "date": window.index.strftime("%Y-%m-%d"),
                            "value": window.values,
                        }).to_csv(index=False)
                if show_z500_minis:
                    render_press_export(mini_fig, f"wavogram_event_{w['event_id']}", csv_text=csv_text)
                    z_fig = build_wave_event_z500_mini_fig(payload_b, w, y_range=z500_range)
                    if z_fig is not None:
                        st.plotly_chart(
                            z_fig, use_container_width=True,
                            key=f"wave_mini_z500_{w['event_id']}",
                            config=PLOTLY_UI_CONFIG,
                        )
                        packed = wave_event_z500_window(payload_b, w)
                        z_csv = None
                        if packed is not None:
                            dates, anom, z_live, z_clim = packed
                            z_csv = pd.DataFrame({
                                "date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
                                "z500_dam": np.round(z_live, 2),
                                "doy_mean_dam": np.round(z_clim, 2),
                                "z500_anom_dam": np.round(anom, 2),
                            }).to_csv(index=False)
                        with st.container(key=f"wave_event_foot_{w['event_id']}"):
                            render_press_export(
                                z_fig, f"wavogram_event_z500_{w['event_id']}",
                                csv_text=z_csv,
                            )
                            st.button(
                                "Map this event",
                                key=f"wave_open_map_{w['event_id']}",
                                help=HELP["wave_open_map"],
                                on_click=_open_wave_event_on_map,
                                args=(w["start_date"],),
                                kwargs=dict(
                                    is_warm=payload_b.get("is_warm"),
                                    parameter=payload_b.get("parameter"),
                                    threshold_level=payload_b.get("threshold_level"),
                                    event_id=w["event_id"],
                                ),
                            )
                    else:
                        st.button(
                            "Map this event",
                            key=f"wave_open_map_{w['event_id']}",
                            help=HELP["wave_open_map"],
                            on_click=_open_wave_event_on_map,
                            args=(w["start_date"],),
                            kwargs=dict(
                                is_warm=payload_b.get("is_warm"),
                                parameter=payload_b.get("parameter"),
                                threshold_level=payload_b.get("threshold_level"),
                                event_id=w["event_id"],
                            ),
                        )
                else:
                    with st.container(key=f"wave_event_foot_{w['event_id']}"):
                        render_press_export(mini_fig, f"wavogram_event_{w['event_id']}", csv_text=csv_text)
                        st.button(
                            "Map this event",
                            key=f"wave_open_map_{w['event_id']}",
                            help=HELP["wave_open_map"],
                            on_click=_open_wave_event_on_map,
                            args=(w["start_date"],),
                            kwargs=dict(
                                is_warm=payload_b.get("is_warm"),
                                parameter=payload_b.get("parameter"),
                                threshold_level=payload_b.get("threshold_level"),
                                event_id=w["event_id"],
                            ),
                        )
    _wave_section_spacer()

# --- UI & CSS: TOP NAVIGATION BAR ---
st.set_page_config(page_title="AtmoPulse", layout="wide", page_icon="assets/favicon.svg", initial_sidebar_state="expanded")
import atmopulse_theme as _ap_theme
import backend_narrative as _bn
import frontend_plots as _fp
import page_map_tracker as _pmt
import page_meteogram as _pm
importlib.reload(_ap_theme)
importlib.reload(_bn)
importlib.reload(_fp)
importlib.reload(_pmt)
importlib.reload(_pm)
st_plotly_press = _fp.st_plotly_press
align_wave_stats_yranges = _fp.align_wave_stats_yranges
build_wave_event_mini_fig = _fp.build_wave_event_mini_fig
build_wave_event_z500_mini_fig = _fp.build_wave_event_z500_mini_fig
render_press_export = _fp.render_press_export
_output_credit_text = _fp._output_credit_text
PLOTLY_UI_CONFIG = _fp.PLOTLY_UI_CONFIG
wave_event_z500_window = _fp.wave_event_z500_window
_Z500_ANOM_Y_FLOOR = _fp._Z500_ANOM_Y_FLOOR
render_map_tracker = _pmt.render_map_tracker
render_meteogram = _pm.render_meteogram
st.markdown(f"<style>{_ap_theme.atmopulse_streamlit_css(_ap_theme.ATMOPULSE_BRAND)}</style>", unsafe_allow_html=True)
st.html(
    """
<div class="ap-map-fit" style="height:0;overflow:hidden"></div>
<script>
(function () {
  if (window.__apMapFitStop) window.__apMapFitStop();
  var sel = ".st-key-map_a svg.main-svg, .st-key-map_b svg.main-svg, .st-key-map_flicker svg.main-svg, .st-key-map_opacity svg.main-svg";
  function fit() {
    document.querySelectorAll(sel).forEach(function (svg) {
      var w = svg.getAttribute("width");
      var h = svg.getAttribute("height");
      if (!w || !h) return;
      var vb = "0 0 " + w + " " + h;
      if (svg.getAttribute("viewBox") !== vb) svg.setAttribute("viewBox", vb);
      if (svg.getAttribute("preserveAspectRatio") !== "none") svg.setAttribute("preserveAspectRatio", "none");
    });
  }
  function sizeSwipe() {
    document.querySelectorAll(".st-key-swipe_map_frame iframe").forEach(function (frame) {
      var w = frame.clientWidth || frame.getBoundingClientRect().width || 0;
      if (w < 280) return;
      var mapH = Math.round(w * 42 / 70);
      if (mapH < 280) return;
      var extra = 46 + 8;
      try {
        var ctrl = frame.contentDocument && frame.contentDocument.querySelector(".atmopulse-swipe-ctrl");
        if (ctrl) extra = Math.ceil(ctrl.getBoundingClientRect().height) + 8;
      } catch (e) {}
      var total = mapH + extra;
      if (frame.getAttribute("data-ap-swipe-h") === String(total)) return;
      frame.setAttribute("data-ap-swipe-h", String(total));
      frame.style.setProperty("height", total + "px", "important");
      frame.style.setProperty("min-height", total + "px", "important");
      frame.style.setProperty("max-height", total + "px", "important");
      var wrap = frame.parentElement;
      if (wrap) {
        wrap.style.setProperty("height", "auto", "important");
        wrap.style.setProperty("flex", "0 0 auto", "important");
      }
    });
  }
  fit();
  sizeSwipe();
  var obs = new MutationObserver(function () { fit(); sizeSwipe(); });
  obs.observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["width", "height"]});
  window.addEventListener("resize", function () { sizeSwipe(); });
  [80, 240, 600, 1200].forEach(function (t) { setTimeout(function () { sizeSwipe(); }, t); });
  window.__apMapFitStop = function () { obs.disconnect(); };
})();
</script>
""",
    unsafe_allow_javascript=True,
)
inject_monday_weekstart()

geolocator = Nominatim(user_agent="atmopulse_extremes_tracker_2026")


class _PointLocation:
    """Stand-in for a geopy result, so saved places do not need another lookup."""

    def __init__(self, address, latitude, longitude):
        self.address = address
        self.latitude = float(latitude)
        self.longitude = float(longitude)


def _in_map_domain(lat, lon) -> bool:
    return (
        EUROPE_BBOX[0] <= float(lon) <= EUROPE_BBOX[2]
        and EUROPE_BBOX[1] <= float(lat) <= EUROPE_BBOX[3]
    )


def _filter_map_domain(result):
    if result is None:
        return None
    if isinstance(result, list):
        return [r for r in result if _in_map_domain(r.latitude, r.longitude)]
    if not _in_map_domain(result.latitude, result.longitude):
        return None
    return result


def _safe_geocode(query, **kwargs):
    """Nominatim lookup inside the map domain. Must not crash when the network is down."""
    kwargs.setdefault("timeout", 10)
    kwargs.setdefault(
        "viewbox",
        ((EUROPE_BBOX[1], EUROPE_BBOX[0]), (EUROPE_BBOX[3], EUROPE_BBOX[2])),
    )
    kwargs.setdefault("bounded", True)
    kwargs.setdefault("language", "en")
    try:
        found = geolocator.geocode(query, **kwargs)
    except (GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError):
        st.warning(
            "Location search is temporarily unavailable "
            "(OpenStreetMap Nominatim could not be reached). "
            "Check the internet connection and try again."
        )
        return None
    return _filter_map_domain(found)


_DEFAULT_SAVED_LOCATIONS = (
    {"name": "Tallinn", "address": "Tallinn, Estonia", "lat": 59.44, "lon": 24.75},
    {"name": "Berlin", "address": "Berlin, Germany", "lat": 52.52, "lon": 13.40},
    {"name": "Budapest", "address": "Budapest, Hungary", "lat": 47.50, "lon": 19.04},
)


_COUNTRY_EN = {
    "deutschland": "Germany",
    "österreich": "Austria",
    "schweiz": "Switzerland",
    "frankreich": "France",
    "italien": "Italy",
    "spanien": "Spain",
    "polen": "Poland",
    "tschechien": "Czechia",
    "tschechische republik": "Czechia",
    "ungarn": "Hungary",
    "estland": "Estonia",
    "lettland": "Latvia",
    "litauen": "Lithuania",
    "niederlande": "Netherlands",
    "belgien": "Belgium",
    "dänemark": "Denmark",
    "schweden": "Sweden",
    "norwegen": "Norway",
    "finnland": "Finland",
    "vereinigtes königreich": "United Kingdom",
    "großbritannien": "United Kingdom",
    "irland": "Ireland",
    "portugal": "Portugal",
    "griechenland": "Greece",
    "rumänien": "Romania",
    "bulgarien": "Bulgaria",
    "kroatien": "Croatia",
    "slowenien": "Slovenia",
    "slowakei": "Slovakia",
    "luxemburg": "Luxembourg",
}


def _location_country(loc) -> str:
    parts = [part.strip() for part in str(loc.get("address") or "").split(",") if part.strip()]
    if len(parts) < 2:
        return ""
    country = parts[-1]
    if country.lower() == str(loc.get("name") or "").strip().lower():
        return ""
    return _COUNTRY_EN.get(country.lower(), country)


def _saved_loc_label(loc) -> str:
    lat, lon = float(loc["lat"]), float(loc["lon"])
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    coords = f"{abs(lon):.1f}°{ew}, {abs(lat):.1f}°{ns}"
    place = loc["name"]
    country = _location_country(loc)
    if country:
        place = f"{place}, {country}"
    return f"{place} ({coords})"


def _init_saved_locations() -> None:
    if "saved_locations" in st.session_state:
        return
    known = {item["name"]: item for item in _DEFAULT_SAVED_LOCATIONS}
    ordered = []
    for name in st.session_state.get("search_history", [item["name"] for item in _DEFAULT_SAVED_LOCATIONS]):
        if name in known:
            ordered.append(dict(known[name]))
        else:
            ordered.append({"name": name, "address": name, "lat": None, "lon": None})
    have = {item["name"] for item in ordered}
    for item in _DEFAULT_SAVED_LOCATIONS:
        if item["name"] not in have:
            ordered.append(dict(item))
    st.session_state.saved_locations = ordered[:10]


def _remember_location(name, address, lat, lon) -> dict:
    loc = {
        "name": name,
        "address": address,
        "lat": round(float(lat), 2),
        "lon": round(float(lon), 2),
    }
    key = (loc["lat"], loc["lon"])
    kept = [
        item for item in st.session_state.saved_locations
        if item.get("lat") is None or (round(float(item["lat"]), 2), round(float(item["lon"]), 2)) != key
    ]
    st.session_state.saved_locations = [loc, *kept][:10]
    return loc


if "nc_lock" not in st.session_state: st.session_state.nc_lock = threading.Lock()
_init_saved_locations()
if "toggles_warm" not in st.session_state: st.session_state.toggles_warm = {"p75": True, "p90": True, "p95": True, "rec": True}
if "toggles_cold" not in st.session_state: st.session_state.toggles_cold = {"p25": True, "p10": True, "p5": True, "rec": True}
if "offset_slider" not in st.session_state:
    st.session_state.offset_slider = 0
else:
    st.session_state.offset_slider = int(np.clip(st.session_state.offset_slider, -7, 3))

ASSETS_DIR = Path("assets")
DOCUMENTS_DIR = Path("Documents")
LEGAL_MD = ASSETS_DIR / "legal.md"
METHODS_MD = ASSETS_DIR / "methods.md"


def _query_ui_mode():
    raw = st.query_params.get("mode", None)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    return raw if raw in (UI_MODE_STANDARD, UI_MODE_EXPERT) else None


def _init_ui_mode():
    """Seed audience mode once per session. Query ?mode=expert|standard wins on first load."""
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = _query_ui_mode() or UI_MODE_STANDARD
    if "atmopulse_ui_mode" not in st.session_state:
        st.session_state.atmopulse_ui_mode = (
            "Expert" if st.session_state.ui_mode == UI_MODE_EXPERT else "Standard"
        )


def _on_ui_mode_change():
    st.session_state.ui_mode = str(st.session_state.atmopulse_ui_mode).lower()
    st.query_params["mode"] = st.session_state.ui_mode


_EXTREME_LAYER_LABELS = {
    ("warm", "p75"): ("Warm: Moderate", " (> P75)"),
    ("warm", "p90"): ("Warm: Strong", " (> P90)"),
    ("warm", "p95"): ("Warm: Extreme", " (> P95)"),
    ("warm", "rec"): ("Warm: All-Time Record", ""),
    ("cold", "p25"): ("Cold: Moderate", " (< P25)"),
    ("cold", "p10"): ("Cold: Strong", " (< P10)"),
    ("cold", "p5"): ("Cold: Extreme", " (< P5)"),
    ("cold", "rec"): ("Cold: All-Time Record", ""),
}


def extreme_layer_label(kind: str, key: str) -> str:
    """Checkbox copy: impact names in Standard, percentiles appended in Expert."""
    base, expert_suffix = _EXTREME_LAYER_LABELS[(kind, key)]
    if is_expert_mode() and expert_suffix:
        return f"{base}{expert_suffix}"
    return base


_init_ui_mode()
if "forecast_model" not in st.session_state:
    st.session_state.forecast_model = FORECAST_MODEL_IFS

def _load_markdown_page(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _documents_image(*filenames):
    for name in filenames:
        p = DOCUMENTS_DIR / name
        if p.exists():
            return str(p)
    return None


def add_day():
    if st.session_state.offset_slider < FORECAST_OFFSET_MAX:
        st.session_state.offset_slider += 1

def sub_day():
    if st.session_state.offset_slider > FORECAST_OFFSET_MIN:
        st.session_state.offset_slider -= 1

def toggle_warm_state():
    current = any(st.session_state.toggles_warm.values())
    for k in st.session_state.toggles_warm: 
        st.session_state.toggles_warm[k] = not current

def toggle_cold_state():
    current = any(st.session_state.toggles_cold.values())
    for k in st.session_state.toggles_cold: 
        st.session_state.toggles_cold[k] = not current

def _fmt_map_year(yr_val) -> str:
    try:
        y = int(float(yr_val))
        return str(y) if y > 0 else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def _slider_window_doys(anchor_date, pad_past=SLIDER_PAD_PAST, pad_future=SLIDER_PAD_FUTURE):
    """All calendar day-of-year values reachable via the Forecast Offset slider
    for a given anchor ('today') date."""
    dates = [anchor_date + pd.Timedelta(days=o) for o in range(-pad_past, pad_future + 1)]
    return tuple(sorted({int(etccdi_doy_365(d)) for d in dates}))

# --- UI LAYOUT: TOP NAVIGATION BAR ---
with st.container(vertical_alignment="center", horizontal=True, horizontal_alignment="left", gap="small", key="atmopulse_nav_bar", border=False):
    nav_selection = st.radio(
        "Navigation",
        list(NAV_ITEMS),
        horizontal=True,
        label_visibility="collapsed",
        key="atmopulse_top_nav",
    )
st.divider()

default_date = pd.Timestamp.now().floor('D')
target_month = default_date.month
is_warm_season = False
if 4 < target_month < 10: 
    is_warm_season = True
elif target_month == 4 and default_date.day >= 16: 
    is_warm_season = True
elif target_month == 10 and default_date.day <= 15: 
    is_warm_season = True

default_wave_idx = 0 if is_warm_season else 1

with st.sidebar:
    st.markdown(
        f"<div class='atmopulse-sidebar-logo'>{atmopulse_wordmark_html()}{LOGO_SVG}</div>",
        unsafe_allow_html=True,
    )
    st.radio(
        "UI mode",
        list(UI_MODE_LABELS),
        horizontal=True,
        key="atmopulse_ui_mode",
        on_change=_on_ui_mode_change,
        label_visibility="collapsed",
        help=HELP["ui_mode"],
    )
    if nav_selection in NAV_ANALYTICS:
        st.header("Control Panel")

        # --- Map Tracker only: Live vs. a historical ERA5 calendar day.
        # The calendar and Prev/Next sit above the map (page_map_tracker),
        # not in this panel. Map-local state (map_archive_mode /
        # map_archive_date) is independent of the Meteogram Archive Year
        # and of the Forecast Offset slider. The date widget is created
        # later on the map page; seed/clamp the key here first and never
        # also pass value= on that widget.
        map_is_archive = False
        if nav_selection == NAV_MAP:
            if st.session_state.get("_active_map_archive_mode") == "Date":
                st.session_state["_active_map_archive_mode"] = "Archive"
            if "map_archive_mode" not in st.session_state:
                st.session_state["map_archive_mode"] = st.session_state.get(
                    "_active_map_archive_mode", "Live",
                )
            elif st.session_state.get("map_archive_mode") == "Date":
                st.session_state["map_archive_mode"] = "Archive"
            st.radio(
                "Map date:",
                ("Live", "Archive"),
                horizontal=True,
                key="map_archive_mode",
                help=HELP["map_archive_date"],
                on_change=_on_map_date_mode_change,
            )
            st.session_state["_active_map_archive_mode"] = st.session_state.map_archive_mode
            map_is_archive = st.session_state.get("map_archive_mode") == "Archive"
            if map_is_archive:
                _max_archive = latest_era5_archive_date().date()
                _min_archive = pd.Timestamp(1940, 1, 1).date()
                if "map_archive_date" not in st.session_state:
                    st.session_state.map_archive_date = st.session_state.get(
                        "_active_map_archive_date", _year_ago_archive_date(),
                    )
                st.session_state.map_archive_date = min(
                    max(st.session_state.map_archive_date, _min_archive), _max_archive,
                )
                st.session_state["_active_map_archive_date"] = st.session_state.map_archive_date
            st.markdown("---")

        if show_expert("forecast_model"):
            st.radio(
                "Forecast Model",
                FORECAST_MODEL_OPTIONS,
                key="forecast_model",
                help=HELP["forecast_model"],
                disabled=map_is_archive,
            )
        else:
            st.session_state.forecast_model = FORECAST_MODEL_IFS
        _fc_tag = "AIFS" if is_aifs_model() else "IFS"
        _cycle = latest_forecast_cycle_label(selected_forecast_model())
        _fc_when = f" run {_cycle}" if _cycle else ""
        _vintage_cls = "atmopulse-data-vintage is-disabled" if map_is_archive else "atmopulse-data-vintage"
        st.markdown(
            f"<p class='{_vintage_cls}'>📡 Data: ERA5 Archive (~ 5 days ago) | "
            f"{_fc_tag} Forecast{_fc_when}.</p>",
            unsafe_allow_html=True,
            help=HELP["data_vintage"],
        )

        # Forecast Offset / Prev-Next-Day step the LIVE target date only.
        # An active Map Archive date ignores the offset entirely (the
        # archive day is stepped above the map, so the Live slider is
        # never silently bent).
        if map_is_archive:
            st.caption("Forecast Offset is inactive while Map date is set to Archive.")
        else:
            st.slider("Forecast Offset (Days):", FORECAST_OFFSET_MIN, FORECAST_OFFSET_MAX, key="offset_slider", help=HELP["forecast_offset"])
            with st.container(key="map_live_day_buttons"):
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.button("Day before  \n←", on_click=sub_day, use_container_width=True)
                with btn_col2:
                    st.button("Day after  \n→", on_click=add_day, use_container_width=True)

        if map_is_archive:
            target_date = pd.Timestamp(st.session_state.map_archive_date)
        else:
            target_date = default_date + pd.Timedelta(days=st.session_state.offset_slider)
            with st.container(key="map_target_date"):
                st.info(f"Target Date: **{target_date.strftime('%d.%m.%Y')}**")
        
        toggles = {}
        
        if nav_selection == NAV_MAP:
            # View is chosen BEFORE the Mapped Variable / Analysis Level
            # controls below: Wave tracking locks TX/TN to the direction in
            # Standard mode and uses its own Strong/Extreme-only level, so
            # both need to know the view first. Shown in BOTH audience
            # modes now (Standard: Daily | Wave; Expert adds Persistence)
            # — LOCKED PRODUCT DECISIONS: "View placement".
            st.markdown("---")
            view_mode = st.radio(
                "**Map view:**",
                MAP_VIEW_OPTIONS_EXPERT if show_expert("persistence_view") else MAP_VIEW_OPTIONS_STANDARD,
                key="map_view_mode",
                help=HELP["map_view_mode"],
            )

            wave_is_warm = is_warm_season
            wave_level = STANDARD_DEFAULTS["map_wave_level"]
            if is_wave_map_view(view_mode):
                st.markdown("---")
                st.markdown("**Wave Direction**", help=HELP["map_wave_direction"])
                if "map_wave_direction" not in st.session_state:
                    st.session_state.map_wave_direction = "Heat" if is_warm_season else "Cold"
                wave_direction = st.radio(
                    "Wave direction", ("Heat", "Cold"),
                    horizontal=True, key="map_wave_direction", label_visibility="collapsed",
                )
                wave_is_warm = wave_direction == "Heat"
                st.markdown("**Wave Level**", help=HELP["map_wave_level"])
                wave_level = st.radio(
                    "Wave level", MAP_WAVE_LEVELS,
                    horizontal=True, key="map_wave_level", label_visibility="collapsed",
                )

            if show_expert("map_tx_tn"):
                st.markdown("---")
                map_var = st.radio(
                    "**Mapped Variable:**",
                    MAP_VAR_OPTIONS,
                    index=0,
                    key="map_var_radio",
                    help=HELP["map_variable"],
                )
                map_var_code = map_var.split('(')[1].strip(')')
            elif is_wave_map_view(view_mode):
                # Standard + Wave tracking: TX(heat)/TN(cold) locked, per
                # LOCKED PRODUCT DECISIONS — never the general TG default.
                map_var_code = wave_locked_var_code(wave_is_warm)
            else:
                map_var = STANDARD_DEFAULTS["map_var"]
                map_var_code = map_var.split('(')[1].strip(')')

            persist_metric = STANDARD_DEFAULTS["persist_metric"]
            if show_expert("map_analysis_level"):
                st.markdown("---")
                top10_threshold = st.radio(
                    "**Analysis Level**",
                    ("Moderate", "Strong", "Extreme", "All-Time Record"),
                    index=1,
                    help=HELP["map_analysis_level"],
                )
            else:
                top10_threshold = STANDARD_DEFAULTS["analysis_level"]
            persist_metric = top10_threshold
            
            if is_daily_map_view(view_mode):
                st.markdown("---")
                st.markdown("**Map Extremes**", help=HELP["map_extremes"])
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    warm_active = any(st.session_state.toggles_warm.values())
                    st.button(
                        "Warm", use_container_width=True,
                        type="primary" if warm_active else "secondary",
                        help=HELP["warm_toggle"], on_click=toggle_warm_state,
                    )
                with m_col2:
                    cold_active = any(st.session_state.toggles_cold.values())
                    st.button(
                        "Cold", use_container_width=True,
                        type="primary" if cold_active else "secondary",
                        help=HELP["cold_toggle"], on_click=toggle_cold_state,
                    )
                if show_expert("percentile_layer_toggles"):
                    st.markdown("<hr style='margin-top:5px; margin-bottom:15px; border-top: 1px dashed gray;'>", unsafe_allow_html=True)
                    st.session_state.toggles_warm["p75"] = st.checkbox(
                        extreme_layer_label("warm", "p75"), value=st.session_state.toggles_warm["p75"],
                    )
                    st.session_state.toggles_warm["p90"] = st.checkbox(
                        extreme_layer_label("warm", "p90"), value=st.session_state.toggles_warm["p90"],
                    )
                    st.session_state.toggles_warm["p95"] = st.checkbox(
                        extreme_layer_label("warm", "p95"), value=st.session_state.toggles_warm["p95"],
                    )
                    st.session_state.toggles_warm["rec"] = st.checkbox(
                        extreme_layer_label("warm", "rec"), value=st.session_state.toggles_warm["rec"],
                    )
                    st.session_state.toggles_cold["p25"] = st.checkbox(
                        extreme_layer_label("cold", "p25"), value=st.session_state.toggles_cold["p25"],
                    )
                    st.session_state.toggles_cold["p10"] = st.checkbox(
                        extreme_layer_label("cold", "p10"), value=st.session_state.toggles_cold["p10"],
                    )
                    st.session_state.toggles_cold["p5"] = st.checkbox(
                        extreme_layer_label("cold", "p5"), value=st.session_state.toggles_cold["p5"],
                    )
                    st.session_state.toggles_cold["rec"] = st.checkbox(
                        extreme_layer_label("cold", "rec"), value=st.session_state.toggles_cold["rec"],
                    )
                else:
                    if any(st.session_state.toggles_warm.values()):
                        for k in st.session_state.toggles_warm:
                            st.session_state.toggles_warm[k] = True
                    if any(st.session_state.toggles_cold.values()):
                        for k in st.session_state.toggles_cold:
                            st.session_state.toggles_cold[k] = True
                st.markdown("---")
                spell_choice = st.selectbox(
                    "Warm/cold spells",
                    SPELL_LABELS,
                    index=SPELL_LABELS.index(STANDARD_DEFAULTS["spell_label"]),
                    key="map_heat_cold_spells",
                    help=HELP["wsdi_csdi_overlay"],
                )
                toggles["hatching"] = spell_choice != SPELL_OFF
                toggles["spell_days"] = (
                    int(spell_choice.split()[0]) if spell_choice != SPELL_OFF else STANDARD_DEFAULTS["spell_days"]
                )

            st.markdown("---")
            with st.container(key="map_synoptic_overlays"):
                toggles["mslp"] = st.checkbox(
                    "Sea-level pressure",
                    value=STANDARD_DEFAULTS["mslp"],
                    help=HELP["mslp_contours"],
                    key="map_overlay_mslp",
                )
                if show_expert("synoptic_anomalies"):
                    toggles["mslp_anom"] = st.checkbox(
                        "Sea-level pressure anomaly",
                        value=STANDARD_DEFAULTS["mslp_anom"],
                        help=HELP["mslp_anomaly"],
                        key="map_overlay_mslp_anom",
                    )
                else:
                    toggles["mslp_anom"] = STANDARD_DEFAULTS["mslp_anom"]
                if show_expert("z500"):
                    # Leading word-joiner: Streamlit markdown otherwise treats a
                    # label that starts with digits as a numbered list and renders
                    # it at a larger size than the MSLP checkbox.
                    toggles["z500"] = st.checkbox(
                        "\u200b500 hPa height",
                        value=STANDARD_DEFAULTS["z500"],
                        help=HELP["z500_contours"],
                        key="map_overlay_z500",
                    )
                else:
                    toggles["z500"] = STANDARD_DEFAULTS["z500"]
                if show_expert("synoptic_anomalies"):
                    toggles["z500_anom"] = st.checkbox(
                        "\u200b500 hPa height anomaly",
                        value=STANDARD_DEFAULTS["z500_anom"],
                        help=HELP["z500_anomaly"],
                        key="map_overlay_z500_anom",
                    )
                else:
                    toggles["z500_anom"] = STANDARD_DEFAULTS["z500_anom"]
                if show_expert("jet"):
                    # Leading word-joiner: see the Z500 checkbox above --
                    # otherwise Streamlit renders a label starting with a
                    # digit as a numbered list item at a larger font size.
                    toggles["jet"] = st.checkbox(
                        "\u200b300 hPa jet",
                        value=STANDARD_DEFAULTS["jet"],
                        help=HELP["jet_wind"],
                        key="map_overlay_jet",
                    )
                else:
                    toggles["jet"] = STANDARD_DEFAULTS["jet"]
                if show_expert("synoptic_anomalies"):
                    st.caption(
                        "Anomaly isolines are relative to each map's reference period. "
                        "Solid = above that DOY mean, dashed = below."
                    )
            
        elif nav_selection in (NAV_METEO, NAV_WAVE):
            st.markdown("---")
            st.markdown("**Location Settings**")
            
            if nav_selection == NAV_METEO:
                if show_expert("meteo_tx_tn") or show_expert("t850"):
                    meteo_var = st.radio(
                        "Variable:",
                        MAP_VAR_OPTIONS,
                        help=HELP["map_variable"],
                    )
                else:
                    meteo_var = STANDARD_DEFAULTS["meteo_var"]
                if show_expert("meteo_envelope"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    meteo_env = st.selectbox("Background Envelope:", ["Moderate", "Strong", "Extreme", "All-Time"], index=1, help=HELP["meteogram_envelope"])
                else:
                    meteo_env = STANDARD_DEFAULTS["meteo_env"]
                if show_expert("meteo_wsdi_csdi"):
                    meteo_count = st.radio(
                        "Select between:",
                        METEO_COUNT_OPTIONS,
                        help=HELP["meteo_annual_count"],
                        key="meteo_annual_count",
                    )
                    meteo_spell = meteo_count == METEO_COUNT_SPELL
                else:
                    meteo_spell = False
            
            if nav_selection == NAV_WAVE:
                wave_focus = st.radio("Wave Event Type:", ("Heatwaves", "Coldwaves"), index=default_wave_idx, help=HELP["wave_event_type"])
                is_warm = "Heatwaves" in wave_focus
                if show_expert("meteo_tx_tn") or show_expert("t850"):
                    if "wave_var" not in st.session_state:
                        st.session_state.wave_var = (
                            "Maximum Temperature (TX)" if is_warm else "Minimum Temperature (TN)"
                        )
                    wave_var = st.radio(
                        "Variable:",
                        MAP_VAR_OPTIONS,
                        key="wave_var",
                        help=HELP["map_variable"],
                    )
                else:
                    wave_var = "Maximum Temperature (TX)" if is_warm else "Minimum Temperature (TN)"
                wave_thresh = st.radio("Wave Intensity Threshold:", ("Strong", "Extreme"), help=HELP["wave_intensity_threshold"])
                wave_stack_metric = st.radio(
                    "Wave statistic:",
                    ("Intensity", "Days"),
                    index=0,
                    help=HELP["wave_stack_metric"],
                    key="wave_stack_metric",
                )
                if show_expert("z500"):
                    wave_z500_outline = st.checkbox(
                        "\u200b500 hPa ridge/trough outline",
                        value=STANDARD_DEFAULTS["wave_z500_outline"],
                        help=HELP["wave_z500_outline"],
                        key="wave_z500_outline",
                    )
                else:
                    wave_z500_outline = STANDARD_DEFAULTS["wave_z500_outline"]

        st.markdown(
            f"<p class='atmopulse-panel-credit'>{_output_credit_text()}</p>",
            unsafe_allow_html=True,
        )

if nav_selection == NAV_WELCOME:
    st.markdown(f"### Welcome to {atmopulse_wordmark_html()}", unsafe_allow_html=True)
    st.markdown(f"""
    **{atmopulse_wordmark_html()}** merges real-time extreme weather tracking with shifting climate baselines. It provides interactive, synoptic-scale mapping and deep-dive local profiles. Currently focused on extreme temperatures, {atmopulse_wordmark_html()} aims to integrate further atmospheric variables in the future.
    <br><br>
    #### Understanding Percentiles
    {atmopulse_wordmark_html()} relies heavily on percentiles to contextualize current weather against historical norms. In our maps and meteograms, percentiles are calculated using a **centered 5-day moving window** across the reference periods (1961–1990 and 1996–2025). 
    For instance, the 90th percentile (P90) is a threshold exceeded only 10% of the time during the historical baseline. We track **Moderate** (P75/P25), **Strong** (P90/P10), and **Extreme** (P95/P5) thresholds to dynamically classify the severity of synoptic events.
    <br><br>
    #### The Importance of Event Duration
    The impact of extreme temperatures on sectors like human health, agriculture and infrastructure scales drastically with duration. A single hot day is a weather event; a prolonged sequence becomes a systemic hazard. 
    In the **Map Tracker** tab, you can visualize this through the **Persistence duration** layer, showing how many consecutive days an extreme has lasted (unbroken run back from the map date, up to 100 days). Daily maps can optionally show a **warm/cold spell** overlay for 6, 15 or 30 consecutive days (off by default; enable it in the sidebar). 6 days matches the WSDI/CSDI definition.
    <br><br>
    #### Local Wave Definitions
    In the **Point Wavogram** tab, {atmopulse_wordmark_html()} uses a sophisticated definition (adapted from Kyselý) to track heatwaves and coldwaves:
    * **Heatwaves:** Detected year-round. Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) threshold for at least 3 consecutive days. The ridge focuses on May–September and widens if an event falls outside those months.
    * **Coldwaves:** Detected from 1 July to 30 June. Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) threshold for at least 3 consecutive days. The ridge focuses on November–March and widens if an event falls outside those months.

    The **Map Tracker** tab's **Wave tracking** view applies this exact same Kyselý definition spatially — every grid cell across Europe, not just one point — so you can see where an active heatwave or coldwave currently covers the continent, and how intensely (accumulated excess-temperature "K·days") each cell has been inside it. **Strong** and **Extreme** are independent detections there (different threshold pairs), not nested severity tiers of one event.
    """, unsafe_allow_html=True)
    
    img_col1, img_col2 = st.columns(2)
    _warm_img = _documents_image("Warm.JPG", "Warm.jpg")
    _kalt_img = _documents_image("Kalt.JPG", "Kalt.jpg")
    with img_col1:
        if _warm_img:
            st.image(_warm_img, use_container_width=True, caption="Erfassung von Hitzewellen")
        else:
            st.caption("Warm example image not found in Documents/.")
    with img_col2:
        if _kalt_img:
            st.image(_kalt_img, use_container_width=True, caption="Erfassung von Kältewellen")
        else:
            st.caption("Cold example image not found in Documents/.")

elif nav_selection == NAV_MAP:
    render_map_tracker(
        map_var_code, view_mode, persist_metric, top10_threshold, toggles, target_date, default_date,
        map_is_archive=map_is_archive,
        map_anchor_date=(target_date if map_is_archive else default_date),
        wave_is_warm=wave_is_warm, wave_level=wave_level,
    )

elif nav_selection in (NAV_METEO, NAV_WAVE):
    # The location box only exists on these pages. Restore the last choice
    # from a plain shadow key so a detour through Map Tracker does not wipe it.
    if "loc_field" not in st.session_state and st.session_state.get("_active_loc_field"):
        st.session_state.loc_field = st.session_state["_active_loc_field"]

    saved = [
        item for item in st.session_state.saved_locations
        if item.get("lat") is not None and item.get("lon") is not None
    ]
    saved_labels = [_saved_loc_label(item) for item in saved]
    saved_by_label = dict(zip(saved_labels, saved))

    current = st.session_state.get("loc_field")
    pending = list(st.session_state.get("loc_pending_matches") or [])
    if current in saved_labels:
        pending = []
        st.session_state.loc_pending_matches = []
    elif current and current not in {item["label"] for item in pending}:
        name = str(current).split(" (")[0].split(",")[0].strip()
        known = next(
            (
                label for label in saved_labels
                if label.startswith(name + " (") or label.startswith(name + ", ")
            ),
            None,
        )
        if known:
            st.session_state.loc_field = known
            pending = []
        elif name:
            with st.spinner("Searching..."):
                found = _safe_geocode(name, exactly_one=False, limit=5) or []
            pending = []
            for hit in found:
                short = hit.address.split(",")[0].strip()
                pending.append({
                    "label": _saved_loc_label({
                        "name": short, "address": hit.address,
                        "lat": hit.latitude, "lon": hit.longitude,
                    }),
                    "name": short,
                    "address": hit.address,
                    "lat": hit.latitude,
                    "lon": hit.longitude,
                })
            if pending:
                st.session_state.loc_field = pending[0]["label"]
            else:
                st.session_state.loc_field = None
                st.warning("No results inside the map area.")
        st.session_state.loc_pending_matches = pending

    pending_labels = [item["label"] for item in pending]
    options = list(dict.fromkeys([*pending_labels, *saved_labels]))
    if st.session_state.get("loc_field") not in options:
        st.session_state.pop("loc_field", None)
    select_kwargs = {
        "key": "loc_field",
        "accept_new_options": True,
        "placeholder": "Select a location",
    }
    if "loc_field" not in st.session_state:
        select_kwargs["index"] = None
    with st.container(key="target_location_row"):
        loc_head, loc_field_col = st.columns([1, 3], vertical_alignment="center", gap="small")
        with loc_head:
            st.markdown(
                "<p class='atmopulse-target-location-label'>Target Location:</p>",
                unsafe_allow_html=True,
            )
        with loc_field_col:
            choice = st.selectbox(
                "Select a location",
                options,
                label_visibility="collapsed",
                **select_kwargs,
            ) if options else None
    st.session_state["_active_loc_field"] = choice

    location = None
    if choice in saved_by_label:
        chosen_saved = saved_by_label[choice]
        st.session_state.loc_pending_matches = []
        location = _PointLocation(chosen_saved["address"], chosen_saved["lat"], chosen_saved["lon"])
    else:
        pending_hit = next((item for item in pending if item["label"] == choice), None)
        if pending_hit is not None:
            remembered = _remember_location(
                pending_hit["name"], pending_hit["address"], pending_hit["lat"], pending_hit["lon"],
            )
            location = _PointLocation(remembered["address"], remembered["lat"], remembered["lon"])

    lat_target, lon_target = 52.52, 13.40
    if location:
        lat_target, lon_target = round(location.latitude, 2), round(location.longitude, 2)
        if not _in_map_domain(lat_target, lon_target):
            st.warning(f"📍 Location {location.address} is outside the map area.")
            location = None

    if location:
        # Non-widget keys, persisted across nav changes so the Map Tracker's
        # own location marker can draw a small circle+name once a location
        # has actually been chosen — never a hardcoded fallback point
        # (LOCKED PRODUCT DECISIONS: "no marker until location actually
        # chosen (no Berlin 52.52,13.40 fallback)").
        st.session_state["map_marker_lat"] = lat_target
        st.session_state["map_marker_lon"] = lon_target
        st.session_state["map_marker_name"] = location.address

        render_grid_cell_profile(location.address, lat_target, lon_target)
        if nav_selection == NAV_METEO:
            render_meteogram(
                location, lat_target, lon_target, meteo_var, meteo_env, target_date,
                meteo_spell=meteo_spell,
            )

        elif nav_selection == NAV_WAVE:
            _chart_layouts = (LAYOUT_SINGLE_CHART, LAYOUT_SIDE_BY_SIDE)
            if st.session_state.get("wave_layout") not in _chart_layouts:
                st.session_state.wave_layout = LAYOUT_SINGLE_CHART
            _period_choice = (epoch_short_label("A"), epoch_short_label("B"))
            show_period = st.session_state.get("wave_layout", LAYOUT_SINGLE_CHART) == LAYOUT_SINGLE_CHART
            if show_period and st.session_state.get("wave_ep") not in _period_choice:
                raw = st.session_state.get("wave_ep", "")
                st.session_state.wave_ep = (
                    _period_choice[0] if epoch_from_label(raw or "B") == "A" else _period_choice[1]
                )
            with st.container(key="wave_top_controls"):
                wave_cols = st.columns(2 if show_period else 1)
                with wave_cols[0]:
                    map_layout = st.radio(
                        "Layout:",
                        _chart_layouts,
                        horizontal=True,
                        key="wave_layout",
                    )
                if show_period:
                    with wave_cols[1]:
                        st.radio(
                            "Select Reference Period:",
                            _period_choice,
                            horizontal=True,
                            key="wave_ep",
                        )
            param_code = meteo_var_code(wave_var)
            if is_aifs_model() and param_code in ("TX", "TN"):
                st.warning(AIFS_TXTN_WARNING)
            else:
                with st.spinner("Generating Historical Waves..."):
                    wave_payload_a = _compute_wave_payload(
                        lat_target, lon_target, param_code, "A", wave_thresh, is_warm=is_warm,
                    )
                    wave_payload_b = _compute_wave_payload(
                        lat_target, lon_target, param_code, "B", wave_thresh, is_warm=is_warm,
                    )
                    from frontend_plots import build_kysely_wave_figs, union_wave_xrange

                    def _wave_figs(payload, x_range):
                        return build_kysely_wave_figs(
                            payload, z500_outline=wave_z500_outline,
                            stack_metric=wave_stack_metric, x_range=x_range,
                        )

                    side_by_side = map_layout == LAYOUT_SIDE_BY_SIDE
                    if side_by_side:
                        shared_x = union_wave_xrange(wave_payload_a, wave_payload_b)
                        (fig_m_a, fig_s_a, fig_f_a) = _wave_figs(wave_payload_a, shared_x)
                        (fig_m_b, fig_s_b, fig_f_b) = _wave_figs(wave_payload_b, shared_x)
                    else:
                        use_a_build = epoch_from_label(
                            st.session_state.get("wave_ep", _period_choice[1])
                        ) == "A"
                        shown = wave_payload_a if use_a_build else wave_payload_b
                        fig_m_a, fig_s_a, fig_f_a = _wave_figs(shown, None)
                        fig_m_b, fig_s_b, fig_f_b = fig_m_a, fig_s_a, fig_f_a
                    wave_drill_ctx = (lat_target, lon_target, is_warm, param_code, wave_thresh)

                if side_by_side and fig_s_a.data and fig_s_b.data:
                    align_wave_stats_yranges(fig_s_a, fig_s_b, fig_f_a, fig_f_b)

                stack_heading = (
                    "Intensity [days]"
                    if str(wave_stack_metric).lower().startswith("day")
                    else "Intensity [K]"
                )
                wave_kind = "Heatwaves" if is_warm else "Coldwaves"
                var_bit = MAP_VAR_LABELS.get(param_code, param_code)

                def _wave_export_title(panel: str, epoch: str) -> str:
                    return (
                        f"{panel} · {var_bit} ({param_code}) {wave_kind} | "
                        f"{epoch_period_label(epoch)}"
                    )

                def _show_wave_rank(epoch: str) -> None:
                    banner = render_point_wavogram_narrative(
                        location.address, lat_target, lon_target,
                        parameter=param_code, selected_epoch=epoch,
                        threshold_level=wave_thresh, target_date=target_date,
                        is_warm=is_warm,
                    )
                    if banner:
                        st.markdown(banner, unsafe_allow_html=True)

                if map_layout == LAYOUT_SIDE_BY_SIDE:
                    rank_l, rank_r = st.columns(2, gap="small")
                    with rank_l:
                        _show_wave_rank("A")
                    with rank_r:
                        _show_wave_rank("B")
                    with st.container(key="atmopulse_split_wave_ridge"):
                        w_col1, w_col2 = st.columns(2, gap="small")
                        with w_col1:
                            click_a = st_plotly_press(
                                fig_m_a, "wavogram_ridge_historical",
                                on_select="rerun", selection_mode=("points",), key="wave_ridge_click_a",
                            )
                        with w_col2:
                            click_b = st_plotly_press(
                                fig_m_b, "wavogram_ridge_recent",
                                on_select="rerun", selection_mode=("points",), key="wave_ridge_click_b",
                            )
                    # One Event Drill-down block under both ridges. Side by side
                    # shows both periods in the ridges; the drill-down follows
                    # the recent period and names it in the heading.
                    _render_wave_drilldown(
                        wave_payload_a, wave_payload_b, wave_stack_metric, wave_drill_ctx,
                        click_events=(
                            ("wave_ridge_click_a", click_a),
                            ("wave_ridge_click_b", click_b),
                        ),
                        epoch="B",
                    )
                    # Fresh column pair for the stack row: reusing w_col1/
                    # w_col2 here would silently push the drill-down block
                    # (written above, outside those columns) below BOTH
                    # column blocks — Streamlit renders everything ever
                    # written into a given `st.columns()` pair as one
                    # contiguous row wherever that pair was first created,
                    # so it can't be interleaved with content in between.
                    with st.container(key="atmopulse_split_wave_stats"):
                        st_col1, st_col2 = st.columns(2, gap="small")
                        with st_col1:
                            st.markdown(
                                f"**{stack_heading}**",
                                help=HELP["wave_annual_stack"],
                            )
                            st_plotly_press(
                                fig_s_a, "wavogram_stats_historical",
                                export_title=_wave_export_title(stack_heading, "A"),
                            )
                            if show_expert("wave_annual_cycle"):
                                _wave_section_spacer()
                                st.markdown(
                                    "**Frequency [%]**",
                                    help=HELP["wave_annual_cycle"],
                                )
                                st_plotly_press(
                                    fig_f_a, "wavogram_freq_historical",
                                    export_title=_wave_export_title("Frequency [%]", "A"),
                                )
                        with st_col2:
                            st.markdown(
                                f"**{stack_heading}**",
                                help=HELP["wave_annual_stack"],
                            )
                            st_plotly_press(
                                fig_s_b, "wavogram_stats_recent",
                                export_title=_wave_export_title(stack_heading, "B"),
                            )
                            if show_expert("wave_annual_cycle"):
                                _wave_section_spacer()
                                st.markdown(
                                    "**Frequency [%]**",
                                    help=HELP["wave_annual_cycle"],
                                )
                                st_plotly_press(
                                    fig_f_b, "wavogram_freq_recent",
                                    export_title=_wave_export_title("Frequency [%]", "B"),
                                )
                else:
                    use_a = epoch_from_label(st.session_state.get("wave_ep", _period_choice[1])) == "A"
                    ep = "A" if use_a else "B"
                    _show_wave_rank(ep)
                    click_ep = st_plotly_press(
                        fig_m_a if use_a else fig_m_b, f"wavogram_ridge_{ep}",
                        on_select="rerun", selection_mode=("points",), key=f"wave_ridge_click_{ep}",
                    )
                    _render_wave_drilldown(
                        wave_payload_a, wave_payload_b, wave_stack_metric, wave_drill_ctx,
                        click_events=((f"wave_ridge_click_{ep}", click_ep),),
                        epoch=ep,
                    )
                    st.markdown(
                        f"**{stack_heading}**",
                        help=HELP["wave_annual_stack"],
                    )
                    st_plotly_press(
                        fig_s_a if use_a else fig_s_b, f"wavogram_stats_{ep}",
                        export_title=_wave_export_title(stack_heading, ep),
                    )
                    if show_expert("wave_annual_cycle"):
                        _wave_section_spacer()
                        st.markdown(
                            "**Frequency [%]**",
                            help=HELP["wave_annual_cycle"],
                        )
                        st_plotly_press(
                            fig_f_a if use_a else fig_f_b, f"wavogram_freq_{ep}",
                            export_title=_wave_export_title("Frequency [%]", ep),
                        )

                if show_expert("z500") and wave_z500_outline:
                    st.caption(HELP["wave_z500_outline"])

elif nav_selection == NAV_METHODS:
    methods_md = _load_markdown_page(METHODS_MD)
    if methods_md:
        st.markdown(methods_md)
    else:
        st.markdown("### Methods & Resources")

elif nav_selection == NAV_LEGAL:
    legal_md = _load_markdown_page(LEGAL_MD)
    if legal_md:
        st.markdown(legal_md, unsafe_allow_html=True)
    else:
        st.error("Legal text not found. Expected `assets/legal.md`.")