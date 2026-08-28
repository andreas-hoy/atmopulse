"""
AtmoPulse Automated Narrative Generation (backend_narrative.py)

This module generates the auto-summary text/metrics shown alongside the Map
Tracker, Point Meteogram, and Point Wavogram views. It is the single place
where "what does the UI say" is derived, and it is deliberately written so
the generated text can never drift from what the charts show:

- Map Tracker / Point Meteogram narratives strictly branch on the caller's
  `ui_state` ("single_*" vs "compare_*"), mirroring the exact UI toggle
  (`map_layout` / `met_layout` in app.py) that decided what was rendered.
- Point Wavogram narratives are intentionally STATIC with respect to that
  same UI state: they always evaluate the full, continuous ERA5 record
  since 1940 via `backend_waves.get_wave_historical_rank`, independent of
  which baseline/layout is on screen.

All spatial statistics are area-weighted with cos(lat) (a flat lat/lon grid
over-represents high latitudes) and fully vectorized with NumPy — no Python
loops over grid cells.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from backend_waves import get_wave_historical_rank

EPOCH_LABELS: dict[str, str] = {"A": "1961–1990", "B": "1996–2025"}

# Mask codes as produced by app.py::_build_display_mask(). Codes are ordinal
# by severity WITHIN each direction: warm ascends 5->8, cold descends 4->1.
# _build_display_mask() already overwrites low->high severity per cell, so
# a cell's final code is its SINGLE highest severity reached — cumulative
# ("at least tier T") therefore means "code at least as severe as tier T's
# code", i.e. mask >= warm_code (warm) or mask <= cold_code (cold).
_WARM_CODE: dict[str, int] = {"moderate": 5, "strong": 6, "extreme": 7, "record": 8}
_COLD_CODE: dict[str, int] = {"moderate": 4, "strong": 3, "extreme": 2, "record": 1}
_TIER_ORDER = ("moderate", "strong", "extreme", "record")
_TIER_LABEL = {
    "moderate": "moderate (P75/25)",
    "strong": "strong (P90/10)",
    "extreme": "extreme (P95/5)",
    "record": "record-breaking",
}


# ---------------------------------------------------------------------------
# 1. MAP TRACKER — area-weighted spatial percentage (UI-state sensitive)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def spatial_extreme_footprint(
    mask: np.ndarray, valid_domain: np.ndarray, lat2d: np.ndarray,
    *, target_date: str = "", baseline_name: str = "",
) -> dict:
    """
    Area-weighted (cos(lat)), CUMULATIVE percentage of the European domain
    classified at or beyond each severity tier of the map's discrete extreme
    mask (see app.py::_build_display_mask):

    - Record  = area at the Record threshold ONLY.
    - Extreme = area at/beyond the Extreme threshold (INCLUDES Record).
    - Strong  = area at/beyond the Strong threshold (INCLUDES Extreme + Record).
    - Moderate = area at/beyond the Moderate threshold (INCLUDES Strong + Extreme + Record).

    Cached (@st.cache_data): `mask`/`valid_domain`/`lat2d` are the small,
    map-resolution NumPy arrays that fully determine the result, and
    `target_date`/`baseline_name` are cheap string cache-key context — so
    Streamlit hashes only lightweight inputs and skips recomputation on any
    UI rerun (e.g. flipping the Map Layout radio) where none of these
    actually changed.

    Parameters
    ----------
    mask : ndarray
        Discrete classification grid (values 1-8) from `_build_display_mask`;
        NaN where no active layer classifies the cell.
    valid_domain : ndarray (bool-like)
        Marks cells physically inside the domain (e.g. `np.isfinite(v_curr)`).
        Kept separate from `mask` because `mask` is NaN both for "outside the
        domain" AND for "inside the domain but not extreme" — conflating the
        two would silently shrink the denominator and inflate every percentage.
    lat2d : ndarray
        2D latitude grid (degrees), broadcastable to `mask`'s shape.
    target_date, baseline_name : str
        Cheap, human-legible cache-key context (e.g. "2026-08-28", "A").
        Not used in the numeric core — the arrays alone determine the output
        — but keeps cache entries stable and easy to reason about per call site.

    Returns
    -------
    dict: {"moderate": {...}, "strong": {...}, "extreme": {...}, "record": {...}},
    each with "warm_pct", "cold_pct", "total_pct" (float, 0-100, area-weighted,
    CUMULATIVE at/beyond that tier).
    """
    mask = np.asarray(mask, dtype=np.float64)
    domain = np.asarray(valid_domain, dtype=bool)
    weights = np.cos(np.deg2rad(np.asarray(lat2d, dtype=np.float64)))

    total_weight = float(np.sum(weights, where=domain)) if domain.any() else 0.0
    empty = {"warm_pct": 0.0, "cold_pct": 0.0, "total_pct": 0.0}
    if total_weight <= 0:
        return {tier: dict(empty) for tier in _TIER_ORDER}

    out = {}
    for tier in _TIER_ORDER:
        # Cumulative masks: warm severity ascends with the code (>=), cold
        # severity ascends as the code descends (<=). NaN cells naturally
        # fail both comparisons (IEEE754), so no separate isfinite() guard
        # is needed on top of `domain`.
        warm_cum = mask >= _WARM_CODE[tier]
        cold_cum = mask <= _COLD_CODE[tier]
        warm_w = float(np.sum(weights, where=domain & warm_cum))
        cold_w = float(np.sum(weights, where=domain & cold_cum))
        out[tier] = {
            "warm_pct": 100.0 * warm_w / total_weight,
            "cold_pct": 100.0 * cold_w / total_weight,
            "total_pct": 100.0 * (warm_w + cold_w) / total_weight,
        }
    return out


def render_map_tracker_narrative(
    ui_state: str,
    *,
    direction: str = "total_pct",
    footprint_single: dict | None = None,
    epoch_single: str = "B",
    footprint_a: dict | None = None,
    footprint_b: dict | None = None,
) -> None:
    """
    UI-state-sensitive Map Tracker narrative. Strictly listens to `ui_state`
    — never both branches at once, so the metric/table on screen always
    matches the map layout actually rendered (`LAYOUT_FLICKER` vs
    `LAYOUT_SIDE_BY_SIDE` / `LAYOUT_OPACITY` in app.py).

    Always displays ALL FOUR cumulative severity tiers (Moderate, Strong,
    Extreme, Record) — Moderate INCLUDES Strong/Extreme/Record, Strong
    INCLUDES Extreme/Record, Extreme INCLUDES Record, Record is exclusive.

    Parameters
    ----------
    ui_state : "single_map" | "compare_maps"
    direction : "warm_pct" | "cold_pct" | "total_pct" — which slice of
        `spatial_extreme_footprint()`'s output to narrate.
    footprint_single, footprint_a, footprint_b : outputs of
        `spatial_extreme_footprint()` for the relevant baseline(s).
    epoch_single : "A" | "B" — which baseline is active in single-map mode.
    """
    if ui_state == "single_map":
        if footprint_single is None:
            st.info("No map data available for the current selection.")
            return
        epoch_label = EPOCH_LABELS[epoch_single]
        st.markdown(f"**Based on the {epoch_label} baseline**, cumulative area of Europe affected:")
        cols = st.columns(len(_TIER_ORDER))
        for col, tier in zip(cols, _TIER_ORDER):
            with col:
                st.metric(tier.title(), f"{footprint_single[tier][direction]:.1f}%")

    elif ui_state == "compare_maps":
        if footprint_a is None or footprint_b is None:
            st.info("No map data available for the current selection.")
            return
        render_map_tracker_table({"A": footprint_a, "B": footprint_b}, direction=direction)
    else:
        raise ValueError(f"Unknown ui_state for Map Tracker narrative: {ui_state!r}")


def render_map_tracker_table(
    footprints: dict[str, dict], epoch_labels: dict[str, str] | None = None, direction: str = "total_pct",
) -> None:
    """
    Markdown-table Map Tracker output listing ALL FOUR cumulative severity
    tiers (Moderate, Strong, Extreme, Record) for one or more baselines side
    by side. `footprints` maps an epoch key (e.g. "A", "B") to the output of
    `spatial_extreme_footprint()`.
    """
    epoch_labels = epoch_labels or EPOCH_LABELS
    keys = list(footprints)
    header = "| Severity (cumulative) | " + " | ".join(epoch_labels.get(k, k) for k in keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    rows = [header, sep]
    for tier in _TIER_ORDER:
        cells = [f"{footprints[k][tier][direction]:.1f}%" for k in keys]
        rows.append(f"| {tier.title()} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(rows))


# ---------------------------------------------------------------------------
# 2. POINT METEOGRAM — current time-step classification (UI-state sensitive)
# ---------------------------------------------------------------------------

def _is_finite(x) -> bool:
    return x is not None and np.isfinite(x)


def classify_point_severity(value: float, p_warm: dict, p_cold: dict) -> tuple[str, str | None]:
    """
    Point-in-time ETCCDI-percentile classification for a single coordinate.

    `value` MUST be the exact scalar temperature for the specific
    `target_date` being narrated, and `p_warm`/`p_cold` MUST be the exact
    scalar percentile thresholds for that same calendar day — never an
    aggregate (e.g. `.max()`) over the whole forecast/live-series array,
    which would silently narrate the wrong day's extremum.

    STRICT TOP-DOWN climatological structure (warm ladder entirely, most
    extreme first, then the cold ladder, most extreme first, then Normal as
    the final catch-all strictly between P25 and P75 exclusive):

        if   value >= record_max: ("record", "warm")
        elif value >= P95:        ("extreme", "warm")
        elif value >= P90:        ("strong", "warm")
        elif value >  P75:        ("moderate", "warm")
        elif value <= record_min: ("record", "cold")
        elif value <= P5:         ("extreme", "cold")
        elif value <= P10:        ("strong", "cold")
        elif value <  P25:        ("moderate", "cold")
        else:                     ("normal", None)   # strictly within [P25, P75]

    p_warm / p_cold: dicts with keys "p75"/"p25", "p90"/"p10", "p95"/"p5",
    "rec" (already correctly signed thresholds for that direction).

    Returns (tier, direction), e.g. ("strong", "warm") or ("normal", None).
    Callers can build the exact condition string via `f"{tier} {direction}"`.
    """
    if not np.isfinite(value):
        return "normal", None

    rec_w, p95, p90, p75 = p_warm.get("rec"), p_warm.get("p95"), p_warm.get("p90"), p_warm.get("p75")
    rec_c, p5, p10, p25 = p_cold.get("rec"), p_cold.get("p5"), p_cold.get("p10"), p_cold.get("p25")

    if _is_finite(rec_w) and value >= rec_w:
        return "record", "warm"
    elif _is_finite(p95) and value >= p95:
        return "extreme", "warm"
    elif _is_finite(p90) and value >= p90:
        return "strong", "warm"
    elif _is_finite(p75) and value > p75:
        return "moderate", "warm"
    elif _is_finite(rec_c) and value <= rec_c:
        return "record", "cold"
    elif _is_finite(p5) and value <= p5:
        return "extreme", "cold"
    elif _is_finite(p10) and value <= p10:
        return "strong", "cold"
    elif _is_finite(p25) and value < p25:
        return "moderate", "cold"
    else:
        return "normal", None


def render_point_meteogram_narrative(
    ui_state: str,
    location_name: str,
    value: float,
    p_warm_b: dict,
    p_cold_b: dict,
    p_warm_a: dict | None = None,
    p_cold_a: dict | None = None,
) -> tuple:
    """
    UI-state-sensitive Point Meteogram narrative for the current time-step.

    - "single_baseline": text anchored ONLY to the active reference period
      (the thresholds passed as `p_warm_b`/`p_cold_b`).
    - "compare_baselines": additionally classifies against `p_warm_a`/
      `p_cold_a` and quantifies the shift between both baselines.

    Returns the classification(s) so callers can reuse them (e.g. for a
    colored badge) without reclassifying.
    """
    cat_b, dir_b = classify_point_severity(value, p_warm_b, p_cold_b)

    if ui_state == "single_baseline":
        if cat_b == "normal":
            st.markdown(
                f"The area of **{location_name}** is currently within its normal range relative to "
                f"the **{EPOCH_LABELS['B']}** baseline."
            )
        else:
            st.markdown(
                f"The area of **{location_name}** is currently experiencing **{cat_b} {dir_b}** "
                f"conditions relative to the **{EPOCH_LABELS['B']}** baseline."
            )
        return cat_b, dir_b

    if ui_state == "compare_baselines":
        if p_warm_a is None or p_cold_a is None:
            raise ValueError("compare_baselines requires p_warm_a/p_cold_a thresholds")
        cat_a, dir_a = classify_point_severity(value, p_warm_a, p_cold_a)

        b_txt = "within its normal range" if cat_b == "normal" else f"**{cat_b} {dir_b}** conditions"
        a_txt = "within its normal range" if cat_a == "normal" else f"**{cat_a} {dir_a}** conditions"
        st.markdown(
            f"The area of **{location_name}** is currently experiencing {b_txt} relative to the "
            f"**{EPOCH_LABELS['B']}** baseline. However, compared to the historical **{EPOCH_LABELS['A']}** "
            f"baseline, this equates to {a_txt}."
        )
        return (cat_b, dir_b), (cat_a, dir_a)

    raise ValueError(f"Unknown ui_state for Point Meteogram narrative: {ui_state!r}")


# ---------------------------------------------------------------------------
# 3. POINT WAVOGRAM — historical ranking since 1940 (STATIC, UI-state agnostic)
# ---------------------------------------------------------------------------

def render_point_wavogram_narrative(
    location_name: str,
    lat: float,
    lon: float,
    parameter: str = "TX",
    selected_epoch: str = "B",
    threshold_level: str = "Strong (P90/10)",
    target_date=None,
) -> dict | None:
    """
    Point Wavogram narrative. Deliberately STATIC with respect to `ui_state`
    (map layout / baseline compare toggles): it always ranks the current
    event against the full, continuous ERA5 record since 1940 via
    `backend_waves.get_wave_historical_rank`, which does not take a UI
    layout argument at all — there is nothing here for the visual overlay
    state to change.

    Renders nothing (returns None) unless there is a currently active
    heatwave/coldwave AND it ranks in the Top 20 longest events on record.
    """
    rank_info = get_wave_historical_rank(
        lat, lon, parameter=parameter, selected_epoch=selected_epoch,
        threshold_level=threshold_level, target_date=target_date,
    )
    if rank_info is None:
        return None

    st.markdown(
        f"The area of **{location_name}** is currently experiencing its "
        f"**{rank_info['rank_ordinal']} longest** {rank_info['severity']} {rank_info['wave_type']} "
        f"since the start of the ERA5 record in 1940."
    )
    return rank_info
