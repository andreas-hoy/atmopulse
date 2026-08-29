"""
AtmoPulse Map & Point Analytics (backend_analytics.py)

Cached, pure-data analytics extracted from app.py: the area-weighted spatial
extreme-footprint percentages behind the Map Tracker narrative, and the
Top-10 country impact ranking. Both mirror frontend_plots.build_baseline_map's
value/threshold retrieval and mask construction exactly, so narrated and
tabulated results always match the rendered map.

`get_persistence_arrays` / `get_country_weight_grid` are still owned by
app.py (they depend on the live/archive dataset loaders defined there); they
are imported locally, inside the functions that need them, to avoid a
circular import at module load time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backend_maps import etccdi_doy_365
from backend_narrative import spatial_extreme_footprint
from config import DATA_ROOT, TOP10_MASK_VERSION, TOP10_MIN_PCT, is_daily_map_view, selected_forecast_model

# --- Pre-computed Analytics (batch_precompute_analytics.py) ---
PRECOMPUTED_ANALYTICS_DIR: Path = DATA_ROOT / "Precomputed_Analytics"

# Footprint results are threshold-independent (compute_map_footprint returns
# all four tiers at once) — batch_precompute_analytics.py nonetheless writes
# one identical copy per threshold to satisfy the shared naming convention
# with the Top-10 files. Any of the four is equally valid on read; this is
# just the fixed one the lazy-loading wrapper below looks for.
_FOOTPRINT_CACHE_THRESHOLD = "Strong"


def _synoptic_array(field):
    if field is None:
        return None
    if isinstance(field, np.ndarray):
        return field
    return np.asarray(getattr(field, "values", field))


def _synoptic_temp_pair(map_phys_data):
    """TX/TN arrays for the map renderer; fall back to TG when extremes are absent (AIFS)."""
    tx = map_phys_data.get("tx")
    tn = map_phys_data.get("tn")
    if tx is not None and tn is not None:
        return _synoptic_array(tx), _synoptic_array(tn)
    tg = map_phys_data.get("tg")
    if tg is not None:
        arr = _synoptic_array(tg)
        return arr, arr
    sample = next((map_phys_data[k] for k in ("mslp", "z500") if k in map_phys_data), None)
    if sample is None:
        return None, None
    nan = np.full(np.asarray(getattr(sample, "values", sample)).shape, np.nan)
    return nan, nan


def _synoptic_lonlat(map_phys_data):
    if map_phys_data is None:
        return None, None
    if "_lons" in map_phys_data and "_lats" in map_phys_data:
        return np.asarray(map_phys_data["_lons"]), np.asarray(map_phys_data["_lats"])
    sample = map_phys_data.get("mslp", map_phys_data.get("tg", map_phys_data.get("tx")))
    return np.asarray(sample.longitude.values), np.asarray(sample.latitude.values)


def _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold):
    """Replicate the map's discrete extreme classification (same overwrite order)."""
    valid = np.isfinite(v_curr)
    mask = np.full(v_curr.shape, np.nan)
    if t_cold["p25"]: mask = np.where(valid & np.isfinite(v_p25) & (v_curr <= v_p25), 4, mask)
    if t_warm["p75"]: mask = np.where(valid & np.isfinite(v_p75) & (v_curr >= v_p75), 5, mask)
    if t_cold["p10"]: mask = np.where(valid & np.isfinite(v_p10) & (v_curr <= v_p10), 3, mask)
    if t_warm["p90"]: mask = np.where(valid & np.isfinite(v_p90) & (v_curr >= v_p90), 6, mask)
    if t_cold["p5"]:  mask = np.where(valid & np.isfinite(v_p5) & (v_curr <= v_p5), 2, mask)
    if t_warm["p95"]: mask = np.where(valid & np.isfinite(v_p95) & (v_curr >= v_p95), 7, mask)
    if t_cold["rec"]: mask = np.where(valid & np.isfinite(v_rec_c) & (v_curr <= v_rec_c), 1, mask)
    if t_warm["rec"]: mask = np.where(valid & np.isfinite(v_rec_w) & (v_curr >= v_rec_w), 8, mask)
    return mask


def _yyyymmdd_year_grid(grid) -> np.ndarray:
    """YYYYMMDD int grids from the climatology → calendar year as float."""
    arr = np.asarray(grid, dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > 10_000_000), np.floor(arr / 10_000.0), np.nan)


def _yyyymmdd_dot_date(val) -> str:
    """Format a climatology YYYYMMDD int as 'DD.MM.YYYY'; empty string if missing."""
    try:
        n = int(val)
    except (TypeError, ValueError):
        return ""
    if n < 10_000_000:
        return ""
    y, m, d = n // 10000, (n // 100) % 100, n % 100
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return ""
    return f"{d:02d}.{m:02d}.{y:04d}"


def _yyyymmdd_dot_date_arr(grid) -> np.ndarray:
    """Vector of ' (DD.MM.YYYY)' suffixes (or '') for Plotly hover customdata."""
    arr = np.asarray(grid).reshape(-1)
    out = np.empty(arr.size, dtype=object)
    for i, v in enumerate(arr):
        ds = _yyyymmdd_dot_date(v)
        out[i] = f" ({ds})" if ds else ""
    return out


def _map_historical_records(ref_data, doy: int, target_date, map_var: str, shape, anchor_date=None):
    """All-time warm/cold values for map display, from the reference climatology.

    `ref_data` is the caller's own reference-climatology handle (the same
    object callers already pass in as `ref_data`/`_ref_data`), not a module
    global — this keeps the function self-contained and import-safe.
    """
    nan = (np.full(shape, np.nan),) * 4
    if ref_data is None:
        return nan
    try:
        daily_ref = ref_data.sel(dayofyear=int(doy))
    except Exception:
        return nan
    if map_var == "TX":
        wkey, ckey, wd, cd = "tx_max_val", "tx_min_val", "tx_max_date", "tx_min_date"
    elif map_var == "TN":
        wkey, ckey, wd, cd = "tn_max_val", "tn_min_val", "tn_max_date", "tn_min_date"
    else:
        wkey, ckey, wd, cd = "tg_max_val", "tg_min_val", "tg_max_date", "tg_min_date"
    if wkey not in daily_ref or ckey not in daily_ref:
        return nan
    rec_w = np.asarray(daily_ref[wkey].values, dtype=np.float64)
    rec_c = np.asarray(daily_ref[ckey].values, dtype=np.float64)
    yr_w = _yyyymmdd_year_grid(daily_ref[wd].values) if wd in daily_ref else np.full(shape, np.nan)
    yr_c = _yyyymmdd_year_grid(daily_ref[cd].values) if cd in daily_ref else np.full(shape, np.nan)
    return rec_w, rec_c, yr_w, yr_c


# Map z-bin indices (must match frontend_plots.build_baseline_map's mask assignment order).
_TOP10_COLD_BINS = {
    "Moderate": {4},
    "Strong": {3, 2, 1},
    "Extreme": {2, 1},
    "All-Time Record": {1},
}
_TOP10_WARM_BINS = {
    "Moderate": {5},
    "Strong": {6, 7, 8},
    "Extreme": {7, 8},
    "All-Time Record": {8},
}


def _top10_analysis_key(top10_threshold: str) -> str:
    if "All-Time" in top10_threshold:
        return "All-Time Record"
    if "Extreme" in top10_threshold:
        return "Extreme"
    if "Strong" in top10_threshold:
        return "Strong"
    return "Moderate"


def _footprint_parquet_path(target_date_str, map_var, baseline_type) -> Path:
    return PRECOMPUTED_ANALYTICS_DIR / f"footprint_{target_date_str}_{map_var}_{baseline_type}_{_FOOTPRINT_CACHE_THRESHOLD}.parquet"


def _top10_parquet_paths(target_date_str, map_var, baseline_type, top10_threshold) -> tuple[Path, Path]:
    base = f"top10_{target_date_str}_{map_var}_{baseline_type}_{top10_threshold}"
    return (
        PRECOMPUTED_ANALYTICS_DIR / f"{base}_warm.parquet",
        PRECOMPUTED_ANALYTICS_DIR / f"{base}_cold.parquet",
    )


def _toggles_all_true(t_warm: dict, t_cold: dict) -> bool:
    """True only when every warm/cold percentile toggle is active — the exact
    assumption batch_precompute_analytics.py bakes into its Parquet output."""
    return all(t_warm.values()) and all(t_cold.values())


def flatten_footprint(footprint: dict) -> pd.DataFrame:
    """{"moderate": {"warm_pct": .., "cold_pct": .., "total_pct": ..}, ...} ->
    a single-row flat DataFrame (columns like "moderate_warm_pct"), i.e. the
    exact on-disk shape written/read by the Parquet pre-computation pipeline."""
    row = {f"{tier}_{metric}": value for tier, metrics in footprint.items() for metric, value in metrics.items()}
    return pd.DataFrame([row])


def unflatten_footprint(df: pd.DataFrame) -> dict:
    """Inverse of `flatten_footprint`: rebuilds the exact nested dict shape
    `compute_map_footprint()` callers expect (`footprint[tier]["total_pct"]`, etc.)."""
    row = df.iloc[0]
    out: dict = {}
    for col, value in row.items():
        tier, metric = col.split("_", 1)
        out.setdefault(tier, {})[metric] = float(value)
    return out


@st.cache_data(show_spinner=False)
def _calc_compute_map_footprint_raw(_ref_data, _map_phys_data, target_date_str, t_warm, t_cold, baseline_type="A", map_var="TG", anchor_date_str=None):
    """
    Area-weighted, CUMULATIVE Moderate/Strong/Extreme/Record spatial footprint
    for the Map Tracker narrative (backend_narrative.spatial_extreme_footprint).
    Mirrors frontend_plots.build_baseline_map()'s value/threshold retrieval and
    mask construction exactly, so the narrated percentages always match the map.

    PERFORMANCE: leading-underscore `_ref_data`/`_map_phys_data` are excluded
    from Streamlit's cache-key hash (Streamlit's convention for cache_data /
    cache_resource) — they are a large xarray Dataset and a dict of full-grid
    DataArrays, and re-hashing them on every rerun was the actual source of
    the reported frontend latency. The real cache key is the remaining,
    cheap arguments (`target_date_str`, `t_warm`, `t_cold`, `baseline_type`,
    `map_var`, `anchor_date_str`), so reruns triggered by unrelated widgets
    (e.g. the Map Layout radio) hit the cache instantly instead of
    recomputing the mask from scratch.
    """
    if _ref_data is None or _map_phys_data is None:
        return None
    target_date = pd.Timestamp(target_date_str)
    anchor_date = pd.Timestamp(anchor_date_str) if anchor_date_str else None
    suffix, doy = ("A" if baseline_type == "A" else "B"), etccdi_doy_365(target_date)
    tx_curr, tn_curr = _synoptic_temp_pair(_map_phys_data)
    if tx_curr is None or tn_curr is None:
        return None
    lons, lats = _synoptic_lonlat(_map_phys_data)
    daily_ref = _ref_data.sel(dayofyear=doy).reindex(latitude=lats, longitude=lons, method="nearest")

    def safe_get(var_key, fallback=np.nan):
        if var_key in daily_ref.variables:
            return daily_ref[var_key].values
        return np.full(tx_curr.shape, fallback)

    if map_var == "TX":
        v_curr = tx_curr
        v_p95, v_p90, v_p75 = safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
    elif map_var == "TN":
        v_curr = tn_curr
        v_p95, v_p90, v_p75 = safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
    else:
        tg_curr = _map_phys_data.get("tg")
        v_curr = _synoptic_array(tg_curr) if tg_curr is not None else (tx_curr + tn_curr) / 2.0
        v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
        v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
        v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
        v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
        v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
        v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2

    v_rec_w, v_rec_c, _, _ = _map_historical_records(_ref_data, doy, target_date, map_var, tx_curr.shape, anchor_date)
    mask = _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold)
    valid_domain = np.isfinite(v_curr)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return spatial_extreme_footprint(mask, valid_domain, lat2d, target_date=target_date_str, baseline_name=baseline_type)


@st.cache_data(show_spinner=False)
def compute_map_footprint(_ref_data, _map_phys_data, target_date_str, t_warm, t_cold, baseline_type="A", map_var="TG", anchor_date_str=None):
    """
    Lazy-loading front door for the spatial extreme footprint.

    Serves the Parquet output of `batch_precompute_analytics.py` when it
    exists for this exact (date, variable, epoch) — an instant disk read,
    with zero xarray/mask work — and transparently falls back to the
    on-the-fly `_calc_compute_map_footprint_raw()` computation otherwise
    (toggles other than "all active", dates outside the precomputed -7..+3
    window, or before the batch script has ever been run). Same signature
    and return shape as the original function, so every call site keeps
    working unmodified and the app never breaks on a cache miss.
    """
    if _toggles_all_true(t_warm, t_cold):
        parquet_path = _footprint_parquet_path(target_date_str, map_var, baseline_type)
        if parquet_path.exists():
            try:
                return unflatten_footprint(pd.read_parquet(parquet_path))
            except Exception:
                pass  # corrupted/partial Parquet file -> fall through to raw computation
    return _calc_compute_map_footprint_raw(
        _ref_data, _map_phys_data, target_date_str, t_warm, t_cold,
        baseline_type=baseline_type, map_var=map_var, anchor_date_str=anchor_date_str,
    )


# --- TOP 10 COUNTRY IMPACT (Heat & Cold Extremes) ---
@st.cache_data(show_spinner=False)
def _calc_calculate_top10_raw(
    _ref_data, _map_phys_data, target_date, t_warm, t_cold, view_mode, persist_metric, top10_threshold,
    baseline_type="A", map_var="TG", anchor_date=None, _mask_version=TOP10_MASK_VERSION,
    _get_persistence_arrays=None, _get_country_weight_grid=None,
):
    """
    `_get_persistence_arrays` / `_get_country_weight_grid` are app.py-owned
    cached loaders, passed in explicitly by the caller instead of imported
    here: Streamlit runs app.py as the entrypoint script (not as an
    importable module named "app"), so a `from app import ...` inside this
    function would re-execute the whole script from scratch on every call
    and crash on already-instantiated widgets. Leading underscores keep
    these two out of the cache-key hash, same as `_ref_data`/`_map_phys_data`.
    """
    get_persistence_arrays = _get_persistence_arrays
    get_country_weight_grid = _get_country_weight_grid

    if _ref_data is None or _map_phys_data is None:
        return pd.DataFrame(), pd.DataFrame()

    suffix, doy = ("A" if baseline_type == "A" else "B"), etccdi_doy_365(target_date)
    lons, lats = _synoptic_lonlat(_map_phys_data)
    tx, tn = _synoptic_temp_pair(_map_phys_data)
    if tx is None or tn is None:
        return pd.DataFrame(), pd.DataFrame()
    heat_mask, cold_mask = np.zeros(tx.shape, dtype=bool), np.zeros(tx.shape, dtype=bool)

    if is_daily_map_view(view_mode):
        daily_ref = _ref_data.sel(dayofyear=doy).reindex(
            latitude=lats, longitude=lons, method="nearest"
        )
        def safe_get(var_key, fallback=np.nan):
            if var_key in daily_ref.variables:
                return daily_ref[var_key].values
            return np.full(tx.shape, fallback)

        if map_var == "TX":
            v_curr = tx
            v_p95, v_p90, v_p75 = safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
            v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        elif map_var == "TN":
            v_curr = tn
            v_p95, v_p90, v_p75 = safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
            v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        else:
            tg = _map_phys_data.get("tg")
            v_curr = _synoptic_array(tg) if tg is not None else (tx + tn) / 2.0
            v_p95 = (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2
            v_p90 = (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2
            v_p75 = (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2
            v_p25 = (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2
            v_p10 = (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2
            v_p5 = (safe_get(f'tx_p5_doy_{suffix}') + safe_get(f'tn_p5_doy_{suffix}')) / 2

        v_rec_w, v_rec_c, _, _ = _map_historical_records(_ref_data, doy, target_date, map_var, tx.shape, anchor_date)

        display_mask = _build_display_mask(v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5, v_rec_w, v_rec_c, t_warm, t_cold)
        level = _top10_analysis_key(top10_threshold)
        heat_mask = np.isin(display_mask, list(_TOP10_WARM_BINS[level]))
        cold_mask = np.isin(display_mask, list(_TOP10_COLD_BINS[level]))
    else:
        anchor_date_str = anchor_date.strftime('%Y-%m-%d') if anchor_date is not None else None
        streaks = get_persistence_arrays(
            target_date.strftime('%Y-%m-%d'), baseline_type, map_var, anchor_date_str,
            forecast_model=selected_forecast_model(),
        )
        if streaks is not None:
            mapping = {"Moderate": (0, 4), "Strong": (1, 5), "Extreme": (2, 6), "All-Time Record": (3, 7)}
            h_idx, c_idx = mapping.get(persist_metric, (1, 5))
            heat_mask, cold_mask = streaks[h_idx] >= 6, streaks[c_idx] >= 6

    weights, sizes = get_country_weight_grid(tuple(lons), tuple(lats))
    res_h, res_c = [], []
    for name, w in weights.items():
        tot = float(w.sum())
        if tot <= 0: continue
        fh, fc = float((heat_mask * w).sum() / tot * 100), float((cold_mask * w).sum() / tot * 100)
        if fh >= TOP10_MIN_PCT: res_h.append({"Country": name, "Warm Impact (%)": fh, "_size": sizes[name]})
        if fc >= TOP10_MIN_PCT: res_c.append({"Country": name, "Cold Impact (%)": fc, "_size": sizes[name]})

    def _rank(rows, col):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values(by=[col, "_size"], ascending=[False, False]).head(10)
        return df[["Country", col]].reset_index(drop=True)

    return _rank(res_h, "Warm Impact (%)"), _rank(res_c, "Cold Impact (%)")


@st.cache_data(show_spinner=False)
def calculate_top10(
    _ref_data, _map_phys_data, target_date, t_warm, t_cold, view_mode, persist_metric, top10_threshold,
    baseline_type="A", map_var="TG", anchor_date=None, _mask_version=TOP10_MASK_VERSION,
    _get_persistence_arrays=None, _get_country_weight_grid=None,
):
    """
    Lazy-loading front door for the Top-10 country impact tables.

    Serves the two Parquet tables written by `batch_precompute_analytics.py`
    when the request matches exactly what the batch script produces (daily
    snapshot view, every warm/cold toggle active, date inside the -7..+3
    precomputed window) — an instant disk read, skipping the country-mask
    weighting loop entirely — and transparently falls back to
    `_calc_calculate_top10_raw()` for everything else (Persistence view,
    partial toggle states, or a missing/corrupted cache file). Same
    signature and return shape as the original function, so every call
    site keeps working unmodified and the app never breaks on a cache miss.
    """
    if is_daily_map_view(view_mode) and _toggles_all_true(t_warm, t_cold):
        target_date_str = pd.Timestamp(target_date).strftime('%Y-%m-%d')
        warm_path, cold_path = _top10_parquet_paths(target_date_str, map_var, baseline_type, top10_threshold)
        if warm_path.exists() and cold_path.exists():
            try:
                return pd.read_parquet(warm_path), pd.read_parquet(cold_path)
            except Exception:
                pass  # corrupted/partial Parquet file -> fall through to raw computation
    return _calc_calculate_top10_raw(
        _ref_data, _map_phys_data, target_date, t_warm, t_cold, view_mode, persist_metric, top10_threshold,
        baseline_type=baseline_type, map_var=map_var, anchor_date=anchor_date, _mask_version=_mask_version,
        _get_persistence_arrays=_get_persistence_arrays, _get_country_weight_grid=_get_country_weight_grid,
    )
