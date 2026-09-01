"""
AtmoPulse High-Performance Pre-computation Batch Job (batch_precompute_analytics.py)

Offline batch script that walks the entire Forecast Offset window
(-7 .. +3 days from "today", the same window exposed by the sidebar slider)
and precomputes, for every (variable, epoch, threshold) combination, the two
heavy Map Tracker analytics that were previously recalculated on every
Streamlit rerun:

  - the area-weighted spatial extreme footprint (backend_narrative.
    spatial_extreme_footprint, via backend_analytics._calc_compute_map_footprint_raw)
  - the Top-10 country impact ranking (backend_analytics._calc_calculate_top10_raw)

Results are written as Parquet files under `DATA_ROOT / "Precomputed_Analytics"`.
`backend_analytics.compute_map_footprint()` / `calculate_top10()` transparently
read these files at request time (see backend_analytics.py) and fall back to
the on-the-fly computation whenever a Parquet file is missing.

Run this script on a schedule (e.g. whenever a new ERA5/IFS batch lands) to
keep the Map Tracker's -7..+3 day window fully warm.

Toggle assumption: every warm/cold percentile toggle is treated as active
(the app's own default), matching the exact cache key `compute_map_footprint()`
/ `calculate_top10()` check for before serving a precomputed file.

Usage:
    python batch_precompute_analytics.py
"""

from __future__ import annotations

import sys
import threading
import types
from functools import lru_cache

import numpy as np
import pandas as pd

# Headless cron must never import the real Streamlit runtime. MagicMock is
# also unsafe here: it records every attribute access (a leak) and its fake
# ``__path__`` can still let ``streamlit.runtime`` load ScriptRunContext.
class _SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


def _identity_cache(*args, **kwargs):
    """No-op stand-in for ``@st.cache_data`` / ``@st.cache_resource``."""

    def decorator(fn):
        return fn

    if args and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


def _install_streamlit_stub() -> types.ModuleType:
    stub = types.ModuleType("streamlit")
    stub._atmopulse_headless_stub = True
    stub.__file__ = "<atmopulse-streamlit-stub>"
    stub.__path__ = []  # package with no on-disk path: blocks real submodules
    stub.session_state = _SessionState()
    stub.cache_data = _identity_cache
    stub.cache_resource = _identity_cache
    stub.set_page_config = lambda *a, **k: None
    stub.stop = lambda *a, **k: None
    stub.sidebar = stub

    def _noop(*_a, **_k):
        return None

    def _getattr(name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _noop

    stub.__getattr__ = _getattr  # type: ignore[method-assign]

    # Pre-register runtime modules so accidental imports cannot load site-packages.
    runtime = types.ModuleType("streamlit.runtime")
    runtime.__path__ = []
    scriptrunner = types.ModuleType("streamlit.runtime.scriptrunner")
    scriptrunner.get_script_run_ctx = lambda: None

    sys.modules["streamlit"] = stub
    sys.modules["streamlit.runtime"] = runtime
    sys.modules["streamlit.runtime.scriptrunner"] = scriptrunner
    return stub


st = _install_streamlit_stub()

from backend_analytics import (
    _calc_calculate_top10_raw,
    _calc_compute_map_footprint_raw,
    flatten_footprint,
)
from backend_io import fetch_cached_synoptic_data, load_reference_climatology
from backend_map_locations import build_country_weight_grid
from config import (
    DATA_ROOT,
    FORECAST_MODEL_IFS,
    FORECAST_OFFSET_MAX,
    FORECAST_OFFSET_MIN,
    MAP_VIEW_DAILY,
)

OUTPUT_DIR = DATA_ROOT / "Precomputed_Analytics"

MAP_VAR_CODES = ("TG", "TX", "TN")
EPOCHS = ("A", "B")
THRESHOLDS = ("Moderate", "Strong", "Extreme", "All-Time Record")

# TASK 1 assumption: batch generation always assumes every warm/cold
# percentile toggle is active — matches backend_analytics._toggles_all_true(),
# the exact gate that decides whether a request is served from these files.
ALL_TOGGLES_WARM = {"p75": True, "p90": True, "p95": True, "rec": True}
ALL_TOGGLES_COLD = {"p25": True, "p10": True, "p5": True, "rec": True}


def _ensure_session_state() -> None:
    """`fetch_cached_synoptic_data` reads `st.session_state.nc_lock`, normally
    seeded by app.py's module-level init. This script runs standalone
    (outside `streamlit run`), so seed the same key here to keep that loader
    working completely unmodified."""
    if "nc_lock" not in st.session_state:
        st.session_state.nc_lock = threading.Lock()


@lru_cache(maxsize=8)
def _cached_country_weight_grid(lons_tuple, lats_tuple):
    return build_country_weight_grid(np.array(lons_tuple), np.array(lats_tuple))


def _footprint_path(target_date_str: str, map_var_code: str, epoch: str, threshold: str):
    return OUTPUT_DIR / f"footprint_{target_date_str}_{map_var_code}_{epoch}_{threshold}.parquet"


def _top10_paths(target_date_str: str, map_var_code: str, epoch: str, threshold: str):
    base = f"top10_{target_date_str}_{map_var_code}_{epoch}_{threshold}"
    return OUTPUT_DIR / f"{base}_warm.parquet", OUTPUT_DIR / f"{base}_cold.parquet"


def precompute_offset(offset: int, anchor_date: pd.Timestamp, ref_clim) -> None:
    target_date = anchor_date + pd.Timedelta(days=offset)
    target_date_str = target_date.strftime("%Y-%m-%d")
    anchor_date_str = anchor_date.strftime("%Y-%m-%d")

    print(f"[offset {offset:+d}] {target_date_str}: fetching synoptic fields...")
    try:
        map_phys_data, map_time_meta = fetch_cached_synoptic_data(
            target_date_str, anchor_date_str, forecast_model=FORECAST_MODEL_IFS,
        )
    except Exception as exc:
        print(f"  SKIPPED {target_date_str} (loader error: {exc})")
        return

    if not map_time_meta.get("available"):
        print(f"  SKIPPED {target_date_str} (no synoptic data available for this offset)")
        return

    lons, lats = map_phys_data.get("_lons"), map_phys_data.get("_lats")
    if lons is None or lats is None:
        print(f"  SKIPPED {target_date_str} (no grid coordinates in synoptic payload)")
        return
    # Warm the country-weight grid cache once per offset; `_calc_calculate_top10_raw`
    # re-derives the same lons/lats from `map_phys_data` internally and calls
    # `_get_country_weight_grid(tuple(lons), tuple(lats))`, which will then
    # hit this lru_cache instead of recomputing the country-mask overlay.
    _cached_country_weight_grid(tuple(lons), tuple(lats))

    for map_var_code in MAP_VAR_CODES:
        for epoch in EPOCHS:
            # Footprint is threshold-independent (all four tiers come back at
            # once) — computed ONCE per (var, epoch), then written out once
            # per threshold below to satisfy the shared naming convention.
            footprint = _calc_compute_map_footprint_raw(
                ref_clim, map_phys_data, target_date_str,
                ALL_TOGGLES_WARM, ALL_TOGGLES_COLD, epoch, map_var_code,
                anchor_date_str=anchor_date_str,
            )
            if footprint is None:
                print(f"  SKIPPED footprint {map_var_code}/{epoch} (no data for {target_date_str})")
                footprint_df = None
            else:
                footprint_df = flatten_footprint(footprint)

            for threshold in THRESHOLDS:
                if footprint_df is not None:
                    footprint_df.to_parquet(_footprint_path(target_date_str, map_var_code, epoch, threshold), index=False)

                df_warm, df_cold = _calc_calculate_top10_raw(
                    ref_clim, map_phys_data, target_date,
                    ALL_TOGGLES_WARM, ALL_TOGGLES_COLD,
                    MAP_VIEW_DAILY, "Strong", threshold,
                    epoch, map_var_code, anchor_date=anchor_date,
                    _get_persistence_arrays=None,
                    _get_country_weight_grid=_cached_country_weight_grid,
                )
                warm_path, cold_path = _top10_paths(target_date_str, map_var_code, epoch, threshold)
                df_warm.to_parquet(warm_path, index=False)
                df_cold.to_parquet(cold_path, index=False)

            print(f"  wrote {map_var_code}/{epoch}: footprint + top10 for {len(THRESHOLDS)} thresholds")


def run() -> None:
    print("=" * 70)
    print("AtmoPulse Analytics Pre-computation Batch Job")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    _ensure_session_state()

    print("Loading reference climatology (cached singleton)...")
    ref_clim = load_reference_climatology()
    if ref_clim is None:
        print("ABORTED: Reference Climatology is missing or corrupted. Please rebuild it first.")
        return

    anchor_date = pd.Timestamp.now().floor("D")
    print(f"Anchor date (today, normalized to midnight): {anchor_date.strftime('%Y-%m-%d')}")
    print(f"Forecast Offset window: {FORECAST_OFFSET_MIN:+d} .. {FORECAST_OFFSET_MAX:+d} days")
    print(f"Variables: {MAP_VAR_CODES} | Epochs: {EPOCHS} | Thresholds: {THRESHOLDS}")
    print("-" * 70)

    total_days = FORECAST_OFFSET_MAX - FORECAST_OFFSET_MIN + 1
    for i, offset in enumerate(range(FORECAST_OFFSET_MIN, FORECAST_OFFSET_MAX + 1), start=1):
        print(f"[{i}/{total_days}] Processing offset {offset:+d}...")
        precompute_offset(offset, anchor_date, ref_clim)

    print("-" * 70)
    print("Batch pre-computation complete.")


if __name__ == "__main__":
    run()