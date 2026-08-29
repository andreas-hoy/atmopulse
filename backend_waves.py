"""
AtmoPulse Wave Detection & Ridge-Plot Analytics (backend_waves.py)

This module handles the extraction, dynamic thresholding, and visualization of 
synoptic extreme events (summer heatwaves and winter coldwaves) using an adapted 
Kyselý definition. 

Core functionalities:
- Extracts point TX/TN exclusively from era5_master_daily_YYYY.nc (IFS/AIFS only for the last 6 days and the forecast).
- Dynamically calculates seasonal climatological thresholds (P95, P90, P75 for JJA; 
  P5, P10, P25 for DJF) based on shifting reference periods (1961-1990 or 1996-2025).
- Excises leap days (Feb 29th) to ensure statistical homoscedasticity per ETCCDI norms.
- Identifies consecutive threshold exceedances and applies trailing tolerance drops.
- Renders highly customized, interactive Plotly ridge-plots and annual intensity 
  bar charts representing cumulative thermal stress (K·days).
"""

import json
import os
import subprocess
import sys
import xarray as xr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from scipy.interpolate import make_interp_spline

from backend_maps import drop_era5t_aux
from atmopulse_theme import (
    ATMOPULSE_COLD, 
    ATMOPULSE_FONTS, 
    ATMOPULSE_WARM, 
    plotly_title_font, 
    plotly_typography
)

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
if not CLIM_FILE.exists(): 
    CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")

# All parameters the Point Wavogram can plot. "var" is the on-disk/Zarr
# variable name; "is_warm" drives season windowing (JJA/May-Sep vs
# DJF/Nov-Mar), percentile direction (p75/90/95 vs p25/10/5), and detection
# direction (>= threshold vs <= threshold) throughout this module.
WAVE_PARAM_CONFIG = {
    "TX": {"var": "tx", "is_warm": True},
    "TN": {"var": "tn", "is_warm": False},
    "TG": {"var": "tg", "is_warm": True},
    "T850": {"var": "t850", "is_warm": True},
}


def _param_var(parameter: str) -> str:
    return WAVE_PARAM_CONFIG.get(parameter, {"var": parameter.lower()})["var"]


def _param_is_warm(parameter: str) -> bool:
    return WAVE_PARAM_CONFIG.get(parameter, {"is_warm": True})["is_warm"]

# --- Ridge-plot layout (tune wave shape / break aesthetics here) ---
WAVE_RIDGE_SPLINE_PTS = 100      # Smoothness of the ridge curve
WAVE_RIDGE_SKEW_FACTOR = 2.5     # Horizontal bulge vs. intensity (0 = symmetric)
WAVE_RIDGE_HEIGHT_SCALE = 20.0   # Vertical extent in axis-year units (÷ intensity)
WAVE_BREAK_TAIL_LEN = 1.0        # X-axis length of the post-peak decay tail
WAVE_BREAK_TAIL_STEPS = 30       # Number of points along the decay tail
WAVE_BREAK_CTRL_X = -0.25        # Bezier ctrl-x (× tail_len); negative -> mid-fall bulges left
WAVE_BREAK_CTRL_Y = 0.42         # Bezier ctrl-y (× peak height); shapes the curl
WAVE_LINE_WIDTH = 1.0
WAVE_FILL_ALPHA_BASE = 0.55      # Gradient fill opacity at ridge base
WAVE_FILL_ALPHA_PEAK = 0.88      # Gradient fill opacity at ridge peak
WAVE_LINE_ALPHA = 0.92
WAVE_INTENSITY_CAP_TX = 100.0    # Intensity (K·days) mapped to full warm colour
WAVE_INTENSITY_CAP_TN = 200.0    # Intensity (K·days) mapped to full cold colour


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_hex(c0: str, c1: str, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    r0, g0, b0 = _hex_to_rgb(c0)
    r1, g1, b1 = _hex_to_rgb(c1)
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def _wave_ridge_colors(parameter: str, norm_val: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """AtmoPulse map palette: warm p75->rec / cold p25->rec by severity."""
    if _param_is_warm(parameter):
        base = _hex_to_rgb(ATMOPULSE_WARM["p75"])
        peak = _lerp_hex(ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["rec"], norm_val)
    else:
        base = _hex_to_rgb(ATMOPULSE_COLD["p25"])
        peak = _lerp_hex(ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["rec"], norm_val)
    return base, peak


def _wave_break_tail(x_end: float, y_peak: float) -> tuple[np.ndarray, np.ndarray]:
    """Visual-only post-peak closure (Bezier); Kyselý data ends at the peak."""
    if y_peak <= 0:
        return np.array([x_end]), np.array([0.0])
    
    t = np.linspace(0, 1, WAVE_BREAK_TAIL_STEPS)
    x_ctrl = x_end + WAVE_BREAK_TAIL_LEN * WAVE_BREAK_CTRL_X
    y_ctrl = y_peak * WAVE_BREAK_CTRL_Y
    x_out = x_end + WAVE_BREAK_TAIL_LEN
    
    x_break = (1 - t) ** 2 * x_end + 2 * (1 - t) * t * x_ctrl + t ** 2 * x_out
    y_break = (1 - t) ** 2 * y_peak + 2 * (1 - t) * t * y_ctrl
    return x_break, y_break


_POINT_SERIES_EXTRACT_SCRIPT = r"""
import json, sys
import numpy as np, pandas as pd, xarray as xr
path, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
try:
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        ds = ds.rename({"mx2t": "tx"})
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        ds = ds.rename({"mn2t": "tn"})
    keep = [v for v in ("tx", "tn", "tg", "t850") if v in ds.data_vars]
    if not keep:
        raise SystemExit(2)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    t_name = "valid_time" if "valid_time" in ds.dims else "time"
    pt = ds[keep].sel({lat_name: lat, lon_name: lon}, method="nearest")
    times = pd.to_datetime(pt[t_name].values)
    days = pd.DatetimeIndex(times).tz_localize(None).normalize() if getattr(times, "tz", None) else pd.DatetimeIndex(times).normalize()
    def _c(v):
        if v not in pt:
            return [None] * len(days)
        a = np.squeeze(np.asarray(pt[v].values, dtype=float))
        a = np.atleast_1d(a)
        finite = a[np.isfinite(a)]
        if finite.size and float(np.mean(finite)) > 100:
            a = a - 273.15
        return [None if not np.isfinite(x) else float(x) for x in a]
    payload = {"Date": [d.strftime("%Y-%m-%d") for d in days]}
    for v in keep:
        payload[v] = _c(v)
    json.dump(payload, sys.stdout)
finally:
    ds.close()
"""


def _point_series_from_master_isolated(path, lat, lon) -> pd.DataFrame:
    """Read current-year master TX/TN/TG/T850 at one point in a child process (HDF5 abort safety)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _POINT_SERIES_EXTRACT_SCRIPT, str(path), str(lat), str(lon)],
            capture_output=True, text=True, timeout=90, env=env, creationflags=flags,
        )
    except (subprocess.TimeoutExpired, OSError):
        return pd.DataFrame()
    if proc.returncode != 0 or not proc.stdout.strip():
        return pd.DataFrame()
    payload = json.loads(proc.stdout)
    df = pd.DataFrame(payload)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


WAVE_ARCHIVE_VARS = ("tx", "tn", "tg", "t850")


def _kelvin_to_celsius_if_needed(arr: np.ndarray) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(arr, dtype=np.float64))
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.mean(finite)) > 100:
        arr = arr - 273.15
    return arr


def _era5_master_point_series_from_zarr(lat: float, lon: float) -> pd.DataFrame:
    """Millisecond-scale point read of TX/TN/TG/T850 from the point-extraction-
    optimal Zarr mirror (see batch_convert_netcdf_to_zarr.py). Returns an
    empty DataFrame on ANY problem (missing store, missing variables,
    corrupt/partial write, ...) so the caller falls back to the legacy
    year-by-year NetCDF loop unconditionally."""
    from backend_io import ZARR_MASTER_TIME_SERIES

    if not ZARR_MASTER_TIME_SERIES.exists():
        return pd.DataFrame()
    try:
        zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=True)
    except Exception:
        try:
            zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=False)
        except Exception:
            return pd.DataFrame()

    keep = [v for v in WAVE_ARCHIVE_VARS if v in zds.data_vars]
    if not keep:
        return pd.DataFrame()

    try:
        pt = zds[keep].sel(latitude=lat, longitude=lon, method="nearest").compute()
        t = pd.to_datetime(pt["valid_time"].values)
        if getattr(t, "tz", None) is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        rec = {"Date": pd.DatetimeIndex(t).normalize()}
        for v in keep:
            rec[v] = _kelvin_to_celsius_if_needed(pt[v].values)
        return pd.DataFrame(rec)
    except Exception:
        return pd.DataFrame()


def _era5_master_point_series_from_netcdf(lat: float, lon: float) -> list[pd.DataFrame]:
    """Legacy fallback: year-by-year NetCDF loop over era5_master_daily_YYYY.nc,
    extracting whichever of TX/TN/TG/T850 are present in each file."""
    files = sorted(DATA_DIR.glob("era5_master_daily_*.nc"))
    this_year = int(pd.Timestamp.utcnow().year)
    frames = []
    for path in files:
        try:
            year = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            year = None
        if year == this_year:
            sub = _point_series_from_master_isolated(path, lat, lon)
            if sub is not None and not sub.empty:
                frames.append(sub)
            continue
        ds = None
        try:
            ds = xr.open_dataset(path, engine="netcdf4", decode_timedelta=False)
            if "time" in ds.dims and "valid_time" not in ds.dims:
                ds = ds.rename({"time": "valid_time"})
            if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
                ds = ds.rename({"mx2t": "tx"})
            if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
                ds = ds.rename({"mn2t": "tn"})
            keep = [v for v in WAVE_ARCHIVE_VARS if v in ds.data_vars]
            if not keep:
                continue
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            pt = ds[keep].sel({lat_name: lat, lon_name: lon}, method="nearest")
            t = pd.to_datetime(pt.valid_time.values)
            if getattr(t, "tz", None) is not None:
                t = t.tz_convert("UTC").tz_localize(None)
            rec = {"Date": pd.DatetimeIndex(t).normalize()}
            for v in keep:
                rec[v] = _kelvin_to_celsius_if_needed(pt[v].values)
            frames.append(pd.DataFrame(rec))
        except Exception:
            continue
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
    return frames


@st.cache_data(show_spinner=False)
def _era5_master_point_series(lat: float, lon: float, _archive_version=7) -> pd.DataFrame:
    """Full 1940–present TX/TN/TG/T850 at one grid cell.

    Fast path: reads the point-extraction-optimal Zarr mirror of the master
    archive (see batch_convert_netcdf_to_zarr.py / backend_io.ZARR_MASTER_TIME_SERIES)
    — millisecond reads instead of the old year-by-year NetCDF loop. Falls
    back to that legacy loop whenever the Zarr store doesn't exist yet or
    fails to read for any reason, so wave charts never break before/without
    the migration having been run.

    IFS/AIFS is overlaid solely for the last 6 days through the forecast,
    for whichever of tx/tn/tg/t850 the live forecast dataset carries.
    """
    from backend_maps import _open_live_forecast_ds, LIVE_OVERLAY_PAST_DAYS

    frames = []
    zarr_df = _era5_master_point_series_from_zarr(lat, lon)
    if not zarr_df.empty:
        frames.append(zarr_df)
    else:
        frames.extend(_era5_master_point_series_from_netcdf(lat, lon))

    lf = _open_live_forecast_ds()
    if lf is not None:
        try:
            tdim = "valid_time" if "valid_time" in lf.dims else "time"
            cut = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=LIVE_OVERLAY_PAST_DAYS)
            pt = lf.sel(latitude=lat, longitude=lon, method="nearest").sel({tdim: slice(cut, None)})
            tx_name = "tx" if "tx" in pt.data_vars else "mx2t"
            tn_name = "tn" if "tn" in pt.data_vars else "mn2t"
            t = pd.to_datetime(pt[tdim].values)
            if getattr(t, "tz", None) is not None:
                t = t.tz_convert("UTC").tz_localize(None)
            rec = {"Date": pd.DatetimeIndex(t).normalize()}
            has_any = False
            if tx_name in pt.data_vars and tn_name in pt.data_vars:
                rec["tx"] = _kelvin_to_celsius_if_needed(pt[tx_name].values)
                rec["tn"] = _kelvin_to_celsius_if_needed(pt[tn_name].values)
                has_any = True
            for extra in ("tg", "t850"):
                if extra in pt.data_vars:
                    rec[extra] = _kelvin_to_celsius_if_needed(pt[extra].values)
                    has_any = True
            if has_any:
                frames.append(pd.DataFrame(rec))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    # Keep 29 Feb here — this feeds both the live wave-detection pipeline
    # (must show it) and the seasonal-threshold baseline (which excises it
    # itself, in _wave_season_thresholds).
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _load_waves_archive_ds(_archive_version=4):
    """Existence check: wavogram values come from era5_master_daily_*.nc, not Zarr."""
    files = sorted(DATA_DIR.glob("era5_master_daily_*.nc"))
    return files if files else None


@st.cache_resource(show_spinner=False)
def _load_waves_climatology():
    """SINGLETON POINTER for the reference climatology file used by the wave charts."""
    if not CLIM_FILE.exists():
        return None
    return xr.open_dataset(CLIM_FILE, engine='netcdf4')


@st.cache_data(show_spinner=False)
def _wave_season_thresholds(lat: float, lon: float, suffix: str, parameter: str = "TX") -> dict:
    """
    Kyselý seasonal thresholds from true 24h daily max (JJA, warm params:
    TX/TG/T850) / min (DJF, TN) of the requested `parameter`, read via
    `_era5_master_point_series` (Zarr fast path, NetCDF fallback). IFS/AIFS
    is used only for the last 6 days and the forecast — never as a
    historical fill.

    Returned dict keys are prefixed with the parameter's variable name
    (e.g. "tx_p75"/"tg_p90"/"t850_p95") so multiple parameters' thresholds
    never collide, plus "epoch_years".
    """
    var_key = _param_var(parameter)
    is_warm = _param_is_warm(parameter)

    df_pt = _era5_master_point_series(lat, lon)
    if df_pt.empty or var_key not in df_pt.columns:
        return {}

    times = pd.to_datetime(df_pt["Date"])
    leap = (times.dt.month == 2) & (times.dt.day == 29)
    df_pt = df_pt.loc[~leap.values].copy()
    times = pd.to_datetime(df_pt["Date"])

    if suffix == "A":
        y0, y1 = 1961, 1990
    else:
        y1 = int(times.max().year if times.max().month >= 12 else times.max().year - 1)
        y0 = y1 - 29

    raw = np.asarray(df_pt[var_key].values, dtype=np.float64)
    df = pd.DataFrame({var_key: raw}, index=times).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    agg = "max" if is_warm else "min"
    daily = df.resample("D").agg(agg)[var_key]
    daily = daily[~((daily.index.month == 2) & (daily.index.day == 29))]

    yr_ok = (daily.index.year >= y0) & (daily.index.year <= y1)
    season_months = [6, 7, 8] if is_warm else [12, 1, 2]
    season_vals = daily[yr_ok & daily.index.month.isin(season_months)]

    result = {"epoch_years": f"{y0}–{y1}"}
    if is_warm:
        p75, p90, p95 = np.nanpercentile(season_vals, [75, 90, 95])
        result[f"{var_key}_p75"] = float(p75)
        result[f"{var_key}_p90"] = float(p90)
        result[f"{var_key}_p95"] = float(p95)
    else:
        p5, p10, p25 = np.nanpercentile(season_vals, [5, 10, 25])
        result[f"{var_key}_p5"] = float(p5)
        result[f"{var_key}_p10"] = float(p10)
        result[f"{var_key}_p25"] = float(p25)
    return result


def _prepare_wave_season_df(lat, lon, parameter="TX") -> tuple[pd.DataFrame, str, dict]:
    """
    Shared data-prep pipeline for Kyselý wave analytics: point extraction,
    Kelvin normalization, Feb-29 excision, true-24h daily resampling and
    seasonal windowing (May-Sep for heatwaves, Nov-Mar for coldwaves).

    Factored out of `get_kiesely_waves_figs` so the ridge-plot renderer and
    the historical-rank lookup (`get_wave_historical_rank`) run the exact
    same season/day construction and can never silently drift apart.

    Returns (df_season, group_key, diagnostics) where `diagnostics` carries
    the raw-value QA fields used by the ridge-plot's debug panel.
    """
    df_pt = _era5_master_point_series(lat, lon)
    if df_pt.empty:
        return pd.DataFrame(), "", {}

    var_key = _param_var(parameter)
    is_warm = _param_is_warm(parameter)
    if var_key not in df_pt.columns:
        return pd.DataFrame(), "", {}

    raw_vals = np.asarray(df_pt[var_key].values, dtype=np.float64)
    finite_vals = raw_vals[np.isfinite(raw_vals)]
    is_kelvin = False

    df_raw = pd.DataFrame({"Temp": raw_vals}, index=pd.to_datetime(df_pt["Date"]))
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()

    if is_warm:
        df_raw = df_raw.resample("D").max()
    else:
        df_raw = df_raw.resample("D").min()
    # 29 Feb stays as a real day in the live wave-detection/rendering series
    # (a coldwave spanning it must show as one unbroken streak). Only the
    # seasonal P-thresholds (_wave_season_thresholds) excise it as baseline.

    df = df_raw.copy()

    max_valid_date = pd.Timestamp.now() + pd.Timedelta(days=6)
    df = df[df.index <= max_valid_date]
    df['year'], df['month'], df['date'] = df.index.year, df.index.month, df.index.normalize()

    if is_warm:
        df_season = df[df['month'].isin([5, 6, 7, 8, 9])].copy()
        group_key = 'year'
        df_season['plot_x'] = (df_season['date'] - pd.to_datetime(df_season['year'].astype(str) + '-05-01')).dt.days + 1
    else:
        df_season = df[df['month'].isin([11, 12, 1, 2, 3])].copy()
        df_season['winter_year'] = np.where(df_season['month'] <= 3, df_season['year'] - 1, df_season['year'])
        group_key = 'winter_year'
        df_season['plot_x'] = (df_season['date'] - pd.to_datetime(df_season['winter_year'].astype(str) + '-11-01')).dt.days + 1

    diagnostics = {
        "var_key": var_key,
        "is_kelvin_raw": bool(is_kelvin),
        "n_total": int(raw_vals.size),
        "n_nan_raw": int(np.isnan(raw_vals).sum()),
        "n_nan_after_resample": int(df['Temp'].isna().sum()),
        "raw_min": float(np.nanmin(raw_vals)) if np.isfinite(raw_vals).any() else None,
        "raw_max": float(np.nanmax(raw_vals)) if np.isfinite(raw_vals).any() else None,
        "raw_mean": float(np.nanmean(raw_vals)) if np.isfinite(raw_vals).any() else None,
    }
    return df_season, group_key, diagnostics


def _detect_kysely_waves(df_season: pd.DataFrame, group_key: str, p_thresh: float, p_drop: float, parameter: str) -> list[dict]:
    """
    Core Kyselý wave-detection loop (>=3 consecutive days past `p_thresh`,
    continues while the running mean stays past it, breaks on a single-day
    drop past the looser `p_drop` tolerance or once the running mean itself
    crosses back). Shared by the ridge-plot renderer and the historical-rank
    lookup so both always see the identical set of detected events.

    Each returned dict adds 'duration_days' (=len(xs), i.e. consecutive
    season-days in the event) on top of the ridge-plot's native fields.
    """
    waves_data: list[dict] = []
    is_warm = _param_is_warm(parameter)

    for yr, group in df_season.groupby(group_key):
        group = group.drop_duplicates(subset=['date'], keep='first')

        if is_warm:
            full_dates = pd.date_range(f"{int(yr)}-05-01", f"{int(yr)}-09-30")
        else:
            full_dates = pd.date_range(f"{int(yr)}-11-01", f"{int(yr)+1}-03-31")

        # 29 Feb stays on this index as a real calendar day (live rendering);
        # a coldwave streak crossing it is one continuous run, not a gap.
        # Missing ERA5 days (true holes) still surface as NaN — never filled.
        group = group.set_index('date').reindex(full_dates)
        group['plot_x'] = np.arange(1, len(full_dates) + 1)

        temps, xs = group['Temp'].values, group['plot_x'].values
        dates = group.index.values
        n = len(temps)

        i = 0
        while i < n - 2:
            if np.isnan(temps[i:i+3]).any():
                i += 1
                continue

            if all((temps[i+k] >= p_thresh) if is_warm else (temps[i+k] <= p_thresh) for k in range(3)):
                cand_temps, cand_xs, cand_dates = [], [], []
                j = i
                while j < n and not np.isnan(temps[j]):
                    cand_temps.append(temps[j])
                    cand_xs.append(xs[j])
                    cand_dates.append(dates[j])

                    drop_break = (temps[j] < p_drop) if is_warm else (temps[j] > p_drop)
                    mean_break = (np.mean(cand_temps) < p_thresh) if is_warm else (np.mean(cand_temps) > p_thresh)

                    if drop_break or mean_break:
                        cand_temps.pop()
                        cand_xs.pop()
                        cand_dates.pop()
                        break
                    j += 1

                intensity = sum(abs(t - p_thresh) for t in cand_temps if ((t >= p_thresh) if is_warm else (t <= p_thresh)))

                if intensity > 0 and len(cand_temps) >= 3:
                    waves_data.append({
                        'year': yr,
                        'xs': cand_xs,
                        'temps': cand_temps,
                        'intensity': intensity,
                        'start_date': cand_dates[0],
                        'end_date': cand_dates[-1],
                        'duration_days': len(cand_temps),
                    })
                i = j if j > i else i + 1
            else:
                i += 1

    return waves_data


_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else _ORDINAL_SUFFIXES.get(n % 10, "th")
    return f"{n}{suffix}"


def get_wave_historical_rank(
    lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)",
    target_date=None, top_n: int = 20,
) -> dict | None:
    """
    Point Wavogram historical-rank narrative logic (backend_narrative.py
    calls this directly; the visual overlay/UI compare state never changes it).

    Evaluates the currently active heatwave/coldwave (the detected event
    whose [start_date, end_date] window contains `target_date`) against every
    event of the same type detected across the full, continuous ERA5 record
    since 1940, using the identical Kyselý detection as the ridge-plot.

    Returns None if there is no currently active event, or if its duration
    rank falls outside the Top `top_n` (default 20) longest events on record
    -- per the "Trigger only for Top 20 events" constraint.
    """
    suffix = "A" if selected_epoch == "A" else "B"
    var_key = _param_var(parameter)
    is_warm = _param_is_warm(parameter)
    thr = _wave_season_thresholds(lat, lon, suffix, parameter)
    if not thr:
        return None

    if is_warm:
        p_t_ext, p_d_ext = thr[f"{var_key}_p95"], thr[f"{var_key}_p90"]
        p_t_str, p_d_str = thr[f"{var_key}_p90"], thr[f"{var_key}_p75"]
    else:
        p_t_ext, p_d_ext = thr[f"{var_key}_p5"], thr[f"{var_key}_p10"]
        p_t_str, p_d_str = thr[f"{var_key}_p10"], thr[f"{var_key}_p25"]
    p_thresh, p_drop = (p_t_ext, p_d_ext) if "Extreme" in threshold_level else (p_t_str, p_d_str)
    if np.isnan(p_thresh) or np.isnan(p_drop):
        return None

    df_season, group_key, _ = _prepare_wave_season_df(lat, lon, parameter)
    if df_season.empty:
        return None

    waves_data = _detect_kysely_waves(df_season, group_key, p_thresh, p_drop, parameter)
    if not waves_data:
        return None

    target = pd.Timestamp(target_date) if target_date is not None else pd.Timestamp.now()
    current = next(
        (w for w in waves_data if pd.Timestamp(w['start_date']) <= target <= pd.Timestamp(w['end_date'])),
        None,
    )
    if current is None:
        return None

    # Rank by duration (days), descending; ties share the best-available rank.
    durations = np.array([w['duration_days'] for w in waves_data])
    rank = int(np.sum(durations > current['duration_days'])) + 1
    if rank > top_n:
        return None

    return {
        "rank": rank,
        "rank_ordinal": _ordinal(rank),
        "duration_days": int(current['duration_days']),
        "intensity": float(current['intensity']),
        "start_date": pd.Timestamp(current['start_date']),
        "end_date": pd.Timestamp(current['end_date']),
        "parameter": parameter,
        "wave_type": "heatwave" if is_warm else "coldwave",
        "severity": "extreme" if "Extreme" in threshold_level else "strong",
        "n_events_on_record": len(waves_data),
    }


def get_kiesely_waves_figs(lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)", stat_metric="Cumulative Annual Wave Intensity"):
    empty_fig = go.Figure().add_annotation(text="Data Missing or Processing.", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="red", family=ATMOPULSE_FONTS["sora_css"]))
    empty_fig.update_layout(**plotly_typography())

    ds_archive = _load_waves_archive_ds()
    if ds_archive is None:
        return empty_fig, empty_fig

    suffix = "A" if selected_epoch == "A" else "B"
    var_key = _param_var(parameter)
    is_warm = _param_is_warm(parameter)
    thr = _wave_season_thresholds(lat, lon, suffix, parameter)
    if not thr:
        return empty_fig, empty_fig

    if is_warm:
        p_t_ext, p_d_ext = thr[f"{var_key}_p95"], thr[f"{var_key}_p90"]
        p_t_str, p_d_str = thr[f"{var_key}_p90"], thr[f"{var_key}_p75"]
    else:
        p_t_ext, p_d_ext = thr[f"{var_key}_p5"], thr[f"{var_key}_p10"]
        p_t_str, p_d_str = thr[f"{var_key}_p10"], thr[f"{var_key}_p25"]

    p_thresh, p_drop = (p_t_ext, p_d_ext) if "Extreme" in threshold_level else (p_t_str, p_d_str)
    
    if np.isnan(p_thresh) or np.isnan(p_drop): 
        return empty_fig, empty_fig

    df_season, group_key, diagnostics = _prepare_wave_season_df(lat, lon, parameter)
    if df_season.empty:
        return empty_fig, empty_fig

    if is_warm:
        tick_vals, tick_text = [16, 46, 77, 107, 138], ["MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER"]
        start_plot_x, end_plot_x = 1, 153
        grid_lines = [1, 32, 62, 93, 124, 154]
    else:
        tick_vals, tick_text = [16, 46, 77, 107, 136], ["NOV", "DEC", "JAN", "FEB", "MAR"]
        start_plot_x, end_plot_x = 1, 152
        grid_lines = [1, 31, 62, 93, 121, 152]

    # SUMMER-BUG FIX: _detect_kysely_waves() reindexes each season group onto
    # a full native Pandas date range internally, forcing clean date/day counting.
    waves_data = _detect_kysely_waves(df_season, group_key, p_thresh, p_drop, parameter)

    # TEMP DIAGNOSTICS (see app.py debug panel) — safe to remove once the
    # TX-vs-TN wave-count discrepancy is root-caused.
    debug_info = {
        **diagnostics,
        "p_thresh": float(p_thresh),
        "p_drop": float(p_drop),
        "p_t_ext": float(p_t_ext), "p_d_ext": float(p_d_ext),
        "p_t_str": float(p_t_str), "p_d_str": float(p_d_str),
        "threshold_source": "daily_extrema_jja_djf_true24h_era5",
        "threshold_epoch": thr["epoch_years"],
        "threshold_note": "ERA5 tx/tn true 24h daily statistics (era5_master_daily_*.nc)",
        "n_years_all_nan_season": int(sum(
            1 for _, g in df_season.groupby(group_key) if g['Temp'].isna().all()
        )),
        "n_days_above_p_thresh_season": int((df_season['Temp'] >= p_thresh).sum()) if is_warm else int((df_season['Temp'] <= p_thresh).sum()),
        "season_daily_max": float(df_season['Temp'].max()) if not df_season.empty else None,
        "n_waves_detected": int(len(waves_data)),
    }

    fig_main = go.Figure()
    start_year, end_year = 1940, 2026
    y_ticks_vals = list(range(start_year, end_year + 1))
    y_ticks_text = [str(y) if is_warm else f"{y-1}/{str(y)[2:]}" for y in y_ticks_vals]
    
    t_suff = "Heatwaves" if is_warm else "Coldwaves"
    lvl_text = "Extreme Level (P95/5)" if "Extreme" in threshold_level else "Strong Level (P90/10)"
    
    fig_main.update_layout(
        **plotly_typography(),
        title=dict(
            text=f"Duration and Intensity of Local {parameter} {t_suff} (1940–2026) | {lvl_text}<br><span style='font-size:11px;color:gray;'>Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}</span>",
            font=plotly_title_font(size=13),
        ),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - (5.0 if is_warm else 15.0)], showgrid=False, zeroline=False, showline=False),
        height=750, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=55, r=20, t=50, b=40),
        meta=debug_info,
    )
    
    for gl in grid_lines: 
        fig_main.add_vline(x=gl, line_width=1.2, line_color="rgba(30,30,30,0.6)", layer="below")
    for yr in range(start_year, end_year + 1, 5): 
        fig_main.add_hline(y=yr, line_width=0.7, line_color="rgba(100,100,100,0.4)", layer="below")
        
    if waves_data:
        for w in waves_data:
            y_base, w_xs, w_ts = w['year'], np.array(w['xs']), np.array(w['temps'])
            cum_sum = np.cumsum(np.maximum(0, w_ts - p_thresh) if is_warm else np.maximum(0, p_thresh - w_ts))
            
            w_df = pd.DataFrame({'x': w_xs, 'y': cum_sum}).drop_duplicates(subset=['x']).sort_values('x')
            w_xs, cum_sum = w_df['x'].values, w_df['y'].values
            
            if len(w_xs) >= 3:
                x_fine = np.linspace(w_xs[0], w_xs[-1], WAVE_RIDGE_SPLINE_PTS)
                y_fine = np.clip(make_interp_spline(w_xs, cum_sum, k=2)(x_fine), 0, None)
                x_skewed = x_fine + (y_fine / (max(y_fine) if max(y_fine) > 0 else 1)) * WAVE_RIDGE_SKEW_FACTOR
            else:
                x_skewed, y_fine = w_xs, cum_sum

            break_x, y_break = _wave_break_tail(x_skewed[-1], y_fine[-1])

            x_full = np.concatenate(([x_skewed[0]], x_skewed, break_x, [break_x[-1]]))
            y_full = np.concatenate(([0.0], y_fine, y_break, [0.0]))
            y_coords = y_base - (y_full / WAVE_RIDGE_HEIGHT_SCALE)

            cap = WAVE_INTENSITY_CAP_TX if is_warm else WAVE_INTENSITY_CAP_TN
            norm_val = min(w['intensity'] / cap, 1.0)
            (r_b, g_b, b_b), (r, g, b) = _wave_ridge_colors(parameter, norm_val)

            sd_str, ed_str = pd.to_datetime(w['start_date']).strftime('%d.%m.'), pd.to_datetime(w['end_date']).strftime('%d.%m.%Y')
            
            fig_main.add_trace(go.Scatter(
                x=x_full, y=y_coords, mode='lines',
                line=dict(color=f"rgba({r},{g},{b},{WAVE_LINE_ALPHA})", width=WAVE_LINE_WIDTH, shape='spline'),
                fill='toself',
                fillgradient=dict(type='vertical', colorscale=[
                    [0, f"rgba({r_b},{g_b},{b_b},{WAVE_FILL_ALPHA_BASE})"],
                    [1, f"rgba({r},{g},{b},{WAVE_FILL_ALPHA_PEAK})"],
                ]),
                hoverinfo='text',
                text=f"<b>Duration: {sd_str}–{ed_str}</b><br>Length: {len(w_xs)} days<br>Severity: {w['intensity']:.1f} K",
                showlegend=False,
            ))
    else: 
        fig_main.add_annotation(text="No wave events detected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="gray", family=ATMOPULSE_FONTS["sora_css"]))

    if stat_metric == "Annual Cycle Frequency":
        df_season['is_str'] = (df_season['Temp'] >= p_t_str) if is_warm else (df_season['Temp'] <= p_t_str)
        df_season['is_ext'] = (df_season['Temp'] >= p_t_ext) if is_warm else (df_season['Temp'] <= p_t_ext)
        f_str = (df_season.groupby('plot_x')['is_str'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        f_ext = (df_season.groupby('plot_x')['is_ext'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        
        fig_stats = go.Figure()
        if is_warm:
            c_str, c_ext = ATMOPULSE_WARM["p90"], ATMOPULSE_WARM["p95"]
        else:
            c_str, c_ext = ATMOPULSE_COLD["p10"], ATMOPULSE_COLD["p5"]
            
        fig_stats.add_trace(go.Scatter(x=f_str.index, y=f_str.values, mode='lines', line=dict(color=c_str, width=2), name="Strong", hovertemplate='%{y:.1f}%<extra></extra>'))
        fig_stats.add_trace(go.Scatter(x=f_ext.index, y=f_ext.values, mode='lines', line=dict(color=c_ext, width=2), name="Extreme", hovertemplate='%{y:.1f}%<extra></extra>'))
        
        fig_stats.update_layout(
            **plotly_typography(), 
            title=f"Annual Cycle Frequency (5-Day Smoothing) | Reference {'1961–1990' if suffix=='A' else '1996–2025'}", 
            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, showgrid=True), 
            yaxis_title="Relative Frequency (%)", 
            height=350, template="plotly_white", 
            margin=dict(t=40, b=10, l=10, r=10), 
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        return fig_main, fig_stats

    stats = pd.DataFrame(index=np.arange(start_year, end_year + 1))
    stats['max_int'], stats['sum_int'], stats['total_heat'] = 0.0, 0.0, 0.0
    
    for yr in stats.index:
        y_waves = [w for w in waves_data if w['year'] == yr]
        if y_waves:
            stats.loc[yr, 'max_int'] = max(w['intensity'] for w in y_waves)
            stats.loc[yr, 'sum_int'] = sum(w['intensity'] for w in y_waves)
        y_df = df_season[df_season[group_key] == yr]
        stats.loc[yr, 'total_heat'] = sum(t - p_thresh for t in y_df['Temp'] if t >= p_thresh) if is_warm else sum(p_thresh - t for t in y_df['Temp'] if t <= p_thresh)

    col_map = {"Cumulative Annual Wave Intensity": 'sum_int', "Maximum Annual Wave Intensity": 'max_int', "Cumulative Heat/Cold Intensity": 'total_heat'}
    sel_col = col_map.get(stat_metric, 'sum_int')
    y_titles = {
        'sum_int': 'Σ wave intensity (K·days)',
        'max_int': 'Max wave intensity (K·days)',
        'total_heat': f'Σ excess vs. threshold (K·days)',
    }

    fig_stats = go.Figure()
    if is_warm:
        bar_color, mean_color = "#E8A8A0", ATMOPULSE_WARM["p95"]
    else:
        bar_color, mean_color = "#9EC5E8", ATMOPULSE_COLD["p5"]

    fig_stats.add_trace(go.Bar(
        x=stats.index, y=stats[sel_col], marker_color=bar_color, name="Intensity",
        hovertemplate='Year: %{x}<br>Value: %{y:.1f} K<extra></extra>',
    ))
    fig_stats.add_trace(go.Scatter(
        x=stats.index, y=stats[sel_col].rolling(11, center=True).mean(), mode='lines',
        line=dict(color=mean_color, width=2.5), name="11-yr Mean",
        hovertemplate='Year: %{x}<br>11-year mean: %{y:.1f} K<extra></extra>',
    ))

    valid = stats[sel_col].dropna()
    if len(valid) > 2:
        z = np.polyfit(valid.index, valid.values, 1)
        fig_stats.add_trace(go.Scatter(
            x=valid.index, y=np.poly1d(z)(valid.index), mode='lines',
            line=dict(color=mean_color, width=1.5, dash='dot'), name="Trend", hoverinfo='skip',
        ))

    fig_stats.update_layout(
        **plotly_typography(),
        title=f"{stat_metric} | Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}",
        height=350, template="plotly_white",
        margin=dict(t=40, b=10, l=55, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(180,180,180,0.35)", gridwidth=1, dtick=10, zeroline=False),
        yaxis=dict(title=y_titles.get(sel_col, "Intensity (K·days)"), showgrid=True, gridcolor="rgba(180,180,180,0.35)", gridwidth=1, zeroline=False),
        bargap=0.15,
    )

    return fig_main, fig_stats