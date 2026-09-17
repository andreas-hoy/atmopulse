"""
AtmoPulse Wave Detection Analytics (backend_waves.py)

This module handles the extraction and dynamic thresholding of synoptic
extreme events (summer heatwaves and winter coldwaves) using an adapted
Kyselý definition. Compute-only: pandas/numpy/xarray, no Plotly, no
atmopulse_theme. The Point Wavogram's Plotly ridge-plots and annual
intensity bar charts are built by `frontend_plots.build_kysely_wave_figs`
from the dict returned by `compute_kysely_waves_data` below.

Core functionalities:
- Extracts point TX/TN exclusively from era5_master_daily_YYYY.nc (IFS/AIFS only for the last 6 days and the forecast).
- Dynamically calculates seasonal climatological thresholds (P95, P90, P75 for JJA; 
  P5, P10, P25 for DJF) based on shifting reference periods (1961-1990 or 1996-2025).
- Excises leap days (Feb 29th) to ensure statistical homoscedasticity per ETCCDI norms.
- Identifies consecutive threshold exceedances year-round (calendar year for
  heatwaves, July–June for coldwaves) and applies trailing tolerance drops.
  The ridge plot may crop the x-axis to a core season; that is display-only.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import streamlit as st

from backend_maps import drop_era5t_aux, etccdi_doy_365
from config import ZARR_MASTER_TIME_SERIES

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
if not CLIM_FILE.exists(): 
    CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")

# All parameters the Point Wavogram can plot. "var" is the on-disk/Zarr
# variable name; "is_warm" is the default season/direction when the UI does
# not pass an event-type override (Heatwaves vs Coldwaves).
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


def _resolve_is_warm(parameter: str, is_warm=None) -> bool:
    """Heatwaves/Coldwaves from the UI wins; otherwise the parameter default."""
    if is_warm is None:
        return _param_is_warm(parameter)
    return bool(is_warm)


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
    keep = [v for v in ("tx", "tn", "tg", "t850", "z500") if v in ds.data_vars]
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
    def _z(v):
        if v not in pt:
            return [None] * len(days)
        a = np.squeeze(np.asarray(pt[v].values, dtype=float))
        a = np.atleast_1d(a)
        finite = a[np.isfinite(a)]
        if finite.size:
            sample = float(np.mean(finite))
            if sample > 10000:
                a = a / 9.80665 / 10.0
            elif sample > 2000:
                a = a / 10.0
        return [None if not np.isfinite(x) else float(x) for x in a]
    payload = {"Date": [d.strftime("%Y-%m-%d") for d in days]}
    for v in keep:
        payload[v] = _z(v) if v == "z500" else _c(v)
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
WAVE_SYNOPTIC_VARS = ("z500",)
# Wave-mean Z500 anomaly (dam) that counts as a supporting ridge (heat)
# or trough (cold). Two map-tracker anomaly isolines (8 dam), not one.
WAVE_Z500_RIDGE_DAM = 8.0
# Detection windows (thresholds stay JJA / DJF). Display may crop to a
# shorter core season in frontend_plots — that must not truncate events.
WAVE_PLOT_X_MAX = 365


def _wave_plot_x(dates, is_warm: bool) -> np.ndarray:
    """Season x-coordinate on the ETCCDI 365-day axis (same as the month ticks).

    Warm: day-of-year 1–365. Cold: days since 1 July on that same calendar
    (1 July = 1, 30 June = 365). 29 February shares the 1 March slot, so
    leap years no longer sit one day to the right of non-leap years.
    """
    d = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(dates)))
    if d.tz is not None:
        d = d.tz_convert("UTC").tz_localize(None)
    doy = np.asarray(etccdi_doy_365(d), dtype=np.int64)
    if is_warm:
        return doy.astype(np.float64)
    months = d.month.to_numpy()
    return np.where(months >= 7, doy - 181, doy + 184).astype(np.float64)


def _wave_season_origin(yr: int, is_warm: bool) -> pd.Timestamp:
    yr = int(yr)
    if is_warm:
        return pd.Timestamp(year=yr, month=1, day=1)
    return pd.Timestamp(year=yr, month=7, day=1)


def _wave_season_end(yr: int, is_warm: bool) -> pd.Timestamp:
    yr = int(yr)
    if is_warm:
        return pd.Timestamp(year=yr, month=12, day=31)
    return pd.Timestamp(year=yr + 1, month=6, day=30)


def _decode_point_var(name: str, values) -> np.ndarray:
    if name == "z500":
        from backend_io import _z500_to_dam
        return _z500_to_dam(values)
    return _kelvin_to_celsius_if_needed(values)


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
    if not ZARR_MASTER_TIME_SERIES.exists():
        return pd.DataFrame()
    try:
        zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=True)
    except Exception:
        try:
            zds = xr.open_zarr(ZARR_MASTER_TIME_SERIES, consolidated=False)
        except Exception:
            return pd.DataFrame()

    keep = [v for v in WAVE_ARCHIVE_VARS + WAVE_SYNOPTIC_VARS if v in zds.data_vars]
    if not keep:
        return pd.DataFrame()

    try:
        pt = zds[keep].sel(latitude=lat, longitude=lon, method="nearest").compute()
        t = pd.to_datetime(pt["valid_time"].values)
        if getattr(t, "tz", None) is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        rec = {"Date": pd.DatetimeIndex(t).normalize()}
        for v in keep:
            rec[v] = _decode_point_var(v, pt[v].values)
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
            keep = [v for v in WAVE_ARCHIVE_VARS + WAVE_SYNOPTIC_VARS if v in ds.data_vars]
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
                rec[v] = _decode_point_var(v, pt[v].values)
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
def _era5_master_point_series(lat: float, lon: float, include_z500: bool = True, _archive_version=8) -> pd.DataFrame:
    """Full 1940–present TX/TN/TG/T850 (and Z500) at one grid cell.

    Fast path: reads the point-extraction-optimal Zarr mirror of the master
    archive (see batch_convert_netcdf_to_zarr.py / config.ZARR_MASTER_TIME_SERIES)
    — millisecond reads instead of the old year-by-year NetCDF loop. Falls
    back to that legacy loop whenever the Zarr store doesn't exist yet or
    fails to read for any reason, so wave charts never break before/without
    the migration having been run.

    IFS/AIFS is overlaid solely for the last 6 days through the forecast,
    for whichever of tx/tn/tg/t850/z500 the live forecast dataset carries.
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
            if "z500" in pt.data_vars:
                rec["z500"] = _decode_point_var("z500", pt["z500"].values)
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
def _wave_season_thresholds(lat: float, lon: float, suffix: str, parameter: str = "TX", is_warm=None) -> dict:
    """
    Kyselý seasonal thresholds from true 24h daily max (JJA, heat) / min
    (DJF, cold) of the requested `parameter`, read via
    `_era5_master_point_series` (Zarr fast path, NetCDF fallback). IFS/AIFS
    is used only for the last 6 days and the forecast — never as a
    historical fill. `is_warm` follows Heatwaves/Coldwaves when the UI
    passes it; otherwise the parameter default (TX/TG/T850 warm, TN cold).

    Returned dict keys are prefixed with the parameter's variable name
    (e.g. "tx_p75"/"tg_p90"/"t850_p95") so multiple parameters' thresholds
    never collide, plus "epoch_years".
    """
    var_key = _param_var(parameter)
    is_warm = _resolve_is_warm(parameter, is_warm)

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


def _prepare_wave_season_df(lat, lon, parameter="TX", is_warm=None) -> tuple[pd.DataFrame, str, dict]:
    """
    Shared data-prep pipeline for Kyselý wave analytics: point extraction,
    Kelvin normalization, Feb-29 excision, true-24h daily resampling and
    year grouping (calendar year for heatwaves, July–June for coldwaves).
    Detection is year-round; JJA/DJF percentiles stay the thresholds.

    Factored out so `compute_kysely_waves_data` (ridge-plot data) and
    the historical-rank lookup (`get_wave_historical_rank`) run the exact
    same season/day construction and can never silently drift apart.
    `is_warm` is the Heatwaves/Coldwaves override when the UI passes it.

    Returns (df_season, group_key, diagnostics) where `diagnostics` carries
    the raw-value QA fields used by the ridge-plot's debug panel.
    """
    df_pt = _era5_master_point_series(lat, lon)
    if df_pt.empty:
        return pd.DataFrame(), "", {}

    var_key = _param_var(parameter)
    is_warm = _resolve_is_warm(parameter, is_warm)
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

    df_season = df.copy()
    if is_warm:
        group_key = 'year'
        df_season['plot_x'] = _wave_plot_x(df_season['date'], True)
    else:
        df_season['winter_year'] = np.where(
            df_season['month'] >= 7, df_season['year'], df_season['year'] - 1,
        )
        group_key = 'winter_year'
        df_season['plot_x'] = _wave_plot_x(df_season['date'], False)

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


def _detect_kysely_waves(df_season: pd.DataFrame, group_key: str, p_thresh: float, p_drop: float, parameter: str, is_warm=None) -> list[dict]:
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
    is_warm = _resolve_is_warm(parameter, is_warm)

    for yr, group in df_season.groupby(group_key):
        group = group.drop_duplicates(subset=['date'], keep='first')

        full_dates = pd.date_range(
            _wave_season_origin(yr, is_warm), _wave_season_end(yr, is_warm),
        )

        # 29 Feb stays on this index as a real calendar day (live rendering);
        # a coldwave streak crossing it is one continuous run, not a gap.
        # Missing ERA5 days (true holes) still surface as NaN — never filled.
        group = group.set_index('date').reindex(full_dates)
        group['plot_x'] = _wave_plot_x(group.index, is_warm)

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
                    # Stable identity for the drill-down UI: (start, end) as
                    # ISO date strings, not the trace/list index (which shifts
                    # whenever the epoch, threshold or variable changes).
                    event_id = (
                        f"{pd.Timestamp(cand_dates[0]).date().isoformat()}_"
                        f"{pd.Timestamp(cand_dates[-1]).date().isoformat()}"
                    )
                    waves_data.append({
                        'year': yr,
                        'xs': cand_xs,
                        'temps': cand_temps,
                        'intensity': intensity,
                        'start_date': cand_dates[0],
                        'end_date': cand_dates[-1],
                        'duration_days': len(cand_temps),
                        'event_id': event_id,
                    })
                i = j if j > i else i + 1
            else:
                i += 1

    return waves_data


def _attach_wave_z500_anomalies(waves_data: list[dict], lat, lon, suffix: str) -> pd.Series | None:
    """Mean Z500 anomaly (dam) over each wave window vs that epoch's DOY mean.

    Does not change Kyselý detection. Missing Z500 or climatology leaves
    the anomaly fields unset / NaN. Mean, max and min are all over the
    full [start, end] window. Returns the point Z500 series (dam) for the
    expert drill-down mini-charts, or None if Z500 is unavailable.
    """
    from backend_io import load_synoptic_climatology, synoptic_clim_point_doy

    df_pt = _era5_master_point_series(lat, lon)
    if df_pt.empty or "z500" not in df_pt.columns:
        return None

    dates_pt = pd.DatetimeIndex(pd.to_datetime(df_pt["Date"]))
    if dates_pt.tz is not None:
        dates_pt = dates_pt.tz_convert("UTC").tz_localize(None)
    dates_pt = dates_pt.normalize()
    z_vals = np.asarray(df_pt["z500"].values, dtype=np.float64)
    series = pd.Series(z_vals, index=dates_pt, name="z500")

    if not waves_data:
        return series
    clim = synoptic_clim_point_doy(load_synoptic_climatology(), "z500", suffix, lat, lon)
    if clim is None or clim.size < 365:
        return series

    for w in waves_data:
        start = pd.Timestamp(w["start_date"]).normalize()
        end = pd.Timestamp(w["end_date"]).normalize()
        mask = (dates_pt >= start) & (dates_pt <= end)
        if not bool(np.any(mask)):
            w["z500_anom_mean"] = float("nan")
            w["z500_anom_max"] = float("nan")
            w["z500_anom_min"] = float("nan")
            continue
        doys = etccdi_doy_365(dates_pt[mask])
        z_clim = clim[np.clip(doys - 1, 0, len(clim) - 1)]
        anom = z_vals[mask] - z_clim
        if np.isfinite(anom).any():
            w["z500_anom_mean"] = float(np.nanmean(anom))
            w["z500_anom_max"] = float(np.nanmax(anom))
            w["z500_anom_min"] = float(np.nanmin(anom))
        else:
            w["z500_anom_mean"] = float("nan")
            w["z500_anom_max"] = float("nan")
            w["z500_anom_min"] = float("nan")
    return series


_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else _ORDINAL_SUFFIXES.get(n % 10, "th")
    return f"{n}{suffix}"


def _wave_day_frequency(waves_data, df_season, group_key, n_plot_x) -> pd.Series:
    """Percent of years in which that season-day sits inside a detected wave.

    Isolated threshold exceedances that never formed a 3-day Kyselý event
    do not count. Zeros stay zero so the frequency panel can hide empty
    stretches of the season. A 5-day centered rolling mean matches the
    previous threshold-day smoother.
    """
    idx = list(range(1, int(n_plot_x) + 1))
    if df_season is None or df_season.empty:
        return pd.Series(np.nan, index=idx)

    wave_days = set()
    for w in waves_data or []:
        yr = int(w["year"])
        for x in w.get("xs", []):
            wave_days.add((yr, int(x)))

    valid = df_season.dropna(subset=["Temp"]).copy()
    if valid.empty:
        return pd.Series(np.nan, index=idx)

    px = pd.to_numeric(valid["plot_x"], errors="coerce")
    valid = valid.loc[px.notna()].copy()
    valid["plot_x"] = px.loc[valid.index].astype(int)
    n_map = valid.groupby("plot_x")[group_key].nunique().to_dict()
    n_map = {int(k): int(v) for k, v in n_map.items()}
    counts = Counter(x for (_yr, x) in wave_days)
    pct = []
    for x in idx:
        n = n_map.get(x, 0)
        if n <= 0:
            pct.append(np.nan)
        else:
            pct.append(100.0 * counts.get(x, 0) / n)
    return pd.Series(pct, index=idx).rolling(5, center=True, min_periods=1).mean()


def rank_waves_by_metric(waves_data: list[dict], metric: str = "Intensity") -> list[dict]:
    """Sorted copy of `waves_data` for the Point Wavogram drill-down: same
    metric as the annual intensity-stack switch (Intensity = Kyselý K·days
    excess, Days = duration_days), descending. Ties use the *other* metric
    (also descending), then earlier start_date only if both are equal.
    Adds a 1-based 'rank' (over the full list, not just the returned slice)
    to each copied dict.

    Read-only w.r.t. `waves_data` / detection — this only reorders the
    already-detected events for display, it does not call
    `_detect_kysely_waves` again or change thresholds.
    """
    use_days = str(metric).lower().startswith("day")

    def _sort_key(w):
        intensity = float(w.get("intensity", 0) or 0)
        duration = float(w.get("duration_days", 0) or 0)
        start = pd.Timestamp(w["start_date"])
        if use_days:
            return (-duration, -intensity, start)
        return (-intensity, -duration, start)

    ordered = sorted(waves_data or [], key=_sort_key)
    ranked = []
    for i, w in enumerate(ordered, start=1):
        w2 = dict(w)
        w2["rank"] = i
        ranked.append(w2)
    return ranked


def get_wave_historical_rank(
    lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)",
    target_date=None, top_n: int = 20, is_warm=None,
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
    is_warm = _resolve_is_warm(parameter, is_warm)
    thr = _wave_season_thresholds(lat, lon, suffix, parameter, is_warm=is_warm)
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

    df_season, group_key, _ = _prepare_wave_season_df(lat, lon, parameter, is_warm=is_warm)
    if df_season.empty:
        return None

    waves_data = _detect_kysely_waves(df_season, group_key, p_thresh, p_drop, parameter, is_warm=is_warm)
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


@st.cache_data(show_spinner=False)
def compute_kysely_waves_data(lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)", stat_metric="Cumulative Annual Wave Intensity", is_warm=None, z500_ctx=5):
    """
    Compute-only payload for the Point Wavogram (ridge-plot + stats panel).
    Pure pandas/numpy/xarray — no Plotly, no theme, no drawing-only spline
    smoothing (that lives in `frontend_plots.build_kysely_wave_figs`, along
    with the WAVE_RIDGE_*/WAVE_BREAK_* aesthetic constants). Cached on the
    hashable (lat, lon, parameter, selected_epoch, threshold_level, is_warm)
    inputs. Annual intensity stacks and the seasonal frequency series are
    always computed together. The two rendered Figures are built from this
    unhashed payload separately so they never have to be pickled/hashed by
    Streamlit's cache.

    `empty=True` means "the caller should render the placeholder figure" —
    same trigger conditions (missing archive, missing thresholds, empty
    season data, NaN thresholds) as the previous `get_kiesely_waves_figs`.
    Frequency is the share of years in which that season-day belongs to a
    detected Strong / Extreme Kyselý wave, not isolated threshold days.
    """
    suffix = "A" if selected_epoch == "A" else "B"
    is_warm = _resolve_is_warm(parameter, is_warm)
    base = {
        "empty": True,
        "parameter": parameter,
        "epoch": suffix,
        "threshold_level": threshold_level,
        "is_warm": is_warm,
    }

    ds_archive = _load_waves_archive_ds()
    if ds_archive is None:
        return base

    var_key = _param_var(parameter)
    thr = _wave_season_thresholds(lat, lon, suffix, parameter, is_warm=is_warm)
    if not thr:
        return base

    if is_warm:
        p_t_ext, p_d_ext = thr[f"{var_key}_p95"], thr[f"{var_key}_p90"]
        p_t_str, p_d_str = thr[f"{var_key}_p90"], thr[f"{var_key}_p75"]
    else:
        p_t_ext, p_d_ext = thr[f"{var_key}_p5"], thr[f"{var_key}_p10"]
        p_t_str, p_d_str = thr[f"{var_key}_p10"], thr[f"{var_key}_p25"]

    p_thresh, p_drop = (p_t_ext, p_d_ext) if "Extreme" in threshold_level else (p_t_str, p_d_str)
    if np.isnan(p_thresh) or np.isnan(p_drop):
        return base

    df_season, group_key, diagnostics = _prepare_wave_season_df(lat, lon, parameter, is_warm=is_warm)
    if df_season.empty:
        return base

    # SUMMER-BUG FIX: _detect_kysely_waves() reindexes each season group onto
    # a full native Pandas date range internally, forcing clean date/day counting.
    waves_str = _detect_kysely_waves(df_season, group_key, p_t_str, p_d_str, parameter, is_warm=is_warm)
    waves_ext = _detect_kysely_waves(df_season, group_key, p_t_ext, p_d_ext, parameter, is_warm=is_warm)
    waves_data = waves_ext if "Extreme" in threshold_level else waves_str
    z500_series = _attach_wave_z500_anomalies(waves_data, lat, lon, suffix)

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

    payload = {
        "empty": False,
        "parameter": parameter,
        "epoch": suffix,
        "threshold_level": threshold_level,
        "is_warm": is_warm,
        "lat": float(lat),
        "lon": float(lon),
        "var_key": var_key,
        # Wave table / ridge series: one dict per detected event (year, xs,
        # temps, intensity, start_date, end_date, duration_days, event_id).
        "waves_data": waves_data,
        # Raw daily series (indexed by calendar date, epoch-independent) for
        # the drill-down mini-charts — lets them draw the +/-3 display days
        # around an event without re-deriving season groups. Same 'Temp'
        # column _detect_kysely_waves works from, just not reindexed/split
        # into season blocks.
        "daily_series": (
            df_season.drop_duplicates(subset=["date"], keep="first")
            .set_index("date")["Temp"]
        ),
        "z500_series": z500_series,
        "z500_ridge_dam": WAVE_Z500_RIDGE_DAM,
        "group_key": group_key,
        "p_thresh": p_thresh,
        "p_drop": p_drop,
        "p_t_ext": p_t_ext, "p_d_ext": p_d_ext,
        "p_t_str": p_t_str, "p_d_str": p_d_str,
        "epoch_years": thr["epoch_years"],
        "debug_info": debug_info,
    }

    start_year, end_year = 1940, 2026

    n_plot_x = WAVE_PLOT_X_MAX
    payload["freq_series"] = {
        "f_str": _wave_day_frequency(waves_str, df_season, group_key, n_plot_x),
        "f_ext": _wave_day_frequency(waves_ext, df_season, group_key, n_plot_x),
    }

    stats = pd.DataFrame(index=np.arange(start_year, end_year + 1))
    stats["max_int"] = 0.0
    stats["sum_int"] = 0.0
    stats["total_heat"] = 0.0
    stats["max_days"] = 0.0
    stats["sum_days"] = 0.0
    stats["isolated_days"] = 0.0
    stats["max_z500_mean"] = np.nan
    stats["max_z500_ext"] = np.nan
    stats["max_supporting"] = 0.0

    wave_day_keys = {
        (int(w["year"]), int(x))
        for w in waves_data
        for x in w.get("xs", [])
    }

    for yr in stats.index:
        y_waves = [w for w in waves_data if int(w["year"]) == int(yr)]
        if y_waves:
            stats.loc[yr, "max_int"] = max(w["intensity"] for w in y_waves)
            stats.loc[yr, "sum_int"] = sum(w["intensity"] for w in y_waves)
            top = max(y_waves, key=lambda w: w["intensity"])
            stats.loc[yr, "max_days"] = float(top["duration_days"])
            stats.loc[yr, "sum_days"] = float(sum(w["duration_days"] for w in y_waves))
            z_mean = top.get("z500_anom_mean")
            z_mean = float(z_mean) if z_mean is not None and np.isfinite(z_mean) else float("nan")
            z_ext = top.get("z500_anom_max") if is_warm else top.get("z500_anom_min")
            z_ext = float(z_ext) if z_ext is not None and np.isfinite(z_ext) else float("nan")
            stats.loc[yr, "max_z500_mean"] = z_mean
            stats.loc[yr, "max_z500_ext"] = z_ext
            if np.isfinite(z_mean):
                supporting = (z_mean >= WAVE_Z500_RIDGE_DAM) if is_warm else (z_mean <= -WAVE_Z500_RIDGE_DAM)
                stats.loc[yr, "max_supporting"] = 1.0 if supporting else 0.0
        y_df = df_season[df_season[group_key] == yr]
        temps = np.asarray(y_df["Temp"], dtype=float)
        px = pd.to_numeric(y_df["plot_x"], errors="coerce")
        if is_warm:
            stats.loc[yr, "total_heat"] = sum(t - p_thresh for t in temps if t >= p_thresh)
            exc = temps >= p_thresh
        else:
            stats.loc[yr, "total_heat"] = sum(p_thresh - t for t in temps if t <= p_thresh)
            exc = temps <= p_thresh
        in_wave = np.array([
            (int(yr), int(x)) in wave_day_keys if np.isfinite(x) else False
            for x in px
        ], dtype=bool)
        stats.loc[yr, "isolated_days"] = float(np.sum(np.isfinite(temps) & exc & ~in_wave))

    payload["annual_stats"] = stats
    return payload