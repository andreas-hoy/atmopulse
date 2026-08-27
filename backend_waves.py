"""
AtmoPulse Wave Detection & Ridge-Plot Analytics (backend_waves.py)

This module handles the extraction, dynamic thresholding, and visualization of 
synoptic extreme events (summer heatwaves and winter coldwaves) using an adapted 
Kyselý definition. 

Core functionalities:
- Manages high-speed, Zarr-backed extraction of ERA5 point time-series.
- Dynamically calculates seasonal climatological thresholds (P95, P90, P75 for JJA; 
  P5, P10, P25 for DJF) based on shifting reference periods (1961-1990 or 1996-2025).
- Excises leap days (Feb 29th) to ensure statistical homoscedasticity per ETCCDI norms.
- Identifies consecutive threshold exceedances and applies trailing tolerance drops.
- Renders highly customized, interactive Plotly ridge-plots and annual intensity 
  bar charts representing cumulative thermal stress (K·days).
"""

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
    if parameter == "TX":
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


@st.cache_resource(show_spinner=False)
def _load_waves_archive_ds(_archive_version=2):
    """
    SINGLETON POINTER (ZARR OPTIMIZED): Opens the pre-consolidated Zarr archive 
    exactly ONCE per server process and keeps the lazy handle alive in memory.

    Migrated from xr.open_mfdataset() over 18 NetCDF batch files (spatially
    compressed, so a single lat/lon time-series extraction had to decompress
    large spatial blocks into RAM) to xr.open_zarr() on a store re-chunked
    for time-series access. Downstream code still only `.sel(lat, lon, method='nearest')`s
    this one shared handle.
    """
    zarr_path = DATA_DIR / "era5_txtn_archive.zarr"
    if not zarr_path.exists():
        return None
    ds = xr.open_zarr(zarr_path, consolidated=True)
    rename = {}
    if "mx2t" in ds.data_vars and "tx" not in ds.data_vars:
        rename["mx2t"] = "tx"
    if "mn2t" in ds.data_vars and "tn" not in ds.data_vars:
        rename["mn2t"] = "tn"
    if rename:
        ds = ds.rename(rename)
    return drop_era5t_aux(ds)


@st.cache_resource(show_spinner=False)
def _load_waves_climatology():
    """SINGLETON POINTER for the reference climatology file used by the wave charts."""
    if not CLIM_FILE.exists():
        return None
    return xr.open_dataset(CLIM_FILE, engine='netcdf4')


@st.cache_data(show_spinner=False)
def _wave_season_thresholds(lat: float, lon: float, suffix: str) -> dict:
    """
    Kyselý seasonal thresholds from true 24h daily TX max (JJA) / TN min (DJF),
    read from the era5_txtn_archive.zarr point-timeseries archive (tx/tn,
    already one true daily statistic per calendar day since the migration
    to era5_master_daily_*.nc / "derived-era5-single-levels-daily-statistics").
    """
    ds_archive = _load_waves_archive_ds()
    if ds_archive is None:
        return {}
    
    pt = ds_archive.sel(latitude=lat, longitude=lon, method="nearest").compute()
    times = pd.to_datetime(pt.valid_time.values)

    # ETCCDI-STANDARD: Remove February 29th to ensure statistical homoscedasticity.
    pt = pt.sel(valid_time=~((times.month == 2) & (times.day == 29)))
    times = pd.to_datetime(pt.valid_time.values)

    if suffix == "A":
        y0, y1 = 1961, 1990
    else:
        y1 = int(times.max().year if times.max().month >= 12 else times.max().year - 1)
        y0 = y1 - 29

    def _daily_series(var: str) -> pd.Series:
        """
        One true 24h daily value per calendar day (tx/tn from the master
        batches). resample('D') is kept as a harmless idempotent safety net
        in case any transitional file still ships duplicate/sub-daily steps.
        """
        raw = np.asarray(pt[var].values, dtype=np.float64)
        if np.nanmean(raw[np.isfinite(raw)]) > 100:
            raw -= 273.15
        
        df = pd.DataFrame({var: raw}, index=times).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        agg = "max" if var == "tx" else "min"
        return df.resample("D").agg(agg)[var]

    tx_daily = _daily_series("tx")
    tn_daily = _daily_series("tn")
    yr_ok = lambda s: (s.index.year >= y0) & (s.index.year <= y1)

    jja = tx_daily[yr_ok(tx_daily) & tx_daily.index.month.isin([6, 7, 8])]
    djf = tn_daily[yr_ok(tn_daily) & tn_daily.index.month.isin([12, 1, 2])]

    tx_p75, tx_p90, tx_p95 = np.nanpercentile(jja, [75, 90, 95])
    tn_p5, tn_p10, tn_p25 = np.nanpercentile(djf, [5, 10, 25])
    
    return {
        "tx_p75": float(tx_p75), "tx_p90": float(tx_p90), "tx_p95": float(tx_p95),
        "tn_p5": float(tn_p5), "tn_p10": float(tn_p10), "tn_p25": float(tn_p25),
        "epoch_years": f"{y0}–{y1}",
    }


def get_kiesely_waves_figs(lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)", stat_metric="Cumulative Annual Wave Intensity"):
    empty_fig = go.Figure().add_annotation(text="Data Missing or Processing.", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="red", family=ATMOPULSE_FONTS["sora_css"]))
    empty_fig.update_layout(**plotly_typography())

    ds_archive = _load_waves_archive_ds()
    if ds_archive is None:
        return empty_fig, empty_fig

    suffix = "A" if selected_epoch == "A" else "B"
    thr = _wave_season_thresholds(lat, lon, suffix)
    if not thr:
        return empty_fig, empty_fig

    if parameter == "TX":
        p_t_ext, p_d_ext = thr["tx_p95"], thr["tx_p90"]
        p_t_str, p_d_str = thr["tx_p90"], thr["tx_p75"]
    else:
        p_t_ext, p_d_ext = thr["tn_p5"], thr["tn_p10"]
        p_t_str, p_d_str = thr["tn_p10"], thr["tn_p25"]

    p_thresh, p_drop = (p_t_ext, p_d_ext) if "Extreme" in threshold_level else (p_t_str, p_d_str)
    
    if np.isnan(p_thresh) or np.isnan(p_drop): 
        return empty_fig, empty_fig

    var_key = 'tx' if parameter == "TX" else 'tn'
    pt = ds_archive.sel(latitude=lat, longitude=lon, method='nearest').compute()

    # Robust cleanup for Zarr-sourced point extraction: some batches/tiles
    # can pack corrupted or unit-inconsistent values, which silently zeroed
    # out tx wave detection while tn happened to stay Celsius-clean.
    raw_vals = np.asarray(pt[var_key].values, dtype=np.float64)
    finite_vals = raw_vals[np.isfinite(raw_vals)]
    is_kelvin = finite_vals.size > 0 and np.nanmean(finite_vals) > 100.0
    
    if is_kelvin:
        raw_vals = raw_vals - 273.15

    df_raw = pd.DataFrame({'Temp': raw_vals}, index=pd.to_datetime(pt.valid_time.values))
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')].sort_index()
    
    # ETCCDI-STANDARD: Excision of February 29th from Kyselý wave detection.
    df_raw = df_raw[~((df_raw.index.month == 2) & (df_raw.index.day == 29))]
    
    # tx/tn are now true 24h daily statistics (one value per calendar day)
    # from era5_master_daily_*.nc. resample('D') is kept as a harmless
    # idempotent safety net against any leftover duplicate/sub-daily steps.
    if parameter == "TX":
        df_raw = df_raw.resample('D').max()
    else:
        df_raw = df_raw.resample('D').min()
        
    df = df_raw.copy()
    
    # Interpolate isolated NaNs (Zarr packing gaps) before building the
    # Kyselý wave-detection series; leaves the detection loop itself untouched.
    df['Temp'] = df['Temp'].interpolate(method='linear', limit_area='inside', limit=3)
    
    max_valid_date = pd.Timestamp.now() + pd.Timedelta(days=6)
    df = df[df.index <= max_valid_date]
    df['year'], df['month'], df['date'] = df.index.year, df.index.month, df.index.normalize()

    if parameter == "TX":
        df_season = df[df['month'].isin([5, 6, 7, 8, 9])].copy()
        group_key, tick_vals, tick_text = 'year', [16, 46, 77, 107, 138], ["MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER"]
        start_plot_x, end_plot_x = 1, 153
        grid_lines = [1, 32, 62, 93, 124, 154]
    else:
        df_season = df[df['month'].isin([11, 12, 1, 2, 3])].copy()
        df_season['winter_year'] = np.where(df_season['month'] <= 3, df_season['year'] - 1, df_season['year'])
        group_key, tick_vals, tick_text = 'winter_year', [16, 46, 77, 107, 136], ["NOV", "DEC", "JAN", "FEB", "MAR"]
        start_plot_x, end_plot_x = 1, 152
        grid_lines = [1, 31, 62, 93, 121, 152]

    if parameter == "TX":
        df_season['plot_x'] = (df_season['date'] - pd.to_datetime(df_season['year'].astype(str) + '-05-01')).dt.days + 1
    else:
        df_season['plot_x'] = (df_season['date'] - pd.to_datetime(df_season['winter_year'].astype(str) + '-11-01')).dt.days + 1

    waves_data = []

    # SUMMER-BUG FIX: Native Pandas reindexing forces clean date and day counting.
    for yr, group in df_season.groupby(group_key):
        group = group.drop_duplicates(subset=['date'], keep='first')
        
        if parameter == "TX":
            full_dates = pd.date_range(f"{int(yr)}-05-01", f"{int(yr)}-09-30")
        else:
            full_dates = pd.date_range(f"{int(yr)}-11-01", f"{int(yr)+1}-03-31")
            
        group = group.set_index('date').reindex(full_dates)
        group['Temp'] = group['Temp'].interpolate(method='linear', limit_area='inside')
        group['plot_x'] = np.arange(1, len(full_dates) + 1)
        
        temps, xs = group['Temp'].values, group['plot_x'].values
        dates = group.index.values
        n = len(temps)
        
        i = 0
        while i < n - 2:
            if np.isnan(temps[i:i+3]).any(): 
                i += 1
                continue
                
            if all((temps[i+k] >= p_thresh) if parameter == "TX" else (temps[i+k] <= p_thresh) for k in range(3)):
                cand_temps, cand_xs, cand_dates = [], [], []
                j = i
                while j < n and not np.isnan(temps[j]):
                    cand_temps.append(temps[j])
                    cand_xs.append(xs[j])
                    cand_dates.append(dates[j])
                    
                    drop_break = (temps[j] < p_drop) if parameter == "TX" else (temps[j] > p_drop)
                    mean_break = (np.mean(cand_temps) < p_thresh) if parameter == "TX" else (np.mean(cand_temps) > p_thresh)
                    
                    if drop_break or mean_break:
                        cand_temps.pop()
                        cand_xs.pop()
                        cand_dates.pop()
                        break
                    j += 1
                
                intensity = sum(abs(t - p_thresh) for t in cand_temps if ((t >= p_thresh) if parameter == "TX" else (t <= p_thresh)))
                
                if intensity > 0 and len(cand_temps) >= 3: 
                    waves_data.append({
                        'year': yr, 
                        'xs': cand_xs, 
                        'temps': cand_temps, 
                        'intensity': intensity, 
                        'start_date': cand_dates[0], 
                        'end_date': cand_dates[-1]
                    })
                i = j if j > i else i + 1
            else: 
                i += 1

    # TEMP DIAGNOSTICS (see app.py debug panel) — safe to remove once the
    # TX-vs-TN wave-count discrepancy is root-caused.
    debug_info = {
        "var_key": var_key,
        "is_kelvin_raw": bool(is_kelvin),
        "n_total": int(raw_vals.size),
        "n_nan_raw": int(np.isnan(raw_vals).sum()),
        "n_nan_after_pt_interp": int(df['Temp'].isna().sum()),
        "raw_min": float(np.nanmin(raw_vals)) if np.isfinite(raw_vals).any() else None,
        "raw_max": float(np.nanmax(raw_vals)) if np.isfinite(raw_vals).any() else None,
        "raw_mean": float(np.nanmean(raw_vals)) if np.isfinite(raw_vals).any() else None,
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
        "n_days_above_p_thresh_season": int((df_season['Temp'] >= p_thresh).sum()) if parameter == "TX" else int((df_season['Temp'] <= p_thresh).sum()),
        "season_daily_max": float(df_season['Temp'].max()) if not df_season.empty else None,
        "n_waves_detected": int(len(waves_data)),
    }

    fig_main = go.Figure()
    start_year, end_year = 1940, 2026
    y_ticks_vals = list(range(start_year, end_year + 1))
    y_ticks_text = [str(y) if parameter == "TX" else f"{y-1}/{str(y)[2:]}" for y in y_ticks_vals]
    
    t_suff = "Heatwaves" if parameter == "TX" else "Coldwaves"
    lvl_text = "Extreme Level (P95/5)" if "Extreme" in threshold_level else "Strong Level (P90/10)"
    
    fig_main.update_layout(
        **plotly_typography(),
        title=dict(
            text=f"Duration and Intensity of Local {t_suff} (1940–2026) | {lvl_text}<br><span style='font-size:11px;color:gray;'>Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}</span>",
            font=plotly_title_font(size=13),
        ),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - (5.0 if parameter == "TX" else 15.0)], showgrid=False, zeroline=False, showline=False),
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
            cum_sum = np.cumsum(np.maximum(0, w_ts - p_thresh) if parameter == "TX" else np.maximum(0, p_thresh - w_ts))
            
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

            cap = WAVE_INTENSITY_CAP_TX if parameter == "TX" else WAVE_INTENSITY_CAP_TN
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
        df_season['is_str'] = (df_season['Temp'] >= p_t_str) if parameter == "TX" else (df_season['Temp'] <= p_t_str)
        df_season['is_ext'] = (df_season['Temp'] >= p_t_ext) if parameter == "TX" else (df_season['Temp'] <= p_t_ext)
        f_str = (df_season.groupby('plot_x')['is_str'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        f_ext = (df_season.groupby('plot_x')['is_ext'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        
        fig_stats = go.Figure()
        if parameter == "TX":
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
        stats.loc[yr, 'total_heat'] = sum(t - p_thresh for t in y_df['Temp'] if t >= p_thresh) if parameter == "TX" else sum(p_thresh - t for t in y_df['Temp'] if t <= p_thresh)

    col_map = {"Cumulative Annual Wave Intensity": 'sum_int', "Maximum Annual Wave Intensity": 'max_int', "Cumulative Heat/Cold Intensity": 'total_heat'}
    sel_col = col_map.get(stat_metric, 'sum_int')
    y_titles = {
        'sum_int': 'Σ wave intensity (K·days)',
        'max_int': 'Max wave intensity (K·days)',
        'total_heat': f'Σ excess vs. threshold (K·days)',
    }

    fig_stats = go.Figure()
    if parameter == "TX":
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