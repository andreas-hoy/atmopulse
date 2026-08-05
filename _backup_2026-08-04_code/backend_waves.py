import xarray as xr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.interpolate import make_interp_spline

def get_kiesely_waves_figs(lat, lon, parameter="TX", selected_epoch="B", threshold_level="Strong (P90/10)", stat_metric="Cumulative Annual Wave Intensity"):
    DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
    CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
    if not CLIM_FILE.exists(): CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")
    
    empty_fig = go.Figure().add_annotation(text="Data Missing or Processing.", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="red"))
    files = sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))
    if not files or not CLIM_FILE.exists(): return empty_fig, empty_fig

    with xr.open_dataset(CLIM_FILE) as ds_clim:
        pt_clim = ds_clim.sel(latitude=lat, longitude=lon, method='nearest').load()
    
    suffix = "A" if selected_epoch == "A" else "B"
    
    def get_const(name_const):
        if name_const in pt_clim.variables:
            return float(np.nanmean(pt_clim[name_const].values))
        return np.nan

    if parameter == "TX":
        p_t_ext, p_d_ext, p_t_str, p_d_str = get_const(f'tx_p95_{suffix}'), get_const(f'tx_p90_{suffix}'), get_const(f'tx_p90_{suffix}'), get_const(f'tx_p75_{suffix}')
    else:
        p_t_ext, p_d_ext, p_t_str, p_d_str = get_const(f'tn_p5_{suffix}'), get_const(f'tn_p10_{suffix}'), get_const(f'tn_p10_{suffix}'), get_const(f'tn_p25_{suffix}')

    p_thresh, p_drop = (p_t_ext, p_d_ext) if "Extreme" in threshold_level else (p_t_str, p_d_str)
    if np.isnan(p_thresh) or np.isnan(p_drop): return empty_fig, empty_fig

    var_key = 'mx2t' if parameter == "TX" else 'mn2t'
    all_temps, all_times = [], []
    for f in files:
        with xr.open_dataset(f, engine='netcdf4') as ds:
            pt = ds.sel(latitude=lat, longitude=lon, method='nearest').compute()
            all_temps.append(pt[var_key].values - 273.15)
            all_times.append(pt.valid_time.values)
            
    df_raw = pd.DataFrame({'Temp': np.concatenate(all_temps)}, index=pd.to_datetime(np.concatenate(all_times)))
    df = df_raw[~df_raw.index.duplicated(keep='first')].sort_index().copy()
    
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

    waves_data = []

    # SOMMER-BUG FIX: Natives Pandas Reindexing erzwingt saubere Datums- und Tageszählung
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
            if np.isnan(temps[i:i+3]).any(): i += 1; continue
                
            if all((temps[i+k] >= p_thresh) if parameter == "TX" else (temps[i+k] <= p_thresh) for k in range(3)):
                cand_temps, cand_xs, cand_dates = [], [], []
                j = i
                while j < n and not np.isnan(temps[j]):
                    cand_temps.append(temps[j]); cand_xs.append(xs[j]); cand_dates.append(dates[j])
                    if ((temps[j] < p_drop) if parameter == "TX" else (temps[j] > p_drop)) or ((np.mean(cand_temps) < p_thresh) if parameter == "TX" else (np.mean(cand_temps) > p_thresh)): 
                        break
                    j += 1
                
                intensity = sum(abs(t - p_thresh) for t in cand_temps if ((t >= p_thresh) if parameter == "TX" else (t <= p_thresh)))
                if intensity > 0 and len(cand_temps) >= 3: 
                    waves_data.append({'year': yr, 'xs': cand_xs, 'temps': cand_temps, 'intensity': intensity, 'start_date': cand_dates[0], 'end_date': cand_dates[-1]})
                i = j if j > i else i + 1
            else: i += 1

    fig_main = go.Figure()
    start_year, end_year = 1940, 2026
    y_ticks_vals = list(range(start_year, end_year + 1))
    y_ticks_text = [str(y) if parameter == "TX" else f"{y-1}/{str(y)[2:]}" for y in y_ticks_vals]
    
    t_suff = "Heatwaves" if parameter == "TX" else "Coldwaves"
    lvl_text = "Extreme Level (P95/5)" if "Extreme" in threshold_level else "Strong Level (P90/10)"
    
    fig_main.update_layout(
        title=dict(text=f"Duration and Intensity of Local {t_suff} (1940–2026) | {lvl_text}<br><span style='font-size:11px;color:gray;'>Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}</span>", font=dict(size=13)),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[start_plot_x, end_plot_x], showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=y_ticks_vals[::5], ticktext=y_ticks_text[::5], range=[2026.5, start_year - 10.0], showgrid=False, zeroline=False, showline=True, linecolor="rgba(100,100,100,0.3)"),
        height=750, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=55, r=20, t=50, b=40)
    )
    for gl in grid_lines: fig_main.add_vline(x=gl, line_width=1.2, line_color="rgba(30,30,30,0.6)", layer="below")
    for yr in range(start_year, end_year + 1, 5): fig_main.add_hline(y=yr, line_width=0.7, line_color="rgba(100,100,100,0.4)", layer="below")
        
    if waves_data:
        for w in waves_data:
            y_base, w_xs, w_ts = w['year'], np.array(w['xs']), np.array(w['temps'])
            cum_sum = np.cumsum(np.maximum(0, w_ts - p_thresh) if parameter == "TX" else np.maximum(0, p_thresh - w_ts))
            
            w_df = pd.DataFrame({'x': w_xs, 'y': cum_sum}).drop_duplicates(subset=['x']).sort_values('x')
            w_xs, cum_sum = w_df['x'].values, w_df['y'].values
            
            if len(w_xs) >= 3:
                x_fine = np.linspace(w_xs[0], w_xs[-1], 100)
                y_fine = np.clip(make_interp_spline(w_xs, cum_sum, k=2)(x_fine), 0, None)
                # ASYMMETRIE FAKTOR = 2.5
                x_skewed = x_fine + (y_fine / (max(y_fine) if max(y_fine)>0 else 1)) * 2.5 
            else: x_skewed, y_fine = w_xs, cum_sum
            
            break_x = np.linspace(x_skewed[-1], x_skewed[-1] + 1.2, 30)
            y_break = y_fine[-1] * np.exp(-3.0 * (break_x - x_skewed[-1]))
            
            x_full, y_full = np.concatenate(([x_skewed[0]], x_skewed, break_x, [break_x[-1]])), np.concatenate(([0.0], y_fine, y_break, [0.0]))
            y_coords = y_base - (y_full / 20.0)
            
            # Farbe basiert auf Intensität (TX cap 100, TN cap 200)
            norm_val = min(w['intensity'] / 100.0, 1.0) if parameter == "TX" else min(w['intensity'] / 200.0, 1.0)
            r, g, b = (int(255 - 127*norm_val), int(165 - 165*norm_val), int(150*norm_val)) if parameter == "TX" else (int(75*norm_val), int(191 - 191*norm_val), int(255 - 125*norm_val))
            r_b, g_b, b_b = (255, 230, 153) if parameter == "TX" else (204, 238, 255) 
            
            sd_str, ed_str = pd.to_datetime(w['start_date']).strftime('%d.%m.'), pd.to_datetime(w['end_date']).strftime('%d.%m.%Y')
            fig_main.add_trace(go.Scatter(x=x_full, y=y_coords, mode='lines', line=dict(color=f"rgba({r},{g},{b}, 0.9)", width=1.5, shape='spline'), fill='toself', fillgradient=dict(type='vertical', colorscale=[[0, f"rgba({r_b},{g_b},{b_b},0.6)"], [1, f"rgba({r},{g},{b},0.85)"]]), hoverinfo='text', text=f"<b>Duration: {sd_str}–{ed_str}</b><br>Length: {len(w_xs)} days<br>Severity: {w['intensity']:.1f} K", showlegend=False))
    else: fig_main.add_annotation(text="No wave events detected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="gray"))

    if stat_metric == "Annual Cycle Frequency":
        df_season['is_str'] = (df_season['Temp'] >= p_t_str) if parameter == "TX" else (df_season['Temp'] <= p_t_str)
        df_season['is_ext'] = (df_season['Temp'] >= p_t_ext) if parameter == "TX" else (df_season['Temp'] <= p_t_ext)
        f_str = (df_season.groupby('plot_x')['is_str'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        f_ext = (df_season.groupby('plot_x')['is_ext'].mean() * 100).rolling(5, center=True, min_periods=1).mean()
        
        fig_stats = go.Figure()
        c_str, c_ext = ('darkorange', 'firebrick') if parameter == "TX" else ('deepskyblue', 'darkblue')
        fig_stats.add_trace(go.Scatter(x=f_str.index, y=f_str.values, mode='lines', line=dict(color=c_str, width=2), name="Strong", hovertemplate='%{y:.1f}%<extra></extra>'))
        fig_stats.add_trace(go.Scatter(x=f_ext.index, y=f_ext.values, mode='lines', line=dict(color=c_ext, width=2), name="Extreme", hovertemplate='%{y:.1f}%<extra></extra>'))
        fig_stats.update_layout(title=f"Annual Cycle Frequency (5-Day Smoothing) | Reference {'1961–1990' if suffix=='A' else '1996–2025'}", xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, showgrid=True), yaxis_title="Relative Frequency (%)", height=350, template="plotly_white", margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
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
    
    fig_stats = go.Figure()
    mc, lc = ('rgba(210, 100, 100, 0.7)', 'maroon') if parameter == "TX" else ('rgba(100, 150, 210, 0.7)', 'navy')
    
    fig_stats.add_trace(go.Bar(x=stats.index, y=stats[sel_col], marker_color=mc, name="Value", hovertemplate='Value: %{y:.1f} K<extra></extra>'))
    fig_stats.add_trace(go.Scatter(x=stats.index, y=stats[sel_col].rolling(11, center=True).mean(), mode='lines', line=dict(color=lc, width=2), name="11-yr Mean", hovertemplate='11-yr Mean: %{y:.1f} K<extra></extra>'))
    
    valid = stats[sel_col].dropna()
    if len(valid) > 2:
        z = np.polyfit(valid.index, valid.values, 1)
        fig_stats.add_trace(go.Scatter(x=valid.index, y=np.poly1d(z)(valid.index), mode='lines', line=dict(color=lc, width=1.5, dash='dot'), name="Trend", hoverinfo='skip'))

    fig_stats.update_layout(title=f"{stat_metric} | Reference Period {'1961–1990' if suffix=='A' else '1996–2025'}", height=350, template="plotly_white", margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))

    return fig_main, fig_stats