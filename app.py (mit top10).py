import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import threading
from pathlib import Path
from geopy.geocoders import Nominatim

from backend_maps import get_synoptic_map_data
from backend_time_series import get_live_timeseries
from backend_waves import get_kiesely_waves_figs

# --- UI & CSS: TOP NAVIGATION BAR ---
st.set_page_config(page_title="SynEx 🌊", layout="wide", page_icon="🌊", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* Sleek Top Navigation Pill-Bar */
.nav-container {
    background-color: #e0f2ff;
    padding: 10px 20px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
.nav-logo {
    font-size: 22px;
    font-weight: normal;
    color: #0056b3;
    margin-right: 40px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex; flex-direction: row; gap: 10px; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background-color: transparent; padding: 8px 18px; border-radius: 8px;
    font-size: 16px; font-weight: 500; color: #0056b3; cursor: pointer; transition: all 0.2s;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover { background-color: #cce5ff; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] { background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p { color: #0056b3 !important; font-weight: bold; }
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child { display: none; }

/* Shrink Sidebar Spacing */
section[data-testid="stSidebar"] .stRadio > div { gap: 0rem; }
section[data-testid="stSidebar"] .stCheckbox { margin-top: -8px; }
</style>
""", unsafe_allow_html=True)

geolocator = Nominatim(user_agent="extreme_climate_tool_2026")

if "nc_lock" not in st.session_state: st.session_state.nc_lock = threading.Lock()
if "search_history" not in st.session_state: st.session_state.search_history = ["Berlin", "Tallinn", "Budapest"]
if "toggles_warm" not in st.session_state: st.session_state.toggles_warm = {"p75": True, "p90": True, "p95": True, "rec": True}
if "toggles_cold" not in st.session_state: st.session_state.toggles_cold = {"p25": True, "p10": True, "p5": True, "rec": True}
if "offset_slider" not in st.session_state: st.session_state.offset_slider = 0

def add_day():
    if st.session_state.offset_slider < 6: st.session_state.offset_slider += 1
def sub_day():
    if st.session_state.offset_slider > -6: st.session_state.offset_slider -= 1
def toggle_warm_state():
    current = any(st.session_state.toggles_warm.values())
    for k in st.session_state.toggles_warm: st.session_state.toggles_warm[k] = not current
def toggle_cold_state():
    current = any(st.session_state.toggles_cold.values())
    for k in st.session_state.toggles_cold: st.session_state.toggles_cold[k] = not current

@st.cache_data(show_spinner=False)
def load_reference_climatology():
    clim_path = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")
    return xr.open_dataset(clim_path) if clim_path.exists() else None
ref_clim = load_reference_climatology()

@st.cache_resource(show_spinner=False)
def get_master_files():
    DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
    return sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))

@st.cache_resource(show_spinner=False)
def get_europe_borders_trace():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
    try:
        data = requests.get(url, timeout=10).json()
        x, y = [], []
        for feature in data['features']:
            geom = feature.get('geometry')
            if not geom: continue
            if geom['type'] == 'Polygon':
                for poly in geom['coordinates']:
                    for p in poly: x.append(p[0]); y.append(p[1])
                    x.append(None); y.append(None)
            elif geom['type'] == 'MultiPolygon':
                for multi in geom['coordinates']:
                    for poly in multi:
                        for p in poly: x.append(p[0]); y.append(p[1])
                        x.append(None); y.append(None)
        return go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=1.0), hoverinfo='skip', showlegend=False)
    except: return None
border_trace = get_europe_borders_trace()

@st.cache_data(show_spinner=False)
def fetch_cached_synoptic_data(date_str):
    with st.session_state.nc_lock: return get_synoptic_map_data(date_str)

@st.cache_data(show_spinner=False)
def get_persistence_arrays(target_date_str, baseline_type, map_var="TG"):
    if ref_clim is None: return None
    files = get_master_files()
    end_date = pd.to_datetime(target_date_str)
    start_date = end_date - pd.Timedelta(days=59)
    with st.session_state.nc_lock:
        try:
            with xr.open_mfdataset(files, combine='nested', concat_dim='valid_time', engine='netcdf4') as ds:
                ds = ds.sel(valid_time=slice(start_date, end_date)).compute()
        except: return None

    tx_hist, tn_hist = ds['mx2t'].values - 273.15, ds['mn2t'].values - 273.15
    doys = pd.to_datetime(ds.valid_time.values).dayofyear
    suffix = "A" if baseline_type == "A" else "B"
    n_days, n_lats, n_lons = tx_hist.shape
    
    def safe_get(var_key, fallback=None):
        if var_key in ref_clim: return ref_clim[var_key].values
        elif fallback is not None: return fallback
        else: return np.full((366, n_lats, n_lons), np.nan)

    if map_var == "TX":
        v_h, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5 = tx_hist, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}'), safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tx_max_val'), safe_get('tx_min_val')
    elif map_var == "TN":
        v_h, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5 = tn_hist, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}'), safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tn_max_val'), safe_get('tn_min_val')
    else: 
        v_h = (tx_hist + tn_hist) / 2.0
        v_p95 = safe_get(f'tg_p95_doy_{suffix}', (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}'))/2)
        v_p90 = safe_get(f'tg_p90_doy_{suffix}', (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}'))/2)
        v_p75 = safe_get(f'tg_p75_doy_{suffix}', (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}'))/2)
        v_p25 = safe_get(f'tg_p25_doy_{suffix}', (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}'))/2)
        v_p10 = safe_get(f'tg_p10_doy_{suffix}', (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}'))/2)
        v_p5  = safe_get(f'tg_p5_doy_{suffix}',  (safe_get(f'tx_p5_doy_{suffix}')  + safe_get(f'tn_p5_doy_{suffix}'))/2)
        v_r_w, v_r_c = safe_get('tg_max_val', (safe_get('tx_max_val') + safe_get('tn_max_val'))/2), safe_get('tg_min_val', (safe_get('tx_min_val') + safe_get('tn_min_val'))/2)

    streaks = np.zeros((8, n_lats, n_lons), dtype=int)
    exc = np.zeros((8, n_days, n_lats, n_lons), dtype=bool)
    
    for i, d in enumerate(doys):
        d_idx = d - 1
        exc[0, i], exc[1, i], exc[2, i], exc[3, i] = v_h[i] >= v_p75[d_idx], v_h[i] >= v_p90[d_idx], v_h[i] >= v_p95[d_idx], v_h[i] >= v_r_w[d_idx]
        exc[4, i], exc[5, i], exc[6, i], exc[7, i] = v_h[i] <= v_p25[d_idx], v_h[i] <= v_p10[d_idx], v_h[i] <= v_p5[d_idx], v_h[i] <= v_r_c[d_idx]
        
    for lvl in range(8): streaks[lvl] = np.sum(np.cumprod(exc[lvl][::-1, :, :], axis=0), axis=0)
    return streaks

def build_baseline_map(ref_data, map_phys_data, target_date, t_warm, t_cold, toggles, view_mode, persist_metric, top10_threshold, baseline_type="A", map_var="TG"):
    if ref_data is None or map_phys_data is None: return go.Figure()
    suffix, doy = ("A" if baseline_type == "A" else "B"), target_date.dayofyear
    tx_curr, tn_curr = map_phys_data["tx"].values, map_phys_data["tn"].values
    lons, lats = map_phys_data['mslp'].longitude.values, map_phys_data['mslp'].latitude.values
    daily_ref = ref_data.sel(dayofyear=doy)
    
    def safe_get(var_key, fallback=None):
        return daily_ref[var_key].values if var_key in daily_ref else (fallback if fallback is not None else np.full(tx_curr.shape, np.nan))

    if map_var == "TX":
        v_curr, v_p95, v_p90, v_p75 = tx_curr, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        v_rec_w, v_rec_c, yr_w, yr_c = safe_get('tx_max_val'), safe_get('tx_min_val'), safe_get('tx_max_year'), safe_get('tx_min_year')
    elif map_var == "TN":
        v_curr, v_p95, v_p90, v_p75 = tn_curr, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        v_rec_w, v_rec_c, yr_w, yr_c = safe_get('tn_max_val'), safe_get('tn_min_val'), safe_get('tn_max_year'), safe_get('tn_min_year')
    else:
        v_curr = (tx_curr + tn_curr) / 2.0
        v_p95 = safe_get(f'tg_p95_doy_{suffix}', (safe_get(f'tx_p95_doy_{suffix}') + safe_get(f'tn_p95_doy_{suffix}')) / 2.0)
        v_p90 = safe_get(f'tg_p90_doy_{suffix}', (safe_get(f'tx_p90_doy_{suffix}') + safe_get(f'tn_p90_doy_{suffix}')) / 2.0)
        v_p75 = safe_get(f'tg_p75_doy_{suffix}', (safe_get(f'tx_p75_doy_{suffix}') + safe_get(f'tn_p75_doy_{suffix}')) / 2.0)
        v_p25 = safe_get(f'tg_p25_doy_{suffix}', (safe_get(f'tx_p25_doy_{suffix}') + safe_get(f'tn_p25_doy_{suffix}')) / 2.0)
        v_p10 = safe_get(f'tg_p10_doy_{suffix}', (safe_get(f'tx_p10_doy_{suffix}') + safe_get(f'tn_p10_doy_{suffix}')) / 2.0)
        v_p5  = safe_get(f'tg_p5_doy_{suffix}',  (safe_get(f'tx_p5_doy_{suffix}')  + safe_get(f'tn_p5_doy_{suffix}'))  / 2.0)
        v_rec_w, v_rec_c = safe_get('tg_max_val', (safe_get('tx_max_val')+safe_get('tn_max_val'))/2), safe_get('tg_min_val', (safe_get('tx_min_val')+safe_get('tn_min_val'))/2)
        yr_w, yr_c = safe_get('tg_max_year', safe_get('tx_max_year')), safe_get('tg_min_year', safe_get('tn_min_year'))

    fig = go.Figure()
    
    # NACHKOMMA FIX: REINE STRINGS ERZWINGEN
    diff_w_str = np.array([f"{x:+.1f}" if not np.isnan(x) else "N/A" for x in (v_curr - v_rec_w).flat]).reshape(v_curr.shape)
    diff_c_str = np.array([f"{x:+.1f}" if not np.isnan(x) else "N/A" for x in (v_curr - v_rec_c).flat]).reshape(v_curr.shape)
    
    c_data = np.empty((*v_curr.shape, 10), dtype=object)
    c_data[..., 0] = np.round(v_curr, 1)
    c_data[..., 1], c_data[..., 2] = np.round(v_p90, 1), np.round(v_p10, 1)
    c_data[..., 3], c_data[..., 4] = np.round(v_rec_w, 1), np.round(v_rec_c, 1)
    c_data[..., 5], c_data[..., 6] = yr_w, yr_c
    c_data[..., 7], c_data[..., 8] = diff_w_str, diff_c_str

    if "Standard" in view_mode:
        mask = np.full(v_curr.shape, np.nan)
        if t_cold["p25"]: mask = np.where(v_curr <= v_p25, 4, mask)
        if t_warm["p75"]: mask = np.where(v_curr >= v_p75, 5, mask)
        if t_cold["p10"]: mask = np.where(v_curr <= v_p10, 3, mask)
        if t_warm["p90"]: mask = np.where(v_curr >= v_p90, 6, mask)
        if t_cold["p5"]:  mask = np.where(v_curr <= v_p5, 2, mask)
        if t_warm["p95"]: mask = np.where(v_curr >= v_p95, 7, mask)
        if t_cold["rec"]: mask = np.where(v_curr <= v_rec_c, 1, mask)
        if t_warm["rec"]: mask = np.where(v_curr >= v_rec_w, 8, mask)

        colorscale = [[0.0, "#4b0082"], [0.125, "#4b0082"], [0.125, "#0000ff"], [0.25, "#0000ff"], [0.25, "#3399ff"], [0.375, "#3399ff"], [0.375, "#ccf2ff"], [0.5, "#ccf2ff"], [0.5, "#ffe699"], [0.625, "#ffe699"], [0.625, "#ff9933"], [0.75, "#ff9933"],  [0.75, "#cc0000"], [0.875, "#cc0000"], [0.875, "#ff1493"], [1.0, "#ff1493"]]
        
        fig.add_trace(go.Heatmap(
            x=lons, y=lats, z=mask, customdata=c_data, colorscale=colorscale, showscale=False, opacity=0.85, zmin=1, zmax=8, zsmooth='best',
            hovertemplate=(
                "<b>Lat: %{y:.1f}, Lon: %{x:.1f}</b><br><br>"
                f"{map_var}: <b>%{{customdata[0]:.1f}} °C</b> (P90: %{{customdata[1]:.1f}}, P10: %{{customdata[2]:.1f}})<br>"
                "All-Time Warm: %{customdata[3]:.1f} °C in Year %{customdata[5]:.0f} <i>(%{customdata[7]} K diff)</i><br>"
                "All-Time Cold: %{customdata[4]:.1f} °C in Year %{customdata[6]:.0f} <i>(%{customdata[8]} K diff)</i><extra></extra>"
            )
        ))
        
        if toggles.get("hatching", False):
            streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var)
            if streaks is not None:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                if "All-Time" in top10_threshold: h_idx, c_idx = 3, 7
                elif "Extreme" in top10_threshold: h_idx, c_idx = 2, 6
                elif "Strong" in top10_threshold: h_idx, c_idx = 1, 5 
                else: h_idx, c_idx = 0, 4
                
                hatch_mask = (streaks[h_idx] >= 6) | (streaks[c_idx] >= 6)
                if np.any(hatch_mask):
                    h_lons, h_lats = lon_grid[hatch_mask][::2], lat_grid[hatch_mask][::2]
                    fig.add_trace(go.Scatter(x=h_lons, y=h_lats, mode='markers', marker=dict(symbol='x', color='rgba(0,0,0,0.15)', size=3), hoverinfo='skip', showlegend=False))
                    
    else:
        streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var)
        if streaks is not None:
            mapping = {"Warm Moderate": (0, 60, "Reds"), "Warm Strong": (1, 30, "Reds"), "Warm Extreme": (2, 20, "Reds"), "Warm All-Time Record": (3, 15, "Reds"), "Cold Moderate": (4, 60, "Blues"), "Cold Strong": (5, 30, "Blues"), "Cold Extreme": (6, 20, "Blues"), "Cold All-Time Record": (7, 15, "Blues")}
            lvl_idx, max_days, cmap = mapping.get(persist_metric, (1, 30, "Reds"))
            p_mask = np.where(streaks[lvl_idx] == 0, np.nan, streaks[lvl_idx])
                
            fig.add_trace(go.Heatmap(x=lons, y=lats, z=p_mask, zmin=1, zmax=max_days, colorscale=cmap, showscale=True, opacity=0.9, zsmooth='best', colorbar=dict(title="Days", len=0.6, y=0.5, thickness=15), hovertemplate="<b>Lat: %{y:.1f}, Lon: %{x:.1f}</b><br>Persistence: <b>%{z} Days</b><extra></extra>"))
            
    if border_trace is not None: fig.add_trace(border_trace)
    
    if toggles.get("mslp", False): fig.add_trace(go.Contour(x=lons, y=lats, z=map_phys_data['mslp'].values, colorscale=[[0, '#006400'], [1, '#006400']], contours=dict(start=980, end=1040, size=5, showlabels=True), contours_coloring='lines', showscale=False, line_width=2.5, opacity=0.8, hoverinfo="skip"))
    if toggles.get("z500", False): fig.add_trace(go.Contour(x=lons, y=lats, z=map_phys_data['z500'].squeeze().values, colorscale=[[0, 'blue'], [1, 'blue']], contours=dict(start=500, end=600, size=8, showlabels=True), contours_coloring='lines', showscale=False, line_width=2.5, opacity=0.8, hoverinfo="skip"))

    # DATENQUELLEN BOX IN DER KARTE VERANKERT
    fig.add_annotation(text="Data: ECMWF ERA5 (Archive) & IFS (Forecast) | Copernicus C3S", xref="paper", yref="paper", x=0.99, y=0.03, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=11, color="black"), bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1, borderpad=4)

    fig.update_layout(
        uirevision='map_sync_state', 
        xaxis=dict(range=[lons.min(), lons.max()], showgrid=False, zeroline=False, visible=False, constrain="domain"), 
        yaxis=dict(range=[lats.min(), lats.max()], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, visible=False, constrain="domain"), 
        margin={"r":0,"t":10,"l":0,"b":0}, height=850, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- TOP 10 LISTE ---
@st.cache_data(show_spinner=False)
def build_top10_table(df_live, is_warm):
    col_target = 'TX' if is_warm else 'TN'
    if col_target not in df_live.columns: return pd.DataFrame()
    df_sorted = df_live[['Date', col_target]].dropna().sort_values(by=col_target, ascending=not is_warm)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date']).dt.strftime('%Y-%m-%d')
    df_sorted.rename(columns={col_target: f"{col_target} (°C)"}, inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1
    return df_sorted.head(10)

# --- MASTERPIECE METEOGRAMM (Climate Stripes) ---
def build_meteogram(df_live, ref_clim, lat, lon, target_date, epoch, show_air, show_app, env_level_up, env_level_dn, is_warm):
    fig = go.Figure()
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)
    doys = (df_live['Date'].dt.dayofyear.values - 1) % 366
    dates = df_live['Date']
    
    p_up_key = f'tx_{env_level_up}_doy_{epoch}' if env_level_up != "max_val" else 'tx_max_val'
    p_dn_key = f'tn_{env_level_dn}_doy_{epoch}' if env_level_dn != "min_val" else 'tn_min_val'
    
    env_upper = pt_clim[p_up_key].values[doys]
    env_lower = pt_clim[p_dn_key].values[doys]
    
    fig.add_trace(go.Scatter(x=dates, y=env_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=env_lower, mode='lines', fill='tonexty', fillcolor='rgba(200,200,200,0.4)', line=dict(width=0), name='Climate Envelope', hoverinfo='skip'))

    col_target = 'TX' if is_warm else 'TN'
    t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values
    d_hist = dates[dates <= tgt_dt_norm]
    
    c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
    c_hist = c_base[dates <= tgt_dt_norm]

    # Dynamisches Schicht-Rendering
    if is_warm:
        p75, p90, p95 = pt_clim[f'tx_p75_doy_{epoch}'].values[doys][dates <= tgt_dt_norm], pt_clim[f'tx_p90_doy_{epoch}'].values[doys][dates <= tgt_dt_norm], pt_clim[f'tx_p95_doy_{epoch}'].values[doys][dates <= tgt_dt_norm]
        y1 = np.where(t_hist > c_hist, np.minimum(t_hist, p75), c_hist)
        y2 = np.where(t_hist > p75, np.minimum(t_hist, p90), y1)
        y3 = np.where(t_hist > p90, np.minimum(t_hist, p95), y2)
        y4 = np.where(t_hist > p95, t_hist, y3)
        fig.add_trace(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y1, mode='lines', fill='tonexty', fillcolor='rgba(255,180,180,0.6)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y2, mode='lines', fill='tonexty', fillcolor='rgba(255,100,100,0.7)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y3, mode='lines', fill='tonexty', fillcolor='rgba(200,0,0,0.8)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y4, mode='lines', fill='tonexty', fillcolor='rgba(100,0,50,0.9)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    else:
        p25, p10, p5 = pt_clim[f'tn_p25_doy_{epoch}'].values[doys][dates <= tgt_dt_norm], pt_clim[f'tn_p10_doy_{epoch}'].values[doys][dates <= tgt_dt_norm], pt_clim[f'tn_p5_doy_{epoch}'].values[doys][dates <= tgt_dt_norm]
        y1 = np.where(t_hist < c_hist, np.maximum(t_hist, p25), c_hist)
        y2 = np.where(t_hist < p25, np.maximum(t_hist, p10), y1)
        y3 = np.where(t_hist < p10, np.maximum(t_hist, p5), y2)
        y4 = np.where(t_hist < p5, t_hist, y3)
        fig.add_trace(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y1, mode='lines', fill='tonexty', fillcolor='rgba(180,200,255,0.6)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y2, mode='lines', fill='tonexty', fillcolor='rgba(100,150,255,0.7)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y3, mode='lines', fill='tonexty', fillcolor='rgba(0,50,200,0.8)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_hist, y=y4, mode='lines', fill='tonexty', fillcolor='rgba(0,0,100,0.9)', line=dict(width=0), showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(color='black', width=2), name='Climatology Base'))

    fcst_mask = dates >= tgt_dt_norm
    if show_app:
        col_app = 'AT_Max' if is_warm else 'AT_Min'
        fig.add_trace(go.Scatter(x=d_hist, y=df_live.loc[dates <= tgt_dt_norm, col_app], mode='lines', name='Apparent Temperature', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_app], mode='lines', line=dict(color='orange', width=2, dash='dot'), showlegend=False))
    
    if show_air:
        c_main = 'red' if is_warm else 'blue'
        fig.add_trace(go.Scatter(x=d_hist, y=t_hist, mode='lines', name='Air Temperature', line=dict(color=c_main, width=2.5)))
        fig.add_trace(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_target], mode='lines', line=dict(color=c_main, width=2.5, dash='dot'), showlegend=False))
        
    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8)
    fig.add_annotation(x=tgt_dt_norm.timestamp() * 1000, y=0.98, yref="paper", text="<b>HISTORY | FORECAST</b>", showarrow=False, font=dict(size=11, color="black"), xanchor="center", yanchor="top")
    
    fig.update_traces(hovertemplate='%{y:.1f} °C')
    fig.update_layout(title=f"Reference Period {'1961–1990' if epoch=='A' else '1996–2025'}", hovermode="x unified", height=400, template="plotly_white", margin=dict(t=40, b=10), yaxis_title="Temperature (°C)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_data(show_spinner=False)
def build_yearly_extremes_chart(lat, lon, epoch, is_warm):
    files = get_master_files()
    if not files or ref_clim is None: return go.Figure()
    
    with st.session_state.nc_lock:
        try:
            with xr.open_mfdataset(files, combine='nested', concat_dim='valid_time', engine='netcdf4') as ds:
                pt_data = ds.sel(latitude=lat, longitude=lon, method='nearest').compute()
        except: return go.Figure()
        
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    val = (pt_data['mx2t'].values - 273.15) if is_warm else (pt_data['mn2t'].values - 273.15)
    
    df = pd.DataFrame({'time': pt_data.valid_time.values, 'val': val}).drop_duplicates(subset=['time'])
    dates = pd.to_datetime(df['time'])
    df['year'], doys = dates.dt.year, dates.dt.dayofyear - 1
    v = df['val'].values
    
    if is_warm:
        c_75, c_90, c_95, c_rec = pt_clim[f'tx_p75_doy_{epoch}'].values[doys], pt_clim[f'tx_p90_doy_{epoch}'].values[doys], pt_clim[f'tx_p95_doy_{epoch}'].values[doys], pt_clim['tx_max_val'].values[doys]
        df['p75'], df['p90'], df['p95'], df['rec'] = (v >= c_75) & (v < c_90), (v >= c_90) & (v < c_95), (v >= c_95) & (v < c_rec), v >= c_rec
    else:
        c_25, c_10, c_5, c_rec = pt_clim[f'tn_p25_doy_{epoch}'].values[doys], pt_clim[f'tn_p10_doy_{epoch}'].values[doys], pt_clim[f'tn_p5_doy_{epoch}'].values[doys], pt_clim['tn_min_val'].values[doys]
        df['p25'], df['p10'], df['p5'], df['rec'] = (v <= c_25) & (v > c_10), (v <= c_10) & (v > c_5), (v <= c_5) & (v > c_rec), v <= c_rec

    res = df.groupby('year').sum()
    fig = go.Figure()
    if is_warm:
        fig.add_trace(go.Bar(x=res.index, y=res['p75'], name='Moderate', marker_color='#ffe699'))
        fig.add_trace(go.Bar(x=res.index, y=res['p90'], name='Strong', marker_color='#ff9933'))
        fig.add_trace(go.Bar(x=res.index, y=res['p95'], name='Extreme', marker_color='#cc0000'))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color='#ff1493'))
    else:
        fig.add_trace(go.Bar(x=res.index, y=res['p25'], name='Moderate', marker_color='#ccf2ff'))
        fig.add_trace(go.Bar(x=res.index, y=res['p10'], name='Strong', marker_color='#3399ff'))
        fig.add_trace(go.Bar(x=res.index, y=res['p5'],  name='Extreme', marker_color='#0000ff'))
        fig.add_trace(go.Bar(x=res.index, y=res['rec'], name='Records', marker_color='#4b0082'))

    fig.update_layout(barmode='stack', title=f"Days exceeding thresholds per year | Reference Period {'1961–1990' if epoch=='A' else '1996–2025'}", height=300, margin=dict(t=30, b=10), template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_data(show_spinner=False)
def calculate_top10(_ref_data, _map_phys_data, target_date, t_warm, t_cold, view_mode, persist_metric, top10_threshold, baseline_type, map_var="TG"):
    if _ref_data is None or _map_phys_data is None: return pd.DataFrame(), pd.DataFrame()
    suffix, doy = ("A" if baseline_type == "A" else "B"), target_date.dayofyear
    lons, lats = _map_phys_data['mslp'].longitude.values, _map_phys_data['mslp'].latitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    heat_mask, cold_mask = np.zeros(lon_grid.shape, dtype=bool), np.zeros(lon_grid.shape, dtype=bool)

    if "Standard" in view_mode:
        tx, tn = _map_phys_data["tx"].values, _map_phys_data["tn"].values
        dr = _ref_data.sel(dayofyear=doy)
        def safe_get(var_key, fallback=None): return dr[var_key].values if var_key in dr else fallback
            
        if map_var == "TX":
            v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5 = tx, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}'), safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        elif map_var == "TN":
            v_curr, v_p95, v_p90, v_p75, v_p25, v_p10, v_p5 = tn, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}'), safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        else:
            v_curr = (tx + tn) / 2.0
            v_p95, v_p90, v_p75 = safe_get(f'tg_p95_doy_{suffix}', (safe_get(f'tx_p95_doy_{suffix}')+safe_get(f'tn_p95_doy_{suffix}'))/2), safe_get(f'tg_p90_doy_{suffix}', (safe_get(f'tx_p90_doy_{suffix}')+safe_get(f'tn_p90_doy_{suffix}'))/2), safe_get(f'tg_p75_doy_{suffix}', (safe_get(f'tx_p75_doy_{suffix}')+safe_get(f'tn_p75_doy_{suffix}'))/2)
            v_p25, v_p10, v_p5 = safe_get(f'tg_p25_doy_{suffix}', (safe_get(f'tx_p25_doy_{suffix}')+safe_get(f'tn_p25_doy_{suffix}'))/2), safe_get(f'tg_p10_doy_{suffix}', (safe_get(f'tx_p10_doy_{suffix}')+safe_get(f'tn_p10_doy_{suffix}'))/2), safe_get(f'tg_p5_doy_{suffix}', (safe_get(f'tx_p5_doy_{suffix}')+safe_get(f'tn_p5_doy_{suffix}'))/2)

        if top10_threshold == "Extreme":
            if t_warm["p95"]: heat_mask |= (v_curr >= v_p95)
            if t_cold["p5"]: cold_mask |= (v_curr <= v_p5)
        elif top10_threshold == "Strong": 
            if t_warm["p90"]: heat_mask |= (v_curr >= v_p90)
            if t_cold["p10"]: cold_mask |= (v_curr <= v_p10)
        elif top10_threshold == "Moderate":
            if t_warm["p75"]: heat_mask |= (v_curr >= v_p75)
            if t_cold["p25"]: cold_mask |= (v_curr <= v_p25)
    else:
        streaks = get_persistence_arrays(target_date.strftime('%Y-%m-%d'), baseline_type, map_var)
        if streaks is not None:
            if "All-Time" in top10_threshold: h_idx, c_idx = 3, 7
            elif "Extreme" in top10_threshold: h_idx, c_idx = 2, 6
            elif "Strong" in top10_threshold: h_idx, c_idx = 1, 5
            else: h_idx, c_idx = 0, 4
            heat_mask, cold_mask = streaks[h_idx] >= 6, streaks[c_idx] >= 6

    bounds = {"Spain": (36,44,-9,3), "Italy": (37,47,7,18), "Greece": (35,41,19,28), "France": (42,51,-5,8), "Germany": (47,55,6,15), "Poland": (49,54,14,24), "Romania": (43,48,20,29), "Hungary": (45,48,16,22), "Austria": (46,49,9,17), "UK": (50,59,-8,2), "Sweden": (57,60,21,29), "Finland": (60,70,20,30)}
    res_h, res_c = [], []
    for c, (lat_min, lat_max, lon_min, lon_max) in bounds.items():
        s_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
        tot = np.sum(s_mask)
        if tot > 0:
            fh, fc = (np.sum(heat_mask & s_mask)/tot)*100, (np.sum(cold_mask & s_mask)/tot)*100
            if fh > 0: res_h.append({"Country": c, "Warm Impact (%)": fh})
            if fc > 0: res_c.append({"Country": c, "Cold Impact (%)": fc})
                
    return pd.DataFrame(res_h).sort_values(by="Warm Impact (%)", ascending=False).head(10) if res_h else pd.DataFrame(), pd.DataFrame(res_c).sort_values(by="Cold Impact (%)", ascending=False).head(10) if res_c else pd.DataFrame()

def render_dynamic_legend(threshold_level):
    s_p75, s_p90, s_p95, s_rec_h = "background-color:#ffe699; color:black;", "background-color:#ff9933; color:black;", "background-color:#cc0000; color:white;", "background-color:#ff1493; color:white;"
    s_p25, s_p10, s_p5, s_rec_c = "background-color:#ccf2ff; color:black;", "background-color:#3399ff; color:black;", "background-color:#0000ff; color:white;", "background-color:#4b0082; color:white;"
    pop = "font-weight: bold; border: 2px solid black; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
    if "Extreme" in threshold_level: s_p95 += pop;  s_rec_h += pop; s_p5 += pop;  s_rec_c += pop
    elif "Strong" in threshold_level: s_p90 += pop;  s_p10 += pop
    elif "Moderate" in threshold_level: s_p75 += pop;  s_p25 += pop
    else: s_rec_h += pop; s_rec_c += pop

    leg_html = f"""
    <div style='margin-bottom: 15px;'>
        <b>Map Legends (Dynamic Focus: <span style='color: #4b0082;'>{threshold_level.split(' ')[0]}</span>)</b><br>
        Warm: <span style='{s_p75} padding: 2px 6px; border-radius: 3px;'>P75</span> 
              <span style='{s_p90} padding: 2px 6px; border-radius: 3px;'>P90</span> 
              <span style='{s_p95} padding: 2px 6px; border-radius: 3px;'>P95</span> 
              <span style='{s_rec_h} padding: 2px 6px; border-radius: 3px;'>All-Time Max</span><br>
        <div style='margin-top: 4px;'>
        Cold: &nbsp;<span style='{s_p25} padding: 2px 6px; border-radius: 3px;'>P25</span> 
              <span style='{s_p10} padding: 2px 6px; border-radius: 3px;'>P10</span> 
              <span style='{s_p5} padding: 2px 6px; border-radius: 3px;'>P5</span> 
              <span style='{s_rec_c} padding: 2px 6px; border-radius: 3px;'>All-Time Min</span>
        </div>
    </div>
    """
    st.markdown(leg_html, unsafe_allow_html=True)


# --- UI LAYOUT: TOP NAVIGATION BAR ---
nav_col1, nav_col2 = st.columns([1, 6])
with nav_col1:
    st.markdown("<div style='height: 50px; display: flex; align-items: center; font-size: 24px; font-weight: normal; color: #0056b3; padding-left: 10px;'>SynEx 🌊</div>", unsafe_allow_html=True)
with nav_col2:
    st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
    nav_selection = st.radio("Navigation", ["Start", "Synoptic Maps", "Location Meteograms", "Location Waves", "Imprint & Disclaimer"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# --- DYNAMISCHER JAHRESZEITEN DEFAULT ---
default_date = pd.Timestamp.now().floor('D')
target_month = default_date.month
is_warm_season = False
if 4 < target_month < 10: is_warm_season = True
elif target_month == 4 and default_date.day >= 16: is_warm_season = True
elif target_month == 10 and default_date.day <= 15: is_warm_season = True
default_wave_idx = 0 if is_warm_season else 1

# --- SIDEBAR (Nur sichtbare Kontrollen basierend auf Navigation) ---
with st.sidebar:
    if nav_selection in ["Synoptic Maps", "Location Meteograms", "Location Waves"]:
        st.header("Control Panel")
        
        st.slider("Forecast Offset (Days):", -6, 6, st.session_state.offset_slider, key="offset_slider", help="Adjusts the target date. Negative values analyze the past (ERA5 reanalysis), positive values look into the future (IFS forecast).")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1: st.button("⬅️ Prev Day", on_click=sub_day, use_container_width=True)
        with btn_col2: st.button("Next Day ➡️", on_click=add_day, use_container_width=True)
        
        target_date = default_date + pd.Timedelta(days=st.session_state.offset_slider)
        st.info(f"Target Date: **{target_date.strftime('%d.%m.%Y')}**")
        
        toggles = {}
        
        if nav_selection == "Synoptic Maps":
            st.markdown("---")
            map_var = st.radio("**Mapped Variable:**", ("Mean Temp (TG)", "Max Temp (TX)", "Min Temp (TN)"), index=0, help="TG: Daily Mean Temperature. The best proxy for the total thermal energy of the day. \n\nTX: Daily Maximum Temperature. Represents daytime warming. \n\nTN: Daily Minimum Temperature. Represents nighttime cooling.")
            map_var_code = map_var.split('(')[1].strip(')')
            
            st.markdown("---")
            view_mode = st.radio("**Map View Mode:**", ("Standard (Daily Extremes)", "Duration (Cumulative Persistence)"), help="Toggle between daily snap-shot and the persistence duration of synoptic events.")
            persist_metric = "Warm Strong"
            st.markdown("---")
            top10_threshold = st.radio("**Analysis Level**", ("Moderate", "Strong", "Extreme", "All-Time Record"), index=1, help="Drives overlays and tables. Percentiles are calculated using a centered 5-day moving window for the reference period. All-time records use the same window but consider the entire timeframe since 1940.")
            
            if "Standard" in view_mode:
                st.markdown("---")
                st.markdown("**Map Extremes**")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if st.button("Warm: OFF" if any(st.session_state.toggles_warm.values()) else "Warm: ON", use_container_width=True, help="Toggle all warm anomaly layers."): toggle_warm_state(); st.rerun()
                with m_col2:
                    if st.button("Cold: OFF" if any(st.session_state.toggles_cold.values()) else "Cold: ON", use_container_width=True, help="Toggle all cold anomaly layers."): toggle_cold_state(); st.rerun()
                st.markdown("<hr style='margin-top:5px; margin-bottom:15px; border-top: 1px dashed gray;'>", unsafe_allow_html=True)
                st.session_state.toggles_warm["p75"] = st.checkbox("Warm: Moderate", value=st.session_state.toggles_warm["p75"])
                st.session_state.toggles_warm["p90"] = st.checkbox("Warm: Strong", value=st.session_state.toggles_warm["p90"])
                st.session_state.toggles_warm["p95"] = st.checkbox("Warm: Extreme", value=st.session_state.toggles_warm["p95"])
                st.session_state.toggles_warm["rec"] = st.checkbox("Warm: All-Time Record", value=st.session_state.toggles_warm["rec"])
                st.session_state.toggles_cold["p25"] = st.checkbox("Cold: Moderate", value=st.session_state.toggles_cold["p25"])
                st.session_state.toggles_cold["p10"] = st.checkbox("Cold: Strong", value=st.session_state.toggles_cold["p10"])
                st.session_state.toggles_cold["p5"]  = st.checkbox("Cold: Extreme", value=st.session_state.toggles_cold["p5"])
                st.session_state.toggles_cold["rec"] = st.checkbox("Cold: All-Time Record", value=st.session_state.toggles_cold["rec"])
                st.markdown("---")
                toggles["hatching"] = st.checkbox("Show 6-Day WSDI/CSDI Overlay", value=True, help="Hatched areas highlight regions experiencing at least 6 consecutive days above the 90th percentile (WSDI) or below the 10th percentile (CSDI).")
            else:
                st.markdown("---")
                st.markdown("**Persistence Visualization**")
                persist_metric = st.radio("Select Variable to map duration:", ("Warm Moderate", "Warm Strong", "Warm Extreme", "Warm All-Time Record", "Cold Moderate", "Cold Strong", "Cold Extreme", "Cold All-Time Record"))
                
            st.markdown("---")
            toggles["mslp"] = st.checkbox("Show MSLP Contours", value=True, help="Mean Sea Level Pressure (hPa)")
            toggles["z500"] = st.checkbox("Show Z500 Contours", value=False, help="500 hPa Geopotential Height (gpm) – indicates upper-level ridges and troughs.")
            
        elif nav_selection in ["Location Meteograms", "Location Waves"]:
            st.markdown("---")
            st.markdown("**Location Settings**")
            wave_focus = st.radio("Wave Event Type:", ("Heatwaves", "Coldwaves"), index=default_wave_idx, help="Heatwaves: Triggered by 3+ days above the summer threshold for daily maximum temperature. Sustained as long as the average remains above the threshold. Terminated by a single day dropping below the lower tolerance limit.\n\nColdwaves: Triggered by 3+ days below the winter threshold for daily minimum temperature. Sustained as long as the average remains below the threshold. Terminated by a single day rising above the upper tolerance limit.")
            wave_thresh = st.radio("Wave Intensity Threshold:", ("Strong", "Extreme"), help="Strong: Calculates waves using the 90th (heat) or 10th (cold) percentile as the main trigger.\n\nExtreme: Calculates waves using the stricter 95th (heat) or 5th (cold) percentile as the main trigger.")
            
            if nav_selection == "Location Meteograms":
                meteo_env = st.selectbox("Background Envelope:", ["Moderate", "Strong", "Extreme", "All-Time"], index=1, help="Displays the corresponding climate boundaries (percentile-based) behind the temperature curve: Uses the 75th (warm) and 25th (cold) percentile for moderate, 90th (warm) and 10th (cold) for strong and 95th (warm) and 5th (cold) for extreme conditions within the reference period. All-time records are given for the full period (starting 1940) prior to the current year.")
                show_air_temp = st.checkbox("Show Air Temperature", value=True, help="Solid Line: Actual air temperature at 2 meters above ground.")
                show_app_temp = st.checkbox("Show Apparent Temperature", value=False, help="Dotted Line: 'Feels-like' temperature, combining 2m air temperature, relative humidity and wind speed.")
            
            if nav_selection == "Location Waves":
                st.markdown("---")
                wave_stat_metric = st.radio(
                    "Wave Statistic Metric:", 
                    ("Cumulative Annual Wave Intensity", "Maximum Annual Wave Intensity", "Cumulative Heat/Cold Intensity"),
                    help="Cumulative Annual Wave Intensity: The sum of the intensities (in Kelvin) of ALL distinct waves that occurred in a given year.\n\nMaximum Annual Wave Intensity: The intensity (in Kelvin) of the SINGLE strongest wave event of the year.\n\nCumulative Heat/Cold Intensity: The total accumulated intensity of ALL days exceeding the threshold per year, even if they don't form a consecutive 3-day wave."
                )

if ref_clim is None: st.error("Reference Climatology missing or corrupted! Please rebuild."); st.stop()


# --- TAB 1: START PAGE ---
if nav_selection == "Start":
    st.markdown("### Welcome to the Synoptic Extremes Tracker (SynEx)")
    st.markdown("""
    **SynEx** merges real-time extreme weather tracking with shifting climate baselines. It provides interactive, synoptic-scale mapping and deep-dive local profiles. Currently focused on extreme temperatures, SynEx aims to integrate further atmospheric variables in the future.
    
    <br>
    
    #### Understanding Percentiles
    SynEx relies heavily on percentiles to contextualize current weather against historical norms. In our maps and meteograms, percentiles are calculated using a **centered 5-day moving window** across the reference periods (1961–1990 and 1996–2025). 
    
    For instance, the 90th percentile (P90) is a threshold exceeded only 10% of the time during the historical baseline. We track **Moderate** (P75/P25), **Strong** (P90/P10), and **Extreme** (P95/P5) thresholds to dynamically classify the severity of synoptic events.
    
    <br>
    
    #### The Importance of Event Duration
    The impact of extreme temperatures on sectors like human health, agriculture and infrastructure scales drastically with duration. A single hot day is a weather event; a prolonged sequence becomes a systemic hazard. 
    
    In the **Synoptic Maps** tab, you can visualize this through the **Cumulative Persistence** layer, showing how many days an extreme event has lasted. By default, the maps also display an overlay for **WSDI and CSDI** conditions. The Warm/Cold Spell Duration Index ([WSDI & CSDI](https://www.climdex.org/learn/indices/)) identifies regions experiencing at least 6 consecutive days above the 90th percentile (WSDI) or below the 10th percentile (CSDI).
    
    <br>
    
    #### Local Wave Definitions
    In the **Location Waves** tab, SynEx uses a sophisticated definition (adapted from Kyselý) to track seasonally-bound heatwaves and coldwaves:
    * **Heatwaves:** Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) threshold for at least 3 consecutive days. The wave continues as long as the *average* TX remains above this threshold, and terminates immediately if a single day drops below a secondary, lower tolerance threshold.
    * **Coldwaves:** Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) threshold for at least 3 consecutive days. It continues while the *average* TN remains below this threshold, and ends if a single day rises above the upper tolerance limit.
    """, unsafe_allow_html=True)


# --- TAB 2: SYNOPTIC MAPS ---
elif nav_selection == "Synoptic Maps":
    map_layout = st.radio("Map Layout:", ("Side-by-Side Compare", "Single Map Flicker"), horizontal=True)
    if map_layout == "Single Map Flicker":
        flicker_epoch = st.radio("Select Reference Period:", ("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), horizontal=True)

    if "Standard" in view_mode:
        s_p75, s_p90, s_p95, s_rec_h = "background-color:#ffe699; color:black;", "background-color:#ff9933; color:black;", "background-color:#cc0000; color:white;", "background-color:#ff1493; color:white;"
        s_p25, s_p10, s_p5, s_rec_c = "background-color:#ccf2ff; color:black;", "background-color:#3399ff; color:black;", "background-color:#0000ff; color:white;", "background-color:#4b0082; color:white;"
        pop = "font-weight: bold; border: 2px solid black; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
        if "Extreme" in top10_threshold: s_p95 += pop;  s_rec_h += pop; s_p5 += pop;  s_rec_c += pop
        elif "Strong" in top10_threshold: s_p90 += pop;  s_p10 += pop
        elif "Moderate" in top10_threshold: s_p75 += pop;  s_p25 += pop
        else: s_rec_h += pop; s_rec_c += pop
        st.markdown(f"<div style='margin-bottom: 15px;'><b>Map Legends (Dynamic Focus: <span style='color: #4b0082;'>{top10_threshold.split(' ')[0]}</span>)</b><br>Warm: <span style='{s_p75} padding: 2px 6px; border-radius: 3px;'>P75</span> <span style='{s_p90} padding: 2px 6px; border-radius: 3px;'>P90</span> <span style='{s_p95} padding: 2px 6px; border-radius: 3px;'>P95</span> <span style='{s_rec_h} padding: 2px 6px; border-radius: 3px;'>All-Time Max</span><br><div style='margin-top: 4px;'>Cold: &nbsp;<span style='{s_p25} padding: 2px 6px; border-radius: 3px;'>P25</span> <span style='{s_p10} padding: 2px 6px; border-radius: 3px;'>P10</span> <span style='{s_p5} padding: 2px 6px; border-radius: 3px;'>P5</span> <span style='{s_rec_c} padding: 2px 6px; border-radius: 3px;'>All-Time Min</span></div></div>", unsafe_allow_html=True)
    else: st.info(f"**Persistence Mode Active:** Showing number of consecutive days with target percentiles, ending on {target_date.strftime('%d.%m.%Y')}.")

    try:
        with st.spinner("Loading synoptic fields..."): map_phys_data = fetch_cached_synoptic_data(target_date.strftime('%Y-%m-%d'))
        def render_top10_local(df, col_name):
            if not df.empty: st.dataframe(df, column_config={col_name: st.column_config.ProgressColumn(col_name, format="%.1f%%", min_value=0, max_value=100)}, hide_index=True, use_container_width=True)

        if map_layout == "Side-by-Side Compare":
            map_col1, map_col2 = st.columns(2)
            with map_col1:
                st.markdown("<h4 style='margin-bottom: -25px; margin-top: 0px;'>Reference Period A (1961–1990)</h4>", unsafe_allow_html=True)
                st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code), use_container_width=True, key="map_a")
                df_h_a, df_c_a = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "A", map_var_code)
                t_col1, t_col2 = st.columns(2)
                with t_col1: render_top10_local(df_h_a, "Warm Impact (%)")
                with t_col2: render_top10_local(df_c_a, "Cold Impact (%)")

            with map_col2:
                st.markdown("<h4 style='margin-bottom: -25px; margin-top: 0px;'>Reference Period B (1996–2025)</h4>", unsafe_allow_html=True)
                st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code), use_container_width=True, key="map_b")
                df_h_b, df_c_b = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, "B", map_var_code)
                t_col3, t_col4 = st.columns(2)
                with t_col3: render_top10_local(df_h_b, "Warm Impact (%)")
                with t_col4: render_top10_local(df_c_b, "Cold Impact (%)")
        else:
            ep_sel = "A" if "A" in flicker_epoch else "B"
            st.markdown(f"<h4 style='margin-bottom: -25px; margin-top: 0px;'>{flicker_epoch}</h4>", unsafe_allow_html=True)
            st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, ep_sel, map_var_code), use_container_width=True, key="map_flicker")
            df_h, df_c = calculate_top10(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, view_mode, persist_metric, top10_threshold, ep_sel, map_var_code)
            t_col1, t_col2 = st.columns(2)
            with t_col1: render_top10_local(df_h, "Warm Impact (%)")
            with t_col2: render_top10_local(df_c, "Cold Impact (%)")
    except Exception as e: st.error(f"Error loading maps: {e}")

@st.cache_data(show_spinner=False)
def build_top10_table(df_live, is_warm):
    col_target = 'TX' if is_warm else 'TN'
    if col_target not in df_live.columns: return pd.DataFrame()
    df_sorted = df_live[['Date', col_target]].dropna().sort_values(by=col_target, ascending=not is_warm)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date']).dt.strftime('%d.%m.%Y')
    df_sorted.rename(columns={col_target: f"{col_target} (°C)"}, inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1
    return df_sorted.head(10)

# --- TAB 3 & 4: LOCATION METEOGRAMS & WAVES ---
elif nav_selection in ["Location Meteograms", "Location Waves"]:
    st.subheader("🏙️ Target Location")
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1: loc_history_sel = st.selectbox("Select recent location:", ["Select..."] + st.session_state.search_history)
    with search_col2: new_loc_input = st.text_input("Or select new location (Press Enter to see options):")

    location = None
    if new_loc_input:
        if new_loc_input != st.session_state.get("last_query"):
            st.session_state.last_query = new_loc_input
            with st.spinner("Searching..."):
                results = geolocator.geocode(new_loc_input, exactly_one=False, limit=5)
                st.session_state.geocode_results = results
        
        results = st.session_state.get("geocode_results")
        if results:
            opts = {f"{r.address} (Lat: {r.latitude:.2f}, Lon: {r.longitude:.2f})": r for r in results}
            chosen = st.selectbox("Multiple matches found. Select exact location:", list(opts.keys()))
            location = opts[chosen]
            short_name = location.address.split(",")[0].strip()
            if short_name not in st.session_state.search_history:
                st.session_state.search_history.insert(0, short_name)
                if len(st.session_state.search_history) > 10: st.session_state.search_history.pop()
        else: st.warning("No results found.")
    elif loc_history_sel != "Select...":
        location = geolocator.geocode(loc_history_sel, timeout=10)

    lat_target, lon_target = 52.52, 13.40 
    if location:
        lat_target, lon_target = round(location.latitude, 2), round(location.longitude, 2)
        if not (-25 <= lon_target <= 45 and 30 <= lat_target <= 72):
            st.warning(f"📍 Location {location.address} ({lat_target}°N, {lon_target}°E) is outside the ERA5 Europe domain.")
            location = None
        else:
            st.success(f"📍 **Location Matrix Active:** {location.address} | **{lat_target}°N, {lon_target}°E**")

    if location:
        if nav_selection == "Location Meteograms":
            with st.spinner("Fetching Meteogram data..."):
                df_live = get_live_timeseries(lat_target, lon_target)
            if not df_live.empty:
                met_col1, met_col2 = st.columns(2)
                env_map = {"Moderate": ("p75", "p25"), "Strong": ("p90", "p10"), "Extreme": ("p95", "p5"), "All-Time": ("max_val", "min_val")}
                env_up, env_dn = env_map.get(meteo_env.split(" ")[0], ("p90", "p10"))
                
                with met_col1:
                    st.plotly_chart(build_meteogram(df_live, ref_clim, lat_target, lon_target, target_date, "A", show_air_temp, show_app_temp, env_up, env_dn, "Heatwaves" in wave_focus), use_container_width=True)
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A", "Heatwaves" in wave_focus), use_container_width=True)
                with met_col2:
                    st.plotly_chart(build_meteogram(df_live, ref_clim, lat_target, lon_target, target_date, "B", show_air_temp, show_app_temp, env_up, env_dn, "Heatwaves" in wave_focus), use_container_width=True)
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "B", "Heatwaves" in wave_focus), use_container_width=True)
                    
                st.markdown("---")
                st.markdown(f"#### Top 10 Historical Extremes ({'Maximum' if 'Heatwaves' in wave_focus else 'Minimum'} Temperatures)")
                st.dataframe(build_top10_table(df_live, "Heatwaves" in wave_focus), use_container_width=True)
                    
        elif nav_selection == "Location Waves":
            with st.spinner("Generating Historical Waves & Annual Cycles..."):
                param_code = "TX" if "Heatwaves" in wave_focus else "TN"
                
                fig_m_a, fig_s_a, fig_c_a = get_kiesely_waves_figs(lat_target, lon_target, parameter=param_code, selected_epoch="A", threshold_level=wave_thresh, stat_metric=wave_stat_metric)
                fig_m_b, fig_s_b, fig_c_b = get_kiesely_waves_figs(lat_target, lon_target, parameter=param_code, selected_epoch="B", threshold_level=wave_thresh, stat_metric=wave_stat_metric)
                
                w_col1, w_col2 = st.columns(2)
                with w_col1: st.plotly_chart(fig_m_a, use_container_width=True)
                with w_col2: st.plotly_chart(fig_m_b, use_container_width=True)
                
                st.markdown("---")
                s_col1, s_col2 = st.columns(2)
                with s_col1: st.plotly_chart(fig_s_a, use_container_width=True)
                with s_col2: st.plotly_chart(fig_s_b, use_container_width=True)
                
                st.markdown("---")
                c_col1, c_col2 = st.columns(2)
                with c_col1: st.plotly_chart(fig_c_a, use_container_width=True)
                with c_col2: st.plotly_chart(fig_c_b, use_container_width=True)

# --- TAB 5: INFO & DISCLAIMER ---
elif nav_selection == "Imprint & Disclaimer":
    st.markdown("""
    ### Imprint & Contact
    **Operator / Scientific Contact:** *This tool is operated for scientific and informational purposes.* Email: [Your-Email@example.com]  
    
    *(Note: If operating commercially within the EU, a full postal address may be required under the TMG. For private or non-commercial scientific research, a contact email typically suffices to handle inquiries.)*
    
    ---
    
    ### Liability Disclaimer
    The data and visualizations provided by the **Synoptic Extremes Tracker (SynEx)** are for informational and research purposes only. 
    
    While every effort is made to ensure accuracy through the use of high-quality Copernicus C3S and ECMWF datasets, **no liability is accepted for the correctness, completeness, or timeliness** of the information displayed. 
    
    SynEx is not an official warning system. The classifications of heatwaves, coldwaves, and extreme synoptic events presented here **do not replace official severe weather warnings** issued by national or international meteorological services (e.g., DWD, MeteoAlarm). 
    
    No liability is assumed for any damages, losses, or actions taken based on the use of this tool and its forecasts.
    """)