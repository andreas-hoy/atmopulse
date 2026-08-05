import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import threading
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime

from backend_maps import get_synoptic_map_data
from backend_time_series import get_live_timeseries
from backend_waves import get_kiesely_waves_figs

# --- UI & CSS: EXPERT TOP NAVIGATION BAR ---
st.set_page_config(page_title="SynEx 🌊", layout="wide", page_icon="🌊", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* Mega-Balken NUR für die Top-Navigation */
div.nav-container {
    background-color: #e6f2ff; border-radius: 12px; padding: 10px 20px;
    align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 25px; display: flex;
}
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] { 
    display: flex; flex-direction: row; gap: 10px; margin: 0; padding: 0;
}
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background-color: transparent; padding: 12px 20px; border-radius: 8px; cursor: pointer; transition: all 0.2s; margin: 0;
}
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover { background-color: #cce5ff; }
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] { 
    background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
}
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] > label p { 
    font-size: 26px !important; font-weight: normal !important; color: #0056b3 !important; 
}
div.nav-container div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child { display: none; }

/* Seitenleisten Abstände und normales Design erzwingen */
section[data-testid="stSidebar"] .stRadio > div { gap: 0rem; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p { font-size: 14px !important; font-weight: 400 !important; color: inherit !important; }
section[data-testid="stSidebar"] .stCheckbox { margin-top: -12px; }
</style>
""", unsafe_allow_html=True)

geolocator = Nominatim(user_agent="synex_extremes_tracker_2026")

if "nc_lock" not in st.session_state: st.session_state.nc_lock = threading.Lock()
if "search_history" not in st.session_state: st.session_state.search_history = ["Berlin", "Tallinn", "Budapest"]
if "toggles_warm" not in st.session_state: st.session_state.toggles_warm = {"p75": True, "p90": True, "p95": True, "rec": True}
if "toggles_cold" not in st.session_state: st.session_state.toggles_cold = {"p25": True, "p10": True, "p5": True, "rec": True}
if "offset_slider" not in st.session_state: st.session_state.offset_slider = 0

def add_day():
    if st.session_state.offset_slider < 6: st.session_state.offset_slider += 1
def sub_day():
    if st.session_state.offset_slider > -6: st.session_state.offset_slider -= 1
def update_offset_slider():
    st.session_state.offset_slider = st.session_state.offset_slider_widget
def toggle_warm_state():
    current = any(st.session_state.toggles_warm.values())
    for k in st.session_state.toggles_warm: st.session_state.toggles_warm[k] = not current
def toggle_cold_state():
    current = any(st.session_state.toggles_cold.values())
    for k in st.session_state.toggles_cold: st.session_state.toggles_cold[k] = not current

# --- DATA LOADERS ---
@st.cache_data(show_spinner=False)
def load_reference_climatology():
    clim_path = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
    if not clim_path.exists(): clim_path = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference.nc")
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
    with st.session_state.nc_lock: 
        data = get_synoptic_map_data(date_str)
        if 'valid_time' in data['mslp'].dims:
            data['mslp'] = data['mslp'].sel(valid_time=date_str, method='nearest')
            data['z500'] = data['z500'].sel(valid_time=date_str, method='nearest')
        return data

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
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in ref_clim.variables: return ref_clim[var_key].values
        return np.full((366, n_lats, n_lons), fallback)

    if map_var == "TX":
        v_h, v_p95, v_p90, v_p75 = tx_hist, safe_get(f'tx_p95_doy_{suffix}'), safe_get(f'tx_p90_doy_{suffix}'), safe_get(f'tx_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tx_p25_doy_{suffix}'), safe_get(f'tx_p10_doy_{suffix}'), safe_get(f'tx_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tx_max_val'), safe_get('tx_min_val')
    elif map_var == "TN":
        v_h, v_p95, v_p90, v_p75 = tn_hist, safe_get(f'tn_p95_doy_{suffix}'), safe_get(f'tn_p90_doy_{suffix}'), safe_get(f'tn_p75_doy_{suffix}')
        v_p25, v_p10, v_p5 = safe_get(f'tn_p25_doy_{suffix}'), safe_get(f'tn_p10_doy_{suffix}'), safe_get(f'tn_p5_doy_{suffix}')
        v_r_w, v_r_c = safe_get('tn_max_val'), safe_get('tn_min_val')
    else: 
        v_h = (tx_hist + tn_hist) / 2.0
        v_p95, v_p90, v_p75 = (safe_get(f'tx_p95_doy_{suffix}')+safe_get(f'tn_p95_doy_{suffix}'))/2, (safe_get(f'tx_p90_doy_{suffix}')+safe_get(f'tn_p90_doy_{suffix}'))/2, (safe_get(f'tx_p75_doy_{suffix}')+safe_get(f'tn_p75_doy_{suffix}'))/2
        v_p25, v_p10, v_p5 = (safe_get(f'tx_p25_doy_{suffix}')+safe_get(f'tn_p25_doy_{suffix}'))/2, (safe_get(f'tx_p10_doy_{suffix}')+safe_get(f'tn_p10_doy_{suffix}'))/2, (safe_get(f'tx_p5_doy_{suffix}')+safe_get(f'tn_p5_doy_{suffix}'))/2
        v_r_w, v_r_c = (safe_get('tx_max_val')+safe_get('tn_max_val'))/2, (safe_get('tx_min_val')+safe_get('tn_min_val'))/2

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
    
    def safe_get(var_key, fallback=np.nan):
        if var_key in daily_ref.variables: return daily_ref[var_key].values
        return np.full(tx_curr.shape, fallback)

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
        v_p95, v_p90, v_p75 = (safe_get(f'tx_p95_doy_{suffix}')+safe_get(f'tn_p95_doy_{suffix}'))/2, (safe_get(f'tx_p90_doy_{suffix}')+safe_get(f'tn_p90_doy_{suffix}'))/2, (safe_get(f'tx_p75_doy_{suffix}')+safe_get(f'tn_p75_doy_{suffix}'))/2
        v_p25, v_p10, v_p5 = (safe_get(f'tx_p25_doy_{suffix}')+safe_get(f'tn_p25_doy_{suffix}'))/2, (safe_get(f'tx_p10_doy_{suffix}')+safe_get(f'tn_p10_doy_{suffix}'))/2, (safe_get(f'tx_p5_doy_{suffix}')+safe_get(f'tn_p5_doy_{suffix}'))/2
        v_rec_w, v_rec_c = (safe_get('tx_max_val')+safe_get('tn_max_val'))/2, (safe_get('tx_min_val')+safe_get('tn_min_val'))/2
        yr_w, yr_c = safe_get('tx_max_year'), safe_get('tn_min_year')

    fig = go.Figure()
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
                f"All-Time Cold: %{{customdata[4]:.1f}} °C in Year %{{customdata[6]:.0f}} <i>({str('%{customdata[7]}').replace('7', 'diff_c')} K diff)</i><extra></extra>"
            ).replace("diff_c", "8") 
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
    if toggles.get("mslp", False): fig.add_trace(go.Contour(x=lons, y=lats, z=map_phys_data['mslp'].values.squeeze(), colorscale=[[0, '#006400'], [1, '#006400']], contours=dict(start=980, end=1040, size=5, showlabels=True), contours_coloring='lines', showscale=False, line_width=2.5, opacity=0.8, hoverinfo="skip"))
    if toggles.get("z500", False): fig.add_trace(go.Contour(x=lons, y=lats, z=map_phys_data['z500'].values.squeeze(), colorscale=[[0, 'blue'], [1, 'blue']], contours=dict(start=500, end=600, size=8, showlabels=True), contours_coloring='lines', showscale=False, line_width=2.5, opacity=0.8, hoverinfo="skip"))

    # DATA TEXT UNTEN LINKS
    fig.add_annotation(text="Data: ECMWF ERA5 (Archive) & IFS (Forecast) | Copernicus C3S", xref="paper", yref="paper", x=0.0, y=-0.05, xanchor="left", yanchor="top", showarrow=False, font=dict(size=11, color="gray"))

    fig.update_layout(uirevision='map_sync_state', xaxis=dict(range=[lons.min(), lons.max()], showgrid=False, zeroline=False, visible=False, constrain="domain"), yaxis=dict(range=[lats.min(), lats.max()], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, visible=False, constrain="domain"), margin={"r":0,"t":10,"l":0,"b":30}, height=850, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- METEOGRAMM CORE TRACES (Für Subplots) ---
def get_meteogram_traces(df_live, ref_clim, lat, lon, target_date, epoch, show_air, show_app, meteo_env, meteo_var="TG"):
    traces = []
    pt_clim = ref_clim.sel(latitude=lat, longitude=lon, method='nearest')
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)
    doys = (df_live['Date'].dt.dayofyear.values - 1) % 366
    dates = df_live['Date']
    
    if meteo_var == "Mean Temp (TG)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
    elif meteo_var == "Max Temp (TX)":
        c_base = (pt_clim[f'tx_p75_doy_{epoch}'].values[doys] + pt_clim[f'tx_p25_doy_{epoch}'].values[doys]) / 2.0
    else:
        c_base = (pt_clim[f'tn_p75_doy_{epoch}'].values[doys] + pt_clim[f'tn_p25_doy_{epoch}'].values[doys]) / 2.0
        
    env_map = {"Moderate": ("p75", "p25"), "Strong": ("p90", "p10"), "Extreme": ("p95", "p5"), "All-Time": ("max_val", "min_val")}
    el_up, el_dn = env_map.get(meteo_env, ("p90", "p10"))
    p_up_key = f'tx_{el_up}_doy_{epoch}' if el_up != "max_val" else 'tx_max_val'
    p_dn_key = f'tn_{el_dn}_doy_{epoch}' if el_dn != "min_val" else 'tn_min_val'
    
    env_upper = pt_clim[p_up_key].values[doys] if p_up_key in pt_clim.variables else np.full(len(doys), np.nan)
    env_lower = pt_clim[p_dn_key].values[doys] if p_dn_key in pt_clim.variables else np.full(len(doys), np.nan)
    
    traces.append(go.Scatter(x=dates, y=env_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=dates, y=env_lower, mode='lines', fill='tonexty', fillcolor='rgba(220,220,220,0.5)', line=dict(width=0), name='Climate Boundaries Envelope', legendgroup='env', hoverinfo='skip'))

    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    t_hist = df_live.loc[dates <= tgt_dt_norm, col_target].values if col_target in df_live.columns else ((df_live.loc[dates <= tgt_dt_norm, 'TX'].values + df_live.loc[dates <= tgt_dt_norm, 'TN'].values) / 2.0)
    
    d_hist = dates[dates <= tgt_dt_norm]
    c_hist = c_base[dates <= tgt_dt_norm]

    # Warm Anomalies
    p75 = pt_clim[f'tx_p75_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p75_doy_{epoch}' in pt_clim else c_hist
    p90 = pt_clim[f'tx_p90_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p90_doy_{epoch}' in pt_clim else c_hist
    p95 = pt_clim[f'tx_p95_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tx_p95_doy_{epoch}' in pt_clim else c_hist
    y_w1 = np.where(t_hist > c_hist, np.minimum(t_hist, p75), c_hist)
    y_w2 = np.where(t_hist > p75, np.minimum(t_hist, p90), y_w1)
    y_w3 = np.where(t_hist > p90, np.minimum(t_hist, p95), y_w2)
    y_w4 = np.where(t_hist > p95, t_hist, y_w3)
    
    # Cold Anomalies
    p25 = pt_clim[f'tn_p25_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p25_doy_{epoch}' in pt_clim else c_hist
    p10 = pt_clim[f'tn_p10_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p10_doy_{epoch}' in pt_clim else c_hist
    p5  = pt_clim[f'tn_p5_doy_{epoch}'].values[doys][dates <= tgt_dt_norm] if f'tn_p5_doy_{epoch}' in pt_clim else c_hist
    y_c1 = np.where(t_hist < c_hist, np.maximum(t_hist, p25), c_hist)
    y_c2 = np.where(t_hist < p25, np.maximum(t_hist, p10), y_c1)
    y_c3 = np.where(t_hist < p10, np.maximum(t_hist, p5), y_c2)
    y_c4 = np.where(t_hist < p5, t_hist, y_c3)

    sh = True if epoch == "A" else False # Legende nur einmal zeichnen

    traces.append(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w1, mode='lines', fill='tonexty', fillcolor='rgba(255,200,200,0.5)', line=dict(width=0), name='Warm Moderate', legendgroup='wm', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w2, mode='lines', fill='tonexty', fillcolor='rgba(255,130,130,0.6)', line=dict(width=0), name='Warm Strong', legendgroup='ws', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w3, mode='lines', fill='tonexty', fillcolor='rgba(220,40,40,0.7)', line=dict(width=0), name='Warm Extreme', legendgroup='we', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_w4, mode='lines', fill='tonexty', fillcolor='rgba(130,0,20,0.85)', line=dict(width=0), name='Warm Record', legendgroup='wr', showlegend=sh, hoverinfo='skip'))

    traces.append(go.Scatter(x=d_hist, y=c_hist, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c1, mode='lines', fill='tonexty', fillcolor='rgba(200,225,255,0.5)', line=dict(width=0), name='Cold Moderate', legendgroup='cm', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c2, mode='lines', fill='tonexty', fillcolor='rgba(120,175,255,0.6)', line=dict(width=0), name='Cold Strong', legendgroup='cs', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c3, mode='lines', fill='tonexty', fillcolor='rgba(30,90,220,0.7)', line=dict(width=0), name='Cold Extreme', legendgroup='ce', showlegend=sh, hoverinfo='skip'))
    traces.append(go.Scatter(x=d_hist, y=y_c4, mode='lines', fill='tonexty', fillcolor='rgba(5,20,120,0.85)', line=dict(width=0), name='Cold Record', legendgroup='cr', showlegend=sh, hoverinfo='skip'))

    traces.append(go.Scatter(x=dates, y=c_base, mode='lines', line=dict(color='black', width=2), name='Climatology Base', legendgroup='base', showlegend=sh))

    fcst_mask = dates >= tgt_dt_norm
    if show_app:
        col_app = 'AT_Max' if meteo_var == "Max Temp (TX)" else ('AT_Min' if meteo_var == "Min Temp (TN)" else 'AT_Mean')
        if col_app in df_live.columns:
            traces.append(go.Scatter(x=d_hist, y=df_live.loc[dates <= tgt_dt_norm, col_app], mode='lines', name='Apparent Temperature', legendgroup='app', showlegend=sh, line=dict(color='#2ca02c', width=1.5), hovertemplate="Apparent Temperature: %{y:.1f}°C<extra></extra>"))
            traces.append(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_app], mode='lines', line=dict(color='#2ca02c', width=1.5, dash='dot'), legendgroup='app', showlegend=False, hovertemplate="Apparent Temperature (Fcst): %{y:.1f}°C<extra></extra>"))
    
    if show_air:
        c_ref_w = pt_clim['tx_max_val'].values[doys][dates <= tgt_dt_norm] if 'tx_max_val' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_c = pt_clim['tn_min_val'].values[doys][dates <= tgt_dt_norm] if 'tn_min_val' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_yw = pt_clim['tx_max_year'].values[doys][dates <= tgt_dt_norm] if 'tx_max_year' in pt_clim else np.full(len(d_hist), np.nan)
        c_ref_yc = pt_clim['tn_min_year'].values[doys][dates <= tgt_dt_norm] if 'tn_min_year' in pt_clim else np.full(len(d_hist), np.nan)

        c_data = np.empty((len(d_hist), 5), dtype=object)
        c_data[:, 0] = np.round(c_hist, 1)
        c_data[:, 1] = np.round(c_ref_w, 1)
        c_data[:, 2] = np.round(c_ref_c, 1)
        c_data[:, 3] = c_ref_yw
        c_data[:, 4] = c_ref_yc

        traces.append(go.Scatter(x=d_hist, y=t_hist, mode='lines', customdata=c_data, name='Air Temperature', legendgroup='air', showlegend=sh, line=dict(color='rgba(0,0,0,0.7)', width=1.5), 
            hovertemplate="Air Temperature: %{y:.1f}°C<br>Reference Value: %{customdata[0]:.1f}°C<br>Hist. Max: %{customdata[1]:.1f}°C (%{customdata[3]:.0f})<br>Hist. Min: %{customdata[2]:.1f}°C (%{customdata[4]:.0f})<extra></extra>"))
        if col_target in df_live.columns:
            traces.append(go.Scatter(x=dates[fcst_mask], y=df_live.loc[fcst_mask, col_target], mode='lines', name='Air Temp Forecast', legendgroup='air', showlegend=False, line=dict(color='gray', width=2.5, dash='dot'), hovertemplate="Air Temperature (Fcst): %{y:.1f}°C<extra></extra>"))
        
    return traces

def build_top10_table(df_live, meteo_var):
    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    if col_target not in df_live.columns: return pd.DataFrame()
    df_sorted = df_live[['Date', col_target]].dropna().sort_values(by=col_target, ascending=(meteo_var == "Min Temp (TN)"))
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date']).dt.strftime('%Y-%m-%d')
    df_sorted.rename(columns={col_target: f"{col_target} (°C)"}, inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1
    return df_sorted.head(10)

# --- DATETIME64 CRASH BUGFIX ---
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
    
    if is_warm and f'tx_p95_doy_{epoch}' in pt_clim.variables:
        c_75, c_90, c_95, c_rec = pt_clim[f'tx_p75_doy_{epoch}'].values[doys], pt_clim[f'tx_p90_doy_{epoch}'].values[doys], pt_clim[f'tx_p95_doy_{epoch}'].values[doys], pt_clim['tx_max_val'].values[doys]
        df['p75'], df['p90'], df['p95'], df['rec'] = (v >= c_75) & (v < c_90), (v >= c_90) & (v < c_95), (v >= c_95) & (v < c_rec), v >= c_rec
    elif not is_warm and f'tn_p5_doy_{epoch}' in pt_clim.variables:
        c_25, c_10, c_5, c_rec = pt_clim[f'tn_p25_doy_{epoch}'].values[doys], pt_clim[f'tn_p10_doy_{epoch}'].values[doys], pt_clim[f'tn_p5_doy_{epoch}'].values[doys], pt_clim['tn_min_val'].values[doys]
        df['p25'], df['p10'], df['p5'], df['rec'] = (v <= c_25) & (v > c_10), (v <= c_10) & (v > c_5), (v <= c_5) & (v > c_rec), v <= c_rec
    else: return go.Figure().add_annotation(text="Data Missing.", showarrow=False)

    cols_to_sum = ['year', 'p75', 'p90', 'p95', 'rec'] if is_warm else ['year', 'p25', 'p10', 'p5', 'rec']
    res = df[cols_to_sum].groupby('year').sum()
    
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

    fig.update_layout(barmode='stack', title=f"Days exceeding thresholds | {'1961–1990' if epoch=='A' else '1996–2025'}", height=300, margin=dict(t=30, b=10), template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
    return fig

# --- WAVE CACHE WRAPPER ---
@st.cache_data(show_spinner=False)
def fetch_wave_figs(lat_target, lon_target, param_code, selected_epoch, wave_thresh, wave_stat_metric):
    return get_kiesely_waves_figs(lat_target, lon_target, parameter=param_code, selected_epoch=selected_epoch, threshold_level=wave_thresh, stat_metric=wave_stat_metric)

# --- UI LAYOUT: TOP NAVIGATION BAR ---
nav_col1, nav_col2 = st.columns([1, 6])
with nav_col1:
    st.markdown("<div style='height: 50px; display: flex; align-items: center; font-size: 60px; font-weight: bold; color: #0056b3; padding-left: 10px;'>SynEx 🌊</div>", unsafe_allow_html=True)
with nav_col2:
    st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
    nav_selection = st.radio("Navigation", ["Start", "Synoptic Maps", "Location Meteograms", "Location Waves", "Imprint & Disclaimer"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
st.divider()

default_date = pd.Timestamp.now().floor('D')
target_month = default_date.month
is_warm_season = False
if 4 < target_month < 10: is_warm_season = True
elif target_month == 4 and default_date.day >= 16: is_warm_season = True
elif target_month == 10 and default_date.day <= 15: is_warm_season = True
default_wave_idx = 0 if is_warm_season else 1

with st.sidebar:
    if nav_selection in ["Synoptic Maps", "Location Meteograms", "Location Waves"]:
        tooltip_text = (
            "Data Origin (Hybrid System Specifications):\n\n"
            "1. ERA5 Reanalysis: The primary climate reference dataset. Fully quality-assured "
            "data is typically available with a latency of 2 to 3 months behind real-time.\n\n"
            "2. ERA5T (Preliminary): Preliminary daily updates that seamlessly close the gap "
            "between the final ERA5 release and approximately 5 days prior to the present.\n\n"
            "3. ECMWF IFS (Analysis & HRES Forecast): Operative model runs that bridge the remaining "
            "5-day latency to real-time (using analysis data) and provide the short- to medium-range "
            "weather forecasts."
        )
        st.markdown(
            f"📡 **Data Vintage:** ERA5 Archive (~ 5 days ago) | IFS Forecast ({default_date.strftime('%d.%m.%Y')} 12 UTC)", 
            help=tooltip_text
        )
        st.header("Control Panel")
        
        st.slider("Forecast Offset (Days):", -6, 6, st.session_state.offset_slider, key="offset_slider_widget", on_change=update_offset_slider, help="Adjusts the target date. Negative values analyze the past (ERA5 reanalysis), positive values look into the future (IFS forecast).")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1: st.button("⬅️ Prev Day", on_click=sub_day, use_container_width=True)
        with btn_col2: st.button("Next Day ➡️", on_click=add_day, use_container_width=True)
        
        target_date = default_date + pd.Timedelta(days=st.session_state.offset_slider)
        st.info(f"Target Date: **{target_date.strftime('%d.%m.%Y')}**")
        
        toggles = {}
        
        if nav_selection == "Synoptic Maps":
            st.markdown("---")
            map_var = st.radio("**Mapped Variable:**", ("Mean Temperature (TG)", "Maximum Temperature (TX)", "Minimum Temperature (TN)"), index=0, help="Mean Temperature: Daily Mean Temperature (TG). The best proxy for the total thermal energy of the day. \n\nMaximum Temperature: Daily Maximum Temperature (TX). Represents daytime warming (or lack thereof). \n\nMinimum Temperature: Daily Minimum Temperature (TN). Represents nighttime cooling (or lack thereof).")
            map_var_code = map_var.split('(')[1].strip(')')
            
            st.markdown("---")
            view_mode = st.radio("**Map View Mode:**", ("Standard (Daily Extremes)", "Duration (Cumulative Persistence)"), help="Toggle between daily snap-shot and the persistence duration of synoptic events.")
            persist_metric = "Warm Strong"
            st.markdown("---")
            top10_threshold = st.radio("**Analysis Level**", ("Moderate", "Strong", "Extreme", "All-Time Record"), index=1, help="The following percentile-based levels can be selected: moderate (P75/25), strong (P90/10) and extreme (P95/5) levels, as well as all-time records.")
            
            if "Standard" in view_mode:
                st.markdown("---")
                st.markdown("**Map Extremes**", help="The following percentile-based levels can be selected: moderate (P75/25), strong (P90/10) and extreme (P95/5) levels, as well as all-time records")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    if st.button("Warm: OFF" if any(st.session_state.toggles_warm.values()) else "Warm: ON", use_container_width=True, help="Toggle all warm anomaly layers"): toggle_warm_state(); st.rerun()
                with m_col2:
                    if st.button("Cold: OFF" if any(st.session_state.toggles_cold.values()) else "Cold: ON", use_container_width=True, help="Toggle all cold anomaly layers"): toggle_cold_state(); st.rerun()
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
            
            if nav_selection == "Location Meteograms":
                meteo_var = st.radio("Variable:", ["Mean Temp (TG)", "Max Temp (TX)", "Min Temp (TN)"])
                st.markdown("<br>", unsafe_allow_html=True)
                meteo_env = st.selectbox("Background Envelope:", ["Moderate", "Strong", "Extreme", "All-Time"], index=1, help="Displays the corresponding climate boundaries (percentile-based) behind the temperature curve: Uses the 75th (warm) and 25th (cold) percentile for moderate, 90th (warm) and 10th (cold) for strong and 95th (warm) and 5th (cold) for extreme conditions within the reference period. All-time records are given for the full period (starting 1940) prior to the current year.")
                st.markdown("<br>", unsafe_allow_html=True)
                show_air_temp = st.checkbox("Show Air Temperature Colors", value=True, help="Colors the space below the curve for cold anomalies and above the curve for warm anomalies.")
                show_app_temp = st.checkbox("Show Apparent Temperature", value=False, help="Dotted Line: 'Feels-like' temperature, combining 2m air temperature, relative humidity and wind speed.")
            
            if nav_selection == "Location Waves":
                wave_focus = st.radio("Wave Event Type:", ("Heatwaves", "Coldwaves"), index=default_wave_idx, help="Heatwaves: Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) threshold for at least 3 consecutive days. The wave continues as long as the average TX remains above this threshold, and terminates immediately if a single day drops below a secondary, lower tolerance threshold.\n\nColdwaves: Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) threshold for at least 3 consecutive days. It continues while the average TN remains below this threshold, and ends if a single day rises above the upper tolerance limit.")
                wave_thresh = st.radio("Wave Intensity Threshold:", ("Strong", "Extreme"), help="Strong: Calculates waves using the 90th (heat) or 10th (cold) percentile as the main trigger.\n\nExtreme: Calculates waves using the stricter 95th (heat) or 5th (cold) percentile as the main trigger.")
                st.markdown("---")
                wave_stat_metric = st.radio(
                    "Wave Statistic Metric:", 
                    ("Cumulative Annual Wave Intensity", "Maximum Annual Wave Intensity", "Cumulative Heat/Cold Intensity", "Annual Cycle Frequency"),
                    help="Cumulative Annual Wave Intensity: The sum of the intensities (in Kelvin) of ALL distinct waves that occurred in a given year.\n\nMaximum Annual Wave Intensity: The intensity (in Kelvin) of the SINGLE strongest wave event of the year.\n\nCumulative Heat/Cold Intensity: The total accumulated intensity of ALL days exceeding the threshold per year, even if they don't form a consecutive 3-day wave.\n\nAnnual Cycle Frequency: Shows the 5-day-smoothed relative frequency of events throughout the year."
                )

if ref_clim is None: st.error("Reference Climatology missing or corrupted! Please rebuild."); st.stop()

if nav_selection == "Start":
    st.markdown("### Welcome to the Synoptic Extremes Tracker (SynEx)")
    st.markdown("""
    **SynEx** merges real-time extreme weather tracking with shifting climate baselines. It provides interactive, synoptic-scale mapping and deep-dive local profiles. Currently focused on extreme temperatures, SynEx aims to integrate further atmospheric variables in the future.
    <br><br>
    #### Understanding Percentiles
    SynEx relies heavily on percentiles to contextualize current weather against historical norms. In our maps and meteograms, percentiles are calculated using a **centered 5-day moving window** across the reference periods (1961–1990 and 1996–2025). 
    For instance, the 90th percentile (P90) is a threshold exceeded only 10% of the time during the historical baseline. We track **Moderate** (P75/P25), **Strong** (P90/P10), and **Extreme** (P95/P5) thresholds to dynamically classify the severity of synoptic events.
    <br><br>
    #### The Importance of Event Duration
    The impact of extreme temperatures on sectors like human health, agriculture and infrastructure scales drastically with duration. A single hot day is a weather event; a prolonged sequence becomes a systemic hazard. 
    In the **Synoptic Maps** tab, you can visualize this through the **Cumulative Persistence** layer, showing how many days an extreme event has lasted. By default, the maps also display an overlay for **WSDI and CSDI** conditions.
    <br><br>
    #### Local Wave Definitions
    In the **Location Waves** tab, SynEx uses a sophisticated definition (adapted from Kyselý) to track seasonally-bound heatwaves and coldwaves:
    * **Heatwaves:** Triggered when the daily maximum temperature (TX) exceeds the local summer (June–August) threshold for at least 3 consecutive days. 
    * **Coldwaves:** Triggered when the daily minimum temperature (TN) falls below the local winter (December–February) threshold for at least 3 consecutive days.
    """, unsafe_allow_html=True)
    
    img_col1, img_col2 = st.columns(2)
    with img_col1: st.image("Warm.jpg", use_container_width=True, caption="Erfassung von Hitzewellen")
    with img_col2: st.image("Kalt.jpg", use_container_width=True, caption="Erfassung von Kältewellen")

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
        
        if map_layout == "Side-by-Side Compare":
            map_col1, map_col2 = st.columns(2)
            with map_col1:
                st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "A", map_var_code).update_layout(title="Reference Period A (1961–1990)"), use_container_width=True, key="map_a")
            with map_col2:
                st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, "B", map_var_code).update_layout(title="Reference Period B (1996–2025)"), use_container_width=True, key="map_b")
        else:
            ep_sel = "A" if "A" in flicker_epoch else "B"
            st.plotly_chart(build_baseline_map(ref_clim, map_phys_data, target_date, st.session_state.toggles_warm, st.session_state.toggles_cold, toggles, view_mode, persist_metric, top10_threshold, ep_sel, map_var_code).update_layout(title=flicker_epoch), use_container_width=True, key="map_flicker")
    except Exception as e: st.error(f"Error loading maps: {e}")

elif nav_selection in ["Location Meteograms", "Location Waves"]:
    st.subheader("🏙️ Target Location")
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1: loc_history_sel = st.selectbox("Select recent location:", ["Select..."] + st.session_state.search_history)
    with search_col2: new_loc_input = st.text_input("Or select new location (Press Enter to see options):")
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

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
            st.warning(f"📍 Location {location.address} is outside the Europe domain.")
            location = None
        else: st.success(f"📍 **Location Matrix Active:** {location.address} | **{lat_target}°N, {lon_target}°E**")

    if location:
        if nav_selection == "Location Meteograms":
            map_layout = st.radio("Layout:", ("Side-by-Side Compare", "Single Map Flicker"), horizontal=True, key="met_layout")
            with st.spinner("Fetching Meteogram data..."): df_live = get_live_timeseries(lat_target, lon_target)
            if not df_live.empty:
                col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
                t_arr = df_live[col_target].values if col_target in df_live.columns else ((df_live['TX'].values + df_live['TN'].values)/2.0)
                global_min, global_max = np.nanmin(t_arr) - 3, np.nanmax(t_arr) + 3
                tgt_dt_norm = pd.to_datetime(target_date).tz_localize(None)

                traces_a = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "A", show_air_temp, show_app_temp, meteo_env, meteo_var)
                traces_b = get_meteogram_traces(df_live, ref_clim, lat_target, lon_target, target_date, "B", show_air_temp, show_app_temp, meteo_env, meteo_var)
                
                if map_layout == "Side-by-Side Compare":
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference Period A (1961–1990)", "Reference Period B (1996–2025)"), shared_yaxes=True)
                    for trace in traces_a: fig.add_trace(trace, row=1, col=1)
                    for trace in traces_b: fig.add_trace(trace, row=1, col=2)
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=1)
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8, row=1, col=2)
                    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", showgrid=True, gridcolor='rgba(200,200,200,0.3)')
                    fig.update_yaxes(range=[global_min, global_max])
                    fig.update_layout(hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A", meteo_var != "Min Temp (TN)"), use_container_width=True)
                    with c2: st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "B", meteo_var != "Min Temp (TN)"), use_container_width=True)
                else:
                    flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="met_ep")
                    fig = go.Figure(data=traces_a if "A" in flicker_epoch else traces_b)
                    fig.add_vline(x=tgt_dt_norm.timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.8)
                    fig.update_xaxes(dtick="M2", tickformat="%b\n%Y", showgrid=True, gridcolor='rgba(200,200,200,0.3)')
                    fig.update_yaxes(range=[global_min, global_max])
                    fig.update_layout(title=f"Reference Period {'1961–1990' if 'A' in flicker_epoch else '1996–2025'}", hovermode="x unified", height=500, template="plotly_white", margin=dict(t=40, b=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(build_yearly_extremes_chart(lat_target, lon_target, "A" if "A" in flicker_epoch else "B", meteo_var != "Min Temp (TN)"), use_container_width=True)
                
                st.markdown("#### Top 10 Extreme Events (Historical)")
                st.dataframe(build_top10_table(df_live, meteo_var), use_container_width=True)

        elif nav_selection == "Location Waves":
            map_layout = st.radio("Layout:", ("Side-by-Side Compare", "Single Map Flicker"), horizontal=True, key="wave_layout")
            with st.spinner("Generating Historical Waves..."):
                param_code = "TX" if "Heatwaves" in wave_focus else "TN"
                fig_m_a, fig_s_a = fetch_wave_figs(lat_target, lon_target, param_code, "A", wave_thresh, wave_stat_metric)
                fig_m_b, fig_s_b = fetch_wave_figs(lat_target, lon_target, param_code, "B", wave_thresh, wave_stat_metric)
                
                if fig_s_a.data and fig_s_b.data:
                    max_a = max([max(t.y) for t in fig_s_a.data if t.y is not None and len(t.y)>0])
                    max_b = max([max(t.y) for t in fig_s_b.data if t.y is not None and len(t.y)>0])
                    g_max = max(max_a, max_b) * 1.1
                    fig_s_a.update_yaxes(range=[0, g_max]); fig_s_b.update_yaxes(range=[0, g_max])

                if map_layout == "Side-by-Side Compare":
                    w_col1, w_col2 = st.columns(2)
                    with w_col1:
                        st.plotly_chart(fig_m_a, use_container_width=True)
                        st.plotly_chart(fig_s_a, use_container_width=True)
                    with w_col2:
                        st.plotly_chart(fig_m_b, use_container_width=True)
                        st.plotly_chart(fig_s_b, use_container_width=True)
                else:
                    flicker_epoch = st.radio("Select Reference Period:", ("A (1961–1990)", "B (1996–2025)"), horizontal=True, key="wave_ep")
                    st.plotly_chart(fig_m_a if "A" in flicker_epoch else fig_m_b, use_container_width=True)
                    st.plotly_chart(fig_s_a if "A" in flicker_epoch else fig_s_b, use_container_width=True)

elif nav_selection == "Imprint & Disclaimer":
    st.markdown("""
    ### Imprint & Contact
    **Operator / Scientific Contact:** Dr. Andreas Hoy  
    Tallinn, Estonia  
    Email: ahoy.dresden@gmail.com  
    
    ---
    
    ### Liability Disclaimer
    The data and visualizations provided by the **Synoptic Extremes Tracker (SynEx)** are for informational and research purposes only. 
    
    While every effort is made to ensure accuracy through the use of high-quality Copernicus C3S and ECMWF datasets, **no liability is accepted for the correctness, completeness, or timeliness** of the information displayed. 
    """)