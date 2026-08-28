"""
AtmoPulse Reusable UI Widgets (frontend_widgets.py)

Self-contained Streamlit widgets shared across pages: the ERA5 grid-cell
physiography profile (Point Meteogram / Point Wavogram tabs) and the
Top-10 hottest/coldest-day table (Point Meteogram tab).
"""

import math

import pandas as pd
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

from backend_io import _create_gridcell_map, load_invariant_fields
from labels import HELP

INVARIANT_VARS = ("lsm", "z", "sdor")


def _classify_roughness(sdor_m):
    if sdor_m < 20:
        return "flat"
    if sdor_m < 75:
        return "hilly"
    if sdor_m < 200:
        return "low mountains"
    return "high mountains"


def render_grid_cell_profile(location_name, lat, lon):
    """Collapsible ERA5 grid-cell physical metadata + scientific disclaimer,
    used only in the Point Meteogram / Point Wavogram tabs."""
    with st.expander("ℹ️ ERA5 Grid Cell Profile & Spatial Limits", expanded=False):
        inv_ds = load_invariant_fields()
        if inv_ds is None or not all(v in inv_ds.variables for v in INVARIANT_VARS):
            st.caption("Grid cell physiography data is currently unavailable.")
            return

        pt = inv_ds[list(INVARIANT_VARS)].sel(latitude=lat, longitude=lon, method="nearest")

        lsm = float(pt["lsm"].values)
        elevation = float(pt["z"].values) / 9.80665
        sdor = float(pt["sdor"].values)

        ns_extent = 27.8
        ew_extent = 27.8 * math.cos(math.radians(lat))
        area = ns_extent * ew_extent
        roughness_class = _classify_roughness(sdor)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
**Grid Cell Dimensions:**
The meteorological data for [{location_name}; {lat}°N, {lon}°E] is calculated based on a macro-scale ERA5 grid cell covering a total area of **{area:.1f} km²** (North-South: 27.8 km | East-West: {ew_extent:.1f} km).

**Modeled Physical Profile:**
* **Surface Cover:** {lsm*100:.0f}% Land | {(1-lsm)*100:.0f}% Water
* **Topography:** Mean elevation {elevation:.0f} m a.s.l. (Terrain roughness: {roughness_class})

⚠️ **Scientific Disclaimer:**
*AtmoPulse tracks large-scale synoptic anomalies. The displayed values represent a spatial and thermodynamic average over this entire {area:.1f} km² grid cell. Local on-the-ground measurements—especially within urbanized areas, smaller islands and/or highly structured terrain—will deviate significantly from these macro-scale baselines. Note: The underlying 0.25° model physics do not explicitly resolve urban infrastructure.*
""")
        with col2:
            if _FOLIUM_AVAILABLE:
                st_folium(
                    _create_gridcell_map(lat, lon),
                    width="100%",
                    height=300,
                    returned_objects=[],
                    key=f"gridcell_map_{lat:.2f}_{lon:.2f}",
                )
            else:
                st.caption("Map preview unavailable: install `folium` and `streamlit-folium` to enable it.")


def _top10_header_html(title: str) -> str:
    return f'<span class="atmopulse-subsection-label" title="{HELP["top10_table"]}"><b>{title}</b> ℹ️</span>'


@st.cache_data(show_spinner=False)
def build_top10_table(df_live, meteo_var):
    """Trailing 12-month (365-day) hottest/coldest-day table below the main meteogram.

    Pure data-processing (returns a DataFrame, not a Plotly figure) -> cached
    with @st.cache_data so re-sorting/re-slicing df_live doesn't re-run on
    every unrelated widget rerun (e.g. the Layout radio).
    """
    col_target = 'TG' if meteo_var == "Mean Temp (TG)" else ('TX' if meteo_var == "Max Temp (TX)" else 'TN')
    if col_target not in df_live.columns: 
        return pd.DataFrame()
    df_sorted = df_live[['Date', col_target]].dropna().sort_values(by=col_target, ascending=(meteo_var == "Min Temp (TN)"))
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date']).dt.strftime('%Y-%m-%d')
    df_sorted.rename(columns={col_target: f"{col_target} (°C)"}, inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.index += 1
    return df_sorted.head(10)
