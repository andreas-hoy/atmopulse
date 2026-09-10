"""
Schritt-B payload-size doc script (AtmoPulse map tracker optimization).

Standalone by design: it does NOT import frontend_plots / app.py / backend_io
(those need a live NetCDF store + Streamlit runtime). Instead it rebuilds the
exact same two hover strategies -- old 2D-HTML-hovertext vs. new
customdata+hovertemplate -- on a synthetic grid with the real production
dimensions (EUROPE_BBOX = -25..45 lon, 30..72 lat, native 0.25 degree ERA5
grid), so `len(fig.to_json())` is representative of the real Daily TG /
single-epoch / no-isolines case described in the task.

Run:  python measure_step_b_payload.py
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

# --- synthetic grid matching production dims (EUROPE_BBOX, 0.25 deg) ---
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -25.0, 30.0, 45.0, 72.0
lons = np.arange(LON_MIN, LON_MAX + 1e-9, 0.25)
lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, 0.25)
nlon, nlat = len(lons), len(lats)
rng = np.random.default_rng(0)

print(f"Grid: {nlat} x {nlon} = {nlat * nlon} cells (Daily TG, 1 epoch, no isolines)")

# Synthetic but realistic fields (same shapes as production tx/tn/tg + records)
v_curr = rng.uniform(-10, 30, size=(nlat, nlon)).astype(np.float64)
v_rec_w = v_curr + rng.uniform(0, 5, size=(nlat, nlon))
v_rec_c = v_curr - rng.uniform(0, 5, size=(nlat, nlon))
yr_w = rng.integers(1961, 2025, size=(nlat, nlon)).astype(np.float64)
yr_c = rng.integers(1961, 2025, size=(nlat, nlon)).astype(np.float64)
diff_w = v_curr - v_rec_w
diff_c = v_curr - v_rec_c
mask = rng.integers(1, 9, size=(nlat, nlon)).astype(np.float64)
# A few sea/NaN cells, like the real display mask (must render "N/A", not crash)
nan_frac = rng.random((nlat, nlon)) < 0.03
for arr in (v_curr, v_rec_w, v_rec_c, yr_w, yr_c, diff_w, diff_c):
    arr[nan_frac] = np.nan
loc_labels = np.array(
    [[f"Cell {i}-{j}" for j in range(nlon)] for i in range(nlat)], dtype=object
)
var_label = "Mean Temp (TG)"


# --- OLD (pre-Schritt-B): 2D HTML hovertext grid ---
def fmt_num(v):
    return f"{float(v):.1f}" if np.isfinite(v) else "N/A"


def fmt_diff(v):
    return f"{float(v):+.1f}" if np.isfinite(v) else "N/A"


def fmt_year(v):
    return str(int(float(v))) if np.isfinite(v) and float(v) > 0 else "N/A"


vfmt_num, vfmt_diff, vfmt_year = (
    np.vectorize(fmt_num), np.vectorize(fmt_diff), np.vectorize(fmt_year),
)
lon2d, lat2d = np.meshgrid(lons, lats)
old_hovertext = (
    "<b>" + loc_labels.astype(str) + "</b><br>"
    "Latitude: " + vfmt_num(lat2d) + ", Longitude: " + vfmt_num(lon2d) + "<br><br>"
    + var_label + ": " + vfmt_num(v_curr) + " \u00b0C<br>"
    "All-Time Warm: " + vfmt_num(v_rec_w) + " \u00b0C (Year " + vfmt_year(yr_w) + "; " + vfmt_diff(diff_w) + " \u00b0C diff)<br>"
    "All-Time Cold: " + vfmt_num(v_rec_c) + " \u00b0C (Year " + vfmt_year(yr_c) + "; " + vfmt_diff(diff_c) + " \u00b0C diff)"
)
fig_old = go.Figure(go.Heatmap(
    x=lons, y=lats, z=mask, hovertext=old_hovertext, showscale=False,
    zmin=1, zmax=8, zsmooth="best",
    hovertemplate="%{hovertext}<extra></extra>",
))

# --- NEW (Schritt B): customdata (nlat, nlon, 7) + text (short labels) ---
vfmt_num_o = np.vectorize(fmt_num, otypes=[object])
vfmt_diff_o = np.vectorize(fmt_diff, otypes=[object])
vfmt_year_o = np.vectorize(fmt_year, otypes=[object])
customdata = np.stack([
    vfmt_num_o(v_curr), vfmt_num_o(v_rec_w), vfmt_year_o(yr_w), vfmt_diff_o(diff_w),
    vfmt_num_o(v_rec_c), vfmt_year_o(yr_c), vfmt_diff_o(diff_c),
], axis=-1)
new_hovertemplate = (
    "<b>%{text}</b><br>"
    "Latitude: %{y:.2f}, Longitude: %{x:.2f}<br><br>"
    + var_label + ": %{customdata[0]} \u00b0C<br>"
    "All-Time Warm: %{customdata[1]} \u00b0C (Year %{customdata[2]}; %{customdata[3]} \u00b0C diff)<br>"
    "All-Time Cold: %{customdata[4]} \u00b0C (Year %{customdata[5]}; %{customdata[6]} \u00b0C diff)"
    "<extra></extra>"
)
fig_new = go.Figure(go.Heatmap(
    x=lons, y=lats, z=mask, text=loc_labels, customdata=customdata, showscale=False,
    zmin=1, zmax=8, zsmooth=False,
    hovertemplate=new_hovertemplate,
))

json_old = fig_old.to_json()
json_new = fig_new.to_json()
size_old = len(json_old.encode("utf-8"))
size_new = len(json_new.encode("utf-8"))

print(f"OLD  hovertext-grid  fig.to_json(): {size_old / 1e6:8.2f} MB  (len={len(json_old)})")
print(f"NEW  customdata+text fig.to_json(): {size_new / 1e6:8.2f} MB  (len={len(json_new)})")
print(f"Reduction: {(1 - size_new / size_old) * 100:.1f}%")

# Sanity: hover content must be reproducible from the new payload alone.
pj = json.loads(json_new)
tr = pj["data"][0]
assert tr["zsmooth"] is False, "zsmooth must be False (hard class edges), got %r" % tr.get("zsmooth")
assert "hovertext" not in tr, "hovertext must be gone (dead HTML-grid path)"
assert tr["hoverinfo"] != "skip" if "hoverinfo" in tr else True, "hoverinfo=skip is forbidden"
print("OK: zsmooth=False, no hovertext grid, hoverinfo not 'skip'.")
