# AtmoPulse: Real-Time Synoptic Extreme Tracking

**AtmoPulse** is an interactive Streamlit dashboard designed to dynamically track, visualize, and contextualize European temperature extremes against shifting climate baselines (1961–1990 vs. 1996–2025). Scheduled for a Beta release in Autumn 2026, the tool bridges the gap between public climate communication and meteorological state-of-the-art.

## Core Methodological Innovation
* **Predictive Persistence:** Transitions complex ETCCDI persistence indices (WSDI/CSDI) into live, forward-looking forecast overlays.
* **Physical Consistency:** Fuses live atmospheric forecasting (IFS/AIFS) with historical climate baselines (ERA5). Surface temperature parameters are rigorously calibrated via area-weighted interpolation and Quantile Mapping (CDF-Matching) to neutralize model biases.
* **Dual-Mode Architecture:** Features a Standard Mode for media/public (impact rankings, heatwave tracking) and an Expert Mode for tropospheric analytics (T850, Z500, jet stream data) alongside a trailing 5-day UTCI metric based on ERA5.

---

## Technical Setup & Developer Guide

### Quick start (local)
```powershell
cd "C:\Users\liina\Andreas ERA5"
conda activate cee_env
streamlit run app.py
```
The app opens at http://localhost:8501.

### Required data (not in git)
Place NetCDF files under `ERA5_ClimateTool/`:
* `Reference_Climatology/`: `climatology_reference_complete.nc` or `climatology_reference.nc`
* `Master_Batches/`: `era5_txtn_batch_*.nc`, `era5_mslp_batch_*.nc`, `era5_z500_batch_*.nc`
* `Live_Forecasts/`: `live_forecast_*.nc` (optional; IFS bridge for last ~5 days)

*Note: Without reference climatology, the app stops with an error on startup. Check `PROTOTYPE_NOTICE.txt` for prototype ERA5 parameters (TX/TN).*

### Data update (ERA5 + IFS)
1. **Credentials:** Register at https://cds.climate.copernicus.eu and place your API key in `~/.cdsapirc` or `%USERPROFILE%\.cdsapirc`. (IFS open data via `fetch_ifs_forecast.py` needs no key).
2. **Environment:** Ensure `eccodes`, `cfgrib`, and `ecmwf-opendata` are installed in your Conda environment.
3. **Execution:**
```powershell
conda activate cee_env
cd "C:\Users\liina\Andreas ERA5"
python update_recent_data.py
```
*ERA5 batches extend to today − 5 days. Options: `--era5-only`, `--ifs-only`, `--dry-run`, `--end-date 2026-07-31`.*

### Project Layout
* `app.py` – Streamlit frontend
* `backend_maps.py` – Synoptic map data
* `backend_waves.py` – Kyselý heat/cold waves
* `backend_time_series.py` – Location meteograms (Open-Meteo API)
* `run_pipeline.py` – Event detection pipeline

---
*Developed by Dr. Andreas Hoy | Applied Climatologist & Digital Climate Service Developer*
