# SynEx – Synoptic Extremes Tracker

Interactive Streamlit dashboard for European temperature extremes against shifting climate baselines (1961–1990 vs 1996–2025).

## Quick start (local)

```powershell
cd "C:\Users\liina\Andreas ERA5"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501

## Required data (not in git)

Place NetCDF files under `ERA5_ClimateTool/`:

| Folder | Content |
|--------|---------|
| `Reference_Climatology/` | `climatology_reference_complete.nc` or `climatology_reference.nc` |
| `Master_Batches/` | `era5_txtn_batch_*.nc`, `era5_mslp_batch_*.nc`, `era5_z500_batch_*.nc` |
| `Live_Forecasts/` | `live_forecast_*.nc` (optional; IFS bridge for last ~5 days) |

Without reference climatology the app stops with an error on startup.

## CDS / ECMWF credentials

For ERA5 downloads (`era5_update_2026.py`, downloaders):

1. Register at https://cds.climate.copernicus.eu
2. Copy `~/.cdsapirc` (Linux/Mac) or `%USERPROFILE%\.cdsapirc` (Windows):

```
url: https://cds.climate.copernicus.eu/api
key: <UID>:<API-KEY>
```

IFS open data (`fetch_ifs_forecast.py`) needs no key.

## Data update (ERA5 + IFS)

From project root with **cee_env** active (Anaconda Prompt):

```powershell
conda activate cee_env
cd "C:\Users\liina\Andreas ERA5"
python update_recent_data.py
```

- ERA5 batches extend to **today − 5 days** (Copernicus CDS queue: often 2–10 min)
- IFS live bridge writes `ERA5_ClimateTool/Live_Forecasts/live_forecast_*.nc`
- Options: `--era5-only`, `--ifs-only`, `--dry-run`, `--end-date 2026-07-31`

Requires `~/.cdsapirc` (CDS API key). IFS needs ecCodes in cee_env:

```powershell
conda install -n cee_env -c conda-forge eccodes cfgrib ecmwf-opendata
```


```powershell
python -c "import streamlit, xarray, plotly, scipy, geopy; print('OK')"
streamlit run app.py
```

## Project layout

- `app.py` – Streamlit frontend
- `backend_maps.py` – synoptic map data
- `backend_waves.py` – Kyselý heat/cold waves
- `backend_time_series.py` – location meteograms (Open-Meteo API)
- `run_pipeline.py` – event detection pipeline (optional batch processing)

See `ERA5_ClimateTool/PROTOTYPE_NOTICE.txt` – TX/TN use prototype ERA5 parameters; replace before production.
