# AtmoPulse: Real-Time Synoptic Extreme Tracking

**AtmoPulse** is an interactive Streamlit dashboard that tracks, visualises, and contextualises European temperature extremes against shifting climate baselines (1961–1990 vs. 1996–2025). A public beta is planned for autumn 2026.

## Method

* **Predictive persistence:** ETCCDI WSDI/CSDI-style persistence shown as a live, forward-looking overlay on IFS/AIFS forecasts.
* **Physical consistency:** ECMWF IFS and AIFS forecasts are regridded onto the ERA5 grid (conservative / fracarea) and surface temperatures can be calibrated with quantile delta mapping (QDM).
* **Dual-mode UI:** Standard mode for impact rankings and heat/cold-wave tracking; expert mode for T850, Z500, jet, and trailing ERA5 UTCI.

Daily TX/TN/TG are defined on a **00–00 UTC** calendar day. Synoptic drivers (MSLP, Z500, T850, U300/V300) use **12 UTC**. Percentiles use a centred **5-day** day-of-year window on a homogeneous **ETCCDI 365-day** calendar (29 February excised).

---

## Quick start (local)

```powershell
cd "C:\Users\liina\Andreas ERA5"
conda activate cee_env
streamlit run app.py
```

The app opens at http://localhost:8501. NetCDF archives are **not** in git; without reference climatology the app stops on startup.

### Required data (local only)

Place files under `ERA5_ClimateTool/`:

* `Reference_Climatology/` — `climatology_reference_complete.nc`, `climatology_synoptics.nc`, optional `qdm_transfer_functions.nc` and `regrid_weights_cdo.nc`
* `Master_Batches/` — `era5_master_daily_YYYY.nc`
* `Live_Forecasts/` — `ifs_daily_forecast_*.nc`, `aifs_daily_forecast_*.nc` (optional)

### Credentials

* **ERA5 / CDS:** [Copernicus CDS](https://cds.climate.copernicus.eu) API key in `%USERPROFILE%\.cdsapirc` (never commit this file).
* **IFS / AIFS open data:** `ecmwf-opendata` needs no key. Optional MARS/ECMWF API credentials belong in `%USERPROFILE%\.ecmwfapirc`, not in the repo.

### Daily operational update

```powershell
conda activate cee_env
cd "C:\Users\liina\Andreas ERA5"
run_atmopulse_pipeline.bat
```

This runs `aifs_ingestion.py`, `ifs_ingestion.py`, then `era5_daily_updater.py` (ERA5 lag: today − 5 days, 90-day rolling window).

### Historical ERA5 archive (one-off / rebuild)

1. `era5_phase1_synoptics.py` — 12 UTC MSLP, Z500, T850, U300, V300 (1940–2025)
2. `era5_phase2_2m-temp.py` — hourly TX/TN then TG, merged into master files
3. `era5_phase3_utci.py` — daily UTCI max/min
4. `era5_init_2026.py` — first fill of year 2026 (only if `era5_master_daily_2026.nc` does not exist)
5. `era5_climatology_builder.py` / `era5_synoptic_climatology.py` — percentile and synoptic baselines
6. `evaluate_master.py` — CF/thermodynamic QA after a rebuild
7. `calculate_qdm_bias.py` — IFS hindcast vs ERA5 QDM lookup (needs `ERA5_ClimateTool/IFS_Hindcasts/`)

---

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit frontend |
| `atmopulse_theme.py` | Brand, palettes, CSS |
| `backend_maps.py` | Synoptic map fields |
| `backend_map_locations.py` | Country weights and place labels |
| `backend_time_series.py` | Location meteograms |
| `backend_waves.py` | Kyselý heat/cold waves |
| `run_atmopulse_pipeline.bat` | Daily AIFS + IFS + ERA5 update |
| `era5_daily_updater.py` | Operational ERA5 rolling update |
| `ifs_ingestion.py` / `aifs_ingestion.py` | Live forecast ingest |

Python 3 with the packages in `requirements.txt` (`cee_env`).

---

*Developed by Dr. Andreas Hoy | Applied Climatologist & Digital Climate Service Developer*
