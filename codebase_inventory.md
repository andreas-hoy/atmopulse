# AtmoPulse codebase inventory (read-only)

Scope: every `*.py`, batch script, and configuration file under the workspace. LOC uses **Total** = physical lines, **SLOC** = non-blank / non-comment / non-docstring, **Comments** = `#` lines plus docstrings. Dates are last-write on disk (local). Relative paths are from `C:\Users\liina\Andreas ERA5`.

---

## Section 1 — Structured summary

### 1A. Active application and data pipeline

| File | Path | Size / Modified | Total / SLOC / Comments | Role | Core dependencies | Streamlit cache | Architectural risk |
|---|---|---|---|---|---|---|---|
| `app.py` | `.` | 148.3 KB / 2026-08-28 | 2825 / 2203 / 351 | **UI/Frontend** + analytics + NetCDF I/O orchestrator | Internal: `backend_maps`, `backend_map_locations`, `backend_waves`, `backend_narrative`, `labels`, `atmopulse_theme`. Heavy: `streamlit`, `xarray`, `pandas`, `numpy`, `plotly`, `requests`, `geopy`, optional `folium`/`streamlit-folium`. `subprocess` for isolated HDF5 reads | **Heavy.** `@st.cache_resource`: `load_reference_climatology`, `load_invariant_fields`, `get_master_files`, `get_master_archive_ds`, `get_live_txtn_ds`, `_load_persistence_window_source`, `_load_qdm_bias_ds`, `get_europe_borders_trace`, `fetch_cached_synoptic_data`, `get_map_historical_records_bundle`, `build_yearly_extremes_chart`. `@st.cache_data`: `_create_gridcell_map`, `_load_persistence_daily_series`, `get_live_point_series`, `get_map_location_labels`, `get_country_weight_grid`, `get_persistence_arrays`, `compute_map_footprint`, `calculate_top10`, `build_top10_table`, `_load_point_archive_series`, `fetch_wave_figs` | **Critical.** Monolith (~2.2k SLOC). Mixes UI, I/O, ranking, meteograms. `importlib.reload` of backends on **every** rerun. Three parallel archive loaders vs maps/waves. Hardcoded `ERA5_ClimateTool/...` paths. Duplicate `_harmonize_master_archive` vs `backend_maps`. Module-level `Nominatim` + `ref_clim` open |
| `backend_maps.py` | `.` | 21.4 KB / 2026-08-28 | 538 / 351 / 108 | **Pipeline/Backend** — synoptic merge of yearly masters + live IFS/AIFS | Internal: none. Heavy: `xarray`, `numpy`, `pandas`, `streamlit`. Engines: `netcdf4`, fallback `h5netcdf` | `@st.cache_resource`: `load_global_datasets`, `load_windowed_synoptic_arrays` | Dual-open of in-progress year files (HDF error on Windows). Docstring still says “dask-backed”; loaders now prefer unchunked `netcdf4`. Process-global `_synoptic_anchor`. No shared path constants |
| `backend_waves.py` | `.` | 33.5 KB / 2026-08-28 | 740 / 568 / 64 | **Analytics** — Kyselý waves, ridge plots, 1940–present point TX/TN | Internal: `backend_maps.drop_era5t_aux`, `_open_live_forecast_ds`; `atmopulse_theme`. Heavy: `xarray`, `numpy`, `pandas`, `plotly`, `scipy.interpolate`, `streamlit`, `subprocess` | `@st.cache_data`: `_era5_master_point_txtn`, `get_kiesely_waves_figs`. `@st.cache_resource`: `_load_waves_archive_ds`, `_load_waves_climatology` | **High.** Year-by-year NetCDF opens (not the app mfdataset). Duplicated subprocess JSON extractor vs `app.py`. Cold-start point extract can be minutes. `_load_waves_archive_ds` is existence-only despite the name |
| `backend_narrative.py` | `.` | 15.1 KB / 2026-08-28 | 344 / 157 / 155 | **Analytics** — map/meteogram/wavogram copy | Internal: `backend_waves.get_wave_historical_rank`. Heavy: `numpy`, `streamlit` | `@st.cache_data`: `spatial_extreme_footprint` | Low. Tight and vectorized. Tight coupling to `app.py` mask codes (1–8) |
| `backend_map_locations.py` | `.` | 10.4 KB / 2026-08-25 | 279 / 186 / 49 | **Analytics** — Natural Earth STRtree, Top-10 country weights, hover labels | Internal: none. Heavy: `requests`, `numpy`, **`shapely`**, **`pyproj`** (neither in `requirements.txt`) | **None** in this file. App wraps callers with `@st.cache_data`. Module singleton `_LOCATION_INDEX` | Nested Python loops over every grid cell. Runtime GitHub GeoJSON fetch. Missing declared deps. Uncached if imported outside app wrappers |
| `backend_time_series.py` | `.` | 1.1 KB / 2026-08-28 | 24 / 5 / 18 | **Deprecated stub** (former Open-Meteo meteogram) | None (raises `RuntimeError`) | None | **Orphan.** README still lists it as live. Replacement lives in `app.get_live_point_series` |
| `atmopulse_theme.py` | `.` | 30.2 KB / 2026-08-28 | 819 / 724 / 39 | **UI/Frontend** — palettes, CSS, Plotly typography | stdlib `urllib.parse` only | None | Large CSS string in Python. Reloaded every Streamlit rerun via `importlib.reload` |
| `labels.py` | `.` | 6.2 KB / 2026-08-28 | 107 / 78 / 24 | **UI/Frontend** — tooltip copy | None | None | Low. Clean split from layout |
| `aifs_ingestion.py` | `.` | 11.4 KB / 2026-08-27 | 357 / 272 / 21 | **Ingestion** — ECMWF AIFS → ERA5 grid daily NetCDF | Internal: none. Heavy: `xarray`, `numpy`, `scipy.sparse`, `ecmwf.opendata`, `cfgrib` | None (CLI) | Near-clone of `ifs_ingestion.py`. `TMP_DIR.mkdir` at **import**. Hardcoded `ERA5_ClimateTool`. Needs `climatology_synoptics.nc` + CDO weights |
| `ifs_ingestion.py` | `.` | 10.6 KB / 2026-08-27 | 321 / 240 / 25 | **Ingestion** — ECMWF IFS HRES TX/TN/TG + synoptics | Same as AIFS | None (CLI) | Same duplication. Called by `run_atmopulse_pipeline.bat`. cfgrib time-axis fragility |
| `era5_daily_updater.py` | `.` | 8.6 KB / 2026-08-27 | 201 / 149 / 21 | **Ingestion** — operational 90-day ERA5T rolling update | `cdsapi`, `xarray` | None (CLI) | Hardcoded `ERA5_ClimateTool/Master_Batches`. Parallel CDS + in-place yearly NetCDF rewrite (HDF5 lock risk vs live app) |
| `era5_init_2026.py` | `.` | 8.0 KB / 2026-08-27 | 220 / 173 / 14 | **Ingestion** — one-shot 2026 master bootstrap | `cdsapi`, `xarray` | None | Logic overlap with daily updater (`harmonize_time`, CDS area, Kelvin→°C). Not in the daily `.bat` |
| `era5_phase1_synoptics.py` | `.` | 7.6 KB / 2026-08-25 | 181 / 120 / 30 | **Ingestion** — historical 12 UTC synoptics 1940–2025 | `cdsapi`, `xarray`, `concurrent.futures` | None | 4-worker CDS saturation. Years hardcoded through 2025. Duplicate `DEFAULT_AREA` / logging boilerplate |
| `era5_phase2_2m-temp.py` | `.` | 9.9 KB / 2026-08-25 | 222 / 149 / 31 | **Ingestion** — historical hourly TX/TN then TG merge | `cdsapi`, `xarray`, `concurrent.futures` | None | Two-pass CDS. Tape vs disk year split. Same boilerplate as phase 1 |
| `era5_phase3_utci.py` | `.` | 6.7 KB / 2026-08-25 | 205 / 154 / 14 | **Ingestion** — UTCI max/min into masters | `cdsapi`, `xarray`, `zipfile` | None | ZIP CDS workaround. In-place rewrite of all yearly masters |
| `era5_climatology_builder.py` | `.` | 9.4 KB / 2026-08-25 | 217 / 144 / 33 | **Pipeline/Backend** — TX/TN/TG/T850 percentiles + records | `xarray`, `numpy`, `pandas` (`open_mfdataset(..., parallel=True)`) | None | Full 1940–present mfdataset in RAM. Hardcoded `CUTOFF_DATE = '2026-07-31'`. Duplicated `preprocess_era5t` / `get_window_doys` vs synoptic builder |
| `era5_synoptic_climatology.py` | `.` | 4.9 KB / 2026-08-25 | 161 / 118 / 11 | **Pipeline/Backend** — MSLP/Z500 DOY means | `xarray`, `numpy` | None | Twin of climatology builder (same epochs, cutoff, ERA5T preprocess) |
| `calculate_qdm_bias.py` | `.` | 6.5 KB / 2026-08-25 | 208 / 150 / 17 | **Analytics** — IFS vs ERA5 QDM lookup | `xarray`, `numpy` (`open_mfdataset`) | None | Needs `IFS_Hindcasts/` (gitignored). Duplicate ETCCDI DOY helper vs `backend_maps.etccdi_doy_365`. Output optionally loaded by app |
| `evaluate_master.py` | `.` | 5.2 KB / 2026-08-25 | 140 / 100 / 17 | **Analytics** — CF/thermodynamic QA of masters | `xarray`, `numpy` | None | Diagnostic only. Hardcoded Pärnu check point |
| `download_era5_invariants.py` | `.` | 1.7 KB / 2026-08-28 | 57 / 27 / 20 | **Ingestion** — one-shot physiography NetCDF | `cdsapi` | None | Documented standalone. App reads `era5_invariants.nc` if present |

### 1B. Configuration, batch, and marker files

| File | Path | Size / Modified | Total / SLOC / Comments | Role | Core dependencies | Streamlit cache | Architectural risk |
|---|---|---|---|---|---|---|---|
| `run_atmopulse_pipeline.bat` | `.` | 1.5 KB / 2026-08-27 | 54 / 41 / 6 | **Configuration** — daily AIFS → IFS → ERA5 | Calls `aifs_ingestion.py`, `ifs_ingestion.py`, `run_era5_update.bat` | n/a | **Hardcoded** `C:\Users\liina\miniconda3\...` and `C:\Users\liina\Andreas ERA5`. AIFS failure is a warning; IFS failure is an error but pipeline still continues to ERA5 |
| `run_era5_update.bat` | `.` | 0.3 KB / 2026-08-25 | 9 / 4 / 3 | **Configuration** — wrapper for daily ERA5 | `era5_daily_updater.py` | n/a | Same absolute Conda + project paths |
| `requirements.txt` | `.` | 0.6 KB / 2026-08-28 | 30 / 21 / 5 | **Configuration** — declared Python deps | streamlit, xarray, netCDF4, pandas, numpy, plotly, scipy, requests, geopy, folium, cdsapi, ecmwf-opendata, cfgrib, eccodes; optional regionmask, openmeteo-*, matplotlib, statsmodels | n/a | **Gap:** `shapely` and `pyproj` used but undeclared. `h5netcdf` used as fallback, undeclared. Optional `regionmask` / Open-Meteo unused in live `*.py`. Open-Meteo leftover after stubbing `backend_time_series.py` |
| `.streamlit/config.toml` | `.streamlit/` | 0.1 KB / 2026-08-27 | 6 / 6 / 0 | **Configuration** — Streamlit theme | n/a | n/a | Theme colors diverge from `atmopulse_theme.py` brand CSS (two sources of truth) |
| `.vscode/settings.json` | `.vscode/` | 0.1 KB / 2026-08-26 | 4 / 4 / 0 | **Configuration** — Conda as default env | n/a | n/a | Low |
| `.gitignore` | `.` | 0.6 KB / 2026-08-27 | 44 / 31 / 8 | **Configuration** — ignore data, secrets, backups | n/a | n/a | Correctly excludes `ERA5_ClimateTool/`, `*.nc`, API keys. `backup_*/` is listed but `backup_2026-08-25_code/` is still in the tree |
| `Downloading` | `.` | 0.0 KB / 2026-08-28 | 1 / 1 / 0 | Status stub (`--- IFS (physics) forecast...`) | none | n/a | Not a script; not referenced by Python |
| `Updating` | `.` | 0.0 KB / 2026-08-28 | 1 / 1 / 0 | Status stub (`--- ERA5 baseline...`) | none | n/a | Same |

### 1C. Ancillary and frozen snapshot (not on the live import graph)

| File / group | Path | Size / Modified | Total / SLOC / Comments | Role | Notes |
|---|---|---|---|---|---|
| `generate_plots.py` | `Documents/LinkedIn Figures/` | 11.4 KB / 2026-07-15 | 234 / 175 / 15 | Offline matplotlib/statsmodels LinkedIn figures | Hardcoded `P90-HWs 9 stations_3.xlsm`. Not imported by the app |
| `backup_2026-08-25_code/` | 23 `*.py` + old `.bat` / `requirements.txt` | ~248 KB combined / mostly 2026-07–08-18 | ~4.7k total lines | Frozen pre-refactor snapshot (`synex_theme.py`, `run_pipeline.py`, `update_recent_data.py`, `compute_*.py`, Kelvin patches, etc.) | Not imported. `.gitignore` intends to exclude `backup_*/` |

---

## Section 2 — Standalone / orphaned scripts

**Import graph of the live app**

`app.py` → `backend_maps`, `backend_map_locations`, `backend_waves`, `backend_narrative`, `labels`, `atmopulse_theme`  
`backend_waves` → `backend_maps`, `atmopulse_theme`  
`backend_narrative` → `backend_waves`

**Operational pipeline** (`run_atmopulse_pipeline.bat`)

`aifs_ingestion.py` → `ifs_ingestion.py` → `run_era5_update.bat` → `era5_daily_updater.py`

### Dead in the live tree

| File | Why it is orphaned |
|---|---|
| `backend_time_series.py` | Nothing imports it. It only raises. README still says it powers location meteograms |
| `Documents/LinkedIn Figures/generate_plots.py` | Standalone Excel→PDF/PNG. No callers |
| `Downloading`, `Updating` | One-line status markers, unused |
| Entire `backup_2026-08-25_code/` | Snapshot of retired modules (`compute_climate_masks.py`, `compute_persistence.py`, `detect_extremes.py`, `run_pipeline.py`, `synex_theme.py`, `update_recent_data.py`, Kelvin patches) |

### Documented one-offs (not wired to app or daily `.bat`)

These are **not** imported, but README treats them as the historical rebuild kit. Outputs *are* consumed if the NetCDF files exist:

- `era5_phase1_synoptics.py`, `era5_phase2_2m-temp.py`, `era5_phase3_utci.py`, `era5_init_2026.py`
- `era5_climatology_builder.py`, `era5_synoptic_climatology.py`
- `evaluate_master.py`, `calculate_qdm_bias.py`
- `download_era5_invariants.py` (explicitly “run once”; app caches `era5_invariants.nc`)

---

## Section 3 — Critical hotspots and refactoring candidates (by urgency)

### 1. `app.py` as a 2.8k-line god module — **P0**

It owns Streamlit layout, geocoding, three NetCDF access paths, persistence cubes, Top-10 ranking, meteogram traces, QDM application, Folium, and cache keys. Cache coverage is actually **dense** (11 `cache_resource` + 11 `cache_data`), so the pain is cohesion, not missing decorators.

Highest-leverage splits: (a) archive I/O into `backend_maps` only, (b) point series / meteograms out of the UI file, (c) Top-10 / footprint next to `backend_map_locations`.

### 2. `importlib.reload` on every Streamlit rerun — **P0 (production)**

```49:61:app.py
# DEV SAFETY NET: Streamlit's autoreload only re-executes THIS script on
# save; it does NOT automatically re-import already-loaded local modules
...
importlib.reload(_backend_maps_mod)
importlib.reload(_backend_waves_mod)
importlib.reload(_backend_narrative_mod)
importlib.reload(_atmopulse_theme_mod)
```

This is a local-dev workaround that re-executes backend modules, re-binds functions, and can fight `@st.cache_*` identity. Gate it behind a debug flag; never leave it on for a public beta.

### 3. Triple read of the same yearly masters — **P0 (I/O / Windows HDF5)**

| Path | Strategy |
|---|---|
| `backend_maps._open_synoptic_range` | Nearby years, unchunked `open_dataset` |
| `app.get_master_archive_ds` | All historical years via `open_mfdataset` + current year isolated |
| `backend_waves._era5_master_point_txtn` | Loop every `era5_master_daily_YYYY.nc`; current year in a **child process** |

`app.py` and `backend_waves.py` each embed a nearly identical `-c` JSON extractor (`_POINT_EXTRACT_SCRIPT` / `_POINT_TXTN_EXTRACT_SCRIPT`) to survive HDF5 aborts on the in-progress year file. That is a real operational constraint, but the duplication means two subprocess taxonomies, two Kelvin heuristics, and two cache keys for the same grid cell.

Unify on one “safe point extract” + one lazy archive handle. Keep the subprocess fence in one place.

### 4. Near-duplicate IFS / AIFS ingestion — **P1**

`ifs_ingestion.py` (321 lines) and `aifs_ingestion.py` (357 lines) share client setup, cfgrib load, sparse CDO regrid, ERA5 grid target, and `doy_365`. Drift risk is high (AIFS lacks native TX/TN; the UI already special-cases this). Extract a shared `opendata_regrid.py`.

### 5. Undeclared geospatial stack + nested-loop Top-10 — **P1**

`backend_map_locations.py` imports `shapely` and `pyproj`; `backend_maps.py` falls back to `h5netcdf`. None of these are in `requirements.txt`. `build_country_weight_grid` walks every lat/lon in Python. First call is wrapped by `get_country_weight_grid` (`@st.cache_data`), so reruns are fine; **cold start** and cache eviction are not.

Also: `get_europe_borders_trace` and the location index both download Natural Earth GeoJSON from GitHub at runtime (`@st.cache_resource` / module singleton). Offline or GitHub-down = empty borders / failed Top-10.

### 6. Hardcoded machine paths and data roots — **P1**

Batch files pin `C:\Users\liina\miniconda3` and `C:\Users\liina\Andreas ERA5`. Almost every Python module repeats `Path("ERA5_ClimateTool/...")`. Climatology builders freeze `CUTOFF_DATE = "2026-07-31"` and epochs A/B in three files. One `paths.py` / env-based root would unlock portability and stop cutoff drift.

### 7. Climatology / ERA5T preprocess copied four ways — **P2**

`preprocess_era5t` + `get_window_doys` appear in `era5_climatology_builder.py` and `era5_synoptic_climatology.py`. `drop_era5t_aux` in `backend_maps` is a third variant. `harmonize_time` is copied across `era5_daily_updater.py` and `era5_init_2026.py`. Historical phase scripts share `DEFAULT_AREA`, logging, and 4-worker CDS patterns.

### 8. Module-level side effects — **P2**

- `ifs_ingestion` / `aifs_ingestion`: `mkdir` on import
- `app.py`: `ref_clim = load_reference_climatology()` and `border_trace = get_europe_borders_trace()` at import (README: app **stops** without climatology)
- `Nominatim(...)` constructed at import even for map-only sessions

### 9. Cache design debt (not “uncached”) — **P2**

Maps and point analytics **are** cached. Remaining issues:

- `backend_map_locations` itself is uncached (OK only because `app.py` wraps it).
- `_load_point_archive_series` still does `.sel(...).compute()` under a thread lock on a multi-year mfdataset — comments still call this “multi-minute”.
- `build_yearly_extremes_chart` uses `@st.cache_resource` for a Plotly figure (avoids deepcopy; figures are not invalidated the same way as `cache_data`).
- Version bump integers (`_harmonize_version=10`, `_archive_version=6`, `TOP10_GRID_VERSION = 5`) are manual cache busts scattered through signatures.

### 10. Requirements vs reality — **P2**

Unused in live code: `regionmask`, `openmeteo-requests`, `requests-cache`, `retry-requests` (Open-Meteo was removed). Used but missing: `shapely`, `pyproj`, possibly `h5netcdf`. Theme colors exist in both `.streamlit/config.toml` and `atmopulse_theme.py`.

---

### Architecture in one paragraph

AtmoPulse is a **Streamlit monolith** (`app.py`) over a **file-based climate warehouse** (`ERA5_ClimateTool/`, gitignored NetCDF). Ingestion is a **batch CLI belt** (CDS for ERA5, `ecmwf-opendata`+cfgrib for IFS/AIFS), not an in-process pipeline. The live app’s backends are reasonably split (maps / waves / narrative / theme / labels), but **I/O and analytics leaked back into `app.py`**, while waves reimplemented a second archive reader. Caching is mature on the UI path; the remaining cost is **duplicate NetCDF strategy**, **Windows HDF5 isolation**, **dev-only module reload**, and **undeclared geospatial dependencies**. Historical rebuild scripts are a separate, duplicated CDS toolkit and are correctly offline — they should stay out of the app import graph, but they should share path/ERA5T/ETCCDI helpers with the runtime.
