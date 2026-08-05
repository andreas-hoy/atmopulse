import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings

# Unterdrückt Warnungen für Ozean-Pixel (All-NaN)
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

DATA_DIR = Path("ERA5_ClimateTool/Master_Batches")
OUT_DIR = Path("ERA5_ClimateTool/Reference_Climatology")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "climatology_reference.nc"
TEMP_PROGRESS_FILE = OUT_DIR / "climatology_progress_temp.nc"

def build_climatology():
    print("🚀 Starte beschleunigten Climatology Builder (inkl. P5/P95 & Auto-Resume)...")
    
    txtn_files = sorted(list(DATA_DIR.glob("era5_txtn_batch_*.nc")))
    if not txtn_files:
        raise FileNotFoundError(f"Keine Batches gefunden in {DATA_DIR}")
    
    # 1. Metadaten lazily laden und sortieren
    print("Lese Dateistrukturen...")
    ds = xr.open_mfdataset(txtn_files, combine='nested', concat_dim='valid_time')
    ds = ds.sortby('valid_time')
    
    # Zeitachsen extrahieren
    times = pd.to_datetime(ds.valid_time.values)
    _, index = np.unique(times, return_index=True)
    ds = ds.isel(valid_time=index)
    times = pd.to_datetime(ds.valid_time.values)
    
    # Globale Arrays für extrem schnelles Suchen (ohne Xarray-Overhead)
    doys = times.dayofyear.values
    years = times.year.values
    
    current_year = times.max().year if times.max().month >= 12 else times.max().year - 1
    
    epoch_A_start, epoch_A_end = 1961, 1990
    epoch_B_start, epoch_B_end = current_year - 29, current_year
    
    lats = ds.latitude.values
    lons = ds.longitude.values
    n_lats, n_lons = len(lats), len(lons)
    
    # --- PRÜFEN OB EIN CHECKPOINT EXISTIERT ---
    start_doy = 1
    if TEMP_PROGRESS_FILE.exists():
        try:
            print("🔄 Bestehenden Fortschritt gefunden! Analysiere Checkpoint...")
            ds_clim = xr.open_dataset(TEMP_PROGRESS_FILE).load()
            
            # Prüfe, ob die neuen P5/P95 Variablen im alten Checkpoint existieren
            if "tx_p95_doy_A" not in ds_clim.data_vars:
                print("⚠️ Alter Checkpoint enthält keine P5/P95 Struktur. Starte sauber neu...")
                ds_clim.close()
                TEMP_PROGRESS_FILE.unlink(missing_ok=True)
                start_doy = 1
            else:
                test_var = ds_clim["tx_p90_doy_A"].values
                for d in range(366):
                    if np.all(np.isnan(test_var[d])):
                        start_doy = d + 1
                        break
                else:
                    start_doy = 367
            
            if start_doy <= 366 and "tx_p95_doy_A" in ds_clim.data_vars:
                print(f"▶️ Setze Berechnung nahtlos bei Tag {start_doy}/366 fort!")
            elif start_doy > 366:
                print("✅ Alle Tage bereits berechnet. Konsolidiere finale Datei...")
                ds_clim.close()
                if OUT_FILE.exists(): os.remove(OUT_FILE)
                os.rename(TEMP_PROGRESS_FILE, OUT_FILE)
                return
        except Exception as e:
            print(f"⚠️ Checkpoint beschädigt ({e}). Starte Berechnung neu...")
            TEMP_PROGRESS_FILE.unlink(missing_ok=True)
            start_doy = 1
            
    if start_doy == 1:
        print("🆕 Initialisiere neue Datenstruktur (inklusive P5 und P95)...")
        # Datensatz komplett mit P5 und P95 vorbereiten
        empty_3d = lambda: (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), np.nan, dtype=np.float32))
        
        ds_clim = xr.Dataset(
            coords={"dayofyear": np.arange(1, 367), "latitude": lats, "longitude": lons},
            data_vars={
                "tx_p75_doy_A": empty_3d(), "tx_p90_doy_A": empty_3d(), "tx_p95_doy_A": empty_3d(),
                "tn_p25_doy_A": empty_3d(), "tn_p10_doy_A": empty_3d(), "tn_p5_doy_A": empty_3d(),
                
                "tx_p75_doy_B": empty_3d(), "tx_p90_doy_B": empty_3d(), "tx_p95_doy_B": empty_3d(),
                "tn_p25_doy_B": empty_3d(), "tn_p10_doy_B": empty_3d(), "tn_p5_doy_B": empty_3d(),
                
                "tx_max_val": empty_3d(), "tx_max_year": (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), -1, dtype=np.int32)),
                "tn_min_val": empty_3d(), "tn_min_year": (("dayofyear", "latitude", "longitude"), np.full((366, n_lats, n_lons), -1, dtype=np.int32)),
            }
        )

    # --- TEIL 1: SAISONALE KONSTANTEN (NUR BEI START_DOY == 1) ---
    if start_doy == 1:
        print("\n--- TEIL 1: Berechne saisonale Konstanten für Wellen (JJA / DJF) ---")
        for ep_name, start_yr, end_yr in [("A", epoch_A_start, epoch_A_end), ("B", epoch_B_start, epoch_B_end)]:
            print(f"Lade JJA & DJF für Epoche {ep_name}...")
            
            # Masken auf NumPy-Ebene für Speed
            jja_idx = np.where((years >= start_yr) & (years <= end_yr) & (np.isin(times.month, [6, 7, 8])))[0]
            ds_jja = ds.isel(valid_time=jja_idx).compute()
            tx_jja = ds_jja['mx2t'].values - 273.15
            
            # Gebündelte Berechnung
            tx_pcts = np.nanpercentile(tx_jja, [75, 90, 95], axis=0)
            ds_clim[f"tx_p75_{ep_name}"] = (("latitude", "longitude"), tx_pcts[0])
            ds_clim[f"tx_p90_{ep_name}"] = (("latitude", "longitude"), tx_pcts[1])
            ds_clim[f"tx_p95_{ep_name}"] = (("latitude", "longitude"), tx_pcts[2])
            del ds_jja, tx_jja, tx_pcts
            
            djf_idx = np.where((years >= start_yr) & (years <= end_yr) & (np.isin(times.month, [12, 1, 2])))[0]
            ds_djf = ds.isel(valid_time=djf_idx).compute()
            tn_djf = ds_djf['mn2t'].values - 273.15
            
            tn_pcts = np.nanpercentile(tn_djf, [5, 10, 25], axis=0)
            ds_clim[f"tn_p5_{ep_name}"]  = (("latitude", "longitude"), tn_pcts[0])
            ds_clim[f"tn_p10_{ep_name}"] = (("latitude", "longitude"), tn_pcts[1])
            ds_clim[f"tn_p25_{ep_name}"] = (("latitude", "longitude"), tn_pcts[2])
            del ds_djf, tn_djf, tn_pcts

    # --- TEIL 2: GLEITENDE FENSTER (HIGH SPEED & CHECKPOINT) ---
    print("\n--- TEIL 2: Berechne tägliche 5-Tage-Fenster ---")
    
    # Referenzen aufbauen für direktes Schreiben
    v_tx_p75_A, v_tx_p90_A, v_tx_p95_A = ds_clim["tx_p75_doy_A"].values, ds_clim["tx_p90_doy_A"].values, ds_clim["tx_p95_doy_A"].values
    v_tn_p25_A, v_tn_p10_A, v_tn_p5_A  = ds_clim["tn_p25_doy_A"].values, ds_clim["tn_p10_doy_A"].values, ds_clim["tn_p5_doy_A"].values
    
    v_tx_p75_B, v_tx_p90_B, v_tx_p95_B = ds_clim["tx_p75_doy_B"].values, ds_clim["tx_p90_doy_B"].values, ds_clim["tx_p95_doy_B"].values
    v_tn_p25_B, v_tn_p10_B, v_tn_p5_B  = ds_clim["tn_p25_doy_B"].values, ds_clim["tn_p10_doy_B"].values, ds_clim["tn_p5_doy_B"].values
    
    v_tx_max_val, v_tx_max_yr = ds_clim["tx_max_val"].values, ds_clim["tx_max_year"].values
    v_tn_min_val, v_tn_min_yr = ds_clim["tn_min_val"].values, ds_clim["tn_min_year"].values

    for target_doy in range(start_doy, 367):
        print(f"  -> Verarbeite Tag {target_doy}/366... [Speed-Mode & Auto-Save]", end="\r", flush=True)
        
        window_doys = []
        for offset in range(-2, 3):
            d = target_doy + offset
            if d < 1: d += 366
            elif d > 366: d -= 366
            window_doys.append(d)
            
        # Hocheffizientes Integer-Indexing statt Boolean/Datetime Search
        win_idx = np.where(np.isin(doys, window_doys))[0]
        
        # Daten für diese Indizes physisch in RAM laden
        ds_win = ds.isel(valid_time=win_idx).compute()
        tx_win = ds_win['mx2t'].values - 273.15
        tn_win = ds_win['mn2t'].values - 273.15
        win_years = years[win_idx]
        
        idx = target_doy - 1
        
        # --- EPOCHE A ---
        mask_A = (win_years >= epoch_A_start) & (win_years <= epoch_A_end)
        if np.any(mask_A):
            # 3 Perzentile in einer einzigen Vektoroperation berechnen!
            pct_tx_A = np.nanpercentile(tx_win[mask_A], [75, 90, 95], axis=0)
            v_tx_p75_A[idx], v_tx_p90_A[idx], v_tx_p95_A[idx] = pct_tx_A[0], pct_tx_A[1], pct_tx_A[2]
            
            pct_tn_A = np.nanpercentile(tn_win[mask_A], [5, 10, 25], axis=0)
            v_tn_p5_A[idx], v_tn_p10_A[idx], v_tn_p25_A[idx] = pct_tn_A[0], pct_tn_A[1], pct_tn_A[2]
            
        # --- EPOCHE B ---
        mask_B = (win_years >= epoch_B_start) & (win_years <= epoch_B_end)
        if np.any(mask_B):
            pct_tx_B = np.nanpercentile(tx_win[mask_B], [75, 90, 95], axis=0)
            v_tx_p75_B[idx], v_tx_p90_B[idx], v_tx_p95_B[idx] = pct_tx_B[0], pct_tx_B[1], pct_tx_B[2]
            
            pct_tn_B = np.nanpercentile(tn_win[mask_B], [5, 10, 25], axis=0)
            v_tn_p5_B[idx], v_tn_p10_B[idx], v_tn_p25_B[idx] = pct_tn_B[0], pct_tn_B[1], pct_tn_B[2]
            
        # --- REKORDE ---
        max_idx = np.nanargmax(tx_win, axis=0)
        v_tx_max_val[idx] = np.take_along_axis(tx_win, np.expand_dims(max_idx, axis=0), axis=0).squeeze()
        yr_grid_tx = np.broadcast_to(win_years[:, None, None], tx_win.shape)
        v_tx_max_yr[idx] = np.take_along_axis(yr_grid_tx, np.expand_dims(max_idx, axis=0), axis=0).squeeze()
        
        min_idx = np.nanargmin(tn_win, axis=0)
        v_tn_min_val[idx] = np.take_along_axis(tn_win, np.expand_dims(min_idx, axis=0), axis=0).squeeze()
        yr_grid_tn = np.broadcast_to(win_years[:, None, None], tn_win.shape)
        v_tn_min_yr[idx] = np.take_along_axis(yr_grid_tn, np.expand_dims(min_idx, axis=0), axis=0).squeeze()

        # --- ZURÜCKSCHREIBEN IN DATASET ---
        ds_clim["tx_p75_doy_A"] = (("dayofyear", "latitude", "longitude"), v_tx_p75_A)
        ds_clim["tx_p90_doy_A"] = (("dayofyear", "latitude", "longitude"), v_tx_p90_A)
        ds_clim["tx_p95_doy_A"] = (("dayofyear", "latitude", "longitude"), v_tx_p95_A)
        ds_clim["tn_p25_doy_A"] = (("dayofyear", "latitude", "longitude"), v_tn_p25_A)
        ds_clim["tn_p10_doy_A"] = (("dayofyear", "latitude", "longitude"), v_tn_p10_A)
        ds_clim["tn_p5_doy_A"]  = (("dayofyear", "latitude", "longitude"), v_tn_p5_A)
        
        ds_clim["tx_p75_doy_B"] = (("dayofyear", "latitude", "longitude"), v_tx_p75_B)
        ds_clim["tx_p90_doy_B"] = (("dayofyear", "latitude", "longitude"), v_tx_p90_B)
        ds_clim["tx_p95_doy_B"] = (("dayofyear", "latitude", "longitude"), v_tx_p95_B)
        ds_clim["tn_p25_doy_B"] = (("dayofyear", "latitude", "longitude"), v_tn_p25_B)
        ds_clim["tn_p10_doy_B"] = (("dayofyear", "latitude", "longitude"), v_tn_p10_B)
        ds_clim["tn_p5_doy_B"]  = (("dayofyear", "latitude", "longitude"), v_tn_p5_B)

        ds_clim["tx_max_val"] = (("dayofyear", "latitude", "longitude"), v_tx_max_val)
        ds_clim["tx_max_year"] = (("dayofyear", "latitude", "longitude"), v_tx_max_yr)
        ds_clim["tn_min_val"] = (("dayofyear", "latitude", "longitude"), v_tn_min_val)
        ds_clim["tn_min_year"] = (("dayofyear", "latitude", "longitude"), v_tn_min_yr)

        # CHECKPOINT SPEICHERN (Atomares Schreiben gegen Korruption)
        TEMP_WRITE_FILE = OUT_DIR / "progress_step_temp.nc"
        ds_clim.to_netcdf(TEMP_WRITE_FILE)
        ds_clim.close() # Schließt den Speicherzugriff
        
        if TEMP_PROGRESS_FILE.exists(): TEMP_PROGRESS_FILE.unlink()
        os.rename(TEMP_WRITE_FILE, TEMP_PROGRESS_FILE)
        
        # Wieder öffnen für den nächsten Schleifendurchlauf
        ds_clim = xr.open_dataset(TEMP_PROGRESS_FILE).load()

    print("\n🎉 Alle 366 Tage inklusive P5/P95 erfolgreich abgeschlossen!")
    ds_clim.close()
    
    if OUT_FILE.exists(): os.remove(OUT_FILE)
    os.rename(TEMP_PROGRESS_FILE, OUT_FILE)
    print("✅ Klimatologie final, wasserdicht und extrem schnell auf der Festplatte gesichert!")

if __name__ == "__main__":
    build_climatology()