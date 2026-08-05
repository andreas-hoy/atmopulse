import os
import time
from pathlib import Path

# Wir importieren unsere eigenen Skripte! 
# (Voraussetzung: Sie liegen im selben Ordner wie diese Datei)
from compute_climate_masks import compute_complete_masks
from detect_extremes import detect_daily_anomalies
from compute_persistence import calculate_persistence_tracks
from spatial_aggregation import aggregate_by_country

# --- KONFIGURATION ---
BASE_DIR = "C:/Users/liina/Andreas ERA5/ERA5_ClimateTool"

# Wir definieren die beiden Baselines
BASELINES = {
    "historical_1961_1990": range(1961, 1991),
    "modern_1996_2025": range(1996, 2026)
}

def build_baseline_mask(baseline_name: str, years: range):
    """Sammelt alle Jahre einer Baseline und berechnet die 30-Jahres-Maske"""
    print(f"\n{'='*50}\nPrüfe Baseline: {baseline_name} ({years.start}-{years.stop-1})")
    
    tx_files = [f"{BASE_DIR}/TX_daily/era5_tx_daily_{y}.nc" for y in years]
    tn_files = [f"{BASE_DIR}/TN_daily/era5_tn_daily_{y}.nc" for y in years]
    
    # Prüfen, ob schon alle Daten heruntergeladen wurden
    missing = [f for f in tx_files + tn_files if not os.path.exists(f)]
    if missing:
        print(f"-> Überspringe Berechnung. Es fehlen noch {len(missing)/2:.0f} Jahre (Downloader läuft noch).")
        return False
        
    out_file = f"{BASE_DIR}/Reference_Masks/mask_{baseline_name}.nc"
    if os.path.exists(out_file):
        print(f"-> Maske existiert bereits: {out_file}")
        return True
        
    print(f"-> Alle 30 Jahre vorhanden! Starte Mammut-Berechnung für {baseline_name}...")
    # Wir übergeben die Listen an unser modifiziertes Skript
    compute_complete_masks(tx_files, tn_files, out_file)
    return True

def process_target_year(year: int, baseline_name: str):
    """Jagt ein einzelnes Jahr durch die gesamte Detektor- und Länder-Pipeline"""
    print(f"\n{'='*50}\nStarte Event-Pipeline für Jahr {year} (Referenz: {baseline_name})")
    
    tx_file = f"{BASE_DIR}/TX_daily/era5_tx_daily_{year}.nc"
    tn_file = f"{BASE_DIR}/TN_daily/era5_tn_daily_{year}.nc"
    ref_mask = f"{BASE_DIR}/Reference_Masks/mask_{baseline_name}.nc"
    
    if not (os.path.exists(tx_file) and os.path.exists(ref_mask)):
        print("-> Fehler: Quelldaten oder Referenzmaske fehlen.")
        return
        
    # Dateipfade für die Zwischen- und Endergebnisse
    events_file = f"{BASE_DIR}/Detected_Events/events_{year}_vs_{baseline_name}.nc"
    pers_file = f"{BASE_DIR}/Persistence_Data/persistence_{year}_vs_{baseline_name}.nc"
    stats_csv = f"{BASE_DIR}/Aggregated_Stats/country_stats_{year}_vs_{baseline_name}.csv"
    
    # 1. Anomalien detektieren (Phase 1)
    detect_daily_anomalies(tx_file, tn_file, ref_mask, events_file)
    
    # 2. Persistenz berechnen (Phase 2)
    calculate_persistence_tracks(events_file, pers_file)
    
    # 3. Länder-Statistiken generieren (Phase 3)
    aggregate_by_country(pers_file, stats_csv)
    
    print(f"\n+++ Jahr {year} komplett verarbeitet! Daten bereit für Streamlit. +++")

if __name__ == "__main__":
    total_start = time.time()
    
    # SCHRITT 1: Baselines bauen (Überspringt automatisch, solange Downloader läuft)
    for name, year_range in BASELINES.items():
        build_baseline_mask(name, year_range)
        
    # SCHRITT 2: Manueller Testlauf explizit für den Monat 2026-01
    test_mask = f"{BASE_DIR}/Reference_Masks/klimamasken_januar_test.nc"
    tx_test = f"{BASE_DIR}/TX_daily_known_1h_issue/era5_tx_daily_2026-01.nc"
    tn_test = f"{BASE_DIR}/TN_daily_known_1h_issue/era5_tn_daily_2026-01.nc"
    
    if os.path.exists(test_mask) and os.path.exists(tx_test):
        print("\n==================================================")
        print("Starte Event-Pipeline für Test-Monat 2026-01")
        
        events_file = f"{BASE_DIR}/Detected_Events/events_2026-01_test.nc"
        pers_file = f"{BASE_DIR}/Persistence_Data/persistence_2026-01_test.nc"
        stats_csv = f"{BASE_DIR}/Aggregated_Stats/country_stats_2026-01_test.csv"
        
        # Direktes Aufrufen der 3 Module
        detect_daily_anomalies(tx_test, tn_test, test_mask, events_file)
        calculate_persistence_tracks(events_file, pers_file)
        aggregate_by_country(pers_file, stats_csv)
        
        print("\n+++ Testmonat 2026-01 komplett verarbeitet! +++")
    else:
        print("Testdaten nicht gefunden.")
        
    print(f"\nPipeline-Lauf beendet nach {(time.time()-total_start):.2f} Sekunden.")