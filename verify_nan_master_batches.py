import xarray as xr
from pathlib import Path

# Pfad aus deiner V4-Architektur übernehmen
root_dir = Path("ERA5_ClimateTool/Master_Batches").resolve()
years_to_check = range(2010, 2026)

print("=== NaN Diagnose für ERA5 Master-Batches (2010-2025) ===")

for year in years_to_check:
    file_path = root_dir / f"era5_master_daily_{year}.nc"
    
    if not file_path.exists():
        print(f"[{year}] ⚠️ Übersprungen: Datei nicht gefunden.")
        continue
        
    try:
        # load() zwingt die Daten in den RAM, wichtig für korrekte NaN-Zählung
        with xr.open_dataset(file_path) as ds:
            ds = ds.load() 
            
            has_nans = ds.isnull().any().to_array().any().item()
            
            if not has_nans:
                print(f"[{year}] ✅ Sauber. Keine NaNs gefunden.")
            else:
                print(f"[{year}] ❌ ALARM: NaNs detektiert!")
                nan_counts = ds.isnull().sum(dim=ds.dims)
                for var in ds.data_vars:
                    count = nan_counts[var].item()
                    if count > 0:
                        print(f"    -> {var}: {count} NaNs")
    except Exception as e:
        print(f"[{year}] ⚠️ Fehler beim Lesen der Datei: {e}")