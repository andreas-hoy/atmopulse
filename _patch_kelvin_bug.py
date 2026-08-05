"""
Fast in-place fix for the Kelvin/Celsius unit bug in climatology_reference_complete.nc.

era5_climatology_updater.py wrote these 12 variables in Kelvin (missing -273.15):
  tx_p5/10/25_doy_A/B  (cold-side TX percentiles)
  tn_p75/90/95_doy_A/B (warm-side TN percentiles)

All other percentile variables were already correct (Celsius) from
era5_climatology_builder.py.  Subtracting 273.15 is equivalent to the full
recomputation and completes in seconds instead of hours.
"""
import xarray as xr
from pathlib import Path

CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
KELVIN_VARS = [
    "tx_p5_doy_A", "tx_p10_doy_A", "tx_p25_doy_A",
    "tx_p5_doy_B", "tx_p10_doy_B", "tx_p25_doy_B",
    "tn_p75_doy_A", "tn_p90_doy_A", "tn_p95_doy_A",
    "tn_p75_doy_B", "tn_p90_doy_B", "tn_p95_doy_B",
]

print(f"1. Oeffne {CLIM_FILE}...", flush=True)
with xr.open_dataset(CLIM_FILE) as ds:
    out = ds.load()

print("2. Konvertiere Kelvin -> Celsius (-273.15)...", flush=True)
for var in KELVIN_VARS:
    if var not in out:
        print(f"   WARNUNG: {var} nicht gefunden, ueberspringe.", flush=True)
        continue
    sample = float(out[var].isel(dayofyear=209, latitude=80, longitude=140).values)
    if sample > 100:
        out[var] = out[var] - 273.15
        print(f"   {var}: korrigiert (war ~{sample:.1f} K)", flush=True)
    else:
        print(f"   {var}: bereits Celsius (~{sample:.1f}), ueberspringe.", flush=True)

TMP = CLIM_FILE.with_suffix(".tmp.nc")
print("3. Speichere...", flush=True)
out.to_netcdf(TMP)
out.close()
TMP.replace(CLIM_FILE)
print("Fertig! climatology_reference_complete.nc korrigiert.", flush=True)
