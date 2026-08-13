"""
Fast in-place fix for the Kelvin/Celsius unit bug in climatology_reference_complete.nc.

era5_climatology_updater.py wrote these 12 variables in Kelvin (missing -273.15):
  tx_p5/10/25_doy_A/B  (cold-side TX percentiles)
  tn_p75/90/95_doy_A/B (warm-side TN percentiles)

All other percentile variables were already correct (Celsius) from
era5_climatology_builder.py.  Subtracting 273.15 is equivalent to the full
recomputation and completes in seconds instead of hours.

NOTE: the bug was NOT a clean "whole variable is in Kelvin" issue - the
updater ran in per-day-of-year chunks, so individual (dayofyear, lat, lon)
cells can be stuck in Kelvin while neighbouring cells were already fixed.
A single sample-point check (as before) only catches the case where the
*entire* variable is Kelvin; it silently misses partially-corrupted arrays
and leaves scattered ~270-300 "Kelvin" cells behind. Those cells then get
compared (in Celsius) in the map-rendering code and always evaluate as an
extreme cold/warm "record" pixel, i.e. the isolated implausible pixels seen
at Gibraltar/Turkey on the Synoptic Map. Fixing this per-cell (not per
sample point) resolves that.
"""
import os
import shutil
import numpy as np
import xarray as xr
from pathlib import Path

CLIM_FILE = Path("ERA5_ClimateTool/Reference_Climatology/climatology_reference_complete.nc")
KELVIN_VARS = [
    "tx_p5_doy_A", "tx_p10_doy_A", "tx_p25_doy_A",
    "tx_p5_doy_B", "tx_p10_doy_B", "tx_p25_doy_B",
    "tn_p75_doy_A", "tn_p90_doy_A", "tn_p95_doy_A",
    "tn_p75_doy_B", "tn_p90_doy_B", "tn_p95_doy_B",
]
# Physically, no ERA5 TX/TN percentile on Earth exceeds ~70 C. Any cell above
# this can only be a leftover un-converted Kelvin value (~200-320 K range).
KELVIN_CELL_THRESHOLD = 100.0

print(f"1. Oeffne {CLIM_FILE}...", flush=True)
with xr.open_dataset(CLIM_FILE) as ds:
    out = ds.load()

print("2. Konvertiere Kelvin -> Celsius (-273.15), pro Gitterzelle...", flush=True)
changed = False
for var in KELVIN_VARS:
    if var not in out:
        print(f"   WARNUNG: {var} nicht gefunden, ueberspringe.", flush=True)
        continue
    data = out[var]
    # skipna=True (default for xarray reductions) so NaN/land-sea mask holes
    # don't get treated as "not Kelvin" and don't blow up the max() check.
    still_kelvin = data > KELVIN_CELL_THRESHOLD
    n_bad = int(still_kelvin.sum(skipna=True).values)
    if n_bad > 0:
        out[var] = xr.where(still_kelvin, data - 273.15, data, keep_attrs=True)
        changed = True
        total = int(still_kelvin.size)
        print(f"   {var}: {n_bad}/{total} Zellen korrigiert (isoliert im Kelvin verblieben).", flush=True)
    else:
        print(f"   {var}: bereits vollstaendig Celsius, ueberspringe.", flush=True)

if not changed:
    out.close()
    TMP = CLIM_FILE.with_suffix(".tmp.nc")
    if TMP.exists():
        TMP.unlink()
    print("Keine Kelvin-Zellen gefunden — Datei unveraendert, Speichern uebersprungen.", flush=True)
    print("Fertig!", flush=True)
    raise SystemExit(0)

TMP = CLIM_FILE.with_suffix(".tmp.nc")
print("3. Speichere...", flush=True)
out.to_netcdf(TMP)
out.close()

try:
    os.replace(TMP, CLIM_FILE)
except PermissionError:
    # Windows: target is often locked by an open Streamlit/xarray handle.
    fallback = CLIM_FILE.with_name(CLIM_FILE.stem + "_patched.nc")
    if fallback.exists():
        fallback.unlink()
    shutil.move(str(TMP), str(fallback))
    print(
        f"\nFEHLER: {CLIM_FILE.name} ist gesperrt (Streamlit/App laeuft?).\n"
        f"Patch gespeichert als: {fallback}\n"
        f"Bitte App schliessen, dann:\n"
        f"  1. {CLIM_FILE.name} loeschen oder umbenennen\n"
        f"  2. {fallback.name} -> {CLIM_FILE.name} umbenennen\n"
        f"  3. App neu starten",
        flush=True,
    )
    raise SystemExit(1)

print("Fertig! climatology_reference_complete.nc korrigiert.", flush=True)
