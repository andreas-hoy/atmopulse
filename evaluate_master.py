#!/usr/bin/env python3
"""Evaluate ERA5 master daily NetCDF files for CF time, variables, and thermodynamics.

Opens each ``era5_master_daily_YYYY.nc`` and reports whether the time
coordinate follows CF convention (``time``, not ``valid_time``), which
surface variables are present, and whether TX/TN/TG obey
``TN <= TG <= TX`` at a fixed European grid point (near Pärnu,
58.5°N, 24.5°E).

This is a diagnostic only. Daily master timestamps cannot prove whether
TX/TN/TG were aggregated on a 00–00 UTC or 23–23 UTC window; that
requires the hourly source series. After a 00–00 UTC re-download, use
this script to confirm units, completeness, and physical consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

MASTER_DIR = Path("ERA5_ClimateTool/Master_Batches")
CHECK_LAT = 58.5
CHECK_LON = 24.5
CELSIUS_MAX_PLAUSIBLE = 100.0


def _inferred_unit(values: np.ndarray) -> str:
    """Infer Kelvin vs Celsius from magnitude (diagnostic label only)."""
    peak = float(np.nanmax(values))
    return "K" if peak > CELSIUS_MAX_PLAUSIBLE else "°C"


def evaluate_master_file(file_path: Path) -> None:
    """Run CF, variable, and thermodynamic checks on one master file."""
    print("=" * 50)
    print(f"EVALUATION: {file_path.name}")
    print("=" * 50)

    if not file_path.exists():
        print(f"ERROR: file not found: {file_path}")
        return

    try:
        with xr.open_dataset(file_path) as ds:
            print("\n1) DIMENSIONS & METADATA")
            dims = list(ds.dims)
            if "valid_time" in dims or "valid_time" in ds.coords:
                print("CRITICAL ERROR: 'valid_time' is still present!")
            elif "time" not in dims:
                print("CRITICAL ERROR: dimension 'time' is missing!")
            else:
                print("CF standard 'time' is in place.")
                hours = np.unique(ds["time"].dt.hour.values)
                t0 = np.datetime_as_string(ds["time"].values[0], unit="h")
                t1 = np.datetime_as_string(ds["time"].values[-1], unit="h")
                print(f"  time range: {t0} → {t1}")
                print(f"  unique hours on the time axis: {hours.tolist()}")
                print(
                    "  note: daily labels (usually 00:00) do not reveal "
                    "whether the source aggregation was 00–00 or 23–23 UTC."
                )

            print("\n2) VARIABLE INTEGRITY")
            vars_in_file = list(ds.data_vars)
            print(f"Variables present: {vars_in_file}")

            has_tx_tn = "tx" in vars_in_file and "tn" in vars_in_file
            has_tg = "tg" in vars_in_file

            print(
                f"\n3) THERMODYNAMICS (grid point near Pärnu: "
                f"{CHECK_LAT}°N, {CHECK_LON}°E)"
            )
            ds_parnu = ds.sel(
                latitude=CHECK_LAT, longitude=CHECK_LON, method="nearest"
            )

            if has_tx_tn:
                tx_raw = ds_parnu["tx"].values
                tn_raw = ds_parnu["tn"].values
                unit = _inferred_unit(tx_raw)

                print("Annual extrema at check point:")
                print(f"  -> Max TX:  {np.nanmax(tx_raw):.2f} {unit}")
                print(f"  -> Min TN:  {np.nanmin(tn_raw):.2f} {unit}")

                inconsistent_tx_tn = np.sum(tx_raw < tn_raw)
                if inconsistent_tx_tn > 0:
                    print(
                        f"PHYSICAL ERROR: TN > TX on "
                        f"{inconsistent_tx_tn} days!"
                    )
                else:
                    print("Thermodynamics intact: TX >= TN on all days.")
            else:
                print("ERROR: tx or tn missing; cannot run the check!")
                tx_raw = tn_raw = None

            if has_tg:
                tg_raw = ds_parnu["tg"].values
                unit = _inferred_unit(tg_raw)
                print(f"  -> Mean TG: {np.nanmean(tg_raw):.2f} {unit}")

                if tx_raw is None or tn_raw is None:
                    print("WARNING: cannot test TN <= TG <= TX without tx/tn.")
                else:
                    inconsistent_tg = np.sum(
                        (tg_raw > tx_raw) | (tg_raw < tn_raw)
                    )
                    if inconsistent_tg > 0:
                        print(
                            f"WARNING: TG outside TX/TN on "
                            f"{inconsistent_tg} days!"
                        )
                    else:
                        print(
                            "Mean-value logic intact: "
                            "TN <= TG <= TX on all days."
                        )
            else:
                print("NOTE: 'tg' is not yet in this file (Pass 2 pending).")

    except Exception as exc:
        print(f"Critical error during evaluation: {exc}")


def main() -> None:
    """Evaluate every master daily file under MASTER_DIR."""
    files = sorted(MASTER_DIR.glob("era5_master_daily_*.nc"))
    if not files:
        print(f"No master files found in {MASTER_DIR.resolve()}")
        return
    for file_path in files:
        evaluate_master_file(file_path)


if __name__ == "__main__":
    main()
