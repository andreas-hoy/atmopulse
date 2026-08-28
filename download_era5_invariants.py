#!/usr/bin/env python3
"""
AtmoPulse: ERA5 Time-Invariant Fields Downloader

Downloads the static ERA5 physiography fields used by the Point Meteogram /
Point Wavogram "Grid Cell Profile" expander: geopotential (-> elevation),
land-sea mask, sub-grid standard deviation of orography (-> roughness class),
and high/low vegetation cover.

These fields are time-invariant, so a single representative date/time is
sufficient. Requires the `cdsapi` package and a configured `~/.cdsapirc`
(see https://cds.climate.copernicus.eu/how-to-api).

This script is standalone and NOT executed automatically by AtmoPulse.
Run it manually once:

    python download_era5_invariants.py
"""

from pathlib import Path
import cdsapi

# Pan-European domain, consistent with the rest of the AtmoPulse pipeline.
AREA = [72.0, -25.0, 30.0, 45.0]  # North, West, South, East

OUTPUT_FILE = Path("ERA5_ClimateTool/Reference_Climatology/era5_invariants.nc")

REQUEST = {
    "product_type": "reanalysis",
    "variable": [
        "geopotential",
        "high_vegetation_cover",
        "land_sea_mask",
        "low_vegetation_cover",
        "standard_deviation_of_orography",
    ],
    # Time-invariant fields: any single historical date/time is representative.
    "year": "2024",
    "month": "01",
    "day": "01",
    "time": "12:00",
    "area": AREA,
    "format": "netcdf",
}


def download_era5_invariants():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", REQUEST, str(OUTPUT_FILE))

    print(f"✅ ERA5 invariant fields downloaded successfully to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    download_era5_invariants()
