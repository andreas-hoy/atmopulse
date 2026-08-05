#!/usr/bin/env python3
import logging
import os
import sys
import time
from pathlib import Path
import cdsapi

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / "utci_download.log")]
    )

def main() -> int:
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=500, sleep_max=120)

    # 10-Jahres-Blöcke
    decades = [
        (2020, 2025), (2010, 2019), (2000, 2009),
        (1990, 1999), (1980, 1989), (1970, 1979),
        (1960, 1969), (1950, 1959), (1940, 1949),
    ]

    for start, end in decades:
        target = root / f"era5_utci_batch_{start}_{end}.nc"
        if target.exists():
            logging.info("SKIP | UTCI Block %s_%s existiert.", start, end)
            continue
            
        logging.info("START | UTCI Block %s bis %s", start, end)
        years = [str(y) for y in range(start, end + 1)]
        
        # UTCI aus dem daily statistics dataset abrufen
        req = {
            "product_type": "reanalysis",
            "variable": [
                "universal_thermal_climate_index" 
            ],
            "year": years,
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            # Tägliches Maximum und Minimum anfordern
            "daily_statistic": ["daily_maximum", "daily_minimum"],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": DEFAULT_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        part = target.with_suffix(".part")
        started = time.monotonic()
        try:
            # Wir nutzen hier den daily-statistics Endpunkt, analog zu TX/TN
            client.retrieve("derived-era5-single-levels-daily-statistics", req, str(part))
            os.replace(part, target)
            logging.info("DONE | UTCI Block %s_%s in %.1f min", start, end, (time.monotonic() - started) / 60)
        except Exception:
            part.unlink(missing_ok=True)
            logging.exception("FAILED | UTCI Block %s_%s", start, end)

    return 0

if __name__ == "__main__":
    sys.exit(main())