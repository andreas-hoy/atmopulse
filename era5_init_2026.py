#!/usr/bin/env python3
"""Initial 2026 ERA5 master-file download (TX, TN, TG, MSLP, T850, Z500, jets).

Issues decoupled CDS requests so temperature fields do not fall back to ZIP
archives. Synoptic drivers are 12 UTC; surface TX/TN/TG use hourly 00–23 UTC
then calendar-day resample (max / min / mean) with Kelvin→Celsius conversion.
Writes ``era5_master_daily_2026.nc`` only if that file does not yet exist.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import xarray as xr

DEFAULT_AREA = [72.0, -25.0, 30.0, 45.0]


def setup_logging(root: Path) -> None:
    """Configure INFO logging to stdout; ensure the output directory exists."""
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def harmonize_time(ds: xr.Dataset) -> xr.Dataset:
    """Rename ``valid_time`` dimension/coordinate to CF ``time``."""
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _collapse_aux_for_agg(da: xr.DataArray) -> xr.DataArray:
    """Drop ERA5T expver/number so daily resample cannot broadcast to all-NaN."""
    if "expver" in da.dims:
        da = da.dropna(dim="expver", how="all")
        if "expver" in da.dims:
            reduce_dims = [d for d in da.dims if d != "expver"]
            n_finite = da.notnull().sum(dim=reduce_dims)
            da = da.isel(expver=int(n_finite.argmax().item()), drop=True)
    if "number" in da.dims and da.sizes.get("number", 0) == 1:
        da = da.isel(number=0, drop=True)
    for name in ("expver", "number"):
        if name in da.coords:
            da = da.drop_vars(name, errors="ignore")
    return da


def _daily_on_index(da: xr.DataArray, how: str, target_time) -> xr.DataArray:
    """Calendar-day stat reindexed onto the synoptic (already-floored) time axis."""
    da = _collapse_aux_for_agg(da)
    agg = getattr(da.resample(time="1D"), how)()
    agg = agg.assign_coords(time=agg["time"].dt.floor("D"))
    return agg.reindex(time=target_time)


def main() -> None:
    """Download 2026 ERA5 fields from 1 January through today minus 5 days."""
    root = Path("ERA5_ClimateTool/Master_Batches").resolve()
    setup_logging(root)
    client = cdsapi.Client(wait_until_complete=True, retry_max=50, sleep_max=120)

    target_year = 2026
    end_date = datetime.utcnow() - timedelta(days=5)

    start_date_str = f"{target_year}-01-01"
    end_date_str = end_date.strftime("%Y-%m-%d")
    date_string = f"{start_date_str}/{end_date_str}"

    master_file = root / f"era5_master_daily_{target_year}.nc"
    if master_file.exists():
        logging.error(
            "Master file %s already exists. Delete it or use the update script.",
            master_file.name,
        )
        sys.exit(1)

    logging.info(
        "START INIT DOWNLOAD FOR %s | period: %s", target_year, date_string
    )
    tmp_dir = root / ".tmp_init"
    tmp_dir.mkdir(exist_ok=True)

    try:
        logging.info("Requesting synoptics (MSLP)...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "mean_sea_level_pressure",
                "date": date_string,
                "time": "12:00",
                "area": DEFAULT_AREA,
                "format": "netcdf",
            },
            str(tmp_dir / "syn_mslp.nc"),
        )

        logging.info("Requesting synoptics (pressure levels)...")
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                ],
                "pressure_level": ["300", "500", "850"],
                "date": date_string,
                "time": "12:00",
                "area": DEFAULT_AREA,
                "format": "netcdf",
            },
            str(tmp_dir / "syn_press.nc"),
        )

        # Surface extremes (00–23 UTC), decoupled requests.
        base_req = {
            "product_type": "reanalysis",
            "date": date_string,
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": DEFAULT_AREA,
            "format": "netcdf",
        }

        logging.info("Requesting temperature (TG)...")
        req_tg = base_req.copy()
        req_tg["variable"] = "2m_temperature"
        client.retrieve(
            "reanalysis-era5-single-levels", req_tg, str(tmp_dir / "t2m.nc")
        )

        logging.info("Requesting temperature (TX)...")
        req_tx = base_req.copy()
        req_tx["variable"] = (
            "maximum_2m_temperature_since_previous_post_processing"
        )
        client.retrieve(
            "reanalysis-era5-single-levels", req_tx, str(tmp_dir / "mx2t.nc")
        )

        logging.info("Requesting temperature (TN)...")
        req_tn = base_req.copy()
        req_tn["variable"] = (
            "minimum_2m_temperature_since_previous_post_processing"
        )
        client.retrieve(
            "reanalysis-era5-single-levels", req_tn, str(tmp_dir / "mn2t.nc")
        )

        logging.info("Processing and converting data in RAM...")
        with xr.open_dataset(tmp_dir / "syn_mslp.nc") as ds_mslp, \
             xr.open_dataset(tmp_dir / "syn_press.nc") as ds_press, \
             xr.open_dataset(tmp_dir / "t2m.nc") as ds_t2m, \
             xr.open_dataset(tmp_dir / "mx2t.nc") as ds_mx2t, \
             xr.open_dataset(tmp_dir / "mn2t.nc") as ds_mn2t:

            ds_mslp = harmonize_time(ds_mslp.load())
            ds_press = harmonize_time(ds_press.load())
            ds_t2m = harmonize_time(ds_t2m.load())
            ds_mx2t = harmonize_time(ds_mx2t.load())
            ds_mn2t = harmonize_time(ds_mn2t.load())

            ds_out = xr.Dataset({"mslp": ds_mslp["msl"].astype("float32")})
            ds_out["z500"] = (
                ds_press["z"].sel(pressure_level=500, drop=True).astype("float32")
            )
            ds_out["t850"] = (
                ds_press["t"].sel(pressure_level=850, drop=True) - 273.15
            ).astype("float32")
            ds_out["u300"] = (
                ds_press["u"].sel(pressure_level=300, drop=True).astype("float32")
            )
            ds_out["v300"] = (
                ds_press["v"].sel(pressure_level=300, drop=True).astype("float32")
            )
            if "pressure_level" in ds_out.coords:
                ds_out = ds_out.drop_vars("pressure_level")

            ds_out["time"] = ds_out["time"].dt.floor("D")
            ds_out["tx"] = (
                _daily_on_index(ds_mx2t["mx2t"], "max", ds_out.time) - 273.15
            ).astype("float32")
            ds_out["tn"] = (
                _daily_on_index(ds_mn2t["mn2t"], "min", ds_out.time) - 273.15
            ).astype("float32")
            ds_out["tg"] = (
                _daily_on_index(ds_t2m["t2m"], "mean", ds_out.time) - 273.15
            ).astype("float32")

            for var in ["tx", "tn", "tg", "t850"]:
                ds_out[var].attrs["units"] = "Celsius"

            encoding = {
                v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_out.data_vars
            }
            ds_out.to_netcdf(master_file, encoding=encoding)
            logging.info("INITIALISATION COMPLETE: %s", master_file.name)

    except Exception as exc:
        logging.error("ERROR: %s", exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
