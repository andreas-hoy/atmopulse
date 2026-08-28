"""
DEPRECATED — AtmoPulse no longer calls out to Open-Meteo.

`get_live_timeseries()` used to fetch the Point Meteogram's "current
conditions" series from api.open-meteo.com / archive-api.open-meteo.com.
That created a data-provenance mismatch: the classification thresholds
(P75/P90/P95/records) are computed from this app's own ERA5 archive, while
the value being classified came from a *different* model/provider's point
interpolation — the two are not guaranteed to agree on any given day.

Replacement: `get_live_point_series()` in app.py, built strictly from this
app's own ERA5 master archive (history) + its own IFS/AIFS live forecast
(future), with an optional QDM mean-bias correction on the forecast segment
(see calculate_qdm_bias.py). No external network calls.

This module is kept only so old imports fail loudly instead of silently
reintroducing an external dependency.
"""

def get_live_timeseries(lat: float, lon: float):
    raise RuntimeError(
        "get_live_timeseries() (Open-Meteo) has been removed. "
        "Use app.get_live_point_series(lat, lon, forecast_model) instead."
    )
