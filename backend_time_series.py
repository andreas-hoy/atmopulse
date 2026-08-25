"""
AtmoPulse Local Time-Series Extraction (backend_time_series.py)

This module handles the retrieval of location-specific meteorological time-series.
It bridges historical data with operational forecasts by cleanly splitting the 
API requests, ensuring continuous data availability for AtmoPulse point meteograms.
"""

import pandas as pd
import requests

def get_live_timeseries(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetches data (365 days history + 6 days forecast) for the selected coordinates.
    Splits the query into Archive API (for the past) and Forecast API (for the future)
    to elegantly bypass the 90-day limit of the Open-Meteo service.
    """
    
    # 1. Historical Data (From 365 days ago until yesterday)
    end_hist = pd.Timestamp.utcnow().tz_localize(None).floor('D') - pd.Timedelta(days=1)
    start_hist = end_hist - pd.Timedelta(days=364)
    
    url_hist = "https://archive-api.open-meteo.com/v1/archive"
    params_hist = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_hist.strftime('%Y-%m-%d'),
        "end_date": end_hist.strftime('%Y-%m-%d'),
        "daily": [
            "temperature_2m_max", 
            "temperature_2m_min", 
            "apparent_temperature_max", 
            "apparent_temperature_min"
        ],
        "timezone": "UTC"
    }
    
    # 2. Forecast Data (Today + 6 days into the future)
    url_fcst = "https://api.open-meteo.com/v1/forecast"
    params_fcst = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max", 
            "temperature_2m_min", 
            "apparent_temperature_max", 
            "apparent_temperature_min"
        ],
        "timezone": "UTC",
        "past_days": 0,
        "forecast_days": 7
    }
    
    try:
        # Query both APIs in parallel
        # Added a 15-second timeout to prevent infinite hanging of the web app
        resp_hist = requests.get(url_hist, params=params_hist, timeout=15).json()
        resp_fcst = requests.get(url_fcst, params=params_fcst, timeout=15).json()
        
        df_hist = pd.DataFrame(resp_hist.get('daily', {}))
        df_fcst = pd.DataFrame(resp_fcst.get('daily', {}))
        
        # Seamlessly merge data
        df = pd.concat([df_hist, df_fcst], ignore_index=True).drop_duplicates(subset=['time'])
        df = df.rename(columns={
            "time": "Date",
            "temperature_2m_max": "TX",
            "temperature_2m_min": "TN",
            "apparent_temperature_max": "AT_Max",
            "apparent_temperature_min": "AT_Min"
        })
        return df
    except Exception as e:
        print(f"API Error: {e}")
        return pd.DataFrame()