import pandas as pd
import requests

def get_live_timeseries(lat: float, lon: float) -> pd.DataFrame:
    """
    Holt die Daten (Historie 365 Tage + Forecast 6 Tage) für die gewählten Koordinaten.
    Splittet die Abfrage in Archive-API (für Vergangenheit) und Forecast-API (für Zukunft),
    um das 90-Tage-Limit von Open-Meteo elegant zu umgehen.
    """
    
    # 1. Historische Daten (Vor 365 Tagen bis Gestern)
    end_hist = pd.Timestamp.utcnow().tz_localize(None).floor('D') - pd.Timedelta(days=1)
    start_hist = end_hist - pd.Timedelta(days=364)
    
    url_hist = "https://archive-api.open-meteo.com/v1/archive"
    params_hist = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_hist.strftime('%Y-%m-%d'),
        "end_date": end_hist.strftime('%Y-%m-%d'),
        "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min"],
        "timezone": "UTC"
    }
    
    # 2. Vorhersage Daten (Heute + 6 Tage in die Zukunft)
    url_fcst = "https://api.open-meteo.com/v1/forecast"
    params_fcst = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min"],
        "timezone": "UTC",
        "past_days": 0,
        "forecast_days": 7
    }
    
    try:
        # Beide APIs parallel abfragen
        resp_hist = requests.get(url_hist, params=params_hist).json()
        resp_fcst = requests.get(url_fcst, params=params_fcst).json()
        
        df_hist = pd.DataFrame(resp_hist.get('daily', {}))
        df_fcst = pd.DataFrame(resp_fcst.get('daily', {}))
        
        # Daten nahtlos verknüpfen
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