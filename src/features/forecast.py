"""
Forecast Module - Fetches weather AND pollutant forecast from Open-Meteo
Forecasts start from TODAY'S MIDNIGHT for consistent daily predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
from src.database import get_db_client
import src.config as config

# PKT timezone (UTC+5)
PKT = timezone(timedelta(hours=5))


def get_pkt_now():
    """Get current datetime in PKT timezone."""
    return datetime.now(PKT)


def fetch_weather_forecast_from_mongo():
    """
    Fetch weather + pollutant forecast from MongoDB (stored by ingest_data.py).
    Returns 72 hours starting from TODAY'S MIDNIGHT.
    """
    print("[INFO] Fetching forecast from MongoDB...")
    
    db = get_db_client()
    collection = db["weather_forecast"]
    
    # Get today's midnight in PKT
    now = get_pkt_now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    
    # Fetch forecast data from today's midnight
    data = list(collection.find(
        {"time": {"$gte": today_midnight}},
        {"_id": 0}
    ).sort("time", 1).limit(96))
    
    if not data:
        print("[WARNING] No forecast data in MongoDB, fetching fresh from API...")
        return fetch_weather_forecast_from_api()
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    
    print(f"[SUCCESS] Loaded {len(df)} hours of forecast from MongoDB")
    print(f"   Starts: {df['time'].min().strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"   Ends: {df['time'].max().strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def fetch_weather_forecast_from_api(hours=96):
    """
    Fetch weather AND pollutant forecast directly from Open-Meteo API.
    Fallback when MongoDB doesn't have recent forecast.
    """
    now = get_pkt_now()
    # Start from TODAY's midnight
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    
    print(f"[INFO] Fetching {hours}-hour forecast from API starting {today_midnight.strftime('%Y-%m-%d %H:%M')}...")
    
    # 1. Fetch WEATHER forecast
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                   "wind_speed_10m", "weather_code"],
        "forecast_days": 4, # 4 days = 96 hours
        "timezone": "Asia/Karachi"
    }
    
    try:
        w_resp = requests.get(w_url, params=w_params, timeout=30).json()
        
        if "error" in w_resp:
            print(f"[ERROR] Weather Forecast API Error: {w_resp.get('reason', 'Unknown')}")
            return None
        
        df = pd.DataFrame(w_resp['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        
        # Rename columns
        df.rename(columns={
            'temperature_2m': 'temp',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'rain',
            'wind_speed_10m': 'wind_speed'
        }, inplace=True)
        
    except Exception as e:
        print(f"[ERROR] Weather API fetch failed: {e}")
        return None
    
    # 2. Fetch POLLUTANT forecast + US AQI
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["pm10", "nitrogen_dioxide", "ozone", "us_aqi"],  # Added us_aqi
        "forecast_days": 4,
        "timezone": "Asia/Karachi"
    }
    
    try:
        a_resp = requests.get(a_url, params=a_params, timeout=30).json()
        if "error" not in a_resp:
            df_air = pd.DataFrame(a_resp['hourly'])
            df_air['time'] = pd.to_datetime(df_air['time'])
            df_air.rename(columns={'nitrogen_dioxide': 'no2'}, inplace=True)
            
            # Merge with weather data
            df = df.merge(df_air, on='time', how='left')
            
        # Fill any missing pollutant values
        df['pm10'] = df.get('pm10', pd.Series([50.0] * len(df))).fillna(50.0)
        df['no2'] = df.get('no2', pd.Series([15.0] * len(df))).fillna(15.0)
        df['ozone'] = df.get('ozone', pd.Series([70.0] * len(df))).fillna(70.0)
        
    except Exception as e:
        print(f"[WARNING] Pollutant API fetch failed: {e}")
        df['pm10'] = 50.0
        df['no2'] = 15.0
        df['ozone'] = 70.0
    
    # Filter from today's midnight
    df = df[df['time'] >= today_midnight].head(hours)
    
    print(f"[SUCCESS] Fetched {len(df)} hours from API")
    print(f"   Columns: {list(df.columns)}")
    return df


def fetch_weather_forecast_data():
    """
    Main function to get weather + pollutant forecast.
    Tries MongoDB first, falls back to API.
    Returns DataFrame with: time, temp, humidity, wind_speed, rain, weather_code, pm10, no2, ozone
    """
    # Try MongoDB first
    df = fetch_weather_forecast_from_mongo()
    
    # If MongoDB is empty or stale, fetch from API
    if df is None or df.empty:
        df = fetch_weather_forecast_from_api()
    
    return df


def generate_forecast_with_weather(model, df_history, weather_forecast):
    """
    Generate AQI forecast using weather predictions.
    Imported from clean_forecast for backward compatibility.
    """
    from src.prediction.clean_forecast import clean_forecast
    return clean_forecast(model, weather_forecast, df_history)
