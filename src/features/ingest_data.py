"""
Dynamic Data Ingestion Module for AQI Prediction
- Stores all timestamps in PKT (Pakistan Standard Time = UTC+5)
- Automatically fetches past data from last stored date to latest available
- Stores weather forecast for next 72 hours starting from current hour
- Fully serverless - runs via GitHub Actions daily and can be called hourly
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.database import get_db_client
import src.config as config

# PKT timezone (UTC+5)
PKT = timezone(timedelta(hours=5))


def get_pkt_now():
    """Get current datetime in PKT timezone."""
    return datetime.now(PKT)


def get_latest_stored_date(collection_name):
    """Get the latest date stored in MongoDB collection."""
    db = get_db_client()
    collection = db[collection_name]
    
    latest = collection.find_one(
        {},
        sort=[("time", -1)],
        projection={"time": 1}
    )
    
    if latest and "time" in latest:
        return latest["time"]
    return None


def fetch_open_meteo_historical(start_date, end_date):
    """
    Fetch historical weather and AQI data from Open-Meteo Archive API.
    Note: Archive API has ~2-5 day delay for recent data.
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"[INFO] Fetching historical data: {start_str} to {end_str}")
    
    # 1. Fetch WEATHER (Archive API)
    w_url = "https://archive-api.open-meteo.com/v1/archive"
    w_params = {
        "latitude": config.LAT, 
        "longitude": config.LON,
        "start_date": start_str, 
        "end_date": end_str,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "weather_code"],
        "timezone": "Asia/Karachi"  # Returns data in PKT
    }
    
    try:
        w_resp = requests.get(w_url, params=w_params, timeout=30).json()
        if "error" in w_resp:
            print(f"[ERROR] Weather API Error: {w_resp.get('reason', 'Unknown')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Weather Request Failed: {e}")
        return pd.DataFrame()
    
    # 2. Fetch AIR QUALITY (Archive API) - including US AQI directly
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": config.LAT, 
        "longitude": config.LON,
        "start_date": start_str, 
        "end_date": end_str,
        "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi"],  # Added us_aqi
        "timezone": "Asia/Karachi"
    }
    
    try:
        a_resp = requests.get(a_url, params=a_params, timeout=30).json()
        if "error" in a_resp:
            print(f"[ERROR] AQI API Error: {a_resp.get('reason', 'Unknown')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] AQI Request Failed: {e}")
        return pd.DataFrame()
    
    # 3. Process & Merge
    if 'hourly' not in w_resp or 'hourly' not in a_resp:
        print("[WARNING] No data available for this date range")
        return pd.DataFrame()
    
    df_w = pd.DataFrame(w_resp['hourly'])
    df_a = pd.DataFrame(a_resp['hourly'])
    df = pd.merge(df_w, df_a, on='time')
    
    # Rename columns
    df.rename(columns={
        'temperature_2m': 'temp',
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed',
        'nitrogen_dioxide': 'no2'
    }, inplace=True)
    
    # Parse time as PKT (Open-Meteo returns Asia/Karachi times)
    df['time'] = pd.to_datetime(df['time'])
    df['city'] = "Hyderabad"
    
    # Fill missing values
    df = df.infer_objects(copy=False).interpolate(method='linear').bfill().ffill()
    
    return df


def fetch_weather_forecast(hours=72):
    """
    Fetch weather AND pollutant forecast from Open-Meteo for next N hours.
    Starts from TODAY'S MIDNIGHT for consistent daily forecasts.
    
    Fetches:
    - Weather: temp, humidity, rain, wind_speed, weather_code
    - Pollutants: pm10, no2, ozone (NOT pm2_5 - that's our target!)
    """
    now = get_pkt_now()
    # Start from TODAY's midnight, not current hour
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    
    print(f"\n[INFO] Fetching {hours}-hour forecast from {today_midnight.strftime('%Y-%m-%d %H:%M')} PKT...")
    
    # 1. Fetch WEATHER forecast
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                   "wind_speed_10m", "weather_code"],
        "forecast_days": 4,  # Get 4 days to ensure 72+ hours
        "timezone": "Asia/Karachi"  # Returns data in PKT
    }
    
    try:
        w_resp = requests.get(w_url, params=w_params, timeout=30).json()
        if "error" in w_resp:
            print(f"[ERROR] Weather Forecast API Error: {w_resp.get('reason', 'Unknown')}")
            return pd.DataFrame()
        
        df_weather = pd.DataFrame(w_resp['hourly'])
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        
        # Rename columns to match schema
        df_weather.rename(columns={
            'temperature_2m': 'temp',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'rain',
            'wind_speed_10m': 'wind_speed'
        }, inplace=True)
        
    except Exception as e:
        print(f"[ERROR] Weather forecast fetch failed: {e}")
        return pd.DataFrame()
    
    # 2. Fetch POLLUTANT forecast + US AQI directly
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["pm10", "nitrogen_dioxide", "ozone", "us_aqi"],  # Added us_aqi for comparison
        "forecast_days": 4,
        "timezone": "Asia/Karachi"
    }
    
    try:
        a_resp = requests.get(a_url, params=a_params, timeout=30).json()
        if "error" in a_resp:
            print(f"[WARNING] Air Quality Forecast API Error: {a_resp.get('reason', 'Unknown')}")
            # Continue with just weather data
            df_weather['pm10'] = 50.0  # Default
            df_weather['no2'] = 15.0
            df_weather['ozone'] = 70.0
        else:
            df_air = pd.DataFrame(a_resp['hourly'])
            df_air['time'] = pd.to_datetime(df_air['time'])
            df_air.rename(columns={'nitrogen_dioxide': 'no2'}, inplace=True)
            
            # Merge with weather data
            df_weather = df_weather.merge(df_air, on='time', how='left')
            
            # Fill any missing pollutant values
            df_weather['pm10'] = df_weather['pm10'].fillna(50.0)
            df_weather['no2'] = df_weather['no2'].fillna(15.0)
            df_weather['ozone'] = df_weather['ozone'].fillna(70.0)
            
    except Exception as e:
        print(f"[WARNING] Pollutant forecast fetch failed: {e}")
        # Continue with default values
        df_weather['pm10'] = 50.0
        df_weather['no2'] = 15.0
        df_weather['ozone'] = 70.0
    
    # Filter to start from today's midnight
    df = df_weather[df_weather['time'] >= today_midnight].head(hours)
    
    end_time = df['time'].max()
    print(f"[SUCCESS] Forecast: {len(df)} hours from {today_midnight.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"   Columns: {list(df.columns)}")
    return df


def fetch_recent_history(days=7):
    """
    Fetch recent history (last N days) using the FORECAST API (past_days parameter).
    This fills the gap between Archive API (2-5 day delay) and current time.
    """
    print(f"\n[INFO] Fetching RECENT history (last {days} days) via Forecast API...")
    
    # 1. Fetch WEATHER + AIR QUALITY from Forecast API using past_days
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "weather_code"],
        "timezone": "Asia/Karachi",
        "past_days": days,
        "forecast_days": 1 
    }
    
    try:
        # Weather
        w_resp = requests.get(url, params=params, timeout=30).json()
        df_weather = pd.DataFrame(w_resp['hourly'])
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        
        df_weather.rename(columns={
            'temperature_2m': 'temp',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'rain',
            'wind_speed_10m': 'wind_speed'
        }, inplace=True)
        
        # Air Quality
        a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        a_params = {
            "latitude": config.LAT,
            "longitude": config.LON,
            "hourly": ["pm10", "nitrogen_dioxide", "ozone", "us_aqi"],
            "timezone": "Asia/Karachi",
            "past_days": days,
            "forecast_days": 1
        }
        
        a_resp = requests.get(a_url, params=a_params, timeout=30).json()
        df_air = pd.DataFrame(a_resp['hourly'])
        df_air['time'] = pd.to_datetime(df_air['time'])
        df_air.rename(columns={'nitrogen_dioxide': 'no2'}, inplace=True)
        
        # Mergex
        df = df_weather.merge(df_air, on='time', how='inner')
        
        # Filter: Keep only PAST data (up to current HOUR boundary, not current minute)
        # This prevents the Forecast API from leaking future forecast hours into history
        now = datetime.now()  # System time (naive)
        current_hour_boundary = now.replace(minute=0, second=0, microsecond=0)
        if df['time'].dt.tz is not None:
             df['time'] = df['time'].dt.tz_localize(None)
             
        df = df[df['time'] < current_hour_boundary]
        
        print(f"[SUCCESS] Fetched {len(df)} recent history records")
        if df.empty:
             print(f"[DEBUG] filtered df empty. max time: {df['time'].max() if not df.empty else 'None'}, now: {now}")
             
        return df
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to fetch recent history: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def upload_historical_to_mongo(df):
    """Upload historical data to MongoDB, avoiding duplicates."""
    if df.empty:
        print("[WARNING] Historical DataFrame empty. Skipping.")
        return
    
    db = get_db_client()
    collection = db[config.FEATURE_COLLECTION]
    
    records = df.to_dict("records")
    print(f"[INFO] Uploading {len(records)} historical records...")
    
    # Delete existing range to prevent duplicates
    start_time = df['time'].min()
    end_time = df['time'].max()
    deleted = collection.delete_many({"time": {"$gte": start_time, "$lte": end_time}})
    print(f"[INFO] Replaced {deleted.deleted_count} existing records")
    
    collection.insert_many(records)
    print(f"[SUCCESS] Historical data uploaded: {start_time} to {end_time} PKT")


def upload_forecast_to_mongo(df):
    """
    Upload weather forecast to MongoDB.
    Replaces existing forecast data for real-time updates.
    """
    if df.empty:
        print("[WARNING] Forecast DataFrame empty. Skipping.")
        return
    
    db = get_db_client()
    collection = db["weather_forecast"]
    
    # Clear old forecast data
    collection.delete_many({})
    
    records = df.to_dict("records")
    collection.insert_many(records)
    
    print(f"[SUCCESS] Forecast uploaded: {len(records)} hours to 'weather_forecast' collection")


def ingest_historical_data():
    """
    DYNAMIC ingestion:
    1. Find latest date in MongoDB
    2. Fetch from (latest + 1 day) to (today - 5 days) [API has delay]
    3. Upload to MongoDB
    """
    print("\n" + "=" * 60)
    print("📡 DYNAMIC HISTORICAL DATA INGESTION (PKT Timezone)")
    print("=" * 60)
    
    now = get_pkt_now().replace(tzinfo=None)  # Use naive datetime
    # 1. Fetch Historical Data (Archive API - up to 5 days ago)
    # Check what we have in DB
    last_date = get_latest_stored_date(config.FEATURE_COLLECTION)
    
    if last_date:
        start_date = last_date + timedelta(hours=1)
        print(f"[INFO] Latest data in MongoDB: {last_date}")
    else:
        # If no data, fetch last 6 months
        start_date = datetime.now() - timedelta(days=180)
        print("[INFO] No existing data. Fetching 6 months of historical data...")
        
    end_date = datetime.now()
    
    # Only fetch if we are missing data
    if start_date < end_date:
        print(f"[INFO] Fetching: {start_date.date()} to {end_date.date()}")
        
        df = fetch_open_meteo_historical(start_date, end_date)
        
        if not df.empty:
            upload_historical_to_mongo(df)
            print(f"[SUCCESS] Added {len(df)} new records to MongoDB")
        else:
            print("[WARNING] No new data fetched")
    else:
        print("[INFO] Historical data is up to date. No new data to fetch.")


def ingest_weather_forecast():
    """
    Fetch and store 72-hour weather forecast starting from current hour.
    This can run hourly to keep forecast up-to-date.
    """
    print("\n" + "=" * 60)
    print("🌤️ WEATHER FORECAST INGESTION (Next 72 Hours from NOW)")
    print("=" * 60)
    
    df = fetch_weather_forecast(hours=72)
    
    if not df.empty:
        upload_forecast_to_mongo(df)
    else:
        print("[WARNING] Forecast not available")


if __name__ == "__main__":
    now = get_pkt_now()
    
    print("\n" + "=" * 60)
    print("🚀 AQI DATA INGESTION PIPELINE")
    print(f"📍 Location: Hyderabad, Sindh, Pakistan ({config.LAT}°N, {config.LON}°E)")
    print(f"📅 Current Time: {now.strftime('%Y-%m-%d %H:%M')} PKT")
    print("=" * 60)
    
    # Step 1: Ingest historical data (Archive API - has 2-5 day delay)
    ingest_historical_data()
    
    # Step 2: Fill the gap between Archive API and NOW using Forecast API past_days
    # This is CRITICAL — without this, there's a multi-day data gap
    print("\n" + "=" * 60)
    print("🔄 FILLING RECENT DATA GAP (Forecast API past_days)")
    print("=" * 60)
    
    df_recent = fetch_recent_history(days=7)
    if not df_recent.empty:
        # Add city column if missing
        if 'city' not in df_recent.columns:
            df_recent['city'] = "Hyderabad"
        upload_historical_to_mongo(df_recent)
        print(f"[SUCCESS] Filled {len(df_recent)} recent records")
    else:
        print("[WARNING] No recent history available")
    
    # Step 3: Ingest weather forecast (next 72 hours from NOW)
    ingest_weather_forecast()
    
    # Summary
    db = get_db_client()
    hist_count = db[config.FEATURE_COLLECTION].count_documents({})
    forecast_count = db["weather_forecast"].count_documents({})
    
    latest = get_latest_stored_date(config.FEATURE_COLLECTION)
    
    forecast_range = list(db["weather_forecast"].find({}, {"time": 1, "_id": 0}).sort("time", 1))
    forecast_start = forecast_range[0]['time'] if forecast_range else None
    forecast_end = forecast_range[-1]['time'] if forecast_range else None
    
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)
    print(f"   Historical Records: {hist_count}")
    print(f"   Latest Historical: {latest}")
    print(f"   Forecast Hours: {forecast_count}")
    if forecast_start and forecast_end:
        print(f"   Forecast Range: {forecast_start} to {forecast_end} PKT")
    print("=" * 60)