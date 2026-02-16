"""
AQI Forecast Module - Clean Recursive Prediction

Strategy:
- Checks delay between current time and last available REAL data from Open-Meteo
- ALWAYS starts predicting from the hour after the last REAL data point
- Uses weather forecast inputs (temp, humidity, wind, pm10, no2, ozone) - NOT target variable
- Recursive: feeds predicted AQI back into history for lag features
- Returns (forecast_df, reference_df, last_real_time) so dashboard can cut history correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
from src.database import get_db_client
import src.config as config

# PKT timezone
PKT = timezone(timedelta(hours=5))


def get_real_data_boundary():
    """
    Find the exact boundary between REAL (measured) and MODEL (forecast) data
    in Open-Meteo's Air Quality API.
    
    Uses forecast_days=0 to get ONLY real/measured data.
    
    Returns:
        tuple: (delay_hours, last_real_time, last_real_aqi)
    """
    now = datetime.now(PKT).replace(tzinfo=None)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["us_aqi"],
        "past_days": 7,
        "forecast_days": 0,  # ONLY real/measured data — no model outputs
        "timezone": "Asia/Karachi"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30).json()
        if 'hourly' not in resp:
            print(f"[WARNING] API returned no data: {resp}")
            return 999, now, 50.0
        
        times = resp['hourly']['time']
        aqi_values = resp['hourly']['us_aqi']
        
        # Find last non-null entry = last REAL reading
        for i in range(len(aqi_values) - 1, -1, -1):
            if aqi_values[i] is not None:
                last_real_time = datetime.fromisoformat(times[i])
                delay_hours = (now - last_real_time).total_seconds() / 3600
                last_real_aqi = aqi_values[i]
                
                print(f"[INFO] Last REAL AQI: {last_real_aqi} at {times[i]}")
                print(f"[INFO] Delay from now: {delay_hours:.1f} hours")
                print(f"[INFO] This is REAL measured data (forecast_days=0)")
                
                return delay_hours, last_real_time, last_real_aqi
        
        return 999, now, 50.0
        
    except Exception as e:
        print(f"[ERROR] Could not check real data boundary: {e}")
        return 999, now, 50.0


def fetch_forecast_inputs(start_time, hours=72):
    """
    Fetch weather + pollutant FORECAST inputs from Open-Meteo.
    These are the INPUT variables for our model — NOT the target (us_aqi).
    
    Returns DataFrame with: time, temp, humidity, wind_speed, rain, weather_code, pm10, no2, ozone
    """
    print(f"\n[INFO] Fetching {hours}h forecast inputs from {start_time.strftime('%Y-%m-%d %H:%M')}...")
    
    # Weather forecast
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                    "wind_speed_10m", "weather_code"],
        "past_days": 2,
        "forecast_days": 5,  # Enough to cover 72h from any point
        "timezone": "Asia/Karachi"
    }
    
    try:
        w_resp = requests.get(w_url, params=w_params, timeout=30).json()
        df_w = pd.DataFrame(w_resp['hourly'])
        df_w['time'] = pd.to_datetime(df_w['time'])
        df_w.rename(columns={
            'temperature_2m': 'temp',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'rain',
            'wind_speed_10m': 'wind_speed'
        }, inplace=True)
    except Exception as e:
        print(f"[ERROR] Weather forecast failed: {e}")
        return pd.DataFrame()
    
    # Pollutant forecast (pm10, no2, ozone — NOT us_aqi since that's our target)
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["pm10", "nitrogen_dioxide", "ozone"],  # NO us_aqi — that's our target!
        "past_days": 2,
        "forecast_days": 5,
        "timezone": "Asia/Karachi"
    }
    
    try:
        a_resp = requests.get(a_url, params=a_params, timeout=30).json()
        df_a = pd.DataFrame(a_resp['hourly'])
        df_a['time'] = pd.to_datetime(df_a['time'])
        df_a.rename(columns={'nitrogen_dioxide': 'no2'}, inplace=True)
        
        df = df_w.merge(df_a, on='time', how='left')
    except Exception as e:
        print(f"[WARNING] Pollutant forecast failed: {e}")
        df = df_w
    
    # Fill missing pollutants
    for col, default in [('pm10', 50.0), ('no2', 15.0), ('ozone', 70.0)]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    
    # Filter to start from the requested time and take N hours
    df = df[df['time'] >= start_time].head(hours).reset_index(drop=True)
    
    if not df.empty:
        print(f"[SUCCESS] Forecast inputs: {len(df)} hours")
        print(f"   From: {df['time'].min().strftime('%Y-%m-%d %H:%M')}")
        print(f"   To:   {df['time'].max().strftime('%Y-%m-%d %H:%M')}")
    
    return df


def fetch_open_meteo_reference(start_time, hours=72):
    """
    Fetch Open-Meteo's own AQI FORECAST for comparison/reference on the dashboard.
    This is NOT used as model input — only for visual comparison.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": ["us_aqi"],
        "forecast_days": 5,
        "timezone": "Asia/Karachi"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30).json()
        df = pd.DataFrame(resp['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df = df[df['time'] >= start_time].head(hours)
        return df
    except Exception as e:
        print(f"[WARNING] Could not fetch Open-Meteo reference: {e}")
        return pd.DataFrame()


def clean_forecast(model, df_weather_forecast=None, df_history=None):
    """
    Generate AQI forecast starting from the last REAL data point.
    
    Strategy:
    1. Query Open-Meteo with forecast_days=0 to find last REAL measured data
    2. Start predictions from the hour AFTER last real data
    3. Use weather forecast inputs (NOT target variable)
    4. Recursive: feed predicted AQI back into history for lag features
    
    Args:
        model: Trained model (LightGBM targeting us_aqi)  
        df_weather_forecast: Pre-fetched weather forecast (optional)
        df_history: Historical data (optional)
    
    Returns:
        tuple: (result_df, reference_df, last_real_time)
        - result_df: DataFrame with ['time', 'aqi', 'pm2_5']
        - reference_df: DataFrame with Open-Meteo's own forecast for comparison
        - last_real_time: datetime — boundary between real and predicted data
    """
    from src.features.feature_engineering import create_forecast_features
    
    print("\n" + "=" * 60)
    print("AQI FORECAST GENERATION")
    print("=" * 60)
    
    # ===== Step 1: Find real data boundary =====
    delay_hours, last_real_time, last_real_aqi = get_real_data_boundary()
    
    # ALWAYS start forecast from 1 hour after last real data
    forecast_start = last_real_time + timedelta(hours=1)
    now = datetime.now(PKT).replace(tzinfo=None)
    
    # Calculate how many hours to predict
    hours_to_predict = max(72, int(delay_hours) + 72)
    # Cap at reasonable max
    hours_to_predict = min(hours_to_predict, 120)
    
    print(f"[STRATEGY] Last real data: {last_real_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"[STRATEGY] Delay: {delay_hours:.1f}h")
    print(f"[STRATEGY] Forecast starts: {forecast_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"[STRATEGY] Hours to predict: {hours_to_predict}")
    
    # ===== Step 2: Fetch forecast inputs =====
    if df_weather_forecast is not None and not df_weather_forecast.empty:
        df_weather_forecast['time'] = pd.to_datetime(df_weather_forecast['time'])
        df_inputs = df_weather_forecast[df_weather_forecast['time'] >= forecast_start].head(hours_to_predict)
    else:
        df_inputs = fetch_forecast_inputs(forecast_start, hours=hours_to_predict)
    
    if df_inputs.empty:
        print("[ERROR] No forecast input data available!")
        return pd.DataFrame(), pd.DataFrame(), last_real_time
    
    # ===== Step 3: Load history for lag features =====
    db = get_db_client()
    history_docs = list(db[config.FEATURE_COLLECTION].find(
        {"time": {"$lte": last_real_time}},  # Only up to last REAL time
        {"_id": 0, "us_aqi": 1}
    ).sort("time", -1).limit(72))
    
    if history_docs:
        history_aqi = [d.get('us_aqi', 50.0) for d in reversed(history_docs)]
    else:
        history_aqi = [last_real_aqi] * 72
    
    print(f"[INFO] History buffer: {len(history_aqi)} values")
    print(f"   Last actual AQI: {history_aqi[-1]:.1f}")
    
    # ===== Step 4: Get expected feature names from model =====
    expected_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
    print(f"[INFO] Model expects {len(expected_features)} features")
    
    # ===== Step 5: Recursive prediction loop =====
    predictions = []
    
    for i_step, (idx, row) in enumerate(df_inputs.iterrows()):
        # Create features using current weather row + history
        features_dict = create_forecast_features(
            last_data=row,
            forecast_hour=0,
            history_values=history_aqi,
            target_col='us_aqi'
        )
        
        # Build model input vector in correct order
        feature_vector = [features_dict.get(f, 0.0) for f in expected_features]
        
        # Predict
        try:
            prediction = model.predict([feature_vector])[0]
        except Exception as e:
            print(f"[WARNING] Prediction error at step {i_step}: {e}")
            prediction = history_aqi[-1] if history_aqi else 50.0
        
        # Clip to valid AQI range
        prediction = max(0, min(prediction, 500))
        
        # Feed prediction back into history (recursive)
        history_aqi.append(prediction)
        
        predictions.append({
            'time': row['time'],
            'aqi': prediction,
            'pm2_5': prediction / 2.5  # Approximate reverse
        })
        
        if i_step < 3 or i_step % 24 == 0:
            print(f"   t={i_step}: AQI={prediction:.1f} (weather: temp={row.get('temp', '?')}, hum={row.get('humidity', '?')})")
    
    result_df = pd.DataFrame(predictions)
    
    # ===== Step 6: Fetch Open-Meteo reference for comparison =====
    reference_df = fetch_open_meteo_reference(forecast_start, hours=hours_to_predict)
    
    # ===== Summary =====
    print(f"\n[SUCCESS] Generated {len(result_df)} hour forecast")
    if not result_df.empty:
        print(f"   AQI range: {result_df['aqi'].min():.0f} - {result_df['aqi'].max():.0f}")
        print(f"   Mean AQI: {result_df['aqi'].mean():.1f}")
    
    return result_df, reference_df, last_real_time
