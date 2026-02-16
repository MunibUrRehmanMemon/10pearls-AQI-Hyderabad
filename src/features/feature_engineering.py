"""
Enhanced Feature Engineering Module for AQI Prediction
Includes comprehensive feature creation with temporal, lag, and interaction features
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Creates comprehensive feature set for AQI prediction.
    
    Features include:
    - RAW WEATHER FEATURES (temp, humidity, wind_speed, rain) - for forecast-based predictions
    - Cyclical time encodings (hour, day of week, month)
    - Lag features (1h, 3h, 6h, 12h, 24h, 48h)
    - Rolling statistics (mean, std, min, max)
    - Interaction features
    - Weather stability indicators
    
    Args:
        df: DataFrame with datetime index and raw features
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # ===== 0. RAW WEATHER FEATURES (CRITICAL FOR WEATHER-BASED PREDICTIONS) =====
    # These are the main features that come from weather forecasts
    # Keep original columns: temp, humidity, wind_speed, rain, weather_code, pm10, no2, ozone
    # They are already in df from ingestion, we just ensure they're present
    
    print(f"[INFO] Raw weather columns available: {[c for c in ['temp', 'humidity', 'wind_speed', 'rain', 'weather_code', 'pm10', 'no2', 'ozone'] if c in df.columns]}")
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            print("[INFO] Setting 'time' column as index for feature engineering...")
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            print("[WARNING] No 'time' column found and index is not DatetimeIndex. Cyclical features may be wrong.")
            # Depending on usage, might want to raise error or return
            # For now, let it crash if it must, or skip
            pass
    
    # ===== 1. CYCLICAL TIME FEATURES =====
    # Hour of day (captures daily pollution cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day of week (weekday vs weekend patterns)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Month (seasonal patterns)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # ===== 1B. EXPLICIT TIME FEATURES (stronger signal than sin/cos) =====
    # Raw hour - lets model learn exact hourly patterns
    df['hour'] = df.index.hour
    
    # Time period indicators - explicit pattern encoding
    # Night (8pm-5am): typically HIGH PM2.5 due to low atmospheric mixing
    df['is_night'] = ((df.index.hour >= 20) | (df.index.hour <= 5)).astype(int)
    
    # Morning rush (6am-10am): high pollution from traffic
    df['is_morning_rush'] = ((df.index.hour >= 6) & (df.index.hour <= 10)).astype(int)
    
    # Afternoon (11am-5pm): typically LOW PM2.5 due to solar heating/mixing
    df['is_afternoon'] = ((df.index.hour >= 11) & (df.index.hour <= 17)).astype(int)
    
    # Evening rush (5pm-8pm): moderate pollution
    df['is_evening_rush'] = ((df.index.hour >= 17) & (df.index.hour <= 20)).astype(int)
    
    # Weekend indicator
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # ===== 2. LAG FEATURES - REINTRODUCED (ROBUST ONLY) =====
    # We add ONLY 24h lags/rolling features to capture daily persistence
    # without introducing short-term autoregressive oscillation (like 1h lag).
    
    target_for_lag = 'us_aqi' if 'us_aqi' in df.columns else 'pm2_5'
    print(f"[INFO] Creating lag features based on: {target_for_lag}")
    
    # 24h Lag (Yesterday's value at same time)
    df[f'{target_for_lag}_lag_24h'] = df[target_for_lag].shift(24)
    
    # 24h Rolling Mean (General daily level)
    df[f'{target_for_lag}_rolling_mean_24h'] = df[target_for_lag].rolling(window=24, min_periods=1).mean()
    
    # 24h Rolling Std (Volatility)
    df[f'{target_for_lag}_rolling_std_24h'] = df[target_for_lag].rolling(window=24, min_periods=1).std()
    
    # Fill NAs for the first 24 hours (backfill to avoid dropping too much data)
    df[f'{target_for_lag}_lag_24h'] = df[f'{target_for_lag}_lag_24h'].bfill().fillna(method='ffill')
    df[f'{target_for_lag}_rolling_mean_24h'] = df[f'{target_for_lag}_rolling_mean_24h'].bfill()
    df[f'{target_for_lag}_rolling_std_24h'] = df[f'{target_for_lag}_rolling_std_24h'].fillna(0)

    
    # ===== 5. INTERACTION FEATURES =====
    # Wind can disperse pollutants
    if 'wind_speed' in df.columns and 'temp' in df.columns:
        df['wind_x_temp'] = df['wind_speed'] * df['temp']
    
    # Humidity affects particle behavior
    if 'humidity' in df.columns and 'temp' in df.columns:
        df['humidity_x_temp'] = df['humidity'] * df['temp']
    
    # Wind and humidity interaction
    if 'wind_speed' in df.columns and 'humidity' in df.columns:
        df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
        
    # NEW: Pollutant Interactions
    if 'pm10' in df.columns and 'humidity' in df.columns:
        df['pm10_x_humidity'] = df['pm10'] * df['humidity']
        
    if 'hour' in df.columns and 'ozone' in df.columns:
        df['hour_x_ozone'] = df['hour'] * df['ozone']
    
    # PM10 to PM2.5 ratio (aerosol size distribution)
    # Note: 'pm2_5' usually exists, but we use target_for_lag just in case
    # If pm2_5 is available, use it
    current_pm25 = 'pm2_5' if 'pm2_5' in df.columns else target_for_lag
    if 'pm10' in df.columns:
        df['pm10_pm25_ratio'] = df['pm10'] / (df[current_pm25] + 1e-6)
        df['pm10_pm25_ratio'] = df['pm10'] / (df['pm2_5'] + 1e-6)
    
    # ===== 6. WEATHER STABILITY INDICATORS =====
    # Temperature stability
    if 'temp' in df.columns:
        df['temp_rolling_std_12h'] = df['temp'].rolling(window=12, min_periods=1).std()
        df['temp_diff_1h'] = df['temp'].diff(1)
    
    # Wind stability
    if 'wind_speed' in df.columns:
        df['wind_rolling_mean_12h'] = df['wind_speed'].rolling(window=12, min_periods=1).mean()
        df['wind_rolling_std_12h'] = df['wind_speed'].rolling(window=12, min_periods=1).std()
    
    # ===== 7. POLLUTANT INTERACTIONS =====
    if 'no2' in df.columns:
        df['no2_lag_1h'] = df['no2'].shift(1)
        df['no2_rolling_mean_12h'] = df['no2'].rolling(window=12, min_periods=1).mean()
    
    if 'ozone' in df.columns:
        df['ozone_lag_1h'] = df['ozone'].shift(1)
        df['ozone_rolling_mean_12h'] = df['ozone'].rolling(window=12, min_periods=1).mean()
    
    # ===== 8. CLEAN UP =====
    # Drop rows with NaN values (from lag and rolling windows)
    df = df.dropna()
    
    return df


def prepare_training_data(df, target_col='us_aqi', prediction_horizon=1):
    """
    Prepares data for training by creating target variable and feature matrix.
    
    CRITICAL: This function explicitly excludes the target column (us_aqi) from
    features to prevent data leakage. The model should learn from:
    - Lag features (aqi_lag_1h, etc.) - historical values only
    - Weather features (temp, wind_speed, etc.)
    - Temporal features (hour_sin, is_morning_rush, etc.)
    
    Args:
        df: DataFrame with engineered features
        target_col: Column name for target variable (default: 'us_aqi' for composite AQI)
        prediction_horizon: How many hours ahead to predict (default: 1)
        
    Returns:
        X: Feature matrix (WITHOUT target column)
        y: Target vector
        feature_names: List of feature names
    """
    df = df.copy()
    
    # Create target: Predict US AQI 'prediction_horizon' hours ahead
    df['target'] = df[target_col].shift(-prediction_horizon)
    
    # Drop rows with missing target
    df = df.dropna(subset=['target'])
    
    # Select only numeric features (exclude city, time if present)
    numeric_df = df.select_dtypes(include=[np.number, bool])
    
    # CRITICAL: Explicitly exclude target column from features to prevent leakage
    # The target_col (pm2_5) should NOT be in X - only lag features are allowed
    exclude_columns = ['target', target_col]
    columns_to_drop = [col for col in exclude_columns if col in numeric_df.columns]
    
    # Separate features and target
    y = numeric_df['target']
    X = numeric_df.drop(columns=columns_to_drop)
    
    # Validation: Ensure target is not in features
    if target_col in X.columns:
        raise ValueError(f"CRITICAL: Target column '{target_col}' found in features! This is data leakage.")
    
    print(f"✅ Prepared training data: {len(X)} samples, {len(X.columns)} features")
    print(f"✅ Target column '{target_col}' excluded from features: {target_col not in X.columns}")
    
    return X, y, list(X.columns)


def create_forecast_features(last_data, forecast_hour, history_values, target_col='us_aqi'):
    """
    Creates features for forecasting future hours.
    
    Args:
        last_data: Most recent data row (dict or Series)
        forecast_hour: Hours into the future to forecast (1, 2, 3...)
        history_values: List of historical target values (e.g. us_aqi)
        target_col: Name of the target column (default 'us_aqi')
        
    Returns:
        Dictionary of features for prediction
    """
    from datetime import timedelta
    
    # Calculate the future time
    if 'time' in last_data:
        current_time = last_data['time']
    elif hasattr(last_data, 'name') and isinstance(last_data.name, pd.Timestamp):
        current_time = last_data.name
    else:
        current_time = pd.Timestamp.now()
        
    # Ensure current_time is a timestamp
    current_time = pd.to_datetime(current_time)
    
    future_time = current_time + timedelta(hours=forecast_hour)
    
    # Base features (assuming relatively stable weather)
    features = {
        'temp': last_data.get('temp', 25),
        'humidity': last_data.get('humidity', 50),
        'wind_speed': last_data.get('wind_speed', 5),
        'rain': last_data.get('rain', 0),
        'weather_code': last_data.get('weather_code', 0),
        'pm10': last_data.get('pm10', 50),
        'no2': last_data.get('no2', 20),
        'ozone': last_data.get('ozone', 50),
    }
    
    # Cyclical time features
    features['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
    features['dow_sin'] = np.sin(2 * np.pi * future_time.dayofweek / 7)
    features['dow_cos'] = np.cos(2 * np.pi * future_time.dayofweek / 7)
    features['month_sin'] = np.sin(2 * np.pi * future_time.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * future_time.month / 12)
    
    # Explicit Time Features (for learning cycles)
    features['hour'] = future_time.hour
    features['is_night'] = 1 if (future_time.hour >= 20 or future_time.hour <= 5) else 0
    features['is_morning_rush'] = 1 if (6 <= future_time.hour <= 10) else 0
    features['is_afternoon'] = 1 if (11 <= future_time.hour <= 17) else 0
    features['is_evening_rush'] = 1 if (17 <= future_time.hour <= 20) else 0
    features['is_weekend'] = 1 if (future_time.dayofweek >= 5) else 0
    
    # Lag/Rolling Features (Critical for Momentum)
    # We need at least 24 hours of history for these features
    # history_values should contain [historic_data..., prediction_h1, prediction_h2...]
    
    # Lag 24h (Yesterday)
    # If we are forecasting h=1, we need hist[-24]
    features[f'{target_col}_lag_24h'] = history_values[-24] if len(history_values) >= 24 else 50
    
    # Rolling 24h Stats
    window_24 = history_values[-24:] if len(history_values) >= 24 else history_values
    features[f'{target_col}_rolling_mean_24h'] = np.mean(window_24) if window_24 else 50
    features[f'{target_col}_rolling_std_24h'] = np.std(window_24) if window_24 else 0
    
    # Interaction Features
    features['wind_x_temp'] = features['wind_speed'] * features['temp']
    features['humidity_x_temp'] = features['humidity'] * features['temp']
    features['wind_x_humidity'] = features['wind_speed'] * features['humidity']
    features['pm10_x_humidity'] = features['pm10'] * features['humidity']
    features['hour_x_ozone'] = features['hour'] * features['ozone']
    
    # Ratios
    # If target is us_aqi, we don't have pm2_5 prediction yet, but pm10 is input
    # We can approximate pm2.5 from aqi or just use aqi in interaction if needed
    # For now, let's skip pm10_pm25_ratio if we don't have pm2.5, or use aqi
    
    return features
