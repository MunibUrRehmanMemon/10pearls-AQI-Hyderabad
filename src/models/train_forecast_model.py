"""
Train 3 Forecast Models - LightGBM, XGBoost, RandomForest
===========================================================
Target: US AQI (0-500 scale)
Features: Based on EDA findings (see eda_output.txt)
- Weather: temp, humidity, wind_speed, rain, weather_code
- Pollutants: pm10, no2, ozone (NOT target - no leakage)
- Time: cyclical + explicit hour markers
- Lag: us_aqi_lag_24h, rolling_mean_24h, rolling_std_24h
- Interactions: wind_x_temp, humidity_x_temp, etc.

All 3 models are saved to MongoDB model registry.
The best model (by MAE) is also saved as 'best_model'.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.database import get_db_client
import src.config as config
from src.models.registry import save_model

TARGET_COL = 'us_aqi'


def engineer_training_features(df, target_col=TARGET_COL):
    """
    Engineer features for training.
    
    CRITICAL: Must produce the EXACT SAME features that create_forecast_features() 
    produces at inference time. Any mismatch = model gets wrong inputs.
    """
    df = df.copy()
    
    # --- Cyclical Time ---
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # --- Explicit Time ---
    df['hour'] = df.index.hour
    df['is_night'] = ((df.index.hour >= 20) | (df.index.hour <= 5)).astype(int)
    df['is_morning_rush'] = ((df.index.hour >= 6) & (df.index.hour <= 10)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 11) & (df.index.hour <= 17)).astype(int)
    df['is_evening_rush'] = ((df.index.hour >= 17) & (df.index.hour <= 20)).astype(int)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # --- Lag Features (from target) ---
    df[f'{target_col}_lag_24h'] = df[target_col].shift(24)
    df[f'{target_col}_rolling_mean_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    df[f'{target_col}_rolling_std_24h'] = df[target_col].rolling(window=24, min_periods=1).std()
    
    # Fill NAs
    df[f'{target_col}_lag_24h'] = df[f'{target_col}_lag_24h'].bfill().ffill()
    df[f'{target_col}_rolling_mean_24h'] = df[f'{target_col}_rolling_mean_24h'].bfill()
    df[f'{target_col}_rolling_std_24h'] = df[f'{target_col}_rolling_std_24h'].fillna(0)
    
    # --- Interaction Features ---
    if 'wind_speed' in df.columns and 'temp' in df.columns:
        df['wind_x_temp'] = df['wind_speed'] * df['temp']
    if 'humidity' in df.columns and 'temp' in df.columns:
        df['humidity_x_temp'] = df['humidity'] * df['temp']
    if 'wind_speed' in df.columns and 'humidity' in df.columns:
        df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
    if 'pm10' in df.columns and 'humidity' in df.columns:
        df['pm10_x_humidity'] = df['pm10'] * df['humidity']
    if 'ozone' in df.columns:
        df['hour_x_ozone'] = df['hour'] * df['ozone']
    
    df = df.dropna()
    return df


# Feature columns (exact match with create_forecast_features)
FEATURE_COLS = [
    # Weather
    'temp', 'humidity', 'wind_speed', 'rain', 'weather_code',
    # Pollutants (NOT target)
    'pm10', 'no2', 'ozone',
    # Cyclical time
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Explicit time
    'hour', 'is_night', 'is_morning_rush', 'is_afternoon',
    'is_evening_rush', 'is_weekend',
    # Lag/Rolling
    f'{TARGET_COL}_lag_24h', f'{TARGET_COL}_rolling_mean_24h',
    f'{TARGET_COL}_rolling_std_24h',
    # Interactions
    'wind_x_temp', 'humidity_x_temp', 'wind_x_humidity',
    'pm10_x_humidity', 'hour_x_ozone',
]


def prepare_data(df, target_col=TARGET_COL):
    """Prepare X, y ensuring no target leakage."""
    y = df[target_col].copy()
    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].copy()
    
    # Safety: remove any target/derived columns
    for col in [target_col, 'pm2_5']:
        if col in X.columns:
            X = X.drop(columns=[col])
            print(f"  [SAFETY] Removed leakage: {col}")
    
    return X, y, list(X.columns)


def get_models():
    """Define the 3 models to train."""
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        xgb_available = False
        print("[WARNING] XGBoost not installed. pip install xgboost")
    
    models = {}
    
    # 1. LightGBM (fast, good with categorical-like features)
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 2. XGBoost (strong gradient boosting)
    if xgb_available:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # 3. RandomForest (robust baseline)
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    return models


def train_all_models():
    """Train all 3 models and save to registry."""
    print("\n" + "=" * 70)
    print("  TRAINING 3 FORECAST MODELS")
    print(f"  Target: {TARGET_COL}")
    print("=" * 70)
    
    # Load data
    db = get_db_client()
    data = list(db[config.FEATURE_COLLECTION].find({}, {"_id": 0}))
    
    if not data:
        raise ValueError("No data found in MongoDB! Run ingestion first.")
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not in data! Columns: {list(df.columns)}")
    
    print(f"\nData: {len(df)} records ({df.index.min()} to {df.index.max()})")
    print(f"Target range: {df[TARGET_COL].min():.0f} - {df[TARGET_COL].max():.0f} (mean: {df[TARGET_COL].mean():.1f})")
    
    # Engineer features
    df_eng = engineer_training_features(df)
    X, y, feature_names = prepare_data(df_eng)
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)} (after dropping NaN)")
    
    # Chronological train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train each model
    models = get_models()
    results = {}
    best_mae = float('inf')
    best_name = None
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"  R2:   {r2:.4f}")
        print(f"  MAE:  {mae:.2f} AQI")
        print(f"  RMSE: {rmse:.2f} AQI")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            fi = sorted(zip(feature_names, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
            print(f"  Top 5 features:")
            for fname, imp in fi[:5]:
                print(f"    {fname:30s} {imp:.4f}")
        
        results[name] = {
            'model': model,
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'test_preds': preds
        }
        
        if mae < best_mae:
            best_mae = mae
            best_name = name
    
    # Retrain all on full dataset and save
    print(f"\n{'='*70}")
    print(f"  RETRAINING ON FULL DATA & SAVING TO REGISTRY")
    print(f"{'='*70}")
    
    for name, model in models.items():
        print(f"\n  Saving {name}...")
        model.fit(X, y)  # Full data retrain
        
        r = results[name]
        is_best = (name == best_name)
        
        metrics = {
            'model_name': name,
            'r2': r['r2'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'target': TARGET_COL,
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_names),
            'n_samples': len(X),
            'features': feature_names,
            'is_best': is_best,
            'description': f'{name} targeting {TARGET_COL}'
        }
        
        # Save with model name
        save_model(model, metrics, model_name=name)
        
        # Also save best as 'best_model'
        if is_best:
            save_model(model, metrics, model_name='best_model')
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Results (evaluated on test set):")
    print(f"  {'Model':<15} {'R2':>8} {'MAE':>8} {'RMSE':>8} {'Best':>6}")
    print(f"  {'-'*47}")
    for name, r in results.items():
        star = " <--" if name == best_name else ""
        print(f"  {name:<15} {r['r2']:8.4f} {r['mae']:8.2f} {r['rmse']:8.2f}  {star}")
    
    print(f"\n  Best model: {best_name} (MAE={best_mae:.2f})")
    print(f"  Saved as: 'best_model' in MongoDB registry")
    print(f"  Features: {feature_names}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_all_models()
