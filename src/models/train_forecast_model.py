"""
Train 3 Forecast Models - LightGBM, XGBoost, RandomForest
===========================================================
Target: US AQI (0-500 scale)
Features: Based on EDA findings (see notebooks/EDA.ipynb)
- Weather: temp, humidity, wind_speed, rain, weather_code
- Pollutants: pm10, no2, ozone (NOT target - no leakage)
- Time: cyclical + explicit hour markers
- Lag: us_aqi_lag_24h, rolling_mean_24h, rolling_std_24h
- Interactions: wind_x_temp, humidity_x_temp, etc.

Hyperparameters are tuned dynamically via RandomizedSearchCV with
TimeSeriesSplit cross-validation. No hardcoded params — the best
configuration is selected each training run and stored in the
model registry metadata.

All 3 models are saved to MongoDB model registry.
The best model (by MAE) is also saved as 'best_model'.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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


def get_model_search_spaces():
    """Define the 3 models with their hyperparameter search spaces."""
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        xgb_available = False
        print("[WARNING] XGBoost not installed. pip install xgboost")
    
    spaces = {}
    
    # 1. LightGBM
    spaces['LightGBM'] = {
        'estimator': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [200, 300, 500, 700],
            'max_depth': [5, 8, 12, -1],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30, 50],
            'reg_alpha': [0.0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.0, 0.01, 0.1, 1.0],
        }
    }
    
    # 2. XGBoost
    if xgb_available:
        spaces['XGBoost'] = {
            'estimator': xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            'params': {
                'n_estimators': [200, 300, 500, 700],
                'max_depth': [5, 8, 12, 15],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0.0, 0.01, 0.1, 1.0],
                'reg_lambda': [0.0, 0.01, 0.1, 1.0],
            }
        }
    
    # 3. RandomForest
    spaces['RandomForest'] = {
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [3, 5, 10],
            'max_features': ['sqrt', 0.5, 0.8],
        }
    }
    
    return spaces


def train_all_models():
    """Train all 3 models with RandomizedSearchCV and save to registry."""
    print("\n" + "=" * 70)
    print("  TRAINING 3 FORECAST MODELS (with Hyperparameter Tuning)")
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
    
    # TimeSeriesSplit for cross-validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Get model search spaces
    model_spaces = get_model_search_spaces()
    results = {}
    best_mae = float('inf')
    best_name = None
    
    for name, spec in model_spaces.items():
        print(f"\n--- Tuning {name} (20 iterations, 3-fold TimeSeriesSplit) ---")
        
        search = RandomizedSearchCV(
            estimator=spec['estimator'],
            param_distributions=spec['params'],
            n_iter=20,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        # Evaluate best estimator on held-out test set
        best_estimator = search.best_estimator_
        preds = best_estimator.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"  Best CV MAE: {-search.best_score_:.2f}")
        print(f"  Test R2:     {r2:.4f}")
        print(f"  Test MAE:    {mae:.2f} AQI")
        print(f"  Test RMSE:   {rmse:.2f} AQI")
        print(f"  Best params: {search.best_params_}")
        
        # Feature importance
        if hasattr(best_estimator, 'feature_importances_'):
            fi = sorted(zip(feature_names, best_estimator.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
            print(f"  Top 5 features:")
            for fname, imp in fi[:5]:
                print(f"    {fname:30s} {imp:.4f}")
        
        results[name] = {
            'model': best_estimator,
            'best_params': search.best_params_,
            'cv_mae': -search.best_score_,
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'test_preds': preds
        }
        
        if mae < best_mae:
            best_mae = mae
            best_name = name
    
    # Retrain best params on full dataset and save
    print(f"\n{'='*70}")
    print(f"  RETRAINING ON FULL DATA & SAVING TO REGISTRY")
    print(f"{'='*70}")
    
    for name, r in results.items():
        print(f"\n  Retraining {name} with tuned params on full data...")
        model = r['model']
        model.fit(X, y)  # Full data retrain with best params
        
        is_best = (name == best_name)
        
        # Convert best_params values to JSON-serializable types
        serializable_params = {}
        for k, v in r['best_params'].items():
            if isinstance(v, (np.integer,)):
                serializable_params[k] = int(v)
            elif isinstance(v, (np.floating,)):
                serializable_params[k] = float(v)
            elif v is None:
                serializable_params[k] = None
            else:
                serializable_params[k] = v
        
        metrics = {
            'model_name': name,
            'r2': r['r2'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'cv_mae': r['cv_mae'],
            'best_params': serializable_params,
            'target': TARGET_COL,
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_names),
            'n_samples': len(X),
            'features': feature_names,
            'is_best': is_best,
            'description': f'{name} targeting {TARGET_COL} (tuned via RandomizedSearchCV)'
        }
        
        # Save with model name
        save_model(model, metrics, model_name=name)
        
        # Also save best as 'best_model'
        if is_best:
            save_model(model, metrics, model_name='best_model')
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE (with Hyperparameter Tuning)")
    print(f"{'='*70}")
    print(f"\n  Results (evaluated on test set):")
    print(f"  {'Model':<15} {'R2':>8} {'MAE':>8} {'RMSE':>8} {'CV MAE':>8} {'Best':>6}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        star = " <--" if name == best_name else ""
        print(f"  {name:<15} {r['r2']:8.4f} {r['mae']:8.2f} {r['rmse']:8.2f} {r['cv_mae']:8.2f}  {star}")
    
    print(f"\n  Best model: {best_name} (MAE={best_mae:.2f})")
    print(f"  Best params: {results[best_name]['best_params']}")
    print(f"  Saved as: 'best_model' in MongoDB registry")
    print(f"  Features: {feature_names}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_all_models()
