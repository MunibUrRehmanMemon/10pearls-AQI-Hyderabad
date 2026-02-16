# AQI Predictor — Hyderabad, Sindh, Pakistan

A machine learning system that predicts the US Air Quality Index (AQI) for Hyderabad, Sindh, Pakistan over a 72-hour horizon. The system automatically ingests hourly weather and air quality data from Open-Meteo, trains gradient boosting models, and serves predictions through an interactive Streamlit dashboard.

## Live Dashboard

Deployed on Streamlit Cloud. The dashboard displays:
- **Historical AQI** (real measured data from Open-Meteo)
- **72-hour predictions** from our trained LightGBM model
- **Open-Meteo reference line** for comparison
- **AQI category zones** (Good, Moderate, USG, Unhealthy)
- **Model performance metrics** (R², MAE, RMSE for all 3 models)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGO_URI="mongodb+srv://..."
export DB_NAME="aqi_predictor"

# Run data ingestion
python -m src.features.ingest_data

# Train models
python -m src.models.train_forecast_model

# Launch dashboard
streamlit run src/app/dashboard.py
```

## Architecture

```
Open-Meteo APIs (Weather + Air Quality)
         │
         ▼
  src/features/ingest_data.py        ← Hourly via GitHub Actions
         │
         ▼
  MongoDB (hyderabad_features + weather_forecast + model_registry)
         │
         ▼
  src/models/train_forecast_model.py ← Daily via GitHub Actions (1 AM PKT)
         │
         ▼
  src/prediction/clean_forecast.py   ← Recursive 72h prediction
         │
         ▼
  src/app/dashboard.py               ← Streamlit dashboard
```

## Models

Three models are trained and compared. The best performing model is automatically selected:

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| **LightGBM** | 0.874 | 6.58 | 9.13 |
| XGBoost | 0.842 | 7.20 | 10.23 |
| RandomForest | 0.748 | 10.08 | 12.94 |

## CI/CD

- **Hourly Ingestion** — Fetches latest data from Open-Meteo every hour
- **Daily Training** — Retrains all 3 models at 1 AM PKT, selects the best

## Project Structure

```
├── src/
│   ├── config.py                 # Configuration (coords, DB settings)
│   ├── database.py               # MongoDB connection
│   ├── features/
│   │   ├── ingest_data.py        # Data ingestion pipeline
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── train_forecast_model.py # 3-model training
│   │   └── registry.py           # GridFS model storage
│   ├── prediction/
│   │   └── clean_forecast.py     # 72h recursive forecast
│   └── app/
│       └── dashboard.py          # Streamlit dashboard
├── .github/workflows/
│   ├── hourly_ingestion.yml
│   └── daily_training.yml
├── .streamlit/config.toml
├── requirements.txt
└── REPORT.md
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGO_URI` | MongoDB connection string |
| `DB_NAME` | Database name (default: `aqi_predictor`) |

Set these as GitHub Secrets for CI/CD and as Streamlit Cloud secrets for deployment.
