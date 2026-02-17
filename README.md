# AQI Predictor — Hyderabad, Sindh, Pakistan

A machine learning system that predicts the **US Air Quality Index (AQI)** for Hyderabad, Sindh, Pakistan over a **72-hour horizon**. It automatically ingests hourly weather and air quality data from Open-Meteo, trains gradient boosting models with hyperparameter tuning, and serves real-time predictions through an interactive Streamlit dashboard.

**[View Live Dashboard](https://10pearls-aqi-hyderabad.streamlit.app/)**

---

## What It Does

Every hour, the system pulls the latest weather and air quality readings from Open-Meteo. Once a day at 1 AM PKT, it retrains three ML models using `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation, picks the best one by MAE, and stores it in MongoDB. The dashboard then loads that best model and runs recursive 72-hour predictions — no manual intervention needed.

The predictions are visualized alongside Open-Meteo's own AQI forecast so you can see how our model compares against the data source itself.

---

## Model Performance

Three models are trained daily with automated hyperparameter tuning. The best model is selected dynamically based on the lowest Mean Absolute Error:

| Model | R² | MAE | RMSE | Status |
|-------|-----|-----|------|--------|
| LightGBM | 0.871 | 6.70 | 9.27 | — |
| **XGBoost** | 0.872 | 6.54 | 9.23 | Best |
| RandomForest | 0.852 | 7.10 | 9.94 | — |

> These scores reflect the latest training run. Since training happens daily with fresh data and randomized hyperparameter search, exact values shift slightly over time as the model adapts to changing conditions.

On a 0–500 AQI scale, a MAE of ~6.5 points means the model is typically within one sub-category of the true reading. It reliably captures transitions between Good, Moderate, and Unhealthy for Sensitive Groups — the three most common categories in Hyderabad. After hyperparameter tuning, all three models perform closely, with XGBoost edging ahead.

---

## Features

- **28 engineered features** across 6 groups (weather, pollutants, cyclical time, explicit time, lag/rolling, interactions)
- **Hyperparameter tuning** via `RandomizedSearchCV` (20 iterations per model, `TimeSeriesSplit` CV)
- **Recursive forecasting** — predictions feed back into lag features for the next hour
- **Real data boundary detection** — the system knows where measured data ends and forecasts begin
- **Interactive dashboard** with AQI category zones, model comparison, and SHAP analysis
- **Fully automated** — no manual steps after initial deployment

---

## Architecture

```
Open-Meteo APIs (Weather + Air Quality)
         │
         ▼
  src/features/ingest_data.py        ← Hourly via GitHub Actions
         │
         ▼
  MongoDB Atlas
  ├── hyderabad_features              (historical data)
  ├── weather_forecast                (72h forecast inputs)
  └── model_registry + GridFS         (trained models + metadata)
         │
         ▼
  src/models/train_forecast_model.py ← Daily at 1 AM PKT via GitHub Actions
  (RandomizedSearchCV + TimeSeriesSplit)
         │
         ▼
  src/prediction/clean_forecast.py   ← Recursive 72h prediction
         │
         ▼
  src/app/dashboard.py               ← Streamlit Cloud (inference only)
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/MunibUrRehmanMemon/10pearls-AQI-Hyderabad.git
cd 10pearls-AQI-Hyderabad

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGO_URI="mongodb+srv://..."
export DB_NAME="aqi_predictor"

# Run data ingestion
python -m src.features.ingest_data

# Train models (with hyperparameter tuning)
python -m src.models.train_forecast_model

# Launch dashboard
streamlit run src/app/dashboard.py
```

---

## CI/CD (GitHub Actions)

| Workflow | Schedule | What It Does |
|----------|----------|-------------|
| **Hourly Ingestion** | Every hour at :05 | Fetches latest weather + AQI data from Open-Meteo, upserts into MongoDB |
| **Daily Training** | 1:00 AM PKT (20:00 UTC) | Re-ingests data, trains all 3 models with `RandomizedSearchCV`, saves best to registry |

Both workflows use `MONGO_URI` and `DB_NAME` from GitHub Secrets.

---

## Project Structure

```
├── src/
│   ├── config.py                      # Coordinates, DB settings, collection names
│   ├── database.py                    # MongoDB connection
│   ├── features/
│   │   ├── ingest_data.py             # 3-step ingestion (archive + recent + forecast)
│   │   ├── feature_engineering.py     # 28-feature pipeline
│   │   └── forecast.py               # Weather forecast fetching
│   ├── models/
│   │   ├── train_forecast_model.py    # RandomizedSearchCV training for 3 models
│   │   └── registry.py               # GridFS model storage + metadata
│   ├── prediction/
│   │   └── clean_forecast.py          # Recursive 72h AQI forecast
│   └── app/
│       └── dashboard.py               # Streamlit dashboard (inference only)
├── notebooks/
│   └── EDA.ipynb                      # Exploratory analysis + tuning validation
├── .github/workflows/
│   ├── hourly_ingestion.yml
│   └── daily_training.yml
├── .streamlit/config.toml             # Dark theme config
├── requirements.txt
├── REPORT.md                          # Detailed project report
└── README.md
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGO_URI` | MongoDB Atlas connection string |
| `DB_NAME` | Database name (default: `aqi_predictor`) |

Set these as **GitHub Secrets** for CI/CD and as **Streamlit Cloud secrets** for the dashboard.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | LightGBM, XGBoost, scikit-learn (RandomForest) |
| Hyperparameter Tuning | RandomizedSearchCV + TimeSeriesSplit |
| Data Source | Open-Meteo (CAMS / Copernicus) |
| Database | MongoDB Atlas (features + model registry via GridFS) |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Hosting | Streamlit Cloud |

---

**Author:** Munib Ur Rehman Memon  
**Organization:** 10Pearls — Internship Project  
**Live App:** [https://10pearls-aqi-hyderabad.streamlit.app/](https://10pearls-aqi-hyderabad.streamlit.app/)
