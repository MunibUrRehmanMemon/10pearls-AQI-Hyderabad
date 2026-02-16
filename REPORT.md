# AQI Predictor — Project Report

**Project:** Air Quality Index Prediction System  
**Location:** Hyderabad, Sindh, Pakistan  
**Author:** Munib Ur Rehman Memon  
**Organization:** 10Pearls — Internship Project

---

## 1. Background

Air quality has become a growing concern for urban areas across South Asia. Hyderabad, the second-largest city in Sindh, regularly experiences periods of moderate to unhealthy air quality, driven primarily by fine particulate matter (PM2.5). While international monitoring services provide current readings, residents and local stakeholders often lack access to localized short-term forecasts.

This project was developed to fill that gap — a lightweight, automated system that predicts the US AQI for Hyderabad over the next 72 hours using publicly available weather and atmospheric data.

---

## 2. Objective

Build a fully automated pipeline that:
- Collects hourly weather and air quality observations from Open-Meteo
- Trains machine learning models to predict future AQI values
- Serves a public-facing dashboard showing historical readings and forecasts
- Runs continuously through scheduled CI/CD workflows

---

## 3. Data Source

All data comes from **Open-Meteo**, a free, open-source weather and air quality API. The API provides:
- Hourly weather variables: temperature, humidity, wind speed, precipitation, weather code
- Hourly air quality variables: PM2.5, PM10, NO₂, ozone, and US AQI

The air quality data is derived from the **Copernicus Atmosphere Monitoring Service (CAMS)**, which runs global atmospheric composition models at 40 km resolution. CAMS updates once daily, which introduces a roughly 24-hour lag between the latest available measured data and the current time. The system accounts for this delay when generating forecasts.

---

## 4. Approach

### Data Ingestion
A three-step ingestion pipeline runs hourly:
1. **Archive API** fetches verified historical data (available with a 2–5 day lag)
2. **Forecast API with past_days** fills the gap between archive and present
3. **Weather Forecast API** fetches the next 72 hours of forecast weather inputs

All ingested records are stored in MongoDB with upsert logic to avoid duplicates.

### Feature Engineering
The model uses 28 features grouped into five categories:
- **Weather:** temperature, humidity, wind speed, rain, weather code
- **Pollutants:** PM10, NO₂, ozone (these are model inputs, not the target)
- **Temporal:** cyclical hour/day/month encodings, plus binary indicators for rush hours and weekends
- **Lag/Rolling:** 24-hour lag, 24-hour rolling mean, and 24-hour rolling standard deviation of AQI
- **Interactions:** cross-terms between weather variables and pollutants (e.g., wind × temperature)

The target variable (US AQI) and PM2.5 are explicitly excluded from the input features to prevent data leakage. PM2.5 is excluded because it is the primary component from which US AQI is calculated — including it would essentially give the model the answer.

### Model Training
Three gradient boosting models are trained and compared:
- **LightGBM** — fast, handles categorical-like features well
- **XGBoost** — strong regularization, robust generalization
- **RandomForest** — ensemble baseline with low overfitting risk

Training uses an 80/20 chronological split (no shuffling, since this is time-series data). The best model is automatically selected based on the lowest Mean Absolute Error on the test set and saved to MongoDB via GridFS.

### Forecasting
Predictions are generated recursively, one hour at a time:
1. The system queries Open-Meteo to find the last available real data point
2. Starting from the next hour, it fetches forecast weather inputs
3. For each hour, it constructs features from predicted weather plus historical AQI values
4. The predicted AQI is fed back into the history buffer for the next step's lag features

This recursive approach allows the model to generate predictions that extend beyond the available weather forecast horizon while maintaining consistency with its own prior predictions.

---

## 5. Results

The best performing model (LightGBM) achieved the following on the held-out test set:

| Metric | Value |
|--------|-------|
| R² | 0.874 |
| MAE | 6.58 AQI points |
| RMSE | 9.13 AQI points |

Given that the AQI scale ranges from 0 to 500, a mean absolute error of approximately 7 points represents a reasonably accurate forecast. The model correctly identifies trends and category transitions between Good, Moderate, and Unhealthy for Sensitive Groups — the three most common categories observed in Hyderabad.

The most influential features, in order, are the 24-hour rolling mean, 24-hour rolling standard deviation, 24-hour lag, PM10, and the humidity-temperature interaction. This aligns with the exploratory analysis, which showed strong autocorrelation in AQI values (r = 0.74 at 24-hour lag) and meaningful correlations between AQI and both wind speed (r = -0.40) and temperature (r = -0.39).

---

## 6. Deployment

The system is deployed as follows:
- **Dashboard:** Streamlit Cloud, publicly accessible
- **Data & Models:** MongoDB Atlas (cloud-hosted)
- **CI/CD:** GitHub Actions
  - Hourly data ingestion (every hour at minute 5)
  - Daily model retraining (1:00 AM PKT)

The dashboard uses Plotly for interactive charts, showing the historical AQI, the model's predictions, and Open-Meteo's own forecast as a reference line. AQI category zones are displayed as colored bands for quick visual interpretation.

---

## 7. Limitations

- **Spatial resolution:** The data represents a 40 km grid cell, not a specific monitoring station. Actual readings at any given point in the city may differ.
- **Data lag:** CAMS data has a 24-hour delay, meaning the model must predict the gap between the last real observation and the present before extending into the future.
- **Single location:** The system is currently configured for Hyderabad only. Expanding to other cities would require minimal code changes but separate data pipelines.
- **No ground truth validation:** The model is trained and evaluated against CAMS-derived AQI values, not local monitoring station data. Differences between model-derived and station-measured PM2.5 are expected.

---

## 8. Future Improvements

- Integrate local ground monitoring station data when available
- Add SHAP-based explanations to the dashboard for transparency
- Expand coverage to include Karachi, Lahore, and Islamabad
- Explore transformer-based time-series models for longer-range forecasts
- Add push notifications for hazardous AQI events

---

## 9. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | LightGBM, XGBoost, scikit-learn |
| Data Source | Open-Meteo (CAMS) |
| Database | MongoDB Atlas |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Hosting | Streamlit Cloud |
