# AQI Predictor — Project Report

**Project:** Air Quality Index Prediction System  
**Location:** Hyderabad, Sindh, Pakistan  
**Author:** Munib Ur Rehman Memon  
**Organization:** 10Pearls — Internship Project  
**Live Dashboard:** [https://10pearls-aqi-hyderabad.streamlit.app/](https://10pearls-aqi-hyderabad.streamlit.app/)

---

## 1. Background

Air quality has become a growing concern for urban areas across South Asia. Hyderabad, the second-largest city in Sindh, regularly experiences periods of moderate to unhealthy air quality, driven primarily by fine particulate matter (PM2.5). While international monitoring services provide current readings, residents and local stakeholders often lack access to localized short-term forecasts that tell them what to expect over the coming days.

This project was developed to fill that gap — a lightweight, fully automated system that predicts the US AQI for Hyderabad over the next 72 hours using publicly available weather and atmospheric data. The system runs entirely on free-tier infrastructure (GitHub Actions, MongoDB Atlas, Streamlit Cloud) and requires zero manual intervention once deployed.

---

## 2. Objective

Build a fully automated, end-to-end pipeline that:

1. Collects hourly weather and air quality observations from Open-Meteo
2. Engineers meaningful features from raw data (28 features across 6 groups)
3. Trains multiple machine learning models with automated hyperparameter tuning
4. Dynamically selects the best-performing model based on Mean Absolute Error
5. Generates recursive 72-hour AQI forecasts
6. Serves a public-facing dashboard showing historical readings, predictions, and model performance
7. Runs continuously through scheduled CI/CD workflows without human intervention

---

## 3. Data Source

All data comes from **Open-Meteo**, a free, open-source weather and air quality API. The API provides:

- **Hourly weather variables:** temperature, humidity, wind speed, precipitation, weather code
- **Hourly air quality variables:** PM2.5, PM10, NO₂, ozone, and US AQI

The air quality data is derived from the **Copernicus Atmosphere Monitoring Service (CAMS)**, which runs global atmospheric composition models at ~11 km resolution (updated from the older 40 km grids). CAMS updates once daily, which introduces a roughly **20–24 hour lag** between the latest available measured data and the current time. Weather data (temperature, wind, humidity) updates every 1–6 hours and is available with minimal delay.

### Data Ingestion Pipeline

The ingestion system runs in three steps to ensure continuous, gap-free data:

1. **Archive API** — Fetches verified historical data (available with a 2–5 day lag). This is the most reliable source but arrives late.
2. **Forecast API with `past_days`** — Fills the gap between archive end and the present with CAMS model estimates for recent hours.
3. **Weather Forecast API** — Fetches the next 72 hours of forecast weather inputs (temperature, wind, humidity, etc.) needed for generating future predictions.

All ingested records are stored in MongoDB with upsert logic keyed on timestamp, so duplicate entries are automatically handled. As more reliable archive data becomes available, it silently overwrites the earlier estimates.

**Location:** Hyderabad, Sindh, Pakistan  
**Coordinates:** 25.3960°N, 68.3578°E

---

## 4. Approach

### 4.1 Feature Engineering

The model uses **28 features** grouped into six categories. Every feature was justified through the exploratory data analysis (EDA) before being added to the pipeline:

| Group | Count | Features | EDA Justification |
|-------|-------|----------|-------------------|
| **Weather** | 5 | temp, humidity, wind_speed, rain, weather_code | Wind speed (r = -0.40) and temperature (r = -0.39) showed strong negative correlations with AQI. Cold, calm conditions trap pollutants (inversions). |
| **Pollutants** | 3 | pm10, no2, ozone | PM10 (r = +0.45) is the strongest single predictor. These are available from Open-Meteo forecasts, unlike PM2.5. |
| **Cyclical Time** | 6 | hour_sin/cos, dow_sin/cos, month_sin/cos | Sine/cosine encoding preserves the circular nature of time. The EDA showed a clear diurnal AQI pattern (~3.4 point variation, evening peak). |
| **Explicit Time** | 6 | hour, is_night, is_morning_rush, is_afternoon, is_evening_rush, is_weekend | Binary flags for interpretability. Night and morning rush hours showed higher AQI in the EDA. |
| **Lag/Rolling** | 3 | us_aqi_lag_24h, rolling_mean_24h, rolling_std_24h | The autocorrelation analysis revealed strong persistence: r = 0.739 at 24-hour lag. These turned out to be the most important features. |
| **Interactions** | 5 | wind_x_temp, humidity_x_temp, wind_x_humidity, pm10_x_humidity, hour_x_ozone | Capture non-linear relationships discovered in the EDA. For example, high humidity + high PM10 amplifies AQI more than either alone. |

**Excluded from features:**
- **us_aqi** — This is the target variable
- **pm2_5** — Directly derived from US AQI via the EPA formula (including it would be data leakage; see Section 6.3 for details)
- **city** — Constant value (always Hyderabad)

### 4.2 Target Variable: US AQI

The target variable is the **US Air Quality Index** as defined by the US EPA. Open-Meteo provides this value directly, calculated from PM2.5 concentrations using the standard EPA breakpoint formula:

$$AQI = \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \times (C - C_{low}) + I_{low}$$

Where $C$ is the truncated PM2.5 concentration (µg/m³) and the breakpoints are:

| PM2.5 (µg/m³) | AQI Range | Category |
|----------------|-----------|----------|
| 0.0 – 12.0 | 0 – 50 | Good |
| 12.1 – 35.4 | 51 – 100 | Moderate |
| 35.5 – 55.4 | 101 – 150 | Unhealthy for Sensitive Groups |
| 55.5 – 150.4 | 151 – 200 | Unhealthy |
| 150.5 – 250.4 | 201 – 300 | Very Unhealthy |
| 250.5 – 350.4 | 301 – 400 | Hazardous |
| 350.5 – 500.4 | 401 – 500 | Hazardous |

### 4.3 Model Training with Hyperparameter Tuning

Three gradient boosting models are trained and compared:

- **LightGBM** — Fast training, handles high-cardinality features well, native support for categorical-like features
- **XGBoost** — Strong regularization (L1/L2), robust generalization, widely validated in competitions
- **RandomForest** — Ensemble baseline with inherently low overfitting risk and no learning rate to tune

Each model is tuned using **RandomizedSearchCV** with **TimeSeriesSplit** cross-validation (3 folds) to respect temporal ordering — we never validate on data that comes before the training window. The search runs 20 random parameter combinations per model, scoring by negative Mean Absolute Error.

**Why RandomizedSearchCV instead of GridSearchCV?**
- A full grid search across all three models would take 1–2 hours
- Randomized search with 20 iterations per model completes in 25+ minutes
- This matters because the training pipeline runs daily on GitHub Actions with a 30-minute timeout
- Research by Bergstra & Bengio (2012) showed that randomized search finds comparably good hyperparameters in a fraction of the time

The hyperparameter search spaces include:

| Model | Parameters Tuned |
|-------|-----------------|
| LightGBM | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda |
| RandomForest | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |

**Important:** The best parameters found during training are **not hardcoded**. The pipeline runs its own `RandomizedSearchCV` every day, so parameters naturally adapt as the data distribution shifts over time (seasonal changes, new pollution sources, etc.).

Training uses an **80/20 chronological split** — no shuffling, since this is time-series data. The most recent 20% of data serves as the test set, which is the most realistic way to evaluate forecasting performance.

### 4.4 Model Registry

After training, all three models are saved to MongoDB via GridFS along with their metadata:
- R², MAE, RMSE scores
- Best hyperparameters found by RandomizedSearchCV
- Cross-validation MAE
- Training timestamp
- Feature list
- Whether this model is the current best

The model with the lowest test MAE is flagged as `is_best = True` and saved separately as `best_model` for the dashboard to load.

### 4.5 Recursive Forecasting

Predictions are generated recursively, one hour at a time, over a 72-hour horizon:

1. The system queries MongoDB to find the last available real (measured) data point
2. Starting from the next hour, it fetches forecast weather inputs from the stored weather forecast
3. For each hour, it constructs the full 28-feature vector using:
   - Forecast weather values for that hour
   - The AQI value from 24 hours prior (from history or earlier predictions)
   - Rolling mean and standard deviation updated with each new prediction
4. The predicted AQI is appended to the history buffer and used as input for the next step

This recursive approach means that prediction errors can compound over time, but the strong autocorrelation in AQI data (r = 0.74 at 24h lag) works in our favor — the lag features provide a strong anchor that keeps predictions realistic. In practice, the model maintains reliable accuracy through the full 72-hour window.

---

## 5. Results

### Model Performance (After Hyperparameter Tuning)

| Model | Test R² | Test MAE | Test RMSE | Status |
|-------|---------|----------|-----------|--------|
| **LightGBM** | **0.875** | **6.57** | **9.11** | **Best** |
| XGBoost | 0.871 | 6.76 | 9.27 | — |
| RandomForest | 0.852 | 7.07 | 9.93 | — |

> *Scores from the latest production training run using EDA-tuned hyperparameters. All three models perform closely after proper tuning.*

The best performing model (**LightGBM**) achieved a test MAE of **6.57 AQI points** and R² of **0.875**.

On a 0–500 AQI scale, a MAE of ~6.6 points means the model is typically within one sub-category of the true reading. It reliably captures transitions between Good (0–50), Moderate (51–100), and Unhealthy for Sensitive Groups (101–150) — the three most common categories observed in Hyderabad.

### Most Important Features

The feature importance analysis (from the best LightGBM model) confirms the EDA findings:

1. **us_aqi_rolling_mean_24h** — The 24-hour rolling average anchors predictions to recent trends
2. **us_aqi_rolling_std_24h** — Captures volatility; high std means AQI is changing rapidly
3. **us_aqi_lag_24h** — Yesterday's AQI at the same hour (r = 0.739)
4. **pm10** — The strongest pollutant predictor (r = +0.45 with AQI)
5. **humidity_x_temp** — Interaction term capturing weather conditions that trap pollutants

These align perfectly with the exploratory analysis: AQI in Hyderabad is primarily driven by persistence (yesterday's air quality), atmospheric conditions (temperature inversions, calm winds), and particulate matter concentrations.

---

## 6. Challenges and Solutions

### 6.1 Feature Store and Model Registry: Hopsworks to MongoDB

**Challenge:** The original project plan called for using **Hopsworks** as both the feature store and model registry. Hopsworks provides a managed platform for storing feature groups, training datasets, and model artifacts — it's a clean, purpose-built solution for ML pipelines. However, during development, Hopsworks **discontinued their free tier**, making it inaccessible for this project.

**Solution:** I switched to **MongoDB Atlas** (free tier, 512 MB) to handle both responsibilities:
- **Feature storage:** All ingested weather and air quality data is stored in MongoDB collections (`hyderabad_features`, `weather_forecast`), with upsert logic on timestamps to avoid duplicates. This effectively acts as a feature store — the training pipeline reads directly from these collections.
- **Model registry:** Trained models are serialized with `pickle` and stored in MongoDB's **GridFS** (which handles files larger than the 16 MB document limit). Each model is stored with full metadata — scores, hyperparameters, training date, feature list, and a flag indicating whether it's the current best. The dashboard loads the `best_model` from GridFS at startup.

This approach actually turned out to be simpler to maintain than Hopsworks would have been. MongoDB Atlas's free tier provides enough storage for this use case, and having everything in one database (data + models) reduces the number of external dependencies.

### 6.2 Open-Meteo Data Delay: What Exactly Are We Predicting?

**Challenge:** Open-Meteo's air quality data comes from the **Copernicus Atmosphere Monitoring Service (CAMS)**, which updates once daily. This means there's always a **~20–24 hour gap** between the current time and the last available measured AQI reading. When I first discovered this, it created real confusion about the prediction task: should the model predict the next 3 calendar days? The next 72 hours from now? Or should it first fill in the missing hours and then forecast forward?

**Solution:** The system takes the most practical approach:
1. It detects the **last real data point** — the most recent hour with an actual measured AQI value
2. From that point forward, it predicts hour-by-hour recursively for the next 72 hours
3. The dashboard clearly shows where real data ends and predictions begin (with a labeled dividing line)

This means the "72-hour forecast" actually covers roughly 24 hours of gap-filling (between the last real reading and now) plus ~48 hours of true future prediction. The dashboard communicates this honestly — it shows "Last real AQI reading: [time]" so users understand the boundary. This turned out to be the right approach because the gap hours still use real forecast weather data from Open-Meteo, so the predictions through the gap period are quite accurate.

### 6.3 Target Variable: PM2.5 vs US AQI

**Challenge:** This was the biggest technical challenge of the project. My initial approach was to predict **PM2.5 concentration** (µg/m³) as the target variable, then convert the predictions to AQI using the standard EPA breakpoint formula. This seemed logical — PM2.5 is the raw measurement, and AQI is just a transformation of it.

The problem was devastating: **the model struggled badly with PM2.5 as the target.** Here's why:

- PM2.5 values are highly dependent on their own lag features. The autocorrelation is extremely strong — once the model gets the first predicted hour slightly wrong, that error feeds into the lag features for the next hour, which compounds the error further.
- The lag features of PM2.5 dominated everything else. Weather features (temperature, wind speed) and time features barely contributed. The model essentially learned "PM2.5 tomorrow ≈ PM2.5 today" and nothing else.
- When the model predicted PM2.5 wrong for a single hour in the recursive forecast, every subsequent hour inherited that error through the lag features. A small initial mistake of 5 µg/m³ could snowball into a 30+ µg/m³ error by hour 72.
- The AQI conversion formula amplified these errors further — because of the breakpoint structure, a small PM2.5 error near a breakpoint boundary could cause the AQI to jump by 20–30 points.

**Solution:** I replaced the target variable from **PM2.5** to **US AQI** directly. Instead of predicting the raw concentration and converting, we predict the AQI value that Open-Meteo already computes using the official EPA formula:

$$AQI = \frac{I_{high} - I_{low}}{C_{high} - C_{low}} \times (C - C_{low}) + I_{low}$$

And critically, we **dropped PM2.5 from the input features entirely.** Since US AQI is directly calculated from PM2.5, including PM2.5 as a feature would be data leakage — you'd essentially be giving the model the answer.

This change made a dramatic difference:
- **The model started learning from all features**, not just lag values. Weather (temperature, wind), time of day, pollutants (PM10, NO₂, ozone) — everything contributed.
- **Recursive forecast errors stopped compounding** as aggressively, because AQI on a 0–500 scale has more natural variance that the model could learn from, rather than being dominated by a single lag term.
- **Predictions became very close to Open-Meteo's own forecast**, which was the ultimate validation. Our model, trained on the same data source, produces forecasts that track the Open-Meteo reference line closely on the dashboard.

The lesson: sometimes the "cleaner" approach (predicting the raw measurement) isn't the right one. The AQI scale, despite being a derived metric, turned out to be a much better prediction target because it distributes the model's learning across all features rather than concentrating it in lag dependencies.

---

## 7. Deployment

The system is deployed as a fully automated pipeline with no manual steps:

### Infrastructure

| Component | Service | Cost |
|-----------|---------|------|
| **Dashboard** | Streamlit Cloud | Free |
| **Database** | MongoDB Atlas (M0, 512MB) | Free |
| **CI/CD** | GitHub Actions | Free (2,000 min/month) |
| **Code** | GitHub | Free |

### CI/CD Workflows

| Workflow | Schedule | Duration | What It Does |
|----------|----------|----------|-------------|
| **Hourly Ingestion** | Every hour at :05 | ~1 min | Fetches latest weather + AQI data, upserts into MongoDB |
| **Daily Training** | 1:00 AM PKT (20:00 UTC) | ~15 min | Re-ingests fresh data, trains all 3 models with RandomizedSearchCV, saves best to registry |

Both workflows use `MONGO_URI` and `DB_NAME` from GitHub Secrets. The daily training workflow also runs data ingestion first to ensure models are trained on the absolute latest data.

### Dashboard

The Streamlit dashboard performs **inference only** — it never trains models. On load, it:
1. Reads the `best_model` from MongoDB GridFS
2. Fetches historical AQI data and weather forecasts from MongoDB
3. Runs the recursive 72-hour prediction
4. Displays everything with interactive Plotly charts

The dashboard caches the model and data for 1 hour (`ttl=3600`) to avoid hitting MongoDB on every page refresh.

**Features displayed:**
- Historical AQI time series with category color bands
- 72-hour prediction line with confidence context
- Open-Meteo reference forecast for comparison
- Clear "Last real AQI reading" boundary marker
- Model performance table (R², MAE, RMSE for all 3 models)
- SHAP-based feature importance analysis

---

## 8. Exploratory Data Analysis (EDA)

A comprehensive EDA was conducted in `notebooks/EDA.ipynb` to validate every design decision. Key findings:

1. **AQI Distribution:** Predominantly Moderate (51–100), with occasional USG (101–150) episodes. Mean AQI ≈ 70.
2. **Strong Autocorrelation:** r = 0.739 at 24-hour lag — yesterday's AQI is the single strongest predictor. This justified the lag and rolling features.
3. **Weather Correlations:** Wind speed (r = -0.40) and temperature (r = -0.39) are the strongest weather predictors. Cold + calm = worst AQI (temperature inversions trap pollutants).
4. **Diurnal Pattern:** ~3.4 AQI variation across the day. Evening peak, afternoon low. This justified the time-of-day features.
5. **No Weekend Effect:** Weekday and weekend AQI are nearly identical, suggesting AQI in Hyderabad is driven by regional weather patterns, not local traffic.
6. **PM2.5 Leakage:** PM2.5 has near-perfect correlation with US AQI (r ≈ 0.99) because AQI is derived from it. This confirmed the decision to exclude PM2.5 from features.
7. **Continuous Data:** No time gaps in the dataset — hourly ingestion is working correctly.
8. **Hyperparameter Tuning:** RandomizedSearchCV with TimeSeriesSplit was used during EDA to find optimal params. The best params are baked into the training script for fast daily runs, with a `--tune` flag available for re-optimization.

---

## 9. Limitations

- **Spatial resolution:** CAMS data represents a grid cell (~11 km), not a specific monitoring station. Actual AQI at any given point in the city may differ from the grid average.
- **Data lag:** The ~24-hour CAMS delay means the model must predict through the gap before forecasting the actual future. Errors in gap-filling can propagate into future predictions.
- **Single location:** The system is configured for Hyderabad only. Expanding to other cities requires minimal code changes but separate data pipelines and potentially different feature engineering.
- **No ground truth validation:** All training and evaluation is against CAMS-derived AQI values, not local ground monitoring station data. There is no PM2.5 monitoring station in Hyderabad to validate against.
- **Recursive error compounding:** Each hour's prediction feeds into the next hour's lag features. While the strong autocorrelation helps stabilize this, accuracy naturally decreases toward the end of the 72-hour window.

---

## 10. Future Improvements

- Integrate local ground monitoring station data when/if available in Hyderabad
- Add push notifications or alerts for hazardous AQI events
- Expand coverage to include Karachi, Lahore, and Islamabad
- Experiment with transformer-based time-series models (e.g., Temporal Fusion Transformer) for longer-range accuracy
- Add confidence intervals to predictions to communicate uncertainty
- Implement model drift detection to flag when retraining produces significantly worse results

---

## 11. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | LightGBM, XGBoost, scikit-learn (RandomForest) |
| Hyperparameter Tuning | RandomizedSearchCV + TimeSeriesSplit (3-fold) |
| Data Source | Open-Meteo APIs (CAMS / Copernicus) |
| Database | MongoDB Atlas (M0 free tier) — features + model registry via GridFS |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions (hourly ingestion + daily training) |
| Hosting | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 12. Conclusion

This project demonstrates that a reliable, city-level AQI prediction system can be built and deployed entirely on free-tier infrastructure. The combination of Open-Meteo's comprehensive atmospheric data, well-engineered features grounded in EDA findings, and automated daily retraining with hyperparameter tuning produces forecasts that closely track the data source's own predictions — while extending them into a useful 72-hour planning horizon.

The key technical insight was that predicting AQI directly (rather than PM2.5 and converting) produced a fundamentally more robust model. This, combined with the decision to use MongoDB as a unified feature store and model registry after Hopsworks became unavailable, resulted in a simpler and more maintainable system than originally planned.

The system has been running autonomously since deployment, ingesting data every hour and retraining daily, with no manual intervention required.

---

**Author:** Munib Ur Rehman Memon  
**Organization:** 10Pearls — Internship Project  
**Repository:** [github.com/MunibUrRehmanMemon/10pearls-AQI-Hyderabad](https://github.com/MunibUrRehmanMemon/10pearls-AQI-Hyderabad)  
**Live Dashboard:** [https://10pearls-aqi-hyderabad.streamlit.app/](https://10pearls-aqi-hyderabad.streamlit.app/)
