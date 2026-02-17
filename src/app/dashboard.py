"""
Enhanced AQI Prediction Dashboard with Multiple Model Comparison
Displays predictions from the best model, compares all 3 models, and shows SHAP analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import sys
import os
import math

# SHAP imports
import shap
import matplotlib.pyplot as plt

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.registry import load_latest_model, get_all_model_metrics, get_best_model_info
from src.database import get_db_client

# FORCE RELOAD MODULES TO PICK UP LATEST CHANGES
import src.features.feature_engineering
import src.prediction.clean_forecast
import src.features.forecast
import importlib
importlib.reload(src.features.feature_engineering) # RELOAD FIRST
importlib.reload(src.prediction.clean_forecast)    # THEN RELOAD THIS
importlib.reload(src.features.forecast)            # RELOAD DATA FETCHING

from src.features.feature_engineering import engineer_features
from src.features.forecast import fetch_weather_forecast_data
from src.prediction.clean_forecast import clean_forecast  # NEW: Clean forecasting
import src.config as config


# ===== HELPER FUNCTIONS =====

def calculate_us_aqi(pm25):
    """Standard US EPA AQI Calculator with correct truncation and breakpoints."""
    if pm25 < 0:
        pm25 = 0.0
    pm25 = math.floor(pm25 * 10) / 10
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi)
    return 500


def get_aqi_category(aqi):
    """Returns category and color for AQI value."""
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy (SG)", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"


# ===== STREAMLIT APP SETUP =====

st.set_page_config(page_title="Hyderabad AQI Predictor", page_icon="🌤️", layout="wide")

st.title("🌤️ Hyderabad Air Quality Index Predictor (Blended v2.1)")
st.markdown("**Real-time Air Quality Predictions for Hyderabad, Sindh, Pakistan**")
st.markdown("Powered by Machine Learning (RandomForest, XGBoost, LightGBM)")

# ===== LOAD DATA AND MODEL =====

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_system_components():
    """Loads model and historical data from MongoDB."""
    status_text = st.empty()
    
    try:
        from datetime import datetime
        
        status_text.info("Connecting to Model Registry...")
        # Load the best model dynamically (selected during training based on R² score)
        model, model_meta = load_latest_model('best_model')
        
        status_text.info("Fetching historical air quality data...")
        db = get_db_client()
        
        # Get historical data (sort descending to find last entry)
        # We need ALL recent history to display, but specifically to find where to start forecast
        # We fetch last 7 days for chart context
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        data = list(db[config.FEATURE_COLLECTION].find(
            {"time": {"$gte": seven_days_ago}},
            {"_id": 0}
        ).sort("time", 1)) # Ascending order for DataFrame
        
        if not data:
            st.warning("No historical data found.")
            return model, pd.DataFrame(), datetime.now() # Return empty df and current time as last_history_time
            
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        
        # CRITICAL FIX: Strict cutoff at Current Time
        # Open-Meteo sometimes returns future hours in "history" query.
        # We must discard them to avoid showing "Future in Past"
        now = datetime.now()
        df = df[df['time'] <= now]
        
        # Determine the START of the forecast based on LAST ACTUAL DATA
        if not df.empty:
            last_history_time = df['time'].max()
        else:
            last_history_time = datetime.now()
            
        # CRITICAL: Set index to time for engineer_features
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
            
        status_text.success("Historical data loaded.")
        return model, df, last_history_time
        
    except Exception as e:
        status_text.error(f"[ERROR] System Load Error: {e}")
        st.stop()


with st.spinner("Loading AI System..."):
    # Load model and history
    model, df_history, last_history_time = load_system_components()
    
    # FETCH FORECAST DYNAMICALLY (Start from last history + 1 hour)
    # We fetch raw forecast (96h usually) and then filter
    df_weather_forecast_raw = fetch_weather_forecast_data()
    
    # Filter forecast to start AFTER history
    # This ensures "Prediction starts from 5pm" if history ends at 4pm
    if df_weather_forecast_raw is not None and not df_weather_forecast_raw.empty:
        df_weather_forecast = df_weather_forecast_raw[df_weather_forecast_raw['time'] > last_history_time].copy()
        # Take next 72 hours from that point
        df_weather_forecast = df_weather_forecast.head(72)
    else:
        df_weather_forecast = pd.DataFrame()


st.success("[SUCCESS] System Online: Model & Data Loaded")

# ===== MODEL COMPARISON SECTION =====

st.markdown("---")
st.subheader("📊 Model Performance Comparison")

# Get model metrics
model_metrics = get_all_model_metrics()
best_model_info = get_best_model_info()

if model_metrics:
    col1, col2, col3 = st.columns(3)
    
    for idx, model_info in enumerate(model_metrics):
        col = [col1, col2, col3][idx % 3]
        
        with col:
            is_best = model_info.get('is_best', False)
            model_name = model_info['model_name']
            r2 = model_info['r2_score']
            mae = model_info['mae']
            
            if is_best:
                st.success(f"🥇 **{model_name}** (BEST)")
            else:
                st.info(f"**{model_name}**")
            
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MAE", f"{mae:.2f} µg/m³")
            
    # Training date
    if best_model_info:
        train_date = best_model_info.get('training_date')
        if train_date:
            st.caption(f"Last trained: {pd.to_datetime(train_date).strftime('%Y-%m-%d %H:%M')}")
else:
    st.warning("⚠️ No model metrics found. Models need to be trained first.")

# ===== ENGINEER FEATURES FOR HISTORY =====

df_engineered = engineer_features(df_history.copy())

# ===== FORECAST GENERATION WITH REAL WEATHER DATA =====

st.markdown("---")

# Show current PKT time
from datetime import datetime as dt, timezone, timedelta as td
pkt = timezone(td(hours=5))
current_pkt = dt.now(pkt)
st.subheader(f"🔮 72-Hour AQI Forecast (from {current_pkt.strftime('%I:%M %p')} PKT)")


st.info(f"Predictions: **{current_pkt.strftime('%a, %b %d %I %p')}** onwards PKT")

# Fetch real weather forecast from Open-Meteo
with st.spinner("Fetching 3-day weather forecast..."):
    weather_forecast = fetch_weather_forecast_data()

# Generate forecast using CLEAN FORECASTING (weather + time -> predict AQI directly)
# clean_forecast now returns (forecast_df, reference_df, last_real_time)
with st.spinner("Generating AQI predictions..."):
    forecast_result = clean_forecast(
        model=model,
        df_weather_forecast=weather_forecast if weather_forecast is not None and not weather_forecast.empty else None,
        df_history=df_history.reset_index()
    )
    
    # Handle 3-tuple return: (result_df, reference_df, last_real_time)
    if isinstance(forecast_result, tuple) and len(forecast_result) == 3:
        df_forecast_raw, open_meteo_reference, last_real_time_boundary = forecast_result
    elif isinstance(forecast_result, tuple) and len(forecast_result) == 2:
        df_forecast_raw, open_meteo_reference = forecast_result
        last_real_time_boundary = None
    else:
        df_forecast_raw = forecast_result
        open_meteo_reference = pd.DataFrame()
        last_real_time_boundary = None

if df_forecast_raw is None or df_forecast_raw.empty:
    st.error("Forecast generation failed. No predictions available.")
    st.stop()

# Show info about real data boundary
if last_real_time_boundary:
    delay_from_now = (current_pkt.replace(tzinfo=None) - last_real_time_boundary).total_seconds() / 3600
    st.info(
        f"Last **real** AQI reading: **{last_real_time_boundary.strftime('%a %b %d, %I:%M %p')} PKT** "
        f"(~{delay_from_now:.0f}h ago). "
        f"Our model predicts from **{df_forecast_raw['time'].min().strftime('%a %b %d, %I %p')} PKT** onwards."
    )

st.success(f"Generated {len(df_forecast_raw)} hour forecast")

# Model now predicts AQI directly - no conversion needed
df_forecast_raw['type'] = 'Forecast'

# Set time as index
df_forecast = df_forecast_raw.set_index('time')

# Calculate historical AQI (from stored us_aqi or convert from pm2_5)
if 'us_aqi' in df_history.columns:
    df_history['aqi'] = df_history['us_aqi']
else:
    df_history['aqi'] = df_history['pm2_5'].apply(calculate_us_aqi)

# ===== DAILY AVERAGE METRICS =====

st.subheader("📅 Daily Average AQI")

daily_means = df_forecast.groupby(df_forecast.index.date)['aqi'].mean()

cols = st.columns(min(len(daily_means), 4))

for i, (date, avg_aqi) in enumerate(daily_means.items()):
    if i < len(cols):
        category, color = get_aqi_category(avg_aqi)
        
        with cols[i]:
            st.metric(
                label=pd.to_datetime(date).strftime('%A, %b %d'),
                value=f"{int(avg_aqi)} AQI"
            )
            st.markdown(f":{color}[**{category}**]")

# ===== MAIN VISUALIZATION =====

st.markdown("---")
st.subheader("📈 AQI Trend Analysis (US EPA Scale)")

# Prepare display data - show ONLY real data in history (cut at last_real_time_boundary)
if last_real_time_boundary:
    # Only show history up to last REAL data point
    df_display_hist = df_history[df_history.index <= last_real_time_boundary].tail(72).copy()
else:
    df_display_hist = df_history.tail(72).copy()
df_display_hist['type'] = 'History'

# Combine history and forecast
df_chart = pd.concat([
    df_display_hist[['aqi', 'type']],
    df_forecast[['aqi', 'type']]
])

# Create interactive plot with DUAL comparison lines
fig = go.Figure()

# Historical data (gray)
hist_data = df_chart[df_chart['type'] == 'History']
fig.add_trace(go.Scatter(
    x=hist_data.index,
    y=hist_data['aqi'],
    mode='lines',
    name='History',
    line=dict(color='gray', width=2),
    fill='tozeroy',
    fillcolor='rgba(128, 128, 128, 0.2)',
    hovertemplate='<b>%{x|%a, %b %d %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>'
))

# Our Model Forecast (RED)
forecast_data = df_chart[df_chart['type'] == 'Forecast']
fig.add_trace(go.Scatter(
    x=forecast_data.index,
    y=forecast_data['aqi'],
    mode='lines',
    name='Our Model (Predicted)',
    line=dict(color='#ff4b4b', width=3),
    hovertemplate='<b>%{x|%a, %b %d %H:%M}</b><br>Our AQI: %{y:.0f}<extra></extra>'
))

# Open-Meteo AQI Forecast (BLUE) - from reference data returned by clean_forecast
if open_meteo_reference is not None and not open_meteo_reference.empty and 'us_aqi' in open_meteo_reference.columns:
    ref_indexed = open_meteo_reference.set_index('time')
    fig.add_trace(go.Scatter(
        x=ref_indexed.index,
        y=ref_indexed['us_aqi'],
        mode='lines',
        name='Open-Meteo AQI (Reference)',
        line=dict(color='#1f77b4', width=2, dash='dash'),
        hovertemplate='<b>%{x|%a, %b %d %H:%M}</b><br>Open-Meteo AQI: %{y:.0f}<extra></extra>'
    ))

# AQI category zones
fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)

fig.update_layout(
    xaxis_title="Time (PKT)",
    yaxis_title="AQI (US EPA Scale)",
    hovermode='x unified',
    height=500,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        tickformat='%a %b %d<br>%H:%M',
        dtick=12*3600000,
        tickangle=0
    )
)

st.plotly_chart(fig, use_container_width=True)

# ===== DETAILED HOURLY TABLE =====

with st.expander("📋 View Detailed Hourly Forecast"):
    forecast_display = df_forecast.copy()
    forecast_display['Category'] = forecast_display['aqi'].apply(lambda x: get_aqi_category(x)[0])
    forecast_display['PM2.5 (µg/m³)'] = forecast_display['pm2_5'].round(2)
    forecast_display['AQI'] = forecast_display['aqi'].astype(int)
    forecast_display['Time'] = forecast_display.index.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        forecast_display[['Time', 'AQI', 'Category', 'PM2.5 (µg/m³)']],
        use_container_width=True,
        height=400
    )

# ===== SHAP ANALYSIS =====

st.markdown("---")
st.subheader("🔍 Model Explainability (SHAP Analysis)")

with st.expander("Click to view SHAP Analysis - Feature Importance"):
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows which features are most important for predictions.
    This analysis is based on the last 50 hours of data.
    """)
    
    try:
        # Prepare input data with engineered features
        X_input = df_engineered.tail(50).copy()
        
        # Select only features that model was trained on
        feature_cols = [col for col in model.feature_names_in_ if col in X_input.columns]
        X_input_aligned = X_input[feature_cols]
        
        # Add missing features with default values
        for col in model.feature_names_in_:
            if col not in X_input_aligned.columns:
                X_input_aligned[col] = 0
        
        # Reorder to match model
        X_input_aligned = X_input_aligned[model.feature_names_in_]
        
        # Generate SHAP plots
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input_aligned)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Importance (Global)**")
            st.caption("Which features matter most?")
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_input_aligned, plot_type="bar", show=False, max_display=15)
            st.pyplot(fig1)
            plt.close()
            
        with col2:
            st.markdown("**Feature Impact on Predictions**")
            st.caption("How feature values affect predictions")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_input_aligned, show=False, max_display=15)
            st.pyplot(fig2)
            plt.close()
            
    except Exception as e:
        st.error(f"Unable to generate SHAP analysis: {e}")
        st.write("Error details:", str(e))

# ===== FOOTER =====

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📡 Data Source: <a href='https://open-meteo.com' target='_blank'>Open-Meteo API</a></p>
    <p>📍 Location: Hyderabad, Sindh, Pakistan (25.3960°N, 68.3578°E)</p>
    <p>🔄 Models retrain daily at 1:00 AM PKT with updated data</p>
    <p>🤖 Best model selected based on R² score</p>
</div>
""", unsafe_allow_html=True)
