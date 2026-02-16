"""
Configuration for AQI Predictor - Hyderabad, Sindh, Pakistan
"""
import os

# Location
CITY = "Hyderabad"
LAT = 25.396
LON = 68.3578

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", os.getenv("MONGODB_URI", ""))
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
FEATURE_COLLECTION = "hyderabad_features"
FORECAST_COLLECTION = "weather_forecast"
