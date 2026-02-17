import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Global Variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "10pearls-aqi")

# Hyderabad, Sindh Coordinates
LAT = 25.3960
LON = 68.3578

# Collections
FEATURE_COLLECTION = "hyderabad_features"
MODEL_COLLECTION = "model_registry"