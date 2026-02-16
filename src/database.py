"""
MongoDB connection utility.
"""
from pymongo import MongoClient
import src.config as config


_client = None

def get_db_client():
    """Returns a MongoDB database client (singleton)."""
    global _client
    if _client is None:
        if not config.MONGO_URI:
            raise ValueError("MONGO_URI environment variable is not set.")
        _client = MongoClient(config.MONGO_URI)
        print("[SUCCESS] Connected to MongoDB!")
    return _client[config.DB_NAME]
