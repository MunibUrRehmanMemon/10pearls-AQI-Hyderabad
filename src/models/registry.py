"""
Model Registry for MongoDB with GridFS
Handles saving and loading multiple models with versioning
"""

import joblib
import io
import gridfs
from datetime import datetime
from src.database import get_db_client
import src.config as config


def save_model(model, metrics, model_name='best_model'):
    """
    Saves a trained model to MongoDB using GridFS.
    
    Args:
        model: Trained sklearn/xgboost/lightgbm model
        metrics: Dictionary of model performance metrics
        model_name: Name identifier for the model
    """
    db = get_db_client()
    fs = gridfs.GridFS(db)
    
    # Serialize model to bytes
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    model_binary = model_buffer.getvalue()
    
    # Create metadata
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "metrics": metrics,
        "version": version,
        "features": getattr(model, "feature_names_in_", []).tolist() if hasattr(model, "feature_names_in_") else [],
        "training_date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "best_params": metrics.get("best_params", {})
    }
    
    # Save to GridFS
    fs.put(model_binary, filename=model_name, metadata=metadata)
    
    print(f"✅ Model '{model_name}' saved to GridFS (Version: {version})")


def save_all_models(models_dict, best_model_name):
    """
    Saves ALL trained models to GridFS with their metrics.
    
    Each model is stored in GridFS with its model_name as the filename,
    allowing independent retrieval. The best model is also saved as 'best_model'.
    
    Args:
        models_dict: Dictionary with structure {'ModelName': {'model': obj, 'metrics': dict}}
        best_model_name: Name of the best performing model
    """
    db = get_db_client()
    fs = gridfs.GridFS(db)
    collection = db[config.MODEL_COLLECTION]
    
    # Clear old comparison data
    try:
        collection.delete_many({})
    except Exception as e:
        print(f"⚠️ Warning: Could not clear old model data: {e}")
    
    # Save metadata and models for ALL models
    timestamp = datetime.now()
    version = timestamp.strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        metrics = model_info['metrics']
        
        # 1. Save metadata to collection
        record = {
            'model_name': model_name,
            'r2_score': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'best_params': metrics.get('best_params', {}),
            'is_best': (model_name == best_model_name),
            'training_date': timestamp,
            'version': version
        }
        
        try:
            collection.insert_one(record)
            print(f"✅ Saved metadata for {model_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save metadata for {model_name}: {e}")
        
        # 2. Save model to GridFS
        try:
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            model_binary = model_buffer.getvalue()
            
            metadata = {
                "metrics": metrics,
                "version": version,
                "features": getattr(model, "feature_names_in_", []).tolist() if hasattr(model, "feature_names_in_") else [],
                "training_date": timestamp.isoformat(),
                "model_type": type(model).__name__,
                "is_best": (model_name == best_model_name),
                "best_params": metrics.get("best_params", {})
            }
            
            # Save with model_name as filename (e.g., 'RandomForest', 'XGBoost', 'LightGBM')
            fs.put(model_binary, filename=model_name, metadata=metadata)
            print(f"✅ Saved {model_name} model to GridFS")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save {model_name} to GridFS: {e}")


def load_latest_model(model_name='best_model'):
    """
    Loads the latest version of a model from GridFS.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (model, metadata_dict)
    """
    db = get_db_client()
    fs = gridfs.GridFS(db)
    
    try:
        # Get latest version
        grid_out = fs.get_last_version(filename=model_name)
        
        # Read and deserialize
        model_binary = grid_out.read()
        model = joblib.load(io.BytesIO(model_binary))
        
        # Extract metadata
        metadata = dict(grid_out.metadata) if grid_out.metadata else {}
        version = metadata.get('version', 'unknown')
        model_type = metadata.get('model_type', 'unknown')
        print(f"[OK] Loaded {model_type} (Version: {version})")
        
        return model, metadata
        
    except gridfs.errors.NoFile:
        raise Exception(f"[ERROR] No model '{model_name}' found in registry. Train models first!")


def get_all_model_metrics():
    """
    Retrieves metrics for all models from the latest training session.
    
    Returns:
        List of dictionaries containing model metrics
    """
    db = get_db_client()
    collection = db[config.MODEL_COLLECTION]
    
    # Get the latest training session
    latest_doc = collection.find_one(sort=[('training_date', -1)])
    
    if not latest_doc:
        return []
    
    latest_date = latest_doc['training_date']
    
    # Get all models from that session
    models = list(collection.find(
        {'training_date': latest_date},
        {'_id': 0}
    ).sort('r2_score', -1))
    
    return models


def get_best_model_info():
    """
    Gets information about the best performing model.
    
    Returns:
        Dictionary with best model information
    """
    db = get_db_client()
    collection = db[config.MODEL_COLLECTION]
    
    best_model = collection.find_one(
        {'is_best': True},
        {'_id': 0},
        sort=[('training_date', -1)]
    )
    
    return best_model if best_model else {}