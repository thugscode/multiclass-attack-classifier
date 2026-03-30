"""
FastAPI Backend for Network Intrusion Detection System
Provides REST API endpoints for making predictions using trained ML models
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
import io
from typing import List, Dict, Any
import joblib

# Add src path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_model_and_scaler, predict_with_confidence, predict_batch, batch_prediction_from_csv

# Initialize FastAPI app
app = FastAPI(
    title="Network Intrusion Detection System API",
    description="ML-based API for classifying network traffic as benign or malicious",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and scaler instances
model = None
scaler = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single sample prediction - expects 82 network features"""
    features: List[float]
    
    class Config:
        example = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 77  # 82 features
        }

class PredictionResponse(BaseModel):
    """Response model for prediction result"""
    prediction: str  # attack type (Benign, DoS, DDoS, Probe, R2L, U2R)
    confidence: float  # confidence score between 0-1
    model_used: str  # which model made the prediction

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    samples: List[List[float]]
    
    class Config:
        example = {
            "samples": [
                [1.0, 2.0, 3.0] + [0.0] * 79,
                [2.0, 3.0, 4.0] + [0.0] * 79
            ]
        }

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Dict[str, Any]]  # List of predictions for each sample
    total_samples: int
    benign_count: int
    attack_count: int

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: int
    scaler_loaded: bool
    encoder_loaded: bool


# Startup event to initialize model and scaler
@app.on_event("startup")
async def startup_event():
    """Load trained model and scaler from disk"""
    global model, scaler
    try:
        model_path = '../models/random_forest_model.joblib'
        scaler_path = '../models/feature_scaler.joblib'
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        print("✓ Model and scaler loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model = None
        scaler = None


# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if model and scaler are loaded"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not initialized")
    
    return HealthResponse(
        status="healthy",
        models_loaded=1,
        scaler_loaded=scaler is not None,
        encoder_loaded=False
    )


# Single Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_single_sample(request: PredictionRequest):
    """
    Make a single prediction for network traffic classification
    
    - **features**: List of 82 network features
    
    Returns the attack type with confidence score
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not initialized")
    
    try:
        # Convert to numpy array
        sample = np.array(request.features).reshape(1, -1)
        
        # Make prediction with confidence
        result = predict_with_confidence(model, scaler, sample)
        
        if isinstance(result, dict):
            prediction_label = result['Predicted_Class']
            confidence = result['Confidence']
        else:
            prediction_label = result.iloc[0]['Predicted_Class']
            confidence = float(result.iloc[0]['Confidence'])
        
        return PredictionResponse(
            prediction=str(prediction_label),
            confidence=float(confidence),
            model_used="Random Forest"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Prediction Endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_samples(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple network traffic samples
    
    - **samples**: List of samples, each with 82 network features
    
    Returns predictions for all samples with statistics
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not initialized")
    
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")
    
    try:
        # Convert to numpy array
        X = np.array(request.samples)
        
        # Make batch predictions
        results = predict_batch(model, scaler, X)
        
        # Format predictions as list of dicts
        predictions_list = []
        for i, label in enumerate(results):
            predictions_list.append({
                'sample_index': i,
                'prediction': str(label),
                'confidence': 0.0
            })
        
        # Count attack types
        benign_count = sum(1 for p in predictions_list if 'Benign' in p['prediction'] or p['prediction'].lower() == 'benign')
        attack_count = len(predictions_list) - benign_count
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total_samples=len(predictions_list),
            benign_count=benign_count,
            attack_count=attack_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CSV Upload Prediction Endpoint
@app.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with network traffic data and get predictions
    
    - **file**: CSV file with 82 columns (network features)
    
    Returns predictions for all rows in the CSV
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not initialized")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf8")))
        
        # Extract first 87 columns as features
        X = df.iloc[:, :82].values if df.shape[1] >= 82 else df.values
        
        # Make predictions
        results = predict_batch(model, scaler, X)
        
        # Format predictions as list of dicts
        predictions_list = []
        for i, label in enumerate(results):
            predictions_list.append({
                'row_index': i,
                'prediction': str(label),
                'confidence': 0.0
            })
        
        # Count attack types
        benign_count = sum(1 for p in predictions_list if 'Benign' in p['prediction'])
        attack_count = len(predictions_list) - benign_count
        
        return {
            "message": f"Successfully processed {len(predictions_list)} samples",
            "total_samples": len(predictions_list),
            "benign_count": benign_count,
            "attack_count": attack_count,
            "predictions": predictions_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# Model Comparison Endpoint
@app.get("/models/compare")
async def compare_models():
    """
    Get information about the loaded model
    
    Returns details about the trained model
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "models_available": ["Random Forest"],
        "total_models": 1,
        "models": [
            {
                "name": "Random Forest",
                "type": type(model).__name__,
                "n_estimators": getattr(model, 'n_estimators', 'N/A'),
                "n_features": getattr(model, 'n_features_in_', 'N/A'),
                "n_classes": len(getattr(model, 'classes_', []))
            }
        ]
    }


# Feature Information Endpoint
@app.get("/features/info")
async def get_features_info():
    """Get information about expected network features"""
    return {
        "total_features": 82,
        "feature_description": "Network traffic features extracted from flow data (after removing irrelevant columns)",
        "classes": ["Benign", "DoS", "DDoS", "Probe", "R2L", "U2R"],
        "preprocessing": {
            "scaler": "StandardScaler (Z-score normalization)",
            "encoder": "LabelEncoder for classification output"
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Network Intrusion Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "predict_single": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "predict_csv": "/predict-csv (POST)",
            "models_info": "/models/compare (GET)",
            "features_info": "/features/info (GET)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    # Access at http://localhost:8000
    # API documentation at http://localhost:8000/docs
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
