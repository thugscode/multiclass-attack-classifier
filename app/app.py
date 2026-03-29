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
import pickle

# Add src path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import IntrusionDetectionPredictor

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

# Global predictor instance
predictor = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single sample prediction - expects 87 network features"""
    features: List[float]
    
    class Config:
        example = {
            "features": [1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 82  # 87 features
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
                [1.0, 2.0, 3.0] + [0.0] * 84,
                [2.0, 3.0, 4.0] + [0.0] * 84
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


# Startup event to initialize predictor
@app.on_event("startup")
async def startup_event():
    """Initialize the predictor with trained models and preprocessing objects"""
    global predictor
    try:
        predictor = IntrusionDetectionPredictor(
            models_folder='../models',
            scaler_path='../data/processed/friday_scaler.pkl',
            encoder_path='../data/processed/friday_label_encoder.pkl'
        )
        print("✓ Predictor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize predictor: {e}")
        predictor = None


# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if all models and preprocessing objects are loaded"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(predictor.models) if predictor.models else 0,
        scaler_loaded=predictor.scaler is not None,
        encoder_loaded=predictor.encoder is not None
    )


# Single Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a single prediction for network traffic classification
    
    - **features**: List of 87 network features
    
    Returns the attack type with confidence score
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    # Validate input
    if len(request.features) != 87:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 87 features, got {len(request.features)}"
        )
    
    try:
        # Convert to numpy array
        sample = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction_result = predictor.predict_single_sample(sample[0])
        
        # Extract confidence from probabilities if available
        confidence = 0.0
        if prediction_result['probabilities'] is not None:
            confidence = float(np.max(prediction_result['probabilities']))
        
        return PredictionResponse(
            prediction=prediction_result['prediction_label'],
            confidence=confidence,
            model_used=prediction_result['model']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Prediction Endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple network traffic samples
    
    - **samples**: List of samples, each with 87 network features
    
    Returns predictions for all samples with statistics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")
    
    try:
        # Validate all samples have correct feature count
        for i, sample in enumerate(request.samples):
            if len(sample) != 87:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample {i}: Expected 87 features, got {len(sample)}"
                )
        
        # Convert to numpy array
        X = np.array(request.samples)
        
        # Make batch predictions
        batch_result = predictor.predict_batch(X)
        
        # Format predictions as list of dicts
        predictions_list = []
        for i, label in enumerate(batch_result['predictions_labels']):
            predictions_list.append({
                'sample_index': i,
                'prediction': label,
                'confidence': 0.0  # Confidence calculation would require probabilities per sample
            })
        
        # Count attack types
        benign_count = sum(1 for p in predictions_list if p['prediction'] == 'Benign')
        attack_count = len(predictions_list) - benign_count
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total_samples=len(predictions_list),
            benign_count=benign_count,
            attack_count=attack_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CSV Upload Prediction Endpoint
@app.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with network traffic data and get predictions
    
    - **file**: CSV file with 87 columns (network features)
    
    Returns predictions for all rows in the CSV
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf8")))
        
        # Validate feature count
        if df.shape[1] < 87:
            raise HTTPException(
                status_code=400,
                detail=f"CSV has {df.shape[1]} columns, expected at least 87 features"
            )
        
        # Extract first 87 columns as features
        X = df.iloc[:, :87].values
        
        # Make predictions
        batch_result = predictor.predict_batch(X)
        
        # Format predictions as list of dicts
        predictions_list = []
        for i, label in enumerate(batch_result['predictions_labels']):
            predictions_list.append({
                'row_index': i,
                'prediction': label,
                'confidence': 0.0
            })
        
        # Count attack types
        benign_count = sum(1 for p in predictions_list if p['prediction'] == 'Benign')
        attack_count = len(predictions_list) - benign_count
        
        return {
            "message": f"Successfully processed {len(predictions_list)} samples",
            "total_samples": len(predictions_list),
            "benign_count": benign_count,
            "attack_count": attack_count,
            "predictions": predictions_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# Model Comparison Endpoint
@app.get("/models/compare")
async def compare_models():
    """
    Get information about all available models
    
    Returns details about each trained model's performance
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if not predictor.models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    return {
        "models_available": list(predictor.models.keys()),
        "total_models": len(predictor.models),
        "models": [
            {
                "name": name,
                "type": type(model).__name__
            }
            for name, model in predictor.models.items()
        ]
    }


# Feature Information Endpoint
@app.get("/features/info")
async def get_features_info():
    """Get information about expected network features"""
    return {
        "total_features": 87,
        "feature_description": "Network traffic features extracted from flow data",
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
