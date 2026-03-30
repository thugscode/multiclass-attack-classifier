"""
Prediction Module for Network Intrusion Detection
==================================================
This module provides functions to make predictions on new, unseen network traffic data.

It includes:
- Loading trained model and preprocessing objects
- Making predictions on new data
- Generating predictions with confidence scores
- Batch prediction processing
- Real-time prediction capability

Key functions:
- load_model_and_scaler(): Load trained model and preprocessing objects
- predict_single(): Predict label for a single sample
- predict_batch(): Predict labels for multiple samples
- predict_with_confidence(): Get predictions with confidence scores
- predict_traffic_flow(): Predict attack type for network flow
"""

import joblib
import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Tuple, Dict


def load_model_and_scaler(model_path, scaler_path):
    """
    Load trained Random Forest model and feature scaler from disk.
    
    These saved objects are essential for making predictions on new data:
    - Model: Contains the trained trees and decision rules
    - Scaler: Contains the mean and std used in training (must apply to new data)
    
    Args:
        model_path (str): Path to saved model (.joblib)
        scaler_path (str): Path to saved scaler (.joblib)
        
    Returns:
        tuple: (trained model, scaler object)
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")
        print(f"  Available attack classes: {model.classes_}")
        return model, scaler
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure model files exist. Run train.py first.")
        raise


def preprocess_input(X, scaler):
    """
    Apply the same preprocessing (scaling) that was used during training.
    
    CRITICAL: New data must be preprocessed with the SAME scaler that was fit
    on the training data. Using a different scaler will produce incorrect predictions.
    
    Args:
        X (pd.DataFrame or np.ndarray): Raw features (unscaled)
        scaler: StandardScaler object from training
        
    Returns:
        np.ndarray: Scaled features ready for prediction
    """
    # Convert DataFrame to numpy array to avoid feature name validation issues
    # The scaler was fit on numpy arrays, so we must transform numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Ensure input is 2D (reshape if single sample is provided as 1D)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    X_scaled = scaler.transform(X)
    
    return X_scaled


def predict_single(model, scaler, sample):
    """
    Predict attack class for a single network flow sample.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: StandardScaler object
        sample (pd.Series or np.ndarray): Single sample with features
        
    Returns:
        str: Predicted attack class
    """
    # Reshape and scale the sample
    if isinstance(sample, pd.Series):
        sample = sample.values.reshape(1, -1)
    elif sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    X_scaled = preprocess_input(sample, scaler)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction


def predict_batch(model, scaler, X_data):
    """
    Make predictions for multiple samples (batch prediction).
    
    Batch prediction is more efficient than single predictions one at a time.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: StandardScaler object
        X_data (pd.DataFrame or np.ndarray): Features (multiple samples)
        
    Returns:
        np.ndarray: Predicted class for each sample
    """
    # Scale the input data
    X_scaled = preprocess_input(X_data, scaler)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions


def predict_with_confidence(model, scaler, X_data):
    """
    Make predictions with confidence scores (probability of each class).
    
    Confidence scores indicate how certain the model is about each prediction.
    High confidence (close to 1.0) indicates high certainty.
    Low confidence (close to 0.5) indicates uncertainty - may need manual review.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: StandardScaler object
        X_data (pd.DataFrame or np.ndarray): Features
        
    Returns:
        dict: Dictionary with predictions and confidence scores
    """
    # Handle single sample
    is_single = False
    if isinstance(X_data, pd.Series):
        X_data = X_data.values.reshape(1, -1)
        is_single = True
    elif isinstance(X_data, np.ndarray) and X_data.ndim == 1:
        X_data = X_data.reshape(1, -1)
        is_single = True
    
    # Scale input
    X_scaled = preprocess_input(X_data, scaler)
    
    # Get predictions and probabilities
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Get maximum confidence for each prediction
    max_confidence = np.max(probabilities, axis=1)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Predicted_Class': predictions,
        'Confidence': max_confidence
    })
    
    # Add probability for each class
    for i, class_name in enumerate(model.classes_):
        results[f'Prob_{class_name}'] = probabilities[:, i]
    
    # Return single result or dataframe
    if is_single:
        return results.iloc[0].to_dict()
    else:
        return results


def predict_traffic_flow(model, scaler, flow_features):
    """
    Predict the attack type for a network traffic flow.
    
    This is a convenient wrapper function for real-world usage where you have
    network flow features ready to classify.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: StandardScaler object
        flow_features (dict or pd.Series): Network flow features
                                           E.g., {
                                               'Flow Duration': 112740690,
                                               'Total Fwd Packet': 32,
                                               'Total Bwd packets': 16,
                                               ...
                                           }
        
    Returns:
        dict: Prediction result with class and confidence
    """
    # Convert to DataFrame if needed
    if isinstance(flow_features, dict):
        flow_df = pd.DataFrame([flow_features])
    else:
        flow_df = flow_features
    
    # Get prediction with confidence
    result = predict_with_confidence(model, scaler, flow_df)
    
    return result


def batch_prediction_from_csv(model, scaler, csv_file_path, output_path=None):
    """
    Load data from CSV and make predictions for all samples.
    
    Useful for batch processing of captured network traffic data.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: StandardScaler object
        csv_file_path (str): Path to CSV file with features
        output_path (str, optional): Path to save predictions (if None, returns DataFrame)
        
    Returns:
        pd.DataFrame: Predictions with confidence scores
    """
    # Load data
    X = pd.read_csv(csv_file_path)
    
    # Get predictions with confidence
    results = predict_with_confidence(model, scaler, X)
    
    # Save if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")
    
    return results


def main():
    """
    Main prediction pipeline with example usage.
    
    Demonstrates:
    1. Loading trained model and scaler
    2. Making single and batch predictions
    3. Working with confidence scores
    4. Processing CSV data
    """
    print("\n" + "="*70)
    print("PREDICTION PIPELINE - NETWORK INTRUSION DETECTION")
    print("="*70)
    
    try:
        # Step 1: Load model and scaler
        print("\nStep 1: Loading trained model...")
        model_path = '../models/random_forest_model.joblib'
        scaler_path = '../models/feature_scaler.joblib'
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        
        # Step 2: Load test data
        print("\nStep 2: Loading test data...")
        X_test = pd.read_csv('../data/processed/monday_X.csv')
        print(f"✓ Loaded {X_test.shape[0]} samples")
        
        # Step 3: Make batch predictions
        print("\nStep 3: Making predictions on first 100 samples...")
        X_sample = X_test.head(100)
        
        # Get predictions with confidence
        predictions = predict_with_confidence(model, scaler, X_sample)
        
        print("\nSample predictions (first 10):")
        print(predictions.head(10).to_string())
        
        # Step 4: Analyze results
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70)
        print(f"Total predictions: {len(predictions)}")
        print(f"\nClass distribution:")
        print(predictions['Predicted_Class'].value_counts())
        print(f"\nConfidence statistics:")
        print(f"  Mean:   {predictions['Confidence'].mean():.4f}")
        print(f"  Min:    {predictions['Confidence'].min():.4f}")
        print(f"  Max:    {predictions['Confidence'].max():.4f}")
        print(f"  Median: {predictions['Confidence'].median():.4f}")
        
        # Step 5: Save predictions
        output_path = '../reports/predictions_sample.csv'
        predictions.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")
        
        # Step 6: Show example uncertain predictions
        print("\n" + "="*70)
        print("UNCERTAIN PREDICTIONS (confidence < 0.95)")
        print("="*70)
        uncertain = predictions[predictions['Confidence'] < 0.95]
        if len(uncertain) > 0:
            print(uncertain.head(10).to_string())
        else:
            print("No uncertain predictions found!")
        
        print("\n✓ Prediction pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please ensure:")
        print("  1. Model and scaler files exist (run train.py)")
        print("  2. Test data files exist (run data_preprocessing.py)")
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
