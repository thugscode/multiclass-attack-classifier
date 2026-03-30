"""
Training Module for Random Forest Classifier
==============================================
This module trains a Random Forest classifier for multi-class attack classification.

It handles:
- Loading preprocessed data
- Splitting data into train/test sets
- Training Random Forest model with optimized hyperparameters
- Saving the trained model and scaler
- Generating training reports

Key functions:
- load_preprocessed_data(): Load features and labels
- train_random_forest(): Train and evaluate the model
- save_model(): Save trained model and associated objects
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path


def load_preprocessed_data(data_file_path):
    """
    Load preprocessed data (features and labels) from CSV files.
    
    The preprocessing step produces two CSV files:
    - {filename}_X.csv: Normalized features (scaled)
    - {filename}_y.csv: Target labels (attack types)
    
    Args:
        data_file_path (str): Path to the raw CSV file (without extension)
                             E.g., '../data/processed/monday'
        
    Returns:
        tuple: (features DataFrame, labels Series)
    """
    # Add extensions to construct full file paths
    X_path = f"{data_file_path}_X.csv"
    y_path = f"{data_file_path}_y.csv"
    
    # Load features and labels from CSV files
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0]  # Get first column as Series
    
    print(f"✓ Loaded data:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Classes: {y.unique()}")
    
    return X, y


def train_random_forest(X_train, y_train, X_test=None, y_test=None):
    """
    Train a Random Forest classifier with optimized hyperparameters.
    
    Hyperparameters are optimized for network intrusion detection:
    - n_estimators=200: Larger ensemble for better generalization
    - max_depth=25: Sufficient depth to capture patterns without overfitting
    - min_samples_split=5: Prevents very small splits
    - min_samples_leaf=2: Ensures leaves have meaningful samples
    - max_features='sqrt': Reduces correlation between trees
    - class_weight='balanced': Handles class imbalance
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame, optional): Test features for evaluation
        y_test (pd.Series, optional): Test labels for evaluation
        
    Returns:
        tuple: (trained RandomForestClassifier, evaluation metrics dict)
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Initialize Random Forest with optimized hyperparameters
    rf = RandomForestClassifier(
        n_estimators=200,          # Number of trees
        max_depth=25,              # Maximum depth of trees
        min_samples_split=5,       # Minimum samples to split a node
        min_samples_leaf=2,        # Minimum samples at leaf nodes
        max_features='sqrt',       # Number of features to consider at each split
        random_state=42,           # For reproducibility
        n_jobs=-1,                 # Use all available processors
        class_weight='balanced'    # Handle class imbalance
    )
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    
    # Train the model
    rf.fit(X_train, y_train)
    print("✓ Training completed!")
    
    # Evaluate on test set if provided
    metrics = {}
    if X_test is not None and y_test is not None:
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"F1-Score (Macro):    {f1_macro:.4f}")
        print(f"\nTest set size: {X_test.shape[0]}")
        print(f"Class distribution:\n{y_test.value_counts()}")
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix shape:", cm.shape)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*60)
    for idx, (feat, imp) in enumerate(feature_importance.head(15).values, 1):
        print(f"{idx:2d}. {feat:40s} : {imp:.6f}")
    
    return rf, metrics, feature_importance


def save_model(model, scaler, output_path):
    """
    Save trained model and scaler to disk.
    
    The model and scaler objects are saved using joblib for:
    - Easy loading with joblib.load()
    - Efficient serialization (handles sklearn objects well)
    - Good compatibility across Python versions
    
    Args:
        model (RandomForestClassifier): Trained classifier
        scaler (StandardScaler): Fitted scaler object
        output_path (str): Directory path to save model files
                          E.g., '../models'
        
    Returns:
        None (prints save confirmation)
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    model_file = os.path.join(output_path, 'random_forest_model.joblib')
    scaler_file = os.path.join(output_path, 'feature_scaler.joblib')
    
    # Save model and scaler
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    print("\n" + "="*60)
    print("MODEL SAVED")
    print("="*60)
    print(f"✓ Model saved to:   {model_file}")
    print(f"✓ Scaler saved to:  {scaler_file}")
    print(f"  Model size:       {model.n_estimators} trees")
    print(f"  Classes:          {', '.join(model.classes_)}")
    print(f"  Features:         {model.n_features_in_}")


def main():
    """
    Main training pipeline.
    
    Workflow:
    1. Load preprocessed data
    2. Split into train/test sets
    3. Train Random Forest model
    4. Evaluate on test set
    5. Save trained model and scaler
    """
    # Define data path (adjust as needed for different datasets)
    data_path = '../data/processed/monday'
    
    try:
        # Step 1: Load data
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        X, y = load_preprocessed_data(data_path)
        
        # Step 2: Train-test split
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"Training set:  {X_train.shape[0]} samples")
        print(f"Test set:      {X_test.shape[0]} samples")
        print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")
        
        # Step 3: Train model
        model, metrics, feature_importance = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Step 4: Load scaler (from the data preprocessing step)
        scaler_path = '../data/processed/monday_scaler.pkl'
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Step 5: Save model
        save_model(model, scaler, '../models')
        
        print("\n✓ Training pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        print("  Make sure to run data_preprocessing.py first")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
