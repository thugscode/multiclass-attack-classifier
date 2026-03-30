"""
Model Evaluation Module
=======================
This module provides comprehensive evaluation metrics and visualizations for the trained model.

It includes:
- Loading saved models
- Generating classification metrics
- Confusion matrix analysis
- Feature importance visualization
- Performance reports

Key functions:
- load_model(): Load trained model from disk
- evaluate_model(): Generate detailed evaluation metrics
- create_evaluation_report(): Generate comprehensive evaluation report
- plot_confusion_matrix(): Visualize confusion matrix
- plot_feature_importance(): Visualize feature importance
"""

import joblib
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path, scaler_path):
    """
    Load trained model and scaler from disk.
    
    Args:
        model_path (str): Path to saved Random Forest model (.joblib)
        scaler_path (str): Path to saved StandardScaler (.joblib)
        
    Returns:
        tuple: (trained model, scaler object)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"  Classes: {model.classes_}")
    print(f"  Features: {model.n_features_in_}")
    print(f"✓ Scaler loaded from: {scaler_path}")
    
    return model, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate Random Forest model on test set.
    
    Computes various classification metrics:
    - Accuracy: Overall correctness
    - Precision: False positive rate per class
    - Recall: False negative rate per class (sensitivity)
    - F1-Score: Harmonic mean of precision and recall
    - Confusion Matrix: True/false positives/negatives per class
    
    Args:
        model: Trained RandomForestClassifier
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return metrics


def create_evaluation_report(model, metrics, output_path='../reports'):
    """
    Create a comprehensive evaluation report.
    
    Args:
        model: Trained RandomForestClassifier
        metrics (dict): Evaluation metrics from evaluate_model()
        output_path (str): Directory to save report
        
    Returns:
        None (saves report to file)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create report content
    report_lines = [
        "=" * 70,
        "RANDOM FOREST MODEL EVALUATION REPORT",
        "=" * 70,
        "",
        "MODEL INFORMATION",
        "-" * 70,
        f"Model Type:              Random Forest Classifier",
        f"Number of Trees:         {model.n_estimators}",
        f"Max Depth:               {model.max_depth}",
        f"Number of Features:      {model.n_features_in_}",
        f"Number of Classes:       {len(model.classes_)}",
        f"Classes:                 {', '.join(model.classes_)}",
        "",
        "PERFORMANCE METRICS",
        "-" * 70,
        f"Accuracy:                {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Precision (Weighted):    {metrics['precision']:.4f}",
        f"Recall (Weighted):       {metrics['recall']:.4f}",
        f"F1-Score (Weighted):     {metrics['f1_weighted']:.4f}",
        f"F1-Score (Macro):        {metrics['f1_macro']:.4f}",
        "",
        "CLASSIFICATION REPORT",
        "-" * 70,
    ]
    
    # Add detailed classification report
    report_text = classification_report(
        metrics['y_test'], metrics['y_pred']
    )
    report_lines.append(report_text)
    
    # Add confusion matrix info
    report_lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 70,
        f"Shape: {metrics['confusion_matrix'].shape}",
        f"Diagonal sum (correct predictions): {np.trace(metrics['confusion_matrix'])}",
        f"Total predictions: {metrics['confusion_matrix'].sum()}",
        "",
        "=" * 70,
    ])
    
    # Write report to file
    report_path = os.path.join(output_path, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Evaluation report saved to: {report_path}")
    
    # Also print to console
    print("\n" + "\n".join(report_lines))


def plot_confusion_matrix(metrics, output_path='../reports', title='Confusion Matrix'):
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        metrics (dict): Evaluation metrics from evaluate_model()
        output_path (str): Directory to save plot
        title (str): Plot title
        
    Returns:
        None (saves plot to file)
    """
    os.makedirs(output_path, exist_ok=True)
    
    cm = metrics['confusion_matrix']
    y_test = metrics['y_test']
    y_pred = metrics['y_pred']
    
    # Get class labels from y_test
    classes = y_test.unique()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix plot saved to: {plot_path}")
    plt.close()


def plot_feature_importance(model, feature_names, output_path='../reports', top_n=20):
    """
    Plot and save feature importance bar chart.
    
    Args:
        model: Trained RandomForestClassifier
        feature_names (list): Names of features
        output_path (str): Directory to save plot
        top_n (int): Number of top features to display
        
    Returns:
        None (saves plot to file)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    top_features = importance_df.head(top_n)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_path, f'feature_importance_top{top_n}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to: {plot_path}")
    plt.close()


def main():
    """
    Main evaluation pipeline.
    
    Workflow:
    1. Load model and scaler
    2. Load test data
    3. Evaluate model
    4. Generate report and visualizations
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Load model
        print("\nLoading model...")
        model_path = '../models/random_forest_model.joblib'
        scaler_path = '../models/feature_scaler.joblib'
        model, scaler = load_model(model_path, scaler_path)
        
        # Step 2: Load test data
        print("\nLoading test data...")
        X_test = pd.read_csv('../data/processed/monday_X.csv')
        y_test = pd.read_csv('../data/processed/monday_y.csv').iloc[:, 0]
        
        # Only use a subset for quick evaluation (optional)
        # X_test = X_test.head(10000)
        # y_test = y_test.head(10000)
        
        print(f"✓ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Step 3: Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 4: Generate report
        print("\nGenerating evaluation report...")
        create_evaluation_report(model, metrics)
        
        # Step 5: Create visualizations
        print("\nGenerating visualizations...")
        plot_confusion_matrix(metrics)
        plot_feature_importance(model, X_test.columns.tolist())
        
        print("\n✓ Evaluation pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        print("  Make sure to run train.py first to create the model")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
