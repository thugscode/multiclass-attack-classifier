"""
Data Preprocessing Module for Network Intrusion Detection System
================================================================
This module handles the complete data preprocessing pipeline for the intrusion detection dataset.
It includes steps for data cleaning, feature engineering, and normalization.

Key functions:
- load_data(): Reads CSV files into pandas DataFrames
- clean_data(): Removes duplicates, NaN values, and infinite values
- remove_irrelevant_columns(): Drops non-informative features
- encode_labels(): Converts categorical labels to numerical values
- scale_features(): Normalizes feature values using StandardScaler
- preprocess_pipeline(): Orchestrates all preprocessing steps
- process_all_files(): Processes all raw data files in batch
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle
from pathlib import Path


def load_data(path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        path (str): File path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data containing all rows and columns from the CSV file
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Clean the dataset by removing duplicates, NaN values, and infinite values.
    
    Data quality is essential for model training. This function ensures the dataset is clean:
    1. Duplicates: Exact same rows that don't add information are removed
    2. Infinite values: May result from mathematical operations (division by zero, log of negative)
    3. Missing values (NaN): Records with missing fields cannot be used for training
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        
    Returns:
        pd.DataFrame: Clean DataFrame with duplicates, NaN, and infinite values removed
    """
    # Step 1: Remove duplicate rows (identical records don't add new information)
    df = df.drop_duplicates()

    # Step 2: Replace infinite values with NaN (both positive and negative infinity)
    # This can happen due to operations like division by zero or log of zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 3: Remove rows containing any NaN/missing values
    # These records cannot be used effectively in model training
    df = df.dropna()

    return df


def remove_irrelevant_columns(df):
    """
    Remove columns that don't contribute to intrusion detection prediction.
    
    These columns are either:
    - Identifiers (Flow ID, Source/Destination IP): Unique per session but not predictive
    - Temporal data (Timestamp): May cause data leakage or temporal bias
    - Local network metadata columns that don't aid classification
    
    Args:
        df (pd.DataFrame): DataFrame with all original columns
        
    Returns:
        pd.DataFrame: DataFrame with irrelevant columns removed
    """
    # Define columns that are not useful for the machine learning model
    # These are metadata/identifiers that don't help predict intrusion
    # Based on NSL-KDD and CIC-IDS2017/2018 dataset structures
    columns_to_drop = [
        'Flow ID', 'Source IP', 'Destination IP', 'Timestamp',
        'Src IP dec', 'Dst IP dec', 'Src Port', 'Dst Port',
        'Local', 'Local_1', 'Local_2', 'Local_3', 'Local_4', 'Local_5',
        'Local_6', 'Local_7', 'Local_8', 'Local_9', 'Local_10', 'Local_11',
        'Local_12', 'Local_13', 'Local_14', 'Attempted Category'
    ]
    
    # Safely drop columns only if they exist in the DataFrame
    # Different CSV files may have different column names
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


def encode_labels(df):
    """
    Handle target labels (keep as categorical strings or encode if needed).
    
    Machine learning algorithms can work with either categorical strings or numeric labels.
    For tree-based models like Random Forest, categorical labels work fine.
    For neural networks, numeric encoding is required.
    
    This function keeps labels as strings but provides an encoder for conversion if needed.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Label' column with text values
        
    Returns:
        tuple: (DataFrame, LabelEncoder object for future use if needed)
    """
    # Create a LabelEncoder to map text labels to integers
    # Though we don't apply it by default, it's useful for algorithms that need numeric labels
    le = LabelEncoder()
    
    # Fit the encoder to the labels (but don't transform them yet)
    # This stores the mapping from text labels to numbers
    le.fit(df['Label'])
    
    # Return the DataFrame unchanged and the encoder for later use
    # Tree-based models like Random Forest handle string labels natively
    return df, le


def scale_features(df):
    """
    Normalize feature values using StandardScaler.
    
    Different features have different scales (e.g., byte counts vs time values).
    Scaling ensures all features contribute equally to model training and improves
    convergence speed for algorithms like neural networks and SVM.
    
    The StandardScaler transforms data to have mean=0 and standard deviation=1.
    Formula: X_scaled = (X - mean) / std_dev
    
    Args:
        df (pd.DataFrame): DataFrame after encoding, with features and Label column
        
    Returns:
        tuple: (scaled features array, label series, StandardScaler object)
    """
    # Separate features (X) from target label (y)
    # Drop the 'Label' column to get only the input features
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Initialize StandardScaler for feature normalization
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform all features
    # This centers features around 0 and scales them to unit variance
    X_scaled = scaler.fit_transform(X)

    # Return the scaled features, labels (as pandas Series), and the scaler object
    # The scaler is saved to normalize new data with the same statistics
    # Labels are kept as Series/values in their original categorical form
    return X_scaled, y.values, scaler


def preprocess_pipeline(path):
    """
    Execute the complete preprocessing pipeline for a single dataset.
    
    This function orchestrates all preprocessing steps in the correct order:
    1. Load raw data from CSV
    2. Clean data (remove duplicates, NaN, infinite values)
    3. Remove irrelevant columns (IDs, timestamps)
    4. Encode target labels to numeric format
    5. Normalize/scale feature values
    
    Args:
        path (str): Path to the raw CSV file
        
    Returns:
        tuple: (scaled features, labels, scaler object, label encoder object)
    """
    # Load the raw data from CSV file
    df = load_data(path)
    
    # Clean the data by removing bad records
    df = clean_data(df)
    
    # Remove columns that won't help with predictions
    df = remove_irrelevant_columns(df)
    
    # Convert text labels to numeric format
    df, le = encode_labels(df)
    
    # Normalize all features to have comparable scales
    X, y, scaler = scale_features(df)

    # Return processed features, labels, and the preprocessing objects
    # These objects are essential for processing new data with identical transformations
    return X, y, scaler, le


def process_all_files():
    """
    Process all CSV files from the raw folder and save preprocessed data to the processed folder.
    
    Workflow:
    1. Identifies all CSV files in the raw data directory
    2. Processes each file through the preprocessing pipeline
    3. Saves the normalized features as CSV files
    4. Saves the preprocessing objects (scaler, label encoder) for later use
    5. Provides status updates and error handling for each file
    
    Output files saved for each input file:
    - {filename}_X.csv: Normalized features (input to models)
    - {filename}_y.csv: Target labels (true values for training)
    - {filename}_scaler.pkl: StandardScaler object (applied to features)
    - {filename}_label_encoder.pkl: LabelEncoder object (converts predictions back to text)
    """
    # Define paths for raw and processed data
    raw_folder = "../data/raw"
    processed_folder = "../data/processed"
    
    # Create the processed folder if it doesn't exist
    # parents=True creates parent directories if needed
    # exist_ok=True prevents error if folder already exists
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files from the raw folder
    # Filter to only include .csv files
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]
    
    # Display how many files will be processed
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        # Construct the full path to the raw file
        file_path = os.path.join(raw_folder, csv_file)
        
        # Extract filename without extension for output file naming
        file_name = csv_file.replace('.csv', '')
        
        # Display processing status without newline (end with space for append)
        print(f"Processing: {csv_file}...", end=" ")
        
        try:
            # Run the complete preprocessing pipeline on this file
            X, y, scaler, le = preprocess_pipeline(file_path)
            
            # Convert numpy arrays to pandas DataFrames for CSV export
            # This makes the data more readable and preserves structure
            X_df = pd.DataFrame(X)
            y_df = pd.DataFrame(y, columns=['Label'])
            
            # Save the normalized features to CSV file
            # index=False prevents writing row numbers to the file
            X_df.to_csv(os.path.join(processed_folder, f"{file_name}_X.csv"), index=False)
            
            # Save the target labels to CSV file
            y_df.to_csv(os.path.join(processed_folder, f"{file_name}_y.csv"), index=False)
            
            # Save the StandardScaler object using pickle
            # This allows us to apply identical scaling to new/test data
            with open(os.path.join(processed_folder, f"{file_name}_scaler.pkl"), 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save the LabelEncoder object using pickle
            # This allows us to convert numeric predictions back to class names
            with open(os.path.join(processed_folder, f"{file_name}_label_encoder.pkl"), 'wb') as f:
                pickle.dump(le, f)
            
            # Show success status and the shape of processed features
            print(f"✓ Shape: {X.shape}")
        
        except Exception as e:
            # Catch any errors during processing and display them
            # This allows remaining files to be processed even if one fails
            print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    """
    Main entry point: Execute the complete data processing pipeline.
    
    This script is run directly (not imported as a module). It processes all raw CSV
    files and saves the preprocessed data to the processed folder.
    """
    # Execute the data processing on all files
    process_all_files()
    
    # Confirmation message showing completion
    print("\n✓ All data preprocessing complete!")