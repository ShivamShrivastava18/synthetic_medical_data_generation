"""
Data Preprocessing Pipeline for Pima Indians Diabetes Dataset

This script creates a reusable preprocessing pipeline that:
1. Handles implicit missing values (zeros in medical measurements)
2. Normalizes numerical features using StandardScaler
3. Performs stratified train-test split (80/20)
4. Prevents data leakage by fitting transformers only on training data

Author: BTech Capstone Project
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
RAW_DATA_PATH = '../data/raw/diabetes.csv'
TRAIN_OUTPUT_PATH = '../data/processed/train.csv'
TEST_OUTPUT_PATH = '../data/processed/test.csv'

# Features that should not have zero values (biologically implausible)
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def load_data(filepath):
    """Load raw dataset from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def handle_missing_values(df):
    """
    Replace zeros with NaN for features where zero is biologically implausible.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        DataFrame with zeros replaced by NaN in specified columns
    """
    print("\nHandling implicit missing values...")
    df_copy = df.copy()
    
    for col in ZERO_AS_MISSING:
        zero_count = (df_copy[col] == 0).sum()
        if zero_count > 0:
            print(f"  {col}: {zero_count} zeros ({zero_count/len(df_copy)*100:.1f}%) → NaN")
            df_copy[col] = df_copy[col].replace(0, np.nan)
    
    return df_copy

def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Split data into train and test sets with stratification on target variable.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data (train: {(1-test_size)*100:.0f}%, test: {test_size*100:.0f}%)...")
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def impute_missing_values(X_train, X_test):
    """
    Impute missing values using median strategy.
    Fit imputer on training data only to prevent data leakage.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train_imputed, X_test_imputed (as DataFrames)
    """
    print("\nImputing missing values with median strategy...")
    
    # Initialize imputer
    imputer = SimpleImputer(strategy='median')
    
    # Fit on training data only
    imputer.fit(X_train)
    
    # Transform both train and test
    X_train_imputed = pd.DataFrame(
        imputer.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"  Imputed values using medians from training data")
    print(f"  Train missing values after imputation: {X_train_imputed.isnull().sum().sum()}")
    print(f"  Test missing values after imputation: {X_test_imputed.isnull().sum().sum()}")
    
    return X_train_imputed, X_test_imputed

def normalize_features(X_train, X_test):
    """
    Normalize features using StandardScaler (zero mean, unit variance).
    Fit scaler on training data only to prevent data leakage.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train_scaled, X_test_scaled (as DataFrames)
    """
    print("\nNormalizing features with StandardScaler...")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    scaler.fit(X_train)
    
    # Transform both train and test
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"  Train set - Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")
    print(f"  Test set - Mean: {X_test_scaled.mean().mean():.6f}, Std: {X_test_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled

def save_processed_data(X_train, X_test, y_train, y_test, train_path, test_path):
    """
    Save processed train and test datasets to CSV files.
    
    Args:
        X_train, X_test: Processed features
        y_train, y_test: Target variables
        train_path: Output path for training data
        test_path: Output path for test data
    """
    print("\nSaving processed datasets...")
    
    # Combine features and target
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    
    test_df = X_test.copy()
    test_df['Outcome'] = y_test
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    # Save to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  Train data saved to: {train_path} ({len(train_df)} samples)")
    print(f"  Test data saved to: {test_path} ({len(test_df)} samples)")

def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("Data Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Load raw data
    df = load_data(RAW_DATA_PATH)
    
    # Step 2: Handle implicit missing values (zeros → NaN)
    df = handle_missing_values(df)
    
    # Step 3: Train-test split (stratified)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 4: Impute missing values (fit on train only)
    X_train, X_test = impute_missing_values(X_train, X_test)
    
    # Step 5: Normalize features (fit on train only)
    X_train, X_test = normalize_features(X_train, X_test)
    
    # Step 6: Save processed datasets
    save_processed_data(X_train, X_test, y_train, y_test, 
                       TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
