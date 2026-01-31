"""
Baseline Model Evaluation on Real Data

This script trains and evaluates three baseline ML models
(Logistic Regression, Random Forest, XGBoost) on the preprocessed
Pima Indians Diabetes dataset and saves performance metrics.

Author: BTech Capstone Project
Date: 2026-01-31
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import logistic_regression
import random_forest
import xgboost_model

# Configuration
TRAIN_DATA_PATH = '../data/processed/train.csv'
TEST_DATA_PATH = '../data/processed/test.csv'
RESULTS_PATH = '../results/baseline_metrics.csv'

def load_data():
    """
    Load preprocessed train and test datasets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading preprocessed data...")
    
    # Load datasets
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Separate features and target
    X_train = train_df.drop('Outcome', axis=1)
    y_train = train_df['Outcome']
    
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluate a trained model and compute metrics.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        X_test: Test features
        y_test: True test labels
        
    Returns:
        Dictionary with model name and metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    if model_name == "Logistic Regression":
        y_pred = logistic_regression.predict(model, X_test)
        y_proba = logistic_regression.predict_proba(model, X_test)
    elif model_name == "Random Forest":
        y_pred = random_forest.predict(model, X_test)
        y_proba = random_forest.predict_proba(model, X_test)
    elif model_name == "XGBoost":
        y_pred = xgboost_model.predict(model, X_test)
        y_proba = xgboost_model.predict_proba(model, X_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'auroc': auroc,
        'f1_score': f1
    }

def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("Baseline Model Evaluation on Real Data")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Store results
    results = []
    
    # 1. Logistic Regression
    print("\n" + "-"*60)
    print("1. Training Logistic Regression...")
    print("-"*60)
    lr_model = logistic_regression.train(X_train, y_train)
    print("  Training complete")
    lr_results = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
    results.append(lr_results)
    
    # 2. Random Forest
    print("\n" + "-"*60)
    print("2. Training Random Forest...")
    print("-"*60)
    rf_model = random_forest.train(X_train, y_train)
    print("  Training complete")
    rf_results = evaluate_model("Random Forest", rf_model, X_test, y_test)
    results.append(rf_results)
    
    # 3. XGBoost
    print("\n" + "-"*60)
    print("3. Training XGBoost...")
    print("-"*60)
    xgb_model = xgboost_model.train(X_train, y_train)
    print("  Training complete")
    xgb_results = evaluate_model("XGBoost", xgb_model, X_test, y_test)
    results.append(xgb_results)
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")
    
    # Display summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Find best model
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_auroc = results_df.loc[results_df['auroc'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    
    print(f"Best Accuracy: {best_accuracy['model']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best AUROC: {best_auroc['model']} ({best_auroc['auroc']:.4f})")
    print(f"Best F1-score: {best_f1['model']} ({best_f1['f1_score']:.4f})")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
