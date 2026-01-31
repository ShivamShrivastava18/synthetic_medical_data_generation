"""
TSTR (Train on Synthetic, Test on Real) Evaluation Pipeline

This script evaluates the utility of synthetic data by training ML models
on synthetic datasets and testing on real data.

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
SYNTHETIC_DATASETS = {
    'CTGAN': '../data/synthetic/ctgan.csv',
    'TVAE': '../data/synthetic/tvae.csv',
    'TabDDPM': '../data/synthetic/tabddpm.csv'
}
TEST_DATA_PATH = '../data/processed/test.csv'
BASELINE_METRICS_PATH = '../results/baseline_metrics.csv'
TSTR_RESULTS_PATH = '../results/tstr_metrics.csv'

def load_test_data():
    """
    Load real test data.
    
    Returns:
        X_test, y_test
    """
    print("Loading real test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    print(f"  Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test

def load_synthetic_data(filepath):
    """
    Load synthetic training data.
    
    Args:
        filepath: Path to synthetic CSV
        
    Returns:
        X_train, y_train
    """
    df = pd.read_csv(filepath)
    
    X_train = df.drop('Outcome', axis=1)
    y_train = df['Outcome']
    
    return X_train, y_train

def train_and_evaluate(model_name, train_func, predict_func, predict_proba_func, 
                       X_train, y_train, X_test, y_test):
    """
    Train a model on synthetic data and evaluate on real test data.
    
    Args:
        model_name: Name of the model
        train_func: Training function
        predict_func: Prediction function
        predict_proba_func: Probability prediction function
        X_train: Synthetic training features
        y_train: Synthetic training labels
        X_test: Real test features
        y_test: Real test labels
        
    Returns:
        Dictionary with metrics
    """
    # Train on synthetic data
    model = train_func(X_train, y_train)
    
    # Predict on real test data
    y_pred = predict_func(model, X_test)
    y_proba = predict_proba_func(model, X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'auroc': auroc,
        'f1_score': f1
    }

def evaluate_synthetic_dataset(synthetic_name, synthetic_path, X_test, y_test):
    """
    Evaluate all models on a synthetic dataset.
    
    Args:
        synthetic_name: Name of synthetic dataset
        synthetic_path: Path to synthetic CSV
        X_test: Real test features
        y_test: Real test labels
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {synthetic_name} Synthetic Data")
    print(f"{'='*60}")
    
    # Load synthetic data
    print(f"\nLoading {synthetic_name} synthetic data...")
    X_train, y_train = load_synthetic_data(synthetic_path)
    print(f"  Synthetic data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Class distribution: {y_train.value_counts().to_dict()}")
    
    results = []
    
    # 1. Logistic Regression
    print(f"\n{'-'*60}")
    print("1. Training Logistic Regression on synthetic data...")
    print(f"{'-'*60}")
    lr_metrics = train_and_evaluate(
        "Logistic Regression",
        logistic_regression.train,
        logistic_regression.predict,
        logistic_regression.predict_proba,
        X_train, y_train, X_test, y_test
    )
    print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"  AUROC: {lr_metrics['auroc']:.4f}")
    print(f"  F1-score: {lr_metrics['f1_score']:.4f}")
    
    results.append({
        'synthetic_data': synthetic_name,
        'model': 'Logistic Regression',
        **lr_metrics
    })
    
    # 2. Random Forest
    print(f"\n{'-'*60}")
    print("2. Training Random Forest on synthetic data...")
    print(f"{'-'*60}")
    rf_metrics = train_and_evaluate(
        "Random Forest",
        random_forest.train,
        random_forest.predict,
        random_forest.predict_proba,
        X_train, y_train, X_test, y_test
    )
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  AUROC: {rf_metrics['auroc']:.4f}")
    print(f"  F1-score: {rf_metrics['f1_score']:.4f}")
    
    results.append({
        'synthetic_data': synthetic_name,
        'model': 'Random Forest',
        **rf_metrics
    })
    
    # 3. XGBoost
    print(f"\n{'-'*60}")
    print("3. Training XGBoost on synthetic data...")
    print(f"{'-'*60}")
    xgb_metrics = train_and_evaluate(
        "XGBoost",
        xgboost_model.train,
        xgboost_model.predict,
        xgboost_model.predict_proba,
        X_train, y_train, X_test, y_test
    )
    print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"  AUROC: {xgb_metrics['auroc']:.4f}")
    print(f"  F1-score: {xgb_metrics['f1_score']:.4f}")
    
    results.append({
        'synthetic_data': synthetic_name,
        'model': 'XGBoost',
        **xgb_metrics
    })
    
    return results

def compare_with_baseline(tstr_df, baseline_df):
    """
    Compare TSTR results with baseline (real data) results.
    
    Args:
        tstr_df: TSTR results DataFrame
        baseline_df: Baseline results DataFrame
    """
    print("\n" + "="*60)
    print("TSTR vs Baseline Comparison")
    print("="*60)
    
    for model in baseline_df['model'].unique():
        print(f"\n{model}:")
        print("-" * 60)
        
        # Get baseline metrics
        baseline_row = baseline_df[baseline_df['model'] == model].iloc[0]
        baseline_acc = baseline_row['accuracy']
        baseline_auroc = baseline_row['auroc']
        baseline_f1 = baseline_row['f1_score']
        
        print(f"  Baseline (Real→Real):")
        print(f"    Accuracy: {baseline_acc:.4f}, AUROC: {baseline_auroc:.4f}, F1: {baseline_f1:.4f}")
        
        # Get TSTR metrics for each synthetic dataset
        tstr_rows = tstr_df[tstr_df['model'] == model]
        
        for _, row in tstr_rows.iterrows():
            synthetic_name = row['synthetic_data']
            tstr_acc = row['accuracy']
            tstr_auroc = row['auroc']
            tstr_f1 = row['f1_score']
            
            # Calculate percentage of baseline
            acc_pct = (tstr_acc / baseline_acc) * 100
            auroc_pct = (tstr_auroc / baseline_auroc) * 100
            f1_pct = (tstr_f1 / baseline_f1) * 100
            
            print(f"\n  {synthetic_name} (Synthetic→Real):")
            print(f"    Accuracy: {tstr_acc:.4f} ({acc_pct:.1f}% of baseline)")
            print(f"    AUROC: {tstr_auroc:.4f} ({auroc_pct:.1f}% of baseline)")
            print(f"    F1: {tstr_f1:.4f} ({f1_pct:.1f}% of baseline)")

def main():
    """Main TSTR evaluation pipeline."""
    print("="*60)
    print("TSTR (Train on Synthetic, Test on Real) Evaluation")
    print("="*60)
    
    # Load real test data
    X_test, y_test = load_test_data()
    
    # Evaluate each synthetic dataset
    all_results = []
    
    for synthetic_name, synthetic_path in SYNTHETIC_DATASETS.items():
        results = evaluate_synthetic_dataset(synthetic_name, synthetic_path, X_test, y_test)
        all_results.extend(results)
    
    # Save TSTR results
    print("\n" + "="*60)
    print("Saving TSTR results...")
    print("="*60)
    
    tstr_df = pd.DataFrame(all_results)
    tstr_df.to_csv(TSTR_RESULTS_PATH, index=False)
    print(f"\nTSTR results saved to: {TSTR_RESULTS_PATH}")
    
    # Display summary
    print("\n" + "="*60)
    print("TSTR RESULTS SUMMARY")
    print("="*60)
    print(tstr_df.to_string(index=False))
    
    # Compare with baseline
    if os.path.exists(BASELINE_METRICS_PATH):
        baseline_df = pd.read_csv(BASELINE_METRICS_PATH)
        compare_with_baseline(tstr_df, baseline_df)
    
    # Find best synthetic dataset per model
    print("\n" + "="*60)
    print("BEST SYNTHETIC DATASET BY MODEL")
    print("="*60)
    
    for model in tstr_df['model'].unique():
        model_results = tstr_df[tstr_df['model'] == model]
        best_row = model_results.loc[model_results['accuracy'].idxmax()]
        print(f"\n{model}:")
        print(f"  Best: {best_row['synthetic_data']}")
        print(f"  Accuracy: {best_row['accuracy']:.4f}, AUROC: {best_row['auroc']:.4f}, F1: {best_row['f1_score']:.4f}")
    
    print("\n" + "="*60)
    print("TSTR evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
