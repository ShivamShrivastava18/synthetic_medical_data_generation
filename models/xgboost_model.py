"""
XGBoost Model for Diabetes Classification

This module implements an XGBoost classifier for binary
classification of diabetes using preprocessed medical data.

Author: BTech Capstone Project
Date: 2026-01-31
"""

from xgboost import XGBClassifier
import numpy as np

# Configuration
RANDOM_SEED = 42

def create_model():
    """
    Create an XGBoost model with default hyperparameters.
    
    Returns:
        XGBClassifier: Configured model instance
    """
    model = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric='logloss'  # Suppress warning
    )
    return model

def train(X_train, y_train):
    """
    Train XGBoost model on training data.
    
    Args:
        X_train: Training features (numpy array or DataFrame)
        y_train: Training labels (numpy array or Series)
        
    Returns:
        Trained XGBClassifier model
    """
    model = create_model()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Make binary predictions on test data.
    
    Args:
        model: Trained XGBClassifier model
        X_test: Test features
        
    Returns:
        numpy array of binary predictions (0 or 1)
    """
    return model.predict(X_test)

def predict_proba(model, X_test):
    """
    Predict class probabilities for test data.
    
    Args:
        model: Trained XGBClassifier model
        X_test: Test features
        
    Returns:
        numpy array of probabilities for positive class
    """
    # Return probability of positive class (class 1)
    return model.predict_proba(X_test)[:, 1]

if __name__ == "__main__":
    # Simple test
    print("XGBoost Model")
    print(f"Random Seed: {RANDOM_SEED}")
    
    # Create dummy data for testing
    X_dummy = np.random.randn(100, 8)
    y_dummy = np.random.randint(0, 2, 100)
    
    # Train model
    model = train(X_dummy, y_dummy)
    print(f"Model trained successfully")
    
    # Test predictions
    predictions = predict(model, X_dummy[:10])
    probabilities = predict_proba(model, X_dummy[:10])
    
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:5]}")
