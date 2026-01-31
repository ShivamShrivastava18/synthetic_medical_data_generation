"""
Logistic Regression Model for Diabetes Classification

This module implements a Logistic Regression classifier for binary
classification of diabetes using preprocessed medical data.

Author: BTech Capstone Project
Date: 2026-01-31
"""

from sklearn.linear_model import LogisticRegression
import numpy as np

# Configuration
RANDOM_SEED = 42
MAX_ITER = 1000

def create_model():
    """
    Create a Logistic Regression model with default hyperparameters.
    
    Returns:
        LogisticRegression: Configured model instance
    """
    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=MAX_ITER
    )
    return model

def train(X_train, y_train):
    """
    Train Logistic Regression model on training data.
    
    Args:
        X_train: Training features (numpy array or DataFrame)
        y_train: Training labels (numpy array or Series)
        
    Returns:
        Trained LogisticRegression model
    """
    model = create_model()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Make binary predictions on test data.
    
    Args:
        model: Trained LogisticRegression model
        X_test: Test features
        
    Returns:
        numpy array of binary predictions (0 or 1)
    """
    return model.predict(X_test)

def predict_proba(model, X_test):
    """
    Predict class probabilities for test data.
    
    Args:
        model: Trained LogisticRegression model
        X_test: Test features
        
    Returns:
        numpy array of probabilities for positive class
    """
    # Return probability of positive class (class 1)
    return model.predict_proba(X_test)[:, 1]

if __name__ == "__main__":
    # Simple test
    print("Logistic Regression Model")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Max Iterations: {MAX_ITER}")
    
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
