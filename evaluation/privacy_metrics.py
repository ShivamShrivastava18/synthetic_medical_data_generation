"""
Privacy Metrics Evaluation

This script evaluates privacy risks in synthetic data using:
1. Nearest Neighbor Distance Ratio (NNDR) - detects memorization
2. Duplicate Detection - finds exact/near duplicates
3. Disclosure Risk - estimates re-identification risk

Author: BTech Capstone Project
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
REAL_DATA_PATH = '../data/processed/train.csv'
SYNTHETIC_DATASETS = {
    'CTGAN': '../data/synthetic/ctgan.csv',
    'TVAE': '../data/synthetic/tvae.csv',
    'TabDDPM': '../data/synthetic/tabddpm.csv'
}
PRIVACY_METRICS_PATH = '../results/privacy_metrics.csv'

# Numerical features for distance calculation (exclude Outcome)
NUMERICAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Thresholds
NEAR_DUPLICATE_THRESHOLD = 0.1  # Euclidean distance threshold
HIGH_RISK_NNDR_THRESHOLD = 0.5  # NNDR below this = high disclosure risk

def load_data():
    """
    Load real and synthetic datasets.
    
    Returns:
        Dictionary with DataFrames
    """
    print("Loading datasets...")
    
    data = {}
    
    # Load real data
    data['Real'] = pd.read_csv(REAL_DATA_PATH)
    print(f"  Real data: {data['Real'].shape[0]} samples")
    
    # Load synthetic data
    for name, path in SYNTHETIC_DATASETS.items():
        data[name] = pd.read_csv(path)
        print(f"  {name} synthetic: {data[name].shape[0]} samples")
    
    return data

def normalize_data(real_data, synthetic_data):
    """
    Normalize data for distance calculations.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        
    Returns:
        Normalized real and synthetic arrays
    """
    # Fit scaler on real data
    scaler = StandardScaler()
    real_normalized = scaler.fit_transform(real_data[NUMERICAL_FEATURES])
    
    # Transform synthetic data
    synthetic_normalized = scaler.transform(synthetic_data[NUMERICAL_FEATURES])
    
    return real_normalized, synthetic_normalized

def compute_nndr(real_normalized, synthetic_normalized):
    """
    Compute Nearest Neighbor Distance Ratio.
    
    Args:
        real_normalized: Normalized real data
        synthetic_normalized: Normalized synthetic data
        
    Returns:
        Average NNDR and list of individual NNDRs
    """
    # Compute pairwise distances
    distances = euclidean_distances(synthetic_normalized, real_normalized)
    
    # For each synthetic record, find 2 nearest real records
    nndr_list = []
    
    for i in range(len(synthetic_normalized)):
        # Sort distances for this synthetic record
        sorted_distances = np.sort(distances[i])
        
        # Get nearest and 2nd nearest distances
        d1 = sorted_distances[0]  # Nearest
        d2 = sorted_distances[1]  # 2nd nearest
        
        # Compute NNDR (avoid division by zero)
        if d2 > 0:
            nndr = d1 / d2
        else:
            nndr = 0.0
        
        nndr_list.append(nndr)
    
    avg_nndr = np.mean(nndr_list)
    
    return avg_nndr, nndr_list

def detect_duplicates(real_normalized, synthetic_normalized):
    """
    Detect exact and near duplicates.
    
    Args:
        real_normalized: Normalized real data
        synthetic_normalized: Normalized synthetic data
        
    Returns:
        Count of exact and near duplicates
    """
    # Compute pairwise distances
    distances = euclidean_distances(synthetic_normalized, real_normalized)
    
    # Find minimum distance for each synthetic record
    min_distances = np.min(distances, axis=1)
    
    # Count exact duplicates (distance = 0)
    exact_duplicates = np.sum(min_distances == 0)
    
    # Count near duplicates (distance < threshold)
    near_duplicates = np.sum(min_distances < NEAR_DUPLICATE_THRESHOLD)
    
    return int(exact_duplicates), int(near_duplicates)

def compute_disclosure_risk(nndr_list):
    """
    Compute disclosure risk percentage.
    
    Args:
        nndr_list: List of NNDR values
        
    Returns:
        Percentage of records at high risk
    """
    high_risk_count = np.sum(np.array(nndr_list) < HIGH_RISK_NNDR_THRESHOLD)
    total_count = len(nndr_list)
    
    disclosure_risk_pct = (high_risk_count / total_count) * 100
    
    return disclosure_risk_pct

def assign_privacy_rating(avg_nndr, exact_duplicates, near_duplicates, disclosure_risk_pct):
    """
    Assign overall privacy rating.
    
    Args:
        avg_nndr: Average NNDR
        exact_duplicates: Count of exact duplicates
        near_duplicates: Count of near duplicates
        disclosure_risk_pct: Disclosure risk percentage
        
    Returns:
        Privacy rating string
    """
    # Excellent: NNDR > 0.8, no duplicates, disclosure < 5%
    if avg_nndr > 0.8 and exact_duplicates == 0 and disclosure_risk_pct < 5:
        return "Excellent"
    
    # Good: NNDR > 0.6, few duplicates, disclosure < 10%
    elif avg_nndr > 0.6 and near_duplicates < 3 and disclosure_risk_pct < 10:
        return "Good"
    
    # Moderate: NNDR > 0.5, some duplicates, disclosure < 20%
    elif avg_nndr > 0.5 and near_duplicates < 5 and disclosure_risk_pct < 20:
        return "Moderate"
    
    # Poor: Otherwise
    else:
        return "Poor"

def analyze_privacy(real_data, synthetic_data, synthetic_name):
    """
    Perform complete privacy analysis on a synthetic dataset.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        synthetic_name: Name of synthetic dataset
        
    Returns:
        Dictionary with privacy metrics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {synthetic_name} Privacy")
    print(f"{'='*60}")
    
    # Normalize data
    print("\nNormalizing data...")
    real_normalized, synthetic_normalized = normalize_data(real_data, synthetic_data)
    
    # Compute NNDR
    print("Computing Nearest Neighbor Distance Ratio...")
    avg_nndr, nndr_list = compute_nndr(real_normalized, synthetic_normalized)
    print(f"  Average NNDR: {avg_nndr:.4f}")
    
    # Interpret NNDR
    if avg_nndr > 0.8:
        nndr_interpretation = "Low memorization risk"
    elif avg_nndr > 0.5:
        nndr_interpretation = "Moderate memorization risk"
    else:
        nndr_interpretation = "High memorization risk"
    print(f"  Interpretation: {nndr_interpretation}")
    
    # Detect duplicates
    print("\nDetecting duplicates...")
    exact_duplicates, near_duplicates = detect_duplicates(real_normalized, synthetic_normalized)
    print(f"  Exact duplicates: {exact_duplicates}")
    print(f"  Near duplicates (distance < {NEAR_DUPLICATE_THRESHOLD}): {near_duplicates}")
    
    # Compute disclosure risk
    print("\nComputing disclosure risk...")
    disclosure_risk_pct = compute_disclosure_risk(nndr_list)
    print(f"  Records at high risk (NNDR < {HIGH_RISK_NNDR_THRESHOLD}): {disclosure_risk_pct:.2f}%")
    
    # Interpret disclosure risk
    if disclosure_risk_pct < 5:
        risk_interpretation = "Low risk"
    elif disclosure_risk_pct < 20:
        risk_interpretation = "Moderate risk"
    else:
        risk_interpretation = "High risk"
    print(f"  Interpretation: {risk_interpretation}")
    
    # Assign overall privacy rating
    privacy_rating = assign_privacy_rating(avg_nndr, exact_duplicates, near_duplicates, disclosure_risk_pct)
    print(f"\nOverall Privacy Rating: {privacy_rating}")
    
    return {
        'synthetic_data': synthetic_name,
        'avg_nndr': avg_nndr,
        'exact_duplicates': exact_duplicates,
        'near_duplicates': near_duplicates,
        'disclosure_risk_pct': disclosure_risk_pct,
        'privacy_rating': privacy_rating
    }

def main():
    """Main privacy evaluation pipeline."""
    print("="*60)
    print("Privacy Metrics Evaluation")
    print("="*60)
    
    # Load data
    data = load_data()
    real_data = data['Real']
    
    # Analyze each synthetic dataset
    results = []
    
    for synthetic_name in ['CTGAN', 'TVAE', 'TabDDPM']:
        synthetic_data = data[synthetic_name]
        privacy_metrics = analyze_privacy(real_data, synthetic_data, synthetic_name)
        results.append(privacy_metrics)
    
    # Save results
    print("\n" + "="*60)
    print("Saving privacy metrics...")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(PRIVACY_METRICS_PATH, index=False)
    print(f"\nPrivacy metrics saved to: {PRIVACY_METRICS_PATH}")
    
    # Display summary
    print("\n" + "="*60)
    print("PRIVACY METRICS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Best privacy
    print("\n" + "="*60)
    print("PRIVACY RANKINGS")
    print("="*60)
    
    # Sort by NNDR (higher is better)
    sorted_by_nndr = results_df.sort_values('avg_nndr', ascending=False)
    print("\nBy NNDR (Higher = Better Privacy):")
    for i, (_, row) in enumerate(sorted_by_nndr.iterrows(), 1):
        print(f"  {i}. {row['synthetic_data']}: {row['avg_nndr']:.4f} ({row['privacy_rating']})")
    
    # Sort by disclosure risk (lower is better)
    sorted_by_risk = results_df.sort_values('disclosure_risk_pct', ascending=True)
    print("\nBy Disclosure Risk (Lower = Better Privacy):")
    for i, (_, row) in enumerate(sorted_by_risk.iterrows(), 1):
        print(f"  {i}. {row['synthetic_data']}: {row['disclosure_risk_pct']:.2f}% ({row['privacy_rating']})")
    
    print("\n" + "="*60)
    print("Privacy evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
