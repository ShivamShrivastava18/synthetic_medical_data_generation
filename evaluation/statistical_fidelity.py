"""
Statistical Fidelity Analysis

This script measures how closely synthetic data matches real data using:
1. Kolmogorov-Smirnov (KS) test for distribution similarity
2. Correlation matrix analysis for feature relationship preservation
3. Visualizations for comparison

Author: BTech Capstone Project
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
REAL_DATA_PATH = '../data/processed/train.csv'
SYNTHETIC_DATASETS = {
    'CTGAN': '../data/synthetic/ctgan.csv',
    'TVAE': '../data/synthetic/tvae.csv',
    'TabDDPM': '../data/synthetic/tabddpm.csv'
}
FIDELITY_METRICS_PATH = '../results/fidelity_metrics.csv'
PLOTS_DIR = '../results/fidelity_plots/'

# Numerical features to analyze (exclude Outcome)
NUMERICAL_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def load_data():
    """
    Load real and synthetic datasets.
    
    Returns:
        Dictionary with real and synthetic DataFrames
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

def perform_ks_test(real_data, synthetic_data, feature):
    """
    Perform Kolmogorov-Smirnov test on a feature.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        feature: Feature name
        
    Returns:
        Dictionary with KS statistic and p-value
    """
    real_values = real_data[feature].values
    synthetic_values = synthetic_data[feature].values
    
    # Perform KS test
    ks_stat, p_value = ks_2samp(real_values, synthetic_values)
    
    # Interpret results
    if ks_stat < 0.1 and p_value > 0.05:
        interpretation = "Excellent"
    elif ks_stat < 0.2 and p_value > 0.01:
        interpretation = "Good"
    elif ks_stat < 0.3:
        interpretation = "Moderate"
    else:
        interpretation = "Poor"
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': p_value,
        'interpretation': interpretation
    }

def compute_correlation_distance(real_data, synthetic_data):
    """
    Compute Frobenius norm distance between correlation matrices.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        
    Returns:
        Frobenius norm distance
    """
    # Compute correlation matrices (only numerical features)
    real_corr = real_data[NUMERICAL_FEATURES].corr()
    synthetic_corr = synthetic_data[NUMERICAL_FEATURES].corr()
    
    # Compute Frobenius norm
    distance = np.linalg.norm(real_corr - synthetic_corr, 'fro')
    
    return distance

def plot_distribution_comparison(real_data, synthetic_data, feature, synthetic_name, ks_stat, p_value):
    """
    Plot distribution comparison for a feature.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        feature: Feature name
        synthetic_name: Name of synthetic dataset
        ks_stat: KS statistic
        p_value: KS p-value
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram comparison
    axes[0].hist(real_data[feature], bins=30, alpha=0.6, label='Real', color='blue', density=True)
    axes[0].hist(synthetic_data[feature], bins=30, alpha=0.6, label=synthetic_name, color='orange', density=True)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{feature} - Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # KDE comparison
    real_data[feature].plot(kind='kde', ax=axes[1], label='Real', color='blue', linewidth=2)
    synthetic_data[feature].plot(kind='kde', ax=axes[1], label=synthetic_name, color='orange', linewidth=2)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'{feature} - KDE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add KS test results
    fig.suptitle(f'{synthetic_name}: KS={ks_stat:.4f}, p={p_value:.4f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'dist_{feature}_{synthetic_name}.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(data, dataset_name):
    """
    Plot correlation heatmap.
    
    Args:
        data: DataFrame
        dataset_name: Name of dataset
    """
    # Compute correlation matrix
    corr = data[NUMERICAL_FEATURES].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f'corr_{dataset_name.lower()}.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

def analyze_synthetic_dataset(real_data, synthetic_data, synthetic_name):
    """
    Perform complete fidelity analysis on a synthetic dataset.
    
    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        synthetic_name: Name of synthetic dataset
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {synthetic_name} Synthetic Data")
    print(f"{'='*60}")
    
    results = []
    
    # KS-test for each feature
    print("\nPerforming KS-tests on numerical features...")
    for feature in NUMERICAL_FEATURES:
        ks_results = perform_ks_test(real_data, synthetic_data, feature)
        
        print(f"  {feature:30s} KS={ks_results['ks_statistic']:.4f}, "
              f"p={ks_results['ks_pvalue']:.4f} ({ks_results['interpretation']})")
        
        # Generate distribution plot
        plot_distribution_comparison(
            real_data, synthetic_data, feature, synthetic_name,
            ks_results['ks_statistic'], ks_results['ks_pvalue']
        )
        
        results.append({
            'synthetic_data': synthetic_name,
            'feature': feature,
            'ks_statistic': ks_results['ks_statistic'],
            'ks_pvalue': ks_results['ks_pvalue'],
            'interpretation': ks_results['interpretation']
        })
    
    # Correlation analysis
    print("\nComputing correlation matrix distance...")
    corr_distance = compute_correlation_distance(real_data, synthetic_data)
    print(f"  Frobenius norm distance: {corr_distance:.4f}")
    
    # Interpret correlation distance
    if corr_distance < 0.5:
        corr_interpretation = "Excellent"
    elif corr_distance < 1.0:
        corr_interpretation = "Good"
    elif corr_distance < 2.0:
        corr_interpretation = "Moderate"
    else:
        corr_interpretation = "Poor"
    
    results.append({
        'synthetic_data': synthetic_name,
        'feature': 'correlation_distance',
        'ks_statistic': corr_distance,
        'ks_pvalue': np.nan,
        'interpretation': corr_interpretation
    })
    
    # Generate correlation heatmap
    print(f"  Generating correlation heatmap...")
    plot_correlation_heatmap(synthetic_data, synthetic_name)
    
    return results

def main():
    """Main statistical fidelity analysis pipeline."""
    print("="*60)
    print("Statistical Fidelity Analysis")
    print("="*60)
    
    # Load data
    data = load_data()
    real_data = data['Real']
    
    # Generate real data correlation heatmap
    print("\nGenerating real data correlation heatmap...")
    plot_correlation_heatmap(real_data, 'Real')
    
    # Analyze each synthetic dataset
    all_results = []
    
    for synthetic_name in ['CTGAN', 'TVAE', 'TabDDPM']:
        synthetic_data = data[synthetic_name]
        results = analyze_synthetic_dataset(real_data, synthetic_data, synthetic_name)
        all_results.extend(results)
    
    # Save results
    print("\n" + "="*60)
    print("Saving fidelity metrics...")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(FIDELITY_METRICS_PATH, index=False)
    print(f"\nFidelity metrics saved to: {FIDELITY_METRICS_PATH}")
    
    # Display summary
    print("\n" + "="*60)
    print("FIDELITY METRICS SUMMARY")
    print("="*60)
    
    # KS-test summary
    ks_results = results_df[results_df['feature'] != 'correlation_distance']
    print("\nAverage KS Statistic by Dataset:")
    print(ks_results.groupby('synthetic_data')['ks_statistic'].mean().to_string())
    
    print("\nInterpretation Distribution:")
    print(ks_results.groupby(['synthetic_data', 'interpretation']).size().unstack(fill_value=0).to_string())
    
    # Correlation distance summary
    print("\n" + "-"*60)
    print("Correlation Matrix Distance:")
    print("-"*60)
    corr_results = results_df[results_df['feature'] == 'correlation_distance']
    for _, row in corr_results.iterrows():
        print(f"  {row['synthetic_data']:10s}: {row['ks_statistic']:.4f} ({row['interpretation']})")
    
    # Best synthetic dataset
    print("\n" + "="*60)
    print("BEST SYNTHETIC DATASET")
    print("="*60)
    
    avg_ks = ks_results.groupby('synthetic_data')['ks_statistic'].mean()
    best_dataset = avg_ks.idxmin()
    best_ks = avg_ks.min()
    
    print(f"\nBest overall: {best_dataset}")
    print(f"  Average KS statistic: {best_ks:.4f}")
    
    # Count plots
    plot_files = os.listdir(PLOTS_DIR)
    print(f"\n{len(plot_files)} plots generated in {PLOTS_DIR}")
    
    print("\n" + "="*60)
    print("Statistical fidelity analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
