"""
Main Execution Script for Capstone Project

This script runs all experiments sequentially and generates final results.
Use this for complete reproducibility of the entire project.

Author: BTech Capstone Project
Date: 2026-01-31
"""

import subprocess
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_script(script_path, description):
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script
        description: Description of what the script does
    """
    print_header(description)
    print(f"Running: {script_path}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_path}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)

def consolidate_results():
    """Consolidate all result CSVs into final summary."""
    print_header("Consolidating Results")
    
    # Load all result files
    baseline = pd.read_csv('results/baseline_metrics.csv')
    tstr = pd.read_csv('results/tstr_metrics.csv')
    fidelity = pd.read_csv('results/fidelity_metrics.csv')
    privacy = pd.read_csv('results/privacy_metrics.csv')
    
    # Create summary DataFrame
    summary_data = []
    
    # For each synthetic dataset
    for dataset in ['CTGAN', 'TVAE', 'TabDDPM']:
        # Get TSTR metrics (average across models)
        tstr_subset = tstr[tstr['synthetic_data'] == dataset]
        avg_accuracy = tstr_subset['accuracy'].mean()
        avg_auroc = tstr_subset['auroc'].mean()
        avg_f1 = tstr_subset['f1_score'].mean()
        
        # Get fidelity metrics
        fidelity_subset = fidelity[fidelity['synthetic_data'] == dataset]
        avg_ks = fidelity_subset[fidelity_subset['feature'] != 'correlation_distance']['ks_statistic'].mean()
        corr_distance = fidelity_subset[fidelity_subset['feature'] == 'correlation_distance']['ks_statistic'].values[0]
        
        # Get privacy metrics
        privacy_subset = privacy[privacy['synthetic_data'] == dataset]
        nndr = privacy_subset['avg_nndr'].values[0]
        disclosure_risk = privacy_subset['disclosure_risk_pct'].values[0]
        privacy_rating = privacy_subset['privacy_rating'].values[0]
        
        summary_data.append({
            'Dataset': dataset,
            'TSTR_Accuracy': avg_accuracy,
            'TSTR_AUROC': avg_auroc,
            'TSTR_F1': avg_f1,
            'Avg_KS_Statistic': avg_ks,
            'Correlation_Distance': corr_distance,
            'NNDR': nndr,
            'Disclosure_Risk_%': disclosure_risk,
            'Privacy_Rating': privacy_rating
        })
    
    # Add baseline (real data)
    baseline_avg_acc = baseline['accuracy'].mean()
    baseline_avg_auroc = baseline['auroc'].mean()
    baseline_avg_f1 = baseline['f1_score'].mean()
    
    summary_data.append({
        'Dataset': 'Real (Baseline)',
        'TSTR_Accuracy': baseline_avg_acc,
        'TSTR_AUROC': baseline_avg_auroc,
        'TSTR_F1': baseline_avg_f1,
        'Avg_KS_Statistic': 0.0,  # Perfect match with itself
        'Correlation_Distance': 0.0,  # Perfect match with itself
        'NNDR': None,
        'Disclosure_Risk_%': None,
        'Privacy_Rating': 'N/A'
    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv('results/final_summary.csv', index=False)
    print("✓ Results consolidated to: results/final_summary.csv")
    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def generate_comparison_plots(summary_df):
    """Generate final comparison plots."""
    print_header("Generating Comparison Plots")
    
    # Filter out baseline for some plots
    synthetic_only = summary_df[summary_df['Dataset'] != 'Real (Baseline)'].copy()
    
    # 1. TSTR Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['TSTR_Accuracy', 'TSTR_AUROC', 'TSTR_F1']
    titles = ['TSTR Accuracy', 'TSTR AUROC', 'TSTR F1-Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Plot synthetic datasets
        axes[i].bar(synthetic_only['Dataset'], synthetic_only[metric], 
                   color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        
        # Add baseline line
        baseline_val = summary_df[summary_df['Dataset'] == 'Real (Baseline)'][metric].values[0]
        axes[i].axhline(y=baseline_val, color='blue', linestyle='--', 
                       linewidth=2, label='Real (Baseline)')
        
        axes[i].set_ylabel(title)
        axes[i].set_ylim(0, 1)
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for j, (dataset, value) in enumerate(zip(synthetic_only['Dataset'], synthetic_only[metric])):
            axes[i].text(j, value + 0.02, f'{value:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('TSTR Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/tstr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: results/plots/tstr_comparison.png")
    
    # 2. Fidelity Metrics Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # KS Statistic
    axes[0].bar(synthetic_only['Dataset'], synthetic_only['Avg_KS_Statistic'],
               color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[0].set_ylabel('Average KS Statistic')
    axes[0].set_title('Distribution Similarity (Lower = Better)')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (dataset, value) in enumerate(zip(synthetic_only['Dataset'], synthetic_only['Avg_KS_Statistic'])):
        axes[0].text(i, value + 0.01, f'{value:.3f}', ha='center', fontsize=9)
    
    # Correlation Distance
    axes[1].bar(synthetic_only['Dataset'], synthetic_only['Correlation_Distance'],
               color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[1].set_ylabel('Correlation Distance')
    axes[1].set_title('Correlation Preservation (Lower = Better)')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (dataset, value) in enumerate(zip(synthetic_only['Dataset'], synthetic_only['Correlation_Distance'])):
        axes[1].text(i, value + 0.05, f'{value:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Statistical Fidelity Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/fidelity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: results/plots/fidelity_comparison.png")
    
    # 3. Privacy Metrics Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # NNDR
    axes[0].bar(synthetic_only['Dataset'], synthetic_only['NNDR'],
               color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[0].set_ylabel('NNDR')
    axes[0].set_title('Nearest Neighbor Distance Ratio (Higher = Better)')
    axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, 
                   label='Threshold (0.8)', alpha=0.7)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (dataset, value) in enumerate(zip(synthetic_only['Dataset'], synthetic_only['NNDR'])):
        axes[0].text(i, value + 0.01, f'{value:.3f}', ha='center', fontsize=9)
    
    # Disclosure Risk
    axes[1].bar(synthetic_only['Dataset'], synthetic_only['Disclosure_Risk_%'],
               color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[1].set_ylabel('Disclosure Risk (%)')
    axes[1].set_title('Disclosure Risk (Lower = Better)')
    axes[1].axhline(y=5, color='red', linestyle='--', linewidth=1.5, 
                   label='Threshold (5%)', alpha=0.7)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (dataset, value) in enumerate(zip(synthetic_only['Dataset'], synthetic_only['Disclosure_Risk_%'])):
        axes[1].text(i, value + 0.05, f'{value:.2f}%', ha='center', fontsize=9)
    
    plt.suptitle('Privacy Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/privacy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: results/plots/privacy_comparison.png")
    
    # 4. Overall Ranking Radar Chart
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics to 0-1 scale (higher is better)
    categories = ['TSTR\nAccuracy', 'TSTR\nAUROC', 'TSTR\nF1', 
                  'Fidelity\n(1-KS)', 'Correlation\nPreservation', 'Privacy\n(NNDR)']
    
    for _, row in synthetic_only.iterrows():
        values = [
            row['TSTR_Accuracy'],
            row['TSTR_AUROC'],
            row['TSTR_F1'],
            1 - row['Avg_KS_Statistic'],  # Invert KS (lower is better)
            1 - (row['Correlation_Distance'] / 2),  # Normalize and invert
            row['NNDR']
        ]
        
        # Complete the circle
        values += values[:1]
        
        # Angles for each axis
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Dataset'])
        ax.fill(angles, values, alpha=0.15)
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Overall Performance Comparison', size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/plots/overall_comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: results/plots/overall_comparison_radar.png")

def main():
    """Main execution pipeline."""
    print_header("CAPSTONE PROJECT - FULL EXECUTION PIPELINE")
    print("This script will run all experiments sequentially.")
    print("Estimated time: 10-15 minutes")
    print("\nPress Ctrl+C to cancel...")
    
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nExecution cancelled by user.")
        sys.exit(0)
    
    start_time = time.time()
    
    # Phase 1: Preprocessing
    run_script('preprocessing/preprocess.py', 
               'PHASE 1: Data Preprocessing')
    
    # Phase 2: Baseline Models
    run_script('experiments/baseline_real.py',
               'PHASE 2: Baseline ML Models (Real → Real)')
    
    # Phase 3: Generate Synthetic Data
    run_script('generators/ctgan_generator.py',
               'PHASE 3a: Generate CTGAN Synthetic Data')
    
    run_script('generators/tvae_generator.py',
               'PHASE 3b: Generate TVAE Synthetic Data')
    
    run_script('generators/tabddpm_generator.py',
               'PHASE 3c: Generate TabDDPM Synthetic Data')
    
    # Phase 4: TSTR Evaluation
    run_script('experiments/tstr_pipeline.py',
               'PHASE 4: TSTR Evaluation (Synthetic → Real)')
    
    # Phase 5: Statistical Fidelity
    run_script('evaluation/statistical_fidelity.py',
               'PHASE 5: Statistical Fidelity Analysis')
    
    # Phase 6: Privacy Evaluation
    run_script('evaluation/privacy_metrics.py',
               'PHASE 6: Privacy Evaluation')
    
    # Phase 7: Consolidate and Visualize
    summary_df = consolidate_results()
    generate_comparison_plots(summary_df)
    
    # Final summary
    total_time = time.time() - start_time
    
    print_header("EXECUTION COMPLETE!")
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print("\nAll results saved to:")
    print("  - results/baseline_metrics.csv")
    print("  - results/tstr_metrics.csv")
    print("  - results/fidelity_metrics.csv")
    print("  - results/privacy_metrics.csv")
    print("  - results/final_summary.csv")
    print("\nVisualization plots saved to:")
    print("  - results/plots/tstr_comparison.png")
    print("  - results/plots/fidelity_comparison.png")
    print("  - results/plots/privacy_comparison.png")
    print("  - results/plots/overall_comparison_radar.png")
    print("  - results/fidelity_plots/ (28 plots)")
    
    print("\n" + "="*70)
    print("  PROJECT IS READY FOR REPORT AND VIVA!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
