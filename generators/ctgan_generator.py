"""
CTGAN (Conditional Tabular GAN) Generator

This module implements CTGAN for generating synthetic tabular medical data
using the Synthetic Data Vault (SDV) library.

Author: BTech Capstone Project
Date: 2026-01-31
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_DATA_PATH = '../data/processed/train.csv'
OUTPUT_PATH = '../data/synthetic/ctgan.csv'
RANDOM_SEED = 42

def load_training_data():
    """
    Load preprocessed training data.
    
    Returns:
        DataFrame with training data
    """
    print("Loading training data...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def create_metadata(df):
    """
    Create metadata for the dataset.
    
    Args:
        df: Training DataFrame
        
    Returns:
        SingleTableMetadata object
    """
    print("\nCreating metadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Ensure Outcome is treated as categorical
    metadata.update_column('Outcome', sdtype='categorical')
    
    print("  Metadata created successfully")
    return metadata

def train_ctgan(df, metadata):
    """
    Train CTGAN model on training data.
    
    Args:
        df: Training DataFrame
        metadata: Dataset metadata
        
    Returns:
        Trained CTGANSynthesizer
    """
    print("\nTraining CTGAN...")
    print("  Initializing synthesizer...")
    
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=300,
        verbose=True
    )
    
    print("  Training model (this may take a few minutes)...")
    synthesizer.fit(df)
    
    print("  Training complete!")
    return synthesizer

def generate_synthetic_data(synthesizer, num_samples):
    """
    Generate synthetic data using trained CTGAN.
    
    Args:
        synthesizer: Trained CTGANSynthesizer
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    print(f"\nGenerating {num_samples} synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=num_samples)
    print(f"  Generated {len(synthetic_data)} samples")
    return synthetic_data

def save_synthetic_data(df, output_path):
    """
    Save synthetic data to CSV file.
    
    Args:
        df: Synthetic DataFrame
        output_path: Path to save CSV
    """
    print(f"\nSaving synthetic data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("  Data saved successfully")

def validate_synthetic_data(synthetic_df, original_df):
    """
    Validate synthetic data.
    
    Args:
        synthetic_df: Generated synthetic data
        original_df: Original training data
    """
    print("\nValidating synthetic data...")
    
    # Check schema
    assert list(synthetic_df.columns) == list(original_df.columns), "Schema mismatch!"
    print("  ✓ Schema matches")
    
    # Check size
    assert len(synthetic_df) == len(original_df), "Size mismatch!"
    print(f"  ✓ Size matches ({len(synthetic_df)} samples)")
    
    # Check for missing values
    missing = synthetic_df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values!"
    print("  ✓ No missing values")
    
    # Display statistics
    print("\n  Synthetic data statistics:")
    print(f"    Outcome distribution: {synthetic_df['Outcome'].value_counts().to_dict()}")
    print(f"    Feature means: {synthetic_df.drop('Outcome', axis=1).mean().mean():.4f}")

def main():
    """Main CTGAN generation pipeline."""
    print("="*60)
    print("CTGAN Synthetic Data Generation")
    print("="*60)
    
    # Load data
    train_df = load_training_data()
    
    # Create metadata
    metadata = create_metadata(train_df)
    
    # Train CTGAN
    synthesizer = train_ctgan(train_df, metadata)
    
    # Generate synthetic data (same size as training data)
    num_samples = len(train_df)
    synthetic_df = generate_synthetic_data(synthesizer, num_samples)
    
    # Validate
    validate_synthetic_data(synthetic_df, train_df)
    
    # Save
    save_synthetic_data(synthetic_df, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("CTGAN generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
