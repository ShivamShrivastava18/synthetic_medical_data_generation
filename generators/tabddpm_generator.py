"""
TabDDPM (Tabular Denoising Diffusion Probabilistic Model) Generator

This module implements a simplified TabDDPM for generating synthetic tabular
medical data using PyTorch.

Note: This is a simplified implementation for demonstration purposes.
A full TabDDPM implementation would require more sophisticated architecture.

Author: BTech Capstone Project
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_DATA_PATH = '../data/processed/train.csv'
OUTPUT_PATH = '../data/synthetic/tabddpm.csv'
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion model for tabular data.
    Uses a simple MLP to denoise data.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        """
        Forward pass with timestep conditioning.
        
        Args:
            x: Noisy data
            t: Timestep
        """
        # Concatenate timestep to input
        t_expanded = t.unsqueeze(1).expand(-1, 1)
        x_t = torch.cat([x, t_expanded], dim=1)
        return self.network(x_t)

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

def prepare_data(df):
    """
    Prepare data for training.
    
    Args:
        df: Training DataFrame
        
    Returns:
        DataLoader and feature names
    """
    print("\nPreparing data...")
    
    # Convert to numpy
    data = df.values.astype(np.float32)
    
    # Create tensor dataset
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"  Data shape: {data.shape}")
    print(f"  Device: {DEVICE}")
    
    return dataloader, list(df.columns)

def add_noise(x, t, num_timesteps=1000):
    """
    Add noise to data based on timestep.
    
    Args:
        x: Clean data
        t: Timestep
        num_timesteps: Total number of timesteps
        
    Returns:
        Noisy data and noise
    """
    # Linear noise schedule
    beta = torch.linspace(0.0001, 0.02, num_timesteps).to(x.device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Get alpha_bar for current timestep
    alpha_bar_t = alpha_bar[t.long()]
    
    # Add noise
    noise = torch.randn_like(x)
    noisy_x = torch.sqrt(alpha_bar_t.unsqueeze(1)) * x + torch.sqrt(1 - alpha_bar_t.unsqueeze(1)) * noise
    
    return noisy_x, noise

def train_tabddpm(dataloader, input_dim, epochs=100):
    """
    Train simplified TabDDPM model.
    
    Args:
        dataloader: Training data loader
        input_dim: Input dimension
        epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    print("\nTraining TabDDPM...")
    print(f"  Epochs: {epochs}")
    print(f"  Input dimension: {input_dim}")
    
    model = SimpleDiffusionModel(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    num_timesteps = 1000
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(DEVICE)
            
            # Random timesteps
            t = torch.randint(0, num_timesteps, (x.shape[0],)).to(DEVICE)
            
            # Add noise
            noisy_x, noise = add_noise(x, t, num_timesteps)
            
            # Predict noise
            predicted_noise = model(noisy_x, t.float() / num_timesteps)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("  Training complete!")
    return model

def generate_synthetic_data(model, num_samples, input_dim, num_timesteps=1000):
    """
    Generate synthetic data using trained model.
    
    Args:
        model: Trained diffusion model
        num_samples: Number of samples to generate
        input_dim: Input dimension
        num_timesteps: Number of diffusion steps
        
    Returns:
        Generated samples as numpy array
    """
    print(f"\nGenerating {num_samples} synthetic samples...")
    
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(num_samples, input_dim).to(DEVICE)
        
        # Reverse diffusion process (simplified)
        for t in reversed(range(0, num_timesteps, 50)):  # Use fewer steps for speed
            t_batch = torch.full((num_samples,), t, dtype=torch.float32).to(DEVICE)
            predicted_noise = model(x, t_batch / num_timesteps)
            
            # Simple denoising step
            x = x - 0.01 * predicted_noise
    
    print(f"  Generated {num_samples} samples")
    return x.cpu().numpy()

def postprocess_data(data, columns):
    """
    Postprocess generated data to match original schema.
    
    Args:
        data: Generated numpy array
        columns: Column names
        
    Returns:
        DataFrame with postprocessed data
    """
    print("\nPostprocessing synthetic data...")
    
    df = pd.DataFrame(data, columns=columns)
    
    # Round Outcome to 0 or 1
    df['Outcome'] = (df['Outcome'] > 0).astype(int)
    
    # Ensure non-negative values for certain columns
    for col in ['Pregnancies', 'Age']:
        df[col] = df[col].clip(lower=0)
    
    print("  Postprocessing complete")
    return df

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
    """Main TabDDPM generation pipeline."""
    print("="*60)
    print("TabDDPM Synthetic Data Generation")
    print("="*60)
    
    # Load data
    train_df = load_training_data()
    
    # Prepare data
    dataloader, columns = prepare_data(train_df)
    input_dim = len(columns)
    
    # Train TabDDPM
    model = train_tabddpm(dataloader, input_dim, epochs=100)
    
    # Generate synthetic data
    num_samples = len(train_df)
    synthetic_data = generate_synthetic_data(model, num_samples, input_dim)
    
    # Postprocess
    synthetic_df = postprocess_data(synthetic_data, columns)
    
    # Validate
    validate_synthetic_data(synthetic_df, train_df)
    
    # Save
    save_synthetic_data(synthetic_df, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("TabDDPM generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
