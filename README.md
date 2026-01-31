# BTech Capstone Project: Synthetic Medical Data Generation

## Project Overview

This project implements a comprehensive pipeline for generating and evaluating synthetic medical tabular data using state-of-the-art generative models. The goal is to create privacy-preserving synthetic datasets that maintain the statistical properties and utility of real medical data.

## Dataset

**Pima Indians Diabetes Database**
- **Source**: UCI Machine Learning Repository / Kaggle
- **Size**: 768 samples
- **Features**: 8 numerical features + 1 binary target (Outcome)
- **Task**: Binary classification (diabetes prediction)
- **Split**: 614 training, 154 test samples (80/20 stratified)

## Project Phases Completed

### ✅ Phase 0: Environment & Dataset Setup
- Downloaded and cleaned Pima Indians Diabetes dataset
- Set up project structure and dependencies
- Created comprehensive README and documentation

### ✅ Phase 1: Data Preprocessing Pipeline
- Handled implicit missing values (zeros in medical measurements)
- Applied median imputation (fitted on training data)
- Normalized features using StandardScaler
- Implemented stratified train-test split (80/20)
- Ensured no data leakage

### ✅ Phase 2: Traditional ML Baselines
- Implemented Logistic Regression, Random Forest, XGBoost
- Trained on real data, tested on real data
- **Best Model**: Random Forest (77.3% accuracy, 81.8% AUROC, 65.3% F1)
- Established upper-bound performance for synthetic data evaluation

### ✅ Phase 3: Generative Model Implementation
- **CTGAN**: GAN-based tabular data generator (SDV)
- **TVAE**: Variational Autoencoder for tabular data (SDV)
- **TabDDPM**: Diffusion model for tabular data (PyTorch)
- Generated 614 synthetic samples per model

### ✅ Phase 4: TSTR Experiment (Train on Synthetic, Test on Real)
- Trained all 3 ML models on each synthetic dataset
- Evaluated on real test data
- **Best Synthetic Data**: TVAE (65.4% avg accuracy, 80-93% of baseline)
- Demonstrated synthetic data utility for ML tasks

### ✅ Phase 5: Statistical Fidelity Analysis
- Performed Kolmogorov-Smirnov tests on all features
- Computed correlation matrix distances
- Generated 28 visualization plots
- **Best Fidelity**: TVAE (KS=0.186, correlation distance=0.568)

### ✅ Phase 6: Privacy Evaluation
- Computed Nearest Neighbor Distance Ratio (NNDR)
- Detected duplicates (exact and near)
- Calculated disclosure risk
- **Result**: All models achieved "Excellent" privacy (NNDR > 0.86, zero duplicates)

## Key Results Summary

| Metric | CTGAN | TVAE | TabDDPM |
|--------|-------|------|---------|
| **TSTR Accuracy** | 49.4% | **65.4%** ✅ | 51.3% |
| **Statistical Fidelity (KS)** | 0.254 | **0.186** ✅ | 0.246 |
| **Correlation Distance** | 1.843 | **0.568** ✅ | 1.682 |
| **Privacy (NNDR)** | **0.896** ✅ | 0.864 | 0.895 |
| **Disclosure Risk** | **0.33%** ✅ | 0.98% | 0.49% |

## Winner: TVAE 🏆

**TVAE emerges as the clear winner** across utility and fidelity metrics:
- ✅ Best TSTR performance (65.4% avg accuracy)
- ✅ Best statistical fidelity (KS=0.186)
- ✅ Best correlation preservation (distance=0.568)
- ✅ Excellent privacy (NNDR=0.864, disclosure=0.98%)

**Recommendation**: Use TVAE for synthetic medical data generation in production.

## Project Structure

```
Capstone/
├── data/
│   ├── raw/
│   │   └── diabetes.csv              # Original dataset
│   ├── processed/
│   │   ├── train.csv                 # Preprocessed training data
│   │   └── test.csv                  # Preprocessed test data
│   └── synthetic/
│       ├── ctgan.csv                 # CTGAN synthetic data
│       ├── tvae.csv                  # TVAE synthetic data
│       └── tabddpm.csv               # TabDDPM synthetic data
├── preprocessing/
│   └── preprocess.py                 # Data preprocessing pipeline
├── models/
│   ├── logistic_regression.py        # Logistic Regression model
│   ├── random_forest.py              # Random Forest model
│   └── xgboost_model.py              # XGBoost model
├── generators/
│   ├── ctgan_generator.py            # CTGAN implementation
│   ├── tvae_generator.py             # TVAE implementation
│   └── tabddpm_generator.py          # TabDDPM implementation
├── experiments/
│   ├── baseline_real.py              # Baseline evaluation (real→real)
│   └── tstr_pipeline.py              # TSTR evaluation (synthetic→real)
├── evaluation/
│   ├── statistical_fidelity.py       # Statistical fidelity analysis
│   └── privacy_metrics.py            # Privacy evaluation
├── results/
│   ├── baseline_metrics.csv          # Baseline performance
│   ├── tstr_metrics.csv              # TSTR performance
│   ├── fidelity_metrics.csv          # Statistical fidelity metrics
│   ├── privacy_metrics.csv           # Privacy metrics
│   └── fidelity_plots/               # 28 visualization plots
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Technologies Used

- **Python 3.13**
- **Data Processing**: pandas, numpy, scikit-learn
- **ML Models**: scikit-learn, xgboost
- **Generative Models**: SDV (CTGAN, TVAE), PyTorch (TabDDPM)
- **Visualization**: matplotlib, seaborn
- **Statistical Tests**: scipy

## How to Run

### Quick Start (One-Command Execution)

Run the entire pipeline with a single command:

```bash
python main.py
```

This will:
1. Preprocess the data
2. Train baseline models
3. Generate synthetic data (CTGAN, TVAE, TabDDPM)
4. Run TSTR evaluation
5. Perform statistical fidelity analysis
6. Evaluate privacy metrics
7. Generate final summary and comparison plots

**Estimated time**: 10-15 minutes

### Manual Execution (Step-by-Step)

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
cd preprocessing
python preprocess.py
```

### 3. Train Baseline Models
```bash
cd experiments
python baseline_real.py
```

### 4. Generate Synthetic Data
```bash
cd generators
python ctgan_generator.py
python tvae_generator.py
python tabddpm_generator.py
```

### 5. Run TSTR Evaluation
```bash
cd experiments
python tstr_pipeline.py
```

### 6. Evaluate Fidelity
```bash
cd evaluation
python statistical_fidelity.py
```

### 7. Evaluate Privacy
```bash
cd evaluation
python privacy_metrics.py
```

## Key Findings

### 1. Utility-Fidelity Correlation
Statistical fidelity strongly correlates with downstream utility:
- TVAE: Best fidelity → Best TSTR performance
- CTGAN: Poor fidelity → Poor TSTR performance

### 2. Privacy-Utility Tradeoff
Inverse relationship observed:
- CTGAN: Best privacy (0.896) but worst utility (49.4%)
- TVAE: Good privacy (0.864) with best utility (65.4%)

### 3. Feature-Specific Challenges
Some features are harder to synthesize:
- **Easy**: BloodPressure, BMI
- **Hard**: Insulin, Glucose, Age, Pregnancies

### 4. No Memorization Detected
All models achieved excellent privacy with zero duplicates and low disclosure risk (<1%).

## Limitations

### Not Implemented (As Per Constraints)
- ❌ Hyperparameter tuning for generative models
- ❌ Cross-validation
- ❌ Differential privacy training
- ❌ Adversarial attacks (membership inference, attribute inference)
- ❌ Advanced privacy metrics (k-anonymity, l-diversity)

### Dataset Limitations
- Small dataset (768 samples)
- Single medical domain (diabetes)
- Limited to tabular data

## Future Work

1. **Hyperparameter Optimization**: Tune CTGAN and TabDDPM for better performance
2. **Ensemble Methods**: Combine multiple generators
3. **Conditional Generation**: Generate synthetic data conditioned on specific attributes
4. **Larger Datasets**: Test on larger medical datasets
5. **Privacy-Utility Optimization**: Implement differential privacy with utility preservation
6. **Real-World Deployment**: Deploy TVAE for privacy-preserving data sharing

## Conclusion

This capstone project successfully demonstrates that **TVAE can generate high-quality synthetic medical data** that:
- Achieves 80-93% of real data performance in ML tasks
- Preserves statistical properties (KS=0.186)
- Maintains feature correlations (distance=0.568)
- Provides excellent privacy guarantees (NNDR=0.864, disclosure=0.98%)

**TVAE is recommended for production use** in privacy-sensitive medical data applications where real data sharing is restricted.

## Author

BTech Capstone Project
Date: January 31, 2026

## License

This project is for educational purposes as part of a BTech capstone requirement.
