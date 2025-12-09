# California House Prices Prediction ML Project

## Overview
This project implements a complete machine learning pipeline for predicting California house prices using the scikit-learn California Housing dataset. The pipeline achieves prediction accuracy through baseline, linear regression, and neural network models.

## Dataset
- **Source**: scikit-learn California Housing dataset
- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude (8 features)
- **Target**: MedHouseVal (median house value)
- **Size**: 20,640 samples

## Getting Started
1. Create a virtual environment: `python -m venv venv` or use uv
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`

## Project Structure
```
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source modules
│   ├── data_exploration.py    # Data loading and initial exploration
│   ├── data_preparation.py    # Train/test splitting
│   ├── data_visualization.py  # Plotting functions (saves to plots/)
│   ├── preprocessing.py       # Log transformations
│   ├── model_training.py      # Model training (baseline, LR, NN)
│   └── evaluation.py          # Model evaluation with R² and MSE
├── models/                # Saved trained models
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   └── nn_model.pth
├── plots/                 # Generated visualization plots
└── utils/                 # Utility functions (empty)
```

## Pipeline
1. **Data Exploration**: Load dataset, check features, handle missing data
2. **Data Visualization**: Generate distribution histograms and correlation heatmap (saved as PNG)
3. **Data Preparation**: 80/20 train/test split
4. **Preprocessing**: Apply log transformation to skewed features (AveRooms, AveBedrms, Population, AveOccup)
5. **Post-Preprocessing Visualization**: Visualize transformed data distributions and correlations
6. **Model Training**: Train baseline, linear regression, and neural network models
7. **Evaluation**: Report R² and MSE metrics for all models on train/test sets

## Models Implemented

### Baseline Model
- Uses mean of training targets as predictions
- No training required

### Linear Regression
- Features normalized with StandardScaler
- Model saved to `models/linear_regression.pkl`
- Scaler saved to `models/scaler.pkl`

### Neural Network
- PyTorch implementation with ReLU activation
- Architecture: Input → 64 → 32 → 1
- Trained for 100 epochs with Adam optimizer
- Model saved to `models/nn_model.pth`

## Metrics Reported
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error (on both train and test sets)

## Output Files
- **Models**: Saved in `models/` directory for future use
- **Plots**: 
  - `plots/feature_distributions.png`: Original feature distributions
  - `plots/correlation_matrix.png`: Original feature correlations
  - `plots/preprocessed_distributions.png`: Post-preprocessing distributions
  - `plots/preprocessed_correlation.png`: Post-preprocessing correlations

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torch
- joblib
