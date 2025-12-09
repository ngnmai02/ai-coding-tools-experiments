# A simple model for California House Prices Prediction

## Project overview
End-to-end ML pipeline to predict California house prices using the built-in scikit-learn California Housing dataset. Targets >80% R² on the test set and includes modular steps, logging, plots, and saved models.

## Getting started
Guide to use this repository to recreate the project locally.
Recommendation: 
- Should have a virtual environment (venv or uv)

## Project structure
```
generated_code_long_context/
├── main.py                 # Run the full pipeline with `python main.py`
├── requirements.txt        # Dependencies
├── src/
│   ├── data_exploration.py # Load and inspect dataset
│   ├── data_preparation.py # Train/test split (80/20)
│   ├── data_visualization.py # Distributions + correlation heatmaps
│   ├── preprocessing.py    # Log transforms + one-hot (if categorical exists)
│   ├── evaluation.py       # R², MSE reporting
│   └── models/
│       ├── baseline.py
│       ├── linear_regression.py
│       └── ann.py
├── utils/
│   └── helpers.py          # Save/load utilities
└── outputs/
    ├── models/             # Saved models/scalers
    └── plots/              # Preprocessing-before plots
    └── plots_post/         # Post-preprocessing plots
```

## Pipeline (implemented)
1) Data exploration  
2) Data preparation (80/20 split)  
3) Data visualization (raw data)  
4) Preprocessing (log transforms + one-hot)  
5) Data visualization (post-preprocessing)  
6) Model training (baseline, linear regression, ANN/ReLU)  
7) Model evaluation (R², MSE on train/test)  

## Information regarding each component of the pipeline

### Data exploration
- Inspect shapes, dtypes, ranges
- Report and drop columns with missing data
- Ensure numeric consistency

### Data preparation
- Split 80/20 train/test

### Data visualization
- Before preprocessing: distributions and correlation heatmap
- After preprocessing: same plots on transformed features

### Preprocessing
- Log-transform: `AveRooms`, `AveBedrms`, `Population`, `AveOccup`
- One-hot encode categorical columns if present (ocean proximity not in current sklearn data)

### Feature engineering
- Not used (working directly with original sklearn features)

### Model training
- Baseline: predict mean of train targets
- Linear Regression: StandardScaler normalization + model fit, save coefficients
- ANN (ReLU): PyTorch MLP (`ModelTrainer`), normalized inputs, saves state dict

### Evaluation and testing
- Metrics: R² and MSE (plus RMSE) on train and test
- Tabular and detailed printed report

## Running the project
1) Install dependencies: `pip install -r requirements.txt`
2) Run the full pipeline: `python main.py`
3) Artifacts:
   - Plots: `outputs/plots/` (raw), `outputs/plots_post/` (post-preprocessing)
   - Models/scalers: `outputs/models/`

## Notes
- Uses only original sklearn California Housing columns (no synthetic totals).
- Log transforms applied to skewed numeric features; one-hot only if categoricals exist.
- Deterministic splits via `random_state=42`.
