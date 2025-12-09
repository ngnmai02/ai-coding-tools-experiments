# California Housing Price Prediction Pipeline

End-to-end machine learning project that predicts California house prices using the scikit-learn **California Housing** dataset. The pipeline performs exploration, preprocessing, model training (baseline, linear regression, ANN), and evaluation with reproducible artifacts.

## Getting Started

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full pipeline:
   ```bash
   python main.py
   ```

Artifacts (plots, serialized models, metrics) are stored in `artifacts/`.

## Project Structure
```
├── main.py
├── requirements.txt
├── src/
│   ├── data/                # Loading, exploration, splitting
│   ├── visualization/       # Raw & preprocessed plots
│   ├── preprocessing/       # Log-scaling + normalization pipeline
│   ├── models/              # Baseline, Linear Regression, ANN (ReLU)
│   ├── evaluation/          # R² / MSE computation + reporting
│   └── utils/               # Configs & shared helpers
└── artifacts/
    ├── figures/             # Raw & preprocessed distributions
    ├── models/              # Persisted models & preprocessors
    └── reports/             # Exploration summaries & metrics
```

## Pipeline Overview

1. **Data Exploration**
   - Load the dataset with original feature names.
   - Save descriptive statistics, dtypes, and missing-value summaries.

2. **Data Preparation**
   - Drop columns containing missing values (if any).
   - Split into 80/20 train/test sets.

3. **Visualization**
   - Plot raw feature distributions and correlation heatmaps.
   - Plot distributions of preprocessed features (after log-scaling + normalization).

4. **Preprocessing**
   - Apply log1p transforms to `AveRooms`, `AveBedrms`, `Population`, `AveOccup`.
   - Standardize all features via a `ColumnTransformer`.

5. **Model Training**
   - **Baseline:** Predict global training mean.
   - **Linear Regression:** Train on normalized data, save coefficients.
   - **ANN (ReLU):** PyTorch MLP trained via `ModelTrainer`.

6. **Evaluation**
   - Compute R² and MSE for train/test splits for each model.
   - Persist results to `artifacts/reports/model_performance.json`.
