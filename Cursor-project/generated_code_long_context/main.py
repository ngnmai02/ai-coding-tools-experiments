"""Main entry point for California Housing Price Prediction pipeline.

This script executes the complete ML pipeline:
1. Data exploration
2. Data preparation
3. Data visualization
4. Preprocessing
5. Feature engineering
6. Model training (Baseline, Linear Regression, ANN)
7. Model evaluation
8. Model saving
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_exploration import load_data, explore_data
from src.data_preparation import split_data
from src.data_visualization import visualize_all
from src.preprocessing import preprocess_data, normalize_data
from src.models.baseline import BaselineModel
from src.models.linear_regression import LinearRegressionModel
from src.models.ann import ModelTrainer
from src.evaluation import evaluate_model, report_results
from utils.helpers import save_model, ensure_dir


def main():
    """Execute the complete ML pipeline."""
    print("\n" + "="*80)
    print("CALIFORNIA HOUSING PRICE PREDICTION - ML PIPELINE")
    print("="*80 + "\n")
    
    # Create output directories
    ensure_dir('outputs/models')
    ensure_dir('outputs/plots')
    
    # ==================== STEP 1: DATA EXPLORATION ====================
    df = load_data()
    df_cleaned = explore_data(df)
    
    # ==================== STEP 2: DATA PREPARATION ====================
    X_train, X_test, y_train, y_test = split_data(df_cleaned, target_column='MedHouseVal')
    
    # ==================== STEP 3: DATA VISUALIZATION ====================
    # Visualize before preprocessing to see original distributions
    visualize_all(df_cleaned, save_dir='outputs/plots')
    
    # ==================== STEP 4: PREPROCESSING ====================
    # Apply preprocessing to training and test sets separately (log transform, one-hot encoding)
    X_train_processed = preprocess_data(X_train)
    X_test_processed = preprocess_data(X_test)
    
    # ==================== STEP 5: DATA VISUALIZATION (POST-PREPROCESSING) ====================
    # Visualize distributions/correlations after preprocessing
    visualize_all(X_train_processed, save_dir='outputs/plots_post')
    
    # Ensure both have the same columns (in case preprocessing creates different columns)
    # This can happen if one set has different values in categorical columns
    common_columns = list(set(X_train_processed.columns) & set(X_test_processed.columns))
    X_train_final = X_train_processed[common_columns]
    X_test_final = X_test_processed[common_columns]
    
    # ==================== STEP 6: MODEL TRAINING ====================
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    results = []
    
    # --- Baseline Model ---
    print("\n1. Training Baseline Model...")
    baseline_model = BaselineModel()
    baseline_model.fit(y_train)
    baseline_result = evaluate_model(baseline_model, X_train_final, y_train, 
                                     X_test_final, y_test, "Baseline Model")
    results.append(baseline_result)
    save_model(baseline_model, 'outputs/models/baseline_model.pkl')
    
    # --- Linear Regression Model ---
    print("\n2. Training Linear Regression Model...")
    # Normalize data for linear regression
    X_train_norm, X_test_norm, scaler_lr = normalize_data(X_train_final, X_test_final)
    
    linear_model = LinearRegressionModel()
    linear_model.fit(X_train_norm, y_train)
    linear_result = evaluate_model(linear_model, X_train_norm, y_train, 
                                   X_test_norm, y_test, "Linear Regression")
    results.append(linear_result)
    save_model(linear_model, 'outputs/models/linear_regression_model.pkl')
    save_model(scaler_lr, 'outputs/models/linear_regression_scaler.pkl')
    
    # --- ANN Model ---
    print("\n3. Training ANN Model (ReLU activation)...")
    # Use normalized data for ANN
    X_train_norm_ann, X_test_norm_ann, scaler_ann = normalize_data(X_train_final, X_test_final)
    
    input_size = X_train_norm_ann.shape[1]
    ann_model = ModelTrainer(input_size=input_size, 
                             hidden_sizes=[128, 64], 
                             epochs=100, 
                             batch_size=32,
                             learning_rate=0.001)
    ann_model.fit(X_train_norm_ann, y_train, verbose=True)
    ann_result = evaluate_model(ann_model, X_train_norm_ann, y_train, 
                                X_test_norm_ann, y_test, "ANN (ReLU)")
    results.append(ann_result)
    save_model(ann_model, 'outputs/models/ann_model.pkl')
    save_model(scaler_ann, 'outputs/models/ann_scaler.pkl')
    
    # ==================== STEP 7: EVALUATION ====================
    report_results(results)
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print("\n✓ Data exploration completed")
    print("✓ Data split into training (80%) and testing (20%) sets")
    print("✓ Data visualizations saved to outputs/plots/")
    print("✓ Preprocessing applied (log transformation, one-hot encoding)")
    print("✓ All models trained:")
    print("  - Baseline Model")
    print("  - Linear Regression")
    print("  - ANN with ReLU activation")
    print("✓ All models evaluated (R² and MSE metrics)")
    print("✓ Model parameters saved to outputs/models/")
    print("\n" + "="*80)
    
    # Check if any model achieved >80% accuracy (R² > 0.8)
    print("\nTarget Achievement Check (R² > 0.8):")
    for result in results:
        status = "✓ ACHIEVED" if result['test_r2'] > 0.8 else "✗ Not achieved"
        print(f"  {result['model_name']}: {status} (Test R² = {result['test_r2']:.4f})")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
