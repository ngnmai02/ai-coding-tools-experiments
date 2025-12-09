"""Evaluation module for model performance assessment.

This module handles:
- Calculating R² scores for all models
- Calculating MSE for all models on training and testing data
- Reporting all evaluation results
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str):
    """
    Evaluate a model and return metrics.
    
    Args:
        model: Trained model with predict() method
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        model_name: Name of the model for reporting
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return {
        'model_name': model_name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': np.sqrt(train_mse),
        'test_rmse': np.sqrt(test_mse)
    }


def report_results(results_list):
    """
    Report all evaluation results in a formatted table.
    
    Args:
        results_list: List of result dictionaries from evaluate_model()
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<15} {'Test MSE':<15}")
    print("-" * 80)
    
    # Results
    for result in results_list:
        print(f"{result['model_name']:<25} "
              f"{result['train_r2']:>10.4f}  "
              f"{result['test_r2']:>10.4f}  "
              f"{result['train_mse']:>13.4f}  "
              f"{result['test_mse']:>13.4f}")
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    
    for result in results_list:
        print(f"\n{result['model_name']}:")
        print(f"  Training Set:")
        print(f"    R² Score:  {result['train_r2']:.6f}")
        print(f"    MSE:       {result['train_mse']:.6f}")
        print(f"    RMSE:      {result['train_rmse']:.6f}")
        print(f"  Testing Set:")
        print(f"    R² Score:  {result['test_r2']:.6f}")
        print(f"    MSE:       {result['test_mse']:.6f}")
        print(f"    RMSE:      {result['test_rmse']:.6f}")
    
    print("\n" + "="*80 + "\n")
