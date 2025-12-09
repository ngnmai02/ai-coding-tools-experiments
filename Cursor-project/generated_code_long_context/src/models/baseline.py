"""Baseline model implementation.

This model predicts the mean value of the training target for all test samples.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


class BaselineModel:
    """
    Baseline model that predicts the mean of training target values.
    """
    
    def __init__(self):
        self.mean_value = None
    
    def fit(self, y_train):
        """
        Fit the baseline model by calculating the mean of training targets.
        
        Args:
            y_train: Training target values
        """
        self.mean_value = np.mean(y_train)
        print(f"   Baseline model fitted. Mean value: {self.mean_value:.4f}")
    
    def predict(self, X):
        """
        Predict using the baseline model (returns mean for all samples).
        
        Args:
            X: Input features (not used, but kept for API consistency)
            
        Returns:
            np.array: Array of mean predictions
        """
        if self.mean_value is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.full(n_samples, self.mean_value)
    
    def score(self, X, y_true):
        """
        Calculate R² score for the model.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)
    
    def get_params(self):
        """
        Get model parameters (mean value).
        
        Returns:
            dict: Model parameters
        """
        return {'mean_value': self.mean_value}
