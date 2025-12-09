"""Linear regression model implementation with normalization."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle


class LinearRegressionModel:
    """
    Linear regression model wrapper with normalization support.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Fit the linear regression model.
        
        Args:
            X_train: Training features (should already be normalized)
            y_train: Training target values
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("   Linear regression model fitted")
    
    def predict(self, X):
        """
        Make predictions using the linear regression model.
        
        Args:
            X: Input features (should be normalized)
            
        Returns:
            np.array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
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
        Get model parameters (coefficients and intercept).
        
        Returns:
            dict: Model parameters
        """
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }
