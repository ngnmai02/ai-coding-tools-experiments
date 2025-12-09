"""Artificial Neural Network model implementation with ReLU activation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error


class ModelTrainer:
    """
    Simple ANN model trainer using ReLU activation function.
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1, 
                 learning_rate=0.001, epochs=100, batch_size=32, random_state=42):
        """
        Initialize the ANN model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (default: [128, 64])
            output_size: Output size (default: 1 for regression)
            learning_rate: Learning rate for optimizer (default: 0.001)
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Build the model
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.is_fitted = False
        self.training_history = []
    
    def _build_model(self):
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        prev_size = self.input_size
        
        # Hidden layers with ReLU activation
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, self.output_size))
        
        return nn.Sequential(*layers)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the ANN model.
        
        Args:
            X_train: Training features (numpy array or pandas DataFrame)
            y_train: Training target values (numpy array or pandas Series)
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            verbose: Whether to print training progress (default: True)
        """
        # Convert to numpy if needed
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        
        # Training loop
        n_samples = X_train_tensor.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_train_tensor = X_train_tensor[indices]
            y_train_tensor = y_train_tensor[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            self.training_history.append(avg_loss)
            
            # Validation if provided
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"   Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif verbose and (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        if verbose:
            print(f"   ANN model trained for {self.epochs} epochs")
    
    def _validate(self, X_val, y_val):
        """Calculate validation loss."""
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values
        
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            loss = self.criterion(outputs, y_val_tensor)
        self.model.train()
        
        return loss.item()
    
    def predict(self, X):
        """
        Make predictions using the trained ANN model.
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            
        Returns:
            np.array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        self.model.train()
        
        return predictions.numpy().flatten()
    
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
        Get model parameters (state dict).
        
        Returns:
            dict: Model parameters (state dict)
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
