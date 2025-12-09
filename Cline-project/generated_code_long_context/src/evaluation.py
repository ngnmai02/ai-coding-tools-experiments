import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import joblib

class ModelTrainer(nn.Module):
    def __init__(self, input_size):
        super(ModelTrainer, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate_models(X_train, X_test, y_train, y_test, r2_scores):
    """
    Load trained models and compute MSE on train and test sets.

    Args:
        X_train, X_test, y_train, y_test, r2_scores (tuple of r2 for baseline, lr, nn)
    """
    # Baseline MSE
    y_mean = y_train.mean()
    y_pred_baseline_train = np.full_like(y_train, y_mean)
    mse_baseline_train = mean_squared_error(y_train, y_pred_baseline_train)
    y_pred_baseline_test = np.full_like(y_test, y_mean)
    mse_baseline_test = mean_squared_error(y_test, y_pred_baseline_test)

    # Linear Regression
    lr = joblib.load('models/linear_regression.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_lr_train = lr.predict(X_train_scaled)
    mse_lr_train = mean_squared_error(y_train, y_pred_lr_train)
    y_pred_lr_test = lr.predict(X_test_scaled)
    mse_lr_test = mean_squared_error(y_test, y_pred_lr_test)

    # NN
    input_size = X_train_scaled.shape[1]
    model = ModelTrainer(input_size)
    model.load_state_dict(torch.load('models/nn_model.pth'))
    model.eval()
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_nn_train = model(X_train_tensor).numpy().flatten()
        mse_nn_train = mean_squared_error(y_train, y_pred_nn_train)
        y_pred_nn_test = model(X_test_tensor).numpy().flatten()
        mse_nn_test = mean_squared_error(y_test, y_pred_nn_test)

    # Report all results
    print("\nEvaluation Results:")
    print(f"Baseline - R²: {r2_scores[0]:.4f}, MSE Train: {mse_baseline_train:.4f}, MSE Test: {mse_baseline_test:.4f}")
    print(f"Linear Regression - R²: {r2_scores[1]:.4f}, MSE Train: {mse_lr_train:.4f}, MSE Test: {mse_lr_test:.4f}")
    print(f"NN - R²: {r2_scores[2]:.4f}, MSE Train: {mse_nn_train:.4f}, MSE Test: {mse_nn_test:.4f}")
