import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def train_models(X_train, X_test, y_train, y_test):
    """
    Train baseline, linear regression, and NN models.
    Normalize data, train, compute R², save models.

    Args:
        X_train, X_test, y_train, y_test

    Returns:
        tuple: r2_baseline, r2_lr, r2_nn
    """
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline model
    y_mean = y_train.mean()
    y_pred_baseline = np.full_like(y_test, y_mean)
    r2_baseline = r2_score(y_test, y_pred_baseline)
    print(f"Baseline R²: {r2_baseline:.4f}")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"Linear Regression R²: {r2_lr:.4f}")

    # Save Linear Regression and scaler
    joblib.dump(lr, 'models/linear_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # NN Model
    input_size = X_train_scaled.shape[1]
    model = ModelTrainer(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Train NN
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Predict with NN
    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_tensor).numpy().flatten()
    r2_nn = r2_score(y_test, y_pred_nn)
    print(f"NN R²: {r2_nn:.4f}")

    # Save NN model
    torch.save(model.state_dict(), 'models/nn_model.pth')

    return r2_baseline, r2_lr, r2_nn
