"""Artificial Neural Network model defined with PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import MODELS_DIR, RANDOM_STATE


def _set_seed() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


class RegressionNet(nn.Module):
    """Simple feedforward network with ReLU activation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class ModelTrainer:
    """Encapsulates training loop for the ANN model."""

    model: RegressionNet
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    epochs: int = 50
    batch_size: int = 256

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        _set_seed()
        dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X).float()
            outputs = self.model(inputs)
        return outputs.numpy().flatten()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


def train_ann(X_train: np.ndarray, y_train: np.ndarray) -> ModelTrainer:
    input_dim = X_train.shape[1]
    model = RegressionNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainer = ModelTrainer(model=model, optimizer=optimizer, criterion=criterion)
    trainer.train(X_train, y_train)
    trainer.save(MODELS_DIR / "ann_model.pt")
    return trainer
