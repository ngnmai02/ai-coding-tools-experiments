"""Linear regression model utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

from src.utils.config import MODELS_DIR


@dataclass
class LinearRegressionModel:
    """Wrapper around scikit-learn's LinearRegression."""

    model: LinearRegression

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self.model, f)


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegressionModel:
    model = LinearRegression()
    model.fit(X_train, y_train)
    wrapper = LinearRegressionModel(model=model)
    wrapper.save(MODELS_DIR / "linear_regression_model.pkl")
    return wrapper
