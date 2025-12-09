"""Baseline model that predicts the mean of the training target."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BaselineModel:
    """Simple baseline predictor using the target mean."""

    mean_value: float

    def predict(self, size: int) -> np.ndarray:
        return np.full(shape=size, fill_value=self.mean_value, dtype=float)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean_value": self.mean_value}, indent=2))


def train_baseline(y_train: np.ndarray) -> BaselineModel:
    mean_value = float(np.mean(y_train))
    model = BaselineModel(mean_value=mean_value)
    return model
