"""Evaluation utilities for computing and persisting model metrics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ModelPerformance:
    """Holds R² and MSE metrics for train/test splits."""

    r2_train: float
    r2_test: float
    mse_train: float
    mse_test: float


def compute_performance(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> ModelPerformance:
    """Compute the performance metrics for a model."""
    return ModelPerformance(
        r2_train=float(r2_score(y_train, y_train_pred)),
        r2_test=float(r2_score(y_test, y_test_pred)),
        mse_train=float(mean_squared_error(y_train, y_train_pred)),
        mse_test=float(mean_squared_error(y_test, y_test_pred)),
    )


def save_report(report_path: Path, performance: Mapping[str, ModelPerformance]) -> None:
    """Persist performance metrics dictionary to disk as JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serializable: Dict[str, Dict[str, float]] = {
        name: asdict(metrics) for name, metrics in performance.items()
    }
    report_path.write_text(json.dumps(serializable, indent=2))
