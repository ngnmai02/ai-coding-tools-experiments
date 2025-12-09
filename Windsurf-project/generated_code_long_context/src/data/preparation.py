"""Data preparation utilities (train/test split, target separation)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import RANDOM_STATE, TEST_SIZE


@dataclass
class PreparedData:
    """Convenience container holding split data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_data(df: pd.DataFrame, target_column: str) -> PreparedData:
    """Split the dataset into train/test portions with deterministic random state."""
    features = df.drop(columns=[target_column])
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return PreparedData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
