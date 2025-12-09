"""Preprocessing transformations including log scaling for skewed features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SKEWED_FEATURES = ["AveRooms", "AveBedrms", "Population", "AveOccup"]


def _log1p_array(X: np.ndarray) -> np.ndarray:
    """Apply log1p to an array while guarding against negatives."""
    return np.log1p(np.clip(X, a_min=0, a_max=None))


@dataclass
class PreprocessingArtifacts:
    """Container with fitted preprocessing objects."""

    pipeline: ColumnTransformer
    feature_names: List[str]


def build_preprocessing_pipeline(feature_columns: List[str]) -> ColumnTransformer:
    """Construct a ColumnTransformer that log-scales skewed features and standardizes all features."""
    log_features = [col for col in feature_columns if col in SKEWED_FEATURES]
    remaining_features = [col for col in feature_columns if col not in SKEWED_FEATURES]

    log_pipeline = Pipeline(
        steps=[
            (
                "log",
                FunctionTransformer(
                    _log1p_array,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
        ]
    )

    transformers = []
    if log_features:
        transformers.append(("log_scaled", log_pipeline, log_features))
    if remaining_features:
        transformers.append(("scaled", numeric_pipeline, remaining_features))

    preprocess = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.0,
    )

    return preprocess


def fit_preprocessor(
    preprocess: ColumnTransformer, X_train: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """Fit the preprocessor on training data and return transformed features plus feature names."""
    transformed = preprocess.fit_transform(X_train)
    feature_names = preprocess.get_feature_names_out()
    return transformed, feature_names


def transform_features(preprocess: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform features using a fitted preprocessor."""
    return preprocess.transform(X)
