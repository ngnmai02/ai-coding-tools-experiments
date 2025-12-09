"""Utilities for downloading and preparing the California Housing dataset."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_california_housing

TARGET_COLUMN = "median_house_value"


@dataclass
class DatasetArtifacts:
    """Container for the dataset and helper metadata."""

    dataframe: pd.DataFrame
    description: str


def load_dataset() -> DatasetArtifacts:
    """Load the California Housing dataset without additional engineered features."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns={"MedHouseVal": TARGET_COLUMN})

    description = dataset.DESCR
    return DatasetArtifacts(dataframe=df, description=description)
