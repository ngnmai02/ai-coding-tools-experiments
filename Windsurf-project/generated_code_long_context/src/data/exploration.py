"""Data exploration utilities for summarizing the California Housing dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class ExplorationSummary:
    """Structured summary of data exploration outputs."""

    feature_summary: pd.DataFrame
    missing_summary: pd.Series
    dtype_summary: pd.Series


def explore_data(df: pd.DataFrame, output_dir: Path) -> ExplorationSummary:
    """Compute and persist exploration artifacts for the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_summary = df.describe(include="all").transpose()
    missing_summary = df.isna().sum()
    dtype_summary = df.dtypes

    feature_summary.to_csv(output_dir / "feature_summary.csv")
    missing_summary.to_csv(output_dir / "missing_summary.csv")
    dtype_summary.to_csv(output_dir / "dtype_summary.csv")

    return ExplorationSummary(
        feature_summary=feature_summary,
        missing_summary=missing_summary,
        dtype_summary=dtype_summary,
    )


def drop_missing_features(df: pd.DataFrame, missing_summary: pd.Series) -> pd.DataFrame:
    """Drop columns with missing values to keep the dataset clean."""
    missing_cols = missing_summary[missing_summary > 0].index.tolist()
    if not missing_cols:
        return df
    return df.drop(columns=missing_cols)
