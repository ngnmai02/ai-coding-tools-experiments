"""Visualization utilities for understanding feature distributions and correlations."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


QUESTIONABLE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> Iterable[Path]:
    """Plot histograms for skewness inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for column in QUESTIONABLE_COLUMNS:
        if column not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True, bins=40)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        file_path = output_dir / f"distribution_{column}.png"
        plt.savefig(file_path)
        plt.close()
        saved_files.append(file_path)

    return saved_files


def plot_correlations(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the feature correlation heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    file_path = output_dir / "correlation_heatmap.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


def plot_preprocessed_distributions(
    data: np.ndarray, feature_names: Sequence[str], output_dir: Path
) -> Iterable[Path]:
    """Visualize distributions for transformed features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    df = pd.DataFrame(data, columns=feature_names)
    for column in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True, bins=40)
        plt.title(f"Preprocessed Distribution of {column}")
        plt.tight_layout()
        file_path = output_dir / f"processed_distribution_{column}.png"
        plt.savefig(file_path)
        plt.close()
        saved_files.append(file_path)

    return saved_files
