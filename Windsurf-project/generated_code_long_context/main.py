"""End-to-end pipeline entrypoint for California housing price modeling."""
from __future__ import annotations

import pickle
from typing import Dict

from src.data.data_loader import TARGET_COLUMN, load_dataset
from src.data.exploration import drop_missing_features, explore_data
from src.data.preparation import prepare_data
from src.evaluation.reporting import ModelPerformance, compute_performance, save_report
from src.models.ann import train_ann
from src.models.baseline import train_baseline
from src.models.linear_regression import train_linear_regression
from src.preprocessing.pipeline import (
    build_preprocessing_pipeline,
    fit_preprocessor,
    transform_features,
)
from src.utils.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.visualization.plots import (
    plot_correlations,
    plot_distributions,
    plot_preprocessed_distributions,
)


def run_pipeline() -> Dict[str, ModelPerformance]:
    """Execute the full ML workflow and return model performance metrics."""
    dataset_artifacts = load_dataset()
    dataframe = dataset_artifacts.dataframe

    exploration_dir = REPORTS_DIR / "exploration"
    exploration_summary = explore_data(dataframe, exploration_dir)
    dataframe = drop_missing_features(dataframe, exploration_summary.missing_summary)

    prepared = prepare_data(dataframe, TARGET_COLUMN)

    distributions_dir = FIGURES_DIR / "distributions"
    plot_distributions(dataframe, distributions_dir)
    plot_correlations(dataframe, FIGURES_DIR)

    X_train = prepared.X_train
    X_test = prepared.X_test

    preprocess = build_preprocessing_pipeline(feature_columns=X_train.columns.tolist())
    X_train_processed, feature_names = fit_preprocessor(preprocess, X_train)
    X_test_processed = transform_features(preprocess, X_test)

    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocess, f)

    processed_figures_dir = FIGURES_DIR / "preprocessed"
    plot_preprocessed_distributions(X_train_processed, feature_names, processed_figures_dir)

    y_train = prepared.y_train.to_numpy()
    y_test = prepared.y_test.to_numpy()

    performance: Dict[str, ModelPerformance] = {}

    baseline_model = train_baseline(y_train)
    baseline_model.save(MODELS_DIR / "baseline_model.json")
    baseline_train_pred = baseline_model.predict(len(y_train))
    baseline_test_pred = baseline_model.predict(len(y_test))
    performance["baseline"] = compute_performance(
        y_train, baseline_train_pred, y_test, baseline_test_pred
    )

    linear_model = train_linear_regression(X_train_processed, y_train)
    linear_train_pred = linear_model.predict(X_train_processed)
    linear_test_pred = linear_model.predict(X_test_processed)
    performance["linear_regression"] = compute_performance(
        y_train, linear_train_pred, y_test, linear_test_pred
    )

    ann_trainer = train_ann(X_train_processed, y_train)
    ann_train_pred = ann_trainer.predict(X_train_processed)
    ann_test_pred = ann_trainer.predict(X_test_processed)
    performance["ann_relu"] = compute_performance(
        y_train, ann_train_pred, y_test, ann_test_pred
    )

    report_path = REPORTS_DIR / "model_performance.json"
    save_report(report_path, performance)

    return performance


def main() -> None:
    performance = run_pipeline()
    for name, metrics in performance.items():
        print(f"Model: {name}")
        print(
            f"\tR^2 Train: {metrics.r2_train:.4f} | R^2 Test: {metrics.r2_test:.4f}"
        )
        print(
            f"\tMSE Train: {metrics.mse_train:.4f} | MSE Test: {metrics.mse_test:.4f}"
        )


if __name__ == "__main__":
    main()
