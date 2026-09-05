"""Probability, ranking and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, probability)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "ks_statistic": float(np.max(tpr - fpr)),
        "expected_calibration_error": expected_calibration_error(y_true, probability),
        "observed_default_rate": float(np.mean(y_true)),
        "mean_predicted_pd": float(np.mean(probability)),
    }


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y_true, "predicted": probability})
    frame["risk_band"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    table = frame.groupby("risk_band", observed=True).agg(
        applications=("observed", "size"),
        mean_predicted_pd=("predicted", "mean"),
        observed_default_rate=("observed", "mean"),
    )
    return table.reset_index(drop=True).assign(decile=lambda value: np.arange(1, len(value) + 1))[
        ["decile", "applications", "mean_predicted_pd", "observed_default_rate"]
    ]
