"""Data drift, calibration drift and decision fairness diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import TARGET
from credit_risk.policy import approval_mask


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_count, _ = np.histogram(reference, bins=edges)
    current_count, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_count / reference_count.sum(), 1e-6, None)
    current_share = np.clip(current_count / current_count.sum(), 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_pd: np.ndarray,
    current_pd: np.ndarray,
) -> pd.DataFrame:
    rows = [
        {
            "feature": "predicted_pd",
            "psi": population_stability_index(reference_pd, current_pd),
        }
    ]
    for feature in ["bureau_score", "debt_to_income", "utilisation", "requested_amount"]:
        rows.append(
            {
                "feature": feature,
                "psi": population_stability_index(reference[feature].to_numpy(), current[feature].to_numpy()),
            }
        )
    report = pd.DataFrame(rows)
    report["status"] = np.select(
        [report["psi"] >= 0.25, report["psi"] >= 0.10],
        ["investigate", "watch"],
        default="stable",
    )
    return report


def group_audit(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    group_column: str = "age_group",
    target_approval_rate: float = 0.60,
) -> pd.DataFrame:
    approved = approval_mask(probability, target_approval_rate)
    audit = pd.DataFrame(
        {
            "group": frame[group_column].astype(str).to_numpy(),
            "approved": approved.astype(int),
            "predicted_pd": probability,
            "default": frame[TARGET].to_numpy(),
        }
    )
    return (
        audit.groupby("group", as_index=False)
        .agg(
            applications=("approved", "size"),
            approval_rate=("approved", "mean"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("default", "mean"),
        )
        .sort_values("group", ignore_index=True)
    )
