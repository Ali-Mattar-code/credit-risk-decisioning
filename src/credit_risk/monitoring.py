"""Data drift, calibration drift and decision fairness diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import TARGET
from credit_risk.policy import approval_mask


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion.

    Wilson intervals stay within ``[0, 1]`` and behave more sensibly than a
    normal approximation for small groups or rates near zero and one.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    if z <= 0:
        raise ValueError("z must be positive")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    )
    return float(centre - margin), float(centre + margin)


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
    report = (
        audit.groupby("group", as_index=False)
        .agg(
            applications=("approved", "size"),
            approvals=("approved", "sum"),
            defaults=("default", "sum"),
            approval_rate=("approved", "mean"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("default", "mean"),
        )
        .sort_values("group", ignore_index=True)
    )
    approval_intervals = [
        wilson_interval(int(row.approvals), int(row.applications)) for row in report.itertuples()
    ]
    default_intervals = [
        wilson_interval(int(row.defaults), int(row.applications)) for row in report.itertuples()
    ]
    report["approval_rate_ci_95_low"] = [interval[0] for interval in approval_intervals]
    report["approval_rate_ci_95_high"] = [interval[1] for interval in approval_intervals]
    report["observed_default_rate_ci_95_low"] = [interval[0] for interval in default_intervals]
    report["observed_default_rate_ci_95_high"] = [interval[1] for interval in default_intervals]
    portfolio_approval_rate = float(report["approvals"].sum() / report["applications"].sum())
    report["approval_rate_gap_pp_vs_portfolio"] = 100.0 * (
        report["approval_rate"] - portfolio_approval_rate
    )
    return report
