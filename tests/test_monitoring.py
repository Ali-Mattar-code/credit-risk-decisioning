from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import TARGET
from credit_risk.monitoring import group_audit, population_stability_index, wilson_interval


def test_psi_detects_distribution_shift() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(0.0, 1.0, 4000)
    stable = rng.normal(0.0, 1.0, 4000)
    shifted = rng.normal(1.2, 1.0, 4000)
    assert population_stability_index(reference, stable) < 0.10
    assert population_stability_index(reference, shifted) > 0.25


def test_wilson_interval_is_bounded_for_small_extreme_groups() -> None:
    assert wilson_interval(0, 5) == pytest.approx((0.0, 0.43448246478317476))
    low, high = wilson_interval(5, 5)
    assert low == pytest.approx(0.5655175352168251)
    assert high == pytest.approx(1.0)


def test_group_audit_reports_counts_uncertainty_and_portfolio_gap() -> None:
    frame = pd.DataFrame(
        {
            "age_group": ["A", "A", "A", "B", "B", "B"],
            TARGET: [0, 0, 1, 0, 1, 1],
        }
    )
    probability = np.array([0.05, 0.10, 0.40, 0.15, 0.30, 0.50])
    report = group_audit(frame, probability, target_approval_rate=0.5)

    assert report["applications"].sum() == len(frame)
    assert report["approvals"].sum() == 3
    assert report["defaults"].sum() == 3
    assert np.all(report["approval_rate_ci_95_low"] <= report["approval_rate"])
    assert np.all(report["approval_rate"] <= report["approval_rate_ci_95_high"])
    assert np.all(report["observed_default_rate_ci_95_low"] <= report["observed_default_rate"])
    assert np.all(report["observed_default_rate"] <= report["observed_default_rate_ci_95_high"])
    weighted_gap = np.average(
        report["approval_rate_gap_pp_vs_portfolio"],
        weights=report["applications"],
    )
    assert weighted_gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("successes,total", [(-1, 5), (6, 5), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)
