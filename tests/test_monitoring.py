from __future__ import annotations

import numpy as np

from credit_risk.monitoring import population_stability_index


def test_psi_detects_distribution_shift() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(0.0, 1.0, 4000)
    stable = rng.normal(0.0, 1.0, 4000)
    shifted = rng.normal(1.2, 1.0, 4000)
    assert population_stability_index(reference, stable) < 0.10
    assert population_stability_index(reference, shifted) > 0.25
