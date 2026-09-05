from __future__ import annotations

import numpy as np

from credit_risk.data import generate_credit_portfolio, temporal_split
from credit_risk.modeling import train_champion_challenger


def test_training_returns_bounded_probabilities() -> None:
    split = temporal_split(generate_credit_portfolio(1000, seed=11))
    result = train_champion_challenger(split, seed=11)
    probability = result.champion.predict_pd(split.holdout)
    assert len(result.challengers) == 2
    assert probability.shape == (200,)
    assert np.all((probability > 0.0) & (probability < 1.0))
