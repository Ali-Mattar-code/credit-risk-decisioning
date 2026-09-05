from __future__ import annotations

import numpy as np

from credit_risk.data import generate_credit_portfolio
from credit_risk.policy import approval_mask, evaluate_policy, policy_frontier, select_policy


def test_approval_mask_hits_requested_rate() -> None:
    probability = np.linspace(0.01, 0.99, 100)
    approved = approval_mask(probability, 0.63)
    assert approved.sum() == 63
    assert probability[approved].max() < probability[~approved].min()


def test_policy_frontier_and_selection_are_valid() -> None:
    frame = generate_credit_portfolio(800)
    probability = np.linspace(0.01, 0.55, len(frame))
    frontier = policy_frontier(frame, probability)
    selected = select_policy(frontier, max_mean_pd=0.14)
    assert len(frontier) == 15
    assert 0.20 <= selected["actual_approval_rate"] <= 0.90
    assert evaluate_policy(frame, probability, 0.50)["approved_applications"] == 400
