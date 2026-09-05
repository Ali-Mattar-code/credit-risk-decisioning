"""Risk-ranked approval policies and transparent unit economics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import TARGET


def approval_mask(probability: np.ndarray, target_approval_rate: float) -> np.ndarray:
    """Approve exactly the lowest-risk fraction, with deterministic tie-breaking."""

    if not 0.0 < target_approval_rate <= 1.0:
        raise ValueError("target_approval_rate must be in (0, 1]")
    approved_count = max(1, int(round(len(probability) * target_approval_rate)))
    order = np.argsort(probability, kind="stable")
    mask = np.zeros(len(probability), dtype=bool)
    mask[order[:approved_count]] = True
    return mask


def evaluate_policy(
    frame: pd.DataFrame,
    probability: np.ndarray,
    target_approval_rate: float,
    *,
    loss_given_default: float = 0.62,
    funding_cost: float = 0.045,
    servicing_cost_rate: float = 0.012,
) -> dict[str, float]:
    approved = approval_mask(probability, target_approval_rate)
    exposure = frame["requested_amount"].to_numpy()[approved]
    pd_approved = probability[approved]
    duration = frame["term_months"].to_numpy()[approved] / 12.0
    annual_rate = frame["interest_rate"].to_numpy()[approved]
    interest_margin = np.maximum(annual_rate - funding_cost - servicing_cost_rate, 0.0)
    expected_interest = exposure * duration * interest_margin * (1.0 - pd_approved)
    expected_loss = exposure * loss_given_default * pd_approved
    expected_profit = expected_interest - expected_loss
    observed_default = frame[TARGET].to_numpy()[approved]
    return {
        "target_approval_rate": float(target_approval_rate),
        "actual_approval_rate": float(approved.mean()),
        "approved_applications": int(approved.sum()),
        "score_cutoff": float(np.max(probability[approved])),
        "mean_approved_pd": float(np.mean(pd_approved)),
        "observed_approved_default_rate": float(np.mean(observed_default)),
        "expected_loss": float(expected_loss.sum()),
        "expected_profit": float(expected_profit.sum()),
        "expected_profit_per_approval": float(expected_profit.mean()),
    }


def policy_frontier(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    rates = np.round(np.arange(0.20, 0.91, 0.05), 2)
    return pd.DataFrame([evaluate_policy(frame, probability, float(rate)) for rate in rates])


def select_policy(frontier: pd.DataFrame, max_mean_pd: float = 0.12) -> dict[str, float]:
    eligible = frontier.loc[frontier["mean_approved_pd"] <= max_mean_pd]
    if eligible.empty:
        selected = frontier.loc[frontier["mean_approved_pd"].idxmin()]
    else:
        selected = eligible.loc[eligible["expected_profit"].idxmax()]
    return {key: float(value) for key, value in selected.to_dict().items()}


def heuristic_score(frame: pd.DataFrame) -> np.ndarray:
    """A deliberately simple bureau-policy proxy used only as a benchmark."""

    bureau_risk = (850.0 - frame["bureau_score"].to_numpy()) / 500.0
    delinquency_risk = np.minimum(frame["previous_delinquencies"].to_numpy() / 4.0, 1.0)
    score = 0.82 * bureau_risk + 0.18 * delinquency_risk
    return np.clip(score, 0.0, 1.0)
