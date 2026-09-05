"""Auditable credit-risk modelling and decisioning toolkit."""

from credit_risk.data import generate_credit_portfolio, temporal_split
from credit_risk.modeling import PDModel, train_champion_challenger

__all__ = [
    "PDModel",
    "generate_credit_portfolio",
    "temporal_split",
    "train_champion_challenger",
]

__version__ = "1.0.0"
