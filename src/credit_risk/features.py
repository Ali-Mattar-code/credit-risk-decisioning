"""Feature contracts for probability-of-default models."""

from __future__ import annotations

import pandas as pd

from credit_risk.data import DATE_COLUMN, ID_COLUMN, SENSITIVE_COLUMNS, TARGET


NUMERIC_FEATURES = (
    "annual_income",
    "requested_amount",
    "term_months",
    "interest_rate",
    "debt_to_income",
    "utilisation",
    "bureau_score",
    "previous_delinquencies",
    "credit_history_months",
    "employment_length_years",
    "open_accounts",
    "loan_to_income",
    "credit_headroom",
    "delinquency_pressure",
)
CATEGORICAL_FEATURES = ("purpose", "channel")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable, interpretable ratios without using future outcomes."""

    features = frame.copy()
    features["loan_to_income"] = features["requested_amount"] / features["annual_income"].clip(lower=1.0)
    features["credit_headroom"] = (1.0 - features["utilisation"]) * features["annual_income"]
    features["delinquency_pressure"] = features["previous_delinquencies"] / (
        1.0 + features["credit_history_months"] / 12.0
    )
    return features


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only approved model inputs; IDs, dates and sensitive fields stay out."""

    engineered = engineer_features(frame)
    return engineered.loc[:, MODEL_FEATURES]


def assert_no_forbidden_features(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = {TARGET, DATE_COLUMN, ID_COLUMN, *SENSITIVE_COLUMNS}
    overlap = forbidden.intersection(columns)
    if overlap:
        raise ValueError(f"Forbidden model features detected: {sorted(overlap)}")
