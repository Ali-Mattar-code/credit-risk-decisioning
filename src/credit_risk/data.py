"""Synthetic portfolio generation and leakage-safe temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET = "default_12m"
DATE_COLUMN = "application_date"
ID_COLUMN = "application_id"
SENSITIVE_COLUMNS = ("age_group",)


@dataclass(frozen=True)
class PortfolioSplit:
    """Chronologically ordered development, calibration and holdout sets."""

    development: pd.DataFrame
    calibration: pd.DataFrame
    holdout: pd.DataFrame


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def generate_credit_portfolio(
    n_applications: int = 15_000,
    *,
    seed: int = 42,
    start: str = "2023-01-01",
    months: int = 24,
) -> pd.DataFrame:
    """Create a deterministic lending portfolio with realistic nonlinear risk.

    The generator is intentionally transparent. It introduces a mild late-period
    macroeconomic shift so the out-of-time holdout tests both performance and
    monitoring behavior. It is not a simulation of any named lender or geography.
    """

    if n_applications < 500:
        raise ValueError("n_applications must be at least 500")

    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp(start)
    day_offsets = np.sort(rng.integers(0, months * 30, size=n_applications))
    dates = start_date + pd.to_timedelta(day_offsets, unit="D")
    time_fraction = day_offsets / max(months * 30 - 1, 1)

    annual_income = np.clip(rng.lognormal(np.log(48_000), 0.55, n_applications), 12_000, 240_000)
    bureau_score = np.clip(rng.normal(650, 72, n_applications), 350, 850).round()
    debt_to_income = np.clip(rng.beta(2.3, 4.2, n_applications) * 0.85, 0.01, 0.85)
    utilisation = np.clip(rng.beta(2.0, 2.7, n_applications), 0.01, 1.0)
    previous_delinquencies = np.clip(rng.poisson(0.55, n_applications), 0, 8)
    credit_history_months = np.clip(rng.gamma(4.5, 24.0, n_applications), 3, 420).round()
    employment_length_years = np.clip(rng.gamma(2.2, 2.8, n_applications), 0, 35)
    open_accounts = np.clip(rng.poisson(6.5, n_applications) + 1, 1, 28)
    requested_amount = np.clip(
        annual_income * rng.uniform(0.06, 0.48, n_applications), 1_000, 45_000
    ).round(2)
    term_months = rng.choice([12, 24, 36, 48, 60], n_applications, p=[0.08, 0.19, 0.36, 0.20, 0.17])
    interest_rate = np.clip(
        0.075
        + (690 - bureau_score) * 0.00022
        + debt_to_income * 0.055
        + rng.normal(0, 0.012, n_applications),
        0.045,
        0.32,
    )
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "vehicle", "education", "small_business"],
        n_applications,
        p=[0.42, 0.19, 0.20, 0.08, 0.11],
    )
    channel = rng.choice(["direct", "partner", "broker"], n_applications, p=[0.52, 0.31, 0.17])
    age = np.clip(rng.normal(39, 11, n_applications), 21, 70).round()
    age_group = pd.cut(
        age,
        bins=[20, 29, 39, 49, 59, 71],
        labels=["21-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    ).astype(str)

    loan_to_income = requested_amount / annual_income
    purpose_effect = pd.Series(purpose).map(
        {
            "debt_consolidation": 0.10,
            "home_improvement": -0.12,
            "vehicle": -0.02,
            "education": 0.05,
            "small_business": 0.28,
        }
    ).to_numpy()
    channel_effect = pd.Series(channel).map({"direct": -0.05, "partner": 0.03, "broker": 0.16}).to_numpy()
    late_cycle_shock = np.maximum(time_fraction - 0.72, 0.0) * 1.8
    nonlinear_stress = ((utilisation > 0.82) & (debt_to_income > 0.42)).astype(float) * 0.65

    log_odds = (
        -3.55
        + 2.15 * debt_to_income
        + 1.35 * utilisation
        + 0.34 * previous_delinquencies
        - 0.0082 * (bureau_score - 650)
        + 1.15 * loan_to_income
        - 0.0022 * (credit_history_months - 90)
        + purpose_effect
        + channel_effect
        + late_cycle_shock
        + nonlinear_stress
        + rng.normal(0, 0.18, n_applications)
    )
    probability = _sigmoid(log_odds)
    default = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            ID_COLUMN: [f"APP-{index:06d}" for index in range(1, n_applications + 1)],
            DATE_COLUMN: dates,
            "annual_income": annual_income.round(2),
            "requested_amount": requested_amount,
            "term_months": term_months,
            "interest_rate": interest_rate.round(5),
            "debt_to_income": debt_to_income.round(5),
            "utilisation": utilisation.round(5),
            "bureau_score": bureau_score.astype(int),
            "previous_delinquencies": previous_delinquencies.astype(int),
            "credit_history_months": credit_history_months.astype(int),
            "employment_length_years": employment_length_years.round(2),
            "open_accounts": open_accounts.astype(int),
            "purpose": purpose,
            "channel": channel,
            "age_group": age_group,
            TARGET: default.astype(int),
        }
    ).sort_values(DATE_COLUMN, ignore_index=True)


def temporal_split(
    frame: pd.DataFrame,
    development_fraction: float = 0.65,
    calibration_fraction: float = 0.15,
) -> PortfolioSplit:
    """Split chronologically, preserving a genuinely unseen final holdout."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if development_fraction + calibration_fraction >= 1.0:
        raise ValueError("development and calibration fractions must sum to less than one")

    ordered = frame.sort_values(DATE_COLUMN, ignore_index=True)
    development_end = int(len(ordered) * development_fraction)
    calibration_end = int(len(ordered) * (development_fraction + calibration_fraction))
    return PortfolioSplit(
        development=ordered.iloc[:development_end].copy(),
        calibration=ordered.iloc[development_end:calibration_end].copy(),
        holdout=ordered.iloc[calibration_end:].copy(),
    )
