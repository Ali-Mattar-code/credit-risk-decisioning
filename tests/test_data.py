from __future__ import annotations

import pandas as pd

from credit_risk.data import generate_credit_portfolio, temporal_split
from credit_risk.features import MODEL_FEATURES


def test_generator_is_deterministic() -> None:
    first = generate_credit_portfolio(600, seed=9)
    second = generate_credit_portfolio(600, seed=9)
    pd.testing.assert_frame_equal(first, second)


def test_temporal_split_has_no_date_overlap() -> None:
    split = temporal_split(generate_credit_portfolio(1000))
    assert split.development["application_date"].max() <= split.calibration["application_date"].min()
    assert split.calibration["application_date"].max() <= split.holdout["application_date"].min()
    assert len(split.development) == 650
    assert len(split.calibration) == 150
    assert len(split.holdout) == 200


def test_sensitive_attribute_is_not_a_model_feature() -> None:
    assert "age_group" not in MODEL_FEATURES
    assert "default_12m" not in MODEL_FEATURES
    assert "application_date" not in MODEL_FEATURES
