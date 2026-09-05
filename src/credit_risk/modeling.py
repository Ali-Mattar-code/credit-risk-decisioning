"""Champion-challenger training with held-out isotonic calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import PortfolioSplit, TARGET
from credit_risk.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    assert_no_forbidden_features,
    model_matrix,
)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)), ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
    )


def candidate_pipelines(seed: int = 42) -> dict[str, Pipeline]:
    """Interpretable baseline and nonlinear challenger."""

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor()),
                ("classifier", LogisticRegression(C=0.7, max_iter=1_500, class_weight="balanced", random_state=seed)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.055,
                        max_leaf_nodes=20,
                        min_samples_leaf=45,
                        l2_regularization=0.8,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


@dataclass
class PDModel:
    """A fitted base estimator plus an independently fitted PD calibrator."""

    name: str
    pipeline: Pipeline
    calibrator: IsotonicRegression
    feature_names: tuple[str, ...] = MODEL_FEATURES

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(model_matrix(frame))[:, 1]

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(frame)
        return np.clip(self.calibrator.predict(raw), 1e-5, 1.0 - 1e-5)


@dataclass(frozen=True)
class TrainingResult:
    champion: PDModel
    challengers: dict[str, PDModel]
    validation_scores: dict[str, dict[str, float]]


def train_champion_challenger(split: PortfolioSplit, *, seed: int = 42) -> TrainingResult:
    """Fit on development, calibrate on calibration, select by calibrated Brier score."""

    assert_no_forbidden_features(MODEL_FEATURES)
    x_development = model_matrix(split.development)
    y_development = split.development[TARGET].to_numpy()
    y_calibration = split.calibration[TARGET].to_numpy()

    fitted: dict[str, PDModel] = {}
    scores: dict[str, dict[str, float]] = {}
    for name, pipeline in candidate_pipelines(seed).items():
        pipeline.fit(x_development, y_development)
        raw_probability = pipeline.predict_proba(model_matrix(split.calibration))[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0 - 1e-5)
        calibrator.fit(raw_probability, y_calibration)
        calibrated_probability = calibrator.predict(raw_probability)
        fitted[name] = PDModel(name=name, pipeline=pipeline, calibrator=calibrator)
        scores[name] = {
            "roc_auc": float(roc_auc_score(y_calibration, raw_probability)),
            "brier_raw": float(brier_score_loss(y_calibration, raw_probability)),
            "brier_calibrated": float(brier_score_loss(y_calibration, calibrated_probability)),
        }

    champion_name = min(scores, key=lambda name: (scores[name]["brier_calibrated"], -scores[name]["roc_auc"]))
    return TrainingResult(champion=fitted[champion_name], challengers=fitted, validation_scores=scores)


def model_metadata(model: PDModel) -> dict[str, Any]:
    return {
        "model_name": model.name,
        "calibration": "isotonic regression on a chronological calibration window",
        "target": TARGET,
        "features": list(model.feature_names),
        "sensitive_features_excluded": ["age_group"],
    }
