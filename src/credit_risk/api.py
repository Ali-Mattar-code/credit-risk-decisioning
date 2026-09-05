"""FastAPI scoring surface for the fitted demonstration model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.serialization import load_model


MODEL_PATH = Path(os.getenv("CREDIT_RISK_MODEL", "artifacts/champion.joblib"))
app = FastAPI(title="Credit Risk Decisioning API", version="1.0.0")


class LoanApplication(BaseModel):
    annual_income: float = Field(gt=0)
    requested_amount: float = Field(gt=0)
    term_months: int = Field(ge=6, le=84)
    interest_rate: float = Field(gt=0, lt=1)
    debt_to_income: float = Field(ge=0, le=1)
    utilisation: float = Field(ge=0, le=1)
    bureau_score: int = Field(ge=300, le=900)
    previous_delinquencies: int = Field(ge=0)
    credit_history_months: int = Field(ge=0)
    employment_length_years: float = Field(ge=0)
    open_accounts: int = Field(ge=0)
    purpose: str
    channel: str


class ScoreResponse(BaseModel):
    probability_of_default: float
    risk_band: str
    model: str


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return load_model(MODEL_PATH)


def _risk_band(probability: float) -> str:
    if probability < 0.08:
        return "low"
    if probability < 0.18:
        return "moderate"
    if probability < 0.32:
        return "high"
    return "very_high"


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/score", response_model=ScoreResponse)
def score(application: LoanApplication) -> ScoreResponse:
    try:
        model = _model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    frame = pd.DataFrame([application.model_dump()])
    probability = float(model.predict_pd(frame)[0])
    return ScoreResponse(probability_of_default=probability, risk_band=_risk_band(probability), model=model.name)
