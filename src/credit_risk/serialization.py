"""Safe local persistence helpers for model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from credit_risk.modeling import PDModel


def save_model(model: PDModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> PDModel:
    return joblib.load(Path(path))


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
