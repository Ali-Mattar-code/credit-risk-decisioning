"""Interactive review surface for generated evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reference"

st.set_page_config(page_title="Credit Risk Decisioning", page_icon="📊", layout="wide")
st.title("Credit Risk Decisioning")
st.caption("Synthetic-data demonstration · chronological holdout · auditable policy layer")

if not (RESULTS / "metrics.json").exists():
    st.error("Reference outputs are missing. Run `credit-risk reproduce` first.")
    st.stop()

summary = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
champion = summary["holdout_metrics"][summary["champion"]["model_name"]]
comparison = summary["constant_approval_comparison"]
policy = summary["selected_policy"]

columns = st.columns(4)
columns[0].metric("ROC-AUC", f"{champion['roc_auc']:.3f}")
columns[1].metric("Brier score", f"{champion['brier_score']:.3f}")
columns[2].metric("Selected approval", f"{policy['actual_approval_rate']:.0%}")
columns[3].metric("Default reduction*", f"{comparison['relative_default_reduction']:.1%}")
st.caption("*Against the transparent heuristic at equal approvals; synthetic out-of-time holdout only.")

left, right = st.columns(2)
with left:
    st.subheader("Policy frontier")
    frontier = pd.read_csv(RESULTS / "policy_frontier.csv")
    st.line_chart(frontier, x="actual_approval_rate", y="observed_approved_default_rate")
with right:
    st.subheader("Calibration")
    calibration = pd.read_csv(RESULTS / "calibration_by_decile.csv")
    st.line_chart(calibration, x="mean_predicted_pd", y="observed_default_rate")

st.subheader("Drift controls")
st.dataframe(pd.read_csv(RESULTS / "drift_report.csv"), use_container_width=True, hide_index=True)
st.subheader("Diagnostic group audit")
st.dataframe(pd.read_csv(RESULTS / "group_audit.csv"), use_container_width=True, hide_index=True)
st.info("This portfolio is synthetic and the dashboard is a technical demonstration, not a lending recommendation.")
