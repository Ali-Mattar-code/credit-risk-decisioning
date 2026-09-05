"""End-to-end reproducible experiment and reporting pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.data import TARGET, generate_credit_portfolio, temporal_split
from credit_risk.evaluation import binary_metrics, calibration_table
from credit_risk.modeling import model_metadata, train_champion_challenger
from credit_risk.monitoring import drift_report, group_audit
from credit_risk.policy import approval_mask, evaluate_policy, heuristic_score, policy_frontier, select_policy
from credit_risk.serialization import save_model, write_json


COLORS = {"navy": "#17365D", "blue": "#2F75B5", "teal": "#1B998B", "gold": "#E2A93B", "red": "#D9534F"}


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _model_comparison_figure(metrics: dict[str, dict[str, float]], path: Path) -> None:
    names = list(metrics)
    aucs = [metrics[name]["roc_auc"] for name in names]
    briers = [metrics[name]["brier_score"] for name in names]
    labels = [name.replace("_", " ").title() for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, aucs, color=[COLORS["blue"], COLORS["teal"]])
    axes[0].set_ylim(max(0.5, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    axes[0].set_title("Out-of-time ROC-AUC")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(labels, briers, color=[COLORS["blue"], COLORS["teal"]])
    axes[1].set_ylim(0, max(briers) * 1.25)
    axes[1].set_title("Out-of-time Brier score ↓")
    axes[1].tick_params(axis="x", rotation=12)
    figure.suptitle("Champion–challenger comparison", color=COLORS["navy"], fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, path)


def _calibration_figure(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#8A94A3", label="Perfect calibration")
    axis.plot(
        table["mean_predicted_pd"],
        table["observed_default_rate"],
        marker="o",
        linewidth=2.2,
        color=COLORS["teal"],
        label="Champion",
    )
    upper = max(table["mean_predicted_pd"].max(), table["observed_default_rate"].max()) * 1.12
    axis.set(xlim=(0, upper), ylim=(0, upper), xlabel="Mean predicted PD", ylabel="Observed default rate")
    axis.set_title("Out-of-time calibration by risk decile", color=COLORS["navy"], fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    _save_figure(figure, path)


def _policy_figure(frontier: pd.DataFrame, selected: dict[str, float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(
        frontier["actual_approval_rate"] * 100,
        frontier["observed_approved_default_rate"] * 100,
        color=COLORS["blue"],
        marker="o",
        label="Observed default rate",
    )
    axis.scatter(
        [selected["actual_approval_rate"] * 100],
        [selected["observed_approved_default_rate"] * 100],
        color=COLORS["gold"],
        edgecolor=COLORS["navy"],
        s=95,
        zorder=3,
        label="Selected policy",
    )
    axis.set(xlabel="Approval rate (%)", ylabel="Approved default rate (%)")
    axis.set_title("Risk–growth policy frontier", color=COLORS["navy"], fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, path)


def _monitoring_figure(report: pd.DataFrame, path: Path) -> None:
    color_map = {"stable": COLORS["teal"], "watch": COLORS["gold"], "investigate": COLORS["red"]}
    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    axis.barh(report["feature"], report["psi"], color=[color_map[value] for value in report["status"]])
    axis.axvline(0.10, color=COLORS["gold"], linestyle="--", linewidth=1.2, label="Watch")
    axis.axvline(0.25, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Investigate")
    axis.set(xlabel="Population Stability Index", title="Out-of-time drift monitor")
    axis.title.set_color(COLORS["navy"])
    axis.title.set_fontweight("bold")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.18)
    figure.tight_layout()
    _save_figure(figure, path)


def _dashboard_figure(
    metrics: dict[str, float],
    selected: dict[str, float],
    comparison: dict[str, float],
    drift: pd.DataFrame,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(12, 6.6))
    grid = figure.add_gridspec(2, 4, hspace=0.48, wspace=0.35)
    cards = [
        ("ROC-AUC", metrics["roc_auc"], "{:.3f}"),
        ("Brier score", metrics["brier_score"], "{:.3f}"),
        ("Selected approval", selected["actual_approval_rate"], "{:.0%}"),
        ("Default reduction*", comparison["relative_default_reduction"], "{:.1%}"),
    ]
    for index, (label, value, template) in enumerate(cards):
        axis = figure.add_subplot(grid[0, index])
        axis.axis("off")
        axis.text(
            0.5,
            0.68,
            template.format(value),
            ha="center",
            va="center",
            fontsize=24,
            color=COLORS["navy"],
            fontweight="bold",
        )
        axis.text(0.5, 0.30, label, ha="center", va="center", fontsize=10, color="#536171")
        axis.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, fill=False, edgecolor="#D8E0EA", linewidth=1.1))
    axis = figure.add_subplot(grid[1, :])
    color_map = {"stable": COLORS["teal"], "watch": COLORS["gold"], "investigate": COLORS["red"]}
    axis.bar(drift["feature"], drift["psi"], color=[color_map[value] for value in drift["status"]])
    axis.axhline(0.10, color=COLORS["gold"], linestyle="--", linewidth=1.2)
    axis.axhline(0.25, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axis.set_ylabel("PSI")
    axis.set_title("Monitoring snapshot", color=COLORS["navy"], fontweight="bold")
    axis.grid(axis="y", alpha=0.18)
    figure.suptitle("Credit Risk Decisioning — Reference Run", fontsize=17, color=COLORS["navy"], fontweight="bold")
    figure.text(
        0.01,
        0.01,
        "*Versus a transparent heuristic at the same approval rate; synthetic holdout only.",
        fontsize=8,
        color="#536171",
    )
    _save_figure(figure, path)


def run_reproduction(
    *,
    output_dir: str | Path = "results/reference",
    model_path: str | Path = "artifacts/champion.joblib",
    n_applications: int = 15_000,
    seed: int = 42,
) -> dict[str, object]:
    output = Path(output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    portfolio = generate_credit_portfolio(n_applications=n_applications, seed=seed)
    split = temporal_split(portfolio)
    training = train_champion_challenger(split, seed=seed)

    holdout_metrics: dict[str, dict[str, float]] = {}
    for name, model in training.challengers.items():
        probability = model.predict_pd(split.holdout)
        holdout_metrics[name] = binary_metrics(split.holdout[TARGET].to_numpy(), probability)

    champion = training.champion
    champion_pd = champion.predict_pd(split.holdout)
    calibration_pd = champion.predict_pd(split.calibration)
    champion_metrics = holdout_metrics[champion.name]
    calibration = calibration_table(split.holdout[TARGET].to_numpy(), champion_pd)
    frontier = policy_frontier(split.holdout, champion_pd)
    selected = select_policy(frontier)
    drift = drift_report(split.calibration, split.holdout, calibration_pd, champion_pd)
    fairness = group_audit(split.holdout, champion_pd, target_approval_rate=selected["actual_approval_rate"])

    benchmark_rate = 0.60
    model_policy = evaluate_policy(split.holdout, champion_pd, benchmark_rate)
    baseline_policy = evaluate_policy(split.holdout, heuristic_score(split.holdout), benchmark_rate)
    baseline_default = baseline_policy["observed_approved_default_rate"]
    model_default = model_policy["observed_approved_default_rate"]
    comparison = {
        "approval_rate": benchmark_rate,
        "model_observed_default_rate": model_default,
        "heuristic_observed_default_rate": baseline_default,
        "relative_default_reduction": float((baseline_default - model_default) / baseline_default),
        "scope": "synthetic out-of-time holdout; not a production result",
    }

    calibration.to_csv(output / "calibration_by_decile.csv", index=False)
    frontier.to_csv(output / "policy_frontier.csv", index=False)
    drift.to_csv(output / "drift_report.csv", index=False)
    fairness.to_csv(output / "group_audit.csv", index=False)
    save_model(champion, model_path)

    summary: dict[str, object] = {
        "evidence_scope": "fully reproducible synthetic-data benchmark",
        "not_a_production_claim": True,
        "seed": seed,
        "sample_sizes": {
            "development": len(split.development),
            "calibration": len(split.calibration),
            "out_of_time_holdout": len(split.holdout),
        },
        "date_windows": {
            "development": [
                str(split.development["application_date"].min().date()),
                str(split.development["application_date"].max().date()),
            ],
            "calibration": [
                str(split.calibration["application_date"].min().date()),
                str(split.calibration["application_date"].max().date()),
            ],
            "out_of_time_holdout": [
                str(split.holdout["application_date"].min().date()),
                str(split.holdout["application_date"].max().date()),
            ],
        },
        "champion": model_metadata(champion),
        "validation_scores": training.validation_scores,
        "holdout_metrics": holdout_metrics,
        "selected_policy": selected,
        "constant_approval_comparison": comparison,
        "monitoring": {
            "max_psi": float(drift["psi"].max()),
            "features_to_investigate": drift.loc[drift["status"] == "investigate", "feature"].tolist(),
        },
        "governance": {
            "sensitive_feature_excluded_from_model": "age_group",
            "group_audit_is_diagnostic_not_a_fairness_certification": True,
            "deployment_status": "demonstration only",
        },
    }
    write_json(summary, output / "metrics.json")

    _model_comparison_figure(holdout_metrics, figures / "model_comparison.png")
    _calibration_figure(calibration, figures / "calibration.png")
    _policy_figure(frontier, selected, figures / "policy_frontier.png")
    _monitoring_figure(drift, figures / "monitoring.png")
    _dashboard_figure(champion_metrics, selected, comparison, drift, figures / "decision_dashboard.png")
    return summary
