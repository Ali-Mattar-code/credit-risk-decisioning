# Engineering decisions

This project began from a small exploratory credit-modelling script. The public reconstruction keeps the useful hypothesis—compare models for default prediction—but redesigns the experiment around decisions, leakage control and evidence quality.

## What changed and why

### Random split → chronological split

Credit portfolios move with customer mix and economic conditions. A random split blends those regimes. The reconstruction reserves the latest 20% of applications as a genuinely later holdout and fits calibration on a separate preceding window.

### Global preprocessing → fitted pipelines

Scaling and imputation before splitting allow information from the test set into development. Every learned transformation now lives inside the scikit-learn pipeline and sees development data only.

### Accuracy → probability and policy metrics

An imbalanced classifier can achieve high accuracy by rejecting the minority class. ROC-AUC and average precision assess ranking; Brier score, log loss and ECE assess PD quality; the policy frontier measures what the score changes operationally.

### Predicted labels in ROC-AUC → predicted probabilities

ROC analysis requires a continuous ranking score. The reconstruction evaluates calibrated probabilities rather than thresholded class predictions.

### Test-set tuning → sealed holdout

The earlier exploration repeatedly inspected a test set. This design selects the champion on calibration data and reports the later holdout once.

### Headline uplift → equal-volume benchmark

Default reduction is meaningless if achieved by approving fewer applicants. The comparison fixes approvals at 60% for both the model and the transparent bureau heuristic.

### Notebook output → reproducible system

The implementation is a typed package with a CLI, deterministic evidence, tests, CI, an API, an interactive dashboard, model and data cards, and a documented evidence boundary.

## Deliberate exclusions

- The original exploratory file and local path are not published.
- No private internship code or dataset is represented as open source.
- No production claim is inferred from the synthetic benchmark.
- Reject inference is discussed but not simulated as if it had been validated.
