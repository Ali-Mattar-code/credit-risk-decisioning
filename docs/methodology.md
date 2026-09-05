# Methodology

## Objective

Estimate a 12-month probability of default (PD), convert that estimate into a transparent approval policy, and expose the diagnostics needed to challenge the model after deployment.

The project optimises neither accuracy nor approval volume in isolation. It treats lending as a constrained decision problem:

\[
\max_{a_i \in \{0,1\}} \sum_i a_i\left[(1-p_i)\,m_i\,EAD_i - p_i\,LGD\,EAD_i\right]
\]

subject to a portfolio risk appetite:

\[
\frac{\sum_i a_i p_i}{\sum_i a_i} \leq 12\%.
\]

Here, \(a_i\) is the approval decision, \(p_i\) the calibrated PD, \(m_i\) the term-adjusted interest margin, \(EAD_i\) exposure at default and \(LGD\) loss given default. This is a compact demonstration objective, not a full lender P&L.

## Experimental protocol

Applications are ordered by application date and split into:

- **65% development:** fit preprocessing and base classifiers.
- **15% calibration:** fit isotonic maps and select the champion.
- **20% holdout:** calculate final metrics exactly once on a later window.

The split is temporal because a random split can make credit models appear more stable than they are. The synthetic generator introduces a mild late-period macro shock so monitoring has a meaningful job.

## Models

The champion–challenger set contains:

1. Regularised logistic regression, providing a familiar and explainable baseline.
2. Histogram gradient boosting, capturing nonlinear interactions without an excessive search space.

Numeric imputation and scaling, plus categorical imputation and one-hot encoding, are fitted inside each pipeline. IDs, dates, outcomes and the monitoring-only age group cannot enter the model matrix.

## Calibration

The raw classifier score \(s\) is mapped to a probability using monotonic isotonic regression:

\[
\hat{p}=f_{iso}(s).
\]

The mapping is fitted on the calibration window only. Brier score selects the champion, with ROC-AUC as a ranking-quality tie-breaker.

## Evaluation

The holdout report covers:

- ROC-AUC and KS for ranking
- average precision for class-imbalance sensitivity
- Brier score, log loss and expected calibration error for probability quality
- observed defaults and expected economics across an approval frontier
- default-rate comparison against a disclosed heuristic at equal approvals

The benchmark heuristic uses only bureau score and prior delinquencies. It is intentionally simple and is not described as an incumbent lender policy.

## Monitoring

Population Stability Index (PSI) compares calibration and holdout distributions:

\[
PSI=\sum_b (q_b-p_b)\ln\left(\frac{q_b}{p_b}\right).
\]

Values below 0.10 are labelled stable, 0.10–0.25 watch, and 0.25 or above investigate. These thresholds are conventions, not universal regulatory limits.

The group audit reports approval and risk by age band. It is a diagnostic surface, not proof of fairness. Age is absent from model inputs.
