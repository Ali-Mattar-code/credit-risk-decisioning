# Model card

## Model details

- **Task:** binary probability of default within 12 months
- **Champion selection:** lowest calibrated Brier score on a chronological calibration window
- **Candidates:** logistic regression and histogram gradient boosting
- **Calibration:** isotonic regression
- **Version:** 1.0.0
- **Licence:** MIT

The committed reference run selects logistic regression. Selection may differ if the seed, data generator or hyperparameters change.

## Intended use

This model demonstrates a sound technical workflow for credit-risk ranking, probability calibration, policy analysis and monitoring. It can support portfolio reviews, interviews and engineering discussion.

It must not be used to make real lending decisions. The training population is synthetic and has not been tested for representativeness, legal compliance or real-world stability.

## Inputs

Income, requested exposure, term, rate, debt-to-income ratio, utilisation, bureau score, delinquencies, credit-history length, employment length, open accounts, purpose and channel. Three transparent ratios are derived from these values.

Age group is generated only for diagnostic monitoring and is explicitly blocked from the model feature contract.

## Performance

See [`results/reference/metrics.json`](../results/reference/metrics.json) for machine-readable results. The current deterministic run reports 0.716 ROC-AUC and 0.154 Brier score on 3,000 later synthetic applications.

These results describe this generator only. They are not production estimates and should not be compared directly with a lender’s reported model without matching population, outcome horizon and policy.

## Limitations

- Synthetic relationships are simpler than real borrower behavior.
- No reject inference is attempted.
- The economic objective omits prepayment, collections timing, capital and operational costs.
- The diagnostic group report is not a complete fairness assessment.
- Group-rate Wilson intervals cover binomial sampling uncertainty only; they do not address selection bias, multiple comparisons or outcome maturity.
- Isotonic calibration can be unstable when calibration samples are small.
- A mature production model requires independent validation, audit trails and monitored outcome maturity.
