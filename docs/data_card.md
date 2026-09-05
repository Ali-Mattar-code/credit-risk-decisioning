# Data card

## Provenance

All records are generated locally by `credit_risk.data.generate_credit_portfolio`. No customer, employer or third-party dataset is stored in this repository.

The generator was designed for this public reconstruction. It does not attempt to reproduce any lender’s portfolio, underwriting rules or protected internal information.

## Population

The default run creates 15,000 applications across 24 synthetic months. Variables cover affordability, exposure, bureau behavior, product structure and acquisition channel. A nonlinear outcome equation introduces interaction risk and a mild late-period shift.

## Target

`default_12m` is sampled from the generator’s latent probability and represents a synthetic binary default outcome within 12 months.

## Sensitive information

`age_group` exists solely to demonstrate post-score group diagnostics. It is excluded by the model feature contract. The synthetic label equation does not use age group.

## Known limitations

- Values are statistically plausible, not institutionally representative.
- Missingness and data-quality failures are not comprehensively simulated.
- No application is a real person and no result describes a real population.
- The late-cycle shock is designed for demonstration, not macroeconomic forecasting.

## Reproducibility

Seed `42` and all generation logic are committed. Changing the generator is a data-version change and should trigger a fresh validation report.
