# AID (Automatic Identification of Demand)

**Run:** `cargo run --example aid`

## What this example demonstrates

Uses `AidAnalyzer` to classify demand time series by type (regular, intermittent, etc.), fit a statistical distribution, detect per-observation anomalies, and rank candidate distributions by information criterion. The builder API allows tuning of anomaly sensitivity, intermittent thresholds, and the choice of AIC vs BIC.

## Sections

1. **Regular demand** -- Analyzes a smooth trending series and prints summary statistics: demand type, fitted distribution, mean, variance, zero proportion, and shape/scale parameters.
2. **Intermittent demand** -- Analyzes a series with many zeros to demonstrate intermittent classification and zero-inflated distribution fitting.
3. **Per-observation anomaly labels** -- Injects outliers into a series, retrieves per-observation labels, prints a breakdown by label type, and lists flagged indices with convenience checks (`has_stockouts`, `is_new_product`, `is_obsolete_product`).
4. **Display formatting** -- Shows the `Display` impl for summary and features structs.
5. **Builder API** -- Configures `anomaly_alpha`, `intermittent_threshold`, `detect_anomalies`, and `ic` (AIC/BIC) to show how settings affect classification results.
6. **Information criterion scores** -- Ranks all candidate distributions by IC value and marks the selected one.
7. **Edge cases** -- Analyzes all-zeros and single-observation series.

## Key types

- `AidAnalyzer` -- builder for configuring and running demand classification
- `AidDemandType` -- enum for demand categories (e.g., regular, intermittent)
- `AidAnomalyLabel` -- per-observation label (Normal, outlier variants, etc.)
- `AidInformationCriterion` -- AIC or BIC selection
- `.summary()` / `.features()` -- access classification results and per-observation diagnostics
