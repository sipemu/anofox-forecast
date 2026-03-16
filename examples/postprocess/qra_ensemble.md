# QRA Ensemble Combining

**Run:** `cargo run --example postprocess_qra_ensemble`

## What this example demonstrates

Shows how to combine multiple point forecasters into calibrated probabilistic forecasts using Quantile Regression Averaging (QRA). Fits both standard and Lasso-regularized QRA models that learn per-quantile weights for each forecaster, then evaluates coverage and interval width on held-out data.

## Sections

1. **Simulate forecasters** -- Creates 3 synthetic forecasters: one with negative bias, one with positive bias, and one unbiased but noisy. Computes simple-average MAE as a baseline.
2. **Standard QRA** -- Fits `QRAPredictor::standard` on quantiles {0.1, 0.5, 0.9} and prints the learned intercept and forecaster coefficients for each quantile.
3. **Lasso QRA** -- Fits `QRAPredictor::lasso` with lambda=0.1 to encourage sparse weights, printing the regularized coefficients.
4. **Future predictions** -- Generates 10 future time steps from all 3 forecasters, runs QRA prediction, and displays the quantile forecasts alongside actuals and individual forecaster values.
5. **Evaluation** -- Computes 80%-interval coverage and average interval width on the future test set.

## Key types

- `QRAPredictor` -- quantile regression averager with constructors `standard` and `lasso`
- `QRAResult` (returned by `fit`) -- stores learned coefficients accessible via `coefficients(quantile_idx)`
- `QuantileForecasts` (returned by `predict`) -- per-time-step quantile values accessible via `at_time(i)`
