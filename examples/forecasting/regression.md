# Regression Forecaster Example

**Run:** `cargo run --example regression`

## What this example demonstrates

The multi-backend `RegressionForecaster` — a unified interface that wraps 11 different
regression estimators from `anofox-regression` behind the `Forecaster` trait.

## Sections

1. **OLS: trend + AR(3)** — Ordinary Least Squares with trend index and 3 autoregressive lags. Recursive multi-step prediction feeds predictions back as lag features.

2. **OLS: trend + Fourier** — Fourier seasonality (period=7, order=3) produces 6 sin/cos columns that capture the weekly cycle without requiring dummy variables.

3. **Ridge** — L2-regularized regression (lambda=0.1) shrinks coefficients toward zero, reducing overfitting when features are correlated.

4. **Elastic Net** — Combined L1 + L2 regularization (lambda=0.1, alpha=0.5). The alpha parameter controls the Lasso/Ridge mix.

5. **Quantile regression** — Estimates the conditional median (tau=0.5) instead of the conditional mean. Robust to outliers and useful for asymmetric loss functions.

6. **WLS (Weighted Least Squares)** — Exponential decay weighting (0.97) emphasizes recent observations. Useful when the data-generating process changes over time.

7. **RLS (Recursive Least Squares)** — Adaptive coefficients via a forgetting factor (0.98). Coefficients evolve as new data arrives.

8. **BLS/NNLS** — Bounded Least Squares with non-negative coefficient constraints. Ensures all coefficients are >= 0.

9. **Poisson GLM** — Generalized linear model for count data. Uses a log link function.

10. **Tweedie GLM** — Generalized linear model for continuous positive data with variance power 1.5 (between Poisson and Gamma).

11. **Dynamic Linear Model** — Time-varying parameters via AICc-weighted model averaging. Coefficients can change over time.

12. **Advanced trend components** — `TrendType::Quadratic` and `TrendType::TheilSen` as regression features. These are fitted during `build_matrices()` and their predictions are used as design-matrix columns.

13. **Dummy seasonality** — One-hot encoding (period=7) captures arbitrary seasonal shapes without smoothness assumptions, unlike Fourier terms.

14. **Changepoint step functions** — `ChangepointFeature` with `StepFunctions` encoding creates binary columns that flip 0→1 at each changepoint index. During prediction, columns are forward-filled with their last training value.

15. **Changepoint regime index** — `ChangepointEncoding::RegimeIndex` produces a single ordinal column (0, 1, 2, …) instead of k binary columns.

16. **Feature safety classification** — `classify_features()` returns per-column safety labels: `Deterministic` (safe in any CV), `DataDependent` (re-fit per fold), `Structural` (forward-filled), `External` (user-supplied).

17. **Combined pipeline** — Ridge + trend + Fourier + changepoint on a series with a level shift. Demonstrates how all feature types compose.

## Key types

- `RegressionForecaster` — the main forecaster struct
- `RegressionFeatures` — builder for feature configuration
- `RegressionBackend` — enum selecting the regression estimator
- `TrendType` — fitted trend components (Linear, Quadratic, Cubic, Exponential, TheilSen)
- `SeasonalSpec` — seasonal features (Fourier, Dummy)
- `ChangepointFeature` / `ChangepointEncoding` — structural regime indicators
- `FeatureSafety` — leakage risk classification
