# Examples

Each example has a companion `.md` file with a detailed description.

## Getting Started

| Example | Run | Description |
|---------|-----|-------------|
| [quickstart](quickstart.rs) | `cargo run --example quickstart` | End-to-end: build a TimeSeries, fit a model, forecast, and evaluate |

## Forecasting Models

| Example | Run | Description |
|---------|-----|-------------|
| [arima](forecasting/arima.rs) | `cargo run --example arima` | ARIMA family: manual orders, auto-selection, seasonal ARIMA |
| [exponential](forecasting/exponential.rs) | `cargo run --example exponential` | ETS models: Simple, Holt, Holt-Winters, damped trends, AutoETS |
| [theta](forecasting/theta.rs) | `cargo run --example theta` | Theta method and Dynamic Theta with seasonal decomposition |
| [baseline](forecasting/baseline.rs) | `cargo run --example baseline` | Naive, SeasonalNaive, MeanForecaster baselines |
| [regression](forecasting/regression.rs) | `cargo run --example regression` | 11 regression backends: OLS, Ridge, ElasticNet, Quantile, WLS, RLS, BLS, NNLS, Poisson, Tweedie, DLM |
| [regression_exog_changepoints](forecasting/regression_exog_changepoints.rs) | `cargo run --example regression_exog_changepoints` | Exogenous regressors, changepoint features, CV with exog, structural break detection |
| [intermittent](forecasting/intermittent.rs) | `cargo run --example intermittent` | Croston, SBA, and TSB for sparse/intermittent demand |
| [ensemble](forecasting/ensemble.rs) | `cargo run --example ensemble` | Model ensembling with equal and optimized weights |
| [batch](forecasting/batch.rs) | `cargo run --example batch` | Batch forecasting of multiple series (sequential and parallel) |
| [hierarchy](forecasting/hierarchy.rs) | `cargo run --example hierarchy` | Hierarchical reconciliation: BottomUp, TopDown, MiddleOut, MinTraceOls |
| [var](forecasting/var.rs) | `cargo run --example var` | Vector Autoregression for multivariate series with Granger causality |
| [kalman](forecasting/kalman.rs) | `cargo run --example kalman` | Kalman filter: local level, local linear trend, state-space models |
| [constraints](forecasting/constraints.rs) | `cargo run --example constraints` | Post-hoc forecast constraints: non-negative, clamped, rounded, integer |
| [explainability](forecasting/explainability.rs) | `cargo run --example explainability` | Forecast decomposition into level, trend, and seasonal components |

## Analysis

| Example | Run | Description |
|---------|-----|-------------|
| [stl_decomposition](analysis/stl_decomposition.rs) | `cargo run --example stl_decomposition` | STL decomposition into trend, seasonal, and remainder |
| [changepoint](analysis/changepoint.rs) | `cargo run --example changepoint` | PELT changepoint detection for structural breaks |
| [changepoint_types](analysis/changepoint_types.rs) | `cargo run --example changepoint_types` | Changepoint type classification: level shift, trend change, variance change |
| [outlier_detection](analysis/outlier_detection.rs) | `cargo run --example outlier_detection` | Statistical outlier detection methods |
| [imputation](analysis/imputation.rs) | `cargo run --example imputation` | Missing value imputation strategies |

## Time Series Features

| Example | Run | Description |
|---------|-----|-------------|
| [basic_features](features/basic_features.rs) | `cargo run --example basic_features` | Summary statistics, trend strength, seasonal strength |
| [distribution](features/distribution.rs) | `cargo run --example distribution` | Distribution fitting and statistical tests |
| [autocorrelation](features/autocorrelation.rs) | `cargo run --example autocorrelation` | ACF, PACF, and Ljung-Box test |
| [entropy](features/entropy.rs) | `cargo run --example entropy` | Entropy-based complexity measures |
| [complexity](features/complexity.rs) | `cargo run --example complexity` | Time series complexity features |

## Transforms

| Example | Run | Description |
|---------|-----|-------------|
| [scaling](transform/scaling.rs) | `cargo run --example scaling` | Min-max, standard, and robust scaling |
| [boxcox](transform/boxcox.rs) | `cargo run --example boxcox` | Box-Cox and log transformations |
| [window](transform/window.rs) | `cargo run --example window` | Rolling window aggregations |
| [temporal_aggregation](transform/temporal_aggregation.rs) | `cargo run --example temporal_aggregation` | Temporal aggregation, downsampling, and upsampling |

## Validation

| Example | Run | Description |
|---------|-----|-------------|
| [metrics](validation/metrics.rs) | `cargo run --example metrics` | Forecast accuracy metrics: MAE, RMSE, MAPE, SMAPE, MASE |
| [cross_validation](validation/cross_validation.rs) | `cargo run --example cross_validation` | Time series cross-validation with expanding and sliding windows |
| [diagnostics](validation/diagnostics.rs) | `cargo run --example diagnostics` | Residual diagnostics and model checking |
| [bootstrap](validation/bootstrap.rs) | `cargo run --example bootstrap` | Bootstrap confidence intervals |
| [forecast_export](validation/forecast_export.rs) | `cargo run --example forecast_export` | Export forecasts to structured formats |
| [feature_export](validation/feature_export.rs) | `cargo run --example feature_export` | Export computed features |
| [aid](validation/aid.rs) | `cargo run --example aid` | AID demand classification: Regular, Intermittent, Lumpy, Erratic |
| [serialization](validation/serialization.rs) | `cargo run --example serialization --features serde` | JSON and bincode model/data persistence |

## Postprocessing

Requires the `postprocess` feature (enabled by default).

| Example | Run | Description |
|---------|-----|-------------|
| [postprocess_quickstart](postprocess/quickstart.rs) | `cargo run --example postprocess_quickstart` | Postprocessing quickstart: prediction intervals from raw forecasts |
| [postprocess_conformal](postprocess/conformal.rs) | `cargo run --example postprocess_conformal` | Conformal prediction intervals |
| [postprocess_quantile_methods](postprocess/quantile_methods.rs) | `cargo run --example postprocess_quantile_methods` | Quantile regression and interval estimation methods |
| [postprocess_qra_ensemble](postprocess/qra_ensemble.rs) | `cargo run --example postprocess_qra_ensemble` | Quantile Regression Averaging for ensemble intervals |
| [postprocess_conformalize](postprocess/conformalize.rs) | `cargo run --example postprocess_conformalize` | Conformalized quantile regression |
| [postprocess_unified_api](postprocess/unified_api.rs) | `cargo run --example postprocess_unified_api` | Unified postprocessing API |
| [postprocess_backtest](postprocess/backtest.rs) | `cargo run --example postprocess_backtest` | Backtesting with postprocessed intervals |

## Orchestration & Pipelines

| Example | Run | Description |
|---------|-----|-------------|
| [orchestration](orchestration.rs) | `cargo run --example orchestration` | Full pipeline: profiling, model selection, evaluation, reporting |
| [trend_components](trend_components.rs) | `cargo run --example trend_components` | Trend integration: composable trend and seasonal components |
| [recency_sensitivity](recency_sensitivity.rs) | `cargo run --example recency_sensitivity` | Recency-aware model weighting and sensitivity analysis |
