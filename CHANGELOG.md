# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-03-26

### Added

- **WASM/JS Parity** — major expansion of JS/WASM bindings to match Rust core:
  - `FeatureGenerator` with cyclical sin/cos, binary indicators, advanced calendar features
  - `generateFutureTimestamps(lastMs, frequency, horizon)` standalone function
  - `TimeSeries.futureTimestamps(horizon)` auto-inferred
  - `JsBootstrapPredictor` with `predictIntervals()` and `predictQuantiles()`
  - `JsConformalPredictor.predictQuantiles()` for multi-quantile conformal forecasts
  - `JsConformalPredictor.fitPerStep()` → `JsPerStepConformalResult`
  - `KalmanForecaster.custom(F, H, Q, R)` for user-defined state-space models
  - `KalmanForecaster.logLikelihood(series)` for model comparison
  - `KalmanForecaster.nState` / `nObs` getters

### Fixed

- `KalmanForecaster.smooth()` was hardcoding `local_level(1.0, 0.1)` instead of using the user-configured state-space model

## [0.4.9] - 2026-03-26

### Added

- `BootstrapPredictor::predict_quantiles()` — multi-quantile forecasts from bootstrap simulation paths (e.g., 10th/25th/50th/75th/90th percentiles)
- `ConformalPredictor::predict_quantiles()` — multi-quantile forecasts from conformal nonconformity scores (symmetric, median = point forecast)

### Fixed

- `RegressionForecaster`: exogenous regressors were silently dropped when differencing was active — now preserved via `TrimOnly` policy (regressors trimmed to match differenced length)

## [0.4.8] - 2026-03-24

### Added

- **Per-Horizon-Step Conformal Prediction** (`postprocess` module)
  - `ConformalPredictor::fit_per_step()` — separate interval widths per forecast step
  - `PerStepConformalResult` with `half_widths()`, `predict()`, `predict_intervals()`
  - Works with Split, CrossVal, and Jackknife+ methods; falls back to pooled quantile when < 2 residuals per step

- **Bootstrap Prediction Intervals** (`postprocess` module)
  - `BootstrapPredictor` — model-agnostic residual resampling with cumulative error path simulation
  - IID and block bootstrap variants
  - Intervals grow with horizon (uncertainty accumulates)

- **Calendar Feature Engineering** (`features` module)
  - `TimeComponent` cyclical sin/cos encoding: Month, Quarter, DayOfWeek, Hour, DayOfYear, etc.
  - `BinaryIndicator`: MonthStart, MonthEnd, QuarterStart, QuarterEnd, YearStart, YearEnd, Weekend
  - `AdvancedFeature`: LeapYear, DaysInMonth

- **Future Timestamp Generation** (`core` module)
  - `generate_future_timestamps()` — calendar-aware (Jan 31 + 1mo = Feb 28/29, not Mar 2)
  - `TimeSeries::future_timestamps(horizon)` — auto-infers frequency from data
  - Preserves original day across monthly/yearly stepping (no month-end drift)

- **Auto-Lag Selection for RegressionForecaster** (`models::regression`)
  - `.auto_lags(max_lag)` — select best lag order by BIC
  - `.auto_lags_with(max_lag, LagSelectionCriterion::Aic)` — select by AIC
  - `LagSelectionCriterion` enum: `Bic`, `Aic`
  - Re-selects on each `fit()` call

- **Differencing for RegressionForecaster** (`models::regression`)
  - `.differencing(d)` — regular differencing, auto-integrates after predict
  - `.seasonal_differencing(D, period)` — chainable for multi-seasonal series
  - `seasonal_integrate()` in `arima/diff.rs` — inverse of `seasonal_difference()`

- **CvFoldGenerator Rework** (`utils::cross_validation`)
  - Backward-anchored fold placement (last fold always covers series end)
  - `n_folds` driven — caps how many folds to take
  - `min_initial_window` as constraint (not driver) — drops early folds with insufficient training
  - `step_size` configurable (default = horizon for non-overlapping test sets)
  - `ConstraintViolation::Error` (default) or `ReduceFolds`

- **ModelSpec Flexibility** (`models::traits`)
  - `ModelSpec.name` changed from `&'static str` to `String` (accepts `impl Into<String>`)
  - `model_type` field for grouping multi-variant models
  - `ModelSpec::with_type()` constructor, `ModelRegistry::by_type()` query

- **Period Validation** — seasonal models reject `period < 2` with `InvalidParameter` error (SeasonalNaive, HoltWinters, SeasonalES, SeasonalWindowAverage, MSTLForecaster, TBATS, DummySeasonality)

### Changed

- **AutoForecast**: removed `SelectionStrategy::InSampleMSE` — always uses cross-validation
- **CvFoldGenerator**: `initial_window` renamed to `min_initial_window`
- Consolidated 4 duplicate `compute_median()` → shared `utils::stats::median()`
- Consolidated 3 duplicate aggregate helpers → shared `utils::stats::aggregate()`
- Removed dead `utils::stats::autocorrelation()` (0 callers)
- Removed batch processing module (`fit_many`, `predict_many`, `fit_predict_many`, `fit_registry`)

### Fixed

- `generate_future_timestamps`: month-end drift (Jan 31→Feb 28→Mar 28) fixed to preserve original day (Jan 31→Feb 28→Mar 31)
- `BootstrapPredictor`: intervals now grow with horizon via cumulative error paths (was flat)
- `auto_lags`: config preserved across re-fit calls (was destroyed after first fit)
- `TimeSeries::future_timestamps`: monthly data now uses `Frequency::Months(1)` instead of `Duration::days(30)`

## [0.4.7] - 2026-03-19

### Added

- **Recency-Aware Trend Components** (`seasonality` module)
  - `Recency` enum — fit on recent data only (`Window(N)`, `Fraction(0.3)`, `Full`, `Auto`) for trend-aware forecasting
  - `Recency::Auto` — PELT-based changepoint detection to automatically determine fitting window
  - `PolynomialTrend` — polynomial trend (degree 1-3) with Vandermonde + Cholesky solve, recency-aware
  - `ExponentialTrend` — log-linear exponential growth/decay trend estimation, recency-aware
  - `LogisticTrend` — logistic S-curve fitting with auto or fixed capacity, recency-aware
  - `TheilSenTrend` — robust Theil-Sen median-of-pairwise-slopes estimator, recency-aware, subsampled for large data
  - `AutoTrend` — automatic selection of best trend component via AICc/BIC/holdout (enum dispatch)
  - `AutoSeasonal` — automatic selection of best seasonal component via AICc/BIC
  - `n_params()` method added to `TrendComponent` and `SeasonalComponent` traits (for IC-based selection)
  - `with_recency()` builder methods on `PiecewiseLinearTrend` and `HodrickPrescottFilter`
  - `trend_components` example demonstrating all components and AutoTrend selection

- **Changepoint-Aware Baseline Models**
  - `SMA::with_changepoint()` — constrains window to post-changepoint data
  - `RandomWalkWithDrift::with_changepoint()` — constrains drift estimation to post-changepoint data

- **Regression Forecaster** (`models::regression` module, requires `postprocess` feature)
  - `RegressionForecaster` — wraps `anofox-regression` backends behind the `Forecaster` trait
  - `RegressionBackend` enum — 11 backends: OLS (default), Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson, BLS, NNLS, Dynamic
  - `RegressionFeatures` — configurable feature engineering: trend types, seasonal specs, AR lags, structural features, exogenous regressors
  - Convenience constructors: `linear_trend()`, `ar(lags)`, `trend_ar(lags)`, `ols(features)`, `ridge()`, `elastic_net()`, `quantile()`, `wls_decay()`, `wls()`, `rls()`, `tweedie()`, `poisson()`, `bls()`, `nnls()`, `dynamic()`, `dynamic_smoothed()`
  - `WeightStrategy` enum for WLS — `ExponentialDecay(f64)` or `Custom(Vec<f64>)`
  - Full `Forecaster` trait implementation: `fit`, `predict`, `predict_with_exog`, `fitted_values`, `residuals`, `supports_exog`
  - Recursive multi-step prediction for AR models (feeds predictions back as lag features)
  - Access to fitted statistics (R², coefficients) via `r_squared()`, `fitted_result()`
  - Generalized fitted model storage via `Box<dyn FittedRegressor>` for backend-agnostic prediction
  - `InformationCriterion` re-exported from `anofox-regression` for Dynamic backend configuration
  - Compatible with `ModelRegistry`, pipelines, ensembles, and cross-validation

- **Feature Safety Classification** (`models::regression` module)
  - `FeatureSafety` enum — classifies features by cross-validation leakage risk: `Deterministic`, `DataDependent`, `Structural`, `External`
  - `classify_features()` method on `RegressionFeatures` — returns per-column safety classification
  - Trend types classified as `DataDependent`, Fourier terms as `Deterministic`, structural features as `Structural`

- **Structural Features** (`models::regression` module)
  - `StructuralFeature` trait — general interface for features that are forward-filled during prediction
  - `ChangepointFeature` — encodes detected changepoints as regression features
  - `ChangepointEncoding` enum — `StepFunctions` (k binary columns), `RegimeIndex` (1 ordinal column), `CumulativeCount` (1 additive column)
  - Forward-fill prediction: model continues in last known regime during forecast period
  - Builder methods: `with_structural()`, `with_changepoint_steps()`, `with_changepoints()`

- **Binned Conformal Prediction** (`postprocess` module)
  - `BinnedConformalPredictor` — heteroscedastic prediction intervals that bin calibration residuals by predicted magnitude
  - Quantile-based bin edges with per-bin conformal quantiles — wider intervals where forecast uncertainty is larger
  - Graceful fallback: bins with < 3 residuals merge with neighbors; insufficient data falls back to single global quantile
  - `BinnedConformalResult` — bin edges, per-bin quantiles, global fallback quantile, coverage level

- **Pre-Regression MSTL Decomposition** (`seasonality` + `models` modules)
  - `MSTL::decompose_with_regressors()` — regress out exogenous effects (OLS: y ~ X) before STL decomposition, preventing regressors correlated with trend/seasonality from distorting the decomposition
  - `MSTLResult` now stores `regressor_coefficients` (OLS result) and `regressor_effect` (X*β) for reconstruction
  - `MSTLForecaster` automatically uses pre-regression when `TimeSeries` has calendar regressors
  - Full `Forecaster` exog support: `supports_exog()`, `has_exog()`, `exog_names()`, `predict_with_exog()`
  - `predict()` guards against missing exog when model was fit with regressors

- **Forecasting Metrics** (`utils::metrics` module)
  - `rmsse()` — Root Mean Squared Scaled Error, scaled by in-sample naive forecast MSE (per-series building block of WRMSSE)
  - `wrmsse()` — Weighted RMSSE, the M5 competition primary metric for aggregating across series by importance weights

- **Exogenous Coefficient Extraction** (`models::traits`)
  - `Forecaster::exog_coefficients()` — inspect OLS pre-regression coefficients (intercept, betas, regressor names) on any fitted model with exogenous regressors
  - Implemented across all exog-supporting models: ARIMA, SARIMA, AutoARIMA, ETS, AutoETS, Theta, AutoTheta, OptimizedTheta, DynamicTheta, Naive, MFLES, MSTL, Pipeline

- **Transform Pipeline** (`transform::pipeline` module)
  - `Pipeline` — composable transform chains around any `Forecaster`, itself implements `Forecaster`
  - `PipelineBuilder` — fluent API: `Pipeline::builder().transform(BoxCoxTransform::auto()).transform(DifferenceTransform::new(1)).model(Box::new(Naive::new())).build()`
  - `Transform` trait with concrete implementations: `DifferenceTransform`, `SeasonalDifferenceTransform`, `BoxCoxTransform`, `ScaleTransform`, `LogTransform`

- **Deterministic Feature Generator** (`features::generator` module)
  - `FeatureGenerator` — generate regressors from timestamps: `fourier()`, `day_of_week()`, `month_of_year()`, `quarter()`, `holiday()`
  - `add_to()` — attach generated features to `TimeSeries` as named regressors

- **Composable Seasonality & Trend Components** (`seasonality` module)
  - `SeasonalComponent` / `TrendComponent` traits — dual-purpose: standalone (fit/predict) and feature extraction
  - `DummySeasonality` — one-hot (dummy variable) seasonal encoding for arbitrary seasonal shapes
  - `SeasonalDifference` — standalone seasonal differencing transform with inverse, strength, and variance-reduction features
  - `HodrickPrescottFilter` — smooth trend extraction with cycle decomposition (quarterly/monthly/annual presets)
  - `PiecewiseLinearTrend` — PELT-based changepoint detection with per-segment linear regression
  - Standalone feature functions: `dummy_seasonal_strength`, `dummy_seasonal_amplitude`, `seasonal_diff_strength`, `seasonal_diff_variance_reduction`, `hp_trend_strength`, `hp_cycle_variance_ratio`, `piecewise_n_segments`, `piecewise_trend_features`

- **Comprehensive Examples** (45 examples with `.md` documentation)
  - 10 new examples: `regression` (11 backends), `hierarchy`, `var`, `kalman`, `constraints`, `explainability`, `aid`, `bootstrap`, `temporal_aggregation`, `serialization`
  - `.md` companion documentation for all examples with section descriptions, key types, and run commands
  - `examples/README.md` — categorized index of all examples

### Removed

- **Orchestration module extracted to [`anofox-orchestration`](https://github.com/sipemu/anofox-orchestration)** (private repo)
  - Moved: `DataProfile`, `PipelineBuilder`, `Pipeline`, `PipelineConfig`, `PipelineResult`, `PipelineReport`, `PipelineStore`, `DecisionLog`, `FallbackChain`, `HorizonAnalysis`, `SelectionConfidence`, `ModelConfidenceSet`, `QualityFloor`, `MetricStrategy`, `EnsembleMode`, `PreprocessMode`, `TrendIntegration`, `ChangepointMode`, `DriftConfig`/`DriftReport`, `ExploreBuilder`, `BacktestBuilder`, structured tool functions
  - Moved: `partition_by_structural_break()` helper
  - Moved examples: `orchestration`, `explore`, `regression_exog_changepoints`
  - `anofox-forecast` remains the open-source foundation; `anofox-orchestration` depends on it via git

- **Batch processing module removed** (`models::batch`)
  - Removed `fit_many()`, `predict_many()`, `fit_predict_many()`, `fit_registry()` — batch multi-series operations belong in the orchestration layer
  - `ModelRegistry` and `ModelSpec` remain (used by `compare_models`, `compare_registry`, cross-validation)

### Internal

- Consolidated 4 duplicate `compute_median()` implementations → shared `utils::stats::median()`
- Consolidated 3 duplicate aggregate helpers → shared `utils::stats::aggregate()`
- Removed dead `utils::stats::autocorrelation()` (0 callers)

### Changed

- **ETS/AutoETS Performance Optimizations** (accuracy-preserving, all 2383 tests pass)
  - Eliminate redundant `initialize_state()` calls in non-seasonal Nelder-Mead closures — pass pre-computed heuristic states instead of recomputing O(n) per evaluation
  - Two-stage seasonal optimization: replace 6× expensive joint optimizations (16-18 dim) with 6× cheap smoothing-only explorations (2-4 dim, 500 iter) + 1× joint refinement from winner
  - AutoETS candidate pruning: require 3× period (was 2×) for seasonal models; ANOVA F-test skips seasonal candidates when no seasonal signal detected (F < 1.0)
  - Multi-start early-exit: skip starting points whose initial likelihood exceeds 3× the best found so far
  - `NelderMeadConfig` derives `Copy` (all fields are `Copy` types), eliminating unnecessary `.clone()` calls across ETS, seasonal ES, GARCH, and Theta models

### Fixed

- ETS: replaced unsafe `unwrap()` calls with heuristic fallbacks for edge cases (constant data, zero variance)
- `cross_validate_all` / `fit_all_and_compare`: reduced cognitive complexity with early returns and helper closures
- `TimeSeries::missing_mask()`: fixed panic on empty-dimension series
- Conformal prediction: added SAFETY comments documenting invariants for `expect()` calls
- `RegressionForecaster::predict_with_intervals`: fixed missing `components` argument in `build_future_matrix` call
- `TimeSeries::slice()`: fixed calendar regressors not being sliced — caused regressor/value misalignment during cross-validation with exogenous features
- `cross_validate`: fixed `evaluate_fold` to call `predict_with_exog()` when model has exogenous regressors (previously always called `predict()`, which errors on exog models)

## [0.4.6] - 2026-03-14

### Added

- **Orchestration Module** (`orchestration` module)
  - `DataProfile` — automated data profiling (stationarity, trend, seasonality, quality score, ACF statistics)
  - `PipelineBuilder` — declarative pipeline construction with model selection, cross-validation, fallback chains, and forecast constraints
  - `Pipeline` / `PipelineConfig` — end-to-end pipeline execution and replay from saved configuration
  - `PipelineResult` — forecast with full diagnostics (profile, decision log, confidence metrics, horizon analysis)
  - `DecisionLog` — structured audit trail of pipeline decisions with categories, outcomes, and timing
  - `FallbackChain` — ordered model failover with automatic recovery
  - `HorizonAnalysis` — per-step-ahead error decomposition (RMSE, MAE, bias per horizon step)
  - `ExecutionMetadata` / `ExecutionTimer` — fit/predict timing and convergence tracking
  - `SelectionConfidence` — Diebold-Mariano pairwise forecast accuracy test (via `anofox-statistics`)
  - `ModelConfidenceSet` — Hansen-Lunde-Nason (2011) model confidence set procedure (bootstrap-based)
  - `QualityFloor` — Hansen (2005) Superior Predictive Ability test vs benchmark model
  - Orchestration prelude (`orchestration::prelude::*`) for convenient imports
  - `PreprocessMode` — automatic preprocessing pipeline (Box-Cox for skewed data, outlier replacement for low-quality data)
  - `PreprocessSteps` — explicit control: `boxcox`, `outlier_treatment`, `outlier_window`
  - `MetricStrategy` — data-aware multi-metric model selection
    - `Auto`: intermittent → MAE+SMAPE, non-negative → MAE+WAPE, general → MAE+SMAPE+MDA
    - `Single(Metric)` / `Composite(Vec<(Metric, f64)>)` for custom weighting
    - MDA (higher-is-better) automatically inverted in composite scores
  - `EnsembleMode` — configurable ensemble construction
    - `Auto`: ensemble models from MCS when > 1 model included
    - `Fixed(CombinationMethod)`: always ensemble with specified method
    - `None`: single best model (default)
  - `PipelineReport` — multi-section structured report from `PipelineResult`
    - Sections: Summary, Data Profile, Preprocessing, Model Selection, Ensemble, Forecast, Horizon Analysis, Decision Log, Execution Metadata
    - `Display` impl with column-aligned table formatting
  - `PipelineStore` trait — abstract storage backend decoupled from serde
    - `Value` IR enum (Null, Bool, Int, Float, String, List, Map) for backend-agnostic serialization
    - `Storable` trait for converting orchestration types to/from `Value`
    - `InMemoryStore` — thread-safe in-memory implementation for testing
    - `RecordKind`: Profile, Config, Result, DecisionLog, HorizonAnalysis, Report
  - Structured tool functions for MCP / agent integration (`orchestration::tools`)
    - `profile_data()` — profile a time series with typed I/O
    - `select_models()` — heuristic model recommendation from data profile
    - `run_pipeline()` — end-to-end pipeline execution
    - `explain_result()` — human-readable explanation (Brief/Normal/Detailed)
  - `PipelineBuilder::preprocess()`, `.metric()`, `.ensemble()` builder methods

- **WASM/npm Orchestration Bindings**
  - `JsDataProfile` — full data profiling with 30+ property getters, `toJSON()`, `fromSeries()`, `fromValues()`
  - `JsPipelineBuilder` — fluent builder: `profile()`, `preprocess()`, `metric()`, `ensemble()`, `addModel()`, `execute()`
  - `JsPipelineResult` — forecast, model name, decision log, quality floor, MCS, ensemble weights, metric scores
  - `JsPipelineReport` — `title`, `sectionCount`, `toString()`, `toJSON()` with typed sections
  - `selectModels(profile, availableModels?)` — model recommendation tool function
  - `explainResult(result, verbosity)` — human-readable explanation (brief/normal/detailed)

- **Forecasting Metrics**
  - `bias()` — signed forecast bias (mean error)
  - `periods_in_stock()` — Periods-in-Stock metric for inventory forecasting

- **Automatic Identification of Demand (AID)**
  - `AidAnalyzer` — wraps `anofox-regression`'s AID classifier with `&[f64]`/`TimeSeries` input
  - Summary statistics: demand type (Regular/Intermittent), best-fitting distribution (Poisson, NegBin, Normal, Gamma, LogNormal, etc.), fitted parameters, zero proportion
  - Per-observation anomaly features: `AidFeatures` with `Vec<AidAnomalyLabel>` (Stockout, NewProduct, ObsoleteProduct, HighOutlier, LowOutlier) matching input length
  - `AidResult::summary()` for aggregate statistics, `AidResult::features()` for per-observation labels
  - Builder API: `AidAnalyzer::new().intermittent_threshold(0.3).detect_anomalies(true).analyze(&data)`
  - Gated behind `postprocess` feature (enabled by default)

## [0.4.5] - 2026-03-13

### Added

- **Parallel Processing & AutoForecast**
  - `compare_models()` / `compare_registry()` — parallel model comparison
  - `AutoForecast` candidate model fits run in parallel when `parallel` feature enabled
  - Bootstrap sampling uses `par_iter` when `parallel` enabled

- **Streaming Cross-Validation**
  - `StreamingCVAggregator` — online metric aggregation using Welford's algorithm
  - `cross_validate_early_stop()` — CV with convergence-based early stopping
  - Eliminates need to store all fold results for aggregation

- **Builder Patterns**
  - `Pelt::new(CostFunction::L2).min_size(5).penalty(5.0).detect(&data)` — PELT builder
  - `StlBuilder::new(period).seasonal_window(7).robust(true).decompose(&data)` — STL builder

- **SIMD Correlation & Autocorrelation**
  - `simd::correlation()` — SIMD-accelerated Pearson correlation
  - `simd::autocorrelation()` — SIMD-accelerated autocorrelation at a given lag

- **Binary Serialization** (requires `serde` feature)
  - `to_bincode()` / `from_bincode()` — compact binary serialization via bincode
  - `save_to_bincode()` / `load_from_bincode()` — file persistence

- **Convenience Methods**
  - `Forecaster::fit_predict()` — fit and predict in a single call
  - `Forecaster::fit_predict_with_intervals()` — fit and predict with confidence intervals

- **Specific Error Variants**
  - `ForecastError::ConvergenceFailure` — optimizer/model convergence failures
  - `ForecastError::SingularMatrix` — linear algebra singularity errors
  - `ForecastError::SerializationError` — serialization/deserialization errors

- **WASM/npm Enhancements**
  - `CalendarAnnotations` — holidays, named regressors, and JSON serialization in JS/TS
  - `TimeSeries.setCalendar()` / `hasCalendar()` / `clearCalendar()` — calendar integration
  - Complete TypeScript type definitions (`types.d.ts`) for all 35 exported classes and 21 functions
  - `package.json` with `"types"` field for TypeScript support

- **CI/CD Improvements**
  - `cargo audit` step for security advisory checks
  - `cargo deny` step for license and supply-chain compliance
  - `deny.toml` configuration for allowed licenses

- **Benchmarks**
  - `ensemble_benchmark` — AutoEnsemble fit/predict benchmarks
  - `cv_benchmark` — cross-validation benchmarks with varying folds/horizons/models

- **Documentation**
  - [Model Selection Guide](docs/model_selection_guide.md) — decision flowchart, model families, common patterns

- **Test Coverage**
  - Postprocess tests: +42 IDR tests, +22 Normal tests, +316 lines backtest, +211 lines QRA
  - Streaming CV tests: 8 new tests for aggregator and early stopping
  - SIMD tests: 18 new tests for correlation and autocorrelation

- **Missing Value Imputation Toolbox**
  - `MissingValuePolicy::BackwardFill` — next-observation-carried-backward
  - `MissingValuePolicy::FillMean` — fill with mean of finite values
  - `MissingValuePolicy::FillMedian` — fill with median of finite values
  - `MissingValuePolicy::Interpolate` — linear interpolation via policy enum
  - `TimeSeries::missing_mask()` — boolean mask of NaN/Inf positions (primary dimension)
  - `TimeSeries::missing_count()` — per-dimension count of missing values
  - `TimeSeries::imputed_forward_backward()` — forward-fill then backward-fill (handles leading + trailing NaN)
  - `TimeSeries::imputed_moving_average(window)` — centered moving average imputation with multi-pass for adjacent gaps
  - `TimeSeries::imputed_seasonal(period)` — seasonal median imputation using same-position values across cycles
  - `TimeSeries::with_imputed_regressors(policy)` — apply imputation policy to regressor vectors independently
  - `nan_mean()` / `nan_median()` — NaN-safe statistics helpers in `utils::stats`

- **OLS NaN/Inf Validation**
  - `ols_fit()` now validates `y` for NaN/Inf (returns `ForecastError::MissingValues`)
  - `ols_fit()` now validates each regressor for NaN/Inf (returns `ForecastError::InvalidParameter`)

- **Hierarchical Forecasting** (`hierarchy` module)
  - `HierarchyTree` — define parent→children structure for grouped forecasts
  - `ReconciliationMethod::BottomUp` — aggregate leaf forecasts upward
  - `ReconciliationMethod::TopDown` — disaggregate top-level using historical proportions
  - `ReconciliationMethod::MinTraceOls` — optimal combination via MinT OLS projection
  - 15 tests covering tree construction, all methods, coherence, and error handling

- **Prophet-style Fourier Seasonality** (`seasonality::fourier` module)
  - `FourierSeasonality` — flexible seasonal modeling using Fourier basis functions
  - `fourier_terms()` — generate sin/cos basis vectors for arbitrary period and order
  - Preset constructors: `daily()`, `weekly()`, `yearly()`
  - Fit via normal equations with Cholesky decomposition (no external dependencies)
  - 25 tests covering recovery, orthogonality, periodicity, edge cases

- **Core Type Improvements**
  - `Serialize`/`Deserialize` for `Forecast`, `TimeSeries`, `CalendarAnnotations` (behind `serde` feature)
  - `Display` impls for `Forecast` and `TimeSeries` with preview summaries
  - `PartialEq` for `Forecast` (epsilon-based) and `CalendarAnnotations`

- **WASM Model Parity**
  - Added JS/WASM bindings: `HoltWintersForecaster`, `SESForecaster`, `CrostonForecaster`, `ADIDAForecaster`, `IMAPAForecaster`, `TSBForecaster`, `GARCHForecaster`
  - TypeScript definitions for all new forecaster classes

- **Validation & Stationarity Tests**
  - 14 new residual diagnostic tests (edge cases, NaN, constant, short series)
  - 9 new stationarity test cases (ADF, KPSS, edge cases)

- **Intermittent Model & GARCH Edge Case Tests**
  - Croston: 6 new tests (all zeros, single demand, negative values, very long gaps, single observation)
  - ADIDA: 6 new tests (same pattern)
  - IMAPA: 6 new tests (same pattern)
  - TSB: 5 new tests (same pattern)
  - GARCH: 10 new tests (constant data, extreme volatility, trending data, NaN handling, short series)

- **VAR (Vector Autoregression)** (`models::var` module)
  - `VAR::new(order)` — VAR(p) model for multivariate time series
  - Equation-by-equation OLS estimation
  - Multi-step forecasting across all variables
  - `granger_causality_test(cause, effect)` — F-statistic for Granger causality
  - 18 tests covering coefficient recovery, dimensions, edge cases

- **Kalman Filter Framework** (`models::kalman` module)
  - `KalmanFilter` — forward filtering and Rauch-Tung-Striebel smoothing
  - `StateSpaceModel` — linear Gaussian state-space specification
  - Convenience constructors: `local_level()`, `local_linear_trend()`
  - `filter()`, `smooth()`, `predict()`, `log_likelihood()` methods
  - Internal dense matrix algebra (no external dependencies)
  - 14 tests covering filtering, smoothing, prediction, edge cases

- **Builder Patterns for Models**
  - `GARCH::builder().p(1).q(1).max_iterations(500).build()`
  - `MFLES::builder().seasonal_period(12).num_rounds(5).learning_rate(0.1).build()`
  - `AutoForecast::builder().seasonal_period(12).include_arima(true).build()`

- **Rolling/Expanding Window Forecast**
  - `rolling_forecast()` — walk-forward evaluation with per-window metrics
  - `RollingForecastConfig` — builder for initial train size, horizon, step size, expanding/rolling mode
  - `RollingForecastResult` — per-window predictions, actuals, and aggregated metrics
  - Parallel window evaluation when `parallel` feature enabled

- **Ensemble Prediction Interval Combination**
  - Widest-envelope interval combination: takes min of lower bounds, max of upper bounds
  - `predict_with_intervals()` now produces meaningful combined intervals

- **STL Buffer Caching**
  - `StlScratch` — pre-allocated scratch buffers for zero-allocation repeated decompositions
  - `STL::decompose_with_scratch()` — decompose with reusable buffers
  - `StlBuilder::decompose_reuse()` — amortized allocation across repeated calls

- **TimeSeries Convenience Methods**
  - `seasonal_strength(period)` — seasonal strength via STL decomposition (0 to 1)
  - `trend_strength(period)` — trend strength via STL decomposition (0 to 1)
  - `with_outliers_replaced(config, window)` — replace outliers with local median
  - `to_json()` / `from_json()` — JSON serialization (requires `serde` feature)
  - `Forecast::to_json()` / `Forecast::from_json()` — forecast serialization

- **WASM PostProcessor Bindings**
  - `JsConformalPredictor`, `JsNormalPredictor`, `JsHistoricalSimulator` — probabilistic intervals in JS
  - `JsPostProcessor` — unified API with `conformal()`, `normal()`, `historicalSim()`
  - `JsBacktestConfig` / `JsBacktestResult` — backtesting support
  - `JsPredictionIntervals` — coverage, widths, midpoints, empirical coverage
  - `MFLESForecaster.predictWithIntervals()` — added to WASM bindings
  - TypeScript definitions for all postprocess types

- **Persistence Module Tests**
  - 27 tests (from 6): JSON/bincode round-trips, file I/O, error cases, helper modules

- **Advanced Forecasting Metrics**
  - `wape()` — Weighted Absolute Percentage Error
  - `mda()` — Mean Directional Accuracy (direction-of-change)
  - `theils_u1()` / `theils_u2()` — Theil's U statistics (absolute and relative to naive)
  - `msis()` — Mean Scaled Interval Score for probabilistic forecast evaluation
  - `coverage()` — Empirical coverage rate for prediction intervals
  - `skill_score()` — Relative improvement over a baseline model
  - `ForecastMetrics::compute()` — all 10 metrics (MAE, MSE, RMSE, MAPE, SMAPE, MASE, WAPE, MDA, U1, U2) in one call

- **Model Warm-Starting**
  - `ETS::with_initial_states(spec, period, level, trend, seasonal_values)` — pre-set ETS states
  - `SES::with_alpha(alpha, level)` — pre-fitted SES (predict without fit)
  - `ARIMA::with_coefficients(p, d, q, ar, ma, intercept)` — pre-fitted ARIMA coefficients
  - `Theta::with_theta_value(theta, alpha, level, b)` — pre-fitted Theta state
  - `Forecaster::fitted_params()` — extract `FittedParams` from any fitted model

- **Forecast Constraints**
  - `ForecastConstraint` enum: `NonNegative`, `LowerBound`, `UpperBound`, `Bounds`, `IntegerRound`, `Custom`
  - `ConstrainedForecast::apply()` — apply constraints to point forecasts and intervals
  - `Forecast::non_negative()`, `.clamp(lo, hi)`, `.round_to_integer()` convenience methods

- **Forecast Combination Convenience**
  - `fit_all_and_compare()` — fit all models in registry, rank by holdout MAE/RMSE/MAPE
  - `cross_validate_all()` — cross-validate all registry models with aggregated metrics
  - `ensemble_best_k()` — auto-select top-k models by performance into an ensemble
  - `ModelComparison` / `CVComparison` with `Display` formatted tables

- **STL Convenience Functions** (`seasonality::convenience` module)
  - `deseasonalize()`, `detrend()`, `seasonal_component()`, `trend_component()`, `remainder_component()`
  - `recompose()` — reconstruct series from trend + seasonal + remainder
  - `seasonal_adjust()` — return new TimeSeries with seasonal component removed
  - `STLResult::deseasonalized()`, `.detrended()`, `.recompose()` methods

- **Intermittent Demand Diagnostics**
  - `IntermittentDiagnostics` — Syntetos-Boylan (2005) demand classification framework
  - `DemandClassification`: Smooth, Erratic, Intermittent, Lumpy (ADI/CV² thresholds)
  - ADI (Average Demand Interval), CV² of non-zero demands, zero fraction
  - `recommended_model()` — suggest Croston/TSB/SES based on classification
  - Coverage rate, bias, and Periods-in-Stock (PIS) metrics

- **Model Diagnostics Pipeline**
  - `ModelDiagnostics::from_residuals()` — Ljung-Box, Jarque-Bera, Breusch-Pagan tests
  - `ModelDiagnostics::from_forecaster()` — extract residuals and run all diagnostics
  - ACF/PACF of residuals, residual mean/std, `passes_all` flag

- **Forecast Explainability**
  - `ForecastExplanation` struct: level, trend, seasonal, residual, named components
  - `Explainable` trait implemented for ETS, Theta, MSTLForecaster
  - Components sum to forecast values

- **TimeSeries Temporal Aggregation**
  - `aggregate(period, method)` — Sum, Mean, Median, First, Last, Min, Max
  - `downsample(factor)` — decimation with timestamp preservation
  - `upsample(factor, method)` — Linear, ForwardFill, BackwardFill, Zero interpolation
  - `sliding_window_aggregate(window, step, method)` — configurable sliding windows

- **Hierarchy Reconciliation Methods**
  - `ReconciliationMethod::MiddleOut { middle_level }` — reconcile from a chosen depth
  - `ReconciliationMethod::MinTraceShrink` — MinT with Ledoit-Wolf shrinkage covariance

- **Ensemble Combination Methods**
  - `CombinationMethod::InverseAIC` — Akaike weights from estimated AIC
  - `CombinationMethod::Stacking { folds }` — non-negative constrained linear combiner
  - `CombinationMethod::HorizonAdaptive` — per-horizon weights from rolling-origin evaluation

- **Error Context**
  - `ForecastError::SubModelError` — wraps sub-model failures with model name context
  - `ForecastError::FitRequired` now carries optional model name

- **Forecaster Trait Adapters**
  - `VARForecaster` — adapts multivariate VAR for univariate Forecaster interface
  - `KalmanForecaster` — adapts Kalman filter with local_level/local_linear_trend constructors

- **CV Embargo**
  - `CvFoldGenerator::embargo(n)` — exclude observations after test sets (financial ML)
  - `CVConfig::with_embargo(n)` — embargo for config-based CV

- **WASM/JS Enhancements**
  - `AutoForecastBuilder` — fluent builder for AutoForecast in JS
  - `EnsembleForecaster.setInverseAic()` / `.setStacking()` / `.setHorizonAdaptive()` / `.setMethod(name)`
  - `Forecast.nonNegative()`, `.clamp()`, `.roundToInteger()` — constraint methods in JS
  - `JsModelDiagnostics.fromResiduals()` — diagnostics in JS with all property accessors
  - `VARForecaster`, `KalmanForecaster` — multivariate and state-space models in JS

- **Benchmarks**
  - STL scratch reuse comparison, MSTL multi-period
  - ARIMA/SARIMA fitting at multiple series lengths
  - Periodicity detection (autocorrelation, Welch periodogram)
  - Model comparison (Naive, SES, Theta, ETS, ARIMA)
  - Hot paths (SIMD ops, forecast construction, TimeSeries slicing)

- **Integration Tests**
  - VAR: 13 tests (coefficient recovery, Granger causality, forecast accuracy)
  - Persistence: 16 tests (JSON/bincode round-trips for all model types)
  - Pipeline: 12 tests (ensemble+constraints+postprocessing, STL+recompose, CV+select)

- **Prediction Interval Improvements**
  - RandomWalkWithDrift: proper drift SE + variance scaling formula
  - SMA/WindowAverage: (1 + 1/w) factor for mean estimation uncertainty

### Changed

- Kalman filter uses flat `DenseMatrix` layout with in-place operations and pre-allocated scratch buffers
- Ensemble supports InverseAIC, Stacking, and HorizonAdaptive combination methods

- `parallel` feature now covers AutoForecast, model comparison, bootstrap, cross-validation folds, and rolling forecast windows (previously only AutoARIMA)
- `serde` feature now includes bincode for binary serialization alongside JSON
- Several `ComputationError` uses migrated to specific error variants (`ConvergenceFailure`, `SingularMatrix`)
- Removed dead code in `mfles.rs` and `tbats/model.rs`
- MSTL uses in-place decomposition for reduced allocations
- STL decomposition supports buffer reuse via `StlScratch`
- ETS uses `Cow<[f64]>` to avoid cloning series values when no regressors are present
- Cross-validation uses direct slice references to avoid intermediate allocations per fold
- Ensemble `predict_with_intervals()` produces widest-envelope combined intervals
- Test coverage increased to 2,480+ tests
- `MissingValuePolicy` enum has 4 new variants (breaking for exhaustive `match` — acceptable under 0.x semver)

## [0.4.1] - 2026-01-16

### Added

- **ETS Notation Parser & FPP3 Taxonomy Compliance**
  - `ETSSpec::from_notation("AAA")` - Create ETS models from standard notation
  - `ETSSpec::is_valid()` - Validate model combinations before fitting
  - Reject unstable ETS combinations (MAA, MAdA) per [FPP3 taxonomy](https://otexts.com/fpp3/taxonomy.html)
  - New convenience constructors: `ana()`, `anm()`, `aada()`, `aadm()`, `mnm()`, `madm()`

- **WASM/npm Package Enhancements**
  - `ETSForecaster.fromNotation("AAA", period)` - Standard notation in JavaScript
  - `ETSForecaster.isValidSpec(error, trend, seasonal)` - Validation helper
  - Constructor validation rejects unstable ETS combinations
  - npm package documentation with ETS notation examples

- **Comprehensive Test Coverage**
  - 76 WASM tests covering all 29 forecaster classes
  - 23 JavaScript integration tests (Node.js)
  - Edge case tests: NaN handling, single data point, negative values, large/small values
  - ETS notation parsing tests (valid, invalid, unstable combinations)

- **CI/CD Improvements**
  - JavaScript integration tests in CI workflow
  - npm OIDC trusted publishing (no tokens required)
  - Requires npm >= 11.5.1 for OIDC support

- **Documentation**
  - "Use Cases" section in README with DuckDB extension and npm package links
  - Updated npm README with ETS notation API documentation
  - FPP3 taxonomy reference in API docs

### Changed

- Test coverage increased to 1,400+ tests (unit + integration + WASM + JS)
- Installation instructions updated to v0.4

## [0.4.0] - 2026-01-12

### Added

- **Probabilistic Forecasting Module** (`postprocess` feature, enabled by default)
  - `PostProcessor` - Unified API for probabilistic forecast calibration
  - `PredictionIntervals` - Multi-level interval representation with coverage guarantees
  - **Conformal Prediction** - Distribution-free prediction intervals
  - **Conformalized Quantile Regression** - Calibrated quantile forecasts
  - **Quantile Regression Averaging (QRA)** - Ensemble-based probabilistic forecasts
  - **Historical Simulation** - Bootstrap-based uncertainty estimation
  - **Normal Approximation** - Parametric prediction intervals
  - **Isotonic Distributional Regression (IDR)** - Non-parametric calibration
  - **Backtesting** - Horizon-aware backtesting with automatic calibration
    - `BacktestConfig` with expanding/rolling windows
    - Per-horizon calibration for improved accuracy
    - Coverage and calibration error metrics

- **Cross-Validation Enhancements**
  - `CvFoldGenerator` - Standalone fold generation for custom workflows
  - `gap` parameter - Prevents data leakage from lagged features
  - `purge` parameter - Removes observations to prevent lookahead bias (financial applications)
  - `FillStrategy` trait - Handle unknown future features during CV
    - Implementations: `LastValueFill`, `MeanFill`, `MedianFill`, `ZeroFill`, `ConstantFill`, `ModeFill`
  - `train_test_split()` - Simple ratio or index-based splitting
  - `train_test_split_at()` - Split at specific index
  - `grouped_cross_validate()` - Multi-series CV with consistent fold boundaries
  - `GroupedCVResults` - Per-group and aggregated metrics
  - `Fold` struct - Explicit train/test index representation

- **Architecture Documentation**
  - ADR explaining CV split design decisions
  - Documents why CV split lives in DuckDB extension vs this crate
  - Component distribution rationale (fold generation, orchestration, etc.)

- **New Examples**
  - `postprocess/quickstart.rs` - Getting started with probabilistic forecasts
  - `postprocess/conformal.rs` - Conformal prediction intervals
  - `postprocess/conformalize.rs` - Conformalized quantile regression
  - `postprocess/qra_ensemble.rs` - QRA ensemble methods
  - `postprocess/quantile_methods.rs` - Various quantile approaches
  - `postprocess/unified_api.rs` - PostProcessor unified API
  - `postprocess/backtest.rs` - Backtesting workflows

### Changed

- Cross-validation now returns `folds` field in `CVResults` for transparency
- Test coverage increased to 1,316+ tests

### Dependencies

- Added `faer` (optional) for linear algebra in postprocessing
- Added `anofox-regression` v0.5.0 (optional) for quantile regression

## [0.3.2] - 2026-01-07

### Fixed

- **Stable Rust Compatibility**
  - Replaced unstable `is_multiple_of()` method with `% 2 == 0`
  - Fixes WASM builds on stable Rust 1.86.0+

## [0.3.1] - 2026-01-07

### Added

- **WASM Target Support**
  - Compilation for `wasm32-unknown-unknown` target
  - `js` feature flag for browser environments (enables getrandom/js)
  - Compile-time guard preventing `parallel` feature on WASM targets

## [0.3.0] - 2026-01-03

### Added

- **Optional Parallel AutoARIMA**
  - Feature-gated Rayon parallelization for model evaluation
  - Enable with `--features parallel` for 4-8x speedup
  - Default: sequential execution (DuckDB compatible)
- **Bootstrap Confidence Intervals**
  - `BootstrapConfig` for configuring bootstrap parameters
  - `bootstrap_intervals()` for empirical confidence intervals
  - `bootstrap_forecast()` convenience function
  - Residual bootstrap and block bootstrap methods
- **True Stepwise Search for AutoARIMA**
  - Neighbor-based hill climbing algorithm
  - Reduces model evaluations by 60-70%
  - Enable with `AutoARIMAConfig::with_true_stepwise(true)`
- **Property-Based Testing**
  - 20+ proptest cases for model invariants
  - Tests forecast length, finite values, interval ordering
  - Tests fitted values + residuals reconstruction
- **Interval Calibration Testing**
  - Rolling origin cross-validation for coverage rate testing
  - Winkler score for interval quality assessment
  - Coverage rate tests for analytical and bootstrap intervals

### Removed

- **Time Series Clustering** - Removed clustering module (not needed)

### Changed

- Improved test coverage with 1,136+ tests total
- Updated dependencies

## [0.2.0] - 2025-12-17

### Added

- **Periodicity Detection Module**
  - `ACFPeriodicityDetector` - time-domain detection using ACF peaks
  - `FFTPeriodicityDetector` - frequency-domain detection using periodogram
  - `Autoperiod` - hybrid FFT+ACF detector (Vlachos et al. 2005)
  - `CFDAutoperiod` - noise-resistant detector with clustering (Puech et al. 2020)
  - `SAZED` - parameter-free ensemble method (Toller et al. 2019)
  - Convenience functions: `detect_period()`, `detect_period_ensemble()`, `detect_period_range()`
  - `PeriodicityDetector` trait for unified API
- **FFT Utilities**
  - `fft_real()` - FFT for real-valued signals
  - `periodogram()` - power spectral density computation
  - `periodogram_peaks()` - significant peak detection
  - `welch_periodogram()` - Welch's method for reduced variance
- **SIMD-Accelerated Operations**
  - Vector sum, mean, variance, standard deviation
  - Dot product and sum of squares
  - Squared Euclidean and Manhattan distances
  - Element-wise operations (add, subtract, multiply, divide, scale)
  - Uses Trueno for AVX2/SSE2/NEON acceleration
- **Validation Tools**
  - CLI tool for periodicity detection (`examples/analysis/detect_period.rs`)
  - Python cross-validation script against pyriodicity
  - Criterion benchmarks for periodicity detection

### Changed

- Updated documentation with periodicity detection examples
- Added `rustfft` dependency for FFT operations

## [0.1.0] - 2025-12-11

### Added

- Initial release of anofox-forecast
- **Core Data Structures**
  - `TimeSeries` for univariate and multivariate time series data
  - `Forecast` for prediction results with confidence intervals
  - `CalendarAnnotations` for holidays and regressors
- **Forecasting Models (35+)**
  - ARIMA and AutoARIMA with automatic order selection
  - Exponential Smoothing: SES, Holt's Linear, Holt-Winters, ETS, AutoETS
  - Baseline methods: Naive, Seasonal Naive, Random Walk with Drift, SMA
  - Theta method
  - Intermittent demand: Croston, ADIDA, TSB
  - Ensemble methods with multiple combination strategies
- **Feature Extraction (76+ features)**
  - Basic statistics (mean, variance, quantiles, etc.)
  - Distribution features (skewness, kurtosis, etc.)
  - Autocorrelation and partial autocorrelation
  - Entropy features (approximate, sample, permutation, binned)
  - Complexity features (C3, CID, Lempel-Ziv)
  - Trend analysis and stationarity tests
- **Seasonality & Decomposition**
  - STL (Seasonal-Trend decomposition using LOESS)
  - MSTL (Multiple Seasonal-Trend decomposition)
- **Changepoint Detection**
  - PELT algorithm with L1, L2, Normal, and Poisson cost functions
- **Anomaly Detection**
  - Statistical methods (IQR, z-score)
  - Automatic threshold selection
- **Time Series Clustering**
  - Dynamic Time Warping (DTW) distance
  - K-Means clustering with multiple distance metrics
- **Data Transformations**
  - Scaling: standardization, min-max, robust scaling
  - Box-Cox transformation with automatic lambda selection
  - Window functions: rolling and expanding statistics, EWM
- **Model Evaluation**
  - Accuracy metrics (MAE, MSE, RMSE, MAPE, etc.)
  - Time series cross-validation
  - Residual testing and stationarity tests
