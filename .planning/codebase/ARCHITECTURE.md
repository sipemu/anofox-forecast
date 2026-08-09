<!-- refreshed: 2026-08-09 -->
# Architecture

**Analysis Date:** 2026-08-09

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────┐
│                          User Applications                            │
│  (Rust lib, WebAssembly/JS, DuckDB extension, Python bindings)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Public API Layer                                │
│  `lib.rs` prelude: TimeSeries, Forecast, Forecaster trait            │
│  Models exposed: AutoETS, ARIMA, ETS, Theta, Ensemble, Laplace       │
│  Utils exposed: cross_validate, calculate_metrics, bootstrap_forecast│
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Model Implementations                             │
│  Univariate:  ARIMA, ETS, Theta, TBATS, GARCH, Baseline              │
│  Multivariate: VAR, Kalman, Hierarchical                             │
│  Ensemble:     AutoEnsemble, SmartForecaster                         │
│  Streaming:    Laplace (distributional)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│            Seasonal Decomposition & Trend Extraction                 │
│  STL, MSTL, Fourier, HP Filter, Hamilton, Polynomial, Piecewise     │
│  Automatic period detection, trend components                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│         Feature Extraction & Series Analysis                         │
│  Basic stats, autocorr, entropy, complexity, change detection        │
│  Forecastability metrics, STI classification, surrogate testing      │
│  Outlier detection, changepoint detection, anomaly detection         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│          Validation & Postprocessing                                 │
│  Cross-validation, bootstrap, residual diagnostics                   │
│  Conformal prediction, calibration, quantile regression              │
│  Historical simulation, IDR, QRA                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│           Core Data Structures & Utilities                           │
│  TimeSeries (timestamps, values, metadata), Forecast (point+intervals)│
│  Math: SIMD ops, linear algebra (faer), optimization (lbfgs)         │
│  Error handling, metrics calculation, time frequency handling        │
└───────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Core Data** | TimeSeries representation, Forecast structure, Constraints | `src/core/` |
| **Models Base** | Forecaster trait, model registry, parameter storage | `src/models/traits.rs` |
| **ARIMA Family** | ARIMA, AutoARIMA with d/D detection, differencing | `src/models/arima/` |
| **ETS Family** | ETS, AutoETS with 30-spec search, Holt/HW variants | `src/models/exponential/` |
| **Theta** | Theta, AutoTheta, dynamic theta with trend extraction | `src/models/theta/` |
| **TBATS** | TBATS, AutoTBATS with Fourier seasonality | `src/models/tbats/` |
| **Baseline** | Naive, SeasonalNaive, SMA, RandomWalk, SeasonalWindow | `src/models/baseline/` |
| **Intermittent** | Croston, Intermittent Demand, IMAPA, TSB, ADIDA | `src/models/intermittent/` |
| **Laplace** | Distributional forecasting, mixture models, streaming leaves | `src/models/laplace/` |
| **Ensemble** | AutoEnsemble, weighted combinations, CV selection | `src/models/ensemble/` |
| **Smart Forecaster** | Family-aware routing (retail/continuous), AID classifier | `src/models/smart.rs` |
| **Seasonality** | STL/MSTL decomposition, Fourier, trend components | `src/seasonality/` |
| **Features** | Statistical metrics, autocorrelation, entropy, complexity | `src/features/` |
| **Forecastability** | Time series classification, MI variants, Lyapunov | `src/forecastability/` |
| **Detection** | Period detection (FFT), outliers, changepoints, anomalies | `src/detection/`, `src/changepoint/`, `src/anomaly/` |
| **Transforms** | Box-Cox, Yeo-Johnson, scaling, windowing, pipelines | `src/transform/` |
| **Validation** | Cross-validation, diagnostics, residual tests, bootstrap | `src/utils/`, `src/validation/` |
| **Postprocessing** | Conformal prediction, calibration, quantile methods | `src/postprocess/` |
| **Batch Processing** | Parallel series forecasting with shared computation | `src/batch.rs` |
| **Monitor** | Sequential hypothesis testing, SPRT | `src/monitor/` |

## Pattern Overview

**Overall:** Modular layered forecasting library with trait-based polymorphism.

**Key Characteristics:**
- **Trait-driven design**: `Forecaster` trait allows any model to fit/predict uniformly
- **Feature-gated subsystems**: `distributional`, `postprocess`, `anomaly`, `forecastability`, `seasonal-detection` optional
- **Hybrid parallelism**: Rayon for series-level parallelism, SIMD for linear algebra
- **Streaming support**: Laplace leaves maintain per-observation state for online updates
- **Error-aware**: Result types propagate context; models validate input before fitting

## Layers

**Core Data Structures:**
- Purpose: Define TimeSeries input format and Forecast output format
- Location: `src/core/`
- Contains: TimeSeries builder with calendar annotations, Forecast with optional intervals, ForecastConstraint
- Depends on: chrono (timestamps), error module
- Used by: All models, utilities

**Model Interface:**
- Purpose: Unify all forecasting algorithms under a common trait contract
- Location: `src/models/traits.rs`
- Contains: `Forecaster` trait (fit/predict/residuals/trends), `FittedParams`, `Inspectable` for explanation
- Depends on: Core data structures, error module
- Used by: Every forecasting model, cross-validation, ensemble logic

**Model Implementations (30+):**
- Purpose: Specific forecasting algorithms with parameter optimization
- Location: `src/models/{arima,exponential,theta,tbats,baseline,intermittent,laplace,ensemble,...}`
- Contains: Model-specific fitting logic, parameter storage, prediction equations
- Depends on: Forecaster trait, seasonality module, linear algebra (lbfgs, faer)
- Used by: API layer, AutoForecast selectors, SmartForecaster

**Seasonality & Trend Extraction:**
- Purpose: Decompose series into trend/seasonal/remainder components
- Location: `src/seasonality/`
- Contains: STL (LOESS-based), MSTL (multiple periods), Fourier, HP filter, polynomial trends
- Depends on: Core structures, mathematical libraries
- Used by: ARIMA (seasonal differencing detection), TBATS, decomposition analysis

**Feature Extraction & Analysis:**
- Purpose: Compute statistical properties and classify time series behavior
- Location: `src/features/`, `src/forecastability/`, `src/detection/`
- Contains: Autocorrelation, entropy, complexity, STI class, anomaly detectors, outlier/changepoint algorithms
- Depends on: Core structures, FFT (via statrs), statistical utilities
- Used by: AutoForecast model selection, SmartForecaster family classification, diagnostics

**Transformation Pipeline:**
- Purpose: Pre/post-processing of values (scaling, power transforms, windowing)
- Location: `src/transform/`
- Contains: Box-Cox, Yeo-Johnson, StandardScaler, Pipeline chaining
- Depends on: Core structures, optimization (lbfgs)
- Used by: Models (internal), feature extraction, validation workflows

**Validation & Postprocessing:**
- Purpose: Uncertainty quantification, cross-validation, model selection
- Location: `src/utils/`, `src/validation/`, `src/postprocess/`
- Contains: Cross-validate, bootstrap, residual diagnostics, conformal/calibration methods
- Depends on: All model layer, error module, metrics
- Used by: Applications requiring uncertainty bands or model tuning

**Batch Processing:**
- Purpose: Efficient multi-series forecasting with parallelism
- Location: `src/batch.rs`
- Contains: `auto_ets()`, `auto_arima()`, `auto_theta()` for parallel series lists
- Depends on: Model implementations, rayon (if `parallel` feature enabled)
- Used by: Large-scale applications, production inference

## Data Flow

### Primary Request Path (Fit-Predict)

1. **Input** — User creates `TimeSeries::univariate()` or `::multivariate()` with timestamps, values (`src/core/time_series.rs:new`)
2. **Validation** — TimeSeries validates frequency inference, detects missing values, stores metadata
3. **Model Creation** — User instantiates a model (e.g., `ARIMA::new(1,1,1)`, `AutoETS::default()`) (`src/models/{arima,exponential,...}/mod.rs`)
4. **Fitting** — Model's `fit()` implementation:
   - Calls `validate_series_complete()` to check for NaN/Inf (`src/models/traits.rs:validate_series_complete`)
   - Extracts values, timestamps from TimeSeries
   - Optimizes parameters using lbfgs, closed-form solutions, or streaming state updates
   - Caches fitted values, residuals, diagnostics
5. **Prediction** — User calls `predict(horizon)` or `predict_with_intervals(horizon, level)` (`src/models/traits.rs:predict`)
   - Model generates point forecasts using cached state
   - If intervals requested, computes bounds (model-specific or via postprocessing)
6. **Output** — Returns `Forecast` containing `point`, `lower`, `upper` vectors (`src/core/forecast.rs`)

### Batch Processing Path

1. User calls `batch::auto_ets(values_list, period, horizon)` (`src/batch.rs:auto_ets`)
2. For each series:
   - Create TimeSeries with dummy timestamps
   - Fit AutoETS with same config (shared Fourier design matrix)
   - Parallel execution via rayon (if `parallel` feature)
3. Collect results into `Vec<Result<(Forecast, ETSSpec)>>`

### Cross-Validation Path

1. User calls `cross_validate()` with model config, TimeSeries, and CV config (`src/utils/cross_validation.rs`)
2. Split TimeSeries into train/test folds using sliding or fixed windows
3. For each fold:
   - Fit model on training window
   - Predict on test horizon
   - Calculate metrics (MASE, MAE, RMSE, SMAPE)
4. Aggregate metrics into CVResults with mean/min/max per horizon

### AutoForecast Selection Path (SmartForecaster)

1. User creates `SmartForecaster::new()` or `LaplaceForecaster::auto_aid()` (`src/models/smart.rs`, `src/models/laplace/forecaster.rs`)
2. `fit()` analyzes series:
   - Compute AID features (demand pattern, zeros/variance ratio) (`src/models/laplace/recommend.rs`)
   - Route to distribution family (Poisson, NegBinom, Gaussian, etc.)
   - Select leaf ensemble (streaming leaves for Laplace, or pool of ETS specs)
3. `predict()` invokes family-specific forecaster
4. Returns point + optional distribution (`Gaussian`, `GaussianMixture`)

**State Management:**
- Models cache fitted values, residuals after fitting (lazy evaluation for decomposable models)
- Laplace leaves maintain per-observation state (`level`, `variance`, `shape`) throughout fitting
- Parameter storage: scalar params in `HashMap<String, f64>`, seasonal state in `Option<Vec<f64>>`
- No global mutable state; all state encapsulated in model instance

## Key Abstractions

**Forecaster Trait:**
- Purpose: Polymorphic contract for all forecasting algorithms
- Examples: `ARIMA`, `ETS`, `Theta`, `TBATS`, `LaplaceForecaster`, `VAR`
- Pattern: Object-safe dyn trait; boxed forecasters enable dynamic dispatch
- Variations: Some models support exogenous regressors (`supports_exog()`, `predict_with_exog()`)

**TimeSeries:**
- Purpose: Immutable input representation with metadata
- Examples: Univariate `vec![1.0, 2.0, ...]`, Multivariate with multiple dimensions
- Pattern: Builder pattern with validation; CalendarAnnotations for exogenous variables
- Variations: Sparse (missing values handled via interpolation), Aggregated (multiple series stacked)

**Forecast:**
- Purpose: Output container with point predictions and optional prediction intervals
- Examples: `Forecast::from_values(vec![10.0, 11.0, 12.0])`, `::from_values_with_intervals(...)`
- Pattern: Holds `Vec<Vec<f64>>` for multivariate support; lower/upper optional
- Variations: Distributional output (Laplace) exposes `GaussianMixture` directly

**DistributionalForecaster:**
- Purpose: Extended trait for models producing probability distributions
- Examples: `LaplaceForecaster`, `BootstrapPredictor`
- Pattern: `forecast_dist()` returns `Vec<Gaussian>` or `Vec<GaussianMixture>`
- Variations: Some combine point + distributional (Laplace); others purely distributional

**SeasonalComponent / TrendComponent:**
- Purpose: Decomposition abstractions for trend extraction
- Examples: `STL::decompose()`, `MSTL::decompose()`
- Pattern: Returns (trend, seasonal, remainder); models can expose via `trend_component()`, `seasonal_component()`
- Variations: Some models decompose on fit (ETS, TBATS); others on-demand (STL)

## Entry Points

**Library Root:**
- Location: `src/lib.rs`
- Triggers: `use anofox_forecast::prelude::*` imports public API
- Responsibilities: Re-export core types, models, utilities; declare feature gates; define `prelude`

**Model Entry (Direct Model Use):**
- Location: `src/models/{arima,exponential,...}/mod.rs`
- Triggers: User instantiates `ARIMA::new()`, `AutoETS::default()`, etc.
- Responsibilities: Struct constructor, method definitions, model-specific state

**Automatic Model Selection:**
- Location: `src/models/auto_forecast.rs`, `src/models/smart.rs`, `src/models/laplace/forecaster.rs`
- Triggers: User creates `AutoForecast`, `SmartForecaster`, or `LaplaceForecaster`
- Responsibilities: Candidate evaluation, model pool traversal, selection logic

**Batch Entry:**
- Location: `src/batch.rs`
- Triggers: User calls `batch::auto_ets()`, `batch::auto_arima()`, etc.
- Responsibilities: Iterator setup, parallelization dispatch, result aggregation

**Validation Entry:**
- Location: `src/utils/cross_validation.rs`
- Triggers: User calls `cross_validate()` with model spec
- Responsibilities: Fold generation, metric calculation, aggregation

## Architectural Constraints

- **Threading:** Single-threaded event loop per model fit; Rayon for cross-series parallelism (when `parallel` feature enabled). WASM target forbids `parallel` feature.
- **Global state:** None. All state encapsulated in model instances or passed as parameters. TimeSeries is immutable.
- **Circular imports:** None by design — modules form a DAG: core → models → higher-level (batch, validation).
- **Feature gates:** `distributional` enables Laplace; `postprocess` enables conformal/QRA; `anomaly` requires `distributional`. Feature combinations validated in `lib.rs` with compile-time checks.
- **Numeric precision:** Models use f64 throughout; no automatic type widening/narrowing.
- **Frequency inference:** Requires regularly-spaced timestamps; gaps trigger `FrequencyInference` error. Frequency detection via `StaticFrequency` or auto-detection.

## Anti-Patterns

### Panicking on Invalid Input

**What happens:** Some internal helper functions use `.unwrap()` or `.expect()` on fallible operations (e.g., matrix operations).
**Why it's wrong:** Library code should never panic. Callers can't handle panics gracefully; error context is lost.
**Do this instead:** Return `Result<T>` and use `?` operator. See error handling in `src/models/arima/model.rs` for correct pattern — check `faer` results, convert to `ForecastError::SingularMatrix`.

### Fitting Without Validation

**What happens:** A few models skip `validate_series_complete()` on entry to `fit()`.
**Why it's wrong:** Missing values silently corrupt parameter estimates. Models should catch invalid input early.
**Do this instead:** All `fit()` implementations **must** call `validate_series_complete(series)` at the start. See `src/models/exponential/ets.rs:fit()` for correct pattern.

### Exposing Model State Mutably

**What happens:** Some older utility functions return `&mut [f64]` for internal caches (fitted values, residuals).
**Why it's wrong:** Callers can corrupt internal state; predictions become invalid. Breaks encapsulation.
**Do this instead:** Return immutable `&[f64]` from public trait methods. Internal caches are private. See `Forecaster::fitted_values()` — returns `Option<&[f64]>` (immutable).

### Duplicate Seasonality Detection

**What happens:** AutoETS and AutoARIMA both infer seasonal periods independently, sometimes selecting different periods.
**Why it's wrong:** Inconsistent model behavior; composite models may use conflicting assumptions.
**Do this instead:** Compute period once at the `TimeSeries` level (or via `fdars` feature if available). See `auto_seasonal.rs` for shared period detection. Feature gate: `seasonal-detection` loads `fdars` for robust detection.

## Error Handling

**Strategy:** Result-based error propagation with context. All public APIs return `Result<T>` except infallible accessors.

**Patterns:**
- **Input validation errors:** Return `ForecastError::InsufficientData`, `::EmptyData`, `::MissingValues` early in `fit()`
- **Fitting failures:** Return `::ConvergenceFailure` if optimization doesn't converge; `::SingularMatrix` if linalg fails
- **Prediction state errors:** Return `::FitRequired { model: Some(name) }` if `predict()` called before `fit()`
- **Composite failures:** Return `::SubModelError { model_name, source }` from ensemble/compound models; propagates inner error with context
- **Errors with hints:** `InsufficientData` includes optional `hint` field for actionable guidance (e.g., "try reducing MA order")

**Example:** See `src/models/arima/auto_arima.rs` — `search_pdq()` catches convergence failures, collects per-order errors, returns best-fitting model or `ConvergenceFailure` if none succeeded.

## Cross-Cutting Concerns

**Logging:** Uses `println!` in examples; no structured logging library (intentional — keep dependencies minimal). Models don't log by default; examples show optional output via print statements.

**Validation:** 
- **Data completeness:** `validate_series_complete()` checks for NaN/Inf
- **Dimension consistency:** Models assert `series.len() >= required_min` with `InsufficientData` error
- **Parameter bounds:** AutoETS/AutoARIMA validate (p, d, q) and (P, D, Q) ranges before fitting

**Authentication:** Not applicable — library does not access external services.

---

*Architecture analysis: 2026-08-09*
