# Codebase Structure

**Analysis Date:** 2026-08-09

## Directory Layout

```
anofox-forecast/
├── src/                          # Main library crate
│   ├── lib.rs                    # Public API, feature gates, prelude
│   ├── error.rs                  # ForecastError enum and Result type
│   ├── core/                     # TimeSeries, Forecast, Constraints
│   ├── models/                   # All forecasting model implementations (30+)
│   ├── seasonality/              # STL/MSTL, Fourier, trend components
│   ├── transform/                # Box-Cox, Yeo-Johnson, scaling, pipelines
│   ├── features/                 # Feature extraction, statistics, complexity
│   ├── forecastability/          # Forecastability metrics, STI class, surrogates (feature-gated)
│   ├── detection/                # Period detection, outliers
│   ├── changepoint/              # Changepoint algorithms (PELT, BINSEG, etc.)
│   ├── anomaly/                  # Streaming anomaly detection (feature-gated, requires distributional)
│   ├── utils/                    # Cross-validation, metrics, bootstrap, OLS, optimization
│   ├── validation/               # Residual diagnostics, stationarity tests, AID
│   ├── postprocess/              # Conformal prediction, calibration, quantile methods (feature-gated)
│   ├── batch.rs                  # Parallel multi-series forecasting
│   ├── monitor/                  # Sequential hypothesis testing, SPRT
│   ├── hierarchy/                # Hierarchical forecasting helpers
│   └── simd.rs                   # SIMD operations for linear algebra
│
├── crates/
│   └── anofox-forecast-js/       # WebAssembly bindings
│       ├── Cargo.toml
│       └── src/lib.rs            # wasm-bindgen exports
│
├── examples/                     # Runnable examples (40+)
│   ├── quickstart.rs             # Basic usage: ARIMA fit/predict
│   ├── forecasting/              # Model examples (ARIMA, ETS, Theta, VAR, etc.)
│   ├── features/                 # Feature extraction, autocorrelation, entropy
│   ├── analysis/                 # STL, changepoint, outlier detection
│   ├── transform/                # Box-Cox, scaling, windowing, pipelines
│   ├── validation/               # Cross-validation, metrics, bootstrap, diagnostics
│   └── postprocess/              # Conformal prediction, QRA, calibration
│
├── tests/                        # Integration tests
│   ├── ets_validation.rs         # ETS accuracy vs Python statsforecast
│   ├── auto_arima_*.rs           # ARIMA selection and validation
│   ├── property_tests.rs         # PropTest property-based tests
│   ├── nixtla_validation.rs      # Comparison vs Nixtla StatsForecast
│   ├── batch_validation.rs       # Batch processing correctness
│   ├── var_integration.rs        # VAR multivariate forecasting
│   ├── persistence_integration.rs # Model serialization (serde)
│   ├── pipeline_integration.rs   # Transform pipeline chaining
│   ├── interval_calibration.rs   # Prediction interval coverage
│   ├── promotion_effect_tests.rs # Exogenous regressor effects
│   ├── statsforecast_comparison.rs # Cross-library validation
│   └── ... (15+ more integration tests)
│
├── benches/                      # Criterion benchmarks
│   ├── simd_benchmark.rs         # SIMD performance
│   ├── arima_benchmark.rs        # ARIMA fit/predict speed
│   ├── ets_benchmark.rs          # ETS model pool search
│   ├── prediction_benchmark.rs   # End-to-end forecasting
│   ├── ensemble_benchmark.rs     # Ensemble combining
│   ├── bootstrap_benchmark.rs    # Bootstrap performance
│   └── comprehensive_benchmark.rs # Full workload
│
├── docs/                         # Documentation
│   ├── ANOMALY_PLAN.md           # Anomaly detection design
│   ├── ARCHITECTURE.md           # (Now in .planning/codebase/)
│   └── rendered/                 # Generated docs (HTML)
│
├── validation/                   # Validation datasets and scripts
│   ├── data/                     # Time series test data (TSF, CSV, Parquet)
│   ├── validation/               # Python validation scripts
│   ├── results/                  # Comparison results
│   └── pyproject.toml            # Python dependencies for validation
│
├── scripts/                      # Build/maintenance scripts
│   └── *.py                      # Data processing, validation helpers
│
├── js/                           # JavaScript build output (wasm-pack)
│   └── (generated at build time)
│
├── Cargo.toml                    # Main workspace and package config
├── Cargo.lock                    # Locked dependencies
├── Makefile                      # Common build targets
├── README.md                     # Project overview and quick start
├── CHANGELOG.md                  # Version history (47KB+)
├── LICENSE                       # MIT license
├── deny.toml                     # Dependency audit config
└── codecov.yml                   # Code coverage thresholds
```

## Directory Purposes

**`src/`** — Main library implementation (~300 files, ~50KB Rust code)
- Purpose: All forecasting algorithms, data structures, utilities
- Key files: `lib.rs` (entry), `error.rs`, `core/`, `models/`
- Public API: Via `prelude` module and individual model exports

**`src/core/`** — Foundational data types
- Purpose: TimeSeries input format, Forecast output format, constraint system
- Key files:
  - `time_series.rs` — Builder, frequency inference, metadata storage
  - `forecast.rs` — Point/interval container with accessor methods
  - `constraints.rs` — ConstrainedForecast for bounded predictions

**`src/models/`** — Forecasting algorithms (140+ files)
- Purpose: Univariate/multivariate models, auto-selectors, ensembles
- Key subdirectories:
  - `arima/` — ARIMA, AutoARIMA with d/D search
  - `exponential/` — ETS, AutoETS, Holt, HW, SES variants
  - `theta/` — Theta, AutoTheta, dynamic
  - `tbats/` — TBATS with Fourier seasonality
  - `baseline/` — Naive, SeasonalNaive, SMA, random walk
  - `intermittent/` — Croston, IMAPA, TSB, ADIDA
  - `laplace/` — Distributional forecasting (streaming leaves, mixtures)
  - `ensemble/` — AutoEnsemble, weighted combining
  - `kalman*` — Kalman filter variants, state-space
  - `var*` — VAR, VARForecaster
  - `garch.rs` — GARCH volatility models
  - `regression.rs` — Linear regression forecasting
  - `smart.rs` — SmartForecaster (family-aware router)
  - `mfles.rs` — Multi-frequency learned smoothing
  - `mstl_forecaster.rs` — MSTL-based forecaster
  - `traits.rs` — Forecaster trait, FittedParams, model registry
  - `explain.rs` — Model introspection/explanation
  - `inspect.rs` — Explanation enum (ARIMA, ETS, Laplace details)
  - `convenience.rs` — Helper constructors
  - `cv_select.rs` — CV-based model selection
- Public API: Trait `Forecaster`, individual model structs exported from root

**`src/seasonality/`** — Decomposition and trend extraction (19 files)
- Purpose: Separate trend, seasonal, remainder components; automatic feature extraction
- Key files:
  - `stl.rs` — STL decomposition (LOESS-based)
  - `mstl.rs` — MSTL (multiple seasonal periods)
  - `fourier.rs` — Fourier basis for seasonality modeling
  - `hp_filter.rs` — Hodrick-Prescott trend extraction
  - `polynomial.rs` — Polynomial trend fitting
  - `piecewise.rs` — Piecewise linear trends
  - `auto_seasonal.rs`, `auto_trend.rs` — Automatic detection
  - `traits.rs` — TrendComponent, SeasonalComponent abstractions
- Used by: ARIMA (d/D detection), TBATS, AutoETS

**`src/transform/`** — Data preprocessing (7 files)
- Purpose: Apply transformations (power, scaling, windowing); reversible pipelines
- Key files:
  - `boxcox.rs` — Box-Cox transformation and inverse
  - `yeo_johnson.rs` — Yeo-Johnson (handles zero/negative)
  - `scale.rs` — StandardScaler, MinMaxScaler, RobustScaler
  - `window.rs` — Rolling window/aggregation
  - `pipeline.rs` — Chainable transform composition
  - `transforms.rs` — Generic transform traits
- Used by: Feature extraction, model preprocessing

**`src/features/`** — Statistical and complexity metrics (13 files)
- Purpose: Extract descriptive features for series analysis and model selection
- Key files:
  - `basic.rs` — Mean, variance, autocorrelation, distribution stats
  - `autocorrelation.rs` — ACF, PACF, lag analysis
  - `entropy.rs` — Approximate entropy, sample entropy
  - `complexity.rs` — Lempel-Ziv complexity, fractal dimension
  - `distribution.rs` — Distribution fitting (Gaussian, exponential, etc.)
  - `change.rs` — Change point features
  - `trend.rs` — Trend strength, curvature
  - `counting.rs` — Zero counts, sign changes
  - `panel.rs` — Cross-series features (for global models)
  - `factory.rs`, `generator.rs` — Feature computation pipelines
  - `selection.rs` — Feature importance ranking
- Used by: SmartForecaster (AID), forecastability classifier, diagnostics

**`src/forecastability/`** — Time series classification (feature-gated)
- Purpose: Compute forecastability metrics; classify difficulty/pattern type
- Key files:
  - `sti_class.rs` — STI class (seasonal/trended/irregular)
  - `scorers.rs` — Forecastability scoring
  - `ami.rs`, `transfer_entropy.rs` — Information-theoretic measures
  - `distance_correlation.rs`, `gcmi.rs`, `knn_mi.rs` — Dependency measures
  - `lyapunov.rs` — Chaos/Lyapunov exponent estimation
  - `fingerprint.rs` — Series signature for matching
  - `triage.rs` — Series routing recommendation
- Gate: `forecastability` feature

**`src/detection/`** — Pattern detection (4 files)
- Purpose: Detect periods, outliers; extract dominant frequencies
- Key files:
  - `period.rs` — Period/seasonality detection via FFT
  - `outlier.rs` — Statistical outlier identification (MAD, IQR)
  - `fft.rs` — FFT utilities

**`src/changepoint/`** — Changepoint detection (20 files)
- Purpose: Identify structural breaks; analyze change types
- Key files:
  - `detector.rs` — Main changepoint detector API
  - `algorithms/` — PELT, BINSEG, bottom-up, dynamic programming, kernelCPD
  - `costs/` — Cost functions (L2, L1, cusum, AR, Mahalanobis, Poisson, normal, etc.)
  - `pelt.rs` — PELT algorithm (Pruned Exact Linear Time)
  - `signal.rs` — Signal amplitude/offset change detection
  - `metrics.rs` — F1, Hausdorff, Jaccard evaluation
- Used by: Time series analysis, change type classification

**`src/anomaly/`** — Streaming anomaly detection (feature-gated)
- Purpose: Detect unusual observations via statistical tests
- Key files:
  - `mahalanobis.rs` — Multivariate Mahalanobis distance (from skaters/microprediction)
  - `parade.rs` — Probabilistic anomaly regression/detection
  - `chi2.rs`, `gpd.rs` — Tail-based anomaly detection
  - `quantile.rs`, `zbank.rs` — Quantile/z-score methods
- Gate: `anomaly` feature (requires `distributional`)

**`src/utils/`** — Validation, metrics, optimization (10 files)
- Purpose: Cross-validation, accuracy metrics, bootstrapping, linear algebra
- Key files:
  - `cross_validation.rs` — `cross_validate()` function, CV configs, fold generation
  - `metrics.rs` — MASE, MAE, RMSE, MAPE, SMAPE, R², PINAW
  - `bootstrap.rs` — Bootstrap forecasting with block structure
  - `ols.rs` — Ordinary Least Squares solver
  - `optimization.rs` — Optimization helpers (bounded search, grid search)
  - `stats.rs` — Statistical utilities (quantiles, distributions)
  - `persistence.rs` — Model serialization helpers
  - `exog_shim.rs` — Exogenous variable handling for univariate models
  - `comparison.rs` — Pairwise model comparison

**`src/validation/`** — Residual analysis and diagnostics (6 files)
- Purpose: Post-fit model validation; residual testing
- Key files:
  - `diagnostics.rs` — Residual normality, ACF, Ljung-Box tests
  - `residual_tests.rs` — Statistical tests for white noise
  - `stationarity.rs` — ADF, KPSS tests
  - `aid.rs` — Intermittent-demand classification (AID)
  - `intermittent_diagnostics.rs` — Demand pattern analysis
  - `multicollinearity.rs` — VIF, correlation checks

**`src/postprocess/`** — Uncertainty quantification (feature-gated)
- Purpose: Convert point forecasts to calibrated prediction intervals
- Key files:
  - `conformal.rs` — Conformal prediction (full/local)
  - `conformalize.rs` — Conformalization wrapper
  - `calibration.rs` — Generic calibration interface
  - `normal.rs` — Gaussian error assumption
  - `historical_sim.rs` — Non-parametric empirical error
  - `idr.rs` — Isotonic Distributional Regression
  - `qra.rs` — Quantile Regression Averaging (ensemble)
  - `cqr.rs` — Conformalized Quantile Regression
  - `bootstrap.rs` — Bootstrap-based intervals
  - `aci.rs`, `enbpi.rs` — Additional calibration methods
  - `processor.rs` — PostProcessor wrapper
- Gate: `postprocess` feature

**`src/batch.rs`** — Parallel series forecasting
- Purpose: Efficient multi-series forecasting with shared computation
- Exports: `auto_ets()`, `auto_arima()`, `auto_theta()`, `ets()`, `arima()`, `theta()`
- Parallelization: rayon (if `parallel` feature); otherwise sequential

**`src/monitor/`** — Sequential monitoring (3 files)
- Purpose: Online hypothesis testing; anomaly detection in streams
- Key files:
  - `sequential.rs` — Sequential test framework
  - `sequential_crit.rs` — Critical value tables
  - `sequential_table.rs` — Test statistics tables

**`src/hierarchy/`** — Hierarchical/grouped forecasting
- Purpose: Coherency constraints for tree-structured time series

**`src/simd.rs`** — SIMD utilities
- Purpose: Vectorized linear algebra operations for performance

**`examples/`** — Runnable examples (50+ files)
- Organization: Subdirectories by topic (forecasting, features, analysis, transform, validation, postprocess)
- Purpose: Demonstrate API usage, serve as integration tests
- Key examples:
  - `quickstart.rs` — ARIMA fit/predict with metrics
  - `forecasting/{arima,exponential,theta,var,kalman}.rs` — Model-specific usage
  - `features/basic_features.rs` — Feature extraction
  - `analysis/stl_decomposition.rs` — Seasonality decomposition
  - `validation/cross_validation.rs` — CV with model comparison
  - `postprocess/conformal.rs` — Uncertainty quantification

**`tests/`** — Integration tests (20+ files)
- Purpose: Validate models against reference implementations, test complex workflows
- Key test suites:
  - `ets_validation.rs` — ETS accuracy vs Python statsforecast
  - `auto_arima_*.rs` — AutoARIMA order selection
  - `property_tests.rs` — PropTest property-based tests
  - `batch_validation.rs` — Batch processing correctness
  - `var_integration.rs` — VAR multivariate
  - `statsforecast_comparison.rs` — Cross-library validation
  - `nixtla_validation.rs` — Comparison vs Nixtla
  - `interval_calibration.rs` — Coverage guarantees
  - `persistence_integration.rs` — Model serialization

**`benches/`** — Criterion benchmarks (8 files)
- Purpose: Performance profiling; regression detection
- Key benchmarks:
  - `arima_benchmark.rs` — ARIMA fitting across orders
  - `ets_benchmark.rs` — ETS model pool search
  - `simd_benchmark.rs` — Linear algebra throughput
  - `comprehensive_benchmark.rs` — Full forecasting pipeline

**`validation/`** — External validation infrastructure
- Purpose: Download, store, validate against competition datasets
- Subdirectories:
  - `data/` — M3, M4, M5, tourism, hospital, NN5 datasets
  - `validation/` — Python validation scripts
  - `results/` — Benchmark output CSVs
  - `cpp_theta_reference/` — Theta reference C++ implementation

**`crates/anofox-forecast-js/`** — WebAssembly bindings
- Purpose: Expose Rust library to JavaScript/TypeScript
- Key files:
  - `Cargo.toml` — wasm-pack metadata, features: js, postprocess, distributional, anomaly
  - `src/lib.rs` — wasm-bindgen exports
- Build: `wasm-pack build --target web` generates JS npm package
- Browser usage: Import compiled WASM + JS wrapper

## Naming Conventions

**Files:**
- Model implementations: `{model_name}.rs` or `{model_name}/mod.rs` (e.g., `arima.rs`, `exponential/mod.rs`)
- Trait definitions: `traits.rs` (centralized trait contract)
- Auto-selectors: `auto_{algo}.rs` (e.g., `auto_arima.rs`, `auto_seasonal.rs`)
- Test files: `{feature}_*.rs` or `*_test.rs` (e.g., `ets_validation.rs`)
- Examples: Descriptive name matching feature (e.g., `cross_validation.rs`, `stl_decomposition.rs`)

**Directories:**
- Public modules: lowercase (e.g., `src/models/`, `src/seasonality/`)
- Sub-packages within models: model-name lowercase (e.g., `src/models/arima/`, `src/models/laplace/`)
- Test data: `tests/data/`, `tests/reference/`

**Functions:**
- Constructor: `new()`, `with_*()` for builders (e.g., `TimeSeries::univariate()`, `ARIMA::new(1,1,1)`)
- Fitting: `fit(&mut self, series: &TimeSeries) -> Result<()>`
- Prediction: `predict(&self, horizon: usize) -> Result<Forecast>`
- Batch operations: `batch::{algo}()` (e.g., `batch::auto_ets()`)
- Accessors: `{field}()` for lazy getters (e.g., `fitted_values()`, `residuals()`)

**Types:**
- Error enum: `ForecastError` (not `Error` to avoid naming collisions)
- Result alias: `pub type Result<T> = std::result::Result<T, ForecastError>`
- Model trait: `Forecaster` (singular, trait-like convention)
- Configs: `{Model}Config` (e.g., `AutoETSConfig`, `BacktestConfig`)
- Result types: `{Operation}Result` (e.g., `CVResults`, `BacktestResult`)

## Where to Add New Code

**New Forecasting Model:**
- Primary code: `src/models/{model_name}/mod.rs` (or `{model_name}.rs` if small)
  - Implement `Forecaster` trait: `fit()`, `predict()`, `fitted_values()`, `residuals()`
  - Store parameters in struct (scalars + optional seasonal state)
  - Example structure: See `src/models/exponential/ets.rs` (medium-sized) or `src/models/baseline/naive.rs` (small)
- Tests: `tests/{model_name}_validation.rs` or add to `tests/property_tests.rs`
- Examples: `examples/forecasting/{model_name}.rs`
- Export: Add to `src/models/mod.rs` public re-exports
- Feature gate: Add optional feature in `Cargo.toml` if model requires heavy dependencies

**New Decomposition Method:**
- Implementation: `src/seasonality/{method_name}.rs`
- Trait implementation: `SeasonalComponent` or `TrendComponent` from `src/seasonality/traits.rs`
- Tests: `tests/seasonality_*.rs`
- Examples: `examples/analysis/{method_name}.rs`
- Export: Add to `src/seasonality/mod.rs`

**New Feature/Metric:**
- Computation: `src/features/{feature_category}.rs` or `src/utils/metrics.rs` (if metric)
- Tests: `tests/` or inline unit tests in feature file
- Examples: `examples/features/{feature_name}.rs`
- Export: Add to `src/features/mod.rs` or `src/utils/mod.rs`

**New Validation/Testing Method:**
- Implementation: `src/utils/` (if utility like cross-validate) or `src/validation/` (if diagnostic)
- Tests: `tests/` to validate against reference implementations
- Export: Add to `src/utils/mod.rs` or `src/validation/mod.rs`
- Example: `examples/validation/{method}.rs`

**Integration Test Suite:**
- Location: `tests/{feature}_integration.rs`
- Pattern:
  ```rust
  #[test]
  fn test_workflow() {
      let ts = TimeSeries::univariate(timestamps, values).unwrap();
      let mut model = SomeModel::new();
      model.fit(&ts).unwrap();
      let forecast = model.predict(10).unwrap();
      assert!(...);
  }
  ```
- Reference data: Use `tests/data/` for fixture CSVs; generate synthetic if needed

**Benchmark:**
- Location: `benches/{feature}_benchmark.rs` (Criterion harness)
- Pattern: Use `criterion::black_box()` to prevent optimization
- Run: `cargo bench --bench {feature}_benchmark`

## Special Directories

**`target/`**
- Purpose: Build output (compiled artifacts, dependencies, docs)
- Generated: Yes (via `cargo build`)
- Committed: No (in `.gitignore`)
- Size: ~2GB (debug build with all examples/tests)

**`js/`**
- Purpose: JavaScript build output from wasm-pack
- Generated: Yes (run `wasm-pack build` in `crates/anofox-forecast-js/`)
- Committed: No (in `.gitignore`)
- Contents: `.wasm` binary, `.js` glue code, TypeScript definitions

**`docs/rendered/`**
- Purpose: Generated HTML documentation
- Generated: Yes (via `cargo doc`)
- Committed: No (in `.gitignore`)

**`validation/data/`**
- Purpose: Large benchmark datasets (M3, M4, M5, tourism, etc.)
- Generated: No (downloaded from competition sources)
- Committed: No (in `Cargo.toml` exclude list; ~1.4 GB)
- Usage: Examples reference but don't require; users can download separately

---

*Structure analysis: 2026-08-09*
