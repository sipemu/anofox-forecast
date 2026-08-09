<!-- GSD:project-start source:PROJECT.md -->

## Project

**anofox-forecast — Performance & Validation Hardening**

`anofox-forecast` is a mature Rust time-series forecasting library (v0.15.8) with 30+ models
behind a common `Forecaster` trait, seasonal decomposition, feature extraction, uncertainty
quantification, and WebAssembly/JS bindings published as `@sipemu/anofox-forecast`.

This project is a **proactive, whole-library performance and validation hardening cycle**: for
each quality dimension, stand up a repeatable measurement harness, capture baseline numbers,
produce a prioritized improvement backlog, then land the highest-value fixes — each proven with a
before/after delta. It is aimed at the maintainers of the library, not a new end-user feature.

**Core Value:** **Every claimed capability is measured, and every improvement is proven with a before/after
number.** No unquantified "it feels faster" or "it looks more accurate" — measurement is the
backbone that makes the improvements trustworthy and non-regressing.

### Constraints

- **Tech stack**: Rust 2021, stable toolchain (CI also tests beta/nightly); WASM via wasm-pack — measurement tooling must fit this toolchain.
- **Compatibility**: Public `Forecaster` API stays backward-compatible; the published npm package `@sipemu/anofox-forecast` must keep building.
- **Performance target philosophy**: improvements are only "done" when backed by a reproducible before/after measurement; no unmeasured optimizations.
- **Feature gates**: work must respect existing feature flags (`distributional`, `postprocess`, `anomaly`, `forecastability`, `seasonal-detection`, `parallel`, `serde`, `js`) and their WASM restrictions.
- **CI hygiene**: clippy `-D warnings` and cargo-audit/deny gates must stay green.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust 2021 edition - Core forecasting library and models
- JavaScript/TypeScript - WebAssembly bindings and npm package
- HTML/CSS - Interactive playground UI

## Runtime

- Rust 1.56+ (stable toolchain required, nightly supported)
- Node.js 22.x - For JavaScript/npm workflows
- Browser environment - WebAssembly runtime for JS bindings
- Cargo 1.56+ - Rust dependency and build management
- npm - JavaScript/Node.js package management

## Frameworks

- Rust standard library - Core language runtime
- wasm-bindgen 0.2 - JavaScript ↔ WebAssembly bindings
- js-sys 0.3 - Raw JavaScript type bindings
- proptest 1.5 - Property-based testing (dev-dependency)
- criterion 0.5 - Benchmarking framework
- wasm-bindgen-test 0.3 - WASM test harness
- wasm-pack - Rust → WebAssembly toolchain (auto-installs via Makefile)
- cargo-fmt - Code formatting
- cargo-clippy - Linting with `-D warnings` enforcement
- cargo-audit - Security advisory checking with transitive denial allowlist
- cargo-deny - Comprehensive dependency scanning (advisories, licenses, sources)

## Key Dependencies

- chrono 0.4 - Date/time handling with serde support
- statrs 0.18 - Statistical computations (distributions, hypothesis tests)
- trueno 0.8 - Core numerical algorithms
- anofox-regression 0.5.3 (optional) - 11 regression backends for feature-based forecasting
- anofox-statistics 0.4 - Custom statistical functions
- lbfgs 0.3 - LBFGS optimization algorithm
- rayon 1.10 (optional via `parallel` feature) - Data parallelism across model fitting
- faer 0.23 (optional via `postprocess` feature) - Linear algebra for interval postprocessing
- fdars-core 0.9 (optional via `seasonal-detection` feature) - Periodicity detection
- serde 1.0 + serde_json 1.0 + bincode 1.3 (optional via `serde` feature) - Model serialization
- thiserror 2.0 - Error type derivation
- rand 0.8.6 - Random number generation (cryptographic seeding via getrandom)
- serde-wasm-bindgen 0.6 - WASM-compatible serde bridge for JavaScript
- js-sys 0.3 - Low-level JavaScript API access
- serde_json 1.0 - JSON serialization for examples and testing
- getrandom 0.2 (JS feature for WASM) - Secure random number generation with JavaScript fallback

## Configuration

- Rust edition: 2021
- WASM targets: `wasm32-unknown-unknown`
- Build profiles:
- `Cargo.toml` - Main workspace manifest
- `Makefile` - Convenience targets for WASM, testing, publishing
- `Cargo.lock` - Locked dependency versions (deterministic builds)
- `js/package.json` - NPM package configuration for `@sipemu/anofox-forecast`

## Platform Requirements

- Rust toolchain (stable/beta/nightly) — CI tests all three
- wasm-pack (auto-installs or pre-installed)
- Chrome (for WASM headless browser tests)
- Make or equivalent
- Node.js 22+ (for npm testing and publishing)
- **Rust crates.io:** Compiled binary as dependency — no external runtime needed
- **WebAssembly/npm:** Browser or Node.js with WASM support (all modern versions)
- **GitHub Pages:** Deployment via Actions for interactive playground
- **npm registry:** Package published with OIDC + provenance attestation (npm 11+)

## Version Information

- `default` - Enables `postprocess` (prediction interval methods)
- `parallel` - Activates rayon for multi-threaded model fitting
- `js` - WASM-only: JavaScript random number generation via getrandom
- `postprocess` - Conformal, bootstrap, quantile regression intervals (requires faer + anofox-regression)
- `forecastability` - Mutual information analysis, transfer entropy, distance correlation
- `seasonal-detection` - Automated periodicity detection (fdars-core)
- `serde` - Model serialization to JSON/bincode
- `distributional` - Distributional forecasting (e.g., GARCH, quantile outputs)
- `anomaly` - Streaming anomaly detection (requires distributional)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Modules use `snake_case`: `simple_exponential_smoothing.rs`, `auto_forecast.rs`
- Model implementations follow their acronym in lowercase: `arima.rs`, `garch.rs`, `theta.rs`
- Directory modules are named `mod.rs` and re-export public types to a parent
- Test files in integration suite use descriptive lowercase: `auto_arima_validation.rs`, `property_tests.rs`, `ets_validation.rs`
- Public functions use `snake_case`: `fit()`, `predict()`, `fit_predict()`, `predict_with_intervals()`
- Constructor methods: `new()`, `auto()` (for auto-optimized variants), `with_dimensions()`
- Internal helper functions prefixed with `_internal`: `predict_internal()`
- Factory functions in tests: `make_ts()`, `make_timestamps()`, `make_test_data()`, `make_values_strategy()`
- Local variables use `snake_case`: `last_value`, `fitted_values`, `horizon`, `alpha`, `residuals`
- Loop counters: `i`, `h` (common in forecasting context for horizon step)
- Collections use plural form: `timestamps`, `values`, `predictions`, `regressors`
- Public structs use `PascalCase`: `Naive`, `SimpleExponentialSmoothing`, `AutoETS`, `ARIMA`, `GARCH`
- All acronyms capitalized: `ETS`, `ARIMA`, `MSTL`, `STL`, `TBATS`, `VAR`, `GARCH`, `SES`
- Type aliases use `PascalCase`: `Result<T>`, `FittedParams`, `Forecast`, `TimeSeries`
- Enums use `PascalCase` variants: `ErrorType::Additive`, `SeasonalType::Additive`, `TrendType::Linear`

## Code Style

- Rust edition 2021
- Standard rustfmt with default settings (no custom config file)
- 4-space indentation
- Clippy enabled with specific pragmatic allowances declared in `lib.rs`:
- Strict mode: CI runs `cargo clippy --all-targets --all-features -- -D warnings` (deny warnings)
- No warnings allowed in production builds

## Import Organization

- No path aliases defined in `Cargo.toml`
- Imports reference full crate path: `crate::core::TimeSeries`, `crate::models::Forecaster`
- Re-exports in `mod.rs` files provide convenient namespacing (e.g., `pub use naive::Naive`)
- Prelude module at `crate::prelude` exports most-common types for convenience imports

## Error Handling

- Custom error enum: `ForecastError` in `src/error.rs` using `thiserror::Error` derive
- Type alias: `pub type Result<T> = std::result::Result<T, ForecastError>`
- All fallible operations return `Result<T>`
- Error variants include context: `InsufficientData { needed: usize, got: usize, hint: Option<String> }`
- Errors support optional hints for user guidance: `hint: Some("window must be positive".into())`
- Sub-model errors wrap source: `SubModelError { model_name: String, source: Box<ForecastError> }`
- Early validation: `validate_series_complete()` called at start of every `fit()` implementation
- Missing values checked before operations: `series.has_missing_values()` returns `ForecastError::MissingValues`
- Fit-before-predict enforced: Methods return `FitRequired { model: Option<String> }` if called before fit

#[derive(Error, Debug, Clone, PartialEq)]

## Logging

- Logging used only in tests for diagnostic output (visibility into test behavior)
- Production code avoids logging (library design principle)
- Test diagnostics include intermediate state: penalty values, changepoint locations, model parameters
- Example from `tests/auto_changepoint_test.rs`:

## Comments

- Complex algorithms receive top-of-file documentation
- Non-obvious parameter bounds explained inline
- Workarounds for language limitations flagged (e.g., WASM constraints)
- Mathematical concepts (eigenvalues, matrix operations) briefly explained
- References to papers or specifications noted
- All public functions documented with `///` before signature
- Doc comments describe purpose, arguments, and typical usage
- Examples provided when usage pattern is non-obvious
- Error conditions documented in doc comment or Error variant description
- Doc examples use `#[doc]` attributes and must be valid Rust (cargo test validates them)
- Rarely used; code preferred to be self-documenting
- Used sparingly in numerical sections where loop intent may be unclear
- Flag changes needed for WASM or platform-specific concerns

## Function Design

- Prefer functions under 50 lines
- Numerical/algorithmic sections may reach 100+ lines if coherent block
- Refactor into helpers when sections exceed 80 lines
- Models accept TimeSeries for training, horizon for prediction
- Configuration passed via builder pattern (e.g., `AutoETSConfig`) or separate config struct
- Exogenous regressors passed as `HashMap<String, Vec<f64>>` on `predict_with_exog()`
- No default parameters; overloads via method naming (`auto()` vs `new(alpha)`)
- `Result<T>` for all fallible operations
- Tuple returns when multiple related values (e.g., `(fitted: Vec<f64>, residuals: Vec<f64>)`)
- Option<T> for computed fields that may not apply (e.g., `trend_component()` → `Result<&[f64]>`)
- Forecasts returned as `Forecast` struct containing `Vec<Vec<f64>>` for dimensions × horizon

## Module Design

- Each module (`mod.rs`) re-exports public types it defines
- Private implementation details remain in submodules (e.g., `naive.rs` private, exported via `mod.rs`)
- Traits and structs for model composition: `Forecaster`, `ModelSpec`, `BoxedForecaster`
- Extensive use in `src/models/mod.rs` to group and re-export model families
- Example from `src/models/mod.rs`:
- Prelude at `src/lib.rs::prelude` provides convenient access:

## Documentation Structure

- Every module starts with `//!` doc comment describing its purpose
- Lists main types and common use cases
- Example from `src/models/exponential/mod.rs`:
- Conditional modules marked with `#[cfg(feature = "...")]`
- Example: `#[cfg(feature = "anomaly")]` guards entire module
- Documented in Cargo.toml with purpose

## Serialization

- When feature `serde` enabled, models derive `Serialize + Deserialize`
- Custom serializer `persistence::nan_vec` handles NaN values in Vec<f64>
- Exogenous regressor data marked `#[serde(skip)]` to avoid serialization
- Serde dependencies optional: `serde = { version = "1.0", features = ["derive"], optional = true }`

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

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

- **Trait-driven design**: `Forecaster` trait allows any model to fit/predict uniformly
- **Feature-gated subsystems**: `distributional`, `postprocess`, `anomaly`, `forecastability`, `seasonal-detection` optional
- **Hybrid parallelism**: Rayon for series-level parallelism, SIMD for linear algebra
- **Streaming support**: Laplace leaves maintain per-observation state for online updates
- **Error-aware**: Result types propagate context; models validate input before fitting

## Layers

- Purpose: Define TimeSeries input format and Forecast output format
- Location: `src/core/`
- Contains: TimeSeries builder with calendar annotations, Forecast with optional intervals, ForecastConstraint
- Depends on: chrono (timestamps), error module
- Used by: All models, utilities
- Purpose: Unify all forecasting algorithms under a common trait contract
- Location: `src/models/traits.rs`
- Contains: `Forecaster` trait (fit/predict/residuals/trends), `FittedParams`, `Inspectable` for explanation
- Depends on: Core data structures, error module
- Used by: Every forecasting model, cross-validation, ensemble logic
- Purpose: Specific forecasting algorithms with parameter optimization
- Location: `src/models/{arima,exponential,theta,tbats,baseline,intermittent,laplace,ensemble,...}`
- Contains: Model-specific fitting logic, parameter storage, prediction equations
- Depends on: Forecaster trait, seasonality module, linear algebra (lbfgs, faer)
- Used by: API layer, AutoForecast selectors, SmartForecaster
- Purpose: Decompose series into trend/seasonal/remainder components
- Location: `src/seasonality/`
- Contains: STL (LOESS-based), MSTL (multiple periods), Fourier, HP filter, polynomial trends
- Depends on: Core structures, mathematical libraries
- Used by: ARIMA (seasonal differencing detection), TBATS, decomposition analysis
- Purpose: Compute statistical properties and classify time series behavior
- Location: `src/features/`, `src/forecastability/`, `src/detection/`
- Contains: Autocorrelation, entropy, complexity, STI class, anomaly detectors, outlier/changepoint algorithms
- Depends on: Core structures, FFT (via statrs), statistical utilities
- Used by: AutoForecast model selection, SmartForecaster family classification, diagnostics
- Purpose: Pre/post-processing of values (scaling, power transforms, windowing)
- Location: `src/transform/`
- Contains: Box-Cox, Yeo-Johnson, StandardScaler, Pipeline chaining
- Depends on: Core structures, optimization (lbfgs)
- Used by: Models (internal), feature extraction, validation workflows
- Purpose: Uncertainty quantification, cross-validation, model selection
- Location: `src/utils/`, `src/validation/`, `src/postprocess/`
- Contains: Cross-validate, bootstrap, residual diagnostics, conformal/calibration methods
- Depends on: All model layer, error module, metrics
- Used by: Applications requiring uncertainty bands or model tuning
- Purpose: Efficient multi-series forecasting with parallelism
- Location: `src/batch.rs`
- Contains: `auto_ets()`, `auto_arima()`, `auto_theta()` for parallel series lists
- Depends on: Model implementations, rayon (if `parallel` feature enabled)
- Used by: Large-scale applications, production inference

## Data Flow

### Primary Request Path (Fit-Predict)

### Batch Processing Path

### Cross-Validation Path

### AutoForecast Selection Path (SmartForecaster)

- Models cache fitted values, residuals after fitting (lazy evaluation for decomposable models)
- Laplace leaves maintain per-observation state (`level`, `variance`, `shape`) throughout fitting
- Parameter storage: scalar params in `HashMap<String, f64>`, seasonal state in `Option<Vec<f64>>`
- No global mutable state; all state encapsulated in model instance

## Key Abstractions

- Purpose: Polymorphic contract for all forecasting algorithms
- Examples: `ARIMA`, `ETS`, `Theta`, `TBATS`, `LaplaceForecaster`, `VAR`
- Pattern: Object-safe dyn trait; boxed forecasters enable dynamic dispatch
- Variations: Some models support exogenous regressors (`supports_exog()`, `predict_with_exog()`)
- Purpose: Immutable input representation with metadata
- Examples: Univariate `vec![1.0, 2.0, ...]`, Multivariate with multiple dimensions
- Pattern: Builder pattern with validation; CalendarAnnotations for exogenous variables
- Variations: Sparse (missing values handled via interpolation), Aggregated (multiple series stacked)
- Purpose: Output container with point predictions and optional prediction intervals
- Examples: `Forecast::from_values(vec![10.0, 11.0, 12.0])`, `::from_values_with_intervals(...)`
- Pattern: Holds `Vec<Vec<f64>>` for multivariate support; lower/upper optional
- Variations: Distributional output (Laplace) exposes `GaussianMixture` directly
- Purpose: Extended trait for models producing probability distributions
- Examples: `LaplaceForecaster`, `BootstrapPredictor`
- Pattern: `forecast_dist()` returns `Vec<Gaussian>` or `Vec<GaussianMixture>`
- Variations: Some combine point + distributional (Laplace); others purely distributional
- Purpose: Decomposition abstractions for trend extraction
- Examples: `STL::decompose()`, `MSTL::decompose()`
- Pattern: Returns (trend, seasonal, remainder); models can expose via `trend_component()`, `seasonal_component()`
- Variations: Some models decompose on fit (ETS, TBATS); others on-demand (STL)

## Entry Points

- Location: `src/lib.rs`
- Triggers: `use anofox_forecast::prelude::*` imports public API
- Responsibilities: Re-export core types, models, utilities; declare feature gates; define `prelude`
- Location: `src/models/{arima,exponential,...}/mod.rs`
- Triggers: User instantiates `ARIMA::new()`, `AutoETS::default()`, etc.
- Responsibilities: Struct constructor, method definitions, model-specific state
- Location: `src/models/auto_forecast.rs`, `src/models/smart.rs`, `src/models/laplace/forecaster.rs`
- Triggers: User creates `AutoForecast`, `SmartForecaster`, or `LaplaceForecaster`
- Responsibilities: Candidate evaluation, model pool traversal, selection logic
- Location: `src/batch.rs`
- Triggers: User calls `batch::auto_ets()`, `batch::auto_arima()`, etc.
- Responsibilities: Iterator setup, parallelization dispatch, result aggregation
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

### Fitting Without Validation

### Exposing Model State Mutably

### Duplicate Seasonality Detection

## Error Handling

- **Input validation errors:** Return `ForecastError::InsufficientData`, `::EmptyData`, `::MissingValues` early in `fit()`
- **Fitting failures:** Return `::ConvergenceFailure` if optimization doesn't converge; `::SingularMatrix` if linalg fails
- **Prediction state errors:** Return `::FitRequired { model: Some(name) }` if `predict()` called before `fit()`
- **Composite failures:** Return `::SubModelError { model_name, source }` from ensemble/compound models; propagates inner error with context
- **Errors with hints:** `InsufficientData` includes optional `hint` field for actionable guidance (e.g., "try reducing MA order")

## Cross-Cutting Concerns

- **Data completeness:** `validate_series_complete()` checks for NaN/Inf
- **Dimension consistency:** Models assert `series.len() >= required_min` with `InsufficientData` error
- **Parameter bounds:** AutoETS/AutoARIMA validate (p, d, q) and (P, D, Q) ranges before fitting

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
