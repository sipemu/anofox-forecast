# Coding Conventions

**Analysis Date:** 2026-08-09

## Naming Patterns

**Files:**
- Modules use `snake_case`: `simple_exponential_smoothing.rs`, `auto_forecast.rs`
- Model implementations follow their acronym in lowercase: `arima.rs`, `garch.rs`, `theta.rs`
- Directory modules are named `mod.rs` and re-export public types to a parent
- Test files in integration suite use descriptive lowercase: `auto_arima_validation.rs`, `property_tests.rs`, `ets_validation.rs`

**Functions:**
- Public functions use `snake_case`: `fit()`, `predict()`, `fit_predict()`, `predict_with_intervals()`
- Constructor methods: `new()`, `auto()` (for auto-optimized variants), `with_dimensions()`
- Internal helper functions prefixed with `_internal`: `predict_internal()`
- Factory functions in tests: `make_ts()`, `make_timestamps()`, `make_test_data()`, `make_values_strategy()`

**Variables:**
- Local variables use `snake_case`: `last_value`, `fitted_values`, `horizon`, `alpha`, `residuals`
- Loop counters: `i`, `h` (common in forecasting context for horizon step)
- Collections use plural form: `timestamps`, `values`, `predictions`, `regressors`

**Types:**
- Public structs use `PascalCase`: `Naive`, `SimpleExponentialSmoothing`, `AutoETS`, `ARIMA`, `GARCH`
- All acronyms capitalized: `ETS`, `ARIMA`, `MSTL`, `STL`, `TBATS`, `VAR`, `GARCH`, `SES`
- Type aliases use `PascalCase`: `Result<T>`, `FittedParams`, `Forecast`, `TimeSeries`
- Enums use `PascalCase` variants: `ErrorType::Additive`, `SeasonalType::Additive`, `TrendType::Linear`

## Code Style

**Formatting:**
- Rust edition 2021
- Standard rustfmt with default settings (no custom config file)
- 4-space indentation

**Linting:**
- Clippy enabled with specific pragmatic allowances declared in `lib.rs`:
  - `#![allow(clippy::upper_case_acronyms)]` — Allow `ARIMA`, `ETS`, `STL` etc.
  - `#![allow(clippy::too_many_arguments)]` — Forecasting models often need many parameters
  - `#![allow(clippy::type_complexity)]` — Generic bounds can be complex
  - `#![allow(clippy::needless_range_loop)]` — Manual loops preferred in SIMD/numerical contexts
  - `#![allow(clippy::manual_memcpy)]` — Explicit loops for clarity in algorithmic code
  - `#![allow(clippy::manual_is_multiple_of)]` — `is_multiple_of` unstable on WASM targets
- Strict mode: CI runs `cargo clippy --all-targets --all-features -- -D warnings` (deny warnings)
- No warnings allowed in production builds

## Import Organization

**Order:**
1. Crate imports from other modules (e.g., `use crate::core::*`)
2. External crate imports (e.g., `use chrono::*`, `use thiserror::Error`)
3. Std lib imports (e.g., `use std::collections::HashMap`)

**Path Aliases:**
- No path aliases defined in `Cargo.toml`
- Imports reference full crate path: `crate::core::TimeSeries`, `crate::models::Forecaster`
- Re-exports in `mod.rs` files provide convenient namespacing (e.g., `pub use naive::Naive`)
- Prelude module at `crate::prelude` exports most-common types for convenience imports

## Error Handling

**Patterns:**
- Custom error enum: `ForecastError` in `src/error.rs` using `thiserror::Error` derive
- Type alias: `pub type Result<T> = std::result::Result<T, ForecastError>`
- All fallible operations return `Result<T>`
- Error variants include context: `InsufficientData { needed: usize, got: usize, hint: Option<String> }`
- Errors support optional hints for user guidance: `hint: Some("window must be positive".into())`
- Sub-model errors wrap source: `SubModelError { model_name: String, source: Box<ForecastError> }`
- Early validation: `validate_series_complete()` called at start of every `fit()` implementation
- Missing values checked before operations: `series.has_missing_values()` returns `ForecastError::MissingValues`
- Fit-before-predict enforced: Methods return `FitRequired { model: Option<String> }` if called before fit

**Example from `src/error.rs`:**
```rust
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ForecastError {
    #[error("empty input data")]
    EmptyData,
    
    #[error("insufficient data: need at least {needed}, got {got}{}", .hint.as_deref()...)]
    InsufficientData {
        needed: usize,
        got: usize,
        hint: Option<String>,
    },
    
    #[error("sub-model '{model_name}' failed: {source}")]
    SubModelError {
        model_name: String,
        source: Box<ForecastError>,
    },
}
```

## Logging

**Framework:** `println!()` macro (no external logging framework)

**Patterns:**
- Logging used only in tests for diagnostic output (visibility into test behavior)
- Production code avoids logging (library design principle)
- Test diagnostics include intermediate state: penalty values, changepoint locations, model parameters
- Example from `tests/auto_changepoint_test.rs`:
  ```rust
  println!("Auto-detected {} changepoints at {:?}",
      result.result.n_changepoints, result.result.changepoints);
  println!("Selected penalty: {:.2}", result.penalty);
  ```

## Comments

**When to Comment:**
- Complex algorithms receive top-of-file documentation
- Non-obvious parameter bounds explained inline
- Workarounds for language limitations flagged (e.g., WASM constraints)
- Mathematical concepts (eigenvalues, matrix operations) briefly explained
- References to papers or specifications noted

**Doc Comments (///):**
- All public functions documented with `///` before signature
- Doc comments describe purpose, arguments, and typical usage
- Examples provided when usage pattern is non-obvious
- Error conditions documented in doc comment or Error variant description
- Doc examples use `#[doc]` attributes and must be valid Rust (cargo test validates them)

**Example from `src/models/exponential/ses.rs`:**
```rust
/// Simple Exponential Smoothing forecaster.
///
/// The model equation is:
/// `level_t = α × y_t + (1-α) × level_{t-1}`
///
/// where α (alpha) is the smoothing parameter (0 < α < 1).
///
/// # Example
/// ```
/// use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
/// use anofox_forecast::models::Forecaster;
/// // ...
/// let mut model = SimpleExponentialSmoothing::new(0.3);
/// model.fit(&ts).unwrap();
/// ```
```

**Inline Comments (///):**
- Rarely used; code preferred to be self-documenting
- Used sparingly in numerical sections where loop intent may be unclear
- Flag changes needed for WASM or platform-specific concerns

## Function Design

**Size:** 
- Prefer functions under 50 lines
- Numerical/algorithmic sections may reach 100+ lines if coherent block
- Refactor into helpers when sections exceed 80 lines

**Parameters:**
- Models accept TimeSeries for training, horizon for prediction
- Configuration passed via builder pattern (e.g., `AutoETSConfig`) or separate config struct
- Exogenous regressors passed as `HashMap<String, Vec<f64>>` on `predict_with_exog()`
- No default parameters; overloads via method naming (`auto()` vs `new(alpha)`)

**Return Values:**
- `Result<T>` for all fallible operations
- Tuple returns when multiple related values (e.g., `(fitted: Vec<f64>, residuals: Vec<f64>)`)
- Option<T> for computed fields that may not apply (e.g., `trend_component()` → `Result<&[f64]>`)
- Forecasts returned as `Forecast` struct containing `Vec<Vec<f64>>` for dimensions × horizon

## Module Design

**Exports:**
- Each module (`mod.rs`) re-exports public types it defines
- Private implementation details remain in submodules (e.g., `naive.rs` private, exported via `mod.rs`)
- Traits and structs for model composition: `Forecaster`, `ModelSpec`, `BoxedForecaster`

**Barrel Files:**
- Extensive use in `src/models/mod.rs` to group and re-export model families
- Example from `src/models/mod.rs`:
  ```rust
  mod arima;
  mod auto_forecast;
  // ... more modules
  
  pub use arima::ARIMA;
  pub use auto_forecast::AutoForecast;
  pub use baseline::Naive;
  ```
- Prelude at `src/lib.rs::prelude` provides convenient access:
  ```rust
  pub mod prelude {
      pub use crate::core::{Forecast, TimeSeries};
      pub use crate::models::Forecaster;
      pub use crate::utils::{cross_validate, AccuracyMetrics};
  }
  ```

## Documentation Structure

**Module-level:**
- Every module starts with `//!` doc comment describing its purpose
- Lists main types and common use cases
- Example from `src/models/exponential/mod.rs`:
  ```rust
  //! Exponential smoothing models.
  //!
  //! This module provides exponential smoothing forecasting methods:
  //! - Simple Exponential Smoothing (SES)
  //! - Holt's Linear Trend
  //! - Holt-Winters (additive and multiplicative seasonality)
  ```

**Feature Flags:**
- Conditional modules marked with `#[cfg(feature = "...")]`
- Example: `#[cfg(feature = "anomaly")]` guards entire module
- Documented in Cargo.toml with purpose

## Serialization

**Serde:**
- When feature `serde` enabled, models derive `Serialize + Deserialize`
- Custom serializer `persistence::nan_vec` handles NaN values in Vec<f64>
- Exogenous regressor data marked `#[serde(skip)]` to avoid serialization
- Serde dependencies optional: `serde = { version = "1.0", features = ["derive"], optional = true }`

---

*Convention analysis: 2026-08-09*
