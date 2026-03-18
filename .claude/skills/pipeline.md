---
name: pipeline
description: How to chain reversible transforms (BoxCox, Difference, Scale, Log) around any Forecaster using Pipeline
user_invocable: true
---

# Transform Pipeline in anofox-forecast

Chain reversible transforms around any `Forecaster`. The `Pipeline` itself implements `Forecaster`, making it composable with cross-validation, ensembles, and batch operations.

## 1. Available Transforms

```rust
use anofox_forecast::transform::{
    BoxCoxTransform, DifferenceTransform, SeasonalDifferenceTransform,
    LogTransform, ScaleTransform, ScaleMethod,
    Pipeline,
};

// Power transform (stabilize variance)
BoxCoxTransform::auto()              // Auto-select lambda via MLE
BoxCoxTransform::with_lambda(0.5)    // Fixed lambda (0.5 = sqrt)

// Differencing (remove trend/seasonality)
DifferenceTransform::new(1)          // First-order difference (d=1)
DifferenceTransform::new(2)          // Second-order difference
SeasonalDifferenceTransform::new(12) // Seasonal difference (period=12)

// Scaling (normalize magnitudes)
ScaleTransform::new(ScaleMethod::Standardize)  // z-score: (x - mean) / std
ScaleTransform::new(ScaleMethod::Normalize)    // min-max: [0, 1]
ScaleTransform::new(ScaleMethod::RobustScale)  // (x - median) / IQR

// Log transform (handles non-positive with auto shift)
LogTransform::new()  // ln(x + shift), shift auto-computed if min <= 0
```

## 2. Build a Pipeline

```rust
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::Forecaster;

let mut pipeline = Pipeline::builder()
    .transform(BoxCoxTransform::auto())       // 1st: stabilize variance
    .transform(DifferenceTransform::new(1))   // 2nd: remove trend
    .transform(ScaleTransform::new(ScaleMethod::Standardize)) // 3rd: normalize
    .model(Box::new(Naive::new()))            // inner model
    .build();
```

Transforms apply in order during `fit()`, inverse in reverse during `predict()`.

## 3. Fit and Predict

```rust
// Pipeline implements Forecaster — use it like any model
pipeline.fit(&ts).unwrap();

// Point forecast (auto inverse-transformed to original scale)
let forecast = pipeline.predict(12).unwrap();

// With prediction intervals
let fc = pipeline.predict_with_intervals(12, 0.95).unwrap();
let lower = fc.lower_series(0).unwrap();
let upper = fc.upper_series(0).unwrap();

// Fitted values and residuals (on original scale)
let fitted = pipeline.fitted_values();
let residuals = pipeline.residuals();
```

## 4. Common Recipes

```rust
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::arima::ARIMA;

// Heteroscedastic data: BoxCox → AutoETS
let mut p = Pipeline::builder()
    .transform(BoxCoxTransform::auto())
    .model(Box::new(AutoETS::new()))
    .build();

// Exponential growth: Log → Difference → model
let mut p = Pipeline::builder()
    .transform(LogTransform::new())
    .transform(DifferenceTransform::new(1))
    .model(Box::new(ARIMA::new(1, 0, 1)))
    .build();

// High-variance data: Scale → model
let mut p = Pipeline::builder()
    .transform(ScaleTransform::new(ScaleMethod::Standardize))
    .model(Box::new(ARIMA::new(1, 1, 1)))
    .build();
```

## 5. Use with Cross-Validation and Comparison

```rust
use anofox_forecast::models::ModelSpec;
use anofox_forecast::utils::comparison::{compare_models, ComparisonConfig};

// Pipeline works anywhere a Forecaster is expected
let factories: Vec<(&str, Box<dyn Fn() -> Box<dyn Forecaster + Send> + Send + Sync>)> = vec![
    ("BoxCox+Naive", Box::new(|| {
        Box::new(Pipeline::builder()
            .transform(BoxCoxTransform::auto())
            .model(Box::new(Naive::new()))
            .build())
    })),
    ("Log+Diff+ARIMA", Box::new(|| {
        Box::new(Pipeline::builder()
            .transform(LogTransform::new())
            .transform(DifferenceTransform::new(1))
            .model(Box::new(ARIMA::new(1, 0, 1)))
            .build())
    })),
];

let results = compare_models(&factories, &ts, &ComparisonConfig::default()).unwrap();
```

## 6. Exogenous Regressors through Pipeline

Pipeline delegates exog support to the inner model:

```rust
pipeline.fit(&ts_with_regressors).unwrap();
assert!(pipeline.has_exog());

let fc = pipeline.predict_with_exog(12, &future_regressors).unwrap();
```

## Key Rules

- Transforms apply **left-to-right** during fit, **right-to-left** during predict.
- `BoxCoxTransform::auto()` requires all positive values. Use `LogTransform` for data with zeros (auto-shifts).
- `DifferenceTransform` consumes `d` observations from the front. `SeasonalDifferenceTransform` consumes `period` observations.
- `Pipeline::builder().build()` **panics** if no model is provided.
- Length-preserving transforms (BoxCox, Scale, Log) have `offset() = 0`. Differencing has `offset() = d` or `period`.
