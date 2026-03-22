---
name: validation
description: How to cross-validate models, compute accuracy metrics, compare models, and run residual diagnostics
user_invocable: true
---

# Validation in anofox-forecast

## 1. Accuracy Metrics

```rust
use anofox_forecast::utils::metrics::{calculate_metrics, AccuracyMetrics};
use anofox_forecast::utils::metrics::{mae, rmse, smape, wape, mda, coverage};

// Full metrics suite
let metrics = calculate_metrics(&actual, &predicted, Some(12)).unwrap();
println!("MAE={:.2} RMSE={:.2} SMAPE={:.2} MASE={:?} R²={:.3}",
    metrics.mae, metrics.rmse, metrics.smape, metrics.mase, metrics.r_squared);

// Individual metrics
let m = mae(&actual, &predicted);
let r = rmse(&actual, &predicted);
let s = smape(&actual, &predicted);
let w = wape(&actual, &predicted);         // Weighted Absolute Percentage Error
let d = mda(&actual, &predicted);          // Mean Directional Accuracy
let c = coverage(&actual, &lower, &upper); // Interval coverage rate
```

## 2. Cross-Validation

### CvFoldGenerator (low-level fold placement)

```rust
use anofox_forecast::utils::cross_validation::{CvFoldGenerator, CVStrategy, ConstraintViolation};

// Folds are placed backwards from the series end, stepping back by step_size.
// Last fold always covers the series end. n_folds caps how many to take.
// min_initial_window drops early folds with insufficient training data.
let folds = CvFoldGenerator::new()
    .n_folds(5)                               // at most 5 folds
    .horizon(12)                              // test size per fold
    .min_initial_window(50)                   // every fold trains on >= 50 obs
    .step_size(12)                            // step between origins (default = horizon)
    .strategy(CVStrategy::Expanding)          // or Rolling
    .generate(365)
    .unwrap();

// step_size < horizon → overlapping test windows
// step_size > horizon → sparse sampling (gaps between test windows)
// step_size = horizon (default) → contiguous, non-overlapping test sets
```

### CVConfig (high-level CV)

```rust
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};
use anofox_forecast::models::arima::ARIMA;

// Expanding window CV
let config = CVConfig::expanding(50, 12)   // min_initial_window=50, horizon=12
    .with_seasonal_period(12);             // for MASE calculation

let results = cross_validate(&config, &ts, || ARIMA::new(1, 1, 1)).unwrap();

println!("Folds: {}", results.n_folds);
println!("Mean MAE: {:.2} ± {:.2}", results.aggregated.mae, results.aggregated.mae_std);

// Rolling window CV (fixed training size)
let config = CVConfig::rolling(50, 12);
```

### CV with Data Leakage Prevention

```rust
let config = CVConfig::expanding(50, 12)
    .with_gap(1)       // 1-step gap between train_end and test_start
    .with_purge(3)     // remove 3 obs from end of training (train_end = origin - purge)
    .with_embargo(2);  // exclude 2 obs after test from next fold's training

// gap: observations excluded between train and test (prevents lagged feature leakage)
// purge: shrinks training end (prevents label overlap in financial ML)
// embargo: shifts next fold's train_start forward (prevents serial correlation)
```

### Early-Stopping CV

```rust
use anofox_forecast::utils::cross_validation::cross_validate_early_stop;

// Stop when MAE stabilizes within 1% tolerance
let results = cross_validate_early_stop(&config, &ts, || ARIMA::new(1, 1, 1), 0.01).unwrap();
println!("Converged after {} folds", results.n_folds);
```

### Rolling Forecast (Walk-Forward)

```rust
use anofox_forecast::utils::cross_validation::{
    rolling_forecast, RollingForecastConfig,
};

let rf_config = RollingForecastConfig {
    initial_train_size: 50,
    horizon: 1,
    step_size: 1,
    expanding: true,
};

let result = rolling_forecast(&ts, &rf_config, || ARIMA::new(1, 1, 1)).unwrap();
println!("Windows: {}", result.windows.len());
println!("Aggregated MAE: {:.2}", result.aggregated.mae);
```

## 3. Model Comparison

```rust
use anofox_forecast::utils::comparison::{compare_models, ComparisonConfig};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;

let factories: Vec<(&str, Box<dyn Fn() -> Box<dyn Forecaster + Send> + Send + Sync>)> = vec![
    ("Naive", Box::new(|| Box::new(Naive::new()))),
    ("ARIMA(1,1,1)", Box::new(|| Box::new(ARIMA::new(1, 1, 1)))),
    ("AutoETS", Box::new(|| Box::new(AutoETS::new()))),
];

// In-sample only
let config = ComparisonConfig::default().with_horizon(12);
let results = compare_models(&factories, &ts, &config).unwrap();

for r in &results {
    println!("{}: RMSE={:.2}, time={}μs", r.model_name, r.in_sample.rmse, r.fit_time_us);
}

// With cross-validation
let cv_config = CVConfig::expanding(50, 12).with_step_size(6);
let config = ComparisonConfig::default().with_cv(cv_config);
let results = compare_models(&factories, &ts, &config).unwrap();

for r in &results {
    if let Some(cv) = &r.cv_metrics {
        println!("{}: CV RMSE={:.2}", r.model_name, cv.rmse);
    }
}
```

## 4. Residual Diagnostics

```rust
use anofox_forecast::validation::{
    diagnose_residuals, ResidualDiagnostics,
    ModelDiagnostics,
    ljung_box, durbin_watson, jarque_bera,
};

// All-in-one from model
let diag = ModelDiagnostics::from_forecaster(&model, 0.05).unwrap();
println!("{}", diag.summary());
println!("All tests pass: {}", diag.passes_all);

// From residuals directly
let diag = ModelDiagnostics::from_residuals(&residuals, 0.05);

// Individual tests
let lb = ljung_box(&residuals, 10, 2);
println!("Ljung-Box p={:.4}, white noise={}", lb.p_value, lb.is_white_noise(0.05));

let dw = durbin_watson(&residuals);
println!("Durbin-Watson stat={:.3}, type={:?}", dw.statistic, dw.autocorrelation_type);

let jb = jarque_bera(&residuals);
println!("Jarque-Bera p={:.4}", jb.p_value);
```

## 5. Stationarity Tests

```rust
use anofox_forecast::validation::{adf_test, kpss_test, test_stationarity};

// Combined test (recommended)
let (adf, kpss, conclusion) = test_stationarity(&series);
println!("Conclusion: {}", conclusion); // "stationary", "non_stationary", or "inconclusive"

// Individual tests
let adf = adf_test(&series, None);
println!("ADF: stat={:.3}, p={:.4}, stationary={}", adf.statistic, adf.p_value, adf.is_stationary);

let kpss = kpss_test(&series, None);
println!("KPSS: stat={:.3}, p={:.4}", kpss.statistic, kpss.p_value);
```

## 6. Intermittent Demand Diagnostics

```rust
use anofox_forecast::validation::IntermittentDiagnostics;

let diag = IntermittentDiagnostics::from_data(&actual);
println!("Classification: {:?}", diag.classification);  // Smooth, Erratic, Intermittent, Lumpy
println!("ADI={:.2}, CV²={:.2}", diag.adi, diag.cv_squared);
println!("Recommended model: {}", diag.recommended_model());

// With forecast evaluation
let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
println!("Bias: {:.2}", diag.bias);

// With interval evaluation
let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
println!("Coverage: {:.1}%", diag.coverage_rate.unwrap() * 100.0);
```

## 7. Train/Test Split

```rust
use anofox_forecast::utils::cross_validation::{train_test_split, train_test_split_at};

// Split by ratio
let (train, test) = train_test_split(&ts, 0.8);  // 80/20 split

// Split by index
let (train, test) = train_test_split_at(&ts, 100);
```

## Key Rules

- Folds are placed **backwards from the series end** — most recent data is always evaluated.
- `n_folds` is a **cap**, not a target. Fewer folds are returned if `min_initial_window` drops early ones.
- `step_size` defaults to `horizon` (non-overlapping). Set smaller for overlap, larger for sparse.
- `CVConfig::expanding(min_window, horizon)` — training grows each fold, starts at 0.
- `CVConfig::rolling(window_size, horizon)` — fixed-size training window slides forward.
- `gap`: observations excluded between `train_end` and `test_start`.
- `purge`: shrinks `train_end` by N observations (origin = train_end + purge).
- `step_size` steps back from origin, `purge` shortens training. They are independent:
  `step_size` controls fold spacing, `purge` controls how close training gets to the test boundary.
- `compare_models` silently skips models that fail to fit.
- `diagnose_residuals` and `ModelDiagnostics` require the model to have been fit first.
