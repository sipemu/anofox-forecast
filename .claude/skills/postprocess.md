---
name: postprocess
description: How to generate prediction intervals and calibrated quantile forecasts using conformal prediction and other postprocessing methods
user_invocable: true
---

# Postprocessing in anofox-forecast

Model-agnostic uncertainty quantification: turn any point forecast into calibrated prediction intervals or quantile forecasts.

## 1. Unified PostProcessor (Recommended)

```rust
use anofox_forecast::postprocess::{
    PostProcessor, PostModel, ConformalMethod,
    PointForecasts, PredictionIntervals, QuantileForecasts,
};

// Step 1: Choose a method
let processor = PostProcessor::conformal(0.90);        // 90% conformal intervals
let processor = PostProcessor::historical_sim(           // Empirical quantiles
    vec![0.05, 0.25, 0.5, 0.75, 0.95],
);
let processor = PostProcessor::normal(                   // Gaussian assumption
    vec![0.05, 0.25, 0.5, 0.75, 0.95],
);
let processor = PostProcessor::idr(                      // Isotonic distributional regression
    vec![0.05, 0.25, 0.5, 0.75, 0.95],
);

// Step 2: Train on historical forecasts vs actuals
let train_forecasts = PointForecasts::from_values(fitted_values.to_vec());
let trained = processor.train(&train_forecasts, &actual_values).unwrap();

// Step 3: Generate intervals or quantiles for new forecasts
let new_forecasts = PointForecasts::from_values(point_predictions.to_vec());
let intervals: PredictionIntervals = processor.predict_intervals(&trained, &new_forecasts).unwrap();
let quantiles: QuantileForecasts = processor.predict_quantiles(&trained, &new_forecasts).unwrap();

// Or train + predict in one call
let quantiles = processor.point_to_quantiles(
    &train_forecasts, &actual_values, &new_forecasts,
).unwrap();
```

## 2. Conformal Prediction (Distribution-Free Guarantees)

```rust
use anofox_forecast::postprocess::{ConformalPredictor, ConformalMethod};

// Split conformal (holdout calibration set)
let cp = ConformalPredictor::split(0.90);

// Cross-validation conformal
let cp = ConformalPredictor::cross_val(0.90, 5);

// Jackknife+ (leave-one-out, finite-sample valid)
let cp = ConformalPredictor::jackknife_plus(0.90);

// Fit and predict
let result = cp.fit(&train_forecasts, &train_actuals).unwrap();
let intervals = cp.predict(&result, &new_forecasts).unwrap();

println!("Interval width: {:.2}", result.quantile_value() * 2.0);
```

## 3. Per-Horizon-Step Conformal (Growing Intervals)

```rust
use anofox_forecast::postprocess::{ConformalPredictor, PerStepConformalResult};

// For multi-step forecasting where error grows with horizon
let cp = ConformalPredictor::split(0.90);

// fold_forecasts[i] and fold_actuals[i] are Vec<f64> of length `horizon`
// from cross-validation or rolling-origin backtesting
let result: PerStepConformalResult = cp.fit_per_step(
    &fold_forecasts,  // &[Vec<f64>], one per fold
    &fold_actuals,    // &[Vec<f64>], one per fold
).unwrap();

// Each step gets its own half-width: h=1 is tighter than h=12
println!("Half-widths: {:?}", result.half_widths());

// Apply to a point forecast
let (lower, upper) = result.predict(&point_forecast);

// Or get PredictionIntervals directly
let intervals = result.predict_intervals(&point_forecast);
```

Works with all methods: `split()`, `cross_val()`, `jackknife_plus()`. Falls back to pooled quantile when a step has < 2 residuals.

## 4. Historical Simulation (Non-Parametric Quantiles)

```rust
use anofox_forecast::postprocess::HistoricalSimulator;

// Full error distribution
let sim = HistoricalSimulator::new(vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]);

// Rolling window variant (for non-stationary errors)
let sim = HistoricalSimulator::with_window(
    vec![0.05, 0.50, 0.95],
    50,  // use last 50 errors only
);

let result = sim.fit(&train_forecasts, &train_actuals).unwrap();
let quantiles = sim.predict(&result, &new_forecasts).unwrap();

// Access specific quantile predictions
let median = quantiles.median();          // 50th percentile
let q05 = quantiles.at_quantile(0);       // 5th percentile values
let at_t3 = quantiles.at_time(3);         // all quantiles at time step 3
```

## 5. Quantile Regression Averaging (Ensemble)

```rust
use anofox_forecast::postprocess::{QRAPredictor, QRARegularization};

// Combine multiple point forecasters
let qra = QRAPredictor::new(
    vec![0.05, 0.25, 0.50, 0.75, 0.95],
    QRARegularization::LassoCV { n_folds: 5 },
);

// forecasts_matrix: [n_observations][n_forecasters]
let forecasts_matrix = vec![
    vec![model1_pred[0], model2_pred[0], model3_pred[0]],
    vec![model1_pred[1], model2_pred[1], model3_pred[1]],
    // ...
];

let result = qra.fit(&forecasts_matrix, &actuals).unwrap();
let quantiles = qra.predict(&result, &new_forecasts_matrix).unwrap();
```

## 6. Binned Conformal (Heteroscedastic Intervals)

```rust
use anofox_forecast::postprocess::BinnedConformalPredictor;

// Wider intervals for larger forecasts, narrower for smaller
let bcp = BinnedConformalPredictor::new(0.90, 5);  // 5 bins
let bcp = BinnedConformalPredictor::default_bins(0.90); // 3 bins

let result = bcp.fit(&train_forecasts, &train_actuals).unwrap();
// result has per-bin quantile values
```

## 7. Recalibrate Existing Quantile Forecasts

```rust
use anofox_forecast::postprocess::conformalize;

// Fix underdispersed or overdispersed quantile forecasts
let recalibrated = conformalize(
    &original_quantile_forecasts,
    &calibration_forecasts,
    &calibration_actuals,
).unwrap();

println!("Adjustments per quantile: {:?}", recalibrated.adjustments());
let fixed_forecasts = recalibrated.into_forecasts();
```

## 8. Evaluate Intervals

```rust
use anofox_forecast::utils::metrics::{coverage, msis};

// Empirical coverage (should be close to nominal level)
let cov = coverage(&test_actuals, intervals.lower(), intervals.upper());
println!("Coverage: {:.1}% (target: 90%)", cov * 100.0);

// Mean Scaled Interval Score (lower = better)
let score = msis(&test_actuals, intervals.lower(), intervals.upper(), 0.10);
println!("MSIS: {:.2}", score);

// Interval widths
let widths = intervals.widths();
println!("Mean width: {:.2}", widths.iter().sum::<f64>() / widths.len() as f64);
```

## 9. Typical Workflow

```rust
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use anofox_forecast::postprocess::{PostProcessor, PointForecasts};

// 1. Fit model
let mut model = ARIMA::new(1, 1, 1);
model.fit(&train_ts).unwrap();

// 2. Get fitted values as calibration data
let fitted = model.fitted_values().unwrap();
let actual = train_ts.primary_values();

// 3. Train postprocessor on in-sample residuals
let processor = PostProcessor::conformal(0.90);
let pf = PointForecasts::from_values(fitted.to_vec());
let trained = processor.train(&pf, actual).unwrap();

// 4. Generate point forecast
let fc = model.predict(12).unwrap();

// 5. Wrap in calibrated intervals
let new_pf = PointForecasts::from_values(fc.primary().to_vec());
let intervals = processor.predict_intervals(&trained, &new_pf).unwrap();
```

## Key Rules

- **Conformal prediction** provides distribution-free coverage guarantees — use when correctness matters.
- **Historical simulation** uses empirical error distribution — good for heavy-tailed or skewed errors.
- **Normal predictor** assumes Gaussian errors — fastest but may undercover for non-normal errors.
- **IDR** learns monotone conditional quantiles — best calibration but needs more data.
- **QRA** combines multiple forecasters — use when you have an ensemble.
- **Binned conformal** produces wider intervals where errors are larger — use for heteroscedastic data.
- All methods require a calibration set: historical `(forecast, actual)` pairs.
- `PointForecasts::from_values()` creates forecasts without timestamps.
- `QuantileForecasts::to_prediction_intervals(coverage)` converts quantiles to intervals.
