---
name: exogenous
description: How to use exogenous regressors and FeatureGenerator with forecasting models
user_invocable: true
---

# Exogenous Regressors in anofox-forecast

All major models support exogenous regressors via OLS pre-regression:
ARIMA, AutoARIMA, SARIMA, ETS, AutoETS, Theta, AutoTheta, MSTL, MFLES, Naive.

## 1. FeatureGenerator — Create Deterministic Regressors

```rust
use anofox_forecast::features::FeatureGenerator;

let gen = FeatureGenerator::new()
    .fourier(7, 2)       // Weekly Fourier terms (4 cols: sin1, cos1, sin2, cos2)
    .fourier(365, 3)     // Annual Fourier terms (6 cols)
    .day_of_week()       // Mon-Sat indicators (6 cols, Sunday = baseline)
    .month_of_year()     // Feb-Dec indicators (11 cols, January = baseline)
    .quarter()           // Q2-Q4 indicators (3 cols, Q1 = baseline)
    .holiday("promo", promo_dates);  // Binary indicator (1 col)

// Generate features from timestamps
let features: HashMap<String, Vec<f64>> = gen.generate(&timestamps);

// Get sorted column names
let names: Vec<String> = gen.feature_names();
```

## 2. Attach to TimeSeries

```rust
use anofox_forecast::core::{CalendarAnnotations, TimeSeries};

let mut ts = TimeSeries::univariate(timestamps, values).unwrap();

// Method A: Use FeatureGenerator (attaches all generated features)
gen.add_to(&mut ts);

// Method B: Manual attachment via CalendarAnnotations
let cal = CalendarAnnotations::new()
    .with_regressor("temperature".to_string(), temp_values)
    .with_regressor("price".to_string(), price_values);
ts.set_calendar(cal);

// Method C: Combine both (add_to preserves existing regressors)
gen.add_to(&mut ts);  // adds Fourier on top of temperature & price
```

## 3. Fit and Predict with Exog

```rust
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;

// Fit — model automatically picks up regressors from TimeSeries
let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

assert!(model.has_exog());
assert!(model.exog_names().is_some());

// predict() will ERROR if model was fit with exog
// Must use predict_with_exog() with future regressor values
let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
future_regs.insert("temperature".to_string(), vec![25.0; horizon]);
future_regs.insert("price".to_string(), vec![9.99; horizon]);

let forecast = model.predict_with_exog(horizon, &future_regs).unwrap();

// With intervals:
let fc = model.predict_with_exog_intervals(horizon, &future_regs, 0.95).unwrap();
```

## 4. Generate Future Features with FeatureGenerator

For Fourier terms, the index must continue from training:

```rust
let n_train = 365;
let horizon = 14;

// Generate for full range (train + future), then slice
let all_timestamps = daily_timestamps(n_train + horizon);
let all_features = gen.generate(&all_timestamps);

let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
for (name, vals) in &all_features {
    future_regs.insert(name.clone(), vals[n_train..].to_vec());
}

// Add external regressors for the future period
future_regs.insert("temperature".to_string(), future_temp);
```

## 5. Extract OLS Coefficients

After fitting with exogenous regressors, extract the OLS pre-regression coefficients:

```rust
use anofox_forecast::utils::OLSResult;

model.fit(&ts).unwrap();

if let Some(ols) = model.exog_coefficients() {
    println!("Intercept: {:.4}", ols.intercept);
    for (name, coef) in ols.regressor_names.iter().zip(&ols.coefficients) {
        println!("  {}: {:.4}", name, coef);
    }

    // Predict exog contribution for new data
    let exog_effect = ols.predict(&future_regs).unwrap();
}
```

`OLSResult` fields:
- `intercept: f64` — regression intercept
- `coefficients: Vec<f64>` — one per regressor (sorted by name)
- `regressor_names: Vec<String>` — names in sorted order

Available on all exog-supporting models: ARIMA, SARIMA, AutoARIMA, ETS, AutoETS, Theta, AutoTheta, OptimizedTheta, DynamicTheta, MSTL, MFLES, Naive, and Pipeline.

## 6. Scenario Analysis

Compare forecasts under different regressor scenarios:

```rust
// Scenario: high vs low temperature
let mut future_hot = future_regs.clone();
future_hot.insert("temperature".to_string(), vec![35.0; horizon]);

let mut future_cold = future_regs.clone();
future_cold.insert("temperature".to_string(), vec![5.0; horizon]);

let fc_hot = model.predict_with_exog(horizon, &future_hot).unwrap();
let fc_cold = model.predict_with_exog(horizon, &future_cold).unwrap();
// Difference ≈ coefficient × (35 - 5)

// Scenario: promotion on/off
let mut promo_on = future_regs.clone();
let mut promo_col = vec![0.0; horizon];
promo_col[7] = 1.0;  // promo on day 8
promo_on.insert("holiday_promo".to_string(), promo_col);

let mut promo_off = future_regs.clone();
promo_off.insert("holiday_promo".to_string(), vec![0.0; horizon]);

let uplift = fc_on.primary()[7] - fc_off.primary()[7];
// uplift ≈ promotion coefficient
```

## 7. Reuse Generator Across Models

```rust
let gen = FeatureGenerator::new().fourier(12, 3).month_of_year();

// Same features for all models
for mut model in [arima, ets, theta, mstl] {
    let mut ts_clone = ts.clone();
    gen.add_to(&mut ts_clone);
    model.fit(&ts_clone).unwrap();
    let fc = model.predict_with_exog(h, &future_regs).unwrap();
}
```

## Key Rules

- `predict()` **errors** when model was fit with exog. Always use `predict_with_exog()`.
- Future regressors must have **exactly `horizon` values** per regressor.
- All regressor names from fitting must be present in future regressors.
- Don't combine Fourier and DOW for the same period (collinearity).
- FeatureGenerator features are deterministic (timestamp-only) — safe for CV.
