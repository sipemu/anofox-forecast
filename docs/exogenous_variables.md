# Exogenous Variable Support

This document describes the exogenous variable (external regressor) support in the Rust forecasting library, including the implementation approach and validation against Nixtla statsforecast.

## Overview

Exogenous variables (also called external regressors) are additional predictor variables that can improve forecast accuracy when they have a known relationship with the target series. Common examples include:

- Price effects on demand
- Promotional indicators
- Weather variables
- Holiday effects
- Economic indicators

## Supported Models

### Matching Nixtla statsforecast

| Model | Exogenous Support | Implementation |
|-------|-------------------|----------------|
| ARIMA | Full (ARIMAX) | OLS regression + ARIMA on residuals |
| SARIMA | Full (SARIMAX) | OLS regression + SARIMA on residuals |
| AutoARIMA | Full | Passes through to selected model |
| MFLES | Full | OLS regression + MFLES on residuals |

### Rust Extensions (beyond Nixtla)

The following models have exogenous support in this Rust library but **NOT in Nixtla statsforecast**:

| Model | Exogenous Support | Implementation |
|-------|-------------------|----------------|
| Theta | Full | OLS regression + Theta on residuals |
| AutoTheta | Full | Passes through to selected model |
| Naive | Basic | OLS regression overlay |

**How Naive+exogenous works:** Since Naive simply forecasts the last value, adding exogenous support means:
1. Fit OLS: `y = β₀ + β₁x₁ + β₂x₂ + ... + ε`
2. Compute residuals: `y_adj = y - X @ β`
3. Apply Naive: forecast the last residual value for all horizons
4. Add exogenous contribution: `forecast = last_residual + X_future @ β`

This allows exogenous regressors to shift the otherwise flat Naive forecast based on future regressor values.

**Models WITHOUT exogenous support** (same as Nixtla):
- ETS/AutoETS (classical formulation doesn't include exogenous)
- TBATS
- GARCH/ARCH
- Croston and intermittent demand models

## API Usage

### Adding Regressors to TimeSeries

```rust
use anofox_forecast::core::{CalendarAnnotations, TimeSeries};

// Create regressors
let price: Vec<f64> = vec![10.0, 12.0, 11.0, ...];
let promo: Vec<f64> = vec![0.0, 1.0, 0.0, ...];

// Add to calendar annotations
let calendar = CalendarAnnotations::new()
    .with_regressor("price".to_string(), price)
    .with_regressor("promo".to_string(), promo);

// Create time series with regressors
let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
ts.set_calendar(calendar);
```

### Fitting Models

Models automatically detect and use regressors from the TimeSeries:

```rust
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;

let mut model = ARIMA::new(1, 1, 1);
model.fit(&ts).unwrap();

// Check if model used exogenous regressors
assert!(model.supports_exog());  // Model capability
assert!(model.has_exog());       // Model was fit with regressors
println!("Regressors: {:?}", model.exog_names());
```

### Forecasting with Future Regressors

When a model is fit with exogenous regressors, you must provide future values:

```rust
use std::collections::HashMap;

// Create future regressor values (must match horizon length)
let mut future_regressors = HashMap::new();
future_regressors.insert("price".to_string(), vec![11.0, 10.5, 12.0, ...]);
future_regressors.insert("promo".to_string(), vec![0.0, 0.0, 1.0, ...]);

// Forecast
let forecast = model.predict_with_exog(horizon, &future_regressors).unwrap();

// With confidence intervals
let forecast = model.predict_with_exog_intervals(horizon, &future_regressors, 0.95).unwrap();
```

### Error Handling

```rust
// predict() fails if model was fit with regressors
let result = model.predict(horizon);
assert!(result.is_err());  // Must use predict_with_exog

// Missing regressor causes error
let mut incomplete = HashMap::new();
incomplete.insert("price".to_string(), vec![11.0; horizon]);
// Missing "promo" regressor
let result = model.predict_with_exog(horizon, &incomplete);
assert!(result.is_err());

// Wrong length causes error
let mut wrong_len = HashMap::new();
wrong_len.insert("price".to_string(), vec![11.0, 10.5]);  // Only 2 values
wrong_len.insert("promo".to_string(), vec![0.0; horizon]);
let result = model.predict_with_exog(horizon, &wrong_len);
assert!(result.is_err());
```

## Implementation Approach

### Nixtla statsforecast Pattern

Following Nixtla's approach, exogenous variables are handled using OLS regression:

1. **Fit Phase:**
   - Fit OLS: `y = β₀ + β₁x₁ + β₂x₂ + ... + ε`
   - Store coefficients β
   - Compute adjusted series: `y_adj = y - X @ β`
   - Fit the base model (ARIMA, MFLES, etc.) on `y_adj`

2. **Predict Phase:**
   - Get base model forecast on adjusted scale
   - Add back exogenous contribution: `forecast + X_future @ β`

### Code Structure

```
src/utils/ols.rs          # OLS regression utilities
  - OLSResult             # Coefficients and regressor names
  - ols_fit()             # Fit OLS using Cholesky decomposition
  - ols_residuals()       # Compute residuals

src/models/traits.rs      # Extended Forecaster trait
  - supports_exog()       # Check model capability
  - has_exog()            # Check if fit with exogenous
  - exog_names()          # Get regressor names
  - predict_with_exog()   # Predict with future values
  - predict_with_exog_intervals()  # With confidence intervals

src/models/arima/model.rs # ARIMA/SARIMA with exog_ols field
src/models/mfles.rs       # MFLES with exog_ols field
src/models/baseline/naive.rs  # Naive with exog_ols field
```

## Validation Against Nixtla statsforecast

### Validation Approach

The implementation is validated against Nixtla statsforecast using:

1. **Synthetic Data Tests:** Generate data with known exogenous effects and verify the model recovers the correct coefficients and forecasts.

2. **Reference Comparison:** Python script generates reference outputs from statsforecast for direct comparison.

3. **Integration Tests:** Comprehensive tests verify:
   - Models correctly detect and use regressors
   - Forecasts change appropriately with different future regressor values
   - Error handling for missing/mismatched regressors
   - Confidence intervals work with exogenous

### Running Validation

**Generate Nixtla Reference (requires Python):**
```bash
pip install statsforecast pandas numpy
python scripts/generate_nixtla_reference.py
```

This creates reference files in `tests/reference/`:
- `test_data_exog.json` - Test data with known coefficients
- `arima_exog_reference.json` - AutoARIMA forecasts
- `mfles_exog_reference.json` - MFLES forecasts
- `naive_exog_reference.json` - Naive forecasts

**Run Rust Integration Tests:**
```bash
cargo test --test exog_integration
```

### Validation Criteria

| Metric | Tolerance |
|--------|-----------|
| OLS coefficients | < 1% relative error |
| Point forecasts | < 2% relative error |
| Confidence intervals | Must overlap |

### Test Results

**Validation against Nixtla statsforecast:**

| Model | MAPE | RMSE | Status |
|-------|------|------|--------|
| AutoARIMA | 0.03% | 0.0169 | ✅ Nearly identical |
| MFLES | 0.54% | 0.3817 | ✅ Very close |
| Forecast Direction | 100% | - | ✅ Perfect match |

**Integration tests (14 tests pass):**
- `arima_with_exogenous_basic`
- `sarima_with_exogenous_basic`
- `auto_arima_with_exogenous_basic`
- `mfles_with_exogenous_basic`
- `naive_with_exogenous_basic`
- `theta_with_exogenous_basic`
- `auto_theta_with_exogenous_basic`
- `theta_without_exogenous_still_works`
- `theta_exog_intervals_work`
- `arima_without_exogenous_still_works`
- `missing_future_regressor_errors`
- `wrong_regressor_length_errors`
- `exog_intervals_work`
- `exogenous_effect_visible_in_forecast`

**Note:** Naive and Theta exogenous support cannot be validated against Nixtla because Nixtla doesn't support exogenous for these models. These are Rust extensions beyond Nixtla's functionality.

## Comparison with Nixtla statsforecast

### API Comparison

| Feature | Nixtla (Python) | Rust |
|---------|-----------------|------|
| Regressor input | DataFrame column | `CalendarAnnotations.with_regressor()` |
| Future regressors | `X_df` parameter | `HashMap<String, Vec<f64>>` |
| Check for exog | N/A | `model.has_exog()` |
| Get regressor names | N/A | `model.exog_names()` |

### Nixtla Python Example

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Data with exogenous
df = pd.DataFrame({
    'unique_id': ['series1'] * n,
    'ds': dates,
    'y': values,
    'price': price_values,
    'promo': promo_values
})

# Future exogenous
X_df = pd.DataFrame({
    'unique_id': ['series1'] * horizon,
    'ds': future_dates,
    'price': future_price,
    'promo': future_promo
})

# Fit and forecast
sf = StatsForecast(models=[AutoARIMA()], freq='D')
sf.fit(df)
forecast = sf.predict(h=horizon, X_df=X_df)
```

### Equivalent Rust Example

```rust
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::Forecaster;
use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use std::collections::HashMap;

// Data with exogenous
let calendar = CalendarAnnotations::new()
    .with_regressor("price".to_string(), price_values)
    .with_regressor("promo".to_string(), promo_values);

let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
ts.set_calendar(calendar);

// Future exogenous
let mut future = HashMap::new();
future.insert("price".to_string(), future_price);
future.insert("promo".to_string(), future_promo);

// Fit and forecast
let mut model = AutoARIMA::new();
model.fit(&ts).unwrap();
let forecast = model.predict_with_exog(horizon, &future).unwrap();
```

## Limitations

1. **Linear relationship only:** Exogenous effects are modeled as linear (OLS). Non-linear relationships require feature engineering.

2. **No automatic lag detection:** Users must align regressors with the target series manually.

3. **Future values required:** Unlike some ML approaches, exogenous values must be known or forecasted for the prediction horizon.

4. **No categorical encoding:** Categorical variables must be one-hot encoded before use.

## Future Enhancements

- [ ] Add remaining baseline models (SeasonalNaive, HistoricAverage, etc.)
- [ ] Support for interaction terms
- [ ] Automatic regressor selection
- [ ] Regularization options (Ridge, Lasso)
- [ ] Handle missing regressor values
