# ARIMA

**Run:** `cargo run --example arima`

## What this example demonstrates

Covers ARIMA model variants and automatic order selection. Fits ARIMA(1,1,1), pure AR(1), and pure MA(1) models, then uses `AutoARIMA` for stepwise selection. Also shows differencing/integration utilities and model comparison by AIC.

## Sections

1. **ARIMA(1,1,1)** -- Fits the model, prints AR/MA coefficients, intercept, AIC, BIC, and 5-step forecasts.
2. **AR(1) model** -- Fits a pure autoregressive model via `ARIMA::ar(1)`.
3. **MA(1) model** -- Fits a pure moving-average model via `ARIMA::ma(1)`.
4. **AutoARIMA (stepwise)** -- Automatically selects the best (p,d,q) order; prints the top 5 models by AIC and 10-step forecasts with 95% confidence intervals.
5. **AutoARIMA model scores** -- Lists all evaluated models and their AIC scores.
6. **Differencing utilities** -- Demonstrates `difference` and `integrate` functions for manual differencing and reversal.
7. **Model comparison** -- Sorts ARIMA(1,1,1), AR(1), and MA(1) by AIC.

## Key types

- `ARIMA` -- ARIMA(p,d,q) model with convenience constructors `ar`, `ma`
- `AutoARIMA` -- stepwise automatic order selection
- `Forecaster` -- trait for `fit`, `predict`, `predict_with_intervals`
- `difference`, `integrate` -- differencing and integration utility functions
