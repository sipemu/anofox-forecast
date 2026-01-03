//! Naive forecasting model.
//!
//! The naive method simply forecasts the last observed value for all future periods.
//!
//! Supports exogenous regressors via TimeSeries.regressors.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use std::collections::HashMap;

/// Naive forecaster that repeats the last value.
///
/// Supports exogenous regressors via TimeSeries.regressors.
#[derive(Debug, Clone, Default)]
pub struct Naive {
    last_value: Option<f64>,
    fitted: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    history: Option<Vec<f64>>,
    /// OLS result for exogenous regressors (if any).
    exog_ols: Option<OLSResult>,
}

impl Naive {
    pub fn new() -> Self {
        Self::default()
    }

    /// Internal prediction method that handles both with and without exogenous cases.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate exogenous contribution if applicable
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            let future = future_regressors.ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors. Future regressor values required."
                        .into(),
                )
            })?;

            // Validate future regressors have correct length
            for name in &ols.regressor_names {
                let values = future.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            // Predict exogenous contribution
            Some(ols.predict(future)?)
        } else {
            if future_regressors.is_some_and(|r| !r.is_empty()) {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            None
        };

        let mut predictions = Vec::with_capacity(horizon);
        for h in 0..horizon {
            let mut pred = last;
            if let Some(ref exog) = exog_contribution {
                pred += exog[h];
            }
            predictions.push(pred);
        }

        Ok(Forecast::from_values(predictions))
    }
}

impl Forecaster for Naive {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let raw_values = series.primary_values();
        if raw_values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        // Check for exogenous regressors
        let values: Vec<f64> = if series.has_regressors() {
            // Extract regressors from TimeSeries
            let regressors = series.all_regressors();

            // Fit OLS: y ~ X
            let ols_result = ols_fit(raw_values, &regressors)?;

            // Calculate residuals (y - OLS prediction)
            let adjusted = ols_residuals(raw_values, &ols_result, &regressors)?;

            // Store OLS result for prediction
            self.exog_ols = Some(ols_result);

            adjusted
        } else {
            self.exog_ols = None;
            raw_values.to_vec()
        };

        self.last_value = Some(*values.last().unwrap());
        self.history = Some(values.clone());

        // Fitted values are shifted history (y_hat[t] = y[t-1])
        let mut fitted = Vec::with_capacity(values.len());
        fitted.push(f64::NAN); // First fitted value is undefined
        fitted.extend_from_slice(&values[..values.len() - 1]);
        self.fitted = Some(fitted.clone());

        // Residuals are first differences (y[t] - y[t-1])
        let residuals: Vec<f64> = (0..values.len())
            .map(|i| {
                if i == 0 {
                    f64::NAN
                } else {
                    values[i] - values[i - 1]
                }
            })
            .collect();
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        // If model was fit with exogenous regressors, require predict_with_exog
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }

        self.predict_internal(horizon, None)
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        self.exog_ols.is_some()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.exog_ols
            .as_ref()
            .map(|ols| ols.regressor_names.as_slice())
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        self.predict_internal(horizon, Some(future_regressors))
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;
        let _history = self.history.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate exogenous contribution
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            // Validate future regressors
            for name in &ols.regressor_names {
                let values = future_regressors.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }
            Some(ols.predict(future_regressors)?)
        } else {
            None
        };

        // Calculate residual standard deviation for confidence intervals
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = if valid_residuals.is_empty() {
            0.0
        } else {
            let n = valid_residuals.len() as f64;
            let variance = crate::simd::sum_of_squares(&valid_residuals) / n;
            variance.sqrt()
        };

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let mut pred = last;
            if let Some(ref exog) = exog_contribution {
                pred += exog[h - 1];
            }
            predictions.push(pred);

            let se = sigma * (h as f64).sqrt();
            lower.push(pred - z * se);
            upper.push(pred + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            predictions,
            lower,
            upper,
        ))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;
        let _history = self.history.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate residual standard deviation for confidence intervals
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        if valid_residuals.is_empty() {
            let predictions = vec![last; horizon];
            return Ok(Forecast::from_values(predictions));
        }

        let n = valid_residuals.len() as f64;
        let variance = crate::simd::sum_of_squares(&valid_residuals) / n;
        let sigma = variance.sqrt();

        // Z-score for the confidence level
        let z = quantile_normal((1.0 + level) / 2.0);

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            predictions.push(last);
            // Confidence interval widens with sqrt(horizon)
            let se = sigma * (h as f64).sqrt();
            lower.push(last - z * se);
            upper.push(last + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            predictions,
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let residuals = self.residuals.as_ref()?;

        // Filter out NaN residuals
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        if valid_residuals.is_empty() {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let n = valid_residuals.len() as f64;
        let variance = crate::simd::sum_of_squares(&valid_residuals) / n;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = variance.sqrt();

        let lower: Vec<f64> = fitted.iter().map(|&f| f - z * sigma).collect();
        let upper: Vec<f64> = fitted.iter().map(|&f| f + z * sigma).collect();

        Some(Forecast::from_values_with_intervals(
            fitted.clone(),
            lower,
            upper,
        ))
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "Naive"
    }
}

/// Approximate quantile function for standard normal distribution.
fn quantile_normal(p: f64) -> f64 {
    // Approximation using Abramowitz and Stegun formula
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i as u32, 0, 0).unwrap())
            .collect()
    }

    #[test]
    fn naive_repeats_last_value() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.primary(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn naive_fitted_values_are_shifted_history() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        assert!(fitted[0].is_nan());
        assert_eq!(&fitted[1..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn naive_residuals_are_first_differences() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let residuals = model.residuals().unwrap();
        assert!(residuals[0].is_nan());
        assert_eq!(&residuals[1..], &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn naive_confidence_intervals_widen_with_horizon() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10)
            .map(|i| (i as f64) + 0.1 * (i as f64).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        // Interval should widen with horizon
        for i in 1..5 {
            let width_prev = upper[i - 1] - lower[i - 1];
            let width_curr = upper[i] - lower[i];
            assert!(
                width_curr > width_prev,
                "Interval at h={} should be wider than h={}",
                i + 1,
                i
            );
        }
    }

    #[test]
    fn naive_handles_empty_data() {
        let ts = TimeSeries::univariate(vec![], vec![]).unwrap();
        let mut model = Naive::new();

        assert!(matches!(model.fit(&ts), Err(ForecastError::EmptyData)));
    }

    #[test]
    fn naive_zero_horizon_returns_empty() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert!(forecast.is_empty());
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn naive_requires_fit_before_predict() {
        let model = Naive::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn naive_name_is_correct() {
        let model = Naive::new();
        assert_eq!(model.name(), "Naive");
    }
}
