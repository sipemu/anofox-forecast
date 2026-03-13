//! VARForecaster: Forecaster trait adapter for the VAR model.
//!
//! Wraps the multivariate VAR model to provide the univariate `Forecaster` interface.
//! When the TimeSeries has regressors, they are included as additional VAR variables
//! (sorted by name for determinism). The first variable's forecasts are returned.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::traits::{validate_series_complete, Forecaster};
use crate::models::var::VAR;
use crate::utils::stats::quantile_normal;

/// Forecaster adapter for the VAR (Vector Autoregression) model.
///
/// Converts the multivariate VAR interface into the univariate `Forecaster` trait.
/// The primary series values are always the first variable. Any regressors present
/// in the `TimeSeries` are included as additional VAR variables, sorted by name
/// for deterministic ordering.
///
/// # Example
/// ```
/// use anofox_forecast::models::var_forecaster::VARForecaster;
/// let model = VARForecaster::new(1); // VAR(1)
/// ```
#[derive(Debug, Clone)]
pub struct VARForecaster {
    var: VAR,
    fitted_values: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    n_obs: usize,
}

impl VARForecaster {
    /// Create a new VARForecaster with the given lag order.
    pub fn new(order: usize) -> Self {
        Self {
            var: VAR::new(order),
            fitted_values: None,
            residuals: None,
            n_obs: 0,
        }
    }
}

impl Forecaster for VARForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;

        let values = series.primary_values().to_vec();
        let n = values.len();
        self.n_obs = n;

        // Build the data matrix: primary values first, then regressors sorted by name.
        let mut data = vec![values];

        if series.has_regressors() {
            let regressors = series.all_regressors();
            let mut names: Vec<String> = regressors.keys().cloned().collect();
            names.sort();
            for name in &names {
                data.push(regressors[name].clone());
            }
        }

        self.var.fit(&data)?;

        // Extract fitted values and residuals for the first variable.
        let p = self.var.order();
        if let Some(fitted_all) = self.var.fitted_values() {
            // VAR fitted values have length n - p. Pad with NaN for the first p values.
            let mut fitted = vec![f64::NAN; p];
            fitted.extend_from_slice(&fitted_all[0]);
            self.fitted_values = Some(fitted);
        }

        if let Some(resid_all) = self.var.residuals() {
            let mut residuals = vec![f64::NAN; p];
            residuals.extend_from_slice(&resid_all[0]);
            self.residuals = Some(residuals);
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.fitted_values.is_none() {
            return Err(ForecastError::FitRequired { model: None });
        }
        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let forecasts = self.var.predict(horizon)?;
        // Return the first variable's forecasts.
        Ok(Forecast::from_values(forecasts[0].clone()))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        if self.fitted_values.is_none() {
            return Err(ForecastError::FitRequired { model: None });
        }
        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let forecasts = self.var.predict(horizon)?;
        let point = &forecasts[0];

        let z = quantile_normal((1.0 + level) / 2.0);

        // Compute residual standard deviation (excluding NaN padding).
        let residuals = self
            .residuals
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let valid_resid: Vec<f64> = residuals
            .iter()
            .copied()
            .filter(|r| r.is_finite())
            .collect();
        let n = valid_resid.len();
        if n == 0 {
            return Ok(Forecast::from_values(point.clone()));
        }
        let mean_r: f64 = valid_resid.iter().sum::<f64>() / n as f64;
        let var_r: f64 = valid_resid
            .iter()
            .map(|r| (r - mean_r).powi(2))
            .sum::<f64>()
            / n as f64;
        let sigma = var_r.sqrt();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        for (h, &pred) in point.iter().enumerate() {
            let width = z * sigma * ((h + 1) as f64).sqrt();
            lower.push(pred - width);
            upper.push(pred + width);
        }

        Ok(Forecast::from_values_with_intervals(
            point.clone(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted_values.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "VARForecaster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::error::ForecastError;

    fn make_var_test_series(n: usize) -> TimeSeries {
        use chrono::{Duration, TimeZone, Utc};
        let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..n).map(|i| base + Duration::days(i as i64)).collect();
        let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn var_forecaster_fit_predict() {
        let ts = make_var_test_series(100);
        let mut model = VARForecaster::new(1);
        model.fit(&ts).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.name(), "VARForecaster");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);

        let fitted = model.fitted_values().unwrap();
        assert_eq!(fitted.len(), 100);
        assert!(fitted[0].is_nan());

        let residuals = model.residuals().unwrap();
        assert_eq!(residuals.len(), 100);
        assert!(residuals[0].is_nan());
    }

    #[test]
    fn var_forecaster_predict_with_intervals() {
        let ts = make_var_test_series(100);
        let mut model = VARForecaster::new(1);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.horizon(), 5);
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let point = forecast.primary();
        for i in 0..5 {
            assert!(lower[i] < point[i]);
            assert!(upper[i] > point[i]);
        }
    }

    #[test]
    fn var_forecaster_requires_fit() {
        let model = VARForecaster::new(1);
        assert!(!model.is_fitted());
        assert!(matches!(
            model.predict(5).unwrap_err(),
            ForecastError::FitRequired { .. }
        ));
    }

    #[test]
    fn var_forecaster_empty_horizon() {
        let ts = make_var_test_series(50);
        let mut model = VARForecaster::new(1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }
}
