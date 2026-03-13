//! KalmanForecaster: Forecaster trait adapter for the Kalman filter.
//!
//! Wraps a Kalman filter with a state-space model to provide the univariate
//! `Forecaster` interface. Supports local level and local linear trend models
//! out of the box, or any custom `StateSpaceModel`.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::kalman::{KalmanFilter, StateSpaceModel};
use crate::models::traits::{validate_series_complete, Forecaster};
use crate::utils::stats::quantile_normal;

/// Forecaster adapter for the Kalman filter.
///
/// Provides convenient constructors for common state-space models and
/// implements the `Forecaster` trait for the univariate forecasting interface.
///
/// # Example
/// ```
/// use anofox_forecast::models::kalman_forecaster::KalmanForecaster;
/// let model = KalmanForecaster::local_level();
/// ```
#[derive(Debug, Clone)]
pub struct KalmanForecaster {
    model: StateSpaceModel,
    filter: Option<KalmanFilter>,
    fitted_values: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
}

impl KalmanForecaster {
    /// Create a KalmanForecaster with a local level (random walk plus noise) model.
    ///
    /// Uses default variance parameters: obs_var=1.0, level_var=0.1.
    pub fn local_level() -> Self {
        Self {
            model: StateSpaceModel::local_level(1.0, 0.1),
            filter: None,
            fitted_values: None,
            residuals: None,
        }
    }

    /// Create a KalmanForecaster with a local linear trend model.
    ///
    /// Uses default variance parameters: obs_var=1.0, level_var=0.1, trend_var=0.01.
    pub fn local_linear_trend() -> Self {
        Self {
            model: StateSpaceModel::local_linear_trend(1.0, 0.1, 0.01),
            filter: None,
            fitted_values: None,
            residuals: None,
        }
    }

    /// Create a KalmanForecaster with a custom state-space model.
    pub fn with_model(model: StateSpaceModel) -> Self {
        Self {
            model,
            filter: None,
            fitted_values: None,
            residuals: None,
        }
    }
}

impl Forecaster for KalmanForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;

        let values = series.primary_values();
        let n = values.len();

        // Convert univariate values to observation vectors.
        let observations: Vec<Vec<f64>> = values.iter().map(|&v| vec![v]).collect();

        let mut kf = KalmanFilter::new(self.model.clone())?;
        let filtered = kf.filter(&observations)?;

        // Extract fitted values (predicted observations) and residuals (innovations).
        let mut fitted = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for state in &filtered {
            fitted.push(state.predicted_obs[0]);
            residuals.push(state.innovation[0]);
        }

        self.fitted_values = Some(fitted);
        self.residuals = Some(residuals);
        self.filter = Some(kf);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let kf = self
            .filter
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions = kf.predict(horizon)?;
        let point: Vec<f64> = predictions.iter().map(|p| p[0]).collect();
        Ok(Forecast::from_values(point))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let kf = self
            .filter
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions = kf.predict(horizon)?;
        let z = quantile_normal((1.0 + level) / 2.0);

        // Compute residual standard deviation.
        let residuals = self
            .residuals
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let n = residuals.len();
        let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
        let var_r: f64 = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
        let sigma = var_r.sqrt();

        let mut point = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for (h, pred_vec) in predictions.iter().enumerate() {
            let pred = pred_vec[0];
            let width = z * sigma * ((h + 1) as f64).sqrt();
            point.push(pred);
            lower.push(pred - width);
            upper.push(pred + width);
        }

        Ok(Forecast::from_values_with_intervals(point, lower, upper))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted_values.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "KalmanForecaster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::error::ForecastError;

    fn make_kalman_test_series(n: usize) -> TimeSeries {
        use chrono::{Duration, TimeZone, Utc};
        let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..n).map(|i| base + Duration::days(i as i64)).collect();
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 + 1.0).collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn kalman_forecaster_local_level_fit_predict() {
        let ts = make_kalman_test_series(50);
        let mut model = KalmanForecaster::local_level();
        model.fit(&ts).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.name(), "KalmanForecaster");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);

        let fitted = model.fitted_values().unwrap();
        assert_eq!(fitted.len(), 50);

        let residuals = model.residuals().unwrap();
        assert_eq!(residuals.len(), 50);
    }

    #[test]
    fn kalman_forecaster_local_linear_trend_fit_predict() {
        let ts = make_kalman_test_series(100);
        let mut model = KalmanForecaster::local_linear_trend();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();
        assert_eq!(forecast.horizon(), 10);

        let preds = forecast.primary();
        for i in 1..preds.len() {
            assert!(preds[i] > preds[i - 1] - 1.0);
        }
    }

    #[test]
    fn kalman_forecaster_predict_with_intervals() {
        let ts = make_kalman_test_series(50);
        let mut model = KalmanForecaster::local_level();
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

        let width_first = upper[0] - lower[0];
        let width_last = upper[4] - lower[4];
        assert!(width_last >= width_first - 1e-10);
    }

    #[test]
    fn kalman_forecaster_requires_fit() {
        let model = KalmanForecaster::local_level();
        assert!(!model.is_fitted());
        assert!(matches!(
            model.predict(5).unwrap_err(),
            ForecastError::FitRequired { .. }
        ));
    }

    #[test]
    fn kalman_forecaster_zero_horizon() {
        let ts = make_kalman_test_series(30);
        let mut model = KalmanForecaster::local_level();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn kalman_forecaster_custom_model() {
        let ssm = StateSpaceModel::local_level(0.5, 0.1);
        let ts = make_kalman_test_series(50);
        let mut model = KalmanForecaster::with_model(ssm);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.horizon(), 3);
    }
}
