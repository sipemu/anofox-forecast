//! ARIMA (Autoregressive Integrated Moving Average) model.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::diff::{difference, integrate};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// ARIMA model specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ARIMASpec {
    /// AR order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// MA order (q)
    pub q: usize,
}

impl ARIMASpec {
    /// Create a new ARIMA specification.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.p + self.q + 1 // AR + MA + intercept
    }
}

impl Default for ARIMASpec {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

/// ARIMA forecasting model.
///
/// ARIMA(p, d, q) combines:
/// - AR(p): Autoregressive component
/// - I(d): Differencing for stationarity
/// - MA(q): Moving average component
#[derive(Debug, Clone)]
pub struct ARIMA {
    /// Model specification.
    spec: ARIMASpec,
    /// AR coefficients.
    ar_coefficients: Vec<f64>,
    /// MA coefficients.
    ma_coefficients: Vec<f64>,
    /// Intercept (mean of differenced series).
    intercept: f64,
    /// Original series (for integration).
    original: Option<Vec<f64>>,
    /// Differenced series.
    differenced: Option<Vec<f64>>,
    /// Fitted values on differenced scale.
    fitted_diff: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// AIC.
    aic: Option<f64>,
    /// BIC.
    bic: Option<f64>,
    /// Series length.
    n: usize,
}

impl ARIMA {
    /// Create a new ARIMA model.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            spec: ARIMASpec::new(p, d, q),
            ar_coefficients: vec![],
            ma_coefficients: vec![],
            intercept: 0.0,
            original: None,
            differenced: None,
            fitted_diff: None,
            residuals: None,
            residual_variance: None,
            aic: None,
            bic: None,
            n: 0,
        }
    }

    /// Create an ARIMA(1,1,1) model.
    pub fn arima_111() -> Self {
        Self::new(1, 1, 1)
    }

    /// Create an AR(p) model (ARIMA with d=0, q=0).
    pub fn ar(p: usize) -> Self {
        Self::new(p, 0, 0)
    }

    /// Create an MA(q) model (ARIMA with p=0, d=0).
    pub fn ma(q: usize) -> Self {
        Self::new(0, 0, q)
    }

    /// Get the model specification.
    pub fn spec(&self) -> ARIMASpec {
        self.spec
    }

    /// Get AR coefficients.
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coefficients
    }

    /// Get MA coefficients.
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coefficients
    }

    /// Get the intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get AIC.
    pub fn aic(&self) -> Option<f64> {
        self.aic
    }

    /// Get BIC.
    pub fn bic(&self) -> Option<f64> {
        self.bic
    }

    /// Calculate the conditional sum of squares for given parameters.
    fn calculate_css(
        diff_series: &[f64],
        p: usize,
        q: usize,
        ar: &[f64],
        ma: &[f64],
        intercept: f64,
    ) -> f64 {
        let n = diff_series.len();
        let start = p.max(q);

        if n <= start {
            return f64::MAX;
        }

        let mut residuals = vec![0.0; n];
        let mut css = 0.0;

        for t in start..n {
            let mut pred = intercept;

            // AR component
            for i in 0..p {
                pred += ar[i] * (diff_series[t - 1 - i] - intercept);
            }

            // MA component
            for i in 0..q {
                pred += ma[i] * residuals[t - 1 - i];
            }

            let error = diff_series[t] - pred;
            residuals[t] = error;
            css += error * error;
        }

        css
    }

    /// Estimate parameters using conditional least squares.
    fn estimate_parameters(&mut self, diff_series: &[f64]) {
        let p = self.spec.p;
        let q = self.spec.q;

        // Calculate mean for intercept initialization
        let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;

        if p == 0 && q == 0 {
            // Just the mean
            self.intercept = mean;
            self.ar_coefficients = vec![];
            self.ma_coefficients = vec![];
            return;
        }

        // Set up optimization
        let n_params = p + q + 1; // AR + MA + intercept
        let mut initial = vec![0.0; n_params];
        initial[0] = mean; // intercept

        // Initialize AR coefficients with small values
        for i in 0..p {
            initial[1 + i] = 0.1 / (i + 1) as f64;
        }
        // Initialize MA coefficients
        for i in 0..q {
            initial[1 + p + i] = 0.1 / (i + 1) as f64;
        }

        // Set up bounds (AR and MA coefficients should be bounded for stationarity/invertibility)
        let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)]; // intercept
        for _ in 0..p {
            bounds.push((-0.99, 0.99)); // AR bounds
        }
        for _ in 0..q {
            bounds.push((-0.99, 0.99)); // MA bounds
        }

        let config = NelderMeadConfig {
            max_iter: 1000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(
            |params| {
                let intercept = params[0];
                let ar: Vec<f64> = params[1..1 + p].to_vec();
                let ma: Vec<f64> = params[1 + p..].to_vec();
                Self::calculate_css(diff_series, p, q, &ar, &ma, intercept)
            },
            &initial,
            Some(&bounds),
            config,
        );

        // Extract optimized parameters
        self.intercept = result.optimal_point[0];
        self.ar_coefficients = result.optimal_point[1..1 + p].to_vec();
        self.ma_coefficients = result.optimal_point[1 + p..].to_vec();
    }

    /// Calculate fitted values and residuals.
    fn calculate_fitted(&mut self, diff_series: &[f64]) {
        let n = diff_series.len();
        let p = self.spec.p;
        let q = self.spec.q;
        let start = p.max(q);

        let mut fitted = vec![f64::NAN; n];
        let mut residuals = vec![0.0; n];

        for t in start..n {
            let mut pred = self.intercept;

            // AR component
            for i in 0..p {
                pred += self.ar_coefficients[i] * (diff_series[t - 1 - i] - self.intercept);
            }

            // MA component
            for i in 0..q {
                pred += self.ma_coefficients[i] * residuals[t - 1 - i];
            }

            fitted[t] = pred;
            residuals[t] = diff_series[t] - pred;
        }

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals[start..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);

            // Calculate information criteria
            let n_eff = valid_residuals.len() as f64;
            let k = self.spec.num_params() as f64;
            let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

            self.aic = Some(-2.0 * ll + 2.0 * k);
            self.bic = Some(-2.0 * ll + k * n_eff.ln());
        }

        self.fitted_diff = Some(fitted);
        self.residuals = Some(residuals);
    }
}

impl Default for ARIMA {
    fn default() -> Self {
        Self::arima_111()
    }
}

impl Forecaster for ARIMA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        let min_len = self.spec.d + self.spec.p.max(self.spec.q) + 2;

        if values.len() < min_len {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
            });
        }

        self.n = values.len();
        self.original = Some(values.to_vec());

        // Apply differencing
        let diff_series = difference(values, self.spec.d);
        self.differenced = Some(diff_series.clone());

        // Estimate parameters
        self.estimate_parameters(&diff_series);

        // Calculate fitted values and residuals
        self.calculate_fitted(&diff_series);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let original = self.original.as_ref().ok_or(ForecastError::FitRequired)?;
        let diff_series = self
            .differenced
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let p = self.spec.p;
        let q = self.spec.q;

        // Forecast on differenced scale
        let mut extended_diff = diff_series.clone();
        let mut extended_residuals = residuals.clone();

        for _ in 0..horizon {
            let t = extended_diff.len();
            let mut pred = self.intercept;

            // AR component
            for i in 0..p {
                if t > i {
                    pred += self.ar_coefficients[i] * (extended_diff[t - 1 - i] - self.intercept);
                }
            }

            // MA component (residuals become 0 for forecasts)
            for i in 0..q {
                if t > i {
                    pred += self.ma_coefficients[i] * extended_residuals[t - 1 - i];
                }
            }

            extended_diff.push(pred);
            extended_residuals.push(0.0); // Future residuals are 0
        }

        // Extract forecast on differenced scale
        let forecast_diff: Vec<f64> = extended_diff[diff_series.len()..].to_vec();

        // Integrate back to original scale
        let predictions = if self.spec.d > 0 {
            integrate(&forecast_diff, original, self.spec.d)
        } else {
            forecast_diff
        };

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        // Calculate cumulative variance for each horizon
        for h in 1..=horizon {
            // Simplified variance calculation
            // For ARIMA, the variance grows with horizon
            let cumulative_var = variance * h as f64;
            let se = cumulative_var.sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted_diff.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "ARIMA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    #[test]
    fn arima_basic_fit() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        assert!(model.ar_coefficients().len() == 1);
        assert!(model.ma_coefficients().len() == 1);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_ar1() {
        let timestamps = make_timestamps(100);
        // Generate AR(1) process: y_t = 0.7 * y_{t-1} + e_t
        let mut values = vec![10.0];
        for i in 1..100 {
            values.push(0.7 * values[i - 1] + (i as f64 * 0.1).sin());
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::ar(1);
        model.fit(&ts).unwrap();

        // AR coefficient should be close to 0.7
        assert!(model.ar_coefficients()[0] > 0.3);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_ma1() {
        let timestamps = make_timestamps(100);
        // Simple series for MA testing
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::ma(1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_011_random_walk() {
        let timestamps = make_timestamps(50);
        // Random walk-like series
        let mut values = vec![10.0];
        for i in 1..50 {
            values.push(values[i - 1] + 0.5 + (i as f64 * 0.1).sin() * 0.1);
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(0, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_with_differencing() {
        let timestamps = make_timestamps(50);
        // Strong trend
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = ARIMA::new(1, 1, 0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Forecast should continue the trend
        assert!(preds[0] > values.last().unwrap() - 5.0);
    }

    #[test]
    fn arima_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        // Intervals should exist and be reasonable
        for i in 0..5 {
            assert!(lower[i].is_finite());
            assert!(upper[i].is_finite());
            assert!(upper[i] >= lower[i]);
        }
    }

    #[test]
    fn arima_information_criteria() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 0, 1);
        model.fit(&ts).unwrap();

        assert!(model.aic().is_some());
        assert!(model.bic().is_some());
    }

    #[test]
    fn arima_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(2, 1, 1);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn arima_requires_fit() {
        let model = ARIMA::new(1, 1, 1);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn arima_zero_horizon() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn arima_spec() {
        let spec = ARIMASpec::new(2, 1, 3);
        assert_eq!(spec.p, 2);
        assert_eq!(spec.d, 1);
        assert_eq!(spec.q, 3);
        assert_eq!(spec.num_params(), 6); // 2 AR + 3 MA + 1 intercept
    }

    #[test]
    fn arima_default() {
        let model = ARIMA::default();
        assert_eq!(model.spec().p, 1);
        assert_eq!(model.spec().d, 1);
        assert_eq!(model.spec().q, 1);
    }

    #[test]
    fn arima_name() {
        let model = ARIMA::new(1, 1, 1);
        assert_eq!(model.name(), "ARIMA");
    }

    #[test]
    fn arima_getters() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        assert!(!model.ar_coefficients().is_empty());
        assert!(!model.ma_coefficients().is_empty());
        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }
}
