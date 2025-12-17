//! Naive forecasting model.
//!
//! The naive method simply forecasts the last observed value for all future periods.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Naive forecaster that repeats the last value.
#[derive(Debug, Clone, Default)]
pub struct Naive {
    last_value: Option<f64>,
    fitted: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    history: Option<Vec<f64>>,
}

impl Naive {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Forecaster for Naive {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        self.last_value = Some(*values.last().unwrap());
        self.history = Some(values.to_vec());

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
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions = vec![last; horizon];
        Ok(Forecast::from_values(predictions))
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
