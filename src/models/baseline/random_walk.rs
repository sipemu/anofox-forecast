//! Random Walk with Drift model.
//!
//! Forecasts based on the last value plus a drift term estimated from historical data.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Random walk with drift forecaster.
///
/// The forecast is: y_hat\[t+h\] = y\[t\] + h * drift
/// where drift is the average change in the series.
#[derive(Debug, Clone, Default)]
pub struct RandomWalkWithDrift {
    last_value: Option<f64>,
    drift: Option<f64>,
    fitted: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    residual_variance: Option<f64>,
}

impl RandomWalkWithDrift {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the estimated drift parameter.
    pub fn drift(&self) -> Option<f64> {
        self.drift
    }
}

impl Forecaster for RandomWalkWithDrift {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: values.len(),
            });
        }

        self.last_value = Some(*values.last().ok_or(ForecastError::EmptyData)?);

        // Calculate drift as average of first differences
        let n = values.len();
        let drift = (values[n - 1] - values[0]) / (n - 1) as f64;
        self.drift = Some(drift);

        // Fitted values: y_hat[t] = y[t-1] + drift
        let mut fitted = Vec::with_capacity(n);
        fitted.push(f64::NAN);
        for i in 1..n {
            fitted.push(values[i - 1] + drift);
        }
        self.fitted = Some(fitted);

        // Residuals: y[t] - y_hat[t]
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                if i == 0 {
                    f64::NAN
                } else {
                    values[i] - (values[i - 1] + drift)
                }
            })
            .collect();

        // Calculate residual variance for prediction intervals
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;
        let drift = self.drift.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions: Vec<f64> = (1..=horizon).map(|h| last + (h as f64) * drift).collect();

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let last = self.last_value.ok_or(ForecastError::FitRequired)?;
        let drift = self.drift.ok_or(ForecastError::FitRequired)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let sigma = variance.sqrt();

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let pred = last + (h as f64) * drift;
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

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let variance = self.residual_variance?;

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
        "RandomWalkWithDrift"
    }
}

fn quantile_normal(p: f64) -> f64 {
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
    use crate::models::baseline::Naive;
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i as u32, 0, 0).unwrap())
            .collect()
    }

    #[test]
    fn random_walk_calculates_drift_correctly() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect linear trend, drift = 1
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.drift().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn random_walk_produces_trending_forecast() {
        let timestamps = make_timestamps(5);
        let values = vec![0.0, 2.0, 4.0, 6.0, 8.0]; // Drift = 2
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        let preds = forecast.primary();

        // Last value is 8, drift is 2
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-10); // 8 + 1*2
        assert_relative_eq!(preds[1], 12.0, epsilon = 1e-10); // 8 + 2*2
        assert_relative_eq!(preds[2], 14.0, epsilon = 1e-10); // 8 + 3*2
    }

    #[test]
    fn random_walk_handles_zero_drift() {
        let timestamps = make_timestamps(5);
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // Constant, drift = 0
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.drift().unwrap(), 0.0, epsilon = 1e-10);

        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.primary(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn random_walk_short_series() {
        let timestamps = make_timestamps(2);
        let values = vec![1.0, 3.0]; // Minimal series
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.drift().unwrap(), 2.0, epsilon = 1e-10);

        let forecast = model.predict(2).unwrap();
        assert_relative_eq!(forecast.primary()[0], 5.0, epsilon = 1e-10); // 3 + 1*2
        assert_relative_eq!(forecast.primary()[1], 7.0, epsilon = 1e-10); // 3 + 2*2
    }

    #[test]
    fn random_walk_confidence_intervals_widen() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10)
            .map(|i| (i as f64) * 2.0 + 0.5 * (i as f64).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        for i in 1..5 {
            let width_prev = upper[i - 1] - lower[i - 1];
            let width_curr = upper[i] - lower[i];
            assert!(width_curr > width_prev);
        }
    }

    #[test]
    fn random_walk_vs_naive_on_trending_data() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| (i as f64) * 3.0).collect(); // Trend
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut rw = RandomWalkWithDrift::new();
        rw.fit(&ts).unwrap();

        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();

        let rw_forecast = rw.predict(3).unwrap();
        let naive_forecast = naive.predict(3).unwrap();

        // Random walk should trend upward
        assert!(rw_forecast.primary()[2] > rw_forecast.primary()[0]);

        // Naive should be flat
        assert_eq!(naive_forecast.primary()[0], naive_forecast.primary()[2]);

        // Random walk should give higher forecasts for upward trend
        assert!(rw_forecast.primary()[2] > naive_forecast.primary()[2]);
    }

    #[test]
    fn random_walk_requires_minimum_data() {
        let timestamps = make_timestamps(1);
        let values = vec![1.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = RandomWalkWithDrift::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { needed: 2, got: 1 })
        ));
    }

    #[test]
    fn random_walk_name_is_correct() {
        let model = RandomWalkWithDrift::new();
        assert_eq!(model.name(), "RandomWalkWithDrift");
    }
}
