//! Seasonal Naive forecasting model.
//!
//! Forecasts by repeating the value from the same season in the previous cycle.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Seasonal Naive forecaster.
///
/// Each forecast is equal to the observation from the same season
/// in the previous year (or previous seasonal period).
#[derive(Debug, Clone)]
pub struct SeasonalNaive {
    period: usize,
    history: Option<Vec<f64>>,
    fitted: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    residual_variance: Option<f64>,
}

impl SeasonalNaive {
    /// Create a new SeasonalNaive model with the given seasonal period.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            history: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
        }
    }

    /// Get the seasonal period.
    pub fn period(&self) -> usize {
        self.period
    }
}

impl Default for SeasonalNaive {
    fn default() -> Self {
        Self::new(12) // Default to monthly seasonality
    }
}

impl Forecaster for SeasonalNaive {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < self.period {
            return Err(ForecastError::InsufficientData {
                needed: self.period,
                got: values.len(),
            });
        }

        self.history = Some(values.to_vec());

        // Fitted values: y_hat[t] = y[t - period]
        let mut fitted = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            if i < self.period {
                fitted.push(f64::NAN);
            } else {
                fitted.push(values[i - self.period]);
            }
        }
        self.fitted = Some(fitted);

        // Residuals: y[t] - y[t - period]
        let residuals: Vec<f64> = (0..values.len())
            .map(|i| {
                if i < self.period {
                    f64::NAN
                } else {
                    values[i] - values[i - self.period]
                }
            })
            .collect();

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();
        if !valid_residuals.is_empty() {
            let variance =
                valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let history = self.history.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let n = history.len();
        let predictions: Vec<f64> = (0..horizon)
            .map(|h| {
                // Find corresponding seasonal value
                let idx = n - self.period + (h % self.period);
                if idx < n {
                    history[idx]
                } else {
                    // Wrap around for forecasts beyond one season
                    history[n - self.period + ((h + n - self.period) % self.period)]
                }
            })
            .collect();

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let history = self.history.as_ref().ok_or(ForecastError::FitRequired)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let sigma = variance.sqrt();

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let n = history.len();

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let idx = n - self.period + (h % self.period);
            let pred = if idx < n {
                history[idx]
            } else {
                history[n - self.period + ((h + n - self.period) % self.period)]
            };
            predictions.push(pred);

            // Standard error increases with number of complete seasons ahead
            let k = (h / self.period) + 1;
            let se = sigma * (k as f64).sqrt();
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

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "SeasonalNaive"
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
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1, i as u32 % 24, 0, 0)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn seasonal_naive_repeats_seasonal_pattern() {
        // Period of 4: [1, 2, 3, 4, 1, 2, 3, 4]
        let timestamps = make_timestamps(8);
        let values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        model.fit(&ts).unwrap();

        let forecast = model.predict(4).unwrap();
        assert_eq!(forecast.primary(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn seasonal_naive_quarterly_data() {
        let timestamps = make_timestamps(8);
        let values = vec![10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        model.fit(&ts).unwrap();

        // Forecast should repeat the last seasonal cycle
        let forecast = model.predict(4).unwrap();
        assert_eq!(forecast.primary(), &[11.0, 21.0, 31.0, 41.0]);
    }

    #[test]
    fn seasonal_naive_weekly_data() {
        let timestamps = make_timestamps(14);
        let values: Vec<f64> = (0..14).map(|i| ((i % 7) + 1) as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(7);
        model.fit(&ts).unwrap();

        let forecast = model.predict(7).unwrap();
        assert_eq!(forecast.primary(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn seasonal_naive_requires_full_season() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { needed: 4, got: 3 })
        ));
    }

    #[test]
    fn seasonal_naive_forecast_beyond_one_season() {
        let timestamps = make_timestamps(8);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        model.fit(&ts).unwrap();

        // Forecast 8 steps (2 seasonal cycles)
        let forecast = model.predict(8).unwrap();
        let preds = forecast.primary();

        // Should repeat: [5,6,7,8, 5,6,7,8]
        assert_eq!(&preds[0..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&preds[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn seasonal_naive_fitted_values_and_residuals() {
        let timestamps = make_timestamps(8);
        let values = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        // First period has no fitted values
        assert!(fitted[0].is_nan());
        assert!(fitted[1].is_nan());
        assert!(fitted[2].is_nan());
        assert!(fitted[3].is_nan());
        // Second period: fitted[i] = values[i-4]
        assert_eq!(&fitted[4..], &[1.0, 2.0, 3.0, 4.0]);

        let residuals = model.residuals().unwrap();
        // Residuals: values[i] - values[i-4]
        assert!(residuals[0].is_nan());
        assert_relative_eq!(residuals[4], 1.0, epsilon = 1e-10); // 2 - 1
        assert_relative_eq!(residuals[5], 1.0, epsilon = 1e-10); // 3 - 2
    }

    #[test]
    fn seasonal_naive_confidence_intervals() {
        let timestamps = make_timestamps(16);
        let values: Vec<f64> = (0..16)
            .map(|i| ((i % 4) as f64) + 0.1 * (i as f64))
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalNaive::new(4);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(8, 0.95).unwrap();
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        // Intervals for second season should be wider than first
        let width_first = upper[0] - lower[0];
        let width_second = upper[4] - lower[4];
        assert!(width_second > width_first);
    }

    #[test]
    fn seasonal_naive_name_is_correct() {
        let model = SeasonalNaive::new(12);
        assert_eq!(model.name(), "SeasonalNaive");
    }
}
