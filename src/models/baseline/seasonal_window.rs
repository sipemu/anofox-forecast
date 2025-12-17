//! Seasonal Window Average forecasting model.
//!
//! Forecasts by averaging observations from the same season across multiple cycles.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Seasonal Window Average forecaster.
///
/// Forecasts are computed as the average of observations from the same
/// season over a specified number of previous cycles (window).
#[derive(Debug, Clone)]
pub struct SeasonalWindowAverage {
    period: usize,
    window: usize, // Number of seasonal cycles to average
    history: Option<Vec<f64>>,
    fitted: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    residual_variance: Option<f64>,
}

impl SeasonalWindowAverage {
    /// Create a new SeasonalWindowAverage model.
    ///
    /// # Arguments
    /// * `period` - The seasonal period
    /// * `window` - Number of seasonal cycles to average (1 = SeasonalNaive)
    pub fn new(period: usize, window: usize) -> Self {
        Self {
            period,
            window: window.max(1), // At least 1
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

    /// Get the window size.
    pub fn window(&self) -> usize {
        self.window
    }
}

impl Default for SeasonalWindowAverage {
    fn default() -> Self {
        Self::new(12, 2) // Monthly with 2-year window
    }
}

impl Forecaster for SeasonalWindowAverage {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < self.period {
            return Err(ForecastError::InsufficientData {
                needed: self.period,
                got: values.len(),
            });
        }

        self.history = Some(values.to_vec());

        // Calculate fitted values
        let n = values.len();
        let mut fitted = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.period {
                fitted.push(f64::NAN);
            } else {
                // Average same-season values from previous cycles
                let mut sum = 0.0;
                let mut count = 0;

                for k in 1..=self.window {
                    // Go back k cycles from position i
                    if i >= k * self.period {
                        let idx = i - k * self.period;
                        sum += values[idx];
                        count += 1;
                    }
                }

                if count > 0 {
                    fitted.push(sum / count as f64);
                } else {
                    fitted.push(f64::NAN);
                }
            }
        }
        self.fitted = Some(fitted.clone());

        // Calculate residuals
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                if fitted[i].is_nan() {
                    f64::NAN
                } else {
                    values[i] - fitted[i]
                }
            })
            .collect();

        // Residual variance
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
        let history = self.history.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let n = history.len();
        let mut predictions = Vec::with_capacity(horizon);

        for h in 0..horizon {
            // What season position are we forecasting?
            // The forecast position is n + h, so its season is (n + h) % period
            let _target_season = (n + h) % self.period;

            // Find all same-season values in the last `window` cycles
            let mut sum = 0.0;
            let mut count = 0;

            // Start from the last occurrence of this season in history
            // Last occurrence: n - 1 - ((n - 1 - target_season) % period) if target_season < n
            // More simply: find all indices i where i % period == target_season
            for k in 1..=self.window {
                // k cycles back from the forecast position
                let forecast_pos = n + h;
                if forecast_pos >= k * self.period {
                    let idx = forecast_pos - k * self.period;
                    if idx < n {
                        sum += history[idx];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                predictions.push(sum / count as f64);
            } else {
                // Fallback to last value
                predictions.push(*history.last().ok_or(ForecastError::EmptyData)?);
            }
        }

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let base = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let sigma = variance.sqrt();

        if horizon == 0 {
            return Ok(base);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = base.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for (h, pred) in preds.iter().enumerate() {
            let k = (h / self.period) + 1;
            let se = sigma * (k as f64).sqrt();
            lower.push(pred - z * se);
            upper.push(pred + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
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
        "SeasonalWindowAverage"
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
    use crate::models::baseline::SeasonalNaive;
    use crate::models::Forecaster;
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
    fn seasonal_window_average_averages_seasonal_values() {
        // 3 cycles of period 4
        let timestamps = make_timestamps(12);
        let values = vec![
            1.0, 2.0, 3.0, 4.0, // Cycle 1
            2.0, 3.0, 4.0, 5.0, // Cycle 2
            3.0, 4.0, 5.0, 6.0, // Cycle 3
        ];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalWindowAverage::new(4, 2);
        model.fit(&ts).unwrap();

        let forecast = model.predict(4).unwrap();
        let preds = forecast.primary();

        // For season 0: average of 2.0 and 3.0 = 2.5
        // For season 1: average of 3.0 and 4.0 = 3.5
        // For season 2: average of 4.0 and 5.0 = 4.5
        // For season 3: average of 5.0 and 6.0 = 5.5
        assert_relative_eq!(preds[0], 2.5, epsilon = 1e-10);
        assert_relative_eq!(preds[1], 3.5, epsilon = 1e-10);
        assert_relative_eq!(preds[2], 4.5, epsilon = 1e-10);
        assert_relative_eq!(preds[3], 5.5, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_window_window_1_equals_seasonal_naive() {
        let timestamps = make_timestamps(12);
        let values: Vec<f64> = (0..12).map(|i| (i as f64) + 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut swa = SeasonalWindowAverage::new(4, 1);
        swa.fit(&ts).unwrap();

        let mut sn = SeasonalNaive::new(4);
        sn.fit(&ts).unwrap();

        let swa_forecast = swa.predict(4).unwrap();
        let sn_forecast = sn.predict(4).unwrap();

        // With window=1, should give same results as SeasonalNaive
        for i in 0..4 {
            assert_relative_eq!(
                swa_forecast.primary()[i],
                sn_forecast.primary()[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn seasonal_window_different_window_sizes() {
        let timestamps = make_timestamps(16);
        // 4 cycles with increasing values
        let values: Vec<f64> = (0..16).map(|i| ((i / 4) + 1) as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model1 = SeasonalWindowAverage::new(4, 1);
        let mut model2 = SeasonalWindowAverage::new(4, 2);
        let mut model3 = SeasonalWindowAverage::new(4, 4);

        model1.fit(&ts).unwrap();
        model2.fit(&ts).unwrap();
        model3.fit(&ts).unwrap();

        let f1 = model1.predict(1).unwrap();
        let f2 = model2.predict(1).unwrap();
        let f3 = model3.predict(1).unwrap();

        // Window 1: last cycle = 4.0
        // Window 2: average of 3.0 and 4.0 = 3.5
        // Window 4: average of 1, 2, 3, 4 = 2.5
        assert_relative_eq!(f1.primary()[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(f2.primary()[0], 3.5, epsilon = 1e-10);
        assert_relative_eq!(f3.primary()[0], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_window_confidence_intervals() {
        let timestamps = make_timestamps(16);
        let values: Vec<f64> = (0..16)
            .map(|i| ((i % 4) as f64) + 0.1 * (i as f64))
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalWindowAverage::new(4, 2);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(8, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        // Intervals should exist
        for i in 0..8 {
            assert!(lower[i] < upper[i]);
        }
    }

    #[test]
    fn seasonal_window_handles_limited_data() {
        // Only 2 cycles but window of 3
        let timestamps = make_timestamps(8);
        let values = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalWindowAverage::new(4, 3);
        model.fit(&ts).unwrap();

        // Should still work, using available cycles
        let forecast = model.predict(4).unwrap();
        assert_eq!(forecast.horizon(), 4);
    }

    #[test]
    fn seasonal_window_smooths_vs_seasonal_naive() {
        // Series with noisy seasonal pattern
        let timestamps = make_timestamps(12);
        let values = vec![
            10.0, 20.0, 30.0, 40.0, // Cycle 1
            12.0, 22.0, 32.0, 42.0, // Cycle 2 (slightly higher)
            8.0, 18.0, 28.0, 38.0, // Cycle 3 (slightly lower)
        ];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut swa = SeasonalWindowAverage::new(4, 3);
        swa.fit(&ts).unwrap();

        let mut sn = SeasonalNaive::new(4);
        sn.fit(&ts).unwrap();

        let swa_forecast = swa.predict(4).unwrap();
        let sn_forecast = sn.predict(4).unwrap();

        // SWA should smooth out the noise
        // SeasonalNaive uses last cycle: [8, 18, 28, 38]
        // SWA averages all cycles: [10, 20, 30, 40]
        assert_relative_eq!(swa_forecast.primary()[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(sn_forecast.primary()[0], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_window_quarterly_seasonality() {
        let timestamps = make_timestamps(8);
        let values = vec![100.0, 120.0, 80.0, 90.0, 110.0, 130.0, 90.0, 100.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalWindowAverage::new(4, 2);
        model.fit(&ts).unwrap();

        let forecast = model.predict(4).unwrap();
        let preds = forecast.primary();

        // Average of corresponding seasons
        assert_relative_eq!(preds[0], (100.0 + 110.0) / 2.0, epsilon = 1e-10);
        assert_relative_eq!(preds[1], (120.0 + 130.0) / 2.0, epsilon = 1e-10);
        assert_relative_eq!(preds[2], (80.0 + 90.0) / 2.0, epsilon = 1e-10);
        assert_relative_eq!(preds[3], (90.0 + 100.0) / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_window_name_is_correct() {
        let model = SeasonalWindowAverage::new(4, 2);
        assert_eq!(model.name(), "SeasonalWindowAverage");
    }
}
