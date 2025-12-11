//! TSB (Teunter-Syntetos-Babai) method for intermittent demand forecasting.
//!
//! TSB models demand probability and demand size separately, then combines them.
//! The implementation matches statsforecast's approach:
//! - SES is applied to the **non-zero demand values only**
//! - SES is applied to a **probability series** (0/1 binary indicating demand occurrence)
//! - The forecast is: demand_level * probability
//!
//! Reference: Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011).
//! "Intermittent demand: Linking forecasting to inventory obsolescence."

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// TSB method for intermittent demand forecasting.
#[derive(Debug, Clone)]
pub struct TSB {
    /// Smoothing parameter for demand size.
    alpha_d: f64,
    /// Smoothing parameter for probability.
    alpha_p: f64,
    /// Estimated demand level (SES on non-zero demands).
    demand_level: Option<f64>,
    /// Estimated demand probability (SES on 0/1 probability series).
    probability: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
}

impl TSB {
    /// Create a new TSB model with default parameters.
    ///
    /// Default alpha_d = 0.1 and alpha_p = 0.1, matching statsforecast defaults.
    pub fn new() -> Self {
        Self {
            alpha_d: 0.1,
            alpha_p: 0.1,
            demand_level: None,
            probability: None,
            fitted: None,
            residuals: None,
            n: 0,
        }
    }

    /// Create TSB model with specified parameters.
    pub fn with_params(mut self, alpha_d: f64, alpha_p: f64) -> Self {
        self.alpha_d = alpha_d.clamp(0.01, 0.99);
        self.alpha_p = alpha_p.clamp(0.01, 0.99);
        self
    }

    /// Get the demand smoothing parameter.
    pub fn alpha_d(&self) -> f64 {
        self.alpha_d
    }

    /// Get the probability smoothing parameter.
    pub fn alpha_p(&self) -> f64 {
        self.alpha_p
    }

    /// Get the estimated demand level.
    pub fn demand_level(&self) -> Option<f64> {
        self.demand_level
    }

    /// Get the estimated demand probability.
    pub fn probability(&self) -> Option<f64> {
        self.probability
    }

    /// Apply Simple Exponential Smoothing to a series.
    ///
    /// Returns (forecast, fitted_values) where forecast is the one-step-ahead prediction.
    /// Matches statsforecast's _ses_forecast implementation.
    fn ses_forecast(x: &[f64], alpha: f64) -> (f64, Vec<f64>) {
        if x.is_empty() {
            return (0.0, vec![]);
        }

        let complement = 1.0 - alpha;
        let mut fitted = vec![f64::NAN; x.len()];
        fitted[0] = x[0];

        for i in 1..x.len() {
            fitted[i] = alpha * x[i - 1] + complement * fitted[i - 1];
        }

        // One-step-ahead forecast
        let forecast = alpha * x[x.len() - 1] + complement * fitted[x.len() - 1];
        fitted[0] = f64::NAN;

        (forecast, fitted)
    }
}

impl Default for TSB {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for TSB {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if values.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: values.len(),
            });
        }

        // Handle all-zero case
        if values.iter().all(|&v| v == 0.0) {
            self.demand_level = Some(0.0);
            self.probability = Some(0.0);
            self.fitted = Some(vec![0.0; values.len()]);
            self.residuals = Some(vec![0.0; values.len()]);
            return Ok(());
        }

        // Extract non-zero demands (statsforecast's _demand function)
        let demands: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();

        // Create probability series: 1.0 if demand, 0.0 otherwise (statsforecast's _probability)
        let probabilities: Vec<f64> = values
            .iter()
            .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
            .collect();

        // Apply SES to demands (non-zero values only)
        let (demand_forecast, demand_fitted) = Self::ses_forecast(&demands, self.alpha_d);

        // Apply SES to probability series (0/1 values)
        let (prob_forecast, prob_fitted) = Self::ses_forecast(&probabilities, self.alpha_p);

        self.demand_level = Some(demand_forecast);
        self.probability = Some(prob_forecast);

        // Expand demand fitted values back to original series length
        // (statsforecast's _expand_fitted_demand)
        let mut demand_fitted_expanded = vec![f64::NAN; values.len()];
        let mut demand_idx = 0;
        for (i, &v) in values.iter().enumerate() {
            if v > 0.0 {
                if demand_idx < demand_fitted.len() {
                    demand_fitted_expanded[i] = demand_fitted[demand_idx];
                } else {
                    demand_fitted_expanded[i] = demand_forecast;
                }
                demand_idx += 1;
            } else {
                // For zero periods, use the last available demand forecast
                demand_fitted_expanded[i] = if demand_idx > 0 && demand_idx <= demand_fitted.len() {
                    demand_fitted[demand_idx - 1]
                } else if demand_idx > 0 {
                    demand_forecast
                } else {
                    f64::NAN
                };
            }
        }

        // Compute fitted values: demand_fitted * prob_fitted
        let fitted: Vec<f64> = demand_fitted_expanded
            .iter()
            .zip(prob_fitted.iter())
            .map(|(&d, &p)| if d.is_nan() || p.is_nan() { f64::NAN } else { d * p })
            .collect();

        // Compute residuals
        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted.iter())
            .map(|(&y, &f)| if f.is_nan() { f64::NAN } else { y - f })
            .collect();

        self.fitted = Some(fitted);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let demand_level = self.demand_level.ok_or(ForecastError::FitRequired)?;
        let probability = self.probability.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // TSB forecast: constant value = demand_level * probability
        // The forecast is flat (same value for all horizons), matching statsforecast behavior.
        // The probability decay during fitting captures historical patterns,
        // but the forecast represents the expected demand rate going forward.
        let forecast_value = demand_level * probability;
        let values = vec![forecast_value; horizon];

        Ok(Forecast::from_values(values))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Compute variance from residuals (excluding NaN values)
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        let std_dev = if valid_residuals.len() > 1 {
            let mean_resid: f64 = valid_residuals.iter().sum::<f64>() / valid_residuals.len() as f64;
            let variance = valid_residuals
                .iter()
                .map(|r| (r - mean_resid).powi(2))
                .sum::<f64>()
                / (valid_residuals.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);

        let lower: Vec<f64> = point_forecast
            .primary()
            .iter()
            .map(|&f| f - z * std_dev)
            .collect();
        let upper: Vec<f64> = point_forecast
            .primary()
            .iter()
            .map(|&f| f + z * std_dev)
            .collect();

        Ok(Forecast::from_values_with_intervals(
            point_forecast.primary().to_vec(),
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
        "TSB"
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

    fn make_intermittent_series() -> TimeSeries {
        let timestamps = make_timestamps(20);
        let values = vec![
            5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
        ];
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn tsb_basic() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        assert!(model.demand_level().is_some());
        assert!(model.probability().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn tsb_flat_forecast() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();

        // TSB forecast should be flat (same value for all horizons)
        // This matches statsforecast behavior where forecast = demand_level * probability
        let first = forecast.primary()[0];
        for (i, &val) in forecast.primary().iter().enumerate() {
            assert!(
                (val - first).abs() < 1e-10,
                "TSB forecast should be flat: {} != {} at index {}",
                val,
                first,
                i
            );
        }
    }

    #[test]
    fn tsb_with_params() {
        let ts = make_intermittent_series();
        let mut model = TSB::new().with_params(0.2, 0.3);
        model.fit(&ts).unwrap();

        assert!((model.alpha_d() - 0.2).abs() < 1e-10);
        assert!((model.alpha_p() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn tsb_probability_range() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let prob = model.probability().unwrap();
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability should be in [0, 1]: {}",
            prob
        );
    }

    #[test]
    fn tsb_insufficient_data() {
        let timestamps = make_timestamps(1);
        let values = vec![1.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TSB::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn tsb_all_zeros() {
        let timestamps = make_timestamps(10);
        let values = vec![0.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        // All-zero series should produce zero forecasts
        let forecast = model.predict(5).unwrap();
        for &val in forecast.primary() {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn tsb_requires_fit() {
        let model = TSB::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn tsb_zero_horizon() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn tsb_confidence_intervals() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn tsb_fitted_and_residuals() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 20);
    }

    #[test]
    fn tsb_name() {
        let model = TSB::new();
        assert_eq!(model.name(), "TSB");
    }

    #[test]
    fn tsb_default() {
        let model = TSB::default();
        assert!((model.alpha_d() - 0.1).abs() < 1e-10);
        assert!((model.alpha_p() - 0.1).abs() < 1e-10);
    }

    /// Validation test comparing TSB output with statsforecast.
    ///
    /// Data: Simple continuous series (all non-zero values)
    /// When all values are non-zero, TSB essentially becomes SES because:
    /// - _demand extracts all values
    /// - _probability is all 1s, so SES on it gives 1.0
    /// - forecast = demand_ses_forecast * 1.0 = demand_ses_forecast
    ///
    /// Reference: statsforecast.models.TSB(alpha_d=0.1, alpha_p=0.1)
    #[test]
    fn tsb_matches_statsforecast_continuous() {
        // Simple continuous data (no zeros)
        let timestamps = make_timestamps(10);
        let values = vec![50.0, 48.0, 52.0, 49.0, 51.0, 48.0, 50.0, 52.0, 49.0, 51.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();

        // Expected from statsforecast.models.TSB(alpha_d=0.1, alpha_p=0.1)
        // For continuous data: forecast = SES(demands) * SES(all-1s) ≈ SES(demands) * 1.0
        let expected = 50.056; // From Python validation

        for &pred in forecast.primary() {
            assert_relative_eq!(pred, expected, epsilon = 0.1);
        }
    }
}
