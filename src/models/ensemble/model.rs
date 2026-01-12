//! Ensemble forecasting methods.
//!
//! Combines multiple forecasting models to produce a single forecast,
//! often with improved accuracy and robustness.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Method for combining forecasts from multiple models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombinationMethod {
    /// Simple average of all forecasts.
    Mean,
    /// Median of all forecasts.
    Median,
    /// Weighted by inverse MSE on fitted values.
    WeightedMSE,
    /// Custom weights provided by user.
    Custom,
}

/// Ensemble forecaster that combines multiple models.
pub struct Ensemble {
    /// The forecasting models.
    models: Vec<Box<dyn Forecaster>>,
    /// Method for combining forecasts.
    method: CombinationMethod,
    /// Custom weights (used when method is Custom).
    custom_weights: Option<Vec<f64>>,
    /// Computed weights after fitting.
    weights: Vec<f64>,
    /// Combined fitted values.
    fitted: Option<Vec<f64>>,
    /// Combined residuals.
    residuals: Option<Vec<f64>>,
    /// Whether models have been fitted.
    is_fitted: bool,
}

impl Ensemble {
    /// Create a new ensemble with the given models.
    pub fn new(models: Vec<Box<dyn Forecaster>>) -> Self {
        let n = models.len();
        Self {
            models,
            method: CombinationMethod::Mean,
            custom_weights: None,
            weights: vec![1.0 / n as f64; n],
            fitted: None,
            residuals: None,
            is_fitted: false,
        }
    }

    /// Set the combination method.
    pub fn with_method(mut self, method: CombinationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set custom weights (must match number of models).
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.custom_weights = Some(weights);
        self.method = CombinationMethod::Custom;
        self
    }

    /// Get the current weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the combination method.
    pub fn method(&self) -> CombinationMethod {
        self.method
    }

    /// Get the number of models in the ensemble.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Combine values using the specified method.
    fn combine_values(&self, values: &[Vec<f64>]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let horizon = values[0].len();
        let mut combined = vec![0.0; horizon];

        match self.method {
            CombinationMethod::Mean => {
                for h in 0..horizon {
                    let sum: f64 = values.iter().map(|v| v[h]).sum();
                    combined[h] = sum / values.len() as f64;
                }
            }
            CombinationMethod::Median => {
                for h in 0..horizon {
                    let mut vals: Vec<f64> = values.iter().map(|v| v[h]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = vals.len();
                    combined[h] = if n.is_multiple_of(2) {
                        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                    } else {
                        vals[n / 2]
                    };
                }
            }
            CombinationMethod::WeightedMSE | CombinationMethod::Custom => {
                for h in 0..horizon {
                    let weighted_sum: f64 = values
                        .iter()
                        .zip(self.weights.iter())
                        .map(|(v, w)| v[h] * w)
                        .sum();
                    combined[h] = weighted_sum;
                }
            }
        }

        combined
    }

    /// Compute weights based on MSE of fitted values.
    fn compute_mse_weights(&mut self, actual: &[f64]) {
        let n = self.models.len();
        let mut mse_values = vec![f64::INFINITY; n];

        for (i, model) in self.models.iter().enumerate() {
            if let Some(fitted) = model.fitted_values() {
                let mse: f64 = actual
                    .iter()
                    .zip(fitted.iter())
                    .map(|(a, f)| (a - f).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64;
                mse_values[i] = mse.max(1e-10); // Avoid division by zero
            }
        }

        // Convert MSE to weights (inverse MSE, normalized)
        let inv_mse: Vec<f64> = mse_values.iter().map(|m| 1.0 / m).collect();
        let sum_inv: f64 = inv_mse.iter().sum();

        self.weights = if sum_inv > 0.0 {
            inv_mse.iter().map(|w| w / sum_inv).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
    }
}

impl Forecaster for Ensemble {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        if self.models.is_empty() {
            return Err(ForecastError::ComputationError(
                "Ensemble has no models".to_string(),
            ));
        }

        // Fit all models
        for model in &mut self.models {
            model.fit(series)?;
        }

        let values = series.primary_values();

        // Compute weights if using MSE-based weighting
        if self.method == CombinationMethod::WeightedMSE {
            self.compute_mse_weights(values);
        } else if self.method == CombinationMethod::Custom {
            if let Some(ref custom) = self.custom_weights {
                if custom.len() != self.models.len() {
                    return Err(ForecastError::ComputationError(format!(
                        "Custom weights length ({}) doesn't match model count ({})",
                        custom.len(),
                        self.models.len()
                    )));
                }
                // Normalize weights
                let sum: f64 = custom.iter().sum();
                self.weights = custom.iter().map(|w| w / sum).collect();
            }
        }

        // Combine fitted values
        let all_fitted: Vec<Vec<f64>> = self
            .models
            .iter()
            .filter_map(|m| m.fitted_values().map(|f| f.to_vec()))
            .collect();

        if !all_fitted.is_empty() {
            let combined_fitted = self.combine_values(&all_fitted);
            let residuals: Vec<f64> = values
                .iter()
                .zip(combined_fitted.iter())
                .map(|(y, f)| y - f)
                .collect();
            self.fitted = Some(combined_fitted);
            self.residuals = Some(residuals);
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if !self.is_fitted {
            return Err(ForecastError::FitRequired);
        }

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // Get forecasts from all models
        let all_forecasts: Vec<Vec<f64>> = self
            .models
            .iter()
            .filter_map(|m| m.predict(horizon).ok())
            .map(|f| f.primary().to_vec())
            .collect();

        if all_forecasts.is_empty() {
            return Err(ForecastError::ComputationError(
                "No models produced valid forecasts".to_string(),
            ));
        }

        let combined = self.combine_values(&all_forecasts);
        Ok(Forecast::from_values(combined))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Get all forecasts with intervals
        let all_forecasts: Vec<Forecast> = self
            .models
            .iter()
            .filter_map(|m| m.predict_with_intervals(horizon, level).ok())
            .collect();

        if all_forecasts.is_empty() {
            return Ok(point_forecast);
        }

        // Combine lower bounds
        let all_lowers: Vec<Vec<f64>> = all_forecasts
            .iter()
            .filter_map(|f| f.lower_series(0).ok().map(|l| l.to_vec()))
            .collect();

        // Combine upper bounds
        let all_uppers: Vec<Vec<f64>> = all_forecasts
            .iter()
            .filter_map(|f| f.upper_series(0).ok().map(|u| u.to_vec()))
            .collect();

        let lower = if !all_lowers.is_empty() {
            self.combine_values(&all_lowers)
        } else {
            point_forecast.primary().to_vec()
        };

        let upper = if !all_uppers.is_empty() {
            self.combine_values(&all_uppers)
        } else {
            point_forecast.primary().to_vec()
        };

        Ok(Forecast::from_values_with_intervals(
            point_forecast.primary().to_vec(),
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

        // Compute variance from residuals
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        if valid_residuals.is_empty() {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let n = valid_residuals.len() as f64;
        let variance = crate::simd::sum_of_squares(&valid_residuals) / n;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);
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
        match self.method {
            CombinationMethod::Mean => "Ensemble (Mean)",
            CombinationMethod::Median => "Ensemble (Median)",
            CombinationMethod::WeightedMSE => "Ensemble (Weighted MSE)",
            CombinationMethod::Custom => "Ensemble (Custom)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::baseline::{Naive, SimpleMovingAverage};
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    fn make_series() -> TimeSeries {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn ensemble_mean_basic() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ensemble_median() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::Median);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ensemble_weighted_mse() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::WeightedMSE);
        ensemble.fit(&ts).unwrap();

        // Weights should be normalized
        let weights = ensemble.weights();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1");
    }

    #[test]
    fn ensemble_custom_weights() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_weights(vec![0.7, 0.3]);
        ensemble.fit(&ts).unwrap();

        let weights = ensemble.weights();
        assert!((weights[0] - 0.7).abs() < 1e-6);
        assert!((weights[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn ensemble_empty() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![];

        let mut ensemble = Ensemble::new(models);
        assert!(ensemble.fit(&ts).is_err());
    }

    #[test]
    fn ensemble_requires_fit() {
        let models: Vec<Box<dyn Forecaster>> = vec![Box::new(Naive::new())];
        let ensemble = Ensemble::new(models);
        assert!(matches!(
            ensemble.predict(5),
            Err(ForecastError::FitRequired)
        ));
    }

    #[test]
    fn ensemble_zero_horizon() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![Box::new(Naive::new())];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn ensemble_confidence_intervals() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn ensemble_fitted_and_residuals() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        assert!(ensemble.fitted_values().is_some());
        assert!(ensemble.residuals().is_some());
    }

    #[test]
    fn ensemble_name() {
        let mean = Ensemble::new(vec![Box::new(Naive::new())]);
        assert_eq!(mean.name(), "Ensemble (Mean)");

        let median =
            Ensemble::new(vec![Box::new(Naive::new())]).with_method(CombinationMethod::Median);
        assert_eq!(median.name(), "Ensemble (Median)");

        let weighted =
            Ensemble::new(vec![Box::new(Naive::new())]).with_method(CombinationMethod::WeightedMSE);
        assert_eq!(weighted.name(), "Ensemble (Weighted MSE)");
    }

    #[test]
    fn ensemble_model_count() {
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let ensemble = Ensemble::new(models);
        assert_eq!(ensemble.model_count(), 3);
    }

    #[test]
    fn ensemble_mean_is_between_individual_forecasts() {
        let ts = make_series();

        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();
        let naive_fc = naive.predict(5).unwrap();

        let mut sma = SimpleMovingAverage::new(5);
        sma.fit(&ts).unwrap();
        let sma_fc = sma.predict(5).unwrap();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();
        let ensemble_fc = ensemble.predict(5).unwrap();

        // Ensemble mean should be between the individual forecasts
        for i in 0..5 {
            let min_val = naive_fc.primary()[i].min(sma_fc.primary()[i]);
            let max_val = naive_fc.primary()[i].max(sma_fc.primary()[i]);
            assert!(
                ensemble_fc.primary()[i] >= min_val - 1e-10
                    && ensemble_fc.primary()[i] <= max_val + 1e-10,
                "Ensemble forecast should be between individual forecasts"
            );
        }
    }
}
