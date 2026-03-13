//! Automatic ensemble construction from top-K models.
//!
//! `AutoEnsemble` uses `AutoForecast` infrastructure to fit multiple model
//! families, then combines the top-K performing models into an ensemble.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::AutoARIMA;
use crate::models::ensemble::model::{CombinationMethod, Ensemble};
use crate::models::exponential::AutoETS;
use crate::models::theta::AutoTheta;
use crate::models::{validate_series_complete, Forecaster};

/// Configuration for automatic ensemble construction.
#[derive(Debug, Clone)]
pub struct AutoEnsembleConfig {
    /// Number of top models to include in the ensemble.
    pub top_k: usize,
    /// Combination method for the ensemble.
    pub combination_method: CombinationMethod,
    /// Seasonal period (None for non-seasonal).
    pub seasonal_period: Option<usize>,
}

impl Default for AutoEnsembleConfig {
    fn default() -> Self {
        Self {
            top_k: 3,
            combination_method: CombinationMethod::WeightedMSE,
            seasonal_period: None,
        }
    }
}

impl AutoEnsembleConfig {
    /// Create a config with a specific seasonal period.
    pub fn with_period(period: usize) -> Self {
        Self {
            seasonal_period: Some(period),
            ..Default::default()
        }
    }

    /// Set the number of top models to include.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }

    /// Set the combination method.
    pub fn with_method(mut self, method: CombinationMethod) -> Self {
        self.combination_method = method;
        self
    }
}

/// Automatic ensemble that selects top-K models across families.
///
/// Fits AutoARIMA, AutoETS, and AutoTheta, ranks them by in-sample MSE,
/// and combines the top-K into an ensemble.
///
/// # Example
/// ```
/// use anofox_forecast::models::ensemble::AutoEnsemble;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..60).map(|i| {
///     Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)
/// }).collect();
/// let values: Vec<f64> = (0..60).map(|i| 10.0 + i as f64 * 0.5).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = AutoEnsemble::new();
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(5).unwrap();
/// ```
pub struct AutoEnsemble {
    config: AutoEnsembleConfig,
    ensemble: Option<Ensemble>,
    /// Scores of all candidates (name, MSE), sorted ascending.
    scores: Vec<(String, f64)>,
}

impl AutoEnsemble {
    /// Create a new AutoEnsemble with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoEnsembleConfig::default(),
            ensemble: None,
            scores: Vec::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: AutoEnsembleConfig) -> Self {
        Self {
            config,
            ensemble: None,
            scores: Vec::new(),
        }
    }

    /// Create a seasonal AutoEnsemble.
    pub fn seasonal(period: usize) -> Self {
        Self::with_config(AutoEnsembleConfig::with_period(period))
    }

    /// Get all candidate scores.
    pub fn all_scores(&self) -> &[(String, f64)] {
        &self.scores
    }

    /// Get the number of models in the final ensemble.
    pub fn model_count(&self) -> usize {
        self.ensemble.as_ref().map_or(0, |e| e.model_count())
    }

    fn calculate_mse(residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return f64::MAX;
        }
        residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64
    }
}

impl Default for AutoEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoEnsemble {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;

        let mut candidates: Vec<(Box<dyn Forecaster>, String, f64)> = Vec::new();

        // Fit AutoARIMA
        {
            let mut model = match self.config.seasonal_period {
                Some(p) if p > 1 => AutoARIMA::seasonal(p),
                _ => AutoARIMA::new(),
            };
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    if mse.is_finite() {
                        let name = model.name().to_string();
                        candidates.push((Box::new(model), name, mse));
                    }
                }
            }
        }

        // Fit AutoETS
        {
            let mut model = match self.config.seasonal_period {
                Some(p) if p > 1 => AutoETS::with_period(p),
                _ => AutoETS::new(),
            };
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    if mse.is_finite() {
                        let name = model.name().to_string();
                        candidates.push((Box::new(model), name, mse));
                    }
                }
            }
        }

        // Fit AutoTheta
        {
            let mut model = match self.config.seasonal_period {
                Some(p) if p > 0 => AutoTheta::seasonal(p),
                _ => AutoTheta::new(),
            };
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    if mse.is_finite() {
                        let name = model.name().to_string();
                        candidates.push((Box::new(model), name, mse));
                    }
                }
            }
        }

        if candidates.is_empty() {
            return Err(ForecastError::ConvergenceFailure(
                "No models could be fitted successfully".into(),
            ));
        }

        // Sort by MSE ascending
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Store scores
        self.scores = candidates.iter().map(|(_, n, s)| (n.clone(), *s)).collect();

        // Take top-K
        let top_k = self.config.top_k.min(candidates.len());
        let top_models: Vec<Box<dyn Forecaster>> = candidates
            .into_iter()
            .take(top_k)
            .map(|(m, _, _)| m)
            .collect();

        let ensemble = Ensemble::new(top_models).with_method(self.config.combination_method);
        // Ensemble is already fitted since all sub-models are fitted
        // We need to call fit to compute combined fitted/residuals
        self.ensemble = Some(ensemble);

        // Re-fit the ensemble to compute combined values
        if let Some(ref mut ens) = self.ensemble {
            ens.fit(series)?;
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        self.ensemble
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?
            .predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        self.ensemble
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?
            .predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.ensemble.as_ref()?.fitted_values()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.ensemble.as_ref()?.residuals()
    }

    fn name(&self) -> &str {
        "AutoEnsemble"
    }

    fn is_fitted(&self) -> bool {
        self.ensemble.as_ref().is_some_and(|e| e.is_fitted())
    }
}

impl std::fmt::Display for AutoEnsemble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AutoEnsemble (top-{})", self.config.top_k)?;
        if !self.scores.is_empty() {
            for (name, score) in &self.scores {
                write!(f, "\n  {}: MSE={:.4}", name, score)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn make_test_series(n: usize) -> TimeSeries {
        let timestamps: Vec<_> = (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect();
        let values: Vec<f64> = (0..n).map(|i| 10.0 + i as f64 * 0.3).collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn auto_ensemble_basic() {
        let ts = make_test_series(60);
        let mut model = AutoEnsemble::new();
        model.fit(&ts).unwrap();

        assert!(model.is_fitted());
        assert!(model.model_count() > 0);
        assert!(!model.all_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_ensemble_custom_top_k() {
        let ts = make_test_series(60);
        let config = AutoEnsembleConfig::default().with_top_k(2);
        let mut model = AutoEnsemble::with_config(config);
        model.fit(&ts).unwrap();

        assert!(model.model_count() <= 2);
    }

    #[test]
    fn auto_ensemble_display() {
        let ts = make_test_series(60);
        let mut model = AutoEnsemble::new();
        model.fit(&ts).unwrap();

        let display = format!("{}", model);
        assert!(display.contains("AutoEnsemble"));
    }

    #[test]
    fn auto_ensemble_seasonal() {
        let ts = make_test_series(60);
        let mut model = AutoEnsemble::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }
}
