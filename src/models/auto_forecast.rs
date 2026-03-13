//! Unified automatic model selection across ARIMA, ETS, and Theta families.
//!
//! `AutoForecast` fits all enabled auto models (AutoARIMA, AutoETS, AutoTheta)
//! and selects the best one based on in-sample MSE or cross-validation error.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::AutoARIMA;
use crate::models::exponential::AutoETS;
use crate::models::theta::AutoTheta;
use crate::models::{validate_series_complete, Forecaster};
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Strategy for selecting the best model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionStrategy {
    /// Compare models by in-sample MSE of residuals (fast, default).
    #[default]
    InSampleMSE,
    /// Compare models by cross-validation error (more robust, slower).
    CrossValidation,
}

/// Configuration for AutoForecast.
#[derive(Debug, Clone)]
pub struct AutoForecastConfig {
    /// Seasonal period (None for non-seasonal data).
    pub seasonal_period: Option<usize>,
    /// Include AutoARIMA in the candidate set.
    pub include_arima: bool,
    /// Include AutoETS in the candidate set.
    pub include_ets: bool,
    /// Include AutoTheta in the candidate set.
    pub include_theta: bool,
    /// Selection strategy for comparing models.
    pub selection: SelectionStrategy,
}

impl Default for AutoForecastConfig {
    fn default() -> Self {
        Self {
            seasonal_period: None,
            include_arima: true,
            include_ets: true,
            include_theta: true,
            selection: SelectionStrategy::InSampleMSE,
        }
    }
}

impl AutoForecastConfig {
    /// Create a config with a specific seasonal period.
    pub fn with_period(period: usize) -> Self {
        Self {
            seasonal_period: Some(period),
            ..Default::default()
        }
    }

    /// Set the selection strategy.
    pub fn with_selection(mut self, strategy: SelectionStrategy) -> Self {
        self.selection = strategy;
        self
    }

    /// Disable AutoARIMA.
    pub fn without_arima(mut self) -> Self {
        self.include_arima = false;
        self
    }

    /// Disable AutoETS.
    pub fn without_ets(mut self) -> Self {
        self.include_ets = false;
        self
    }

    /// Disable AutoTheta.
    pub fn without_theta(mut self) -> Self {
        self.include_theta = false;
        self
    }
}

/// Internal enum holding the selected model. Using an enum keeps Clone and Debug.
#[derive(Debug, Clone)]
enum SelectedAutoModel {
    ARIMA(AutoARIMA),
    ETS(AutoETS),
    Theta(AutoTheta),
}

/// Unified automatic model selection across ARIMA, ETS, and Theta families.
///
/// `AutoForecast` fits all enabled auto models and selects the best one
/// based on the configured selection strategy.
///
/// # Example
/// ```
/// use anofox_forecast::models::auto_forecast::{AutoForecast, AutoForecastConfig};
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
/// let mut model = AutoForecast::new();
/// model.fit(&ts).unwrap();
///
/// println!("Selected: {}", model.selected_model_name().unwrap());
/// let forecast = model.predict(5).unwrap();
/// assert_eq!(forecast.horizon(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct AutoForecast {
    config: AutoForecastConfig,
    selected: Option<SelectedAutoModel>,
    scores: Vec<(String, f64)>,
}

/// Builder for constructing an [`AutoForecast`] model with custom parameters.
///
/// # Example
/// ```
/// use anofox_forecast::models::auto_forecast::AutoForecast;
///
/// let model = AutoForecast::builder()
///     .seasonal_period(12)
///     .include_arima(true)
///     .include_ets(true)
///     .include_theta(false)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct AutoForecastBuilder {
    seasonal_period: Option<usize>,
    include_arima: Option<bool>,
    include_ets: Option<bool>,
    include_theta: Option<bool>,
    selection: Option<SelectionStrategy>,
}

impl AutoForecastBuilder {
    /// Create a new builder with all defaults.
    fn new() -> Self {
        Self {
            seasonal_period: None,
            include_arima: None,
            include_ets: None,
            include_theta: None,
            selection: None,
        }
    }

    /// Set the seasonal period.
    pub fn seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = Some(period);
        self
    }

    /// Include or exclude AutoARIMA from the candidate set.
    pub fn include_arima(mut self, include: bool) -> Self {
        self.include_arima = Some(include);
        self
    }

    /// Include or exclude AutoETS from the candidate set.
    pub fn include_ets(mut self, include: bool) -> Self {
        self.include_ets = Some(include);
        self
    }

    /// Include or exclude AutoTheta from the candidate set.
    pub fn include_theta(mut self, include: bool) -> Self {
        self.include_theta = Some(include);
        self
    }

    /// Set the model selection strategy.
    pub fn selection(mut self, strategy: SelectionStrategy) -> Self {
        self.selection = Some(strategy);
        self
    }

    /// Build the AutoForecast model.
    pub fn build(self) -> AutoForecast {
        let config = AutoForecastConfig {
            seasonal_period: self.seasonal_period,
            include_arima: self.include_arima.unwrap_or(true),
            include_ets: self.include_ets.unwrap_or(true),
            include_theta: self.include_theta.unwrap_or(true),
            selection: self.selection.unwrap_or_default(),
        };

        AutoForecast::with_config(config)
    }
}

impl AutoForecast {
    /// Create a builder for constructing an AutoForecast model.
    pub fn builder() -> AutoForecastBuilder {
        AutoForecastBuilder::new()
    }

    /// Create a new AutoForecast with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoForecastConfig::default(),
            selected: None,
            scores: Vec::new(),
        }
    }

    /// Create a new AutoForecast with custom configuration.
    pub fn with_config(config: AutoForecastConfig) -> Self {
        Self {
            config,
            selected: None,
            scores: Vec::new(),
        }
    }

    /// Create a seasonal AutoForecast.
    pub fn seasonal(period: usize) -> Self {
        Self::with_config(AutoForecastConfig::with_period(period))
    }

    /// Get the name of the selected model, or None if not yet fitted.
    pub fn selected_model_name(&self) -> Option<&str> {
        self.selected.as_ref().map(|m| match m {
            SelectedAutoModel::ARIMA(model) => model.name(),
            SelectedAutoModel::ETS(model) => model.name(),
            SelectedAutoModel::Theta(model) => model.name(),
        })
    }

    /// Get all candidate scores as (model_name, score) pairs, sorted ascending.
    pub fn all_scores(&self) -> &[(String, f64)] {
        &self.scores
    }

    /// Calculate MSE from residuals.
    fn calculate_mse(residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return f64::MAX;
        }
        let n = residuals.len() as f64;
        residuals.iter().map(|r| r * r).sum::<f64>() / n
    }

    /// Fit using in-sample MSE comparison.
    fn fit_in_sample(&mut self, series: &TimeSeries) -> Result<()> {
        // Build a list of factory closures that each create, fit, and score a candidate.
        // Models are created and consumed within the closure so non-Send types never
        // cross thread boundaries.
        let seasonal_period = self.config.seasonal_period;

        let mut factories: Vec<
            Box<dyn Fn(&TimeSeries) -> Option<(SelectedAutoModel, String, f64)> + Send + Sync>,
        > = Vec::new();

        if self.config.include_arima {
            factories.push(Box::new(move |ts: &TimeSeries| {
                let mut model = match seasonal_period {
                    Some(p) if p > 1 => AutoARIMA::seasonal(p),
                    _ => AutoARIMA::new(),
                };
                model.fit(ts).ok()?;
                let residuals = model.residuals()?;
                let mse = Self::calculate_mse(residuals);
                if mse.is_finite() {
                    let name = model.name().to_string();
                    Some((SelectedAutoModel::ARIMA(model), name, mse))
                } else {
                    None
                }
            }));
        }

        if self.config.include_ets {
            factories.push(Box::new(move |ts: &TimeSeries| {
                let mut model = match seasonal_period {
                    Some(p) if p > 1 => AutoETS::with_period(p),
                    _ => AutoETS::new(),
                };
                model.fit(ts).ok()?;
                let residuals = model.residuals()?;
                let mse = Self::calculate_mse(residuals);
                if mse.is_finite() {
                    let name = model.name().to_string();
                    Some((SelectedAutoModel::ETS(model), name, mse))
                } else {
                    None
                }
            }));
        }

        if self.config.include_theta {
            factories.push(Box::new(move |ts: &TimeSeries| {
                let mut model = match seasonal_period {
                    Some(p) if p > 1 => AutoTheta::seasonal(p),
                    _ => AutoTheta::new(),
                };
                model.fit(ts).ok()?;
                let residuals = model.residuals()?;
                let mse = Self::calculate_mse(residuals);
                if mse.is_finite() {
                    let name = model.name().to_string();
                    Some((SelectedAutoModel::Theta(model), name, mse))
                } else {
                    None
                }
            }));
        }

        #[cfg(feature = "parallel")]
        let mut candidates: Vec<(SelectedAutoModel, String, f64)> =
            factories.par_iter().filter_map(|f| f(series)).collect();

        #[cfg(not(feature = "parallel"))]
        let mut candidates: Vec<(SelectedAutoModel, String, f64)> =
            factories.iter().filter_map(|f| f(series)).collect();

        if candidates.is_empty() {
            return Err(ForecastError::ConvergenceFailure(
                "No candidate model could be fitted".to_string(),
            ));
        }

        // Sort by MSE
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Store all scores
        self.scores = candidates.iter().map(|(_, n, s)| (n.clone(), *s)).collect();

        // Select the best
        let (best_model, _, _) = candidates.into_iter().next().unwrap();
        self.selected = Some(best_model);

        Ok(())
    }

    /// Fit using cross-validation comparison.
    fn fit_cross_validation(&mut self, series: &TimeSeries) -> Result<()> {
        use crate::utils::cross_validation::{cross_validate, CVConfig};

        let n = series.len();
        // Use a reasonable CV config: expanding window, horizon = seasonal_period or 5
        let horizon = self
            .config
            .seasonal_period
            .filter(|&p| p > 1)
            .unwrap_or(5)
            .min(n / 4)
            .max(1);
        let initial_window = (n / 2).max(10).min(n - horizon);
        let step_size = horizon.max(1);

        let cv_config = CVConfig::expanding(initial_window, horizon).with_step_size(step_size);

        // Build a list of factory closures that each run CV and refit on full data.
        // Models are created and consumed within the closure so non-Send types never
        // cross thread boundaries.
        let seasonal_period = self.config.seasonal_period;

        let mut factories: Vec<
            Box<
                dyn Fn(&CVConfig, &TimeSeries) -> Option<(SelectedAutoModel, String, f64)>
                    + Send
                    + Sync,
            >,
        > = Vec::new();

        if self.config.include_arima {
            factories.push(Box::new(move |cv_cfg: &CVConfig, ts: &TimeSeries| {
                let period = seasonal_period;
                let cv_result = cross_validate(cv_cfg, ts, move || match period {
                    Some(p) if p > 1 => AutoARIMA::seasonal(p),
                    _ => AutoARIMA::new(),
                });
                if let Ok(results) = cv_result {
                    if results.n_folds > 0 && results.aggregated.rmse.is_finite() {
                        let mut model = match period {
                            Some(p) if p > 1 => AutoARIMA::seasonal(p),
                            _ => AutoARIMA::new(),
                        };
                        if model.fit(ts).is_ok() {
                            let name = model.name().to_string();
                            return Some((
                                SelectedAutoModel::ARIMA(model),
                                name,
                                results.aggregated.rmse,
                            ));
                        }
                    }
                }
                None
            }));
        }

        if self.config.include_ets {
            factories.push(Box::new(move |cv_cfg: &CVConfig, ts: &TimeSeries| {
                let period = seasonal_period;
                let cv_result = cross_validate(cv_cfg, ts, move || match period {
                    Some(p) if p > 1 => AutoETS::with_period(p),
                    _ => AutoETS::new(),
                });
                if let Ok(results) = cv_result {
                    if results.n_folds > 0 && results.aggregated.rmse.is_finite() {
                        let mut model = match period {
                            Some(p) if p > 1 => AutoETS::with_period(p),
                            _ => AutoETS::new(),
                        };
                        if model.fit(ts).is_ok() {
                            let name = model.name().to_string();
                            return Some((
                                SelectedAutoModel::ETS(model),
                                name,
                                results.aggregated.rmse,
                            ));
                        }
                    }
                }
                None
            }));
        }

        if self.config.include_theta {
            factories.push(Box::new(move |cv_cfg: &CVConfig, ts: &TimeSeries| {
                let period = seasonal_period;
                let cv_result = cross_validate(cv_cfg, ts, move || match period {
                    Some(p) if p > 1 => AutoTheta::seasonal(p),
                    _ => AutoTheta::new(),
                });
                if let Ok(results) = cv_result {
                    if results.n_folds > 0 && results.aggregated.rmse.is_finite() {
                        let mut model = match period {
                            Some(p) if p > 1 => AutoTheta::seasonal(p),
                            _ => AutoTheta::new(),
                        };
                        if model.fit(ts).is_ok() {
                            let name = model.name().to_string();
                            return Some((
                                SelectedAutoModel::Theta(model),
                                name,
                                results.aggregated.rmse,
                            ));
                        }
                    }
                }
                None
            }));
        }

        #[cfg(feature = "parallel")]
        let mut cv_scores: Vec<(SelectedAutoModel, String, f64)> = factories
            .par_iter()
            .filter_map(|f| f(&cv_config, series))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let mut cv_scores: Vec<(SelectedAutoModel, String, f64)> = factories
            .iter()
            .filter_map(|f| f(&cv_config, series))
            .collect();

        if cv_scores.is_empty() {
            return Err(ForecastError::ConvergenceFailure(
                "No candidate model produced valid cross-validation results".to_string(),
            ));
        }

        // Sort by CV RMSE
        cv_scores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        self.scores = cv_scores.iter().map(|(_, n, s)| (n.clone(), *s)).collect();

        let (best_model, _, _) = cv_scores.into_iter().next().unwrap();
        self.selected = Some(best_model);

        Ok(())
    }
}

impl Default for AutoForecast {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoForecast {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;

        if series.len() < 10 {
            return Err(ForecastError::InsufficientData {
                needed: 10,
                got: series.len(),
                hint: Some(
                    "AutoForecast requires at least 10 observations for model comparison".into(),
                ),
            });
        }

        self.selected = None;
        self.scores.clear();

        match self.config.selection {
            SelectionStrategy::InSampleMSE => self.fit_in_sample(series),
            SelectionStrategy::CrossValidation => self.fit_cross_validation(series),
        }
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        match self.selected.as_ref() {
            Some(SelectedAutoModel::ARIMA(m)) => m.predict(horizon),
            Some(SelectedAutoModel::ETS(m)) => m.predict(horizon),
            Some(SelectedAutoModel::Theta(m)) => m.predict(horizon),
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        match self.selected.as_ref() {
            Some(SelectedAutoModel::ARIMA(m)) => m.predict_with_intervals(horizon, level),
            Some(SelectedAutoModel::ETS(m)) => m.predict_with_intervals(horizon, level),
            Some(SelectedAutoModel::Theta(m)) => m.predict_with_intervals(horizon, level),
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        match self.selected.as_ref()? {
            SelectedAutoModel::ARIMA(m) => m.fitted_values(),
            SelectedAutoModel::ETS(m) => m.fitted_values(),
            SelectedAutoModel::Theta(m) => m.fitted_values(),
        }
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        match self.selected.as_ref()? {
            SelectedAutoModel::ARIMA(m) => m.fitted_values_with_intervals(level),
            SelectedAutoModel::ETS(m) => m.fitted_values_with_intervals(level),
            SelectedAutoModel::Theta(m) => m.fitted_values_with_intervals(level),
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        match self.selected.as_ref()? {
            SelectedAutoModel::ARIMA(m) => m.residuals(),
            SelectedAutoModel::ETS(m) => m.residuals(),
            SelectedAutoModel::Theta(m) => m.residuals(),
        }
    }

    fn name(&self) -> &str {
        "AutoForecast"
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        match self.selected.as_ref() {
            Some(SelectedAutoModel::ARIMA(m)) => m.has_exog(),
            Some(SelectedAutoModel::ETS(m)) => m.has_exog(),
            Some(SelectedAutoModel::Theta(m)) => m.has_exog(),
            None => false,
        }
    }

    fn exog_names(&self) -> Option<&[String]> {
        match self.selected.as_ref()? {
            SelectedAutoModel::ARIMA(m) => m.exog_names(),
            SelectedAutoModel::ETS(m) => m.exog_names(),
            SelectedAutoModel::Theta(m) => m.exog_names(),
        }
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        match self.selected.as_ref() {
            Some(SelectedAutoModel::ARIMA(m)) => m.predict_with_exog(horizon, future_regressors),
            Some(SelectedAutoModel::ETS(m)) => m.predict_with_exog(horizon, future_regressors),
            Some(SelectedAutoModel::Theta(m)) => m.predict_with_exog(horizon, future_regressors),
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        match self.selected.as_ref() {
            Some(SelectedAutoModel::ARIMA(m)) => {
                m.predict_with_exog_intervals(horizon, future_regressors, level)
            }
            Some(SelectedAutoModel::ETS(m)) => {
                m.predict_with_exog_intervals(horizon, future_regressors, level)
            }
            Some(SelectedAutoModel::Theta(m)) => {
                m.predict_with_exog_intervals(horizon, future_regressors, level)
            }
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }
}

impl fmt::Display for AutoForecast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.selected_model_name() {
            Some(name) => {
                writeln!(f, "AutoForecast (selected: {})", name)?;
                writeln!(f, "Candidate scores:")?;
                for (model_name, score) in &self.scores {
                    writeln!(f, "  {}: {:.4}", model_name, score)?;
                }
                Ok(())
            }
            None => write!(f, "AutoForecast (not fitted)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    fn make_trend_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                50.0 + 0.3 * i as f64
                    + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn auto_forecast_basic_fit_predict() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_model_name().is_some());
        assert!(!model.all_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_forecast_selects_across_families() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        // Should have scores from multiple model families
        let scores = model.all_scores();
        assert!(
            scores.len() >= 2,
            "Expected at least 2 candidates, got {}",
            scores.len()
        );
    }

    #[test]
    fn auto_forecast_scores_sorted() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        for i in 1..scores.len() {
            assert!(
                scores[i].1 >= scores[i - 1].1,
                "Scores not sorted: {} > {}",
                scores[i - 1].1,
                scores[i].1
            );
        }
    }

    #[test]
    fn auto_forecast_confidence_intervals() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn auto_forecast_fitted_and_residuals() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_forecast_requires_fit() {
        let model = AutoForecast::new();
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn auto_forecast_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoForecast::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_forecast_name() {
        let model = AutoForecast::new();
        assert_eq!(model.name(), "AutoForecast");
    }

    #[test]
    fn auto_forecast_default() {
        let model = AutoForecast::default();
        assert!(model.selected_model_name().is_none());
        assert!(model.all_scores().is_empty());
    }

    #[test]
    fn auto_forecast_seasonal() {
        let ts = make_seasonal_series(100, 12);
        let mut model = AutoForecast::seasonal(12);
        model.fit(&ts).unwrap();

        assert!(model.selected_model_name().is_some());
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn auto_forecast_arima_only() {
        let ts = make_trend_series(100);
        let config = AutoForecastConfig::default().without_ets().without_theta();
        let mut model = AutoForecast::with_config(config);
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].0.contains("AutoARIMA"));
    }

    #[test]
    fn auto_forecast_ets_only() {
        let ts = make_trend_series(100);
        let config = AutoForecastConfig::default()
            .without_arima()
            .without_theta();
        let mut model = AutoForecast::with_config(config);
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].0.contains("AutoETS"));
    }

    #[test]
    fn auto_forecast_theta_only() {
        let ts = make_trend_series(100);
        let config = AutoForecastConfig::default().without_arima().without_ets();
        let mut model = AutoForecast::with_config(config);
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].0.contains("AutoTheta"));
    }

    #[test]
    fn auto_forecast_display_fitted() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        let display = format!("{}", model);
        assert!(display.contains("AutoForecast"));
        assert!(display.contains("selected:"));
        assert!(display.contains("Candidate scores:"));
    }

    #[test]
    fn auto_forecast_display_not_fitted() {
        let model = AutoForecast::new();
        let display = format!("{}", model);
        assert_eq!(display, "AutoForecast (not fitted)");
    }

    #[test]
    fn auto_forecast_config_with_selection() {
        let config =
            AutoForecastConfig::default().with_selection(SelectionStrategy::CrossValidation);
        assert_eq!(config.selection, SelectionStrategy::CrossValidation);
    }

    #[test]
    fn auto_forecast_cross_validation_strategy() {
        let ts = make_trend_series(100);
        let config =
            AutoForecastConfig::default().with_selection(SelectionStrategy::CrossValidation);
        let mut model = AutoForecast::with_config(config);
        model.fit(&ts).unwrap();

        assert!(model.selected_model_name().is_some());
        assert!(!model.all_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_forecast_refit_clears_state() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();

        model.fit(&ts).unwrap();
        let first_name = model.selected_model_name().unwrap().to_string();
        let first_scores_len = model.all_scores().len();

        // Refit on the same data -- state should be reset cleanly
        model.fit(&ts).unwrap();
        assert!(model.selected_model_name().is_some());
        // Scores count should be the same (same data, same candidates)
        assert_eq!(model.all_scores().len(), first_scores_len);
        // Selected model should be the same on identical data
        assert_eq!(model.selected_model_name().unwrap(), first_name);
    }

    #[test]
    fn auto_forecast_clone() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::new();
        model.fit(&ts).unwrap();

        let cloned = model.clone();
        assert_eq!(cloned.selected_model_name(), model.selected_model_name());
        assert_eq!(cloned.all_scores().len(), model.all_scores().len());

        let forecast = cloned.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_forecast_no_candidates_enabled() {
        let ts = make_trend_series(100);
        let config = AutoForecastConfig::default()
            .without_arima()
            .without_ets()
            .without_theta();
        let mut model = AutoForecast::with_config(config);

        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::ConvergenceFailure(_))
        ));
    }

    #[test]
    fn auto_forecast_builder_defaults() {
        let model = AutoForecast::builder().build();
        assert!(model.selected_model_name().is_none());
        assert_eq!(model.name(), "AutoForecast");
    }

    #[test]
    fn auto_forecast_builder_custom() {
        let model = AutoForecast::builder()
            .seasonal_period(12)
            .include_arima(true)
            .include_ets(true)
            .include_theta(false)
            .build();

        let ts = make_seasonal_series(100, 12);
        let mut model = model;
        model.fit(&ts).unwrap();

        assert!(model.selected_model_name().is_some());
        let scores = model.all_scores();
        // Theta was excluded, so should have at most 2 candidates
        assert!(scores.len() <= 2);
        for (name, _) in scores {
            assert!(!name.contains("AutoTheta"));
        }
    }

    #[test]
    fn auto_forecast_builder_fit_predict() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::builder()
            .include_arima(true)
            .include_ets(true)
            .build();

        model.fit(&ts).unwrap();
        assert!(model.selected_model_name().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_forecast_builder_arima_only() {
        let ts = make_trend_series(100);
        let mut model = AutoForecast::builder()
            .include_arima(true)
            .include_ets(false)
            .include_theta(false)
            .build();

        model.fit(&ts).unwrap();
        let scores = model.all_scores();
        assert_eq!(scores.len(), 1);
        assert!(scores[0].0.contains("AutoARIMA"));
    }

    #[test]
    fn auto_forecast_builder_with_selection() {
        let model = AutoForecast::builder()
            .selection(SelectionStrategy::CrossValidation)
            .build();
        // Just verify it builds without error
        assert_eq!(model.name(), "AutoForecast");
    }

    #[test]
    fn auto_forecast_builder_seasonal() {
        let ts = make_seasonal_series(100, 12);
        let mut model = AutoForecast::builder().seasonal_period(12).build();

        model.fit(&ts).unwrap();
        assert!(model.selected_model_name().is_some());

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }
}
