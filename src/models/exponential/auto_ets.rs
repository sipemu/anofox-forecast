//! Automatic ETS model selection.
//!
//! AutoETS automatically selects the best ETS model specification
//! based on information criteria (AIC, AICc, or BIC).

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::exponential::ets::{ETSSpec, ErrorType, SeasonalType, TrendType, ETS};
use crate::models::{validate_series_complete, Forecaster};
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Selection criterion for AutoETS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Corrected Akaike Information Criterion
    #[default]
    AICc,
    /// Bayesian Information Criterion
    BIC,
}

/// Configuration for AutoETS.
#[derive(Debug, Clone)]
pub struct AutoETSConfig {
    /// Selection criterion to use.
    pub criterion: SelectionCriterion,
    /// Seasonal period (None for automatic detection or non-seasonal).
    pub seasonal_period: Option<usize>,
    /// Allow multiplicative errors.
    pub allow_multiplicative_error: bool,
    /// Allow multiplicative seasonality.
    pub allow_multiplicative_seasonal: bool,
    /// Allow damped trend.
    pub allow_damped: bool,
    /// Restrict to additive models only.
    pub additive_only: bool,
}

impl Default for AutoETSConfig {
    fn default() -> Self {
        Self {
            criterion: SelectionCriterion::AICc,
            seasonal_period: None,
            allow_multiplicative_error: true,
            allow_multiplicative_seasonal: true,
            allow_damped: true,
            additive_only: false,
        }
    }
}

impl AutoETSConfig {
    /// Create a configuration for non-seasonal data.
    pub fn non_seasonal() -> Self {
        Self {
            seasonal_period: Some(1),
            ..Default::default()
        }
    }

    /// Create a configuration with a specific seasonal period.
    pub fn with_period(period: usize) -> Self {
        Self {
            seasonal_period: Some(period),
            ..Default::default()
        }
    }

    /// Restrict to additive models only.
    pub fn additive_only(mut self) -> Self {
        self.additive_only = true;
        self.allow_multiplicative_error = false;
        self.allow_multiplicative_seasonal = false;
        self
    }

    /// Set the selection criterion.
    pub fn with_criterion(mut self, criterion: SelectionCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

/// Automatic ETS model selection.
///
/// Fits multiple ETS models and selects the best one based on
/// information criteria.
#[derive(Debug, Clone)]
pub struct AutoETS {
    /// Configuration.
    config: AutoETSConfig,
    /// Selected model.
    selected_model: Option<ETS>,
    /// Selected specification.
    selected_spec: Option<ETSSpec>,
    /// All fitted models and their scores.
    model_scores: Vec<(ETSSpec, f64)>,
    /// OLS result for exogenous regressors.
    exog_ols: Option<OLSResult>,
}

impl AutoETS {
    /// Create a new AutoETS with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoETSConfig::default(),
            selected_model: None,
            selected_spec: None,
            model_scores: Vec::new(),
            exog_ols: None,
        }
    }

    /// Create a new AutoETS with custom configuration.
    pub fn with_config(config: AutoETSConfig) -> Self {
        Self {
            config,
            selected_model: None,
            selected_spec: None,
            model_scores: Vec::new(),
            exog_ols: None,
        }
    }

    /// Create AutoETS for non-seasonal data.
    pub fn non_seasonal() -> Self {
        Self::with_config(AutoETSConfig::non_seasonal())
    }

    /// Create AutoETS with a specific seasonal period.
    pub fn with_period(period: usize) -> Self {
        Self::with_config(AutoETSConfig::with_period(period))
    }

    /// Get the selected specification.
    pub fn selected_spec(&self) -> Option<ETSSpec> {
        self.selected_spec
    }

    /// Get all model scores.
    pub fn model_scores(&self) -> &[(ETSSpec, f64)] {
        &self.model_scores
    }

    /// Generate candidate model specifications.
    fn generate_candidates(&self, has_seasonal: bool) -> Vec<ETSSpec> {
        let mut candidates = Vec::new();

        let error_types = if self.config.additive_only || !self.config.allow_multiplicative_error {
            vec![ErrorType::Additive]
        } else {
            vec![ErrorType::Additive, ErrorType::Multiplicative]
        };

        let trend_types = if self.config.allow_damped {
            vec![
                TrendType::None,
                TrendType::Additive,
                TrendType::AdditiveDamped,
            ]
        } else {
            vec![TrendType::None, TrendType::Additive]
        };

        let seasonal_types = if !has_seasonal {
            vec![SeasonalType::None]
        } else if self.config.additive_only || !self.config.allow_multiplicative_seasonal {
            vec![SeasonalType::None, SeasonalType::Additive]
        } else {
            vec![
                SeasonalType::None,
                SeasonalType::Additive,
                SeasonalType::Multiplicative,
            ]
        };

        for &error in &error_types {
            for &trend in &trend_types {
                for &seasonal in &seasonal_types {
                    // Skip invalid combinations
                    // Multiplicative errors with additive components can be problematic
                    if error == ErrorType::Multiplicative
                        && (trend == TrendType::Additive || trend == TrendType::AdditiveDamped)
                        && seasonal == SeasonalType::Additive
                    {
                        continue; // M,A,A and M,Ad,A can be unstable
                    }
                    candidates.push(ETSSpec::new(error, trend, seasonal));
                }
            }
        }

        candidates
    }

    /// Get the criterion value from a model.
    fn get_criterion_static(criterion: SelectionCriterion, model: &ETS) -> Option<f64> {
        match criterion {
            SelectionCriterion::AIC => model.aic(),
            SelectionCriterion::AICc => model.aicc(),
            SelectionCriterion::BIC => model.bic(),
        }
    }

    /// Fit and evaluate a single candidate model (static method for parallel use).
    fn evaluate_candidate_static(
        series: &TimeSeries,
        spec: ETSSpec,
        seasonal_period: usize,
        criterion: SelectionCriterion,
    ) -> Option<(ETSSpec, ETS, f64)> {
        let period = if spec.has_seasonal() {
            seasonal_period
        } else {
            1
        };

        let min_len = if spec.has_seasonal() { 2 * period } else { 2 };
        if series.primary_values().len() < min_len {
            return None;
        }

        let mut model = ETS::new(spec, period);
        if model.fit(series).is_ok() {
            if let Some(score) = Self::get_criterion_static(criterion, &model) {
                return Some((spec, model, score));
            }
        }
        None
    }
}

impl Default for AutoETS {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoETS {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let raw_values = series.primary_values();
        if raw_values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: raw_values.len(),
            });
        }

        // Handle exogenous regressors
        let eval_series = if series.has_regressors() {
            let regressors = series.all_regressors();
            let ols_result = ols_fit(raw_values, &regressors)?;
            let adjusted = ols_residuals(raw_values, &ols_result, &regressors)?;
            self.exog_ols = Some(ols_result);
            // Create adjusted series without regressors for candidate evaluation
            TimeSeries::univariate(series.timestamps().to_vec(), adjusted)?
        } else {
            self.exog_ols = None;
            series.clone()
        };

        let values = eval_series.primary_values();

        // Determine seasonal period
        let seasonal_period = self.config.seasonal_period.unwrap_or(1);
        let has_seasonal = seasonal_period > 1 && values.len() >= 2 * seasonal_period;

        // Generate candidate models
        let candidates = self.generate_candidates(has_seasonal);
        self.model_scores.clear();

        let criterion = self.config.criterion;

        // Fit each candidate and track scores (parallel when feature enabled)
        let results: Vec<(ETSSpec, ETS, f64)>;

        #[cfg(feature = "parallel")]
        {
            results = candidates
                .par_iter()
                .filter_map(|&spec| {
                    Self::evaluate_candidate_static(&eval_series, spec, seasonal_period, criterion)
                })
                .collect();
        }

        #[cfg(not(feature = "parallel"))]
        {
            results = candidates
                .iter()
                .filter_map(|&spec| {
                    Self::evaluate_candidate_static(&eval_series, spec, seasonal_period, criterion)
                })
                .collect();
        }

        let mut best_model: Option<ETS> = None;
        let mut best_spec: Option<ETSSpec> = None;
        let mut best_score = f64::INFINITY;

        for (spec, model, score) in results {
            self.model_scores.push((spec, score));
            if score < best_score {
                best_score = score;
                best_model = Some(model);
                best_spec = Some(spec);
            }
        }

        // Sort model scores
        self.model_scores
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        self.selected_model = best_model;
        self.selected_spec = best_spec;

        if self.selected_model.is_none() {
            return Err(ForecastError::ComputationError(
                "No valid ETS model could be fitted".to_string(),
            ));
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        model.predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog_intervals() and provide future regressor values.".into()
            ));
        }
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        model.predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.selected_model.as_ref()?.fitted_values()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        self.selected_model
            .as_ref()?
            .fitted_values_with_intervals(level)
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.selected_model.as_ref()?.residuals()
    }

    fn name(&self) -> &str {
        "AutoETS"
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        self.exog_ols.is_some()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.exog_ols
            .as_ref()
            .map(|ols| ols.regressor_names.as_slice())
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        // Get base forecast from selected ETS model (fit on adjusted series)
        let base_forecast = model.predict(horizon)?;

        if let Some(ols) = &self.exog_ols {
            // Validate and compute exogenous contribution
            for name in &ols.regressor_names {
                let values = future_regressors.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            let exog_pred = ols.predict(future_regressors)?;
            let adjusted: Vec<f64> = base_forecast
                .primary()
                .iter()
                .zip(exog_pred.iter())
                .map(|(b, e)| b + e)
                .collect();
            Ok(Forecast::from_values(adjusted))
        } else {
            if !future_regressors.is_empty() {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            Ok(base_forecast)
        }
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        // Get base forecast with intervals from selected ETS model
        let base_forecast = model.predict_with_intervals(horizon, level)?;

        if let Some(ols) = &self.exog_ols {
            for name in &ols.regressor_names {
                let values = future_regressors.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            let exog_pred = ols.predict(future_regressors)?;

            let preds: Vec<f64> = base_forecast
                .primary()
                .iter()
                .zip(exog_pred.iter())
                .map(|(b, e)| b + e)
                .collect();

            let lower = if base_forecast.has_lower() {
                base_forecast
                    .lower_series(0)
                    .unwrap()
                    .iter()
                    .zip(exog_pred.iter())
                    .map(|(b, e)| b + e)
                    .collect()
            } else {
                preds.clone()
            };

            let upper = if base_forecast.has_upper() {
                base_forecast
                    .upper_series(0)
                    .unwrap()
                    .iter()
                    .zip(exog_pred.iter())
                    .map(|(b, e)| b + e)
                    .collect()
            } else {
                preds.clone()
            };

            Ok(Forecast::from_values_with_intervals(preds, lower, upper))
        } else {
            if !future_regressors.is_empty() {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            Ok(base_forecast)
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

    #[test]
    fn auto_ets_selects_model() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        model.fit(&ts).unwrap();

        assert!(model.selected_spec().is_some());
        assert!(!model.model_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_ets_with_trend() {
        let timestamps = make_timestamps(40);
        let values: Vec<f64> = (0..40).map(|i| 10.0 + 1.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        model.fit(&ts).unwrap();

        let spec = model.selected_spec().unwrap();
        // Should select a model with trend
        assert!(spec.has_trend());
    }

    #[test]
    fn auto_ets_with_seasonality() {
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 10.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::with_period(12);
        model.fit(&ts).unwrap();

        let spec = model.selected_spec().unwrap();
        // Should select a model with seasonality
        assert!(spec.has_seasonal());
    }

    #[test]
    fn auto_ets_additive_only() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoETSConfig::non_seasonal().additive_only();
        let mut model = AutoETS::with_config(config);
        model.fit(&ts).unwrap();

        let spec = model.selected_spec().unwrap();
        assert_eq!(spec.error, ErrorType::Additive);
    }

    #[test]
    fn auto_ets_model_scores_sorted() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        model.fit(&ts).unwrap();

        let scores = model.model_scores();
        assert!(!scores.is_empty());

        // Check that scores are sorted ascending
        for i in 1..scores.len() {
            assert!(scores[i].1 >= scores[i - 1].1);
        }
    }

    #[test]
    fn auto_ets_confidence_intervals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn auto_ets_fitted_and_residuals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_ets_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoETS::non_seasonal();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_ets_requires_fit() {
        let model = AutoETS::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn auto_ets_different_criteria() {
        let timestamps = make_timestamps(40);
        let values: Vec<f64> = (0..40).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config_aic = AutoETSConfig::non_seasonal().with_criterion(SelectionCriterion::AIC);
        let config_bic = AutoETSConfig::non_seasonal().with_criterion(SelectionCriterion::BIC);

        let mut model_aic = AutoETS::with_config(config_aic);
        let mut model_bic = AutoETS::with_config(config_bic);

        model_aic.fit(&ts).unwrap();
        model_bic.fit(&ts).unwrap();

        // Both should select valid models (may or may not be the same)
        assert!(model_aic.selected_spec().is_some());
        assert!(model_bic.selected_spec().is_some());
    }

    #[test]
    fn auto_ets_name() {
        let model = AutoETS::new();
        assert_eq!(model.name(), "AutoETS");
    }

    #[test]
    fn auto_ets_default() {
        let model = AutoETS::default();
        assert!(model.selected_model.is_none());
        assert!(model.selected_spec.is_none());
    }
}
