//! Automatic ARIMA model selection.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::diff::suggest_differencing;
use crate::models::arima::model::ARIMA;
use crate::models::Forecaster;

/// Configuration for AutoARIMA.
#[derive(Debug, Clone)]
pub struct AutoARIMAConfig {
    /// Maximum AR order to consider.
    pub max_p: usize,
    /// Maximum MA order to consider.
    pub max_q: usize,
    /// Maximum differencing order.
    pub max_d: usize,
    /// Use stepwise search (faster) vs exhaustive.
    pub stepwise: bool,
    /// Selection criterion (use AIC for selection).
    pub use_aic: bool,
}

impl Default for AutoARIMAConfig {
    fn default() -> Self {
        Self {
            max_p: 3,
            max_q: 3,
            max_d: 2,
            stepwise: true,
            use_aic: true,
        }
    }
}

impl AutoARIMAConfig {
    /// Set maximum orders.
    pub fn with_max_orders(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_p = max_p;
        self.max_d = max_d;
        self.max_q = max_q;
        self
    }

    /// Use exhaustive search instead of stepwise.
    pub fn exhaustive(mut self) -> Self {
        self.stepwise = false;
        self
    }
}

/// Automatic ARIMA model selection.
///
/// Automatically selects the best ARIMA(p, d, q) specification
/// based on information criteria.
#[derive(Debug, Clone)]
pub struct AutoARIMA {
    /// Configuration.
    config: AutoARIMAConfig,
    /// Selected model.
    selected_model: Option<ARIMA>,
    /// Selected orders.
    selected_order: Option<(usize, usize, usize)>,
    /// All fitted models and their scores.
    model_scores: Vec<((usize, usize, usize), f64)>,
}

impl AutoARIMA {
    /// Create a new AutoARIMA with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoARIMAConfig::default(),
            selected_model: None,
            selected_order: None,
            model_scores: Vec::new(),
        }
    }

    /// Create AutoARIMA with custom configuration.
    pub fn with_config(config: AutoARIMAConfig) -> Self {
        Self {
            config,
            selected_model: None,
            selected_order: None,
            model_scores: Vec::new(),
        }
    }

    /// Get the selected order (p, d, q).
    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order
    }

    /// Get all model scores.
    pub fn model_scores(&self) -> &[((usize, usize, usize), f64)] {
        &self.model_scores
    }

    /// Generate candidate orders using stepwise search.
    fn stepwise_candidates(&self, d: usize) -> Vec<(usize, usize, usize)> {
        // Start with simple models and expand
        let mut candidates = vec![
            (0, d, 0),
            (1, d, 0),
            (0, d, 1),
            (1, d, 1),
            (2, d, 0),
            (0, d, 2),
            (2, d, 1),
            (1, d, 2),
            (2, d, 2),
        ];

        // Add higher orders if allowed
        if self.config.max_p >= 3 {
            candidates.push((3, d, 0));
            candidates.push((3, d, 1));
        }
        if self.config.max_q >= 3 {
            candidates.push((0, d, 3));
            candidates.push((1, d, 3));
        }

        // Filter by max orders
        candidates
            .into_iter()
            .filter(|&(p, _, q)| p <= self.config.max_p && q <= self.config.max_q)
            .collect()
    }

    /// Generate all candidate orders (exhaustive).
    fn exhaustive_candidates(&self, d: usize) -> Vec<(usize, usize, usize)> {
        let mut candidates = Vec::new();
        for p in 0..=self.config.max_p {
            for q in 0..=self.config.max_q {
                candidates.push((p, d, q));
            }
        }
        candidates
    }

    /// Get the information criterion from a model.
    fn get_criterion(&self, model: &ARIMA) -> Option<f64> {
        if self.config.use_aic {
            model.aic()
        } else {
            model.bic()
        }
    }
}

impl Default for AutoARIMA {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoARIMA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();

        if values.len() < 10 {
            return Err(ForecastError::InsufficientData {
                needed: 10,
                got: values.len(),
            });
        }

        // Determine differencing order
        let d = suggest_differencing(values).min(self.config.max_d);

        // Generate candidates
        let candidates = if self.config.stepwise {
            self.stepwise_candidates(d)
        } else {
            self.exhaustive_candidates(d)
        };

        self.model_scores.clear();
        let mut best_model: Option<ARIMA> = None;
        let mut best_order: Option<(usize, usize, usize)> = None;
        let mut best_score = f64::INFINITY;

        // Fit each candidate
        for (p, d, q) in candidates {
            // Check if we have enough data
            let min_len = d + p.max(q) + 5;
            if values.len() < min_len {
                continue;
            }

            let mut model = ARIMA::new(p, d, q);
            if model.fit(series).is_ok() {
                if let Some(score) = self.get_criterion(&model) {
                    if score.is_finite() {
                        self.model_scores.push(((p, d, q), score));

                        if score < best_score {
                            best_score = score;
                            best_model = Some(model);
                            best_order = Some((p, d, q));
                        }
                    }
                }
            }
        }

        // Sort model scores
        self.model_scores
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        self.selected_model = best_model;
        self.selected_order = best_order;

        if self.selected_model.is_none() {
            return Err(ForecastError::ComputationError(
                "No valid ARIMA model could be fitted".to_string(),
            ));
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        model.predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let model = self
            .selected_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        model.predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.selected_model.as_ref()?.fitted_values()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.selected_model.as_ref()?.residuals()
    }

    fn name(&self) -> &str {
        "AutoARIMA"
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
    fn auto_arima_selects_model() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
        assert!(!model.model_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_arima_with_trend() {
        let timestamps = make_timestamps(100);
        // Add some noise to make fitting easier
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + 1.5 * i as f64 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
    }

    #[test]
    fn auto_arima_ar_process() {
        let timestamps = make_timestamps(100);
        // AR(1) process
        let mut values = vec![10.0];
        for i in 1..100 {
            values.push(0.8 * values[i - 1] + (i as f64 * 0.05).sin());
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let (p, _, _) = model.selected_order().unwrap();
        // Should select AR component
        assert!(p >= 1);
    }

    #[test]
    fn auto_arima_exhaustive() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoARIMAConfig::default().exhaustive();
        let mut model = AutoARIMA::with_config(config);
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
        // Exhaustive should have more candidates
        assert!(model.model_scores().len() > 3);
    }

    #[test]
    fn auto_arima_model_scores_sorted() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let scores = model.model_scores();
        for i in 1..scores.len() {
            assert!(scores[i].1 >= scores[i - 1].1);
        }
    }

    #[test]
    fn auto_arima_confidence_intervals() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn auto_arima_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_arima_requires_fit() {
        let model = AutoARIMA::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn auto_arima_fitted_and_residuals() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_arima_name() {
        let model = AutoARIMA::new();
        assert_eq!(model.name(), "AutoARIMA");
    }

    #[test]
    fn auto_arima_config() {
        let config = AutoARIMAConfig::default()
            .with_max_orders(5, 2, 5)
            .exhaustive();

        assert_eq!(config.max_p, 5);
        assert_eq!(config.max_d, 2);
        assert_eq!(config.max_q, 5);
        assert!(!config.stepwise);
    }
}
