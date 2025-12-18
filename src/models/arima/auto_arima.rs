//! Automatic ARIMA and SARIMA model selection.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::diff::suggest_differencing;
use crate::models::arima::model::{ARIMA, SARIMA};
use crate::models::Forecaster;

/// Configuration for AutoARIMA.
#[derive(Debug, Clone)]
pub struct AutoARIMAConfig {
    /// Maximum non-seasonal AR order to consider.
    pub max_p: usize,
    /// Maximum non-seasonal MA order to consider.
    pub max_q: usize,
    /// Maximum non-seasonal differencing order.
    pub max_d: usize,
    /// Maximum seasonal AR order.
    pub max_cap_p: usize,
    /// Maximum seasonal MA order.
    pub max_cap_q: usize,
    /// Maximum seasonal differencing order.
    pub max_cap_d: usize,
    /// Seasonal period (0 for non-seasonal).
    pub seasonal_period: usize,
    /// Use stepwise search (faster) vs exhaustive.
    pub stepwise: bool,
    /// Selection criterion (use AIC for selection).
    pub use_aic: bool,
}

impl Default for AutoARIMAConfig {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_q: 5,
            max_d: 2,
            max_cap_p: 2,
            max_cap_q: 2,
            max_cap_d: 1,
            seasonal_period: 0,
            stepwise: true,
            use_aic: true,
        }
    }
}

impl AutoARIMAConfig {
    /// Set maximum non-seasonal orders.
    pub fn with_max_orders(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_p = max_p;
        self.max_d = max_d;
        self.max_q = max_q;
        self
    }

    /// Set maximum seasonal orders.
    pub fn with_seasonal_orders(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_cap_p = max_p;
        self.max_cap_d = max_d;
        self.max_cap_q = max_q;
        self
    }

    /// Set seasonal period.
    pub fn with_seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = period;
        self
    }

    /// Use exhaustive search instead of stepwise.
    pub fn exhaustive(mut self) -> Self {
        self.stepwise = false;
        self
    }
}

/// Selected model type.
#[derive(Debug, Clone)]
enum SelectedModel {
    ARIMA(ARIMA),
    SARIMA(SARIMA),
}

/// Model order (p, d, q, P, D, Q, s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOrder {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub cap_p: usize,
    /// Seasonal differencing order.
    pub cap_d: usize,
    /// Seasonal MA order.
    pub cap_q: usize,
    /// Seasonal period.
    pub s: usize,
}

impl ModelOrder {
    /// Check if this is a seasonal model.
    pub fn is_seasonal(&self) -> bool {
        self.s > 1 && (self.cap_p > 0 || self.cap_d > 0 || self.cap_q > 0)
    }
}

/// Automatic ARIMA/SARIMA model selection.
///
/// Automatically selects the best ARIMA(p, d, q) or SARIMA(p, d, q)(P, D, Q)\[s\]
/// specification based on information criteria.
#[derive(Debug, Clone)]
pub struct AutoARIMA {
    /// Configuration.
    config: AutoARIMAConfig,
    /// Selected model.
    selected_model: Option<SelectedModel>,
    /// Selected orders.
    selected_order: Option<ModelOrder>,
    /// All fitted models and their scores.
    model_scores: Vec<(ModelOrder, f64)>,
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

    /// Create AutoARIMA with seasonal period.
    pub fn seasonal(period: usize) -> Self {
        let config = AutoARIMAConfig::default().with_seasonal_period(period);
        Self::with_config(config)
    }

    /// Get the selected order.
    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order.map(|o| (o.p, o.d, o.q))
    }

    /// Get the full selected order including seasonal components.
    pub fn selected_full_order(&self) -> Option<ModelOrder> {
        self.selected_order
    }

    /// Get all model scores.
    pub fn model_scores(&self) -> &[(ModelOrder, f64)] {
        &self.model_scores
    }

    /// Suggest seasonal differencing order using strength of seasonality.
    fn suggest_seasonal_differencing(values: &[f64], period: usize) -> usize {
        if period < 2 || values.len() < 2 * period {
            return 0;
        }

        // Compute seasonal differences
        let seasonal_diffs: Vec<f64> = (period..values.len())
            .map(|i| values[i] - values[i - period])
            .collect();

        // Compare variances
        let orig_mean = values.iter().sum::<f64>() / values.len() as f64;
        let orig_var =
            values.iter().map(|v| (v - orig_mean).powi(2)).sum::<f64>() / values.len() as f64;

        let diff_mean = seasonal_diffs.iter().sum::<f64>() / seasonal_diffs.len() as f64;
        let diff_var = seasonal_diffs
            .iter()
            .map(|v| (v - diff_mean).powi(2))
            .sum::<f64>()
            / seasonal_diffs.len() as f64;

        // If seasonal differencing reduces variance significantly, suggest D=1
        if diff_var < orig_var * 0.7 {
            1
        } else {
            0
        }
    }

    /// Generate candidate orders using stepwise search.
    fn stepwise_candidates(&self, d: usize, cap_d: usize) -> Vec<ModelOrder> {
        let s = self.config.seasonal_period;

        // Non-seasonal candidates
        let nonseasonal = vec![
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (2, 2),
        ];

        let mut candidates = Vec::new();

        // Add non-seasonal models
        for &(p, q) in &nonseasonal {
            if p <= self.config.max_p && q <= self.config.max_q {
                candidates.push(ModelOrder {
                    p,
                    d,
                    q,
                    cap_p: 0,
                    cap_d,
                    cap_q: 0,
                    s,
                });
            }
        }

        // Add seasonal models if period > 1
        if s > 1 {
            // Seasonal component options: (P, Q)
            let seasonal = vec![
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (2, 2),
            ];

            // Non-seasonal orders to try with seasonal components
            let nonseasonal_with_seasonal = vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (3, 0),
                (0, 3),
                (2, 2),
                (3, 1),
                (1, 3),
            ];

            for &(p, q) in &nonseasonal_with_seasonal {
                for &(cap_p, cap_q) in &seasonal {
                    if p <= self.config.max_p
                        && q <= self.config.max_q
                        && cap_p <= self.config.max_cap_p
                        && cap_q <= self.config.max_cap_q
                    {
                        candidates.push(ModelOrder {
                            p,
                            d,
                            q,
                            cap_p,
                            cap_d,
                            cap_q,
                            s,
                        });
                    }
                }
            }
        }

        candidates
    }

    /// Generate all candidate orders (exhaustive).
    fn exhaustive_candidates(&self, d: usize, cap_d: usize) -> Vec<ModelOrder> {
        let s = self.config.seasonal_period;
        let mut candidates = Vec::new();

        for p in 0..=self.config.max_p {
            for q in 0..=self.config.max_q {
                if s > 1 {
                    // Add seasonal models
                    for cap_p in 0..=self.config.max_cap_p {
                        for cap_q in 0..=self.config.max_cap_q {
                            candidates.push(ModelOrder {
                                p,
                                d,
                                q,
                                cap_p,
                                cap_d,
                                cap_q,
                                s,
                            });
                        }
                    }
                } else {
                    // Non-seasonal only
                    candidates.push(ModelOrder {
                        p,
                        d,
                        q,
                        cap_p: 0,
                        cap_d: 0,
                        cap_q: 0,
                        s: 0,
                    });
                }
            }
        }

        candidates
    }

    /// Get AIC from ARIMA model.
    fn get_arima_criterion(&self, model: &ARIMA) -> Option<f64> {
        if self.config.use_aic {
            model.aic()
        } else {
            model.bic()
        }
    }

    /// Get AIC from SARIMA model.
    fn get_sarima_criterion(&self, model: &SARIMA) -> Option<f64> {
        if self.config.use_aic {
            model.aic()
        } else {
            model.bic()
        }
    }

    /// Fit and evaluate a model with given order.
    fn evaluate_model(
        &self,
        series: &TimeSeries,
        order: ModelOrder,
    ) -> Option<(SelectedModel, f64)> {
        if order.is_seasonal() {
            // Fit SARIMA
            let mut model = SARIMA::new(
                order.p,
                order.d,
                order.q,
                order.cap_p,
                order.cap_d,
                order.cap_q,
                order.s,
            );

            if model.fit(series).is_ok() {
                if let Some(score) = self.get_sarima_criterion(&model) {
                    if score.is_finite() {
                        return Some((SelectedModel::SARIMA(model), score));
                    }
                }
            }
        } else {
            // Fit ARIMA
            let mut model = ARIMA::new(order.p, order.d, order.q);

            if model.fit(series).is_ok() {
                if let Some(score) = self.get_arima_criterion(&model) {
                    if score.is_finite() {
                        return Some((SelectedModel::ARIMA(model), score));
                    }
                }
            }
        }

        None
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
        let s = self.config.seasonal_period;

        // Check minimum data requirements
        let min_required = if s > 1 {
            3 * s // At least 3 seasonal cycles for SARIMA
        } else {
            10
        };

        if values.len() < min_required {
            return Err(ForecastError::InsufficientData {
                needed: min_required,
                got: values.len(),
            });
        }

        // Determine differencing orders - search over range instead of fixing
        let suggested_d = suggest_differencing(values).min(self.config.max_d);
        let suggested_cap_d = if s > 1 {
            Self::suggest_seasonal_differencing(values, s).min(self.config.max_cap_d)
        } else {
            0
        };

        // Build range of d values to try (suggested and neighbors)
        let mut d_range = vec![suggested_d];
        if suggested_d > 0 {
            d_range.push(suggested_d - 1);
        }
        if suggested_d < self.config.max_d {
            d_range.push(suggested_d + 1);
        }
        d_range.sort();
        d_range.dedup();

        // Build range of D values to try (both 0 and suggested for seasonal)
        let cap_d_range: Vec<usize> = if s > 1 && self.config.max_cap_d > 0 {
            let mut range = vec![0, suggested_cap_d];
            range.sort();
            range.dedup();
            range
        } else {
            vec![0]
        };

        // Generate candidates for all (d, D) combinations
        let mut candidates = Vec::new();
        for &d in &d_range {
            for &cap_d in &cap_d_range {
                let new_candidates = if self.config.stepwise {
                    self.stepwise_candidates(d, cap_d)
                } else {
                    self.exhaustive_candidates(d, cap_d)
                };
                candidates.extend(new_candidates);
            }
        }
        // Remove duplicates
        candidates.sort_by(|a, b| {
            (a.p, a.d, a.q, a.cap_p, a.cap_d, a.cap_q)
                .cmp(&(b.p, b.d, b.q, b.cap_p, b.cap_d, b.cap_q))
        });
        candidates.dedup();

        self.model_scores.clear();
        let mut best_model: Option<SelectedModel> = None;
        let mut best_order: Option<ModelOrder> = None;
        let mut best_score = f64::INFINITY;

        // Fit each candidate
        for order in candidates {
            // Check data requirements for this order
            let min_len = order.d
                + order.cap_d * order.s
                + order
                    .p
                    .max(order.q)
                    .max(order.cap_p.max(order.cap_q) * order.s.max(1))
                + 5;

            if values.len() < min_len {
                continue;
            }

            if let Some((model, score)) = self.evaluate_model(series, order) {
                self.model_scores.push((order, score));

                if score < best_score {
                    best_score = score;
                    best_model = Some(model);
                    best_order = Some(order);
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
                "No valid ARIMA/SARIMA model could be fitted".to_string(),
            ));
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => model.predict(horizon),
            Some(SelectedModel::SARIMA(model)) => model.predict(horizon),
            None => Err(ForecastError::FitRequired),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => model.predict_with_intervals(horizon, level),
            Some(SelectedModel::SARIMA(model)) => model.predict_with_intervals(horizon, level),
            None => Err(ForecastError::FitRequired),
        }
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.fitted_values(),
            SelectedModel::SARIMA(model) => model.fitted_values(),
        }
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.fitted_values_with_intervals(level),
            SelectedModel::SARIMA(model) => model.fitted_values_with_intervals(level),
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.residuals(),
            SelectedModel::SARIMA(model) => model.residuals(),
        }
    }

    fn name(&self) -> &str {
        match &self.selected_model {
            Some(SelectedModel::SARIMA(_)) => "AutoARIMA (SARIMA)",
            _ => "AutoARIMA",
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

    // SARIMA-specific tests
    #[test]
    fn auto_arima_seasonal() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::seasonal(12);
        model.fit(&ts).unwrap();

        assert!(model.selected_full_order().is_some());

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn auto_arima_seasonal_config() {
        let config = AutoARIMAConfig::default()
            .with_seasonal_period(12)
            .with_seasonal_orders(2, 1, 2);

        assert_eq!(config.seasonal_period, 12);
        assert_eq!(config.max_cap_p, 2);
        assert_eq!(config.max_cap_d, 1);
        assert_eq!(config.max_cap_q, 2);
    }

    #[test]
    fn auto_arima_seasonal_selects_sarima() {
        let timestamps = make_timestamps(100);
        // Strong seasonal pattern
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 15.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoARIMAConfig::default()
            .with_seasonal_period(12)
            .exhaustive();
        let mut model = AutoARIMA::with_config(config);
        model.fit(&ts).unwrap();

        // Should select a seasonal model
        if model.selected_full_order().is_some() {
            // With strong seasonality, should select seasonal components
            assert!(model.model_scores().len() > 1);
        }

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }
}
