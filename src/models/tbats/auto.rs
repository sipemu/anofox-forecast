//! AutoTBATS - Automatic TBATS model selection.
//!
//! Automatically selects the best TBATS configuration by comparing:
//! - With/without Box-Cox transformation
//! - With/without trend
//! - With/without damped trend
//! - Different Fourier harmonic counts

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::tbats::TBATS;
use crate::models::Forecaster;

/// AutoTBATS - automatic TBATS model selection.
///
/// Evaluates multiple TBATS configurations and selects the best by AIC.
///
/// # Example
/// ```
/// use anofox_forecast::models::tbats::AutoTBATS;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..200).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::hours(i)).collect();
/// let values: Vec<f64> = (0..200).map(|i| {
///     50.0 + 0.2 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * (i % 24) as f64 / 24.0).sin()
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = AutoTBATS::new(vec![24]);
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(24).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AutoTBATS {
    /// Seasonal periods to consider.
    seasonal_periods: Vec<usize>,
    /// Whether to try Box-Cox transformation.
    try_box_cox: bool,
    /// Whether to try models without trend.
    try_no_trend: bool,
    /// Whether to try damped trend.
    try_damped: bool,
    /// Maximum Fourier K to try (relative to default).
    max_k_factor: f64,
    /// The selected best model.
    best_model: Option<TBATS>,
    /// AIC of the best model.
    best_aic: f64,
    /// Selected configuration description.
    selected_config: Option<String>,
}

impl AutoTBATS {
    /// Create a new AutoTBATS with given seasonal periods.
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self {
            seasonal_periods,
            try_box_cox: true,
            try_no_trend: true,
            try_damped: true,
            max_k_factor: 1.5,
            best_model: None,
            best_aic: f64::MAX,
            selected_config: None,
        }
    }

    /// Disable Box-Cox transformation search.
    pub fn without_box_cox_search(mut self) -> Self {
        self.try_box_cox = false;
        self
    }

    /// Disable no-trend model search.
    pub fn without_no_trend_search(mut self) -> Self {
        self.try_no_trend = false;
        self
    }

    /// Disable damped trend search.
    pub fn without_damped_search(mut self) -> Self {
        self.try_damped = false;
        self
    }

    /// Get the selected model configuration.
    pub fn selected_config(&self) -> Option<&str> {
        self.selected_config.as_deref()
    }

    /// Get the best AIC.
    pub fn best_aic(&self) -> f64 {
        self.best_aic
    }

    /// Get reference to the selected model.
    pub fn best_model(&self) -> Option<&TBATS> {
        self.best_model.as_ref()
    }

    /// Try a single configuration and update best if improved.
    fn try_config(&mut self, series: &TimeSeries, model: TBATS, config_name: &str) -> bool {
        let mut model = model;
        if model.fit(series).is_err() {
            return false;
        }

        if let Some(aic) = model.aic() {
            if aic < self.best_aic && aic.is_finite() {
                self.best_aic = aic;
                self.best_model = Some(model);
                self.selected_config = Some(config_name.to_string());
                return true;
            }
        }

        false
    }
}

impl Default for AutoTBATS {
    fn default() -> Self {
        Self::new(vec![12])
    }
}

impl Forecaster for AutoTBATS {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();

        let min_required = self
            .seasonal_periods
            .iter()
            .max()
            .copied()
            .unwrap_or(4)
            .max(10);

        if values.len() < min_required {
            return Err(ForecastError::InsufficientData {
                needed: min_required,
                got: values.len(),
            });
        }

        self.best_aic = f64::MAX;
        self.best_model = None;

        // Check if Box-Cox is possible
        let can_box_cox = values.iter().all(|&v| v > 0.0);

        // Base configurations to try
        let mut configs: Vec<(TBATS, String)> = Vec::new();

        // 1. Basic TBATS with trend
        configs.push((
            TBATS::new(self.seasonal_periods.clone()),
            "TBATS(trend)".to_string(),
        ));

        // 2. TBATS without trend
        if self.try_no_trend {
            configs.push((
                TBATS::new(self.seasonal_periods.clone()).without_trend(),
                "TBATS(no_trend)".to_string(),
            ));
        }

        // 3. TBATS with damped trend
        if self.try_damped {
            for phi in [0.9, 0.95, 0.98] {
                configs.push((
                    TBATS::new(self.seasonal_periods.clone()).with_damped_trend(phi),
                    format!("TBATS(damped_phi={:.2})", phi),
                ));
            }
        }

        // Try configurations without Box-Cox first
        for (model, name) in configs.iter() {
            self.try_config(series, model.clone(), name);
        }

        // Try with Box-Cox if possible
        if self.try_box_cox && can_box_cox {
            for lambda in [0.0, 0.25, 0.5, 0.75, 1.0] {
                // Basic with Box-Cox
                let model = TBATS::new(self.seasonal_periods.clone()).with_box_cox(lambda);
                self.try_config(series, model, &format!("TBATS(box_cox={:.2})", lambda));

                // Damped with Box-Cox
                if self.try_damped {
                    let model = TBATS::new(self.seasonal_periods.clone())
                        .with_box_cox(lambda)
                        .with_damped_trend(0.95);
                    self.try_config(
                        series,
                        model,
                        &format!("TBATS(box_cox={:.2},damped)", lambda),
                    );
                }
            }
        }

        // Try different Fourier K values
        let default_k: Vec<usize> = self
            .seasonal_periods
            .iter()
            .map(|&p| TBATS::default_k(p))
            .collect();

        // Try reduced K
        let reduced_k: Vec<usize> = default_k.iter().map(|&k| (k / 2).max(1)).collect();
        let model = TBATS::new(self.seasonal_periods.clone()).with_fourier_k(reduced_k);
        self.try_config(series, model, "TBATS(reduced_k)");

        // Try increased K
        let increased_k: Vec<usize> = self
            .seasonal_periods
            .iter()
            .zip(default_k.iter())
            .map(|(&p, &k)| ((k as f64 * self.max_k_factor) as usize).min(p / 2))
            .collect();
        let model = TBATS::new(self.seasonal_periods.clone()).with_fourier_k(increased_k);
        self.try_config(series, model, "TBATS(increased_k)");

        if self.best_model.is_none() {
            return Err(ForecastError::ComputationError(
                "No valid TBATS configuration found".to_string(),
            ));
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        self.best_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?
            .predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        self.best_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?
            .predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.best_model.as_ref()?.fitted_values()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        self.best_model
            .as_ref()?
            .fitted_values_with_intervals(level)
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.best_model.as_ref()?.residuals()
    }

    fn name(&self) -> &str {
        "AutoTBATS"
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

    fn make_complex_seasonal_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 50.0 + 0.1 * i as f64;
                let daily = 10.0 * (2.0 * std::f64::consts::PI * (i % 24) as f64 / 24.0).sin();
                let noise = ((i * 17) % 7) as f64 * 0.3 - 1.0;
                (trend + daily + noise).max(1.0)
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn auto_tbats_basic() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        assert!(model.selected_config().is_some());
        assert!(model.best_aic() < f64::MAX);

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn auto_tbats_selects_config() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let config = model.selected_config().unwrap();
        assert!(!config.is_empty());
    }

    #[test]
    fn auto_tbats_without_searches() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24])
            .without_box_cox_search()
            .without_damped_search();
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn auto_tbats_confidence_intervals() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(24, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn auto_tbats_fitted_and_residuals() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_tbats_requires_fit() {
        let model = AutoTBATS::new(vec![24]);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn auto_tbats_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTBATS::new(vec![24]);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_tbats_name() {
        let model = AutoTBATS::new(vec![24]);
        assert_eq!(model.name(), "AutoTBATS");
    }

    #[test]
    fn auto_tbats_default() {
        let model = AutoTBATS::default();
        assert_eq!(model.seasonal_periods, vec![12]);
    }

    #[test]
    fn auto_tbats_best_model_reference() {
        let ts = make_complex_seasonal_series(200);
        let mut model = AutoTBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        assert!(model.best_model().is_some());
    }
}
