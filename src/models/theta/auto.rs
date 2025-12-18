//! AutoTheta - Automatic Theta model selection.
//!
//! AutoTheta automatically selects the best Theta variant (STM, OTM, DSTM, DOTM)
//! based on the data characteristics and compares their in-sample performance.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::theta::{DecompositionType, DynamicTheta, OptimizedTheta, Theta};
use crate::models::Forecaster;

/// Type of Theta model selected by AutoTheta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaModelType {
    /// Standard Theta Model (STM) - fixed alpha=0.1, theta=2.0
    STM,
    /// Optimized Theta Model (OTM) - optimized alpha and theta
    OTM,
    /// Dynamic Standard Theta Model (DSTM) - fixed parameters, dynamic coefficients
    DSTM,
    /// Dynamic Optimized Theta Model (DOTM) - optimized parameters, dynamic coefficients
    DOTM,
}

impl std::fmt::Display for ThetaModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThetaModelType::STM => write!(f, "STM"),
            ThetaModelType::OTM => write!(f, "OTM"),
            ThetaModelType::DSTM => write!(f, "DSTM"),
            ThetaModelType::DOTM => write!(f, "DOTM"),
        }
    }
}

/// Internal enum to hold fitted models.
#[derive(Debug, Clone)]
enum FittedModel {
    STM(Theta),
    OTM(OptimizedTheta),
    DSTM(DynamicTheta),
    DOTM(DynamicTheta),
}

/// AutoTheta - Automatic selection of the best Theta variant.
///
/// AutoTheta fits multiple Theta model variants and selects the one
/// with the best in-sample performance (lowest MSE).
///
/// # Model Variants Considered
/// - **STM**: Standard Theta Model with fixed alpha=0.1, theta=2.0
/// - **OTM**: Optimized Theta Model with optimized alpha and theta
/// - **DSTM**: Dynamic Standard Theta Model with dynamic coefficients
/// - **DOTM**: Dynamic Optimized Theta Model (M4 competition winner component)
///
/// # Example
/// ```
/// use anofox_forecast::models::theta::AutoTheta;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..50).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = AutoTheta::new();
/// model.fit(&ts).unwrap();
/// println!("Selected model: {}", model.selected_model().unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct AutoTheta {
    /// Whether to include seasonal decomposition.
    seasonal_period: usize,
    /// Decomposition type for seasonal models.
    decomposition_type: DecompositionType,
    /// The selected model type.
    selected_type: Option<ThetaModelType>,
    /// The fitted model.
    fitted_model: Option<FittedModel>,
    /// MSE scores for each model type.
    model_scores: Option<Vec<(ThetaModelType, f64)>>,
    /// Number of observations.
    n: usize,
}

impl AutoTheta {
    /// Create a new AutoTheta model.
    pub fn new() -> Self {
        Self {
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            selected_type: None,
            fitted_model: None,
            model_scores: None,
            n: 0,
        }
    }

    /// Create a seasonal AutoTheta model.
    pub fn seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            selected_type: None,
            fitted_model: None,
            model_scores: None,
            n: 0,
        }
    }

    /// Create a seasonal AutoTheta with specified decomposition type.
    pub fn seasonal_with_decomposition(period: usize, decomposition: DecompositionType) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: decomposition,
            selected_type: None,
            fitted_model: None,
            model_scores: None,
            n: 0,
        }
    }

    /// Get the selected model type.
    pub fn selected_model(&self) -> Option<ThetaModelType> {
        self.selected_type
    }

    /// Get the MSE scores for all evaluated models.
    pub fn model_scores(&self) -> Option<&[(ThetaModelType, f64)]> {
        self.model_scores.as_deref()
    }

    /// Calculate MSE from residuals.
    fn calculate_mse(residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return f64::MAX;
        }
        // Skip first residual which is often 0
        let valid: Vec<f64> = residuals.iter().skip(1).copied().collect();
        if valid.is_empty() {
            return f64::MAX;
        }
        crate::simd::sum_of_squares(&valid) / valid.len() as f64
    }
}

impl Default for AutoTheta {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoTheta {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < 6 {
            return Err(ForecastError::InsufficientData {
                needed: 6,
                got: values.len(),
            });
        }

        self.n = values.len();

        let mut scores: Vec<(ThetaModelType, f64, FittedModel)> = Vec::new();

        // Try STM (Standard Theta Model)
        {
            let mut model = if self.seasonal_period > 0 {
                Theta::seasonal_with_decomposition(self.seasonal_period, self.decomposition_type)
            } else {
                Theta::new()
            };
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    scores.push((ThetaModelType::STM, mse, FittedModel::STM(model)));
                }
            }
        }

        // Try OTM (Optimized Theta Model)
        {
            let mut model = if self.seasonal_period > 0 {
                OptimizedTheta::seasonal_with_decomposition(
                    self.seasonal_period,
                    self.decomposition_type,
                )
            } else {
                OptimizedTheta::new()
            };
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    scores.push((ThetaModelType::OTM, mse, FittedModel::OTM(model)));
                }
            }
        }

        // Try DSTM (Dynamic Standard Theta Model)
        {
            let mut model = DynamicTheta::new(0.1);
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    scores.push((ThetaModelType::DSTM, mse, FittedModel::DSTM(model)));
                }
            }
        }

        // Try DOTM (Dynamic Optimized Theta Model)
        {
            let mut model = DynamicTheta::optimized();
            if model.fit(series).is_ok() {
                if let Some(residuals) = model.residuals() {
                    let mse = Self::calculate_mse(residuals);
                    scores.push((ThetaModelType::DOTM, mse, FittedModel::DOTM(model)));
                }
            }
        }

        if scores.is_empty() {
            return Err(ForecastError::ComputationError(
                "All Theta model variants failed to fit".to_string(),
            ));
        }

        // Sort by MSE and select best
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_type, _, best_model) = scores.remove(0);

        // Store scores for diagnostics
        let score_summary: Vec<(ThetaModelType, f64)> =
            std::iter::once((best_type, scores.first().map(|s| s.1).unwrap_or(0.0)))
                .chain(scores.iter().map(|(t, s, _)| (*t, *s)))
                .collect();

        self.selected_type = Some(best_type);
        self.fitted_model = Some(best_model);
        self.model_scores = Some(score_summary);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let model = self
            .fitted_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        match model {
            FittedModel::STM(m) => m.predict(horizon),
            FittedModel::OTM(m) => m.predict(horizon),
            FittedModel::DSTM(m) => m.predict(horizon),
            FittedModel::DOTM(m) => m.predict(horizon),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let model = self
            .fitted_model
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        match model {
            FittedModel::STM(m) => m.predict_with_intervals(horizon, confidence),
            FittedModel::OTM(m) => m.predict_with_intervals(horizon, confidence),
            FittedModel::DSTM(m) => m.predict_with_intervals(horizon, confidence),
            FittedModel::DOTM(m) => m.predict_with_intervals(horizon, confidence),
        }
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        match self.fitted_model.as_ref()? {
            FittedModel::STM(m) => m.fitted_values(),
            FittedModel::OTM(m) => m.fitted_values(),
            FittedModel::DSTM(m) => m.fitted_values(),
            FittedModel::DOTM(m) => m.fitted_values(),
        }
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        match self.fitted_model.as_ref()? {
            FittedModel::STM(m) => m.fitted_values_with_intervals(level),
            FittedModel::OTM(m) => m.fitted_values_with_intervals(level),
            FittedModel::DSTM(m) => m.fitted_values_with_intervals(level),
            FittedModel::DOTM(m) => m.fitted_values_with_intervals(level),
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        match self.fitted_model.as_ref()? {
            FittedModel::STM(m) => m.residuals(),
            FittedModel::OTM(m) => m.residuals(),
            FittedModel::DSTM(m) => m.residuals(),
            FittedModel::DOTM(m) => m.residuals(),
        }
    }

    fn name(&self) -> &str {
        "AutoTheta"
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
    fn auto_theta_basic() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_model().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_theta_trending_data() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();

        let selected = model.selected_model().unwrap();
        println!("Selected model for trending data: {}", selected);

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Should continue trend
        assert!(preds[0] > values.last().unwrap() - 10.0);
    }

    #[test]
    fn auto_theta_model_scores() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.model_scores().is_some());
        let scores = model.model_scores().unwrap();
        assert!(!scores.is_empty());

        // All scores should be finite
        for (_, score) in scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn auto_theta_seasonal() {
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn auto_theta_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..5 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }
    }

    #[test]
    fn auto_theta_fitted_and_residuals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_theta_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoTheta::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_theta_requires_fit() {
        let model = AutoTheta::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn auto_theta_name() {
        let model = AutoTheta::new();
        assert_eq!(model.name(), "AutoTheta");
    }

    #[test]
    fn auto_theta_default() {
        let model = AutoTheta::default();
        assert!(model.selected_model().is_none());
    }

    #[test]
    fn theta_model_type_display() {
        assert_eq!(format!("{}", ThetaModelType::STM), "STM");
        assert_eq!(format!("{}", ThetaModelType::OTM), "OTM");
        assert_eq!(format!("{}", ThetaModelType::DSTM), "DSTM");
        assert_eq!(format!("{}", ThetaModelType::DOTM), "DOTM");
    }
}
