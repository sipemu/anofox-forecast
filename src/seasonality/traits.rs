//! Traits for composable seasonality and trend components.
//!
//! Components implementing these traits can be used standalone (fit, predict,
//! get fitted values) or as feature extractors via the `features()` method.

use crate::error::Result;

/// A composable seasonal component.
///
/// Implementations model periodic patterns at a given period. Each component
/// can be fitted to data, used to predict future seasonal values, and queried
/// for named features.
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::seasonality::{DummySeasonality, SeasonalComponent};
///
/// let mut comp = DummySeasonality::new();
/// comp.fit_seasonal(&values, 12)?;
/// let fitted = comp.fitted_seasonal();
/// let forecast = comp.predict_seasonal(24);
/// let features = comp.seasonal_features();
/// ```
pub trait SeasonalComponent {
    /// Fit the seasonal component to `values` with the given `period`.
    fn fit_seasonal(&mut self, values: &[f64], period: usize) -> Result<()>;

    /// Return the fitted seasonal values (in-sample, same length as training data).
    fn fitted_seasonal(&self) -> &[f64];

    /// Predict the seasonal component for the next `n_ahead` steps.
    fn predict_seasonal(&self, n_ahead: usize) -> Vec<f64>;

    /// Extract named features from the fitted component.
    ///
    /// Returns `(name, value)` pairs suitable for ML pipelines and the
    /// feature-extraction module. Feature names are prefixed with the
    /// component name (e.g. `"dummy_seasonal_strength"`).
    fn seasonal_features(&self) -> Vec<(&str, f64)>;

    /// Human-readable name of this component.
    fn seasonal_name(&self) -> &str;
}

/// A composable trend component.
///
/// Implementations extract and model the trend in a time series.
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::seasonality::{PiecewiseLinearTrend, TrendComponent};
///
/// let mut comp = PiecewiseLinearTrend::new();
/// comp.fit_trend(&values)?;
/// let fitted = comp.fitted_trend();
/// let forecast = comp.predict_trend(12);
/// let features = comp.trend_features();
/// ```
pub trait TrendComponent {
    /// Fit the trend component to `values`.
    fn fit_trend(&mut self, values: &[f64]) -> Result<()>;

    /// Return the fitted trend values (in-sample, same length as training data).
    fn fitted_trend(&self) -> &[f64];

    /// Predict the trend for the next `n_ahead` steps.
    fn predict_trend(&self, n_ahead: usize) -> Vec<f64>;

    /// Extract named features from the fitted component.
    ///
    /// Returns `(name, value)` pairs suitable for ML pipelines.
    fn trend_features(&self) -> Vec<(&str, f64)>;

    /// Human-readable name of this component.
    fn trend_name(&self) -> &str;
}
