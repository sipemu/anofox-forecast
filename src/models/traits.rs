//! Forecaster trait defining the common interface for all models.

use crate::core::{Forecast, TimeSeries};
use crate::error::Result;

/// Common interface for all forecasting models.
pub trait Forecaster {
    /// Fit the model to the time series data.
    fn fit(&mut self, series: &TimeSeries) -> Result<()>;

    /// Generate predictions for the specified horizon.
    fn predict(&self, horizon: usize) -> Result<Forecast>;

    /// Generate predictions with confidence intervals.
    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        // Default implementation just returns point predictions
        let _ = level;
        self.predict(horizon)
    }

    /// Get the fitted values (in-sample predictions).
    fn fitted_values(&self) -> Option<&[f64]>;

    /// Get the residuals (actual - fitted).
    fn residuals(&self) -> Option<&[f64]>;

    /// Get the model name.
    fn name(&self) -> &str;

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted_values().is_some()
    }
}
