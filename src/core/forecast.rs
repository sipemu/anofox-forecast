//! Forecast result structure for holding predictions.

use crate::error::{ForecastError, Result};

/// A forecast result containing point predictions and optional intervals.
#[derive(Debug, Clone, Default)]
pub struct Forecast {
    /// Point predictions: point[dimension][step]
    point: Vec<Vec<f64>>,
    /// Lower prediction interval bounds (optional)
    lower: Option<Vec<Vec<f64>>>,
    /// Upper prediction interval bounds (optional)
    upper: Option<Vec<Vec<f64>>>,
}

impl Forecast {
    /// Create an empty forecast.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a forecast with specified dimensions.
    pub fn with_dimensions(dims: usize) -> Self {
        Self {
            point: vec![Vec::new(); dims],
            lower: None,
            upper: None,
        }
    }

    /// Create a univariate forecast from point predictions.
    pub fn from_values(values: Vec<f64>) -> Self {
        Self {
            point: vec![values],
            lower: None,
            upper: None,
        }
    }

    /// Create a univariate forecast with prediction intervals.
    pub fn from_values_with_intervals(
        values: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Self {
        Self {
            point: vec![values],
            lower: Some(vec![lower]),
            upper: Some(vec![upper]),
        }
    }

    /// Ensure the forecast has at least the specified number of dimensions.
    pub fn ensure_dimensions(&mut self, dims: usize) {
        while self.point.len() < dims {
            self.point.push(Vec::new());
        }
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.point.len()
    }

    /// Get the forecast horizon (number of steps).
    pub fn horizon(&self) -> usize {
        self.point.first().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if forecast is empty.
    pub fn is_empty(&self) -> bool {
        self.point.is_empty() || self.point.iter().all(|s| s.is_empty())
    }

    /// Check if forecast is multivariate.
    pub fn is_multivariate(&self) -> bool {
        self.point.len() > 1
    }

    /// Get mutable reference to a series, creating if needed.
    pub fn series_mut(&mut self, dimension: usize) -> &mut Vec<f64> {
        self.ensure_dimensions(dimension + 1);
        &mut self.point[dimension]
    }

    /// Get reference to a series.
    pub fn series(&self, dimension: usize) -> Result<&[f64]> {
        self.point
            .get(dimension)
            .map(|v| v.as_slice())
            .ok_or(ForecastError::IndexOutOfBounds {
                index: dimension,
                size: self.point.len(),
            })
    }

    /// Get mutable reference to the primary (first) series.
    pub fn primary_mut(&mut self) -> &mut Vec<f64> {
        self.series_mut(0)
    }

    /// Get reference to the primary (first) series.
    pub fn primary(&self) -> &[f64] {
        self.point.first().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all point predictions.
    pub fn point(&self) -> &[Vec<f64>] {
        &self.point
    }

    /// Check if lower interval is available.
    pub fn has_lower(&self) -> bool {
        self.lower.is_some()
    }

    /// Check if upper interval is available.
    pub fn has_upper(&self) -> bool {
        self.upper.is_some()
    }

    /// Ensure lower interval matrix exists with specified dimensions.
    pub fn ensure_lower(&mut self, dims: usize) -> &mut Vec<Vec<f64>> {
        if self.lower.is_none() {
            self.lower = Some(vec![Vec::new(); dims]);
        }
        let lower = self.lower.as_mut().unwrap();
        while lower.len() < dims {
            lower.push(Vec::new());
        }
        lower
    }

    /// Ensure upper interval matrix exists with specified dimensions.
    pub fn ensure_upper(&mut self, dims: usize) -> &mut Vec<Vec<f64>> {
        if self.upper.is_none() {
            self.upper = Some(vec![Vec::new(); dims]);
        }
        let upper = self.upper.as_mut().unwrap();
        while upper.len() < dims {
            upper.push(Vec::new());
        }
        upper
    }

    /// Get mutable reference to lower series for a dimension.
    pub fn lower_series_mut(&mut self, dimension: usize) -> &mut Vec<f64> {
        let lower = self.ensure_lower(dimension + 1);
        &mut lower[dimension]
    }

    /// Get mutable reference to upper series for a dimension.
    pub fn upper_series_mut(&mut self, dimension: usize) -> &mut Vec<f64> {
        let upper = self.ensure_upper(dimension + 1);
        &mut upper[dimension]
    }

    /// Get reference to lower series for a dimension.
    pub fn lower_series(&self, dimension: usize) -> Result<&[f64]> {
        self.lower
            .as_ref()
            .and_then(|l| l.get(dimension))
            .map(|v| v.as_slice())
            .ok_or(ForecastError::IndexOutOfBounds {
                index: dimension,
                size: self.lower.as_ref().map(|l| l.len()).unwrap_or(0),
            })
    }

    /// Get reference to upper series for a dimension.
    pub fn upper_series(&self, dimension: usize) -> Result<&[f64]> {
        self.upper
            .as_ref()
            .and_then(|u| u.get(dimension))
            .map(|v| v.as_slice())
            .ok_or(ForecastError::IndexOutOfBounds {
                index: dimension,
                size: self.upper.as_ref().map(|u| u.len()).unwrap_or(0),
            })
    }

    /// Get all lower interval bounds.
    pub fn lower(&self) -> Option<&[Vec<f64>]> {
        self.lower.as_ref().map(|l| l.as_slice())
    }

    /// Get all upper interval bounds.
    pub fn upper(&self) -> Option<&[Vec<f64>]> {
        self.upper.as_ref().map(|u| u.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_lazily_expands_dimensions_and_series() {
        let mut forecast = Forecast::new();

        // Initially empty
        assert!(forecast.is_empty());
        assert_eq!(forecast.dimensions(), 0);
        assert_eq!(forecast.horizon(), 0);
        assert!(!forecast.is_multivariate());

        // Access dimension 0 - should create it
        forecast.series_mut(0).push(1.0);
        assert_eq!(forecast.dimensions(), 1);
        assert!(!forecast.is_multivariate());
        assert_eq!(forecast.horizon(), 1);

        // Access dimension 2 - should expand to 3 dimensions
        forecast.series_mut(2).push(3.0);
        assert_eq!(forecast.dimensions(), 3);
        assert!(forecast.is_multivariate());

        // Primary access works
        assert_eq!(forecast.primary(), &[1.0]);

        // Dimension 1 exists but is empty
        assert_eq!(forecast.series(1).unwrap(), &[]);
    }

    #[test]
    fn forecast_manages_prediction_intervals() {
        let mut forecast = Forecast::new();

        // Add point predictions
        forecast.primary_mut().extend([1.0, 2.0, 3.0]);

        // No intervals initially
        assert!(!forecast.has_lower());
        assert!(!forecast.has_upper());
        assert!(forecast.lower_series(0).is_err());

        // Add lower bounds
        forecast.lower_series_mut(0).extend([0.5, 1.5, 2.5]);
        assert!(forecast.has_lower());
        assert_eq!(forecast.lower_series(0).unwrap(), &[0.5, 1.5, 2.5]);

        // Add upper bounds
        forecast.upper_series_mut(0).extend([1.5, 2.5, 3.5]);
        assert!(forecast.has_upper());
        assert_eq!(forecast.upper_series(0).unwrap(), &[1.5, 2.5, 3.5]);

        // Access non-existent dimension throws error
        assert!(forecast.lower_series(1).is_err());
        assert!(forecast.upper_series(1).is_err());
    }

    #[test]
    fn forecast_empty_state_reflects_missing_values() {
        let forecast = Forecast::new();
        assert!(forecast.is_empty());
        assert_eq!(forecast.horizon(), 0);

        // With dimensions but no values
        let forecast = Forecast::with_dimensions(2);
        assert!(forecast.is_empty()); // Empty vectors are still "empty"
        assert_eq!(forecast.dimensions(), 2);
        assert_eq!(forecast.horizon(), 0);

        // With actual values
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0]);
        assert!(!forecast.is_empty());
        assert_eq!(forecast.horizon(), 3);
    }

    #[test]
    fn forecast_from_values_creates_univariate() {
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0, 4.0]);

        assert!(!forecast.is_empty());
        assert_eq!(forecast.dimensions(), 1);
        assert!(!forecast.is_multivariate());
        assert_eq!(forecast.horizon(), 4);
        assert_eq!(forecast.primary(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn forecast_from_values_with_intervals() {
        let forecast = Forecast::from_values_with_intervals(
            vec![2.0, 3.0],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        );

        assert_eq!(forecast.primary(), &[2.0, 3.0]);
        assert_eq!(forecast.lower_series(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(forecast.upper_series(0).unwrap(), &[3.0, 4.0]);
    }
}
