//! Forecast result structure for holding predictions.

use crate::error::{ForecastError, Result};
use std::fmt;

/// A forecast result containing point predictions and optional intervals.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub fn from_values_with_intervals(values: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>) -> Self {
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
        self.lower.as_deref()
    }

    /// Get all upper interval bounds.
    pub fn upper(&self) -> Option<&[Vec<f64>]> {
        self.upper.as_deref()
    }
}

#[cfg(feature = "serde")]
impl Forecast {
    /// Serialize this forecast to a JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ForecastError::SerializationError(format!("serialization failed: {}", e)))
    }

    /// Deserialize a forecast from a JSON string.
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            ForecastError::SerializationError(format!("deserialization failed: {}", e))
        })
    }
}

/// Epsilon-based equality for floating-point forecast data.
impl PartialEq for Forecast {
    fn eq(&self, other: &Self) -> bool {
        const EPS: f64 = 1e-12;

        let vecs_eq = |a: &[Vec<f64>], b: &[Vec<f64>]| -> bool {
            a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(va, vb)| {
                    va.len() == vb.len()
                        && va.iter().zip(vb.iter()).all(|(x, y)| (x - y).abs() < EPS)
                })
        };

        if !vecs_eq(&self.point, &other.point) {
            return false;
        }

        match (&self.lower, &other.lower) {
            (Some(a), Some(b)) => {
                if !vecs_eq(a, b) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        match (&self.upper, &other.upper) {
            (Some(a), Some(b)) => {
                if !vecs_eq(a, b) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        true
    }
}

impl fmt::Display for Forecast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self.dimensions();
        let h = self.horizon();
        let intervals = match (self.has_lower(), self.has_upper()) {
            (true, true) => "lower+upper",
            (true, false) => "lower only",
            (false, true) => "upper only",
            (false, false) => "none",
        };

        write!(
            f,
            "Forecast(horizon={}, dims={}, intervals={}",
            h, dims, intervals
        )?;

        if h > 0 {
            let primary = self.primary();
            let preview: Vec<String> = primary
                .iter()
                .take(5)
                .map(|v| format!("{:.4}", v))
                .collect();
            let suffix = if h > 5 { ", ..." } else { "" };
            write!(f, ", values=[{}{}]", preview.join(", "), suffix)?;
        }

        write!(f, ")")
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
        let empty: &[f64] = &[];
        assert_eq!(forecast.series(1).unwrap(), empty);
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
        let forecast =
            Forecast::from_values_with_intervals(vec![2.0, 3.0], vec![1.0, 2.0], vec![3.0, 4.0]);

        assert_eq!(forecast.primary(), &[2.0, 3.0]);
        assert_eq!(forecast.lower_series(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(forecast.upper_series(0).unwrap(), &[3.0, 4.0]);
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn forecast_json_round_trip_point_only() {
        let forecast = Forecast::from_values(vec![1.0, 2.5, 3.7, 4.2]);

        let json = forecast.to_json().unwrap();
        let restored = Forecast::from_json(&json).unwrap();

        assert_eq!(forecast, restored);
    }

    #[test]
    fn forecast_json_round_trip_with_intervals() {
        let forecast = Forecast::from_values_with_intervals(
            vec![10.0, 20.0, 30.0],
            vec![8.0, 18.0, 28.0],
            vec![12.0, 22.0, 32.0],
        );

        let json = forecast.to_json().unwrap();
        let restored = Forecast::from_json(&json).unwrap();

        assert_eq!(forecast, restored);
    }

    #[test]
    fn forecast_json_round_trip_empty() {
        let forecast = Forecast::new();

        let json = forecast.to_json().unwrap();
        let restored = Forecast::from_json(&json).unwrap();

        assert!(restored.is_empty());
        assert_eq!(restored.dimensions(), 0);
    }

    #[test]
    fn forecast_json_round_trip_multivariate() {
        let mut forecast = Forecast::with_dimensions(3);
        forecast.series_mut(0).extend([1.0, 2.0]);
        forecast.series_mut(1).extend([3.0, 4.0]);
        forecast.series_mut(2).extend([5.0, 6.0]);

        let json = forecast.to_json().unwrap();
        let restored = Forecast::from_json(&json).unwrap();

        assert_eq!(forecast, restored);
        assert_eq!(restored.dimensions(), 3);
        assert!(restored.is_multivariate());
    }

    #[test]
    fn forecast_from_json_rejects_invalid_json() {
        let result = Forecast::from_json("not valid json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ForecastError::SerializationError(_)),
            "expected SerializationError, got {:?}",
            err
        );
    }
}
