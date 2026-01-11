//! Data types for probabilistic forecasting.
//!
//! This module provides core data structures for representing point forecasts,
//! quantile forecasts, and prediction intervals used in postprocessing methods.

use chrono::{DateTime, Utc};

use crate::error::{ForecastError, Result};

/// Point forecasts with metadata.
///
/// Represents a sequence of point predictions from a forecasting model,
/// along with timestamps and optional model identification.
///
/// # Example
///
/// ```
/// use chrono::{TimeZone, Utc};
/// use anofox_forecast::postprocess::PointForecasts;
///
/// let timestamps = vec![
///     Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
///     Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
/// ];
/// let values = vec![10.0, 12.0];
///
/// let forecasts = PointForecasts::new(timestamps, values).unwrap();
/// assert_eq!(forecasts.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PointForecasts {
    /// Timestamps for each forecast point
    timestamps: Vec<DateTime<Utc>>,
    /// Predicted values
    values: Vec<f64>,
    /// Optional model name that generated these forecasts
    model_name: Option<String>,
}

impl PointForecasts {
    /// Create new point forecasts from timestamps and values.
    ///
    /// # Errors
    ///
    /// Returns an error if timestamps and values have different lengths.
    pub fn new(timestamps: Vec<DateTime<Utc>>, values: Vec<f64>) -> Result<Self> {
        if timestamps.len() != values.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: timestamps.len(),
                got: values.len(),
            });
        }
        Ok(Self {
            timestamps,
            values,
            model_name: None,
        })
    }

    /// Create point forecasts from values only (without timestamps).
    ///
    /// Useful when timestamps are not needed or will be added later.
    pub fn from_values(values: Vec<f64>) -> Self {
        Self {
            timestamps: Vec::new(),
            values,
            model_name: None,
        }
    }

    /// Create an empty PointForecasts instance.
    pub fn empty() -> Self {
        Self {
            timestamps: Vec::new(),
            values: Vec::new(),
            model_name: None,
        }
    }

    /// Set the model name.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Get the number of forecast points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the timestamps.
    pub fn timestamps(&self) -> &[DateTime<Utc>] {
        &self.timestamps
    }

    /// Get the forecast values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get a mutable reference to the values.
    pub fn values_mut(&mut self) -> &mut Vec<f64> {
        &mut self.values
    }

    /// Get the model name if set.
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// Check if timestamps are available.
    pub fn has_timestamps(&self) -> bool {
        !self.timestamps.is_empty()
    }

    /// Get a specific value by index.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    /// Get a specific timestamp by index.
    pub fn get_timestamp(&self, index: usize) -> Option<&DateTime<Utc>> {
        self.timestamps.get(index)
    }

    /// Iterate over (timestamp, value) pairs.
    ///
    /// Only yields pairs where both timestamp and value exist.
    pub fn iter(&self) -> impl Iterator<Item = (&DateTime<Utc>, f64)> {
        self.timestamps.iter().zip(self.values.iter().copied())
    }

    /// Iterate over values only.
    pub fn iter_values(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().copied()
    }
}

/// Quantile forecasts representing a full predictive distribution.
///
/// Contains forecasts at multiple quantile levels, forming a discrete
/// approximation of the conditional distribution at each time point.
///
/// # Example
///
/// ```
/// use chrono::{TimeZone, Utc};
/// use anofox_forecast::postprocess::QuantileForecasts;
///
/// let timestamps = vec![
///     Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
///     Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
/// ];
/// let quantiles = vec![0.1, 0.5, 0.9];
/// // Values matrix: rows = time points, cols = quantiles
/// let values = vec![
///     vec![8.0, 10.0, 12.0],   // t=1: q10, q50, q90
///     vec![9.0, 12.0, 15.0],   // t=2: q10, q50, q90
/// ];
///
/// let forecasts = QuantileForecasts::new(timestamps, quantiles, values).unwrap();
/// assert_eq!(forecasts.n_times(), 2);
/// assert_eq!(forecasts.n_quantiles(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileForecasts {
    /// Timestamps for each forecast point
    timestamps: Vec<DateTime<Utc>>,
    /// Quantile levels (e.g., [0.1, 0.5, 0.9])
    quantiles: Vec<f64>,
    /// Values matrix: values[time_idx][quantile_idx]
    values: Vec<Vec<f64>>,
}

impl QuantileForecasts {
    /// Create new quantile forecasts.
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Timestamps for each forecast point
    /// * `quantiles` - Quantile levels (must be in (0, 1))
    /// * `values` - Matrix where values[t][q] is the q-th quantile at time t
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Timestamps length doesn't match values rows
    /// - Any value row doesn't match quantiles length
    /// - Quantiles are not in (0, 1) range
    /// - Quantiles are not sorted
    pub fn new(
        timestamps: Vec<DateTime<Utc>>,
        quantiles: Vec<f64>,
        values: Vec<Vec<f64>>,
    ) -> Result<Self> {
        // Validate timestamps match values rows
        if !timestamps.is_empty() && timestamps.len() != values.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: timestamps.len(),
                got: values.len(),
            });
        }

        // Validate quantiles are in valid range
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(ForecastError::InvalidParameter(format!(
                    "quantile must be in (0, 1), got {}",
                    q
                )));
            }
        }

        // Validate quantiles are sorted
        for window in quantiles.windows(2) {
            if window[0] >= window[1] {
                return Err(ForecastError::InvalidParameter(
                    "quantiles must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Validate each row has correct number of quantiles
        for (i, row) in values.iter().enumerate() {
            if row.len() != quantiles.len() {
                return Err(ForecastError::DimensionMismatch {
                    expected: quantiles.len(),
                    got: row.len(),
                });
            }
            // Validate quantile values are monotonically increasing
            for window in row.windows(2) {
                if window[0] > window[1] {
                    return Err(ForecastError::InvalidParameter(format!(
                        "quantile values must be non-decreasing at time index {}",
                        i
                    )));
                }
            }
        }

        Ok(Self {
            timestamps,
            quantiles,
            values,
        })
    }

    /// Create quantile forecasts from values only (without timestamps).
    pub fn from_values(quantiles: Vec<f64>, values: Vec<Vec<f64>>) -> Result<Self> {
        Self::new(Vec::new(), quantiles, values)
    }

    /// Create an empty QuantileForecasts instance with specified quantiles.
    pub fn empty(quantiles: Vec<f64>) -> Result<Self> {
        // Validate quantiles
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(ForecastError::InvalidParameter(format!(
                    "quantile must be in (0, 1), got {}",
                    q
                )));
            }
        }
        for window in quantiles.windows(2) {
            if window[0] >= window[1] {
                return Err(ForecastError::InvalidParameter(
                    "quantiles must be sorted in ascending order".to_string(),
                ));
            }
        }
        Ok(Self {
            timestamps: Vec::new(),
            quantiles,
            values: Vec::new(),
        })
    }

    /// Get the number of time points.
    pub fn n_times(&self) -> usize {
        self.values.len()
    }

    /// Get the number of quantiles.
    pub fn n_quantiles(&self) -> usize {
        self.quantiles.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the timestamps.
    pub fn timestamps(&self) -> &[DateTime<Utc>] {
        &self.timestamps
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Get the values matrix.
    pub fn values(&self) -> &[Vec<f64>] {
        &self.values
    }

    /// Check if timestamps are available.
    pub fn has_timestamps(&self) -> bool {
        !self.timestamps.is_empty()
    }

    /// Get values at a specific time index.
    pub fn at_time(&self, time_idx: usize) -> Option<&[f64]> {
        self.values.get(time_idx).map(|v| v.as_slice())
    }

    /// Get values for a specific quantile across all times.
    pub fn at_quantile(&self, quantile_idx: usize) -> Option<Vec<f64>> {
        if quantile_idx >= self.quantiles.len() {
            return None;
        }
        Some(
            self.values
                .iter()
                .map(|row| row[quantile_idx])
                .collect(),
        )
    }

    /// Get the median (0.5 quantile) if available.
    pub fn median(&self) -> Option<Vec<f64>> {
        self.quantiles
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-10)
            .and_then(|idx| self.at_quantile(idx))
    }

    /// Get a specific value by time and quantile index.
    pub fn get(&self, time_idx: usize, quantile_idx: usize) -> Option<f64> {
        self.values
            .get(time_idx)
            .and_then(|row| row.get(quantile_idx))
            .copied()
    }

    /// Extract prediction intervals at a given coverage level.
    ///
    /// Finds the symmetric quantiles closest to (1-coverage)/2 and 1-(1-coverage)/2.
    ///
    /// # Arguments
    ///
    /// * `coverage` - Desired coverage level (e.g., 0.9 for 90% intervals)
    ///
    /// # Returns
    ///
    /// PredictionIntervals if suitable quantiles are found, None otherwise.
    pub fn to_prediction_intervals(&self, coverage: f64) -> Option<PredictionIntervals> {
        if coverage <= 0.0 || coverage >= 1.0 {
            return None;
        }

        let alpha = 1.0 - coverage;
        let lower_target = alpha / 2.0;
        let upper_target = 1.0 - alpha / 2.0;

        // Find closest quantiles
        let lower_idx = self
            .quantiles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - lower_target).abs())
                    .partial_cmp(&((*b - lower_target).abs()))
                    .unwrap()
            })?
            .0;

        let upper_idx = self
            .quantiles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - upper_target).abs())
                    .partial_cmp(&((*b - upper_target).abs()))
                    .unwrap()
            })?
            .0;

        let lower = self.at_quantile(lower_idx)?;
        let upper = self.at_quantile(upper_idx)?;

        Some(PredictionIntervals {
            timestamps: self.timestamps.clone(),
            lower,
            upper,
            coverage,
        })
    }
}

/// Prediction intervals with coverage level.
///
/// Represents lower and upper bounds of prediction intervals
/// at a specified coverage level.
///
/// # Example
///
/// ```
/// use chrono::{TimeZone, Utc};
/// use anofox_forecast::postprocess::PredictionIntervals;
///
/// let timestamps = vec![
///     Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
///     Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
/// ];
///
/// let intervals = PredictionIntervals::new(
///     timestamps,
///     vec![8.0, 9.0],    // lower bounds
///     vec![12.0, 15.0],  // upper bounds
///     0.90,              // 90% coverage
/// ).unwrap();
///
/// assert_eq!(intervals.len(), 2);
/// assert_eq!(intervals.coverage(), 0.90);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionIntervals {
    /// Timestamps for each interval
    timestamps: Vec<DateTime<Utc>>,
    /// Lower bounds of intervals
    lower: Vec<f64>,
    /// Upper bounds of intervals
    upper: Vec<f64>,
    /// Nominal coverage level (e.g., 0.90 for 90%)
    coverage: f64,
}

impl PredictionIntervals {
    /// Create new prediction intervals.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Lower and upper have different lengths
    /// - Timestamps length doesn't match (if non-empty)
    /// - Coverage is not in (0, 1)
    /// - Any lower bound exceeds corresponding upper bound
    pub fn new(
        timestamps: Vec<DateTime<Utc>>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        coverage: f64,
    ) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: lower.len(),
                got: upper.len(),
            });
        }

        if !timestamps.is_empty() && timestamps.len() != lower.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: timestamps.len(),
                got: lower.len(),
            });
        }

        if coverage <= 0.0 || coverage >= 1.0 {
            return Err(ForecastError::InvalidParameter(format!(
                "coverage must be in (0, 1), got {}",
                coverage
            )));
        }

        // Validate lower <= upper
        for (i, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
            if l > u {
                return Err(ForecastError::InvalidParameter(format!(
                    "lower bound {} exceeds upper bound {} at index {}",
                    l, u, i
                )));
            }
        }

        Ok(Self {
            timestamps,
            lower,
            upper,
            coverage,
        })
    }

    /// Create prediction intervals from bounds only (without timestamps).
    pub fn from_bounds(lower: Vec<f64>, upper: Vec<f64>, coverage: f64) -> Result<Self> {
        Self::new(Vec::new(), lower, upper, coverage)
    }

    /// Get the number of intervals.
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Get the timestamps.
    pub fn timestamps(&self) -> &[DateTime<Utc>] {
        &self.timestamps
    }

    /// Get the lower bounds.
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Get the upper bounds.
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Check if timestamps are available.
    pub fn has_timestamps(&self) -> bool {
        !self.timestamps.is_empty()
    }

    /// Get interval at a specific index.
    pub fn get(&self, index: usize) -> Option<(f64, f64)> {
        match (self.lower.get(index), self.upper.get(index)) {
            (Some(&l), Some(&u)) => Some((l, u)),
            _ => None,
        }
    }

    /// Get the width of each interval.
    pub fn widths(&self) -> Vec<f64> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(l, u)| u - l)
            .collect()
    }

    /// Get the midpoint of each interval.
    pub fn midpoints(&self) -> Vec<f64> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(l, u)| (l + u) / 2.0)
            .collect()
    }

    /// Check if a value falls within the interval at each time point.
    pub fn contains(&self, values: &[f64]) -> Vec<bool> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .zip(values.iter())
            .map(|((&l, &u), &v)| v >= l && v <= u)
            .collect()
    }

    /// Calculate empirical coverage given actual values.
    pub fn empirical_coverage(&self, actuals: &[f64]) -> Option<f64> {
        if actuals.len() != self.len() {
            return None;
        }
        let contained = self.contains(actuals);
        let count = contained.iter().filter(|&&x| x).count();
        Some(count as f64 / actuals.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }

    // =========================================================================
    // PointForecasts tests
    // =========================================================================

    mod point_forecasts {
        use super::*;

        #[test]
        fn new_creates_valid_forecasts() {
            let timestamps = make_timestamps(3);
            let values = vec![1.0, 2.0, 3.0];

            let forecasts = PointForecasts::new(timestamps.clone(), values.clone()).unwrap();

            assert_eq!(forecasts.len(), 3);
            assert!(!forecasts.is_empty());
            assert!(forecasts.has_timestamps());
            assert_eq!(forecasts.values(), &values);
            assert_eq!(forecasts.timestamps(), &timestamps);
        }

        #[test]
        fn new_fails_on_length_mismatch() {
            let timestamps = make_timestamps(3);
            let values = vec![1.0, 2.0]; // Wrong length

            let result = PointForecasts::new(timestamps, values);
            assert!(result.is_err());

            match result {
                Err(ForecastError::DimensionMismatch { expected, got }) => {
                    assert_eq!(expected, 3);
                    assert_eq!(got, 2);
                }
                _ => panic!("Expected DimensionMismatch error"),
            }
        }

        #[test]
        fn from_values_creates_without_timestamps() {
            let forecasts = PointForecasts::from_values(vec![1.0, 2.0, 3.0]);

            assert_eq!(forecasts.len(), 3);
            assert!(!forecasts.has_timestamps());
            assert!(forecasts.timestamps().is_empty());
        }

        #[test]
        fn empty_creates_empty_instance() {
            let forecasts = PointForecasts::empty();

            assert!(forecasts.is_empty());
            assert_eq!(forecasts.len(), 0);
            assert!(!forecasts.has_timestamps());
        }

        #[test]
        fn with_model_name_sets_name() {
            let forecasts = PointForecasts::from_values(vec![1.0])
                .with_model_name("ARIMA");

            assert_eq!(forecasts.model_name(), Some("ARIMA"));
        }

        #[test]
        fn model_name_is_none_by_default() {
            let forecasts = PointForecasts::from_values(vec![1.0]);
            assert!(forecasts.model_name().is_none());
        }

        #[test]
        fn get_returns_value_at_index() {
            let forecasts = PointForecasts::from_values(vec![10.0, 20.0, 30.0]);

            assert_eq!(forecasts.get(0), Some(10.0));
            assert_eq!(forecasts.get(1), Some(20.0));
            assert_eq!(forecasts.get(2), Some(30.0));
            assert_eq!(forecasts.get(3), None);
        }

        #[test]
        fn get_timestamp_returns_timestamp_at_index() {
            let timestamps = make_timestamps(3);
            let forecasts = PointForecasts::new(timestamps.clone(), vec![1.0, 2.0, 3.0]).unwrap();

            assert_eq!(forecasts.get_timestamp(0), Some(&timestamps[0]));
            assert_eq!(forecasts.get_timestamp(2), Some(&timestamps[2]));
            assert_eq!(forecasts.get_timestamp(3), None);
        }

        #[test]
        fn iter_yields_timestamp_value_pairs() {
            let timestamps = make_timestamps(3);
            let values = vec![1.0, 2.0, 3.0];
            let forecasts = PointForecasts::new(timestamps.clone(), values.clone()).unwrap();

            let pairs: Vec<_> = forecasts.iter().collect();

            assert_eq!(pairs.len(), 3);
            assert_eq!(pairs[0], (&timestamps[0], 1.0));
            assert_eq!(pairs[1], (&timestamps[1], 2.0));
            assert_eq!(pairs[2], (&timestamps[2], 3.0));
        }

        #[test]
        fn iter_values_yields_only_values() {
            let forecasts = PointForecasts::from_values(vec![1.0, 2.0, 3.0]);

            let values: Vec<_> = forecasts.iter_values().collect();
            assert_eq!(values, vec![1.0, 2.0, 3.0]);
        }

        #[test]
        fn values_mut_allows_modification() {
            let mut forecasts = PointForecasts::from_values(vec![1.0, 2.0, 3.0]);

            forecasts.values_mut()[1] = 20.0;

            assert_eq!(forecasts.values(), &[1.0, 20.0, 3.0]);
        }

        #[test]
        fn clone_creates_independent_copy() {
            let forecasts = PointForecasts::from_values(vec![1.0, 2.0]).with_model_name("Test");
            let cloned = forecasts.clone();

            assert_eq!(forecasts, cloned);
        }
    }

    // =========================================================================
    // QuantileForecasts tests
    // =========================================================================

    mod quantile_forecasts {
        use super::*;

        #[test]
        fn new_creates_valid_forecasts() {
            let timestamps = make_timestamps(2);
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0],
            ];

            let forecasts = QuantileForecasts::new(
                timestamps.clone(),
                quantiles.clone(),
                values.clone(),
            )
            .unwrap();

            assert_eq!(forecasts.n_times(), 2);
            assert_eq!(forecasts.n_quantiles(), 3);
            assert!(!forecasts.is_empty());
            assert!(forecasts.has_timestamps());
        }

        #[test]
        fn new_fails_on_timestamp_mismatch() {
            let timestamps = make_timestamps(3); // 3 timestamps
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0], // Only 2 rows
            ];

            let result = QuantileForecasts::new(timestamps, quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_quantile_row_mismatch() {
            let timestamps = make_timestamps(2);
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0], // Wrong length
            ];

            let result = QuantileForecasts::new(timestamps, quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_invalid_quantile_below_zero() {
            let quantiles = vec![-0.1, 0.5, 0.9];
            let values = vec![vec![8.0, 10.0, 12.0]];

            let result = QuantileForecasts::from_values(quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_invalid_quantile_above_one() {
            let quantiles = vec![0.1, 0.5, 1.0];
            let values = vec![vec![8.0, 10.0, 12.0]];

            let result = QuantileForecasts::from_values(quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_unsorted_quantiles() {
            let quantiles = vec![0.9, 0.5, 0.1]; // Not sorted
            let values = vec![vec![12.0, 10.0, 8.0]];

            let result = QuantileForecasts::from_values(quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_non_monotonic_values() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![vec![10.0, 8.0, 12.0]]; // q50 < q10

            let result = QuantileForecasts::from_values(quantiles, values);
            assert!(result.is_err());
        }

        #[test]
        fn from_values_creates_without_timestamps() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![vec![8.0, 10.0, 12.0]];

            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert!(!forecasts.has_timestamps());
            assert_eq!(forecasts.n_times(), 1);
        }

        #[test]
        fn empty_creates_empty_instance() {
            let forecasts = QuantileForecasts::empty(vec![0.1, 0.5, 0.9]).unwrap();

            assert!(forecasts.is_empty());
            assert_eq!(forecasts.n_times(), 0);
            assert_eq!(forecasts.n_quantiles(), 3);
        }

        #[test]
        fn at_time_returns_row() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0],
            ];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert_eq!(forecasts.at_time(0), Some(&[8.0, 10.0, 12.0][..]));
            assert_eq!(forecasts.at_time(1), Some(&[9.0, 12.0, 15.0][..]));
            assert_eq!(forecasts.at_time(2), None);
        }

        #[test]
        fn at_quantile_returns_column() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0],
            ];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert_eq!(forecasts.at_quantile(0), Some(vec![8.0, 9.0]));
            assert_eq!(forecasts.at_quantile(1), Some(vec![10.0, 12.0]));
            assert_eq!(forecasts.at_quantile(2), Some(vec![12.0, 15.0]));
            assert_eq!(forecasts.at_quantile(3), None);
        }

        #[test]
        fn median_returns_50th_quantile() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0],
            ];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert_eq!(forecasts.median(), Some(vec![10.0, 12.0]));
        }

        #[test]
        fn median_returns_none_when_not_present() {
            let quantiles = vec![0.1, 0.9];
            let values = vec![vec![8.0, 12.0]];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert_eq!(forecasts.median(), None);
        }

        #[test]
        fn get_returns_specific_value() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![
                vec![8.0, 10.0, 12.0],
                vec![9.0, 12.0, 15.0],
            ];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert_eq!(forecasts.get(0, 0), Some(8.0));
            assert_eq!(forecasts.get(1, 2), Some(15.0));
            assert_eq!(forecasts.get(2, 0), None);
            assert_eq!(forecasts.get(0, 3), None);
        }

        #[test]
        fn to_prediction_intervals_extracts_bounds() {
            let quantiles = vec![0.05, 0.1, 0.5, 0.9, 0.95];
            let values = vec![
                vec![5.0, 8.0, 10.0, 12.0, 15.0],
                vec![6.0, 9.0, 12.0, 15.0, 18.0],
            ];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            let intervals = forecasts.to_prediction_intervals(0.90).unwrap();

            assert_eq!(intervals.coverage(), 0.90);
            assert_eq!(intervals.len(), 2);
            // Should pick q0.05 and q0.95 for 90% coverage
            assert_eq!(intervals.lower(), &[5.0, 6.0]);
            assert_eq!(intervals.upper(), &[15.0, 18.0]);
        }

        #[test]
        fn to_prediction_intervals_returns_none_for_invalid_coverage() {
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![vec![8.0, 10.0, 12.0]];
            let forecasts = QuantileForecasts::from_values(quantiles, values).unwrap();

            assert!(forecasts.to_prediction_intervals(0.0).is_none());
            assert!(forecasts.to_prediction_intervals(1.0).is_none());
            assert!(forecasts.to_prediction_intervals(-0.1).is_none());
        }

        #[test]
        fn allows_equal_quantile_values() {
            // Equal values (flat distribution) should be allowed
            let quantiles = vec![0.1, 0.5, 0.9];
            let values = vec![vec![10.0, 10.0, 10.0]];

            let result = QuantileForecasts::from_values(quantiles, values);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // PredictionIntervals tests
    // =========================================================================

    mod prediction_intervals {
        use super::*;

        #[test]
        fn new_creates_valid_intervals() {
            let timestamps = make_timestamps(3);
            let lower = vec![8.0, 9.0, 10.0];
            let upper = vec![12.0, 13.0, 14.0];

            let intervals = PredictionIntervals::new(
                timestamps.clone(),
                lower.clone(),
                upper.clone(),
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.len(), 3);
            assert!(!intervals.is_empty());
            assert!(intervals.has_timestamps());
            assert_eq!(intervals.coverage(), 0.90);
            assert_eq!(intervals.lower(), &lower);
            assert_eq!(intervals.upper(), &upper);
        }

        #[test]
        fn new_fails_on_length_mismatch() {
            let lower = vec![8.0, 9.0];
            let upper = vec![12.0, 13.0, 14.0];

            let result = PredictionIntervals::from_bounds(lower, upper, 0.90);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_timestamp_mismatch() {
            let timestamps = make_timestamps(2);
            let lower = vec![8.0, 9.0, 10.0];
            let upper = vec![12.0, 13.0, 14.0];

            let result = PredictionIntervals::new(timestamps, lower, upper, 0.90);
            assert!(result.is_err());
        }

        #[test]
        fn new_fails_on_invalid_coverage() {
            let lower = vec![8.0];
            let upper = vec![12.0];

            assert!(PredictionIntervals::from_bounds(lower.clone(), upper.clone(), 0.0).is_err());
            assert!(PredictionIntervals::from_bounds(lower.clone(), upper.clone(), 1.0).is_err());
            assert!(PredictionIntervals::from_bounds(lower.clone(), upper.clone(), -0.5).is_err());
            assert!(PredictionIntervals::from_bounds(lower.clone(), upper.clone(), 1.5).is_err());
        }

        #[test]
        fn new_fails_when_lower_exceeds_upper() {
            let lower = vec![8.0, 15.0, 10.0]; // 15.0 > 13.0
            let upper = vec![12.0, 13.0, 14.0];

            let result = PredictionIntervals::from_bounds(lower, upper, 0.90);
            assert!(result.is_err());
        }

        #[test]
        fn from_bounds_creates_without_timestamps() {
            let intervals =
                PredictionIntervals::from_bounds(vec![8.0], vec![12.0], 0.90).unwrap();

            assert!(!intervals.has_timestamps());
        }

        #[test]
        fn get_returns_interval_at_index() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 9.0, 10.0],
                vec![12.0, 13.0, 14.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.get(0), Some((8.0, 12.0)));
            assert_eq!(intervals.get(1), Some((9.0, 13.0)));
            assert_eq!(intervals.get(2), Some((10.0, 14.0)));
            assert_eq!(intervals.get(3), None);
        }

        #[test]
        fn widths_calculates_interval_widths() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 9.0, 10.0],
                vec![12.0, 15.0, 11.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.widths(), vec![4.0, 6.0, 1.0]);
        }

        #[test]
        fn midpoints_calculates_interval_centers() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 10.0],
                vec![12.0, 20.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.midpoints(), vec![10.0, 15.0]);
        }

        #[test]
        fn contains_checks_value_in_interval() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 9.0, 10.0],
                vec![12.0, 13.0, 14.0],
                0.90,
            )
            .unwrap();

            let values = vec![10.0, 15.0, 10.0]; // In, Out, In
            assert_eq!(intervals.contains(&values), vec![true, false, true]);
        }

        #[test]
        fn contains_includes_boundary_values() {
            let intervals =
                PredictionIntervals::from_bounds(vec![8.0], vec![12.0], 0.90).unwrap();

            assert_eq!(intervals.contains(&[8.0]), vec![true]); // Lower bound
            assert_eq!(intervals.contains(&[12.0]), vec![true]); // Upper bound
        }

        #[test]
        fn empirical_coverage_calculates_correctly() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 9.0, 10.0, 11.0],
                vec![12.0, 13.0, 14.0, 15.0],
                0.90,
            )
            .unwrap();

            // 3 out of 4 values are within intervals
            let actuals = vec![10.0, 15.0, 12.0, 13.0];
            assert_eq!(intervals.empirical_coverage(&actuals), Some(0.75));
        }

        #[test]
        fn empirical_coverage_returns_none_on_length_mismatch() {
            let intervals = PredictionIntervals::from_bounds(
                vec![8.0, 9.0],
                vec![12.0, 13.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.empirical_coverage(&[10.0]), None);
        }

        #[test]
        fn empirical_coverage_handles_all_contained() {
            let intervals = PredictionIntervals::from_bounds(
                vec![0.0, 0.0],
                vec![100.0, 100.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.empirical_coverage(&[50.0, 50.0]), Some(1.0));
        }

        #[test]
        fn empirical_coverage_handles_none_contained() {
            let intervals = PredictionIntervals::from_bounds(
                vec![0.0, 0.0],
                vec![10.0, 10.0],
                0.90,
            )
            .unwrap();

            assert_eq!(intervals.empirical_coverage(&[50.0, 50.0]), Some(0.0));
        }

        #[test]
        fn allows_equal_lower_and_upper() {
            // Point intervals should be allowed
            let intervals =
                PredictionIntervals::from_bounds(vec![10.0], vec![10.0], 0.90).unwrap();

            assert_eq!(intervals.widths(), vec![0.0]);
            assert_eq!(intervals.contains(&[10.0]), vec![true]);
        }
    }
}
