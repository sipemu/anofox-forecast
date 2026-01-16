//! TimeSeries wrapper for JavaScript.

use anofox_forecast::core::{Forecast as InnerForecast, TimeSeries as InnerTimeSeries};
use wasm_bindgen::prelude::*;

/// A time series wrapper for JavaScript.
///
/// Represents a univariate time series with values indexed by position.
/// Timestamps are optional and can be provided as milliseconds since Unix epoch.
#[wasm_bindgen]
pub struct TimeSeries {
    inner: InnerTimeSeries,
}

#[wasm_bindgen]
impl TimeSeries {
    /// Create a new time series from an array of values.
    ///
    /// @param values - Array of numeric values
    /// @returns A new TimeSeries instance
    #[wasm_bindgen(constructor)]
    pub fn new(values: &[f64]) -> Result<TimeSeries, JsError> {
        // Create timestamps as sequential integers (as DateTime)
        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..values.len())
            .map(|i| chrono::DateTime::from_timestamp(i as i64, 0).unwrap_or_else(chrono::Utc::now))
            .collect();

        let inner = InnerTimeSeries::univariate(timestamps, values.to_vec())
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(TimeSeries { inner })
    }

    /// Create a time series with timestamps.
    ///
    /// @param values - Array of numeric values
    /// @param timestamps - Array of timestamps as milliseconds since Unix epoch
    /// @returns A new TimeSeries instance
    #[wasm_bindgen(js_name = withTimestamps)]
    pub fn with_timestamps(values: &[f64], timestamps_ms: &[f64]) -> Result<TimeSeries, JsError> {
        if values.len() != timestamps_ms.len() {
            return Err(JsError::new(&format!(
                "values length ({}) must match timestamps length ({})",
                values.len(),
                timestamps_ms.len()
            )));
        }

        let timestamps: Vec<chrono::DateTime<chrono::Utc>> = timestamps_ms
            .iter()
            .map(|&ms| {
                let secs = (ms / 1000.0) as i64;
                let nanos = ((ms % 1000.0) * 1_000_000.0) as u32;
                chrono::DateTime::from_timestamp(secs, nanos).unwrap_or_else(chrono::Utc::now)
            })
            .collect();

        let inner = InnerTimeSeries::univariate(timestamps, values.to_vec())
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(TimeSeries { inner })
    }

    /// Get the number of observations.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Check if the series is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get all values as an array.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.inner.primary_values().to_vec()
    }

    /// Get a slice of values.
    ///
    /// @param start - Start index (inclusive)
    /// @param end - End index (exclusive)
    /// @returns Array of values in the specified range
    pub fn slice(&self, start: usize, end: usize) -> Result<TimeSeries, JsError> {
        let sliced = self
            .inner
            .slice(start, end)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(TimeSeries { inner: sliced })
    }

    /// Check if the series has missing values (NaN).
    #[wasm_bindgen(js_name = hasMissingValues)]
    pub fn has_missing_values(&self) -> bool {
        self.inner.has_missing_values()
    }

    /// Get the inner TimeSeries (for internal use).
    pub(crate) fn inner(&self) -> &InnerTimeSeries {
        &self.inner
    }
}

/// Forecast result wrapper for JavaScript.
///
/// Contains point predictions and optional prediction intervals.
#[wasm_bindgen]
pub struct Forecast {
    inner: InnerForecast,
}

#[wasm_bindgen]
impl Forecast {
    /// Create a new Forecast from internal type.
    pub(crate) fn from_inner(inner: InnerForecast) -> Self {
        Forecast { inner }
    }

    /// Get the forecast horizon (number of predictions).
    #[wasm_bindgen(getter)]
    pub fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    /// Get point predictions.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.inner.primary().to_vec()
    }

    /// Check if lower prediction interval is available.
    #[wasm_bindgen(js_name = hasLower)]
    pub fn has_lower(&self) -> bool {
        self.inner.has_lower()
    }

    /// Check if upper prediction interval is available.
    #[wasm_bindgen(js_name = hasUpper)]
    pub fn has_upper(&self) -> bool {
        self.inner.has_upper()
    }

    /// Get lower prediction interval bounds.
    ///
    /// @returns Array of lower bounds, or undefined if not available
    #[wasm_bindgen(getter)]
    pub fn lower(&self) -> Option<Vec<f64>> {
        self.inner.lower_series(0).ok().map(|s| s.to_vec())
    }

    /// Get upper prediction interval bounds.
    ///
    /// @returns Array of upper bounds, or undefined if not available
    #[wasm_bindgen(getter)]
    pub fn upper(&self) -> Option<Vec<f64>> {
        self.inner.upper_series(0).ok().map(|s| s.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_time_series_creation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(&values).unwrap();

        assert_eq!(ts.length(), 5);
        assert!(!ts.is_empty());
        assert_eq!(ts.values(), values);
    }

    #[wasm_bindgen_test]
    fn test_time_series_with_timestamps() {
        let values = vec![1.0, 2.0, 3.0];
        let timestamps = vec![1000.0, 2000.0, 3000.0]; // milliseconds

        let ts = TimeSeries::with_timestamps(&values, &timestamps).unwrap();
        assert_eq!(ts.length(), 3);
    }

    #[wasm_bindgen_test]
    fn test_time_series_slice() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(&values).unwrap();

        let sliced = ts.slice(1, 4).unwrap();
        assert_eq!(sliced.length(), 3);
        assert_eq!(sliced.values(), vec![2.0, 3.0, 4.0]);
    }
}
