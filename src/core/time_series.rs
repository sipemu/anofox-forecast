//! TimeSeries data structure for representing temporal data.

use crate::error::{ForecastError, Result};
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::HashMap;

/// Layout of multivariate data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueLayout {
    /// Each inner vector is a dimension (column-major).
    #[default]
    Column,
    /// Each inner vector is an observation across dimensions (row-major).
    Row,
}

/// Policy for handling missing values (NaN/Inf).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissingValuePolicy {
    /// Drop observations with missing values.
    Drop,
    /// Fill with a specific value.
    Fill(f64),
    /// Forward fill (use previous valid value).
    ForwardFill,
    /// Return error if missing values found.
    Error,
}

/// Calendar annotations for holidays and regressors.
#[derive(Debug, Clone, Default)]
pub struct CalendarAnnotations {
    /// Holiday dates.
    holidays: Vec<DateTime<Utc>>,
    /// Named regressors with values per timestamp.
    regressors: HashMap<String, Vec<f64>>,
}

impl CalendarAnnotations {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_holidays(mut self, holidays: Vec<DateTime<Utc>>) -> Self {
        self.holidays = holidays;
        self
    }

    pub fn with_regressor(mut self, name: String, values: Vec<f64>) -> Self {
        self.regressors.insert(name, values);
        self
    }

    pub fn holidays(&self) -> &[DateTime<Utc>] {
        &self.holidays
    }

    pub fn regressors(&self) -> &HashMap<String, Vec<f64>> {
        &self.regressors
    }

    pub fn regressor(&self, name: &str) -> Option<&[f64]> {
        self.regressors.get(name).map(|v| v.as_slice())
    }

    pub fn has_regressors(&self) -> bool {
        !self.regressors.is_empty()
    }

    pub fn is_holiday(&self, timestamp: &DateTime<Utc>) -> bool {
        self.holidays.iter().any(|h| {
            // Check if timestamp falls on the same day as any holiday
            h.date_naive() == timestamp.date_naive()
        })
    }

    pub fn is_business_day(&self, timestamp: &DateTime<Utc>) -> bool {
        let weekday = timestamp.weekday();
        !matches!(weekday, chrono::Weekday::Sat | chrono::Weekday::Sun)
            && !self.is_holiday(timestamp)
    }
}

/// A time series with timestamps and values.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    timestamps: Vec<DateTime<Utc>>,
    /// Values stored in column-major format: values[dimension][observation]
    values: Vec<Vec<f64>>,
    labels: Vec<String>,
    metadata: HashMap<String, String>,
    dimension_metadata: Vec<HashMap<String, String>>,
    timezone: Option<String>,
    frequency: Option<Duration>,
    calendar: Option<CalendarAnnotations>,
}

/// Builder for constructing TimeSeries.
#[derive(Debug, Clone, Default)]
pub struct TimeSeriesBuilder {
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<Vec<f64>>,
    layout: ValueLayout,
    labels: Vec<String>,
    metadata: HashMap<String, String>,
    dimension_metadata: Vec<HashMap<String, String>>,
    timezone: Option<String>,
    frequency: Option<Duration>,
    calendar: Option<CalendarAnnotations>,
}

impl TimeSeriesBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn timestamps(mut self, timestamps: Vec<DateTime<Utc>>) -> Self {
        self.timestamps = timestamps;
        self
    }

    /// Set univariate values.
    pub fn values(mut self, values: Vec<f64>) -> Self {
        self.values = vec![values];
        self.layout = ValueLayout::Column;
        self
    }

    /// Set multivariate values with specified layout.
    pub fn multivariate_values(mut self, values: Vec<Vec<f64>>, layout: ValueLayout) -> Self {
        self.values = values;
        self.layout = layout;
        self
    }

    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn dimension_metadata(mut self, metadata: Vec<HashMap<String, String>>) -> Self {
        self.dimension_metadata = metadata;
        self
    }

    pub fn timezone(mut self, tz: String) -> Self {
        self.timezone = Some(tz);
        self
    }

    pub fn frequency(mut self, freq: Duration) -> Self {
        self.frequency = Some(freq);
        self
    }

    pub fn calendar(mut self, calendar: CalendarAnnotations) -> Self {
        self.calendar = Some(calendar);
        self
    }

    pub fn build(self) -> Result<TimeSeries> {
        TimeSeries::new(
            self.timestamps,
            self.values,
            self.layout,
            self.labels,
            self.metadata,
            self.dimension_metadata,
            self.timezone,
            self.frequency,
            self.calendar,
        )
    }
}

impl TimeSeries {
    /// Create a new TimeSeries with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamps: Vec<DateTime<Utc>>,
        values: Vec<Vec<f64>>,
        layout: ValueLayout,
        labels: Vec<String>,
        metadata: HashMap<String, String>,
        dimension_metadata: Vec<HashMap<String, String>>,
        timezone: Option<String>,
        frequency: Option<Duration>,
        calendar: Option<CalendarAnnotations>,
    ) -> Result<Self> {
        // Validate timestamps are strictly increasing
        for i in 1..timestamps.len() {
            if timestamps[i] <= timestamps[i - 1] {
                return Err(ForecastError::TimestampError(
                    "timestamps must be strictly increasing".to_string(),
                ));
            }
        }

        // Convert to column-major if needed and validate dimensions
        let values = match layout {
            ValueLayout::Column => {
                // Each inner vector is a dimension
                // Validate all dimensions have same length as timestamps
                for (dim, series) in values.iter().enumerate() {
                    if series.len() != timestamps.len() {
                        return Err(ForecastError::DimensionMismatch {
                            expected: timestamps.len(),
                            got: series.len(),
                        });
                    }
                    // Ensure dimension_metadata matches if provided
                    if !dimension_metadata.is_empty() && dim >= dimension_metadata.len() {
                        return Err(ForecastError::DimensionMismatch {
                            expected: values.len(),
                            got: dimension_metadata.len(),
                        });
                    }
                }
                values
            }
            ValueLayout::Row => {
                // Each inner vector is an observation across dimensions
                if values.len() != timestamps.len() {
                    return Err(ForecastError::DimensionMismatch {
                        expected: timestamps.len(),
                        got: values.len(),
                    });
                }

                // All rows must have the same number of dimensions
                let dims = if values.is_empty() {
                    0
                } else {
                    values[0].len()
                };

                for row in &values {
                    if row.len() != dims {
                        return Err(ForecastError::DimensionMismatch {
                            expected: dims,
                            got: row.len(),
                        });
                    }
                }

                // Transpose to column-major
                if dims == 0 {
                    vec![]
                } else {
                    (0..dims)
                        .map(|d| values.iter().map(|row| row[d]).collect())
                        .collect()
                }
            }
        };

        // Validate labels count if provided
        if !labels.is_empty() && labels.len() != values.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: values.len(),
                got: labels.len(),
            });
        }

        // Validate dimension metadata count if provided
        if !dimension_metadata.is_empty() && dimension_metadata.len() != values.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: values.len(),
                got: dimension_metadata.len(),
            });
        }

        Ok(Self {
            timestamps,
            values,
            labels,
            metadata,
            dimension_metadata,
            timezone,
            frequency,
            calendar,
        })
    }

    /// Create a simple univariate time series.
    pub fn univariate(timestamps: Vec<DateTime<Utc>>, values: Vec<f64>) -> Result<Self> {
        Self::new(
            timestamps,
            vec![values],
            ValueLayout::Column,
            vec![],
            HashMap::new(),
            vec![],
            None,
            None,
            None,
        )
    }

    /// Get the number of observations.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if the series is empty.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Get the number of dimensions (1 for univariate).
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Check if the series is multivariate.
    pub fn is_multivariate(&self) -> bool {
        self.values.len() > 1
    }

    /// Get timestamps.
    pub fn timestamps(&self) -> &[DateTime<Utc>] {
        &self.timestamps
    }

    /// Get values for a specific dimension.
    pub fn values(&self, dimension: usize) -> Result<&[f64]> {
        self.values
            .get(dimension)
            .map(|v| v.as_slice())
            .ok_or(ForecastError::IndexOutOfBounds {
                index: dimension,
                size: self.values.len(),
            })
    }

    /// Get primary (first dimension) values.
    pub fn primary_values(&self) -> &[f64] {
        self.values.first().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all values organized by dimension.
    pub fn values_by_dimension(&self) -> &[Vec<f64>] {
        &self.values
    }

    /// Get a row (observation at index across all dimensions).
    pub fn row(&self, index: usize) -> Result<Vec<f64>> {
        if index >= self.len() {
            return Err(ForecastError::IndexOutOfBounds {
                index,
                size: self.len(),
            });
        }
        Ok(self.values.iter().map(|dim| dim[index]).collect())
    }

    /// Get dimension labels.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Set dimension labels.
    pub fn set_labels(&mut self, labels: Vec<String>) -> Result<()> {
        if !labels.is_empty() && labels.len() != self.dimensions() {
            return Err(ForecastError::DimensionMismatch {
                expected: self.dimensions(),
                got: labels.len(),
            });
        }
        self.labels = labels;
        Ok(())
    }

    /// Get metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get dimension metadata.
    pub fn dimension_metadata(&self) -> &[HashMap<String, String>] {
        &self.dimension_metadata
    }

    /// Set dimension metadata.
    pub fn set_dimension_metadata(&mut self, metadata: Vec<HashMap<String, String>>) -> Result<()> {
        if !metadata.is_empty() && metadata.len() != self.dimensions() {
            return Err(ForecastError::DimensionMismatch {
                expected: self.dimensions(),
                got: metadata.len(),
            });
        }
        self.dimension_metadata = metadata;
        Ok(())
    }

    /// Get timezone.
    pub fn timezone(&self) -> Option<&str> {
        self.timezone.as_deref()
    }

    /// Set timezone.
    pub fn set_timezone(&mut self, tz: String) {
        self.timezone = Some(tz);
    }

    /// Get frequency.
    pub fn frequency(&self) -> Option<Duration> {
        self.frequency
    }

    /// Set frequency.
    pub fn set_frequency(&mut self, freq: Duration) {
        self.frequency = Some(freq);
    }

    /// Clear frequency.
    pub fn clear_frequency(&mut self) {
        self.frequency = None;
    }

    /// Get calendar annotations.
    pub fn calendar(&self) -> Option<&CalendarAnnotations> {
        self.calendar.as_ref()
    }

    /// Set calendar annotations.
    pub fn set_calendar(&mut self, calendar: CalendarAnnotations) {
        self.calendar = Some(calendar);
    }

    /// Clear calendar annotations.
    pub fn clear_calendar(&mut self) {
        self.calendar = None;
    }

    /// Check if timestamp is a holiday.
    pub fn is_holiday(&self, timestamp: &DateTime<Utc>) -> bool {
        self.calendar
            .as_ref()
            .map(|c| c.is_holiday(timestamp))
            .unwrap_or(false)
    }

    /// Check if timestamp is a business day.
    pub fn is_business_day(&self, timestamp: &DateTime<Utc>) -> bool {
        self.calendar
            .as_ref()
            .map(|c| c.is_business_day(timestamp))
            .unwrap_or({
                // Default: weekdays are business days
                !matches!(
                    timestamp.weekday(),
                    chrono::Weekday::Sat | chrono::Weekday::Sun
                )
            })
    }

    /// Check if series has regressors.
    pub fn has_regressors(&self) -> bool {
        self.calendar
            .as_ref()
            .map(|c| c.has_regressors())
            .unwrap_or(false)
    }

    /// Get regressor values by name.
    pub fn regressor(&self, name: &str) -> Option<&[f64]> {
        self.calendar.as_ref().and_then(|c| c.regressor(name))
    }

    /// Extract a slice of the time series.
    pub fn slice(&self, start: usize, end: usize) -> Result<TimeSeries> {
        if start > end {
            return Err(ForecastError::InvalidParameter(
                "start must be <= end".to_string(),
            ));
        }
        if end > self.len() {
            return Err(ForecastError::IndexOutOfBounds {
                index: end,
                size: self.len(),
            });
        }

        let timestamps = self.timestamps[start..end].to_vec();
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| dim[start..end].to_vec())
            .collect();

        Ok(TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency,
            calendar: self.calendar.clone(),
        })
    }

    /// Check if series has missing values (NaN or Inf).
    pub fn has_missing_values(&self) -> bool {
        self.values
            .iter()
            .any(|dim| dim.iter().any(|v| v.is_nan() || v.is_infinite()))
    }

    /// Return a sanitized copy with missing values handled.
    pub fn sanitized(&self, policy: MissingValuePolicy) -> Result<TimeSeries> {
        match policy {
            MissingValuePolicy::Error => {
                if self.has_missing_values() {
                    return Err(ForecastError::MissingValues);
                }
                Ok(self.clone())
            }
            MissingValuePolicy::Drop => {
                // Find indices of valid observations (all dimensions valid)
                let valid_indices: Vec<usize> = (0..self.len())
                    .filter(|&i| {
                        self.values
                            .iter()
                            .all(|dim| !dim[i].is_nan() && !dim[i].is_infinite())
                    })
                    .collect();

                let timestamps: Vec<_> =
                    valid_indices.iter().map(|&i| self.timestamps[i]).collect();
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| valid_indices.iter().map(|&i| dim[i]).collect())
                    .collect();

                Ok(TimeSeries {
                    timestamps,
                    values,
                    labels: self.labels.clone(),
                    metadata: self.metadata.clone(),
                    dimension_metadata: self.dimension_metadata.clone(),
                    timezone: self.timezone.clone(),
                    frequency: self.frequency,
                    calendar: self.calendar.clone(),
                })
            }
            MissingValuePolicy::Fill(fill_value) => {
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| {
                        dim.iter()
                            .map(|&v| {
                                if v.is_nan() || v.is_infinite() {
                                    fill_value
                                } else {
                                    v
                                }
                            })
                            .collect()
                    })
                    .collect();

                Ok(TimeSeries {
                    timestamps: self.timestamps.clone(),
                    values,
                    labels: self.labels.clone(),
                    metadata: self.metadata.clone(),
                    dimension_metadata: self.dimension_metadata.clone(),
                    timezone: self.timezone.clone(),
                    frequency: self.frequency,
                    calendar: self.calendar.clone(),
                })
            }
            MissingValuePolicy::ForwardFill => {
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| {
                        let mut result = Vec::with_capacity(dim.len());
                        let mut last_valid = None;
                        for &v in dim {
                            if v.is_nan() || v.is_infinite() {
                                result.push(last_valid.unwrap_or(v));
                            } else {
                                last_valid = Some(v);
                                result.push(v);
                            }
                        }
                        result
                    })
                    .collect();

                Ok(TimeSeries {
                    timestamps: self.timestamps.clone(),
                    values,
                    labels: self.labels.clone(),
                    metadata: self.metadata.clone(),
                    dimension_metadata: self.dimension_metadata.clone(),
                    timezone: self.timezone.clone(),
                    frequency: self.frequency,
                    calendar: self.calendar.clone(),
                })
            }
        }
    }

    /// Return a copy with linear interpolation for NaN values.
    pub fn interpolated(&self, fill_edges: bool) -> TimeSeries {
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| interpolate_series(dim, fill_edges))
            .collect();

        TimeSeries {
            timestamps: self.timestamps.clone(),
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency,
            calendar: self.calendar.clone(),
        }
    }

    /// Infer frequency from timestamps.
    pub fn infer_frequency(&self, tolerance: f64) -> Result<Duration> {
        if self.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: self.len(),
            });
        }

        // Calculate all differences
        let diffs: Vec<i64> = self
            .timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).num_seconds())
            .collect();

        // Find modal (most common) difference
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &diff in &diffs {
            *counts.entry(diff).or_insert(0) += 1;
        }

        let (modal_diff, modal_count) = counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&diff, &count)| (diff, count))
            .ok_or(ForecastError::FrequencyInference(
                "empty spacing data".to_string(),
            ))?;

        // Check if modal is unique enough
        let total_count: usize = counts.values().sum();
        let modal_ratio = modal_count as f64 / total_count as f64;

        if modal_ratio < tolerance {
            return Err(ForecastError::FrequencyInference(
                "no unique modal spacing found".to_string(),
            ));
        }

        Ok(Duration::seconds(modal_diff))
    }

    /// Infer frequency respecting business day calendar.
    pub fn infer_frequency_calendar(&self, tolerance: f64) -> Result<Duration> {
        if self.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: self.len(),
            });
        }

        // Filter to business days only if calendar is present
        let business_timestamps: Vec<&DateTime<Utc>> = if self.calendar.is_some() {
            self.timestamps
                .iter()
                .filter(|t| self.is_business_day(t))
                .collect()
        } else {
            self.timestamps.iter().collect()
        };

        if business_timestamps.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: business_timestamps.len(),
            });
        }

        // Calculate differences between consecutive business days
        let diffs: Vec<i64> = business_timestamps
            .windows(2)
            .map(|w| (*w[1] - *w[0]).num_seconds())
            .collect();

        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &diff in &diffs {
            *counts.entry(diff).or_insert(0) += 1;
        }

        let (modal_diff, modal_count) = counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&diff, &count)| (diff, count))
            .ok_or(ForecastError::FrequencyInference(
                "empty spacing data".to_string(),
            ))?;

        let total_count: usize = counts.values().sum();
        let modal_ratio = modal_count as f64 / total_count as f64;

        if modal_ratio < tolerance {
            return Err(ForecastError::FrequencyInference(
                "no unique modal spacing found".to_string(),
            ));
        }

        Ok(Duration::seconds(modal_diff))
    }

    /// Set frequency from timestamps (auto-infer).
    pub fn set_frequency_from_timestamps(&mut self) -> Result<()> {
        let freq = self.infer_frequency(0.5)?;
        self.frequency = Some(freq);
        Ok(())
    }
}

/// Linear interpolation for a series with NaN values.
fn interpolate_series(values: &[f64], fill_edges: bool) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let mut result = values.to_vec();
    let n = result.len();

    // Find segments of NaN values and interpolate
    let mut i = 0;
    while i < n {
        if result[i].is_nan() {
            // Find start of NaN segment
            let start = i;
            while i < n && result[i].is_nan() {
                i += 1;
            }
            let end = i;

            // Get boundary values
            let left = if start > 0 {
                Some(result[start - 1])
            } else {
                None
            };
            let right = if end < n { Some(result[end]) } else { None };

            // Interpolate or fill edges
            match (left, right) {
                (Some(l), Some(r)) => {
                    // Linear interpolation
                    // Gap is from left boundary to right boundary: (end - start + 1) segments
                    let segments = (end - start + 1) as f64;
                    for (j, idx) in (start..end).enumerate() {
                        let t = (j + 1) as f64 / segments;
                        result[idx] = l + t * (r - l);
                    }
                }
                (Some(l), None) if fill_edges => {
                    result[start..end].fill(l);
                }
                (None, Some(r)) if fill_edges => {
                    result[start..end].fill(r);
                }
                _ => {
                    // Leave as NaN if can't interpolate
                }
            }
        } else {
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    fn make_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i as u32, 0, 0).unwrap())
            .collect()
    }

    fn make_daily_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1 + i as u32, 0, 0, 0)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn time_series_constructs_univariate_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

        assert_eq!(ts.len(), 5);
        assert!(!ts.is_empty());
        assert_eq!(ts.dimensions(), 1);
        assert!(!ts.is_multivariate());
        assert_eq!(ts.primary_values(), &values);
        assert_eq!(ts.timestamps(), &timestamps);
    }

    #[test]
    fn time_series_sets_labels_and_metadata() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Set labels
        ts.set_labels(vec!["dim1".to_string()]).unwrap();
        assert_eq!(ts.labels(), &["dim1"]);

        // Set metadata
        ts.set_metadata("source".to_string(), "test".to_string());
        assert_eq!(ts.metadata().get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn time_series_sets_frequency() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(ts.frequency().is_none());

        ts.set_frequency(Duration::hours(1));
        assert_eq!(ts.frequency(), Some(Duration::hours(1)));

        ts.clear_frequency();
        assert!(ts.frequency().is_none());
    }

    #[test]
    fn time_series_sets_timezone() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(ts.timezone().is_none());

        ts.set_timezone("America/New_York".to_string());
        assert_eq!(ts.timezone(), Some("America/New_York"));
    }

    #[test]
    fn time_series_handles_multivariate_column_layout() {
        let timestamps = make_timestamps(3);
        let values = vec![
            vec![1.0, 2.0, 3.0], // dimension 0
            vec![4.0, 5.0, 6.0], // dimension 1
        ];

        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Column)
            .build()
            .unwrap();

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.dimensions(), 2);
        assert!(ts.is_multivariate());
        assert_eq!(ts.values(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(ts.values(1).unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(ts.row(0).unwrap(), vec![1.0, 4.0]);
        assert_eq!(ts.row(1).unwrap(), vec![2.0, 5.0]);
    }

    #[test]
    fn time_series_handles_multivariate_row_layout() {
        let timestamps = make_timestamps(3);
        let values = vec![
            vec![1.0, 4.0], // observation 0: [dim0, dim1]
            vec![2.0, 5.0], // observation 1
            vec![3.0, 6.0], // observation 2
        ];

        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Row)
            .build()
            .unwrap();

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.dimensions(), 2);
        assert_eq!(ts.values(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(ts.values(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn time_series_validates_constructor_input() {
        let timestamps = make_timestamps(3);

        // Mismatched value count
        let values = vec![vec![1.0, 2.0]]; // 2 values for 3 timestamps
        let result = TimeSeriesBuilder::new()
            .timestamps(timestamps.clone())
            .multivariate_values(values, ValueLayout::Column)
            .build();
        assert!(result.is_err());

        // Inconsistent row dimensions
        let values = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0], // Different dimension count
            vec![6.0, 7.0],
        ];
        let result = TimeSeriesBuilder::new()
            .timestamps(timestamps.clone())
            .multivariate_values(values, ValueLayout::Row)
            .build();
        assert!(result.is_err());

        // Invalid label count
        let values = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Column)
            .labels(vec!["only_one".to_string()]) // 1 label for 2 dimensions
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn time_series_rejects_non_increasing_timestamps() {
        // Non-monotonic timestamps
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(), // goes backward
        ];
        let values = vec![1.0, 2.0, 3.0];

        let result = TimeSeries::univariate(timestamps, values);
        assert!(matches!(result, Err(ForecastError::TimestampError(_))));

        // Duplicate timestamps
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(), // duplicate
        ];
        let values = vec![1.0, 2.0, 3.0];

        let result = TimeSeries::univariate(timestamps, values);
        assert!(matches!(result, Err(ForecastError::TimestampError(_))));
    }

    #[test]
    fn time_series_stores_metadata_and_timezone_attributes() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Metadata
        ts.set_metadata("key1".to_string(), "value1".to_string());
        ts.set_metadata("key2".to_string(), "value2".to_string());
        assert_eq!(ts.metadata().len(), 2);

        // Dimension metadata
        let dim_meta = vec![{
            let mut m = HashMap::new();
            m.insert("unit".to_string(), "celsius".to_string());
            m
        }];
        ts.set_dimension_metadata(dim_meta).unwrap();
        assert_eq!(
            ts.dimension_metadata()[0].get("unit"),
            Some(&"celsius".to_string())
        );

        // Timezone
        ts.set_timezone("UTC".to_string());
        assert_eq!(ts.timezone(), Some("UTC"));
    }

    #[test]
    fn time_series_slice_preserves_dimensional_metadata() {
        let timestamps = make_timestamps(5);
        let values = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

        let mut ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Column)
            .labels(vec!["temp".to_string()])
            .build()
            .unwrap();

        ts.set_metadata("source".to_string(), "sensor".to_string());
        ts.set_timezone("Europe/London".to_string());
        ts.set_frequency(Duration::hours(1));

        let sliced = ts.slice(1, 4).unwrap();

        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.labels(), &["temp"]);
        assert_eq!(sliced.metadata().get("source"), Some(&"sensor".to_string()));
        assert_eq!(sliced.timezone(), Some("Europe/London"));
        assert_eq!(sliced.frequency(), Some(Duration::hours(1)));
    }

    #[test]
    fn time_series_sanitizes_missing_values() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        assert!(ts.has_missing_values());

        // Drop policy
        let sanitized = ts.sanitized(MissingValuePolicy::Drop).unwrap();
        assert_eq!(sanitized.len(), 3);
        assert_eq!(sanitized.primary_values(), &[1.0, 3.0, 5.0]);

        // Fill policy
        let sanitized = ts.sanitized(MissingValuePolicy::Fill(0.0)).unwrap();
        assert_eq!(sanitized.len(), 5);
        assert_eq!(sanitized.primary_values(), &[1.0, 0.0, 3.0, 0.0, 5.0]);

        // ForwardFill policy
        let sanitized = ts.sanitized(MissingValuePolicy::ForwardFill).unwrap();
        assert_eq!(sanitized.primary_values(), &[1.0, 1.0, 3.0, 3.0, 5.0]);

        // Error policy
        let result = ts.sanitized(MissingValuePolicy::Error);
        assert!(matches!(result, Err(ForecastError::MissingValues)));
    }

    #[test]
    fn time_series_calendar_annotations_manage_holidays_and_regressors() {
        let timestamps = make_daily_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let holidays = vec![timestamps[1]]; // Day 2 is a holiday
        let calendar = CalendarAnnotations::new()
            .with_holidays(holidays)
            .with_regressor("promo".to_string(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let mut ts = TimeSeries::univariate(timestamps.clone(), values).unwrap();
        ts.set_calendar(calendar);

        assert!(ts.is_holiday(&timestamps[1]));
        assert!(!ts.is_holiday(&timestamps[0]));
        assert!(ts.has_regressors());
        assert_eq!(
            ts.regressor("promo"),
            Some([0.0, 1.0, 0.0, 1.0, 0.0].as_slice())
        );
    }

    #[test]
    fn calendar_aware_frequency_inference_skips_weekends() {
        // Create timestamps for a week (Mon-Fri, skipping Sat-Sun)
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // Mon
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(), // Tue
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(), // Wed
            Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(), // Thu
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(), // Fri
            Utc.with_ymd_and_hms(2024, 1, 8, 0, 0, 0).unwrap(), // Mon (skip weekend)
            Utc.with_ymd_and_hms(2024, 1, 9, 0, 0, 0).unwrap(), // Tue
        ];
        let values: Vec<f64> = (0..7).map(|i| i as f64).collect();

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_calendar(CalendarAnnotations::new());

        let freq = ts.infer_frequency_calendar(0.5).unwrap();
        assert_eq!(freq, Duration::days(1));
    }

    #[test]
    fn time_series_linear_interpolation_fills_gaps() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let interpolated = ts.interpolated(true);

        let result = interpolated.primary_values();
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn time_series_interpolation_fills_edges() {
        let timestamps = make_timestamps(5);
        let values = vec![f64::NAN, f64::NAN, 3.0, 4.0, f64::NAN];

        let ts = TimeSeries::univariate(timestamps.clone(), values).unwrap();

        // With edge filling
        let interpolated = ts.interpolated(true);
        let result = interpolated.primary_values();
        assert_relative_eq!(result[0], 3.0, epsilon = 1e-10); // Filled with first valid
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10); // Filled with last valid

        // Without edge filling
        let interpolated = ts.interpolated(false);
        let result = interpolated.primary_values();
        assert!(result[0].is_nan()); // Not filled
        assert!(result[4].is_nan()); // Not filled
    }

    #[test]
    fn time_series_infers_regular_frequency() {
        // Hourly data
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let freq = ts.infer_frequency(0.5).unwrap();

        assert_eq!(freq, Duration::hours(1));
    }

    #[test]
    fn time_series_frequency_inference_requires_unique_modal_spacing() {
        // Irregular timestamps with no clear pattern
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(), // 1 hour
            Utc.with_ymd_and_hms(2024, 1, 1, 3, 0, 0).unwrap(), // 2 hours
            Utc.with_ymd_and_hms(2024, 1, 1, 6, 0, 0).unwrap(), // 3 hours
            Utc.with_ymd_and_hms(2024, 1, 1, 10, 0, 0).unwrap(), // 4 hours
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.infer_frequency(0.8); // High tolerance

        assert!(matches!(result, Err(ForecastError::FrequencyInference(_))));
    }

    #[test]
    fn time_series_detects_partial_day_holiday_occurrences() {
        // Create timestamps within a single day
        let base_date = Utc.with_ymd_and_hms(2024, 12, 25, 0, 0, 0).unwrap(); // Christmas
        let timestamps: Vec<DateTime<Utc>> =
            (0..24).map(|h| base_date + Duration::hours(h)).collect();
        let values: Vec<f64> = (0..24).map(|i| i as f64).collect();

        let calendar = CalendarAnnotations::new().with_holidays(vec![base_date]);

        let mut ts = TimeSeries::univariate(timestamps.clone(), values).unwrap();
        ts.set_calendar(calendar);

        // All timestamps on Christmas day should be holidays
        for t in &timestamps {
            assert!(ts.is_holiday(t), "Expected {} to be a holiday", t);
        }

        // Non-Christmas day should not be a holiday
        let non_holiday = Utc.with_ymd_and_hms(2024, 12, 26, 12, 0, 0).unwrap();
        assert!(!ts.is_holiday(&non_holiday));
    }

    #[test]
    fn time_series_row_access_out_of_bounds() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(ts.row(0).is_ok());
        assert!(ts.row(2).is_ok());
        assert!(matches!(
            ts.row(3),
            Err(ForecastError::IndexOutOfBounds { index: 3, size: 3 })
        ));
    }

    #[test]
    fn time_series_dimension_access_out_of_bounds() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(ts.values(0).is_ok());
        assert!(matches!(
            ts.values(1),
            Err(ForecastError::IndexOutOfBounds { index: 1, size: 1 })
        ));
    }
}
