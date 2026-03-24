//! TimeSeries data structure for representing temporal data.

use crate::error::{ForecastError, Result};
use crate::utils::stats::{nan_mean, nan_median};
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::HashMap;
use std::fmt;

/// Layout of multivariate data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueLayout {
    /// Each inner vector is a dimension (column-major).
    #[default]
    Column,
    /// Each inner vector is an observation across dimensions (row-major).
    Row,
}

/// Frequency specification for gap filling.
///
/// Supports Polars-style frequency strings like "30m", "1h", "1d", "1w", "1mo", "1q", "1y".
#[derive(Debug, Clone, PartialEq)]
pub enum Frequency {
    /// Duration-based frequency (seconds, minutes, hours, days, weeks).
    Duration(Duration),
    /// Month-based frequency (1mo, 2mo, etc.).
    Months(i32),
    /// Year-based frequency (1y, 2y, etc.).
    Years(i32),
}

impl Frequency {
    /// Parse a Polars-style frequency string.
    ///
    /// Supported formats:
    /// - "30s" or "30sec" - 30 seconds
    /// - "30m" or "30min" - 30 minutes
    /// - "1h" - 1 hour
    /// - "1d" - 1 day
    /// - "1w" - 1 week
    /// - "1mo" - 1 month
    /// - "1q" - 1 quarter (3 months)
    /// - "1y" - 1 year
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();

        // Find where the number ends and the unit begins
        let num_end = s
            .chars()
            .position(|c| !c.is_ascii_digit())
            .unwrap_or(s.len());

        if num_end == 0 {
            return Err(ForecastError::InvalidParameter(format!(
                "invalid frequency string: '{}' (no number found)",
                s
            )));
        }

        let num: i64 = s[..num_end].parse().map_err(|_| {
            ForecastError::InvalidParameter(format!(
                "invalid frequency string: '{}' (invalid number)",
                s
            ))
        })?;

        let unit = &s[num_end..];

        match unit {
            "s" | "sec" | "second" | "seconds" => Ok(Frequency::Duration(Duration::seconds(num))),
            "m" | "min" | "minute" | "minutes" => Ok(Frequency::Duration(Duration::minutes(num))),
            "h" | "hr" | "hour" | "hours" => Ok(Frequency::Duration(Duration::hours(num))),
            "d" | "day" | "days" => Ok(Frequency::Duration(Duration::days(num))),
            "w" | "week" | "weeks" => Ok(Frequency::Duration(Duration::weeks(num))),
            "mo" | "month" | "months" => Ok(Frequency::Months(num as i32)),
            "q" | "quarter" | "quarters" => Ok(Frequency::Months(num as i32 * 3)),
            "y" | "year" | "years" => Ok(Frequency::Years(num as i32)),
            _ => Err(ForecastError::InvalidParameter(format!(
                "unknown frequency unit: '{}' (expected s, m, h, d, w, mo, q, or y)",
                unit
            ))),
        }
    }

    /// Create a frequency from a chrono::Duration.
    pub fn from_duration(duration: Duration) -> Self {
        Frequency::Duration(duration)
    }

    /// Create a frequency from an integer step size.
    /// This is useful for integer-indexed time series.
    pub fn from_step(step: i64) -> Self {
        Frequency::Duration(Duration::seconds(step))
    }
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
    /// Backward fill (use next valid value).
    BackwardFill,
    /// Fill with mean of finite values.
    FillMean,
    /// Fill with median of finite values.
    FillMedian,
    /// Linear interpolation (edges filled with nearest valid value).
    Interpolate,
    /// Return error if missing values found.
    Error,
}

/// Method for aggregating groups of observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Sum all values in the group.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Median value.
    Median,
    /// First value in the group.
    First,
    /// Last value in the group.
    Last,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
}

/// Method for interpolating values during upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation between points.
    Linear,
    /// Forward fill (carry previous value).
    ForwardFill,
    /// Backward fill (use next value).
    BackwardFill,
    /// Fill with zero.
    Zero,
}

/// Calendar annotations for holidays and regressors.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeSeries {
    timestamps: Vec<DateTime<Utc>>,
    /// Values stored in column-major format: values[dimension][observation]
    values: Vec<Vec<f64>>,
    labels: Vec<String>,
    metadata: HashMap<String, String>,
    dimension_metadata: Vec<HashMap<String, String>>,
    timezone: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::utils::persistence::opt_duration_secs")
    )]
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

    /// Get all regressors as a HashMap (clone).
    pub fn all_regressors(&self) -> HashMap<String, Vec<f64>> {
        self.calendar
            .as_ref()
            .map(|c| c.regressors().clone())
            .unwrap_or_default()
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

        // Slice calendar regressors to match the sliced range
        let calendar = self.calendar.as_ref().map(|cal| {
            let mut sliced_cal = CalendarAnnotations::new().with_holidays(cal.holidays().to_vec());
            for (name, vals) in cal.regressors() {
                let sliced_vals = if vals.len() >= end {
                    vals[start..end].to_vec()
                } else if vals.len() > start {
                    vals[start..].to_vec()
                } else {
                    Vec::new()
                };
                sliced_cal = sliced_cal.with_regressor(name.clone(), sliced_vals);
            }
            sliced_cal
        });

        Ok(TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency,
            calendar,
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
            MissingValuePolicy::BackwardFill => {
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| {
                        let mut result = dim.clone();
                        let mut next_valid = None;
                        for i in (0..result.len()).rev() {
                            if result[i].is_nan() || result[i].is_infinite() {
                                if let Some(v) = next_valid {
                                    result[i] = v;
                                }
                            } else {
                                next_valid = Some(result[i]);
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
            MissingValuePolicy::FillMean => {
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| {
                        let m = nan_mean(dim);
                        dim.iter()
                            .map(|&v| if v.is_nan() || v.is_infinite() { m } else { v })
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
            MissingValuePolicy::FillMedian => {
                let values: Vec<Vec<f64>> = self
                    .values
                    .iter()
                    .map(|dim| {
                        let med = nan_median(dim);
                        dim.iter()
                            .map(|&v| {
                                if v.is_nan() || v.is_infinite() {
                                    med
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
            MissingValuePolicy::Interpolate => Ok(self.interpolated(true)),
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

    /// Returns a boolean mask: true where value is NaN or Inf (primary dimension).
    pub fn missing_mask(&self) -> Vec<bool> {
        match self.values.first() {
            Some(dim) => dim.iter().map(|v| v.is_nan() || v.is_infinite()).collect(),
            None => vec![],
        }
    }

    /// Count of missing values per dimension.
    pub fn missing_count(&self) -> Vec<usize> {
        self.values
            .iter()
            .map(|dim| dim.iter().filter(|v| v.is_nan() || v.is_infinite()).count())
            .collect()
    }

    /// Forward-fill then backward-fill — handles both leading and trailing NaNs.
    pub fn imputed_forward_backward(&self) -> TimeSeries {
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                // Forward fill
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
                // Backward fill remaining (leading NaNs)
                let mut next_valid = None;
                for i in (0..result.len()).rev() {
                    if result[i].is_nan() || result[i].is_infinite() {
                        if let Some(v) = next_valid {
                            result[i] = v;
                        }
                    } else {
                        next_valid = Some(result[i]);
                    }
                }
                result
            })
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

    /// Impute NaN using mean of valid values in a centered window.
    ///
    /// Window must be odd. Multi-pass (up to 3) for adjacent NaNs.
    /// Remaining NaNs filled with global mean.
    pub fn imputed_moving_average(&self, window: usize) -> Result<TimeSeries> {
        if window == 0 || window % 2 == 0 {
            return Err(ForecastError::InvalidParameter(
                "moving average window must be odd and > 0".to_string(),
            ));
        }

        let half = window / 2;
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                let mut result = dim.clone();
                let n = result.len();

                // Multi-pass: up to 3 passes to handle adjacent NaNs
                for _ in 0..3 {
                    let mut changed = false;
                    let snapshot = result.clone();
                    for i in 0..n {
                        if !(snapshot[i].is_nan() || snapshot[i].is_infinite()) {
                            continue;
                        }
                        let start = i.saturating_sub(half);
                        let end = (i + half + 1).min(n);
                        let mut sum = 0.0;
                        let mut count = 0usize;
                        for j in start..end {
                            if j != i && snapshot[j].is_finite() {
                                sum += snapshot[j];
                                count += 1;
                            }
                        }
                        if count > 0 {
                            result[i] = sum / count as f64;
                            changed = true;
                        }
                    }
                    if !changed {
                        break;
                    }
                }

                // Fill any remaining NaNs with global mean
                let global_mean = nan_mean(&result);
                for v in &mut result {
                    if v.is_nan() || v.is_infinite() {
                        *v = global_mean;
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

    /// Impute NaN using the median of observed values at the same seasonal position.
    ///
    /// Groups values by (index % period), computes median per group, fills NaN.
    /// Returns error if period is 0 or if >50% of values in any seasonal bucket are NaN.
    pub fn imputed_seasonal(&self, period: usize) -> Result<TimeSeries> {
        if period == 0 {
            return Err(ForecastError::InvalidParameter(
                "seasonal period must be > 0".to_string(),
            ));
        }
        if self.len() < period {
            return Err(ForecastError::InsufficientData {
                needed: period,
                got: self.len(),
                hint: None,
            });
        }

        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                let n = dim.len();

                // Collect finite values per seasonal bucket
                let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); period];
                for (i, &v) in dim.iter().enumerate() {
                    if v.is_finite() {
                        buckets[i % period].push(v);
                    }
                }

                // Compute median per bucket
                let medians: Vec<f64> = buckets
                    .iter()
                    .map(|b| {
                        if b.is_empty() {
                            f64::NAN
                        } else {
                            let mut sorted = b.clone();
                            sorted.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let len = sorted.len();
                            if len % 2 == 0 {
                                (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
                            } else {
                                sorted[len / 2]
                            }
                        }
                    })
                    .collect();

                // Fill NaN with seasonal median
                let mut result = dim.clone();
                for i in 0..n {
                    if result[i].is_nan() || result[i].is_infinite() {
                        result[i] = medians[i % period];
                    }
                }
                result
            })
            .collect();

        // Validate: check that no seasonal bucket had >50% NaN
        for (d, dim) in self.values.iter().enumerate() {
            let mut bucket_total: Vec<usize> = vec![0; period];
            let mut bucket_missing: Vec<usize> = vec![0; period];
            for (i, &v) in dim.iter().enumerate() {
                bucket_total[i % period] += 1;
                if v.is_nan() || v.is_infinite() {
                    bucket_missing[i % period] += 1;
                }
            }
            for (b, (&total, &missing)) in
                bucket_total.iter().zip(bucket_missing.iter()).enumerate()
            {
                if total > 0 && missing as f64 / total as f64 > 0.5 {
                    return Err(ForecastError::InvalidParameter(format!(
                        "dimension {} seasonal bucket {} has >50% missing values ({}/{})",
                        d, b, missing, total
                    )));
                }
            }
        }

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

    /// Impute NaN values in all regressors using the given policy.
    ///
    /// Applies the policy to each regressor vector independently.
    /// Only `Fill`, `ForwardFill`, `BackwardFill`, `FillMean`, `FillMedian`,
    /// and `Interpolate` are supported. `Drop` and `Error` return an error.
    pub fn with_imputed_regressors(&self, policy: MissingValuePolicy) -> Result<TimeSeries> {
        match policy {
            MissingValuePolicy::Drop | MissingValuePolicy::Error => {
                return Err(ForecastError::InvalidParameter(
                    "Drop and Error policies are not supported for regressor imputation"
                        .to_string(),
                ));
            }
            _ => {}
        }

        let mut result = self.clone();
        if let Some(ref mut cal) = result.calendar {
            let mut imputed_regressors = HashMap::new();
            for (name, values) in cal.regressors() {
                let imputed = match policy {
                    MissingValuePolicy::Fill(fill_value) => values
                        .iter()
                        .map(|&v| {
                            if v.is_nan() || v.is_infinite() {
                                fill_value
                            } else {
                                v
                            }
                        })
                        .collect(),
                    MissingValuePolicy::ForwardFill => {
                        let mut res = Vec::with_capacity(values.len());
                        let mut last_valid = None;
                        for &v in values {
                            if v.is_nan() || v.is_infinite() {
                                res.push(last_valid.unwrap_or(v));
                            } else {
                                last_valid = Some(v);
                                res.push(v);
                            }
                        }
                        res
                    }
                    MissingValuePolicy::BackwardFill => {
                        let mut res = values.to_vec();
                        let mut next_valid = None;
                        for i in (0..res.len()).rev() {
                            if res[i].is_nan() || res[i].is_infinite() {
                                if let Some(v) = next_valid {
                                    res[i] = v;
                                }
                            } else {
                                next_valid = Some(res[i]);
                            }
                        }
                        res
                    }
                    MissingValuePolicy::FillMean => {
                        let m = nan_mean(values);
                        values
                            .iter()
                            .map(|&v| if v.is_nan() || v.is_infinite() { m } else { v })
                            .collect()
                    }
                    MissingValuePolicy::FillMedian => {
                        let med = nan_median(values);
                        values
                            .iter()
                            .map(|&v| {
                                if v.is_nan() || v.is_infinite() {
                                    med
                                } else {
                                    v
                                }
                            })
                            .collect()
                    }
                    MissingValuePolicy::Interpolate => interpolate_series(values, true),
                    MissingValuePolicy::Drop | MissingValuePolicy::Error => {
                        unreachable!()
                    }
                };
                imputed_regressors.insert(name.clone(), imputed);
            }
            // Replace regressors in calendar
            let mut new_cal = CalendarAnnotations::new().with_holidays(cal.holidays().to_vec());
            for (name, values) in imputed_regressors {
                new_cal = new_cal.with_regressor(name, values);
            }
            result.calendar = Some(new_cal);
        }
        Ok(result)
    }

    /// Infer frequency from timestamps.
    pub fn infer_frequency(&self, tolerance: f64) -> Result<Duration> {
        if self.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: self.len(),
                hint: None,
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
                hint: None,
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
                hint: None,
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

    /// Generate future timestamps for a forecast horizon.
    ///
    /// Infers the frequency from existing timestamps, then extrapolates forward.
    /// Uses calendar-aware arithmetic for monthly/quarterly data.
    pub fn future_timestamps(&self, horizon: usize) -> Result<Vec<DateTime<Utc>>> {
        if self.timestamps.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        let last = *self.timestamps.last().unwrap();

        // Try calendar-aware frequency first, fall back to duration-based
        let freq = self
            .infer_frequency_calendar(0.5)
            .or_else(|_| self.infer_frequency(0.5))?;

        Ok(generate_future_timestamps(
            &last,
            &Frequency::Duration(freq),
            horizon,
        ))
    }

    /// Set frequency from timestamps (auto-infer).
    pub fn set_frequency_from_timestamps(&mut self) -> Result<()> {
        let freq = self.infer_frequency(0.5)?;
        self.frequency = Some(freq);
        Ok(())
    }

    /// Fill missing timestamps in the time series with NULL (NaN) values.
    ///
    /// This method generates a complete sequence of timestamps based on the specified
    /// frequency, and fills in missing timestamps with NaN values. This is useful for
    /// ensuring a time series has regular intervals before analysis or forecasting.
    ///
    /// # Arguments
    ///
    /// * `frequency` - The frequency to use for gap filling. Can be:
    ///   - Polars-style string: "30m", "1h", "1d", "1w", "1mo", "1q", "1y"
    ///   - Duration: `Frequency::Duration(Duration::hours(1))`
    ///   - Months: `Frequency::Months(1)` for monthly
    ///   - Years: `Frequency::Years(1)` for yearly
    ///
    /// # Returns
    ///
    /// A new `TimeSeries` with all gaps filled with NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// use anofox_forecast::core::{TimeSeries, Frequency};
    /// use chrono::{TimeZone, Utc};
    ///
    /// // Create a time series with gaps
    /// let timestamps = vec![
    ///     Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
    ///     Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
    ///     // Gap at 2:00
    ///     Utc.with_ymd_and_hms(2024, 1, 1, 3, 0, 0).unwrap(),
    /// ];
    /// let values = vec![1.0, 2.0, 4.0];
    ///
    /// let ts = TimeSeries::univariate(timestamps, values).unwrap();
    /// let filled = ts.fill_gaps(Frequency::parse("1h").unwrap()).unwrap();
    ///
    /// assert_eq!(filled.len(), 4); // Now includes 2:00
    /// ```
    pub fn fill_gaps(&self, frequency: Frequency) -> Result<TimeSeries> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        if self.len() == 1 {
            return Ok(self.clone());
        }

        let start = self.timestamps[0];
        // SAFETY: `self.is_empty()` and `self.len() == 1` are handled above, so timestamps has >= 2 elements.
        let end = *self.timestamps.last().unwrap();

        // Generate the expected timestamps based on frequency
        let expected_timestamps = generate_timestamps(start, end, &frequency)?;

        if expected_timestamps.is_empty() {
            return Ok(self.clone());
        }

        // Build a map from existing timestamps to their indices
        let existing: HashMap<DateTime<Utc>, usize> = self
            .timestamps
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, i))
            .collect();

        // Create new timestamps and values
        let mut new_timestamps = Vec::with_capacity(expected_timestamps.len());
        let mut new_values: Vec<Vec<f64>> = (0..self.dimensions())
            .map(|_| Vec::with_capacity(expected_timestamps.len()))
            .collect();

        for ts in expected_timestamps {
            new_timestamps.push(ts);
            if let Some(&idx) = existing.get(&ts) {
                // Use existing value
                for (dim, dim_values) in new_values.iter_mut().enumerate() {
                    dim_values.push(self.values[dim][idx]);
                }
            } else {
                // Fill with NaN
                for dim_values in &mut new_values {
                    dim_values.push(f64::NAN);
                }
            }
        }

        Ok(TimeSeries {
            timestamps: new_timestamps,
            values: new_values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: match &frequency {
                Frequency::Duration(d) => Some(*d),
                _ => self.frequency,
            },
            calendar: self.calendar.clone(),
        })
    }

    /// Fill gaps using a Polars-style frequency string.
    ///
    /// This is a convenience method that parses the frequency string and calls `fill_gaps`.
    ///
    /// # Arguments
    ///
    /// * `frequency` - A Polars-style frequency string like "30m", "1h", "1d", etc.
    ///
    /// # Examples
    ///
    /// ```
    /// use anofox_forecast::core::TimeSeries;
    /// use chrono::{TimeZone, Utc};
    ///
    /// let timestamps = vec![
    ///     Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
    ///     Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(), // Gap at 1:00
    /// ];
    /// let values = vec![1.0, 3.0];
    ///
    /// let ts = TimeSeries::univariate(timestamps, values).unwrap();
    /// let filled = ts.fill_gaps_str("1h").unwrap();
    ///
    /// assert_eq!(filled.len(), 3);
    /// ```
    pub fn fill_gaps_str(&self, frequency: &str) -> Result<TimeSeries> {
        let freq = Frequency::parse(frequency)?;
        self.fill_gaps(freq)
    }

    /// Compute the seasonal strength of the primary dimension via STL decomposition.
    ///
    /// Returns a value between 0 and 1, where values close to 1 indicate
    /// strong seasonality. Requires `period >= 2` and series length `>= 2 * period`.
    ///
    /// This is a convenience wrapper around STL decomposition — if you need both
    /// seasonal and trend strength, call [`STL::decompose`](crate::seasonality::STL)
    /// once and use `STLResult::seasonal_strength()` / `STLResult::trend_strength()`.
    pub fn seasonal_strength(&self, period: usize) -> Result<f64> {
        use crate::seasonality::STL;

        if period < 2 {
            return Err(ForecastError::InvalidParameter(
                "period must be at least 2".into(),
            ));
        }
        let vals = self.primary_values();
        if vals.len() < 2 * period {
            return Err(ForecastError::InsufficientData {
                needed: 2 * period,
                got: vals.len(),
                hint: Some("need at least 2 full seasonal cycles".into()),
            });
        }
        let stl = STL::new(period);
        let result = stl.decompose(vals).ok_or(ForecastError::ComputationError(
            "STL decomposition failed".into(),
        ))?;
        Ok(result.seasonal_strength())
    }

    /// Compute the trend strength of the primary dimension via STL decomposition.
    ///
    /// Returns a value between 0 and 1, where values close to 1 indicate
    /// a strong trend component. Requires `period >= 2` and series length `>= 2 * period`.
    pub fn trend_strength(&self, period: usize) -> Result<f64> {
        use crate::seasonality::STL;

        if period < 2 {
            return Err(ForecastError::InvalidParameter(
                "period must be at least 2".into(),
            ));
        }
        let vals = self.primary_values();
        if vals.len() < 2 * period {
            return Err(ForecastError::InsufficientData {
                needed: 2 * period,
                got: vals.len(),
                hint: Some("need at least 2 full seasonal cycles".into()),
            });
        }
        let stl = STL::new(period);
        let result = stl.decompose(vals).ok_or(ForecastError::ComputationError(
            "STL decomposition failed".into(),
        ))?;
        Ok(result.trend_strength())
    }

    /// Detect outliers in the primary dimension and return a sanitized copy.
    ///
    /// Outlier values are replaced with the local median within a window of
    /// `window_size` around each outlier. The original series is unchanged.
    ///
    /// Uses the specified [`OutlierConfig`](crate::detection::OutlierConfig)
    /// for detection (IQR, Z-score, or Modified Z-score).
    ///
    /// # Example
    ///
    /// ```
    /// use anofox_forecast::core::TimeSeries;
    /// use anofox_forecast::detection::OutlierConfig;
    /// use chrono::{TimeZone, Utc};
    ///
    /// let timestamps: Vec<_> = (0..50)
    ///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
    ///         + chrono::Duration::hours(i))
    ///     .collect();
    /// let mut values: Vec<f64> = (0..50).map(|i| 10.0 + 0.1 * i as f64).collect();
    /// values[25] = 1000.0; // inject outlier
    ///
    /// let ts = TimeSeries::univariate(timestamps, values).unwrap();
    /// let clean = ts.with_outliers_replaced(&OutlierConfig::default(), 5).unwrap();
    ///
    /// // The outlier at index 25 has been replaced
    /// assert!((clean.primary_values()[25] - 1000.0).abs() > 1.0);
    /// ```
    pub fn with_outliers_replaced(
        &self,
        config: &crate::detection::OutlierConfig,
        window_size: usize,
    ) -> Result<TimeSeries> {
        let vals = self.primary_values();
        let outlier_result = crate::detection::detect_outliers(vals, config);

        if outlier_result.outlier_indices.is_empty() {
            return Ok(self.clone());
        }

        let n = vals.len();
        let mut cleaned = vals.to_vec();
        let half = window_size / 2;

        for &idx in &outlier_result.outlier_indices {
            // Collect non-outlier neighbors within the window
            let start = idx.saturating_sub(half);
            let end = (idx + half + 1).min(n);

            let mut neighbors: Vec<f64> = (start..end)
                .filter(|&i| i != idx && !outlier_result.outlier_indices.contains(&i))
                .map(|i| cleaned[i])
                .filter(|v| v.is_finite())
                .collect();

            if neighbors.is_empty() {
                // Fallback: use overall median
                let mut all: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
                all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if !all.is_empty() {
                    cleaned[idx] = all[all.len() / 2];
                }
            } else {
                neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                cleaned[idx] = neighbors[neighbors.len() / 2];
            }
        }

        // Rebuild TimeSeries with cleaned values
        let mut new_values = self.values.clone();
        new_values[0] = cleaned;

        Ok(TimeSeries {
            timestamps: self.timestamps.clone(),
            values: new_values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency,
            calendar: self.calendar.clone(),
        })
    }

    // --- Temporal aggregation / resampling ---

    /// Aggregate observations into groups of `period` consecutive points.
    pub fn aggregate(&self, period: usize, method: AggregationMethod) -> TimeSeries {
        if period <= 1 || self.is_empty() {
            return self.clone();
        }
        let n = self.len();
        let num_groups = n.div_ceil(period);
        let timestamps: Vec<DateTime<Utc>> = (0..num_groups)
            .map(|g| self.timestamps[g * period])
            .collect();
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                (0..num_groups)
                    .map(|g| {
                        let start = g * period;
                        let end = (start + period).min(n);
                        aggregate_slice(&dim[start..end], method)
                    })
                    .collect()
            })
            .collect();
        TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency.map(|f| f * period as i32),
            calendar: None,
        }
    }

    /// Downsample by taking every `factor`-th observation (decimation).
    pub fn downsample(&self, factor: usize) -> TimeSeries {
        if factor <= 1 || self.is_empty() {
            return self.clone();
        }
        let indices: Vec<usize> = (0..self.len()).step_by(factor).collect();
        let timestamps: Vec<DateTime<Utc>> = indices.iter().map(|&i| self.timestamps[i]).collect();
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| indices.iter().map(|&i| dim[i]).collect())
            .collect();
        TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency.map(|f| f * factor as i32),
            calendar: None,
        }
    }

    /// Upsample by inserting `factor - 1` points between each pair of observations.
    pub fn upsample(&self, factor: usize, method: InterpolationMethod) -> TimeSeries {
        if factor <= 1 || self.len() < 2 {
            return self.clone();
        }
        let n = self.len();
        let new_len = (n - 1) * factor + 1;
        let timestamps: Vec<DateTime<Utc>> = (0..new_len)
            .map(|i| {
                let src_idx = i / factor;
                let frac = i % factor;
                if frac == 0 || src_idx >= n - 1 {
                    self.timestamps[src_idx.min(n - 1)]
                } else {
                    self.timestamps[src_idx]
                        + (self.timestamps[src_idx + 1] - self.timestamps[src_idx]) * frac as i32
                            / factor as i32
                }
            })
            .collect();
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                (0..new_len)
                    .map(|i| {
                        let src_idx = i / factor;
                        let frac = i % factor;
                        if frac == 0 {
                            dim[src_idx]
                        } else {
                            let v0 = dim[src_idx];
                            let v1 = dim[(src_idx + 1).min(n - 1)];
                            match method {
                                InterpolationMethod::Linear => {
                                    v0 + (frac as f64 / factor as f64) * (v1 - v0)
                                }
                                InterpolationMethod::ForwardFill => v0,
                                InterpolationMethod::BackwardFill => v1,
                                InterpolationMethod::Zero => 0.0,
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency.map(|f| {
                let s = f.num_seconds() / factor as i64;
                Duration::seconds(s.max(1))
            }),
            calendar: None,
        }
    }

    /// Sliding window aggregation with configurable step size.
    pub fn sliding_window_aggregate(
        &self,
        window: usize,
        step: usize,
        method: AggregationMethod,
    ) -> TimeSeries {
        let step = if step == 0 { 1 } else { step };
        if window == 0 || window > self.len() || self.is_empty() {
            return TimeSeries {
                timestamps: vec![],
                values: self.values.iter().map(|_| vec![]).collect(),
                labels: self.labels.clone(),
                metadata: self.metadata.clone(),
                dimension_metadata: self.dimension_metadata.clone(),
                timezone: self.timezone.clone(),
                frequency: self.frequency,
                calendar: None,
            };
        }
        let n = self.len();
        let starts: Vec<usize> = (0..n)
            .step_by(step)
            .take_while(|&s| s + window <= n)
            .collect();
        let timestamps: Vec<DateTime<Utc>> = starts.iter().map(|&s| self.timestamps[s]).collect();
        let values: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|dim| {
                starts
                    .iter()
                    .map(|&s| aggregate_slice(&dim[s..s + window], method))
                    .collect()
            })
            .collect();
        TimeSeries {
            timestamps,
            values,
            labels: self.labels.clone(),
            metadata: self.metadata.clone(),
            dimension_metadata: self.dimension_metadata.clone(),
            timezone: self.timezone.clone(),
            frequency: self.frequency.map(|f| f * step as i32),
            calendar: None,
        }
    }
}

/// Apply an aggregation method to a slice of values.
fn aggregate_slice(values: &[f64], method: AggregationMethod) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    match method {
        AggregationMethod::Sum => values.iter().sum(),
        AggregationMethod::Mean => nan_mean(values),
        AggregationMethod::Median => nan_median(values),
        AggregationMethod::First => values[0],
        AggregationMethod::Last => values[values.len() - 1],
        AggregationMethod::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggregationMethod::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Generate a sequence of timestamps from start to end (inclusive) with the given frequency.
fn generate_timestamps(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    frequency: &Frequency,
) -> Result<Vec<DateTime<Utc>>> {
    validate_frequency_positive(frequency)?;

    let mut timestamps = Vec::new();
    let mut current = start;
    while current <= end {
        timestamps.push(current);
        current = advance_timestamp(current, frequency);
    }

    Ok(timestamps)
}

/// Validate that a frequency value is positive.
#[inline]
fn validate_frequency_positive(frequency: &Frequency) -> Result<()> {
    let valid = match frequency {
        Frequency::Duration(d) => d.num_seconds() > 0,
        Frequency::Months(m) => *m > 0,
        Frequency::Years(y) => *y > 0,
    };
    if valid {
        Ok(())
    } else {
        Err(ForecastError::InvalidParameter(
            "frequency must be positive".to_string(),
        ))
    }
}

/// Generate future timestamps by advancing from a starting point.
///
/// Uses calendar-aware arithmetic for monthly/quarterly/yearly frequencies
/// (e.g., Jan 31 + 1 month = Feb 28/29, not Mar 02).
///
/// # Arguments
/// * `last` - The last known timestamp
/// * `frequency` - The step frequency
/// * `horizon` - Number of future timestamps to generate
///
/// # Example
/// ```
/// use anofox_forecast::core::time_series::{generate_future_timestamps, Frequency};
/// use chrono::{Datelike, TimeZone, Utc};
///
/// let last = Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap();
/// let future = generate_future_timestamps(&last, &Frequency::Months(1), 3);
///
/// assert_eq!(future[0].day(), 29); // Feb 29 (2024 is leap year)
/// assert_eq!(future[1].month(), 3); // Mar 31
/// assert_eq!(future[2].month(), 4); // Apr 30
/// ```
pub fn generate_future_timestamps(
    last: &DateTime<Utc>,
    frequency: &Frequency,
    horizon: usize,
) -> Vec<DateTime<Utc>> {
    let mut result = Vec::with_capacity(horizon);
    let mut current = *last;
    for _ in 0..horizon {
        current = advance_timestamp(current, frequency);
        result.push(current);
    }
    result
}

/// Advance a timestamp by one frequency step.
#[inline]
fn advance_timestamp(current: DateTime<Utc>, frequency: &Frequency) -> DateTime<Utc> {
    match frequency {
        Frequency::Duration(duration) => current + *duration,
        Frequency::Months(months) => add_months(current, *months),
        Frequency::Years(years) => add_months(current, *years * 12),
    }
}

/// Add months to a DateTime, handling month-end edge cases.
fn add_months(dt: DateTime<Utc>, months: i32) -> DateTime<Utc> {
    use chrono::{NaiveDate, Timelike};

    let year = dt.year();
    let month = dt.month() as i32;
    let day = dt.day();

    let total_months = year * 12 + (month - 1) + months;
    let new_year = total_months / 12;
    let new_month = (total_months % 12 + 1) as u32;

    // Handle month-end edge cases (e.g., Jan 31 + 1 month = Feb 28/29)
    let max_day = days_in_month(new_year, new_month);
    let new_day = day.min(max_day);

    // Build the new date directly using NaiveDate to avoid issues with chrono's with_* methods
    if let Some(naive_date) = NaiveDate::from_ymd_opt(new_year, new_month, new_day) {
        naive_date
            .and_hms_opt(dt.hour(), dt.minute(), dt.second())
            .map(|naive_dt| DateTime::from_naive_utc_and_offset(naive_dt, Utc))
            .unwrap_or(dt)
    } else {
        // Fallback: just add 30 days per month as approximation
        dt + Duration::days(30 * months as i64)
    }
}

/// Get the number of days in a given month.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 30, // Should never happen
    }
}

/// Check if a year is a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Public crate-internal accessor for `days_in_month`.
pub(crate) fn days_in_month_pub(year: i32, month: u32) -> u32 {
    days_in_month(year, month)
}

/// Public crate-internal accessor for `is_leap_year`.
pub(crate) fn is_leap_year_pub(year: i32) -> bool {
    is_leap_year(year)
}

/// Linear interpolation for a series with NaN values.
fn interpolate_series(values: &[f64], fill_edges: bool) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let mut result = values.to_vec();
    let n = result.len();

    // Find and fill NaN segments
    let mut i = 0;
    while i < n {
        if result[i].is_nan() {
            let start = i;
            while i < n && result[i].is_nan() {
                i += 1;
            }
            let left = if start > 0 {
                Some(result[start - 1])
            } else {
                None
            };
            let right = if i < n { Some(result[i]) } else { None };
            fill_nan_segment(&mut result[start..i], left, right, fill_edges);
        } else {
            i += 1;
        }
    }

    result
}

/// Fill a NaN segment using linear interpolation or edge values.
fn fill_nan_segment(segment: &mut [f64], left: Option<f64>, right: Option<f64>, fill_edges: bool) {
    match (left, right) {
        (Some(l), Some(r)) => {
            let segments = (segment.len() + 1) as f64;
            for (j, val) in segment.iter_mut().enumerate() {
                let t = (j + 1) as f64 / segments;
                *val = l + t * (r - l);
            }
        }
        (Some(l), None) if fill_edges => segment.fill(l),
        (None, Some(r)) if fill_edges => segment.fill(r),
        _ => {} // Leave as NaN
    }
}

#[cfg(feature = "serde")]
impl TimeSeries {
    /// Serialize this time series to a JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ForecastError::SerializationError(format!("serialization failed: {}", e)))
    }

    /// Deserialize a time series from a JSON string.
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            ForecastError::SerializationError(format!("deserialization failed: {}", e))
        })
    }
}

impl fmt::Display for TimeSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.len();
        let dims = self.dimensions();

        write!(f, "TimeSeries(len={}, dims={}", len, dims)?;

        if let Some(freq) = self.frequency {
            write!(f, ", freq={}s", freq.num_seconds())?;
        }

        if len > 0 {
            // SAFETY: `len > 0` guarantees timestamps is non-empty.
            let first = self.timestamps.first().unwrap();
            let last = self.timestamps.last().unwrap();
            write!(
                f,
                ", range=[{} .. {}]",
                first.format("%Y-%m-%d %H:%M:%S"),
                last.format("%Y-%m-%d %H:%M:%S")
            )?;
        }

        write!(f, ")")
    }
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

    // Gap filling tests

    #[test]
    fn frequency_parses_duration_based_strings() {
        // Seconds
        assert_eq!(
            Frequency::parse("30s").unwrap(),
            Frequency::Duration(Duration::seconds(30))
        );
        assert_eq!(
            Frequency::parse("1sec").unwrap(),
            Frequency::Duration(Duration::seconds(1))
        );

        // Minutes
        assert_eq!(
            Frequency::parse("30m").unwrap(),
            Frequency::Duration(Duration::minutes(30))
        );
        assert_eq!(
            Frequency::parse("30min").unwrap(),
            Frequency::Duration(Duration::minutes(30))
        );

        // Hours
        assert_eq!(
            Frequency::parse("1h").unwrap(),
            Frequency::Duration(Duration::hours(1))
        );
        assert_eq!(
            Frequency::parse("24h").unwrap(),
            Frequency::Duration(Duration::hours(24))
        );

        // Days
        assert_eq!(
            Frequency::parse("1d").unwrap(),
            Frequency::Duration(Duration::days(1))
        );
        assert_eq!(
            Frequency::parse("7d").unwrap(),
            Frequency::Duration(Duration::days(7))
        );

        // Weeks
        assert_eq!(
            Frequency::parse("1w").unwrap(),
            Frequency::Duration(Duration::weeks(1))
        );
        assert_eq!(
            Frequency::parse("2w").unwrap(),
            Frequency::Duration(Duration::weeks(2))
        );
    }

    #[test]
    fn frequency_parses_calendar_based_strings() {
        // Months
        assert_eq!(Frequency::parse("1mo").unwrap(), Frequency::Months(1));
        assert_eq!(Frequency::parse("3mo").unwrap(), Frequency::Months(3));

        // Quarters
        assert_eq!(Frequency::parse("1q").unwrap(), Frequency::Months(3));
        assert_eq!(Frequency::parse("2q").unwrap(), Frequency::Months(6));

        // Years
        assert_eq!(Frequency::parse("1y").unwrap(), Frequency::Years(1));
        assert_eq!(Frequency::parse("2y").unwrap(), Frequency::Years(2));
    }

    #[test]
    fn frequency_parse_handles_invalid_input() {
        // No number
        assert!(Frequency::parse("h").is_err());
        assert!(Frequency::parse("mo").is_err());

        // Unknown unit
        assert!(Frequency::parse("1x").is_err());
        assert!(Frequency::parse("5foo").is_err());

        // Empty string
        assert!(Frequency::parse("").is_err());
    }

    #[test]
    fn fill_gaps_with_hourly_frequency() {
        // Create timestamps with a gap at hour 2
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            // Gap at 2:00
            Utc.with_ymd_and_hms(2024, 1, 1, 3, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 4, 0, 0).unwrap(),
        ];
        let values = vec![0.0, 1.0, 3.0, 4.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps_str("1h").unwrap();

        assert_eq!(filled.len(), 5);
        assert_eq!(
            filled.timestamps()[2],
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap()
        );

        let vals = filled.primary_values();
        assert_relative_eq!(vals[0], 0.0);
        assert_relative_eq!(vals[1], 1.0);
        assert!(vals[2].is_nan()); // Gap filled with NaN
        assert_relative_eq!(vals[3], 3.0);
        assert_relative_eq!(vals[4], 4.0);
    }

    #[test]
    fn fill_gaps_with_daily_frequency() {
        // Create timestamps with gaps
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            // Gap at Jan 2
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
            // Gap at Jan 4
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0, 5.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps_str("1d").unwrap();

        assert_eq!(filled.len(), 5);

        let vals = filled.primary_values();
        assert_relative_eq!(vals[0], 1.0);
        assert!(vals[1].is_nan()); // Jan 2 - gap
        assert_relative_eq!(vals[2], 3.0);
        assert!(vals[3].is_nan()); // Jan 4 - gap
        assert_relative_eq!(vals[4], 5.0);
    }

    #[test]
    fn fill_gaps_with_weekly_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // Week 1
            Utc.with_ymd_and_hms(2024, 1, 8, 0, 0, 0).unwrap(), // Week 2
            // Gap at Week 3 (Jan 15)
            Utc.with_ymd_and_hms(2024, 1, 22, 0, 0, 0).unwrap(), // Week 4
        ];
        let values = vec![1.0, 2.0, 4.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps_str("1w").unwrap();

        assert_eq!(filled.len(), 4);
        assert!(filled.primary_values()[2].is_nan()); // Gap at week 3
    }

    #[test]
    fn fill_gaps_with_monthly_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            // Gap at Feb
            Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0, 4.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps(Frequency::Months(1)).unwrap();

        assert_eq!(filled.len(), 4);
        assert_eq!(
            filled.timestamps()[1],
            Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).unwrap()
        );
        assert!(filled.primary_values()[1].is_nan()); // Feb is filled with NaN
    }

    #[test]
    fn fill_gaps_with_quarterly_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // Q1
            // Gap at Q2 (Apr)
            Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap(), // Q3
        ];
        let values = vec![1.0, 3.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps_str("1q").unwrap();

        assert_eq!(filled.len(), 3);
        assert_eq!(
            filled.timestamps()[1],
            Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap()
        );
        assert!(filled.primary_values()[1].is_nan());
    }

    #[test]
    fn fill_gaps_with_yearly_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap(),
            // Gap at 2021
            Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0, 4.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps(Frequency::Years(1)).unwrap();

        assert_eq!(filled.len(), 4);
        assert_eq!(
            filled.timestamps()[1],
            Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap()
        );
        assert!(filled.primary_values()[1].is_nan());
    }

    #[test]
    fn fill_gaps_with_30_minute_frequency() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 30, 0).unwrap(),
            // Gap at 1:00
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 30, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 4.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps_str("30m").unwrap();

        assert_eq!(filled.len(), 4);
        assert_eq!(
            filled.timestamps()[2],
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap()
        );
        assert!(filled.primary_values()[2].is_nan());
    }

    #[test]
    fn fill_gaps_preserves_multivariate_data() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            // Gap at 1:00
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![
            vec![1.0, 3.0],   // dimension 0
            vec![10.0, 30.0], // dimension 1
        ];

        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Column)
            .build()
            .unwrap();

        let filled = ts.fill_gaps_str("1h").unwrap();

        assert_eq!(filled.len(), 3);
        assert_eq!(filled.dimensions(), 2);

        // Check dimension 0
        let dim0 = filled.values(0).unwrap();
        assert_relative_eq!(dim0[0], 1.0);
        assert!(dim0[1].is_nan());
        assert_relative_eq!(dim0[2], 3.0);

        // Check dimension 1
        let dim1 = filled.values(1).unwrap();
        assert_relative_eq!(dim1[0], 10.0);
        assert!(dim1[1].is_nan());
        assert_relative_eq!(dim1[2], 30.0);
    }

    #[test]
    fn fill_gaps_preserves_metadata() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0];

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_labels(vec!["temperature".to_string()]).unwrap();
        ts.set_metadata("source".to_string(), "sensor".to_string());
        ts.set_timezone("Europe/London".to_string());

        let filled = ts.fill_gaps_str("1h").unwrap();

        assert_eq!(filled.labels(), &["temperature"]);
        assert_eq!(filled.metadata().get("source"), Some(&"sensor".to_string()));
        assert_eq!(filled.timezone(), Some("Europe/London"));
        assert_eq!(filled.frequency(), Some(Duration::hours(1)));
    }

    #[test]
    fn fill_gaps_handles_empty_series() {
        let ts = TimeSeries::univariate(vec![], vec![]).unwrap();
        let filled = ts.fill_gaps_str("1h").unwrap();
        assert!(filled.is_empty());
    }

    #[test]
    fn fill_gaps_handles_single_element() {
        let timestamps = vec![Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()];
        let values = vec![1.0];

        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
        let filled = ts.fill_gaps_str("1h").unwrap();

        assert_eq!(filled.len(), 1);
        assert_eq!(filled.timestamps(), &timestamps);
        assert_eq!(filled.primary_values(), &values);
    }

    #[test]
    fn fill_gaps_handles_no_gaps() {
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 1, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 1, 1, 2, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 3.0];

        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
        let filled = ts.fill_gaps_str("1h").unwrap();

        assert_eq!(filled.len(), 3);
        assert_eq!(filled.timestamps(), &timestamps);
        assert_eq!(filled.primary_values(), &values);
        assert!(!filled.has_missing_values());
    }

    #[test]
    fn fill_gaps_month_end_handling() {
        // Test that month-end dates are handled correctly
        // Using first-of-month dates to avoid month-end complications
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            // Gap at Feb 1
            Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap(),
        ];
        let values = vec![1.0, 3.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps(Frequency::Months(1)).unwrap();

        assert_eq!(filled.len(), 3);
        // Feb 1 should be generated
        assert_eq!(
            filled.timestamps()[1],
            Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).unwrap()
        );
        assert!(filled.primary_values()[1].is_nan());
    }

    #[test]
    fn fill_gaps_handles_end_of_month_dates() {
        // When using month-end dates, ensure they still align correctly
        // Jan 31 -> Feb 29 (leap year) -> Mar 29 (not 31!)
        let timestamps = vec![
            Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 2, 29, 0, 0, 0).unwrap(), // Feb 29 exists in original
        ];
        let values = vec![1.0, 2.0];

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let filled = ts.fill_gaps(Frequency::Months(1)).unwrap();

        // Jan 31 + 1mo = Feb 29 (clamped), which matches existing timestamp
        assert_eq!(filled.len(), 2);
        assert!(!filled.has_missing_values());
    }

    // === Missing value imputation tests ===

    #[test]
    fn backward_fill_basic() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, f64::NAN, f64::NAN];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::BackwardFill).unwrap();
        // Trailing NaN left as NaN (no next valid value)
        assert_relative_eq!(result.primary_values()[0], 1.0);
        assert_relative_eq!(result.primary_values()[1], 2.0);
        assert_relative_eq!(result.primary_values()[2], 3.0);
        assert!(result.primary_values()[3].is_nan());
        assert!(result.primary_values()[4].is_nan());
    }

    #[test]
    fn backward_fill_interior() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::BackwardFill).unwrap();
        assert_relative_eq!(result.primary_values()[0], 1.0);
        assert_relative_eq!(result.primary_values()[1], 4.0);
        assert_relative_eq!(result.primary_values()[2], 4.0);
        assert_relative_eq!(result.primary_values()[3], 4.0);
        assert_relative_eq!(result.primary_values()[4], 5.0);
    }

    #[test]
    fn backward_fill_leading_nan() {
        let timestamps = make_timestamps(4);
        let values = vec![f64::NAN, f64::NAN, 3.0, 4.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::BackwardFill).unwrap();
        assert_relative_eq!(result.primary_values()[0], 3.0);
        assert_relative_eq!(result.primary_values()[1], 3.0);
        assert_relative_eq!(result.primary_values()[2], 3.0);
        assert_relative_eq!(result.primary_values()[3], 4.0);
    }

    #[test]
    fn fill_mean_basic() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::FillMean).unwrap();
        // Mean of [1, 3, 5] = 3.0
        assert_relative_eq!(result.primary_values()[0], 1.0);
        assert_relative_eq!(result.primary_values()[1], 3.0);
        assert_relative_eq!(result.primary_values()[2], 3.0);
        assert_relative_eq!(result.primary_values()[3], 3.0);
        assert_relative_eq!(result.primary_values()[4], 5.0);
    }

    #[test]
    fn fill_median_basic() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::FillMedian).unwrap();
        // Median of [1, 3, 10] = 3.0
        assert_relative_eq!(result.primary_values()[0], 1.0);
        assert_relative_eq!(result.primary_values()[1], 3.0);
        assert_relative_eq!(result.primary_values()[2], 3.0);
        assert_relative_eq!(result.primary_values()[3], 3.0);
        assert_relative_eq!(result.primary_values()[4], 10.0);
    }

    #[test]
    fn fill_mean_all_nan() {
        let timestamps = make_timestamps(3);
        let values = vec![f64::NAN, f64::NAN, f64::NAN];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::FillMean).unwrap();
        // All-NaN produces NaN fill
        assert!(result.primary_values()[0].is_nan());
        assert!(result.primary_values()[1].is_nan());
        assert!(result.primary_values()[2].is_nan());
    }

    #[test]
    fn fill_mean_with_inf() {
        let timestamps = make_timestamps(4);
        let values = vec![2.0, f64::INFINITY, 4.0, f64::NAN];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.sanitized(MissingValuePolicy::FillMean).unwrap();
        // Mean of [2, 4] = 3.0, Inf and NaN both replaced
        assert_relative_eq!(result.primary_values()[0], 2.0);
        assert_relative_eq!(result.primary_values()[1], 3.0);
        assert_relative_eq!(result.primary_values()[2], 4.0);
        assert_relative_eq!(result.primary_values()[3], 3.0);
    }

    #[test]
    fn interpolate_policy_matches_interpolated() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let via_policy = ts.sanitized(MissingValuePolicy::Interpolate).unwrap();
        let via_method = ts.interpolated(true);

        for (a, b) in via_policy
            .primary_values()
            .iter()
            .zip(via_method.primary_values().iter())
        {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn missing_mask_correct() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let mask = ts.missing_mask();
        assert_eq!(mask, vec![false, true, false, true, false]);
    }

    #[test]
    fn missing_count_multivariate() {
        let timestamps = make_timestamps(4);
        let values = vec![
            vec![1.0, f64::NAN, 3.0, f64::NAN],      // 2 missing
            vec![f64::NAN, 2.0, f64::NAN, f64::NAN], // 3 missing
        ];
        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, ValueLayout::Column)
            .build()
            .unwrap();
        assert_eq!(ts.missing_count(), vec![2, 3]);
    }

    #[test]
    fn forward_backward_handles_leading() {
        let timestamps = make_timestamps(5);
        let values = vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.imputed_forward_backward();
        // Leading NaN filled backward from 3.0, interior NaN filled forward from 3.0
        assert_relative_eq!(result.primary_values()[0], 3.0);
        assert_relative_eq!(result.primary_values()[1], 3.0);
        assert_relative_eq!(result.primary_values()[2], 3.0);
        assert_relative_eq!(result.primary_values()[3], 3.0);
        assert_relative_eq!(result.primary_values()[4], 5.0);
    }

    #[test]
    fn forward_backward_handles_trailing() {
        let timestamps = make_timestamps(4);
        let values = vec![1.0, 2.0, f64::NAN, f64::NAN];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.imputed_forward_backward();
        assert_relative_eq!(result.primary_values()[0], 1.0);
        assert_relative_eq!(result.primary_values()[1], 2.0);
        // Trailing NaN forward-filled from 2.0
        assert_relative_eq!(result.primary_values()[2], 2.0);
        assert_relative_eq!(result.primary_values()[3], 2.0);
    }

    #[test]
    fn moving_average_single_gap() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.imputed_moving_average(3).unwrap();
        // Window of 3: neighbors are 2.0 and 4.0 → mean = 3.0
        assert_relative_eq!(result.primary_values()[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn moving_average_adjacent_gaps() {
        let timestamps = make_timestamps(6);
        let values = vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0, 6.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.imputed_moving_average(3).unwrap();
        // After multi-pass, gaps should be filled
        assert!(result.primary_values()[1].is_finite());
        assert!(result.primary_values()[2].is_finite());
    }

    #[test]
    fn moving_average_rejects_even_window() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        assert!(ts.imputed_moving_average(4).is_err());
    }

    #[test]
    fn moving_average_rejects_zero_window() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        assert!(ts.imputed_moving_average(0).is_err());
    }

    #[test]
    fn seasonal_imputation_basic() {
        // Period 3: positions 0,1,2,0,1,2,0,1,2
        let timestamps = make_timestamps(9);
        let values = vec![
            10.0,
            20.0,
            30.0, // cycle 1
            11.0,
            21.0,
            31.0, // cycle 2
            f64::NAN,
            22.0,
            32.0, // cycle 3 - NaN at position 0
        ];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let result = ts.imputed_seasonal(3).unwrap();
        // Position 0 values: [10.0, 11.0], median = 10.5
        assert_relative_eq!(result.primary_values()[6], 10.5, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_imputation_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, f64::NAN, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        // Period 4 but only 3 data points
        assert!(ts.imputed_seasonal(4).is_err());
    }

    #[test]
    fn seasonal_imputation_rejects_zero_period() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        assert!(ts.imputed_seasonal(0).is_err());
    }

    #[test]
    fn seasonal_imputation_rejects_too_many_missing() {
        // Period 2: bucket 0 has indices [0, 2, 4], bucket 1 has [1, 3, 5]
        let timestamps = make_timestamps(6);
        let values = vec![f64::NAN, 1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        // Bucket 0 is 100% NaN (3/3)
        assert!(ts.imputed_seasonal(2).is_err());
    }

    #[test]
    fn with_imputed_regressors_fill_mean() {
        let timestamps = make_daily_timestamps(4);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let calendar = CalendarAnnotations::new()
            .with_regressor("promo".to_string(), vec![1.0, f64::NAN, 3.0, f64::NAN]);

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_calendar(calendar);

        let result = ts
            .with_imputed_regressors(MissingValuePolicy::FillMean)
            .unwrap();
        let promo = result.regressor("promo").unwrap();
        // Mean of [1, 3] = 2.0
        assert_relative_eq!(promo[0], 1.0);
        assert_relative_eq!(promo[1], 2.0);
        assert_relative_eq!(promo[2], 3.0);
        assert_relative_eq!(promo[3], 2.0);
    }

    // --- Temporal aggregation / resampling tests ---

    #[test]
    fn agg_sum_exact() {
        let ts =
            TimeSeries::univariate(make_timestamps(6), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a = ts.aggregate(3, AggregationMethod::Sum);
        assert_eq!(a.len(), 2);
        assert_relative_eq!(a.primary_values()[0], 6.0);
        assert_relative_eq!(a.primary_values()[1], 15.0);
    }

    #[test]
    fn agg_sum_rem() {
        let ts =
            TimeSeries::univariate(make_timestamps(7), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .unwrap();
        let a = ts.aggregate(3, AggregationMethod::Sum);
        assert_eq!(a.len(), 3);
        assert_relative_eq!(a.primary_values()[2], 7.0);
    }

    #[test]
    fn agg_mean_exact() {
        let ts = TimeSeries::univariate(make_timestamps(6), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
            .unwrap();
        let a = ts.aggregate(2, AggregationMethod::Mean);
        assert_eq!(a.len(), 3);
        assert_relative_eq!(a.primary_values()[0], 3.0);
        assert_relative_eq!(a.primary_values()[1], 7.0);
        assert_relative_eq!(a.primary_values()[2], 11.0);
    }

    #[test]
    fn agg_mean_rem() {
        let ts =
            TimeSeries::univariate(make_timestamps(5), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let a = ts.aggregate(3, AggregationMethod::Mean);
        assert_relative_eq!(a.primary_values()[0], 20.0);
        assert_relative_eq!(a.primary_values()[1], 45.0);
    }

    #[test]
    fn agg_median_test() {
        let ts = TimeSeries::univariate(make_timestamps(5), vec![1.0, 5.0, 3.0, 7.0, 2.0]).unwrap();
        let a = ts.aggregate(3, AggregationMethod::Median);
        assert_relative_eq!(a.primary_values()[0], 3.0);
        assert_relative_eq!(a.primary_values()[1], 4.5);
    }

    #[test]
    fn agg_first_last_min_max() {
        let ts = TimeSeries::univariate(make_timestamps(4), vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        assert_relative_eq!(
            ts.aggregate(2, AggregationMethod::First).primary_values()[0],
            3.0
        );
        assert_relative_eq!(
            ts.aggregate(2, AggregationMethod::Last).primary_values()[0],
            1.0
        );
        assert_relative_eq!(
            ts.aggregate(2, AggregationMethod::Min).primary_values()[0],
            1.0
        );
        assert_relative_eq!(
            ts.aggregate(2, AggregationMethod::Max).primary_values()[0],
            3.0
        );
    }

    #[test]
    fn agg_ts_first() {
        let t = make_timestamps(6);
        let ts = TimeSeries::univariate(t.clone(), vec![1.0; 6]).unwrap();
        let a = ts.aggregate(3, AggregationMethod::Sum);
        assert_eq!(a.timestamps()[0], t[0]);
        assert_eq!(a.timestamps()[1], t[3]);
    }

    #[test]
    fn agg_identity() {
        let ts = TimeSeries::univariate(make_timestamps(4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(ts.aggregate(1, AggregationMethod::Sum).len(), 4);
        assert_eq!(ts.aggregate(0, AggregationMethod::Sum).len(), 4);
    }

    #[test]
    fn agg_empty() {
        assert!(TimeSeries::univariate(vec![], vec![])
            .unwrap()
            .aggregate(3, AggregationMethod::Sum)
            .is_empty());
    }

    #[test]
    fn ds_correct() {
        let t = make_timestamps(10);
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(t.clone(), v).unwrap();
        let d = ts.downsample(3);
        assert_eq!(d.len(), 4);
        assert_relative_eq!(d.primary_values()[1], 3.0);
        assert_eq!(d.timestamps()[1], t[3]);
    }

    #[test]
    fn ds_edge() {
        let ts = TimeSeries::univariate(make_timestamps(5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(ts.downsample(1).len(), 5);
        assert_eq!(ts.downsample(0).len(), 5);
        assert_eq!(ts.downsample(10).len(), 1);
        assert!(TimeSeries::univariate(vec![], vec![])
            .unwrap()
            .downsample(3)
            .is_empty());
    }

    #[test]
    fn us_linear() {
        let ts = TimeSeries::univariate(make_timestamps(3), vec![0.0, 6.0, 12.0]).unwrap();
        let u = ts.upsample(3, InterpolationMethod::Linear);
        assert_eq!(u.len(), 7);
        for (i, e) in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0].iter().enumerate() {
            assert_relative_eq!(u.primary_values()[i], e);
        }
    }

    #[test]
    fn us_ffill() {
        let ts = TimeSeries::univariate(make_timestamps(3), vec![10.0, 20.0, 30.0]).unwrap();
        let u = ts.upsample(2, InterpolationMethod::ForwardFill);
        for (i, e) in [10.0, 10.0, 20.0, 20.0, 30.0].iter().enumerate() {
            assert_relative_eq!(u.primary_values()[i], e);
        }
    }

    #[test]
    fn us_bfill() {
        let ts = TimeSeries::univariate(make_timestamps(3), vec![10.0, 20.0, 30.0]).unwrap();
        let u = ts.upsample(2, InterpolationMethod::BackwardFill);
        for (i, e) in [10.0, 20.0, 20.0, 30.0, 30.0].iter().enumerate() {
            assert_relative_eq!(u.primary_values()[i], e);
        }
    }

    #[test]
    fn us_zero_test() {
        let ts = TimeSeries::univariate(make_timestamps(3), vec![10.0, 20.0, 30.0]).unwrap();
        let u = ts.upsample(2, InterpolationMethod::Zero);
        for (i, e) in [10.0, 0.0, 20.0, 0.0, 30.0].iter().enumerate() {
            assert_relative_eq!(u.primary_values()[i], e);
        }
    }

    #[test]
    fn us_ts_interp() {
        let t = make_timestamps(3);
        let ts = TimeSeries::univariate(t.clone(), vec![0.0, 1.0, 2.0]).unwrap();
        let u = ts.upsample(2, InterpolationMethod::Linear);
        assert_eq!(u.timestamps()[0], t[0]);
        assert_eq!(u.timestamps()[2], t[1]);
        assert_eq!(u.timestamps()[1], t[0] + (t[1] - t[0]) / 2);
    }

    #[test]
    fn us_edge() {
        let ts = TimeSeries::univariate(make_timestamps(3), vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(ts.upsample(1, InterpolationMethod::Linear).len(), 3);
        let ts1 = TimeSeries::univariate(make_timestamps(1), vec![42.0]).unwrap();
        assert_eq!(ts1.upsample(5, InterpolationMethod::Linear).len(), 1);
        assert!(TimeSeries::univariate(vec![], vec![])
            .unwrap()
            .upsample(3, InterpolationMethod::Linear)
            .is_empty());
    }

    #[test]
    fn us_roundtrip() {
        let v: Vec<f64> = (0..11).map(|i| i as f64 * 2.0).collect();
        let ts = TimeSeries::univariate(make_timestamps(11), v.clone()).unwrap();
        let u = ts.downsample(2).upsample(2, InterpolationMethod::Linear);
        for i in 0..11 {
            assert_relative_eq!(u.primary_values()[i], v[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn sw_sum_s1() {
        let t = make_timestamps(5);
        let ts = TimeSeries::univariate(t.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let s = ts.sliding_window_aggregate(3, 1, AggregationMethod::Sum);
        assert_eq!(s.len(), 3);
        assert_relative_eq!(s.primary_values()[0], 6.0);
        assert_relative_eq!(s.primary_values()[1], 9.0);
        assert_eq!(s.timestamps()[0], t[0]);
    }

    #[test]
    fn sw_mean_s2() {
        let ts = TimeSeries::univariate(make_timestamps(6), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
            .unwrap();
        let s = ts.sliding_window_aggregate(3, 2, AggregationMethod::Mean);
        assert_eq!(s.len(), 2);
        assert_relative_eq!(s.primary_values()[0], 4.0);
        assert_relative_eq!(s.primary_values()[1], 8.0);
    }

    #[test]
    fn sw_edge() {
        let ts = TimeSeries::univariate(make_timestamps(4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(ts
            .sliding_window_aggregate(5, 1, AggregationMethod::Sum)
            .is_empty());
        assert_eq!(
            ts.sliding_window_aggregate(4, 1, AggregationMethod::Sum)
                .len(),
            1
        );
        assert!(ts
            .sliding_window_aggregate(0, 1, AggregationMethod::Sum)
            .is_empty());
        let s = ts.sliding_window_aggregate(2, 0, AggregationMethod::Sum);
        assert_eq!(s.len(), 3);
        assert_relative_eq!(s.primary_values()[0], 3.0);
        assert_relative_eq!(s.primary_values()[2], 7.0);
        assert!(TimeSeries::univariate(vec![], vec![])
            .unwrap()
            .sliding_window_aggregate(3, 1, AggregationMethod::Sum)
            .is_empty());
    }

    #[test]
    fn with_imputed_regressors_rejects_drop() {
        let timestamps = make_daily_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let calendar =
            CalendarAnnotations::new().with_regressor("x".to_string(), vec![1.0, f64::NAN, 3.0]);

        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_calendar(calendar);

        assert!(ts
            .with_imputed_regressors(MissingValuePolicy::Drop)
            .is_err());
    }

    // ---------------------------------------------------------------
    // generate_future_timestamps & TimeSeries::future_timestamps tests
    // ---------------------------------------------------------------

    /// Helper: build a UTC datetime from date components.
    fn ymd(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
    }

    /// Helper: build a UTC datetime from date + time components.
    fn ymdhms(year: i32, month: u32, day: u32, h: u32, m: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, h, m, s).unwrap()
    }

    // --- 1. Monthly stepping from month-end dates ---

    #[test]
    fn monthly_step_jan31_clamps_to_feb28_in_non_leap_year() {
        // Jan 31 2023 + 1mo => Feb 28 2023 (non-leap)
        let last = ymd(2023, 1, 31);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ymd(2023, 2, 28)); // Feb 28
        assert_eq!(result[1], ymd(2023, 3, 28)); // Mar 28 (clamped from prev Feb 28)
        assert_eq!(result[2], ymd(2023, 4, 28)); // Apr 28
    }

    #[test]
    fn monthly_step_jan31_clamps_to_feb29_in_leap_year() {
        // Jan 31 2024 + 1mo => Feb 29 2024 (leap year)
        let last = ymd(2024, 1, 31);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], ymd(2024, 2, 29)); // Feb 29 (leap)
        assert_eq!(result[1], ymd(2024, 3, 29)); // Mar 29 (clamped from prev Feb 29)
        assert_eq!(result[2], ymd(2024, 4, 29)); // Apr 29
        assert_eq!(result[3], ymd(2024, 5, 29)); // May 29
    }

    #[test]
    fn monthly_step_mar31_to_apr30_and_beyond() {
        let last = ymd(2024, 3, 31);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 3);
        assert_eq!(result[0], ymd(2024, 4, 30)); // Apr has 30 days
        assert_eq!(result[1], ymd(2024, 5, 30)); // clamped from prev Apr 30
        assert_eq!(result[2], ymd(2024, 6, 30)); // Jun has 30 days
    }

    #[test]
    fn monthly_step_crosses_year_boundary() {
        let last = ymd(2024, 11, 15);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 3);
        assert_eq!(result[0], ymd(2024, 12, 15));
        assert_eq!(result[1], ymd(2025, 1, 15));
        assert_eq!(result[2], ymd(2025, 2, 15));
    }

    // --- 2. Quarterly stepping (Frequency::Months(3)) ---

    #[test]
    fn quarterly_step_from_jan31() {
        let last = ymd(2024, 1, 31);
        let result = generate_future_timestamps(&last, &Frequency::Months(3), 4);
        assert_eq!(result[0], ymd(2024, 4, 30)); // Apr has 30 days
        assert_eq!(result[1], ymd(2024, 7, 30)); // clamped from prev Apr 30
        assert_eq!(result[2], ymd(2024, 10, 30)); // clamped from prev Jul 30
        assert_eq!(result[3], ymd(2025, 1, 30)); // clamped from prev Oct 30
    }

    #[test]
    fn quarterly_step_from_mid_month() {
        let last = ymd(2024, 3, 15);
        let result = generate_future_timestamps(&last, &Frequency::Months(3), 4);
        assert_eq!(result[0], ymd(2024, 6, 15));
        assert_eq!(result[1], ymd(2024, 9, 15));
        assert_eq!(result[2], ymd(2024, 12, 15));
        assert_eq!(result[3], ymd(2025, 3, 15));
    }

    #[test]
    fn quarterly_step_crosses_year_boundary() {
        let last = ymd(2024, 10, 31);
        let result = generate_future_timestamps(&last, &Frequency::Months(3), 2);
        assert_eq!(result[0], ymd(2025, 1, 31));
        assert_eq!(result[1], ymd(2025, 4, 30)); // Apr has 30 days
    }

    // --- 3. Yearly stepping from Feb 29 (leap -> non-leap) ---

    #[test]
    fn yearly_step_from_feb29_leap_to_non_leap() {
        let last = ymd(2024, 2, 29); // 2024 is a leap year
        let result = generate_future_timestamps(&last, &Frequency::Years(1), 4);
        assert_eq!(result[0], ymd(2025, 2, 28)); // 2025 not leap
        assert_eq!(result[1], ymd(2026, 2, 28)); // 2026 not leap
        assert_eq!(result[2], ymd(2027, 2, 28)); // 2027 not leap
        assert_eq!(result[3], ymd(2028, 2, 28)); // 2028 IS leap, but clamped from prev Feb 28
    }

    #[test]
    fn yearly_step_from_regular_date() {
        let last = ymd(2024, 7, 4);
        let result = generate_future_timestamps(&last, &Frequency::Years(1), 3);
        assert_eq!(result[0], ymd(2025, 7, 4));
        assert_eq!(result[1], ymd(2026, 7, 4));
        assert_eq!(result[2], ymd(2027, 7, 4));
    }

    #[test]
    fn yearly_step_two_years() {
        let last = ymd(2024, 2, 29);
        let result = generate_future_timestamps(&last, &Frequency::Years(2), 3);
        assert_eq!(result[0], ymd(2026, 2, 28)); // not leap
        assert_eq!(result[1], ymd(2028, 2, 28)); // leap but clamped from prev 28
        assert_eq!(result[2], ymd(2030, 2, 28)); // not leap
    }

    // --- 4. Daily / weekly / hourly Duration-based stepping ---

    #[test]
    fn daily_duration_step() {
        let last = ymd(2024, 12, 30);
        let result = generate_future_timestamps(&last, &Frequency::Duration(Duration::days(1)), 3);
        assert_eq!(result[0], ymd(2024, 12, 31));
        assert_eq!(result[1], ymd(2025, 1, 1));
        assert_eq!(result[2], ymd(2025, 1, 2));
    }

    #[test]
    fn weekly_duration_step() {
        let last = ymd(2024, 1, 1); // Monday
        let result = generate_future_timestamps(&last, &Frequency::Duration(Duration::weeks(1)), 4);
        assert_eq!(result[0], ymd(2024, 1, 8));
        assert_eq!(result[1], ymd(2024, 1, 15));
        assert_eq!(result[2], ymd(2024, 1, 22));
        assert_eq!(result[3], ymd(2024, 1, 29));
    }

    #[test]
    fn hourly_duration_step() {
        let last = ymdhms(2024, 3, 10, 22, 0, 0);
        let result = generate_future_timestamps(&last, &Frequency::Duration(Duration::hours(1)), 4);
        assert_eq!(result[0], ymdhms(2024, 3, 10, 23, 0, 0));
        assert_eq!(result[1], ymdhms(2024, 3, 11, 0, 0, 0)); // crosses midnight
        assert_eq!(result[2], ymdhms(2024, 3, 11, 1, 0, 0));
        assert_eq!(result[3], ymdhms(2024, 3, 11, 2, 0, 0));
    }

    #[test]
    fn sub_hourly_duration_step_30min() {
        let last = ymdhms(2024, 1, 1, 0, 0, 0);
        let result =
            generate_future_timestamps(&last, &Frequency::Duration(Duration::minutes(30)), 3);
        assert_eq!(result[0], ymdhms(2024, 1, 1, 0, 30, 0));
        assert_eq!(result[1], ymdhms(2024, 1, 1, 1, 0, 0));
        assert_eq!(result[2], ymdhms(2024, 1, 1, 1, 30, 0));
    }

    // --- 5. TimeSeries::future_timestamps auto-inference ---

    #[test]
    fn future_timestamps_infers_daily_frequency() {
        // Build a daily series of 30 points so inference is reliable
        let timestamps: Vec<DateTime<Utc>> = (0..30)
            .map(|i| ymd(2024, 1, 1) + Duration::days(i))
            .collect();
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let future = ts.future_timestamps(5).unwrap();
        assert_eq!(future.len(), 5);
        assert_eq!(future[0], ymd(2024, 1, 31));
        assert_eq!(future[1], ymd(2024, 2, 1));
        assert_eq!(future[2], ymd(2024, 2, 2));
        assert_eq!(future[3], ymd(2024, 2, 3));
        assert_eq!(future[4], ymd(2024, 2, 4));
    }

    #[test]
    fn future_timestamps_infers_weekly_frequency() {
        // Build a weekly series (every Monday) for 10 weeks
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| ymd(2024, 1, 1) + Duration::weeks(i))
            .collect();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let future = ts.future_timestamps(3).unwrap();
        assert_eq!(future.len(), 3);
        // Last data point is 2024-01-01 + 9 weeks = 2024-03-04
        assert_eq!(future[0], ymd(2024, 3, 11));
        assert_eq!(future[1], ymd(2024, 3, 18));
        assert_eq!(future[2], ymd(2024, 3, 25));
    }

    #[test]
    fn future_timestamps_infers_monthly_frequency() {
        // Build a monthly series using first-of-month dates
        let timestamps: Vec<DateTime<Utc>> = (0..12).map(|i| ymd(2024, 1 + i as u32, 1)).collect();
        let values: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // future_timestamps uses infer_frequency_calendar which returns a Duration.
        // For monthly data the modal diff will be ~30 or 31 days.
        let future = ts.future_timestamps(3).unwrap();
        assert_eq!(future.len(), 3);
        // The inferred Duration-based step is the modal diff in seconds.
        // For 1st-of-month data, most gaps are 31 days (Jan, Mar, May, Jul, Aug, Oct
        // have 31 days). The modal diff should be 31*86400 seconds.
        let last = ymd(2024, 12, 1);
        let step = future[0] - last;
        // The step should be consistent across all future timestamps
        assert_eq!(future[1] - future[0], step);
        assert_eq!(future[2] - future[1], step);
    }

    // --- 6. Empty series edge case ---

    #[test]
    fn future_timestamps_empty_series_returns_error() {
        let ts = TimeSeries::univariate(vec![], vec![]).unwrap();
        let result = ts.future_timestamps(5);
        assert!(result.is_err());
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    // --- 7. Horizon=0 returns empty vec ---

    #[test]
    fn generate_future_timestamps_horizon_zero_returns_empty() {
        let last = ymd(2024, 6, 15);
        let result = generate_future_timestamps(&last, &Frequency::Duration(Duration::days(1)), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_future_timestamps_horizon_zero_monthly() {
        let last = ymd(2024, 6, 15);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn generate_future_timestamps_horizon_zero_yearly() {
        let last = ymd(2024, 6, 15);
        let result = generate_future_timestamps(&last, &Frequency::Years(1), 0);
        assert!(result.is_empty());
    }

    // --- 8. Large horizon (100+ steps) ---

    #[test]
    fn large_horizon_daily_produces_correct_count_and_ordering() {
        let last = ymd(2024, 1, 1);
        let horizon = 365;
        let result =
            generate_future_timestamps(&last, &Frequency::Duration(Duration::days(1)), horizon);
        assert_eq!(result.len(), horizon);
        // First and last
        assert_eq!(result[0], ymd(2024, 1, 2));
        assert_eq!(result[364], ymd(2024, 12, 31)); // 2024 is leap => 366 days, so day 366 = Dec 31
                                                    // Strictly increasing
        for w in result.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn large_horizon_monthly_120_steps() {
        let last = ymd(2024, 1, 15);
        let horizon = 120; // 10 years of months
        let result = generate_future_timestamps(&last, &Frequency::Months(1), horizon);
        assert_eq!(result.len(), horizon);
        // First step
        assert_eq!(result[0], ymd(2024, 2, 15));
        // 12th step: Jan 2025
        assert_eq!(result[11], ymd(2025, 1, 15));
        // 120th step: Jan 2034
        assert_eq!(result[119], ymd(2034, 1, 15));
        // Strictly increasing
        for w in result.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn large_horizon_yearly_200_steps() {
        let last = ymd(2024, 6, 15);
        let horizon = 200;
        let result = generate_future_timestamps(&last, &Frequency::Years(1), horizon);
        assert_eq!(result.len(), horizon);
        assert_eq!(result[0], ymd(2025, 6, 15));
        assert_eq!(result[199], ymd(2224, 6, 15));
        for w in result.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn large_horizon_hourly_1000_steps() {
        let last = ymdhms(2024, 1, 1, 0, 0, 0);
        let horizon = 1000;
        let result =
            generate_future_timestamps(&last, &Frequency::Duration(Duration::hours(1)), horizon);
        assert_eq!(result.len(), horizon);
        assert_eq!(result[0], ymdhms(2024, 1, 1, 1, 0, 0));
        // 1000 hours = 41 days + 16 hours
        assert_eq!(result[999], ymdhms(2024, 2, 11, 16, 0, 0));
        for w in result.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    // --- Additional edge cases ---

    #[test]
    fn monthly_step_preserves_time_of_day() {
        let last = ymdhms(2024, 1, 15, 14, 30, 45);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 2);
        assert_eq!(result[0], ymdhms(2024, 2, 15, 14, 30, 45));
        assert_eq!(result[1], ymdhms(2024, 3, 15, 14, 30, 45));
    }

    #[test]
    fn yearly_step_preserves_time_of_day() {
        let last = ymdhms(2024, 7, 4, 10, 15, 0);
        let result = generate_future_timestamps(&last, &Frequency::Years(1), 2);
        assert_eq!(result[0], ymdhms(2025, 7, 4, 10, 15, 0));
        assert_eq!(result[1], ymdhms(2026, 7, 4, 10, 15, 0));
    }

    #[test]
    fn monthly_step_single_step() {
        let last = ymd(2024, 6, 30);
        let result = generate_future_timestamps(&last, &Frequency::Months(1), 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ymd(2024, 7, 30));
    }

    #[test]
    fn quarterly_step_from_quarter_parse() {
        // Frequency::parse("1q") should produce Months(3)
        let freq = Frequency::parse("1q").unwrap();
        assert_eq!(freq, Frequency::Months(3));
        let last = ymd(2024, 3, 31);
        let result = generate_future_timestamps(&last, &freq, 4);
        assert_eq!(result[0], ymd(2024, 6, 30)); // Jun has 30 days
        assert_eq!(result[1], ymd(2024, 9, 30)); // Sep has 30 days
        assert_eq!(result[2], ymd(2024, 12, 30)); // clamped from Sep 30
        assert_eq!(result[3], ymd(2025, 3, 30)); // clamped from Dec 30
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use chrono::TimeZone;

    fn make_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i as u32, 0, 0).unwrap())
            .collect()
    }

    #[test]
    fn time_series_json_round_trip() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

        let json = ts.to_json().unwrap();
        let restored = TimeSeries::from_json(&json).unwrap();

        assert_eq!(restored.len(), 5);
        assert_eq!(restored.dimensions(), 1);
        assert_eq!(restored.primary_values(), &values);
        assert_eq!(restored.timestamps(), &timestamps);
    }

    #[test]
    fn time_series_json_round_trip_with_frequency() {
        let timestamps = make_timestamps(3);
        let values = vec![10.0, 20.0, 30.0];
        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_frequency(Duration::hours(1));

        let json = ts.to_json().unwrap();
        let restored = TimeSeries::from_json(&json).unwrap();

        assert_eq!(restored.frequency(), Some(Duration::hours(1)));
    }

    #[test]
    fn time_series_json_round_trip_with_metadata() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
        ts.set_metadata("source".to_string(), "test".to_string());

        let json = ts.to_json().unwrap();
        let restored = TimeSeries::from_json(&json).unwrap();

        assert_eq!(restored.metadata().get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn time_series_from_json_rejects_invalid_json() {
        let result = TimeSeries::from_json("{invalid}");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ForecastError::SerializationError(_)),
            "expected SerializationError, got {:?}",
            err
        );
    }

    #[test]
    fn time_series_from_json_rejects_wrong_structure() {
        let result = TimeSeries::from_json(r#"{"point": [[1.0, 2.0]]}"#);
        assert!(result.is_err());
    }

    // ── Seasonal/trend strength tests ─────────────────────────────────

    #[test]
    fn seasonal_strength_detects_strong_seasonality() {
        let n = 120;
        let period = 12;
        let timestamps = make_daily_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                seasonal + 0.1 * i as f64 // weak trend
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let strength = ts.seasonal_strength(period).unwrap();
        assert!(
            strength > 0.5,
            "expected strong seasonality, got {}",
            strength
        );
    }

    #[test]
    fn seasonal_strength_weak_for_trend_only() {
        let n = 120;
        let period = 12;
        let timestamps = make_daily_timestamps(n);
        let values: Vec<f64> = (0..n).map(|i| 5.0 * i as f64).collect(); // pure trend
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let strength = ts.seasonal_strength(period).unwrap();
        assert!(
            strength < 0.5,
            "expected weak seasonality for trend-only, got {}",
            strength
        );
    }

    #[test]
    fn trend_strength_detects_strong_trend() {
        let n = 120;
        let period = 12;
        let timestamps = make_daily_timestamps(n);
        let values: Vec<f64> = (0..n).map(|i| 3.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let strength = ts.trend_strength(period).unwrap();
        assert!(strength > 0.8, "expected strong trend, got {}", strength);
    }

    #[test]
    fn seasonal_strength_insufficient_data() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let err = ts.seasonal_strength(12).unwrap_err();
        assert!(matches!(err, ForecastError::InsufficientData { .. }));
    }

    #[test]
    fn seasonal_strength_invalid_period() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(ts.seasonal_strength(1).is_err());
        assert!(ts.seasonal_strength(0).is_err());
    }

    fn make_daily_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
            .collect()
    }

    // ── Outlier replacement tests ─────────────────────────────────────

    #[test]
    fn with_outliers_replaced_fixes_outlier() {
        let n = 50;
        let timestamps = make_daily_timestamps(n);
        let mut values: Vec<f64> = vec![10.0; n];
        values[25] = 1000.0; // inject outlier

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let config = crate::detection::OutlierConfig::iqr(1.5);
        let clean = ts.with_outliers_replaced(&config, 5).unwrap();

        // The outlier should be replaced with a neighbor median (~10.0)
        assert!(
            (clean.primary_values()[25] - 10.0).abs() < 1.0,
            "expected ~10.0, got {}",
            clean.primary_values()[25]
        );
    }

    #[test]
    fn with_outliers_replaced_no_change_when_clean() {
        let n = 50;
        let timestamps = make_daily_timestamps(n);
        let values: Vec<f64> = (0..n).map(|i| 10.0 + 0.01 * i as f64).collect();

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let config = crate::detection::OutlierConfig::z_score(3.0);
        let clean = ts.with_outliers_replaced(&config, 5).unwrap();

        // Should be identical when no outliers
        assert_eq!(clean.primary_values(), ts.primary_values());
    }

    #[test]
    fn with_outliers_replaced_multiple_outliers() {
        let n = 100;
        let timestamps = make_daily_timestamps(n);
        let mut values: Vec<f64> = vec![10.0; n];
        values[10] = 500.0;
        values[50] = -500.0;
        values[90] = 1000.0;

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let config = crate::detection::OutlierConfig::default();
        let clean = ts.with_outliers_replaced(&config, 7).unwrap();

        // All outliers should be close to 10.0
        assert!((clean.primary_values()[10] - 10.0).abs() < 1.0);
        assert!((clean.primary_values()[50] - 10.0).abs() < 1.0);
        assert!((clean.primary_values()[90] - 10.0).abs() < 1.0);
    }

    #[test]
    fn with_outliers_replaced_preserves_length() {
        let n = 50;
        let timestamps = make_daily_timestamps(n);
        let mut values: Vec<f64> = vec![10.0; n];
        values[25] = 1000.0;

        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let config = crate::detection::OutlierConfig::default();
        let clean = ts.with_outliers_replaced(&config, 5).unwrap();

        assert_eq!(clean.len(), ts.len());
    }
}
