---
name: timeseries
description: How to create and manipulate TimeSeries, handle missing data, and work with Forecast results
user_invocable: true
---

# TimeSeries and Forecast in anofox-forecast

## 1. Create a TimeSeries

```rust
use anofox_forecast::core::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

// Simple univariate
let timestamps: Vec<_> = (0..100)
    .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
    .collect();
let values: Vec<f64> = vec![1.0; 100];
let ts = TimeSeries::univariate(timestamps, values).unwrap();

// Access data
let n = ts.len();
let vals = ts.primary_values();    // &[f64]
let times = ts.timestamps();      // &[DateTime<Utc>]
let is_multi = ts.is_multivariate();
```

## 2. TimeSeriesBuilder (Full Control)

```rust
use anofox_forecast::core::{TimeSeries, Frequency, MissingValuePolicy};

let ts = TimeSeries::builder()
    .timestamps(timestamps)
    .values(vec![values_dim1, values_dim2])  // multivariate
    .labels(vec!["temperature".into(), "humidity".into()])
    .frequency(Frequency::parse("1d").unwrap())
    .missing_value_policy(MissingValuePolicy::ForwardFill)
    .build()
    .unwrap();
```

## 3. Calendar Annotations (Regressors & Holidays)

```rust
use anofox_forecast::core::CalendarAnnotations;

let cal = CalendarAnnotations::new()
    .with_regressor("temperature".to_string(), temp_values)
    .with_regressor("price".to_string(), price_values);

let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
ts.set_calendar(cal);

// Query regressors
assert!(ts.has_regressors());
let temp = ts.regressor("temperature").unwrap();
let all = ts.all_regressors();  // &HashMap<String, Vec<f64>>
```

## 4. Handle Missing Data

```rust
use anofox_forecast::core::MissingValuePolicy;

// Check for problems
let has_nans = ts.has_missing_values();
let has_gaps = ts.has_gaps();

// Sanitize (returns new TimeSeries)
let clean = ts.sanitized(MissingValuePolicy::Drop);            // Remove NaN rows
let clean = ts.sanitized(MissingValuePolicy::Fill(0.0));       // Fill with constant
let clean = ts.sanitized(MissingValuePolicy::ForwardFill);     // Last observation carried forward
let clean = ts.sanitized(MissingValuePolicy::BackwardFill);    // Next observation carried backward
let clean = ts.sanitized(MissingValuePolicy::FillMean);        // Fill with series mean
let clean = ts.sanitized(MissingValuePolicy::FillMedian);      // Fill with series median
let clean = ts.sanitized(MissingValuePolicy::Interpolate);     // Linear interpolation
let clean = ts.sanitized(MissingValuePolicy::Error);           // Return error if NaN found
```

## 5. Frequency Operations

```rust
use anofox_forecast::core::{Frequency, AggregationMethod, InterpolationMethod};

// Parse frequency strings (Polars-style)
let freq = Frequency::parse("30m").unwrap();  // 30 minutes
let freq = Frequency::parse("1h").unwrap();   // hourly
let freq = Frequency::parse("1d").unwrap();   // daily
let freq = Frequency::parse("1w").unwrap();   // weekly
let freq = Frequency::parse("1mo").unwrap();  // monthly

// Infer from data
let freq = ts.infer_frequency();

// Fill gaps at target frequency
let filled = ts.fill_gaps(Frequency::parse("1d").unwrap(), InterpolationMethod::Linear);

// Downsample (aggregate to lower frequency)
let weekly = ts.downsample(Frequency::parse("1w").unwrap(), AggregationMethod::Mean);

// Upsample (interpolate to higher frequency)
let hourly = ts.upsample(Frequency::parse("1h").unwrap(), InterpolationMethod::Linear);
```

## 6. Slicing

```rust
// Extract a sub-range (returns new TimeSeries)
let subset = ts.slice(10, 50);  // indices 10..50
```

## 7. Working with Forecast Results

```rust
use anofox_forecast::core::Forecast;

// After model.predict() or model.predict_with_intervals()
let fc: Forecast = model.predict_with_intervals(12, 0.95).unwrap();

// Point forecasts
let points: &[f64] = fc.primary();         // first dimension
let points: &[f64] = fc.series(0).unwrap(); // by dimension index

// Prediction intervals
if fc.has_lower() {
    let lower: &[f64] = fc.lower_series(0).unwrap();
    let upper: &[f64] = fc.upper_series(0).unwrap();
}

// Metadata
let h = fc.horizon();
let dims = fc.dimensions();
```

## 8. Forecast Constraints

```rust
use anofox_forecast::core::ForecastConstraint;

// Non-negative (e.g., demand, counts)
let fc = fc.non_negative();

// Clamp to range
let fc = fc.clamp(0.0, 1000.0);

// Round to integers (point rounded, lower floored, upper ceiled)
let fc = fc.round_to_integer();

// Chain multiple constraints
let fc = fc.constrain(&[
    ForecastConstraint::NonNegative,
    ForecastConstraint::UpperBound(500.0),
    ForecastConstraint::IntegerRound,
]);
```

## 9. Construct Forecasts Manually

```rust
// Point forecast only
let fc = Forecast::from_values(vec![10.0, 11.0, 12.0]);

// With intervals
let fc = Forecast::from_values_with_intervals(
    vec![10.0, 11.0, 12.0],  // point
    vec![8.0, 9.0, 10.0],    // lower
    vec![12.0, 13.0, 14.0],  // upper
);
```

## Key Rules

- `TimeSeries::univariate()` requires timestamps and values of equal length.
- `primary_values()` returns `&[f64]` for the first (or only) dimension.
- Regressors in `CalendarAnnotations` must have the same length as the time series.
- `slice(start, end)` uses half-open indexing: `[start, end)`.
- `Forecast::primary()` returns `&[f64]` of length `horizon`.
- Constraints are applied in order. `non_negative()` is a shortcut for `constrain(&[ForecastConstraint::NonNegative])`.
