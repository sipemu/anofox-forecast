//! Convenience functions for common STL decomposition operations.
//!
//! These functions provide one-call access to individual decomposition components
//! without needing to manually construct an [`STL`] instance and call `decompose`.

use super::stl::{STLResult, STL};
use crate::core::TimeSeries;

/// Remove the seasonal component and return trend + remainder.
///
/// Returns `None` if the series is too short for decomposition (needs `>= 2 * period`).
///
/// # Example
///
/// ```
/// use anofox_forecast::seasonality::convenience::deseasonalize;
///
/// let series: Vec<f64> = (0..120)
///     .map(|i| {
///         0.1 * i as f64
///             + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
///     })
///     .collect();
/// let deseas = deseasonalize(&series, 12).unwrap();
/// assert_eq!(deseas.len(), series.len());
/// ```
pub fn deseasonalize(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let result = STL::new(period).decompose(data)?;
    Some(result.deseasonalized())
}

/// Remove the trend component and return seasonal + remainder.
///
/// Returns `None` if the series is too short for decomposition.
pub fn detrend(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let result = STL::new(period).decompose(data)?;
    Some(result.detrended())
}

/// Extract only the seasonal component.
///
/// Returns `None` if the series is too short for decomposition.
pub fn seasonal_component(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let result = STL::new(period).decompose(data)?;
    Some(result.seasonal)
}

/// Extract only the trend component.
///
/// Returns `None` if the series is too short for decomposition.
pub fn trend_component(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let result = STL::new(period).decompose(data)?;
    Some(result.trend)
}

/// Extract only the remainder component.
///
/// Returns `None` if the series is too short for decomposition.
pub fn remainder_component(data: &[f64], period: usize) -> Option<Vec<f64>> {
    let result = STL::new(period).decompose(data)?;
    Some(result.remainder)
}

/// Reconstruct a series from its trend, seasonal, and remainder components.
///
/// This is the inverse of STL decomposition: `result[i] = trend[i] + seasonal[i] + remainder[i]`.
///
/// # Panics
///
/// Panics if the three slices have different lengths.
pub fn recompose(trend: &[f64], seasonal: &[f64], remainder: &[f64]) -> Vec<f64> {
    assert_eq!(
        trend.len(),
        seasonal.len(),
        "trend and seasonal must have the same length"
    );
    assert_eq!(
        trend.len(),
        remainder.len(),
        "trend and remainder must have the same length"
    );
    trend
        .iter()
        .zip(seasonal.iter())
        .zip(remainder.iter())
        .map(|((t, s), r)| t + s + r)
        .collect()
}

/// Return a new [`TimeSeries`] with the seasonal component removed, preserving timestamps.
///
/// The returned series contains trend + remainder from the STL decomposition
/// of the primary (first) dimension.
///
/// Returns `None` if decomposition fails (e.g., series too short).
pub fn seasonal_adjust(ts: &TimeSeries, period: usize) -> Option<TimeSeries> {
    let vals = ts.primary_values();
    let result = STL::new(period).decompose(vals)?;
    let adjusted = result.deseasonalized();

    // Build a new TimeSeries preserving timestamps and metadata.
    TimeSeries::univariate(ts.timestamps().to_vec(), adjusted).ok()
}

// ── STLResult convenience methods ────────────────────────────────────

impl STLResult {
    /// Return trend + remainder (the series with seasonality removed).
    pub fn deseasonalized(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(self.remainder.iter())
            .map(|(t, r)| t + r)
            .collect()
    }

    /// Return seasonal + remainder (the series with the trend removed).
    pub fn detrended(&self) -> Vec<f64> {
        self.seasonal
            .iter()
            .zip(self.remainder.iter())
            .map(|(s, r)| s + r)
            .collect()
    }

    /// Reconstruct the original series: trend + seasonal + remainder.
    ///
    /// Useful for verifying round-trip accuracy of decomposition.
    pub fn recompose(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(self.seasonal.iter())
            .zip(self.remainder.iter())
            .map(|((t, s), r)| t + s + r)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    /// Generate a synthetic series with trend + seasonal + small noise-like component.
    fn generate_seasonal_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                trend + seasonal
            })
            .collect()
    }

    fn make_daily_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
            .collect()
    }

    // ── Decompose + recompose round-trip ─────────────────────────────

    #[test]
    fn stl_recompose_round_trip() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let result = STL::new(period).decompose(&series).unwrap();
        let reconstructed = result.recompose();

        for (i, (&orig, &rec)) in series.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "round-trip mismatch at index {}: {} vs {}",
                i,
                orig,
                rec,
            );
        }
    }

    #[test]
    fn stl_result_deseasonalized() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let result = STL::new(period).decompose(&series).unwrap();
        let deseas = result.deseasonalized();

        for (i, (&d, (&t, &r))) in deseas
            .iter()
            .zip(result.trend.iter().zip(result.remainder.iter()))
            .enumerate()
        {
            assert!(
                (d - (t + r)).abs() < 1e-10,
                "deseasonalized mismatch at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_result_detrended() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let result = STL::new(period).decompose(&series).unwrap();
        let detr = result.detrended();

        for (i, (&d, (&s, &r))) in detr
            .iter()
            .zip(result.seasonal.iter().zip(result.remainder.iter()))
            .enumerate()
        {
            assert!(
                (d - (s + r)).abs() < 1e-10,
                "detrended mismatch at index {}",
                i,
            );
        }
    }

    // ── Deseasonalized has lower seasonal strength ───────────────────

    #[test]
    fn stl_deseasonalized_has_lower_seasonal_strength() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let original_result = STL::new(period).decompose(&series).unwrap();
        let original_strength = original_result.seasonal_strength();

        let deseas = deseasonalize(&series, period).unwrap();
        let deseas_result = STL::new(period).decompose(&deseas).unwrap();
        let deseas_strength = deseas_result.seasonal_strength();

        assert!(
            deseas_strength < original_strength,
            "deseasonalized strength ({}) should be less than original ({})",
            deseas_strength,
            original_strength,
        );
    }

    // ── Seasonal component has correct period ────────────────────────

    #[test]
    fn stl_seasonal_component_has_correct_period() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let seasonal = seasonal_component(&series, period).unwrap();

        // The seasonal component should approximately repeat every `period` steps.
        // Check that values separated by one full period are close.
        for i in 0..(seasonal.len() - period) {
            let diff = (seasonal[i] - seasonal[i + period]).abs();
            assert!(
                diff < 2.0,
                "seasonal should repeat with period {}: index {} diff = {}",
                period,
                i,
                diff,
            );
        }
    }

    // ── seasonal_adjust preserves timestamps ─────────────────────────

    #[test]
    fn stl_seasonal_adjust_preserves_timestamps() {
        let n = 120;
        let period = 12;
        let timestamps = make_daily_timestamps(n);
        let values = generate_seasonal_series(n, period);

        let ts = TimeSeries::univariate(timestamps.clone(), values).unwrap();
        let adjusted = seasonal_adjust(&ts, period).unwrap();

        assert_eq!(adjusted.len(), ts.len());
        assert_eq!(adjusted.timestamps(), ts.timestamps());
    }

    // ── Free-function convenience wrappers ───────────────────────────

    #[test]
    fn stl_deseasonalize_matches_result_method() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let via_fn = deseasonalize(&series, period).unwrap();

        let result = STL::new(period).decompose(&series).unwrap();
        let via_method = result.deseasonalized();

        for (i, (&a, &b)) in via_fn.iter().zip(via_method.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "deseasonalize fn vs method mismatch at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_detrend_matches_result_method() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let via_fn = detrend(&series, period).unwrap();

        let result = STL::new(period).decompose(&series).unwrap();
        let via_method = result.detrended();

        for (i, (&a, &b)) in via_fn.iter().zip(via_method.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "detrend fn vs method mismatch at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_free_fn_recompose() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let result = STL::new(period).decompose(&series).unwrap();

        let reconstructed = recompose(&result.trend, &result.seasonal, &result.remainder);

        for (i, (&orig, &rec)) in series.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "free-fn recompose mismatch at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_trend_component_fn() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let trend = trend_component(&series, period).unwrap();
        let result = STL::new(period).decompose(&series).unwrap();

        assert_eq!(trend, result.trend);
    }

    #[test]
    fn stl_remainder_component_fn() {
        let period = 12;
        let series = generate_seasonal_series(120, period);
        let rem = remainder_component(&series, period).unwrap();
        let result = STL::new(period).decompose(&series).unwrap();

        assert_eq!(rem, result.remainder);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn stl_convenience_constant_series() {
        let period = 10;
        let series = vec![42.0; 100];
        let result = STL::new(period).decompose(&series).unwrap();

        // All seasonal and remainder values should be near zero for a constant series.
        for &s in &result.seasonal {
            assert!(s.abs() < 1e-6, "seasonal should be ~0 for constant series");
        }
        for &r in &result.remainder {
            assert!(r.abs() < 1e-6, "remainder should be ~0 for constant series");
        }

        // Recompose should still match.
        let reconstructed = result.recompose();
        for (i, &rec) in reconstructed.iter().enumerate() {
            assert!(
                (rec - 42.0).abs() < 1e-6,
                "recompose of constant series failed at index {}",
                i,
            );
        }

        // Deseasonalized should equal original for constant series.
        let deseas = deseasonalize(&series, period).unwrap();
        for (i, &d) in deseas.iter().enumerate() {
            assert!(
                (d - 42.0).abs() < 1e-6,
                "deseasonalize of constant series failed at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_convenience_very_short_series() {
        // Series shorter than 2 * period should return None.
        let period = 12;
        let series = vec![1.0; 10];

        assert!(deseasonalize(&series, period).is_none());
        assert!(detrend(&series, period).is_none());
        assert!(seasonal_component(&series, period).is_none());
        assert!(trend_component(&series, period).is_none());
        assert!(remainder_component(&series, period).is_none());
    }

    #[test]
    fn stl_convenience_minimum_length_series() {
        // Exactly 2 * period should succeed.
        let period = 7;
        let n = 2 * period;
        let series = generate_seasonal_series(n, period);

        assert!(deseasonalize(&series, period).is_some());
        assert!(detrend(&series, period).is_some());
        assert!(seasonal_component(&series, period).is_some());
        assert!(trend_component(&series, period).is_some());
        assert!(remainder_component(&series, period).is_some());
    }

    #[test]
    fn stl_seasonal_adjust_too_short() {
        let period = 12;
        let timestamps = make_daily_timestamps(10);
        let values = vec![1.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(seasonal_adjust(&ts, period).is_none());
    }

    #[test]
    #[should_panic(expected = "trend and seasonal must have the same length")]
    fn stl_recompose_panics_on_length_mismatch() {
        let _ = recompose(&[1.0, 2.0], &[1.0], &[1.0, 2.0]);
    }
}
