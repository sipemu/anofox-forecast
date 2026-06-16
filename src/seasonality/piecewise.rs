//! Piecewise linear trend modeling.
//!
//! Models trend as piecewise linear segments joined at changepoints detected
//! by the PELT algorithm with a `LinearTrend` cost function.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::piecewise::{PiecewiseLinearTrend, piecewise_trend_features};
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! let values: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 1.0).collect();
//! let mut trend = PiecewiseLinearTrend::new();
//! trend.fit_trend(&values).unwrap();
//!
//! let fitted = trend.fitted_trend();
//! assert_eq!(fitted.len(), 50);
//!
//! let forecast = trend.predict_trend(10);
//! assert_eq!(forecast.len(), 10);
//!
//! let features = piecewise_trend_features(&values);
//! assert!(!features.is_empty());
//! ```

use super::traits::{Recency, TrendComponent};
use crate::changepoint::{CostFunction, Pelt};
use crate::error::{ForecastError, Result};

/// A single linear segment within the piecewise trend.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
    /// Slope of the linear fit within this segment.
    pub slope: f64,
    /// Intercept of the linear fit within this segment (at x = start).
    pub intercept: f64,
}

/// Piecewise linear trend component.
///
/// Uses PELT changepoint detection with `LinearTrend` cost to partition the
/// series into segments, then fits ordinary least squares within each segment.
#[derive(Debug, Clone)]
pub struct PiecewiseLinearTrend {
    /// Minimum segment length for changepoint detection.
    min_segment_length: usize,
    /// Penalty parameter for PELT (higher = fewer changepoints).
    /// Ignored when `auto_penalty` is true.
    penalty: f64,
    /// When true, the penalty is chosen per-series via CROPS + elbow
    /// (`Pelt::auto_detect`). The manual `penalty` field is ignored.
    auto_penalty: bool,
    /// Recency window for fitting.
    recency: Recency,
    /// Fitted segments: linear regression per segment.
    segments: Option<Vec<Segment>>,
    /// Fitted trend values.
    fitted: Vec<f64>,
    /// Length of training data.
    n_train: usize,
}

impl PiecewiseLinearTrend {
    /// Create a new piecewise linear trend with default parameters.
    ///
    /// Defaults: penalty = 10.0, min_segment_length = 5, auto_penalty = false.
    pub fn new() -> Self {
        Self {
            min_segment_length: 5,
            penalty: 10.0,
            auto_penalty: false,
            recency: Recency::Full,
            segments: None,
            fitted: Vec::new(),
            n_train: 0,
        }
    }

    /// Set the penalty parameter (builder-style). Higher values produce fewer
    /// changepoints.
    ///
    /// Has no effect when [`Self::with_auto_penalty`] is also set.
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// Enable per-series automatic penalty selection (builder-style).
    ///
    /// At fit time, runs PELT over a geometric range of penalties via
    /// CROPS (Haynes et al. 2017) and picks the *elbow* where adding one
    /// more knot stops paying for itself. Removes the need to hand-tune
    /// the penalty for varying series shapes — useful for batch / global
    /// pipelines and as the [`super::auto_trend::AutoTrend`] candidate.
    pub fn with_auto_penalty(mut self) -> Self {
        self.auto_penalty = true;
        self
    }

    /// Set the minimum segment length (builder-style).
    pub fn with_min_segment_length(mut self, len: usize) -> Self {
        self.min_segment_length = len;
        self
    }

    /// Set the recency window for fitting (builder-style).
    pub fn with_recency(mut self, recency: Recency) -> Self {
        self.recency = recency;
        self
    }

    /// Return a reference to the fitted segments, if available.
    pub fn segments(&self) -> Option<&[Segment]> {
        self.segments.as_deref()
    }
}

impl Default for PiecewiseLinearTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Fit simple OLS linear regression on indices `start..end` of `values`.
///
/// The regression uses absolute indices as the x-variable so that the
/// intercept and slope are expressed in terms of the original index space.
/// This makes extrapolation from the last segment straightforward.
fn fit_segment_ols(values: &[f64], start: usize, end: usize) -> Segment {
    let n = end - start;
    debug_assert!(n > 0);

    if n == 1 {
        return Segment {
            start,
            end,
            slope: 0.0,
            intercept: values[start],
        };
    }

    let n_f = n as f64;

    // x values are the absolute indices: start, start+1, ..., end-1
    // sum_x  = sum_{i=start}^{end-1} i = n*start + n*(n-1)/2
    let sum_x = n_f * start as f64 + n_f * (n_f - 1.0) / 2.0;
    let sum_x2 = {
        // sum of i^2 from start..end-1
        // = sum_{i=0}^{end-1} i^2 - sum_{i=0}^{start-1} i^2
        let s = |k: usize| {
            let k = k as f64;
            k * (k - 1.0) * (2.0 * k - 1.0) / 6.0
        };
        s(end) - s(start)
    };
    let sum_y: f64 = values[start..end].iter().sum();
    let sum_xy: f64 = values[start..end]
        .iter()
        .enumerate()
        .map(|(j, &y)| (start + j) as f64 * y)
        .sum();

    let ss_xx = sum_x2 - sum_x * sum_x / n_f;
    let ss_xy = sum_xy - sum_x * sum_y / n_f;

    let slope = if ss_xx.abs() < 1e-12 {
        0.0
    } else {
        ss_xy / ss_xx
    };
    let intercept = (sum_y - slope * sum_x) / n_f;

    Segment {
        start,
        end,
        slope,
        intercept,
    }
}

/// Evaluate a segment's linear function at a given index.
#[inline]
fn segment_value(seg: &Segment, idx: usize) -> f64 {
    seg.intercept + seg.slope * idx as f64
}

impl TrendComponent for PiecewiseLinearTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = values.len();

        if n < 2 {
            // Single point: trivial fit.
            self.segments = Some(vec![Segment {
                start: 0,
                end: 1,
                slope: 0.0,
                intercept: values[0],
            }]);
            self.fitted = vec![values[0]];
            self.n_train = 1;
            return Ok(());
        }

        let (rec_start, rec_end) = self.recency.resolve_with_data(values);
        let window = &values[rec_start..rec_end];

        // Run PELT changepoint detection on the recency window. When
        // auto_penalty is set, sweep via CROPS + elbow; otherwise use
        // the configured fixed penalty.
        let pelt = Pelt::new(CostFunction::LinearTrend).min_size(self.min_segment_length);
        let result = if self.auto_penalty {
            pelt.auto_detect(window).result
        } else {
            pelt.penalty(self.penalty).detect(window)
        };

        // Fit OLS per segment (indices are relative to window, shift to absolute).
        let segments: Vec<Segment> = result
            .segments
            .iter()
            .map(|&(s, e)| fit_segment_ols(values, s + rec_start, e + rec_start))
            .collect();

        // Compute fitted values — extrapolate backwards for pre-window portion.
        let mut fitted = vec![0.0; n];

        // Fill the recency window and beyond from segments.
        for seg in &segments {
            for i in seg.start..seg.end {
                fitted[i] = segment_value(seg, i);
            }
        }

        // Backwards-extrapolate: use the first segment's model for indices before rec_start.
        if rec_start > 0 {
            if let Some(first_seg) = segments.first() {
                for i in 0..rec_start {
                    fitted[i] = segment_value(first_seg, i);
                }
            }
        }

        self.segments = Some(segments);
        self.fitted = fitted;
        self.n_train = n;

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        let segments = match &self.segments {
            Some(s) => s,
            None => return vec![f64::NAN; n_ahead],
        };

        if segments.is_empty() {
            return vec![f64::NAN; n_ahead];
        }

        let last = &segments[segments.len() - 1];
        (0..n_ahead)
            .map(|i| segment_value(last, self.n_train + i))
            .collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        let segments = match &self.segments {
            Some(s) => s,
            None => return Vec::new(),
        };

        let n_segments = segments.len() as f64;

        let mean_slope = if segments.is_empty() {
            0.0
        } else {
            segments.iter().map(|s| s.slope).sum::<f64>() / n_segments
        };

        let max_slope_change = if segments.len() < 2 {
            0.0
        } else {
            segments
                .windows(2)
                .map(|w| (w[1].slope - w[0].slope).abs())
                .fold(0.0_f64, f64::max)
        };

        let r_squared = {
            let n = self.fitted.len();
            if n < 2 {
                1.0
            } else {
                // We need the original values to compute R².
                // Since we only store fitted values, we compute R² of fitted
                // vs the data that was used during fit. We don't store the
                // original data, so we estimate using the residual information
                // from segments. Instead, we compute SS_res from segments and
                // SS_tot from fitted values' spread (which approximates the
                // original data's spread when fit is good).
                //
                // Actually, we need the original data. We'll store nothing
                // extra and instead compute R² at fit time if we had the data.
                // For now, return NaN and compute it properly below.
                f64::NAN
            }
        };

        let last_slope = segments.last().map(|s| s.slope).unwrap_or(0.0);

        let mut features = vec![
            ("piecewise_n_segments", n_segments),
            ("piecewise_mean_slope", mean_slope),
            ("piecewise_max_slope_change", max_slope_change),
            ("piecewise_r_squared", r_squared),
            ("piecewise_last_slope", last_slope),
        ];

        // R² can't be computed without original data in the trait method,
        // so we leave it as NaN here. The standalone functions compute it
        // properly since they have access to the original values.
        let _ = &mut features;

        features
    }

    fn trend_name(&self) -> &str {
        "piecewise_linear"
    }

    fn n_params(&self) -> usize {
        // 2 params (slope + intercept) per segment
        self.segments.as_ref().map_or(0, |s| 2 * s.len())
    }
}

/// Compute R-squared of fitted values against original values.
fn compute_r_squared(values: &[f64], fitted: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 1.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
    let ss_res: f64 = values
        .iter()
        .zip(fitted.iter())
        .map(|(&v, &f)| (v - f).powi(2))
        .sum();

    if ss_tot < 1e-12 {
        // Constant series: if residuals are also zero, perfect fit.
        if ss_res < 1e-12 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Fit a piecewise linear trend with default parameters and return all features.
///
/// This convenience function creates a `PiecewiseLinearTrend` with default
/// settings, fits it to `values`, and returns the full set of named features
/// including a properly computed R-squared.
pub fn piecewise_trend_features(values: &[f64]) -> Vec<(&str, f64)> {
    let mut trend = PiecewiseLinearTrend::new();
    if trend.fit_trend(values).is_err() {
        return Vec::new();
    }

    let r_squared = compute_r_squared(values, trend.fitted_trend());

    let segments = match trend.segments() {
        Some(s) => s,
        None => return Vec::new(),
    };

    let n_segments = segments.len() as f64;

    let mean_slope = if segments.is_empty() {
        0.0
    } else {
        segments.iter().map(|s| s.slope).sum::<f64>() / n_segments
    };

    let max_slope_change = if segments.len() < 2 {
        0.0
    } else {
        segments
            .windows(2)
            .map(|w| (w[1].slope - w[0].slope).abs())
            .fold(0.0_f64, f64::max)
    };

    let last_slope = segments.last().map(|s| s.slope).unwrap_or(0.0);

    vec![
        ("piecewise_n_segments", n_segments),
        ("piecewise_mean_slope", mean_slope),
        ("piecewise_max_slope_change", max_slope_change),
        ("piecewise_r_squared", r_squared),
        ("piecewise_last_slope", last_slope),
    ]
}

/// Return the number of piecewise linear segments detected (with default settings).
pub fn piecewise_n_segments(values: &[f64]) -> f64 {
    let mut trend = PiecewiseLinearTrend::new();
    if trend.fit_trend(values).is_err() {
        return f64::NAN;
    }
    match trend.segments() {
        Some(s) => s.len() as f64,
        None => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn assert_approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} ~ {}, diff = {}",
            a,
            b,
            (a - b).abs()
        );
    }

    // ── Linear data: single segment ────────────────────────────────────

    #[test]
    fn linear_data_one_segment() {
        // y = 2*x + 1, should be a single segment
        let values: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&values).unwrap();

        let segs = trend.segments().unwrap();
        assert_eq!(segs.len(), 1, "pure linear data should yield 1 segment");
        assert_approx(segs[0].slope, 2.0, 1e-8);
        assert_approx(segs[0].intercept, 1.0, 1e-8);
    }

    #[test]
    fn linear_data_fitted_values() {
        let values: Vec<f64> = (0..30).map(|i| 3.0 * i as f64 - 5.0).collect();
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&values).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 30);
        for (i, (&f, &v)) in fitted.iter().zip(values.iter()).enumerate() {
            assert_approx(f, v, 1e-8);
            let _ = i;
        }
    }

    // ── Data with changepoint: two segments ────────────────────────────

    #[test]
    fn two_segments_detected() {
        // First half: slope +1, second half: slope -1
        let mut values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        values.extend((0..50).map(|i| 49.0 - i as f64));

        let mut trend = PiecewiseLinearTrend::new()
            .with_penalty(5.0)
            .with_min_segment_length(5);
        trend.fit_trend(&values).unwrap();

        let segs = trend.segments().unwrap();
        assert!(
            segs.len() >= 2,
            "should detect at least 2 segments, got {}",
            segs.len()
        );

        // First segment should have positive slope, last should have negative.
        assert!(
            segs[0].slope > 0.5,
            "first segment slope should be positive"
        );
        assert!(
            segs.last().unwrap().slope < -0.5,
            "last segment slope should be negative"
        );
    }

    // ── Predict extrapolates correctly ─────────────────────────────────

    #[test]
    fn predict_extrapolates_from_last_segment() {
        // y = 2*x + 1
        let values: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);

        // The next values should continue the linear trend: y(20)=41, y(21)=43, ...
        for (j, &f) in forecast.iter().enumerate() {
            let expected = 2.0 * (20 + j) as f64 + 1.0;
            assert_approx(f, expected, 1e-6);
        }
    }

    #[test]
    fn predict_unfitted_returns_nan() {
        let trend = PiecewiseLinearTrend::new();
        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        assert!(forecast[0].is_nan());
    }

    // ── Feature extraction ─────────────────────────────────────────────

    #[test]
    fn features_linear_data() {
        let values: Vec<f64> = (0..40).map(|i| 5.0 * i as f64).collect();
        let features = piecewise_trend_features(&values);

        let get = |name: &str| -> f64 {
            features
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("feature {} not found", name))
        };

        assert_approx(get("piecewise_n_segments"), 1.0, TOL);
        assert_approx(get("piecewise_mean_slope"), 5.0, 1e-8);
        assert_approx(get("piecewise_max_slope_change"), 0.0, TOL);
        assert!(
            get("piecewise_r_squared") > 0.999,
            "R² should be ~1 for linear data, got {}",
            get("piecewise_r_squared")
        );
        assert_approx(get("piecewise_last_slope"), 5.0, 1e-8);
    }

    #[test]
    fn features_two_slopes() {
        let mut values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        values.extend((0..50).map(|i| 49.0 - i as f64));

        let features = piecewise_trend_features(&values);
        assert!(!features.is_empty());

        let get = |name: &str| -> f64 {
            features
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap()
        };

        assert!(
            get("piecewise_n_segments") >= 2.0,
            "should have >= 2 segments"
        );
        assert!(
            get("piecewise_max_slope_change") > 0.5,
            "should detect slope change"
        );
    }

    // ── Standalone functions ───────────────────────────────────────────

    #[test]
    fn standalone_n_segments_linear() {
        let values: Vec<f64> = (0..30).map(|i| i as f64 * 3.0).collect();
        let n = piecewise_n_segments(&values);
        assert_approx(n, 1.0, TOL);
    }

    #[test]
    fn standalone_n_segments_empty() {
        let n = piecewise_n_segments(&[]);
        assert!(n.is_nan());
    }

    #[test]
    fn standalone_features_empty() {
        let features = piecewise_trend_features(&[]);
        assert!(features.is_empty());
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn fit_empty_data_error() {
        let mut trend = PiecewiseLinearTrend::new();
        let result = trend.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn fit_single_point() {
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&[42.0]).unwrap();

        let segs = trend.segments().unwrap();
        assert_eq!(segs.len(), 1);
        assert_approx(segs[0].intercept, 42.0, TOL);
        assert_approx(segs[0].slope, 0.0, TOL);

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 1);
        assert_approx(fitted[0], 42.0, TOL);

        let forecast = trend.predict_trend(3);
        assert_eq!(forecast.len(), 3);
        for &f in &forecast {
            assert_approx(f, 42.0, TOL);
        }
    }

    // ── TrendComponent trait ───────────────────────────────────────────

    #[test]
    fn trend_name_is_correct() {
        let trend = PiecewiseLinearTrend::new();
        assert_eq!(trend.trend_name(), "piecewise_linear");
    }

    #[test]
    fn trend_features_before_fit_empty() {
        let trend = PiecewiseLinearTrend::new();
        let features = trend.trend_features();
        assert!(features.is_empty());
    }

    // ── Builder pattern ────────────────────────────────────────────────

    #[test]
    fn builder_penalty() {
        let trend = PiecewiseLinearTrend::new().with_penalty(25.0);
        assert_approx(trend.penalty, 25.0, TOL);
    }

    #[test]
    fn builder_min_segment_length() {
        let trend = PiecewiseLinearTrend::new().with_min_segment_length(10);
        assert_eq!(trend.min_segment_length, 10);
    }

    // ── R-squared computation ──────────────────────────────────────────

    #[test]
    fn r_squared_perfect_fit() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fitted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = compute_r_squared(&values, &fitted);
        assert_approx(r2, 1.0, TOL);
    }

    #[test]
    fn r_squared_constant_series() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let fitted = vec![5.0, 5.0, 5.0, 5.0];
        let r2 = compute_r_squared(&values, &fitted);
        assert_approx(r2, 1.0, TOL);
    }

    #[test]
    fn r_squared_poor_fit() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fitted = vec![3.0, 3.0, 3.0, 3.0, 3.0]; // mean predictor -> R² = 0
        let r2 = compute_r_squared(&values, &fitted);
        assert_approx(r2, 0.0, TOL);
    }

    // ── Default trait ──────────────────────────────────────────────────

    #[test]
    fn default_same_as_new() {
        let a = PiecewiseLinearTrend::new();
        let b = PiecewiseLinearTrend::default();
        assert_approx(a.penalty, b.penalty, TOL);
        assert_eq!(a.min_segment_length, b.min_segment_length);
    }

    // ── Predict zero ahead ─────────────────────────────────────────────

    #[test]
    fn predict_zero_ahead() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(0);
        assert!(forecast.is_empty());
    }

    // ── Auto-penalty (CROPS + elbow) ───────────────────────────────────

    #[test]
    fn auto_penalty_recovers_known_slope_change() {
        // y = 0.5·i for i in 0..40, then 0.5·40 + 2.0·(i − 40) for
        // i in 40..80. One slope change at index 40; auto_penalty
        // should land a knot near 40 and recover both slopes within
        // tolerance, without us having to guess a penalty.
        let n_first = 40;
        let slope_first = 0.5;
        let n_total = 80;
        let slope_second = 2.0;
        let break_at = n_first;
        let level_at_break = slope_first * n_first as f64;
        let values: Vec<f64> = (0..n_total)
            .map(|i| {
                if i < n_first {
                    slope_first * i as f64
                } else {
                    level_at_break + slope_second * (i - n_first) as f64
                }
            })
            .collect();

        let mut trend = PiecewiseLinearTrend::new().with_auto_penalty();
        trend.fit_trend(&values).unwrap();
        let segs = trend.segments().expect("segments after fit");

        assert!(
            segs.len() >= 2,
            "auto_penalty should detect at least one knot on a clear slope change; got {} segments",
            segs.len()
        );

        // Locate the segment containing the constructed break and
        // confirm its boundary is within ±3 of `break_at`. (±3 absorbs
        // the elbow heuristic's small tolerance plus the minimum
        // segment length.)
        let knot_at = segs[1].start as isize;
        assert!(
            (knot_at - break_at as isize).abs() <= 3,
            "knot landed at {}, expected near {}",
            knot_at,
            break_at,
        );

        // First and last segment slopes should recover the constructed
        // values within tolerance (allow some slack for the knot being
        // a few indices off).
        let first = &segs[0];
        let last = segs.last().unwrap();
        assert!(
            (first.slope - slope_first).abs() < 0.1,
            "first slope {} should be near {}",
            first.slope,
            slope_first,
        );
        assert!(
            (last.slope - slope_second).abs() < 0.1,
            "last slope {} should be near {}",
            last.slope,
            slope_second,
        );
    }

    #[test]
    fn auto_penalty_default_is_off() {
        // Backwards compat: a default PiecewiseLinearTrend uses the
        // manual penalty path; flipping auto_penalty must be explicit.
        let manual = PiecewiseLinearTrend::new();
        let auto = PiecewiseLinearTrend::new().with_auto_penalty();
        assert!(!manual.auto_penalty);
        assert!(auto.auto_penalty);
    }

    // ── Constant data ──────────────────────────────────────────────────

    #[test]
    fn constant_data_single_segment() {
        let values = vec![7.0; 30];
        let mut trend = PiecewiseLinearTrend::new();
        trend.fit_trend(&values).unwrap();

        let segs = trend.segments().unwrap();
        assert_eq!(segs.len(), 1);
        assert_approx(segs[0].slope, 0.0, TOL);
        assert_approx(segs[0].intercept, 7.0, TOL);

        // Forecast should be constant.
        let forecast = trend.predict_trend(5);
        for &f in &forecast {
            assert_approx(f, 7.0, TOL);
        }
    }
}
