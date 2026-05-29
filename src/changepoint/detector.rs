//! Core traits for trait-based changepoint detection.
//!
//! `Cost` and `Detector` mirror the abstractions used by the
//! [`ruptures`](https://github.com/deepcharles/ruptures) Python library.
//! Detectors implement `Detector` and use any cost that implements
//! `Cost`, so algorithms and cost functions cross-compose.

use crate::error::{ForecastError, Result};

use super::signal::Signal;

/// A segment-error cost function.
///
/// Implementations precompute any necessary state in [`Cost::fit`] (for
/// example cumulative sums) so that subsequent [`Cost::error`] calls
/// can be O(1) or close to it.
pub trait Cost: std::fmt::Debug {
    /// Bind the cost to a signal. Must be called before [`Cost::error`].
    fn fit(&mut self, signal: &Signal) -> Result<()>;

    /// Cost of the segment `[start, end)`. Caller must ensure
    /// `0 ≤ start < end ≤ n` and `end - start ≥ min_size()`.
    fn error(&self, start: usize, end: usize) -> Result<f64>;

    /// Minimum segment length the cost can score.
    fn min_size(&self) -> usize {
        1
    }

    /// Diagnostic name.
    fn name(&self) -> &str;
}

/// Result of a detector run.
///
/// `bkps` is a sorted list of segment-end indices (exclusive), matching
/// ruptures' convention: the last entry is always `n` (signal length).
///
/// A series of length 100 with one changepoint at position 50 has
/// `bkps = vec![50, 100]`, `n_changepoints() == 1`, and
/// `segments() == vec![(0, 50), (50, 100)]`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorResult {
    /// Segment-end indices (exclusive). The terminal `n` is included.
    pub bkps: Vec<usize>,
}

impl DetectorResult {
    /// Number of detected changepoints (excludes the terminal `n`).
    pub fn n_changepoints(&self) -> usize {
        self.bkps.len().saturating_sub(1)
    }

    /// Segment boundaries as `(start, end)` pairs.
    pub fn segments(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.bkps.len());
        let mut start = 0usize;
        for &b in &self.bkps {
            out.push((start, b));
            start = b;
        }
        out
    }

    /// Changepoint indices (everything in `bkps` except the terminal `n`).
    pub fn changepoints(&self) -> &[usize] {
        if self.bkps.is_empty() {
            &[]
        } else {
            &self.bkps[..self.bkps.len() - 1]
        }
    }
}

/// A changepoint-detection algorithm.
///
/// Detectors are fit to a signal once, then can be queried multiple
/// times in three modes:
///
/// - [`Detector::predict_pen`] — minimise `Σ C(s, t) + pen · m` (where
///   `m` is the number of changepoints).
/// - [`Detector::predict_n_bkps`] — find the segmentation with exactly
///   `n_bkps` changepoints.
/// - [`Detector::predict_eps`] — find the smallest `m` such that
///   `Σ C(s, t) ≤ epsilon`.
///
/// Not every algorithm supports every mode; default impls return
/// [`ForecastError::InvalidParameter`].
pub trait Detector {
    /// Fit the detector (and its internal cost) to a signal.
    fn fit(&mut self, signal: &Signal) -> Result<()>;

    /// Predict breakpoints under penalty `pen`.
    fn predict_pen(&self, pen: f64) -> Result<DetectorResult>;

    /// Predict breakpoints with a fixed target number of changepoints.
    ///
    /// Default impl returns an error.
    fn predict_n_bkps(&self, _n_bkps: usize) -> Result<DetectorResult> {
        Err(ForecastError::InvalidParameter(format!(
            "Detector '{}' does not support n_bkps mode",
            self.name()
        )))
    }

    /// Predict breakpoints with a maximum total error.
    ///
    /// Default impl returns an error.
    fn predict_eps(&self, _epsilon: f64) -> Result<DetectorResult> {
        Err(ForecastError::InvalidParameter(format!(
            "Detector '{}' does not support epsilon mode",
            self.name()
        )))
    }

    /// Convenience: fit then `predict_pen`.
    fn fit_predict_pen(&mut self, signal: &Signal, pen: f64) -> Result<DetectorResult> {
        self.fit(signal)?;
        self.predict_pen(pen)
    }

    /// Convenience: fit then `predict_n_bkps`.
    fn fit_predict_n_bkps(&mut self, signal: &Signal, n_bkps: usize) -> Result<DetectorResult> {
        self.fit(signal)?;
        self.predict_n_bkps(n_bkps)
    }

    /// Diagnostic name.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_result_segments_round_trip() {
        let r = DetectorResult {
            bkps: vec![30, 70, 100],
        };
        assert_eq!(r.n_changepoints(), 2);
        assert_eq!(r.segments(), vec![(0, 30), (30, 70), (70, 100)]);
        assert_eq!(r.changepoints(), &[30, 70]);
    }

    #[test]
    fn detector_result_no_changepoints() {
        let r = DetectorResult { bkps: vec![100] };
        assert_eq!(r.n_changepoints(), 0);
        assert_eq!(r.segments(), vec![(0, 100)]);
        assert_eq!(r.changepoints(), &[] as &[usize]);
    }

    #[test]
    fn detector_result_empty() {
        let r = DetectorResult { bkps: vec![] };
        assert_eq!(r.n_changepoints(), 0);
        assert_eq!(r.segments(), Vec::<(usize, usize)>::new());
    }
}
