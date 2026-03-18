//! Concrete [`Transform`] implementations for use with [`Pipeline`].
//!
//! Each transform wraps an existing utility (Box-Cox, scaling) or implements
//! a new operation (differencing, log) behind the [`Transform`] trait so it
//! can be composed in a [`Pipeline`].
//!
//! [`Pipeline`]: super::pipeline::Pipeline
//! [`Transform`]: super::pipeline::Transform

use super::pipeline::{InverseMode, Transform};
use crate::error::{ForecastError, Result};
use crate::transform::boxcox::{boxcox, boxcox_lambda, inv_boxcox};
use crate::transform::scale::{normalize, robust_scale, standardize, ScaleResult};

// ── DifferenceTransform ─────────────────────────────────────────────────────

/// First-order differencing applied `d` times.
///
/// Each application computes `y'_t = y_t - y_{t-1}`, consuming one
/// observation from the front. After `d` applications the series is
/// `d` elements shorter.
#[derive(Debug, Clone)]
pub struct DifferenceTransform {
    d: usize,
    /// Anchors saved at the *start* of the training series (one per diff level).
    initial_anchors: Vec<f64>,
    /// Anchors saved at the *end* of the training series (one per diff level).
    final_anchors: Vec<f64>,
}

impl DifferenceTransform {
    /// Create a differencing transform of order `d` (commonly 1 or 2).
    pub fn new(d: usize) -> Self {
        Self {
            d,
            initial_anchors: Vec::new(),
            final_anchors: Vec::new(),
        }
    }
}

impl Transform for DifferenceTransform {
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        if values.len() <= self.d {
            return Err(ForecastError::InsufficientData {
                needed: self.d + 1,
                got: values.len(),
                hint: Some(format!(
                    "need more than {} observations for Difference({})",
                    self.d, self.d
                )),
            });
        }

        self.initial_anchors.clear();
        self.final_anchors.clear();

        let mut current = values.to_vec();
        for _ in 0..self.d {
            self.initial_anchors.push(current[0]);
            self.final_anchors.push(*current.last().unwrap());
            let next: Vec<f64> = current.windows(2).map(|w| w[1] - w[0]).collect();
            current = next;
        }
        Ok(current)
    }

    fn inverse(&self, values: &[f64], mode: InverseMode) -> Result<Vec<f64>> {
        if self.initial_anchors.len() != self.d {
            return Err(ForecastError::FitRequired {
                model: Some("DifferenceTransform".into()),
            });
        }

        let mut current = values.to_vec();
        // Undo in reverse order of application.
        for level in (0..self.d).rev() {
            match mode {
                InverseMode::Predict => {
                    // Cumsum anchored at the final training value for that diff level.
                    let anchor = self.final_anchors[level];
                    let mut out = Vec::with_capacity(current.len());
                    let mut acc = anchor;
                    for &v in &current {
                        acc += v;
                        out.push(acc);
                    }
                    current = out;
                }
                InverseMode::Fitted => {
                    // Cumsum anchored at the initial training value for that diff level.
                    let anchor = self.initial_anchors[level];
                    let mut out = Vec::with_capacity(current.len() + 1);
                    out.push(anchor);
                    let mut acc = anchor;
                    for &v in &current {
                        acc += v;
                        out.push(acc);
                    }
                    current = out;
                }
            }
        }
        Ok(current)
    }

    fn offset(&self) -> usize {
        self.d
    }

    fn name(&self) -> &str {
        "Difference"
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

// ── SeasonalDifferenceTransform ─────────────────────────────────────────────

/// Seasonal differencing: `y'_t = y_t - y_{t-period}`.
///
/// Consumes the first `period` observations.
#[derive(Debug, Clone)]
pub struct SeasonalDifferenceTransform {
    period: usize,
    /// First `period` values from training (for Fitted inverse).
    initial_values: Vec<f64>,
    /// Last `period` values from training (for Predict inverse).
    final_values: Vec<f64>,
}

impl SeasonalDifferenceTransform {
    /// Create a seasonal differencing transform with the given period.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            initial_values: Vec::new(),
            final_values: Vec::new(),
        }
    }
}

impl Transform for SeasonalDifferenceTransform {
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        let n = values.len();
        if n <= self.period {
            return Err(ForecastError::InsufficientData {
                needed: self.period + 1,
                got: n,
                hint: Some(format!(
                    "need more than {} observations for SeasonalDifference({})",
                    self.period, self.period
                )),
            });
        }

        self.initial_values = values[..self.period].to_vec();
        self.final_values = values[n - self.period..].to_vec();

        let diff: Vec<f64> = values[self.period..]
            .iter()
            .enumerate()
            .map(|(i, &v)| v - values[i])
            .collect();
        Ok(diff)
    }

    fn inverse(&self, values: &[f64], mode: InverseMode) -> Result<Vec<f64>> {
        if self.initial_values.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("SeasonalDifferenceTransform".into()),
            });
        }

        match mode {
            InverseMode::Fitted => {
                // Prepend initial_values, then cumsum: y_t = y'_t + y_{t-period}
                let mut out = Vec::with_capacity(self.period + values.len());
                out.extend_from_slice(&self.initial_values);
                for (i, &d) in values.iter().enumerate() {
                    out.push(d + out[i]);
                }
                Ok(out)
            }
            InverseMode::Predict => {
                // Anchor from final_values: y_t = y'_t + y_{t-period}
                let mut out = Vec::with_capacity(self.period + values.len());
                out.extend_from_slice(&self.final_values);
                for (i, &d) in values.iter().enumerate() {
                    out.push(d + out[i]);
                }
                // Return only the new values (skip the anchor period).
                Ok(out[self.period..].to_vec())
            }
        }
    }

    fn offset(&self) -> usize {
        self.period
    }

    fn name(&self) -> &str {
        "SeasonalDifference"
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

// ── BoxCoxTransform ─────────────────────────────────────────────────────────

/// Box-Cox power transform, optionally with automatic lambda selection.
#[derive(Debug, Clone)]
pub struct BoxCoxTransform {
    /// `None` → auto-select; `Some(λ)` → use fixed lambda.
    requested_lambda: Option<f64>,
    /// Lambda actually used (set after `fit_transform`).
    fitted_lambda: Option<f64>,
}

impl BoxCoxTransform {
    /// Create a Box-Cox transform that automatically selects lambda.
    pub fn auto() -> Self {
        Self {
            requested_lambda: None,
            fitted_lambda: None,
        }
    }

    /// Create a Box-Cox transform with a fixed lambda.
    pub fn with_lambda(lambda: f64) -> Self {
        Self {
            requested_lambda: Some(lambda),
            fitted_lambda: None,
        }
    }
}

impl Transform for BoxCoxTransform {
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        if values.iter().any(|&x| x <= 0.0) {
            return Err(ForecastError::InvalidParameter(
                "BoxCox requires all positive values".into(),
            ));
        }

        let lambda = self
            .requested_lambda
            .unwrap_or_else(|| boxcox_lambda(values));
        self.fitted_lambda = Some(lambda);
        Ok(boxcox(values, lambda))
    }

    fn inverse(&self, values: &[f64], _mode: InverseMode) -> Result<Vec<f64>> {
        let lambda = self.fitted_lambda.ok_or(ForecastError::FitRequired {
            model: Some("BoxCoxTransform".into()),
        })?;
        Ok(inv_boxcox(values, lambda))
    }

    fn offset(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "BoxCox"
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

// ── ScaleMethod ─────────────────────────────────────────────────────────────

/// Scaling method for [`ScaleTransform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMethod {
    /// Z-score: (x - mean) / std
    Standardize,
    /// Min-max to [0, 1]
    Normalize,
    /// Median / IQR
    RobustScale,
}

// ── ScaleTransform ──────────────────────────────────────────────────────────

/// Scaling / normalization transform.
#[derive(Debug, Clone)]
pub struct ScaleTransform {
    method: ScaleMethod,
    center: f64,
    scale: f64,
    fitted: bool,
}

impl ScaleTransform {
    /// Create a scale transform with the given method.
    pub fn new(method: ScaleMethod) -> Self {
        Self {
            method,
            center: 0.0,
            scale: 1.0,
            fitted: false,
        }
    }
}

impl Transform for ScaleTransform {
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        let result: ScaleResult = match self.method {
            ScaleMethod::Standardize => standardize(values),
            ScaleMethod::Normalize => normalize(values),
            ScaleMethod::RobustScale => robust_scale(values),
        };
        self.center = result.center;
        self.scale = result.scale;
        self.fitted = true;
        Ok(result.data)
    }

    fn inverse(&self, values: &[f64], _mode: InverseMode) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ForecastError::FitRequired {
                model: Some("ScaleTransform".into()),
            });
        }
        Ok(values
            .iter()
            .map(|&x| x * self.scale + self.center)
            .collect())
    }

    fn offset(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "Scale"
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

// ── LogTransform ────────────────────────────────────────────────────────────

/// Natural-log transform with an automatic positive shift.
///
/// If the minimum value is ≤ 0, a shift is added so all values become
/// positive before taking the log.
#[derive(Debug, Clone)]
pub struct LogTransform {
    shift: f64,
    fitted: bool,
}

impl LogTransform {
    /// Create a new log transform.
    pub fn new() -> Self {
        Self {
            shift: 0.0,
            fitted: false,
        }
    }
}

impl Default for LogTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for LogTransform {
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        self.shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };
        self.fitted = true;
        Ok(values.iter().map(|&x| (x + self.shift).ln()).collect())
    }

    fn inverse(&self, values: &[f64], _mode: InverseMode) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ForecastError::FitRequired {
                model: Some("LogTransform".into()),
            });
        }
        Ok(values.iter().map(|&x| x.exp() - self.shift).collect())
    }

    fn offset(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "Log"
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── Difference ──────────────────────────────────────────────────────

    #[test]
    fn difference_d1_forward_and_fitted_inverse() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let mut t = DifferenceTransform::new(1);
        let diff = t.fit_transform(&data).unwrap();

        assert_eq!(diff, vec![2.0, 3.0, 4.0, 5.0]);

        // Fitted inverse should recover the original (including the anchor).
        let recovered = t.inverse(&diff, InverseMode::Fitted).unwrap();
        assert_eq!(recovered.len(), data.len());
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn difference_d1_predict_inverse() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let mut t = DifferenceTransform::new(1);
        t.fit_transform(&data).unwrap();

        // Predict: constant diff of 5 continuing from last value (15).
        let forecast_diff = vec![5.0, 5.0, 5.0];
        let forecast = t.inverse(&forecast_diff, InverseMode::Predict).unwrap();
        assert_eq!(forecast, vec![20.0, 25.0, 30.0]);
    }

    #[test]
    fn difference_d2() {
        let data = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // x^2
        let mut t = DifferenceTransform::new(2);
        let diff2 = t.fit_transform(&data).unwrap();

        // First diff: [3,5,7,9], second diff: [2,2,2]
        assert_eq!(diff2.len(), 3);
        for &v in &diff2 {
            assert_relative_eq!(v, 2.0, epsilon = 1e-12);
        }

        let recovered = t.inverse(&diff2, InverseMode::Fitted).unwrap();
        assert_eq!(recovered.len(), data.len());
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn difference_insufficient_data() {
        let mut t = DifferenceTransform::new(2);
        assert!(t.fit_transform(&[1.0, 2.0]).is_err());
    }

    // ── SeasonalDifference ──────────────────────────────────────────────

    #[test]
    fn seasonal_diff_forward_and_fitted_inverse() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0];
        let mut t = SeasonalDifferenceTransform::new(4);
        let diff = t.fit_transform(&data).unwrap();

        // diff = [5-1, 7-2, 9-3, 11-4] = [4,5,6,7]
        assert_eq!(diff, vec![4.0, 5.0, 6.0, 7.0]);

        let recovered = t.inverse(&diff, InverseMode::Fitted).unwrap();
        assert_eq!(recovered.len(), data.len());
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn seasonal_diff_predict_inverse() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0];
        let mut t = SeasonalDifferenceTransform::new(4);
        t.fit_transform(&data).unwrap();

        // Forecast diffs of 0 → repeat last cycle [5,7,9,11]
        let forecast_diff = vec![0.0, 0.0, 0.0, 0.0];
        let forecast = t.inverse(&forecast_diff, InverseMode::Predict).unwrap();
        assert_eq!(forecast, vec![5.0, 7.0, 9.0, 11.0]);
    }

    // ── BoxCox ──────────────────────────────────────────────────────────

    #[test]
    fn boxcox_auto_roundtrip() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut t = BoxCoxTransform::auto();
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn boxcox_fixed_lambda() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut t = BoxCoxTransform::with_lambda(0.5);
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn boxcox_rejects_non_positive() {
        let data = vec![-1.0, 0.0, 1.0];
        let mut t = BoxCoxTransform::auto();
        assert!(t.fit_transform(&data).is_err());
    }

    // ── Scale ───────────────────────────────────────────────────────────

    #[test]
    fn scale_standardize_roundtrip() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut t = ScaleTransform::new(ScaleMethod::Standardize);
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn scale_normalize_roundtrip() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut t = ScaleTransform::new(ScaleMethod::Normalize);
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    // ── Log ─────────────────────────────────────────────────────────────

    #[test]
    fn log_positive_roundtrip() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut t = LogTransform::new();
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn log_with_shift() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut t = LogTransform::new();
        let transformed = t.fit_transform(&data).unwrap();

        let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn log_empty_errors() {
        let mut t = LogTransform::new();
        assert!(t.fit_transform(&[]).is_err());
    }

    // ── Trait properties ────────────────────────────────────────────────

    #[test]
    fn offsets_are_correct() {
        assert_eq!(DifferenceTransform::new(1).offset(), 1);
        assert_eq!(DifferenceTransform::new(2).offset(), 2);
        assert_eq!(SeasonalDifferenceTransform::new(12).offset(), 12);
        assert_eq!(BoxCoxTransform::auto().offset(), 0);
        assert_eq!(ScaleTransform::new(ScaleMethod::Standardize).offset(), 0);
        assert_eq!(LogTransform::new().offset(), 0);
    }

    #[test]
    fn names_are_correct() {
        assert_eq!(DifferenceTransform::new(1).name(), "Difference");
        assert_eq!(
            SeasonalDifferenceTransform::new(7).name(),
            "SeasonalDifference"
        );
        assert_eq!(BoxCoxTransform::auto().name(), "BoxCox");
        assert_eq!(
            ScaleTransform::new(ScaleMethod::Standardize).name(),
            "Scale"
        );
        assert_eq!(LogTransform::new().name(), "Log");
    }

    #[test]
    fn clone_box_works() {
        let mut t = DifferenceTransform::new(1);
        let data = vec![1.0, 3.0, 6.0, 10.0];
        t.fit_transform(&data).unwrap();

        let cloned = t.clone_box();
        let inv = cloned.inverse(&[2.0, 3.0], InverseMode::Predict).unwrap();
        assert_eq!(inv, vec![12.0, 15.0]);
    }

    #[test]
    fn inverse_before_fit_errors() {
        let t = DifferenceTransform::new(1);
        assert!(t.inverse(&[1.0], InverseMode::Predict).is_err());

        let t = BoxCoxTransform::auto();
        assert!(t.inverse(&[1.0], InverseMode::Predict).is_err());

        let t = ScaleTransform::new(ScaleMethod::Standardize);
        assert!(t.inverse(&[1.0], InverseMode::Predict).is_err());

        let t = LogTransform::new();
        assert!(t.inverse(&[1.0], InverseMode::Predict).is_err());

        let t = SeasonalDifferenceTransform::new(4);
        assert!(t.inverse(&[1.0], InverseMode::Predict).is_err());
    }
}
