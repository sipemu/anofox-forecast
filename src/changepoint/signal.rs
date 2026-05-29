//! Signal carrier for changepoint detection.
//!
//! Wraps a flat `&[f64]` slice with `n × d` shape information so cost
//! functions and detectors can transparently support univariate and
//! multivariate signals.

use crate::error::{ForecastError, Result};

/// A multivariate signal of shape `(n, d)`, stored row-major as a flat
/// `&[f64]` slice. `n` is the number of samples, `d` is the dimension.
///
/// Univariate series can be wrapped via [`Signal::univariate`] or by
/// `Signal::from(values)`.
#[derive(Debug, Clone, Copy)]
pub struct Signal<'a> {
    data: &'a [f64],
    n: usize,
    d: usize,
}

impl<'a> Signal<'a> {
    /// Wrap a univariate sequence (`d = 1`).
    pub fn univariate(values: &'a [f64]) -> Self {
        Self {
            data: values,
            n: values.len(),
            d: 1,
        }
    }

    /// Wrap a multivariate signal stored row-major: `data[i*d + j]` is
    /// the j-th coordinate of the i-th sample.
    pub fn from_row_major(data: &'a [f64], n: usize, d: usize) -> Result<Self> {
        if d == 0 {
            return Err(ForecastError::InvalidParameter(
                "Signal: dimension d must be ≥ 1".into(),
            ));
        }
        if data.len() != n * d {
            return Err(ForecastError::DimensionMismatch {
                expected: n * d,
                got: data.len(),
            });
        }
        Ok(Self { data, n, d })
    }

    /// Number of samples.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension (number of features per sample).
    pub fn d(&self) -> usize {
        self.d
    }

    /// True iff `d == 1`.
    pub fn is_univariate(&self) -> bool {
        self.d == 1
    }

    /// Row `i` as a `&[f64]` of length `d`.
    ///
    /// # Panics
    /// Panics if `i >= n`.
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.d;
        let end = start + self.d;
        &self.data[start..end]
    }

    /// Full row-major slice.
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }
}

impl<'a> From<&'a [f64]> for Signal<'a> {
    fn from(values: &'a [f64]) -> Self {
        Signal::univariate(values)
    }
}

impl<'a, const N: usize> From<&'a [f64; N]> for Signal<'a> {
    fn from(values: &'a [f64; N]) -> Self {
        Signal::univariate(values.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn univariate_from_slice_round_trip() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let s: Signal = (&values[..]).into();
        assert_eq!(s.n(), 4);
        assert_eq!(s.d(), 1);
        assert!(s.is_univariate());
        assert_eq!(s.row(2), &[3.0]);
    }

    #[test]
    fn multivariate_row_major() {
        // n=3 samples, d=2 dims: [1,2, 3,4, 5,6]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = Signal::from_row_major(&data, 3, 2).unwrap();
        assert_eq!(s.n(), 3);
        assert_eq!(s.d(), 2);
        assert!(!s.is_univariate());
        assert_eq!(s.row(0), &[1.0, 2.0]);
        assert_eq!(s.row(1), &[3.0, 4.0]);
        assert_eq!(s.row(2), &[5.0, 6.0]);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let data = vec![1.0, 2.0, 3.0];
        let err = Signal::from_row_major(&data, 2, 2).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }

    #[test]
    fn zero_dimension_rejected() {
        let data = vec![1.0, 2.0];
        let err = Signal::from_row_major(&data, 2, 0).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }
}
