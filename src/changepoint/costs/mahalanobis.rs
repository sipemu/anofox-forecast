//! Mahalanobis-distance cost.
//!
//! Mirrors `ruptures.costs.CostMl` (Mahalanobis-likelihood). For each
//! segment of a multivariate signal, computes
//!
//! ```text
//! error = Σ_i (x_i − mean) · M · (x_i − mean)
//! ```
//!
//! where `M` is a user-supplied positive-definite metric matrix
//! (typically the inverse covariance of the signal).

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Mahalanobis cost with a precomputed metric matrix.
#[derive(Debug, Clone)]
pub struct CostMahalanobis {
    metric: Vec<f64>, // d × d, row-major
    d: usize,
    n: usize,
    cumsum: Vec<f64>,       // (n+1) × d
    cumsum_outer: Vec<f64>, // (n+1) × d × d (row-major flat)
}

impl CostMahalanobis {
    /// Construct with the given `d × d` row-major metric matrix.
    pub fn new(metric: Vec<f64>, d: usize) -> Result<Self> {
        if metric.len() != d * d {
            return Err(ForecastError::DimensionMismatch {
                expected: d * d,
                got: metric.len(),
            });
        }
        Ok(Self {
            metric,
            d,
            n: 0,
            cumsum: Vec::new(),
            cumsum_outer: Vec::new(),
        })
    }
}

impl Cost for CostMahalanobis {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        if signal.d() != self.d {
            return Err(ForecastError::DimensionMismatch {
                expected: self.d,
                got: signal.d(),
            });
        }
        self.n = signal.n();
        let d = self.d;
        self.cumsum = vec![0.0; (self.n + 1) * d];
        self.cumsum_outer = vec![0.0; (self.n + 1) * d * d];
        for i in 0..self.n {
            let row = signal.row(i);
            for j in 0..d {
                self.cumsum[(i + 1) * d + j] = self.cumsum[i * d + j] + row[j];
            }
            for a in 0..d {
                for b in 0..d {
                    let prev = i * d * d + a * d + b;
                    let next = (i + 1) * d * d + a * d + b;
                    self.cumsum_outer[next] = self.cumsum_outer[prev] + row[a] * row[b];
                }
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostMahalanobis: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let d = self.d;
        let len = (end - start) as f64;
        // Per-dim segment means.
        let mut means = vec![0.0_f64; d];
        for j in 0..d {
            let s = self.cumsum[end * d + j] - self.cumsum[start * d + j];
            means[j] = s / len;
        }
        // Σ_i (x_i - mean) · M · (x_i - mean) = Σ_i M_ab x_i_a x_i_b
        //                                       − len · M_ab mean_a mean_b
        //                                       (twice the cross-term cancels via
        //                                        the mean definition)
        let mut total = 0.0_f64;
        for a in 0..d {
            for b in 0..d {
                let m_ab = self.metric[a * d + b];
                let outer_sum = self.cumsum_outer[end * d * d + a * d + b]
                    - self.cumsum_outer[start * d * d + a * d + b];
                total += m_ab * outer_sum;
                total -= m_ab * len * means[a] * means[b];
            }
        }
        Ok(total.max(0.0))
    }

    fn min_size(&self) -> usize {
        self.d
    }

    fn name(&self) -> &str {
        "mahalanobis"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_metric_matches_l2() {
        // With M = I, Mahalanobis cost equals L2 RSS (sum across dims).
        let data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let s = Signal::from_row_major(&data, 4, 2).unwrap();
        let i_matrix = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = CostMahalanobis::new(i_matrix, 2).unwrap();
        c.fit(&s).unwrap();
        // RSS dim 0 = 5, dim 1 = 500 → total 505
        assert_relative_eq!(c.error(0, 4).unwrap(), 505.0, epsilon = 1e-6);
    }

    #[test]
    fn dim_mismatch_at_fit_errors() {
        let data = vec![1.0, 2.0];
        let s = Signal::univariate(&data);
        let mut c = CostMahalanobis::new(vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert!(c.fit(&s).is_err());
    }
}
