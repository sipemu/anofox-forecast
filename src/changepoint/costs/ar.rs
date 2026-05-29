//! Autoregressive cost: RSS after fitting an AR(p) model per segment.
//!
//! Mirrors `ruptures.costs.CostAR`. Detects changes in the
//! autocorrelation structure.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// AR(p) cost. Fits `x_t = Σ φ_i · x_{t-i} + ε_t` (with intercept) on
/// each segment via OLS and returns the residual sum of squares.
///
/// Univariate only. Each `error` call is O(p² · (end − start) + p³),
/// dominated by the normal-equations solve.
#[derive(Debug, Clone)]
pub struct CostAR {
    order: usize,
    n: usize,
    values: Vec<f64>,
}

impl CostAR {
    /// Construct an AR(p) cost. `order >= 1`.
    pub fn new(order: usize) -> Self {
        Self {
            order: order.max(1),
            n: 0,
            values: Vec::new(),
        }
    }
}

impl Cost for CostAR {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        if !signal.is_univariate() {
            return Err(ForecastError::InvalidParameter(
                "CostAR: univariate signals only".into(),
            ));
        }
        self.n = signal.n();
        self.values = (0..self.n).map(|i| signal.row(i)[0]).collect();
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostAR: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let p = self.order;
        let seg_len = end - start;
        if seg_len <= p + 1 {
            return Ok(0.0);
        }
        let m = seg_len - p; // number of usable equations

        // Build normal equations: (X'X) β = X'y where row i (i in 0..m) is
        //   y_i = values[start + p + i]
        //   X_i = [1, values[start + p + i - 1], …, values[start + p + i - p]]
        let cols = p + 1;
        let mut xtx = vec![0.0_f64; cols * cols];
        let mut xty = vec![0.0_f64; cols];
        for i in 0..m {
            let y = self.values[start + p + i];
            let mut x = vec![1.0_f64; cols];
            for k in 1..=p {
                x[k] = self.values[start + p + i - k];
            }
            for a in 0..cols {
                for b in 0..cols {
                    xtx[a * cols + b] += x[a] * x[b];
                }
                xty[a] += x[a] * y;
            }
        }
        // Ridge regularisation for stability.
        for k in 0..cols {
            xtx[k * cols + k] += 1e-9;
        }

        let beta = solve_symmetric(&xtx, &xty, cols).ok_or_else(|| {
            ForecastError::ComputationError(format!(
                "CostAR: AR({}) fit failed on segment [{}, {})",
                p, start, end
            ))
        })?;

        // RSS
        let mut rss = 0.0;
        for i in 0..m {
            let y = self.values[start + p + i];
            let mut yhat = beta[0];
            for k in 1..=p {
                yhat += beta[k] * self.values[start + p + i - k];
            }
            rss += (y - yhat).powi(2);
        }
        Ok(rss.max(0.0))
    }

    fn min_size(&self) -> usize {
        self.order + 2
    }

    fn name(&self) -> &str {
        "ar"
    }
}

/// Solve symmetric positive-definite Ax = b via Cholesky.
fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Cholesky decomposition: A = L·L^T.
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                if l[j * n + j] == 0.0 {
                    return None;
                }
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // Forward solve Ly = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }
    // Backward solve L^T x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_with_strong_autocorr_fits_well() {
        // AR(1) with phi = 0.8: x_t = 0.8 · x_{t-1} + noise(small)
        let n = 200;
        let mut values = vec![0.0_f64; n];
        values[0] = 1.0;
        for i in 1..n {
            let noise = ((i * 7 + 13) % 17) as f64 * 0.001 - 0.008;
            values[i] = 0.8 * values[i - 1] + noise;
        }
        let s = Signal::univariate(&values);
        let mut c = CostAR::new(1);
        c.fit(&s).unwrap();
        let rss = c.error(0, n).unwrap();
        assert!(rss.is_finite());
        // Should be small (noise is tiny).
        assert!(rss < 1.0);
    }

    #[test]
    fn short_segment_returns_zero() {
        let values = vec![1.0, 2.0, 3.0];
        let s = Signal::univariate(&values);
        let mut c = CostAR::new(2);
        c.fit(&s).unwrap();
        assert_eq!(c.error(0, 3).unwrap(), 0.0);
    }
}
