//! Multivariate linear-regression cost.
//!
//! Mirrors `ruptures.costs.CostLinear`. The signal is treated as a
//! multivariate array `(n, 1 + d)` where the first column is the
//! response variable `y` and the remaining columns are covariates `X`.
//!
//! For each segment, fits `y = X · β + ε` by OLS and returns the
//! residual sum of squares.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Multivariate linear regression cost.
#[derive(Debug, Default, Clone)]
pub struct CostLinear {
    n: usize,
    d_x: usize, // number of covariates (excludes response)
    y: Vec<f64>,
    // X stored row-major as n × d_x
    x: Vec<f64>,
}

impl CostLinear {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostLinear {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        if signal.d() < 2 {
            return Err(ForecastError::InvalidParameter(
                "CostLinear: signal must have at least 2 dimensions (1 response + ≥1 covariates)"
                    .into(),
            ));
        }
        self.n = signal.n();
        self.d_x = signal.d() - 1;
        self.y = Vec::with_capacity(self.n);
        self.x = Vec::with_capacity(self.n * self.d_x);
        for i in 0..self.n {
            let row = signal.row(i);
            self.y.push(row[0]);
            self.x.extend_from_slice(&row[1..]);
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostLinear: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let n = end - start;
        let d = self.d_x + 1; // + intercept
        if n < d + 1 {
            return Ok(0.0);
        }

        // Build X'X (d × d) and X'y (d), with x_intercept = 1.
        let mut xtx = vec![0.0_f64; d * d];
        let mut xty = vec![0.0_f64; d];
        for i in 0..n {
            let row_idx = start + i;
            let row_x = &self.x[row_idx * self.d_x..(row_idx + 1) * self.d_x];
            let y_i = self.y[row_idx];
            // Augmented row: [1, x_0, x_1, …]
            let mut full = Vec::with_capacity(d);
            full.push(1.0);
            full.extend_from_slice(row_x);
            for a in 0..d {
                for b in 0..d {
                    xtx[a * d + b] += full[a] * full[b];
                }
                xty[a] += full[a] * y_i;
            }
        }
        for k in 0..d {
            xtx[k * d + k] += 1e-9;
        }

        let beta = solve_chol(&xtx, &xty, d).ok_or_else(|| {
            ForecastError::ComputationError(format!(
                "CostLinear: OLS fit failed on segment [{}, {})",
                start, end
            ))
        })?;

        let mut rss = 0.0;
        for i in 0..n {
            let row_idx = start + i;
            let row_x = &self.x[row_idx * self.d_x..(row_idx + 1) * self.d_x];
            let mut yhat = beta[0];
            for j in 0..self.d_x {
                yhat += beta[j + 1] * row_x[j];
            }
            rss += (self.y[row_idx] - yhat).powi(2);
        }
        Ok(rss.max(0.0))
    }

    fn min_size(&self) -> usize {
        self.d_x + 2
    }

    fn name(&self) -> &str {
        "linear"
    }
}

fn solve_chol(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
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
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }
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
    fn linear_relation_zero_rss() {
        // y = 2 + 3 · x — perfect linear dependence.
        let n = 30;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let x = i as f64;
            let y = 2.0 + 3.0 * x;
            data.push(y);
            data.push(x);
        }
        let s = Signal::from_row_major(&data, n, 2).unwrap();
        let mut c = CostLinear::new();
        c.fit(&s).unwrap();
        let rss = c.error(0, n).unwrap();
        assert!(rss < 1e-6, "expected zero RSS, got {}", rss);
    }

    #[test]
    fn univariate_rejected() {
        let values = vec![1.0, 2.0, 3.0];
        let s = Signal::univariate(&values);
        let mut c = CostLinear::new();
        assert!(c.fit(&s).is_err());
    }
}
