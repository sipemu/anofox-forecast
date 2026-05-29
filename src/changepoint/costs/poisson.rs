//! Poisson log-likelihood cost for count data.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Poisson cost — `error = −2 · (n · λ_hat · log(λ_hat) − n · λ_hat)`
/// per dimension where `λ_hat = mean`. Returns 0 when the mean is 0.
///
/// Univariate-friendly cost for count series. Not multivariate-aware in
/// the same way ruptures' `CostNormal` is; multivariate signals are
/// scored as a sum across dimensions.
#[derive(Debug, Default, Clone)]
pub struct CostPoisson {
    n: usize,
    d: usize,
    cumsum: Vec<f64>,
    cumsum_log: Vec<f64>,
}

impl CostPoisson {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostPoisson {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.n = signal.n();
        self.d = signal.d();
        let stride = self.d;
        self.cumsum = vec![0.0; (self.n + 1) * stride];
        self.cumsum_log = vec![0.0; (self.n + 1) * stride];
        for i in 0..self.n {
            let row = signal.row(i);
            for (j, &v) in row.iter().enumerate() {
                let prev = i * stride + j;
                let next = (i + 1) * stride + j;
                self.cumsum[next] = self.cumsum[prev] + v;
                // log(x!) ≈ x log x − x for x ≥ 1, 0 otherwise
                let log_term = if v >= 1.0 { v * v.ln() - v } else { 0.0 };
                self.cumsum_log[next] = self.cumsum_log[prev] + log_term;
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostPoisson: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let stride = self.d;
        let len = (end - start) as f64;
        let mut total = 0.0;
        for j in 0..self.d {
            let sum = self.cumsum[end * stride + j] - self.cumsum[start * stride + j];
            let mean = sum / len;
            if mean <= 1e-12 {
                continue;
            }
            let log_sum_terms =
                self.cumsum_log[end * stride + j] - self.cumsum_log[start * stride + j];
            // -2 * log-likelihood under λ = mean
            total += -2.0 * (sum * mean.ln() - len * mean - log_sum_terms);
        }
        Ok(total)
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "poisson"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_series_returns_finite_cost() {
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let s = Signal::univariate(&values);
        let mut c = CostPoisson::new();
        c.fit(&s).unwrap();
        let e = c.error(0, 10).unwrap();
        assert!(e.is_finite());
    }

    #[test]
    fn zero_segment_zero_cost() {
        let values = vec![0.0; 10];
        let s = Signal::univariate(&values);
        let mut c = CostPoisson::new();
        c.fit(&s).unwrap();
        let e = c.error(0, 10).unwrap();
        assert_eq!(e, 0.0);
    }
}
