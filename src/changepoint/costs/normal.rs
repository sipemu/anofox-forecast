//! Gaussian log-likelihood cost (variance-changes).

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Normal (Gaussian) log-likelihood cost.
///
/// `error = n · log(σ²)` per dimension, where `σ²` is the segment
/// sample variance (population formula with n divisor). Equivalent to
/// `ruptures.costs.CostNormal` up to constant terms that don't affect
/// changepoint location.
///
/// Detects changes in variance (and mean, indirectly through the
/// pooled variance). Cumulative sums make `error` O(d).
#[derive(Debug, Default, Clone)]
pub struct CostNormal {
    n: usize,
    d: usize,
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
}

impl CostNormal {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostNormal {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.n = signal.n();
        self.d = signal.d();
        let stride = self.d;
        self.cumsum = vec![0.0; (self.n + 1) * stride];
        self.cumsum_sq = vec![0.0; (self.n + 1) * stride];
        for i in 0..self.n {
            let row = signal.row(i);
            for (j, &v) in row.iter().enumerate() {
                let prev = i * stride + j;
                let next = (i + 1) * stride + j;
                self.cumsum[next] = self.cumsum[prev] + v;
                self.cumsum_sq[next] = self.cumsum_sq[prev] + v * v;
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostNormal: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let stride = self.d;
        let len = (end - start) as f64;
        let mut total = 0.0;
        for j in 0..self.d {
            let sum = self.cumsum[end * stride + j] - self.cumsum[start * stride + j];
            let sum_sq = self.cumsum_sq[end * stride + j] - self.cumsum_sq[start * stride + j];
            let mean = sum / len;
            let var = (sum_sq / len - mean * mean).max(1e-12);
            total += len * var.ln();
        }
        Ok(total)
    }

    fn min_size(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "normal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_segment_has_floor_cost() {
        let values = vec![5.0; 10];
        let s = Signal::univariate(&values);
        let mut c = CostNormal::new();
        c.fit(&s).unwrap();
        // Variance = 0 → log(floor) — must be finite.
        let e = c.error(0, 10).unwrap();
        assert!(e.is_finite());
    }

    #[test]
    fn higher_variance_higher_cost() {
        let low: Vec<f64> = (0..20).map(|i| 1.0 + 0.1 * (i % 2) as f64).collect();
        let high: Vec<f64> = (0..20).map(|i| 1.0 + 5.0 * (i % 2) as f64).collect();

        let mut c_low = CostNormal::new();
        c_low.fit(&Signal::univariate(&low)).unwrap();
        let mut c_high = CostNormal::new();
        c_high.fit(&Signal::univariate(&high)).unwrap();

        assert!(c_high.error(0, 20).unwrap() > c_low.error(0, 20).unwrap());
    }
}
