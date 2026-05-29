//! Joint mean-and-variance cost.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Joint mean-and-variance cost. Penalises both changes in mean and
/// changes in variance more aggressively than `CostNormal` alone.
///
/// `error_per_dim = n · (log(σ²) + mean² / σ²)`. Extra to ruptures —
/// retained from the v0.7.x changepoint API.
#[derive(Debug, Default, Clone)]
pub struct CostMeanVariance {
    n: usize,
    d: usize,
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
}

impl CostMeanVariance {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostMeanVariance {
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
                "CostMeanVariance: invalid segment [{}, {}) for n = {}",
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
            total += len * (var.ln() + mean * mean / var);
        }
        Ok(total)
    }

    fn min_size(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "mean_variance"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_on_constant_input() {
        let values = vec![5.0; 10];
        let s = Signal::univariate(&values);
        let mut c = CostMeanVariance::new();
        c.fit(&s).unwrap();
        assert!(c.error(0, 10).unwrap().is_finite());
    }
}
