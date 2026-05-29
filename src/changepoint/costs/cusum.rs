//! CUSUM-style sustained-shift detection cost.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// CUSUM cost: sum of `(x − global_mean)²` within each segment relative
/// to the *global* mean (computed at fit time over the whole signal).
/// Detects sustained shifts away from baseline.
///
/// Extra to ruptures — retained from the v0.7.x changepoint API.
#[derive(Debug, Default, Clone)]
pub struct CostCusum {
    n: usize,
    d: usize,
    global_mean: Vec<f64>,
    cumsum_sq_dev: Vec<f64>,
}

impl CostCusum {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostCusum {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.n = signal.n();
        self.d = signal.d();
        let stride = self.d;
        // Per-dim global means.
        let mut sums = vec![0.0_f64; self.d];
        for i in 0..self.n {
            let row = signal.row(i);
            for (j, &v) in row.iter().enumerate() {
                sums[j] += v;
            }
        }
        self.global_mean = sums.iter().map(|s| s / self.n.max(1) as f64).collect();
        self.cumsum_sq_dev = vec![0.0; (self.n + 1) * stride];
        for i in 0..self.n {
            let row = signal.row(i);
            for (j, &v) in row.iter().enumerate() {
                let prev = i * stride + j;
                let next = (i + 1) * stride + j;
                let d = v - self.global_mean[j];
                self.cumsum_sq_dev[next] = self.cumsum_sq_dev[prev] + d * d;
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostCusum: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let stride = self.d;
        let mut total = 0.0;
        for j in 0..self.d {
            total += self.cumsum_sq_dev[end * stride + j] - self.cumsum_sq_dev[start * stride + j];
        }
        Ok(total)
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "cusum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zero_when_centred() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Global mean = 3
        let s = Signal::univariate(&values);
        let mut c = CostCusum::new();
        c.fit(&s).unwrap();
        // (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4+1+0+1+4 = 10
        assert_relative_eq!(c.error(0, 5).unwrap(), 10.0, epsilon = 1e-10);
    }
}
