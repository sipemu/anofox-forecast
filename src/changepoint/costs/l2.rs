//! L2 (squared-error / variance) cost.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// L2 cost: residual sum of squares from the segment mean.
///
/// For a segment `s = signal[a..b]`,
/// `error = Σ_j Σ_i (s[i, j] - mean_j)² = Σ_j (sum_sq_j − sum_j² / n)`.
///
/// Equivalent to `ruptures.costs.CostL2`. Uses precomputed cumulative
/// sums so each `error` call is O(d).
#[derive(Debug, Default, Clone)]
pub struct CostL2 {
    n: usize,
    d: usize,
    // cumulative sums, length (n+1) * d, row-major. cumsum[(i+1)*d + j] −
    // cumsum[i*d + j] is the running sum at sample i, dim j.
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
}

impl CostL2 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostL2 {
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
                "CostL2: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let stride = self.d;
        let len = (end - start) as f64;
        let mut total = 0.0;
        for j in 0..self.d {
            let sum = self.cumsum[end * stride + j] - self.cumsum[start * stride + j];
            let sum_sq = self.cumsum_sq[end * stride + j] - self.cumsum_sq[start * stride + j];
            total += sum_sq - sum * sum / len;
        }
        // Numerical floor: RSS can dip to a tiny negative from cancellation.
        Ok(total.max(0.0))
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "l2"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn univariate_rss_matches_naive() {
        let values = vec![1.0, 2.0, 3.0, 10.0, 12.0, 14.0];
        let signal = Signal::univariate(&values);
        let mut cost = CostL2::new();
        cost.fit(&signal).unwrap();

        // Full segment: RSS = sum((x - mean)^2)
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let rss: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
        assert_relative_eq!(cost.error(0, 6).unwrap(), rss, epsilon = 1e-10);

        // Sub-segment [0, 3) = [1, 2, 3]
        let mean = 2.0;
        let rss = (1.0_f64 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2);
        assert_relative_eq!(cost.error(0, 3).unwrap(), rss, epsilon = 1e-10);
    }

    #[test]
    fn multivariate_rss_sums_across_dims() {
        // n = 4, d = 2 — row-major
        let data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let signal = Signal::from_row_major(&data, 4, 2).unwrap();
        let mut cost = CostL2::new();
        cost.fit(&signal).unwrap();

        // For each dim, mean = 2.5 (dim 0) and 25.0 (dim 1)
        // RSS dim 0 = (1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)² = 5
        // RSS dim 1 = (10-25)² + (20-25)² + (30-25)² + (40-25)² = 500
        assert_relative_eq!(cost.error(0, 4).unwrap(), 505.0, epsilon = 1e-10);
    }

    #[test]
    fn invalid_segment_errors() {
        let values = vec![1.0, 2.0, 3.0];
        let signal = Signal::univariate(&values);
        let mut cost = CostL2::new();
        cost.fit(&signal).unwrap();

        assert!(cost.error(0, 0).is_err());
        assert!(cost.error(2, 1).is_err());
        assert!(cost.error(0, 4).is_err());
    }
}
