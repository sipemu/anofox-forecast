//! Rank-based cost. Rank-transforms each dimension at fit time, then
//! scores segments using L2 on the ranks — robust to heavy-tailed
//! distributions. Equivalent to `ruptures.costs.CostRank`.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Rank cost.
///
/// Ranks each dimension over the full signal (ties broken by index), then
/// applies L2 on the ranks. Cumulative sums on the rank representation
/// give O(d) per `error` call.
#[derive(Debug, Default, Clone)]
pub struct CostRank {
    n: usize,
    d: usize,
    cumsum: Vec<f64>,
    cumsum_sq: Vec<f64>,
}

impl CostRank {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostRank {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.n = signal.n();
        self.d = signal.d();
        let stride = self.d;
        // Rank each column independently. Average ranks for ties.
        let mut ranks = vec![0.0_f64; self.n * self.d];
        for j in 0..self.d {
            let mut indexed: Vec<(usize, f64)> =
                (0..self.n).map(|i| (i, signal.row(i)[j])).collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            // Walk in sorted order, assigning average ranks for ties.
            let mut k = 0;
            while k < indexed.len() {
                let mut k2 = k + 1;
                while k2 < indexed.len() && indexed[k2].1 == indexed[k].1 {
                    k2 += 1;
                }
                let avg_rank = (k + k2 - 1) as f64 * 0.5 + 1.0;
                for entry in &indexed[k..k2] {
                    ranks[entry.0 * stride + j] = avg_rank;
                }
                k = k2;
            }
        }
        // Build cumulative sums on the ranks.
        self.cumsum = vec![0.0; (self.n + 1) * stride];
        self.cumsum_sq = vec![0.0; (self.n + 1) * stride];
        for i in 0..self.n {
            for j in 0..self.d {
                let r = ranks[i * stride + j];
                let prev = i * stride + j;
                let next = (i + 1) * stride + j;
                self.cumsum[next] = self.cumsum[prev] + r;
                self.cumsum_sq[next] = self.cumsum_sq[prev] + r * r;
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostRank: invalid segment [{}, {}) for n = {}",
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
        Ok(total.max(0.0))
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "rank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ranks_of_strictly_increasing_match_indices() {
        // [10, 20, 30, 40] → ranks [1, 2, 3, 4]
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let s = Signal::univariate(&values);
        let mut c = CostRank::new();
        c.fit(&s).unwrap();
        // RSS of ranks [1, 2, 3, 4] = (1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)² = 5
        assert_relative_eq!(c.error(0, 4).unwrap(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn ranks_handle_ties_via_average() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let s = Signal::univariate(&values);
        let mut c = CostRank::new();
        c.fit(&s).unwrap();
        // All ranks = 2.5 → RSS = 0
        assert_relative_eq!(c.error(0, 4).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn outliers_dont_blow_up_rank_cost() {
        // L2 cost would be dominated by the outlier; rank cost shouldn't be.
        let values = vec![1.0, 2.0, 3.0, 4.0, 1e6];
        let s = Signal::univariate(&values);
        let mut c = CostRank::new();
        c.fit(&s).unwrap();
        // Ranks are [1, 2, 3, 4, 5], finite cost.
        let e = c.error(0, 5).unwrap();
        assert!(e.is_finite() && e < 100.0);
    }
}
