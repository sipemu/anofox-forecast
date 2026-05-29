//! L1 cost: sum of absolute deviations from the median.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// L1 cost — robust analogue of [`CostL2`](super::l2::CostL2).
///
/// `error = Σ_j Σ_i |s[i, j] − median_j|` per segment. Equivalent to
/// `ruptures.costs.CostL1`. Stores the full signal (no closed-form
/// O(1) update), so each `error` call is O((end−start) · d · log(end−start))
/// due to the in-place sort for median.
#[derive(Debug, Default, Clone)]
pub struct CostL1 {
    n: usize,
    d: usize,
    // Per-dim columns; values[j][i] is sample i, dim j (column-major).
    columns: Vec<Vec<f64>>,
}

impl CostL1 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostL1 {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.n = signal.n();
        self.d = signal.d();
        self.columns = (0..self.d)
            .map(|j| (0..self.n).map(|i| signal.row(i)[j]).collect())
            .collect();
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostL1: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let mut total = 0.0;
        let mut buf: Vec<f64> = Vec::with_capacity(end - start);
        for j in 0..self.d {
            buf.clear();
            buf.extend_from_slice(&self.columns[j][start..end]);
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = buf.len();
            let median = if m % 2 == 1 {
                buf[m / 2]
            } else {
                0.5 * (buf[m / 2 - 1] + buf[m / 2])
            };
            for &x in &buf {
                total += (x - median).abs();
            }
        }
        Ok(total)
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "l1"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn univariate_l1_uses_median() {
        let values = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let s = Signal::univariate(&values);
        let mut c = CostL1::new();
        c.fit(&s).unwrap();
        // median = 5, abs deviations = 4 + 2 + 0 + 2 + 4 = 12
        assert_relative_eq!(c.error(0, 5).unwrap(), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn multivariate_sums_across_dims() {
        let data = vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0];
        let s = Signal::from_row_major(&data, 3, 2).unwrap();
        let mut c = CostL1::new();
        c.fit(&s).unwrap();
        // dim 0: median=2, dev=1+0+1=2; dim 1: median=200, dev=100+0+100=200. Total 202.
        assert_relative_eq!(c.error(0, 3).unwrap(), 202.0, epsilon = 1e-10);
    }
}
