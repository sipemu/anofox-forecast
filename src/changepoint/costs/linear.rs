//! Linear-trend cost: residual sum of squares from a linear regression
//! within each segment. Detects slope changes.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Linear-trend cost.
///
/// Within each segment, fits `y = a + b · t` by OLS and returns the
/// residual sum of squares. Detects changes in slope or level.
///
/// Univariate-only at the moment (multivariate would be CostLinear in
/// ruptures, which fits a vector regression; that's planned as a separate
/// `CostLinear` impl in Phase 3).
///
/// Uses precomputed cumulative `Σx`, `Σx²`, `Σt·x` (with `t = 0..n−1`)
/// so each `error` call is O(1).
#[derive(Debug, Default, Clone)]
pub struct CostLinearTrend {
    n: usize,
    cum_x: Vec<f64>,
    cum_x_sq: Vec<f64>,
    cum_tx: Vec<f64>,
}

impl CostLinearTrend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostLinearTrend {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        if !signal.is_univariate() {
            return Err(ForecastError::InvalidParameter(
                "CostLinearTrend: only univariate signals are supported (use CostLinear in Phase 3 for multivariate)".into(),
            ));
        }
        self.n = signal.n();
        self.cum_x = vec![0.0; self.n + 1];
        self.cum_x_sq = vec![0.0; self.n + 1];
        self.cum_tx = vec![0.0; self.n + 1];
        for i in 0..self.n {
            let x = signal.row(i)[0];
            self.cum_x[i + 1] = self.cum_x[i] + x;
            self.cum_x_sq[i + 1] = self.cum_x_sq[i] + x * x;
            self.cum_tx[i + 1] = self.cum_tx[i] + (i as f64) * x;
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostLinearTrend: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let n = (end - start) as f64;
        if n < 2.0 {
            return Ok(0.0);
        }
        let s_x = self.cum_x[end] - self.cum_x[start];
        let s_x_sq = self.cum_x_sq[end] - self.cum_x_sq[start];
        let s_tx = self.cum_tx[end] - self.cum_tx[start];
        // Σt over [start, end-1] = (start + end - 1) · n / 2
        let s_t = (start as f64 + (end as f64 - 1.0)) * n / 2.0;
        // Σt² over [start, end-1] using closed form
        let end_m1 = (end - 1) as f64;
        let start_m1 = if start == 0 {
            -1.0_f64
        } else {
            (start - 1) as f64
        };
        let s_t_sq = sum_squares_0_to(end_m1) - sum_squares_0_to(start_m1);

        let t_mean = s_t / n;
        let x_mean = s_x / n;
        let s_tt = s_t_sq - n * t_mean * t_mean;
        if s_tt.abs() < 1e-12 {
            // Flat-t (shouldn't happen for n > 1, but defensive)
            let rss = s_x_sq - n * x_mean * x_mean;
            return Ok(rss.max(0.0));
        }
        let s_tx_centred = s_tx - n * t_mean * x_mean;
        let slope = s_tx_centred / s_tt;
        let intercept = x_mean - slope * t_mean;
        // RSS = Σ(x − intercept − slope · t)²
        //     = s_x_sq − 2·intercept·s_x − 2·slope·s_tx
        //       + n·intercept² + 2·intercept·slope·s_t + slope²·s_t_sq
        let rss = s_x_sq - 2.0 * intercept * s_x - 2.0 * slope * s_tx
            + n * intercept * intercept
            + 2.0 * intercept * slope * s_t
            + slope * slope * s_t_sq;
        Ok(rss.max(0.0))
    }

    fn min_size(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "linear_trend"
    }
}

#[inline]
fn sum_squares_0_to(end: f64) -> f64 {
    // Σ_{i=0}^{end} i² when end ≥ 0; 0 otherwise.
    if end < 0.0 {
        0.0
    } else {
        let m = end;
        m * (m + 1.0) * (2.0 * m + 1.0) / 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfect_line_zero_rss() {
        // x[i] = 3 + 2 * i — perfectly linear, no noise.
        let values: Vec<f64> = (0..20).map(|i| 3.0 + 2.0 * i as f64).collect();
        let s = Signal::univariate(&values);
        let mut c = CostLinearTrend::new();
        c.fit(&s).unwrap();
        // Sub-segment must also fit perfectly.
        assert_relative_eq!(c.error(5, 15).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(c.error(0, 20).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn quadratic_has_positive_rss() {
        let values: Vec<f64> = (0..20).map(|i| (i as f64).powi(2)).collect();
        let s = Signal::univariate(&values);
        let mut c = CostLinearTrend::new();
        c.fit(&s).unwrap();
        let rss = c.error(0, 20).unwrap();
        assert!(rss > 0.0);
    }

    #[test]
    fn multivariate_rejected() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let s = Signal::from_row_major(&data, 2, 2).unwrap();
        let mut c = CostLinearTrend::new();
        assert!(c.fit(&s).is_err());
    }
}
