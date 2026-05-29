//! Continuous-piecewise-linear cost.
//!
//! Mirrors `ruptures.costs.CostCLinear`. Fits a linear regression on
//! each segment but constrains the regression to pass through the
//! segment endpoints — when concatenated, segments form a continuous
//! piecewise-linear curve.
//!
//! Specifically, for a univariate segment `[start, end)`:
//!
//! ```text
//! cost = Σ_{i = start..end} (x_i − ŷ_i)²
//! ŷ_i  = x_start + (i − start) · (x_{end−1} − x_start) / (end − start − 1)
//! ```
//!
//! Univariate-only.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Continuous piecewise-linear cost.
#[derive(Debug, Default, Clone)]
pub struct CostCLinear {
    n: usize,
    values: Vec<f64>,
}

impl CostCLinear {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cost for CostCLinear {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        if !signal.is_univariate() {
            return Err(ForecastError::InvalidParameter(
                "CostCLinear: univariate signals only".into(),
            ));
        }
        self.n = signal.n();
        self.values = (0..self.n).map(|i| signal.row(i)[0]).collect();
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostCLinear: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let len = end - start;
        if len < 2 {
            return Ok(0.0);
        }
        let x_start = self.values[start];
        let x_end = self.values[end - 1];
        let denom = (len - 1) as f64;
        let mut rss = 0.0_f64;
        for i in 0..len {
            let yhat = x_start + (i as f64) * (x_end - x_start) / denom;
            let r = self.values[start + i] - yhat;
            rss += r * r;
        }
        Ok(rss.max(0.0))
    }

    fn min_size(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "clinear"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfect_line_zero_cost() {
        let values: Vec<f64> = (0..20).map(|i| 3.0 + 2.0 * i as f64).collect();
        let s = Signal::univariate(&values);
        let mut c = CostCLinear::new();
        c.fit(&s).unwrap();
        assert_relative_eq!(c.error(0, 20).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(c.error(5, 15).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn curve_has_positive_cost() {
        let values: Vec<f64> = (0..20).map(|i| (i as f64).powi(2)).collect();
        let s = Signal::univariate(&values);
        let mut c = CostCLinear::new();
        c.fit(&s).unwrap();
        assert!(c.error(0, 20).unwrap() > 0.0);
    }

    #[test]
    fn univariate_only() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let s = Signal::from_row_major(&data, 2, 2).unwrap();
        let mut c = CostCLinear::new();
        assert!(c.fit(&s).is_err());
    }
}
