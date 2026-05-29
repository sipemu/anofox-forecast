//! Cosine kernel cost.
//!
//! Mirrors `ruptures.costs.CostCosine`. Uses
//! `K(x, y) = (x · y) / (‖x‖ · ‖y‖)`. Detects shifts in direction even
//! when magnitude is similar.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Cosine kernel cost.
#[derive(Debug, Default, Clone)]
pub struct CostCosine {
    n: usize,
    cum_diag: Vec<f64>,
    gram2d: Vec<f64>,
}

impl CostCosine {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn gram2d_at(&self, a: usize, b: usize) -> f64 {
        self.gram2d[a * (self.n + 1) + b]
    }
}

impl Cost for CostCosine {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        let n = signal.n();
        self.n = n;
        self.cum_diag = vec![0.0; n + 1];
        self.gram2d = vec![0.0; (n + 1) * (n + 1)];

        let mut norms = vec![0.0_f64; n];
        for i in 0..n {
            let xi = signal.row(i);
            norms[i] = xi.iter().map(|v| v * v).sum::<f64>().sqrt();
        }

        for i in 0..n {
            let xi = signal.row(i);
            // K(x_i, x_i) = 1 when norm > 0, else 0.
            let kii = if norms[i] > 0.0 { 1.0 } else { 0.0 };
            self.cum_diag[i + 1] = self.cum_diag[i] + kii;
            for j in 0..n {
                let xj = signal.row(j);
                let dot: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum();
                let k = if norms[i] > 0.0 && norms[j] > 0.0 {
                    dot / (norms[i] * norms[j])
                } else {
                    0.0
                };
                let a = (i + 1) * (n + 1) + (j + 1);
                let top = self.gram2d[i * (n + 1) + (j + 1)];
                let left = self.gram2d[(i + 1) * (n + 1) + j];
                let diag = self.gram2d[i * (n + 1) + j];
                self.gram2d[a] = top + left - diag + k;
            }
        }
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostCosine: invalid segment [{}, {}) for n = {}",
                start, end, self.n
            )));
        }
        let n_seg = (end - start) as f64;
        let diag = self.cum_diag[end] - self.cum_diag[start];
        let g_ee = self.gram2d_at(end, end);
        let g_es = self.gram2d_at(end, start);
        let g_se = self.gram2d_at(start, end);
        let g_ss = self.gram2d_at(start, start);
        let gram = g_ee - g_es - g_se + g_ss;
        Ok((diag - gram / n_seg).max(0.0))
    }

    fn min_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "cosine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_cost_finite() {
        let data = vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0];
        let s = Signal::from_row_major(&data, 4, 2).unwrap();
        let mut c = CostCosine::new();
        c.fit(&s).unwrap();
        assert!(c.error(0, 4).unwrap().is_finite());
    }
}
