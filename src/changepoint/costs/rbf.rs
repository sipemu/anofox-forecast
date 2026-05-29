//! RBF (radial basis function) kernel cost.
//!
//! Mirrors `ruptures.costs.CostRbf`. Uses
//! `K(x, y) = exp(−γ · ‖x − y‖²)`. `γ` defaults to a median-heuristic
//! value (1 / median of pairwise squared distances) when not specified.

use crate::changepoint::detector::Cost;
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// RBF kernel cost.
///
/// O(n²) memory — gram cumulative sums of the kernel matrix. For
/// long signals prefer `CostL2` plus penalty tuning.
#[derive(Debug, Clone)]
pub struct CostRbf {
    gamma: Option<f64>,
    n: usize,
    cum_diag: Vec<f64>,
    gram2d: Vec<f64>,
}

impl CostRbf {
    /// Construct with median-heuristic gamma (computed at fit time).
    pub fn auto() -> Self {
        Self {
            gamma: None,
            n: 0,
            cum_diag: Vec::new(),
            gram2d: Vec::new(),
        }
    }

    /// Construct with a fixed gamma.
    pub fn with_gamma(gamma: f64) -> Self {
        Self {
            gamma: Some(gamma),
            n: 0,
            cum_diag: Vec::new(),
            gram2d: Vec::new(),
        }
    }

    #[inline]
    fn gram2d_at(&self, a: usize, b: usize) -> f64 {
        self.gram2d[a * (self.n + 1) + b]
    }
}

impl Cost for CostRbf {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        let n = signal.n();
        self.n = n;

        // Estimate gamma if not supplied: 1 / median(||x_i - x_j||²) over
        // the upper-triangle (i < j). Approximate via sampling for large n.
        let gamma = match self.gamma {
            Some(g) => g,
            None => median_heuristic_gamma(signal),
        };

        self.cum_diag = vec![0.0; n + 1];
        self.gram2d = vec![0.0; (n + 1) * (n + 1)];
        for i in 0..n {
            let xi = signal.row(i);
            self.cum_diag[i + 1] = self.cum_diag[i] + 1.0; // K(x, x) = exp(0) = 1
            for j in 0..n {
                let xj = signal.row(j);
                let sq: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                let k = (-gamma * sq).exp();
                let a = (i + 1) * (n + 1) + (j + 1);
                let top = self.gram2d[i * (n + 1) + (j + 1)];
                let left = self.gram2d[(i + 1) * (n + 1) + j];
                let diag = self.gram2d[i * (n + 1) + j];
                self.gram2d[a] = top + left - diag + k;
            }
        }
        // Memoise the gamma actually used for diagnostics.
        self.gamma = Some(gamma);
        Ok(())
    }

    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "CostRbf: invalid segment [{}, {}) for n = {}",
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
        "rbf"
    }
}

fn median_heuristic_gamma(signal: &Signal) -> f64 {
    let n = signal.n();
    if n < 2 {
        return 1.0;
    }
    // Sample up to ~5_000 pairs for speed.
    let max_pairs = 5000;
    let total_pairs = n * (n - 1) / 2;
    let step = (total_pairs / max_pairs).max(1);
    let mut dists = Vec::new();
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if k % step == 0 {
                let xi = signal.row(i);
                let xj = signal.row(j);
                let sq: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                dists.push(sq);
            }
            k += 1;
        }
    }
    if dists.is_empty() {
        return 1.0;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = dists[dists.len() / 2];
    if med <= f64::EPSILON {
        1.0
    } else {
        1.0 / med
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_cost_finite_on_level_shift() {
        let values: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { 10.0 }).collect();
        let s = Signal::univariate(&values);
        let mut c = CostRbf::auto();
        c.fit(&s).unwrap();
        let e = c.error(0, 40).unwrap();
        assert!(e.is_finite() && e >= 0.0);
        // Splitting at 20 should reduce the kernel cost.
        let left = c.error(0, 20).unwrap();
        let right = c.error(20, 40).unwrap();
        assert!(left + right < e + 1e-6);
    }

    #[test]
    fn rbf_cost_with_explicit_gamma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = Signal::univariate(&values);
        let mut c = CostRbf::with_gamma(0.5);
        c.fit(&s).unwrap();
        assert!(c.error(0, 5).unwrap().is_finite());
    }
}
