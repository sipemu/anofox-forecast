//! Kernel-based changepoint detection (`KernelCPD`).
//!
//! Maps the signal into a reproducing-kernel Hilbert space via a chosen
//! kernel and runs exact dynamic programming on the resulting kernel
//! cost. Mirrors `ruptures.detection.KernelCPD`.
//!
//! ## Memory
//!
//! The 2-D Gram cumulative-sum matrix is O(n²) in memory. For very long
//! signals prefer one of the standard detectors with a non-kernel cost.

use crate::changepoint::detector::{Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Kernel choice for [`KernelCpdDetector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelKind {
    /// Linear kernel `K(x, y) = x · y`.
    Linear,
    /// Radial basis function kernel `K(x, y) = exp(−γ ‖x − y‖²)`.
    Rbf { gamma: f64 },
    /// Cosine kernel `K(x, y) = (x · y) / (‖x‖ · ‖y‖)`. Returns 0 when
    /// either vector has zero norm.
    Cosine,
}

/// Kernel changepoint detector.
///
/// Supports `predict_n_bkps(K)` via exact dynamic programming on the
/// kernel cost. Penalty mode is supported via a K-sweep, identical to
/// [`DynpDetector`](super::dynp::DynpDetector).
#[derive(Debug, Clone)]
pub struct KernelCpdDetector {
    kernel: KernelKind,
    min_size: usize,
    jump: usize,
    n: usize,
    // Diagonal cumulative sum: cum_diag[t] = Σ_{i=0..t} K(x_i, x_i).
    cum_diag: Vec<f64>,
    // 2-D cumulative gram: gram2d[(a, b)] = Σ_{i<a, j<b} K(x_i, x_j).
    // Row-major flat layout, shape (n+1) × (n+1).
    gram2d: Vec<f64>,
    fitted: bool,
}

impl KernelCpdDetector {
    pub fn new(kernel: KernelKind) -> Self {
        Self {
            kernel,
            min_size: 2,
            jump: 1,
            n: 0,
            cum_diag: Vec::new(),
            gram2d: Vec::new(),
            fitted: false,
        }
    }

    pub fn min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size.max(1);
        self
    }

    pub fn jump(mut self, jump: usize) -> Self {
        self.jump = jump.max(1);
        self
    }

    pub fn kernel(&self) -> KernelKind {
        self.kernel
    }

    /// Cost of segment `[start, end)`.
    ///
    /// `error = Σ_i K(x_i, x_i) − (1/n_seg) · Σ_{i,j} K(x_i, x_j)`.
    fn error(&self, start: usize, end: usize) -> Result<f64> {
        if !self.fitted {
            return Err(ForecastError::FitRequired {
                model: Some("KernelCpdDetector".into()),
            });
        }
        if end <= start || end > self.n {
            return Err(ForecastError::InvalidParameter(format!(
                "KernelCPD: invalid segment [{}, {})",
                start, end
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

    #[inline]
    fn gram2d_at(&self, a: usize, b: usize) -> f64 {
        self.gram2d[a * (self.n + 1) + b]
    }
}

impl KernelKind {
    fn evaluate(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            KernelKind::Linear => x.iter().zip(y.iter()).map(|(a, b)| a * b).sum(),
            KernelKind::Rbf { gamma } => {
                let sq: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-gamma * sq).exp()
            }
            KernelKind::Cosine => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                let ny: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
                if nx == 0.0 || ny == 0.0 {
                    0.0
                } else {
                    dot / (nx * ny)
                }
            }
        }
    }
}

impl Detector for KernelCpdDetector {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        let n = signal.n();
        self.n = n;
        self.cum_diag = vec![0.0; n + 1];
        self.gram2d = vec![0.0; (n + 1) * (n + 1)];

        // Build the kernel matrix on the fly into the 2-D cumsum.
        // gram2d[a, b] = sum over i<a, j<b of K(x_i, x_j).
        // Use the relation: gram2d[a+1, b+1] = gram2d[a, b+1] + gram2d[a+1, b]
        //                                     − gram2d[a, b] + K(x_a, x_b)
        // and fill row by row.
        for i in 0..n {
            let xi = signal.row(i);
            self.cum_diag[i + 1] = self.cum_diag[i] + self.kernel.evaluate(xi, xi);
            for j in 0..n {
                let xj = signal.row(j);
                let k = self.kernel.evaluate(xi, xj);
                let a = (i + 1) * (n + 1) + (j + 1);
                let top = self.gram2d[i * (n + 1) + (j + 1)];
                let left = self.gram2d[(i + 1) * (n + 1) + j];
                let diag = self.gram2d[i * (n + 1) + j];
                self.gram2d[a] = top + left - diag + k;
            }
        }
        self.fitted = true;
        Ok(())
    }

    fn predict_n_bkps(&self, n_bkps: usize) -> Result<DetectorResult> {
        if !self.fitted {
            return Err(ForecastError::FitRequired {
                model: Some("KernelCpdDetector".into()),
            });
        }
        let n = self.n;
        if n == 0 {
            return Ok(DetectorResult { bkps: vec![0] });
        }
        let min_size = self.min_size;
        let segments_needed = n_bkps + 1;
        if n < segments_needed * min_size {
            return Err(ForecastError::InvalidParameter(format!(
                "KernelCPD: signal of length {} cannot host {} segments of min size {}",
                n, segments_needed, min_size
            )));
        }
        if n_bkps == 0 {
            return Ok(DetectorResult { bkps: vec![n] });
        }

        // Same DP shape as DynpDetector.
        let mut f = vec![vec![f64::INFINITY; n + 1]; segments_needed];
        let mut prev = vec![vec![0usize; n + 1]; segments_needed];
        for t in min_size..=n {
            f[0][t] = self.error(0, t)?;
        }
        for k in 1..segments_needed {
            let s_lo = k * min_size;
            for t in (k + 1) * min_size..=n {
                let mut best = f64::INFINITY;
                let mut best_s = 0usize;
                let s_hi = t - min_size;
                let mut s = s_lo;
                while s <= s_hi {
                    if f[k - 1][s].is_finite() {
                        let c = self.error(s, t)?;
                        let total = f[k - 1][s] + c;
                        if total < best {
                            best = total;
                            best_s = s;
                        }
                    }
                    s += self.jump;
                    if s > s_hi {
                        break;
                    }
                }
                f[k][t] = best;
                prev[k][t] = best_s;
            }
        }

        let mut bkps = vec![n];
        let mut t = n;
        for k in (1..segments_needed).rev() {
            t = prev[k][t];
            bkps.push(t);
        }
        bkps.reverse();
        if bkps.first() == Some(&0) {
            bkps.remove(0);
        }
        Ok(DetectorResult { bkps })
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        if !self.fitted {
            return Err(ForecastError::FitRequired {
                model: Some("KernelCpdDetector".into()),
            });
        }
        let n = self.n;
        if n == 0 {
            return Ok(DetectorResult { bkps: vec![0] });
        }
        let min_size = self.min_size;
        let k_max = n
            .saturating_sub(min_size)
            .checked_div(min_size)
            .unwrap_or(0);
        let mut best = f64::INFINITY;
        let mut best_r: Option<DetectorResult> = None;
        for k in 0..=k_max {
            let r = self.predict_n_bkps(k)?;
            let mut total = 0.0;
            let mut start = 0usize;
            for &end in &r.bkps {
                if end > start {
                    total += self.error(start, end)?;
                }
                start = end;
            }
            let scored = total + pen * k as f64;
            if scored < best {
                best = scored;
                best_r = Some(r);
            }
        }
        best_r.ok_or_else(|| {
            ForecastError::ComputationError("KernelCPD: penalty sweep yielded no result".into())
        })
    }

    fn name(&self) -> &str {
        "KernelCPD"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level_shift(n_per: usize, levels: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(n_per * levels.len());
        for &lvl in levels {
            out.extend(std::iter::repeat(lvl).take(n_per));
        }
        out
    }

    #[test]
    fn kernel_cpd_linear_locates_level_shift() {
        let series = level_shift(20, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = KernelCpdDetector::new(KernelKind::Linear).min_size(5);
        Detector::fit(&mut d, &s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        assert_eq!(r.bkps[0], 20);
    }

    #[test]
    fn kernel_cpd_rbf_locates_level_shift() {
        let series = level_shift(15, &[0.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = KernelCpdDetector::new(KernelKind::Rbf { gamma: 0.1 }).min_size(3);
        Detector::fit(&mut d, &s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        let cp = r.bkps[0];
        assert!((13..=17).contains(&cp));
    }

    #[test]
    fn kernel_cpd_zero_bkps() {
        let series = level_shift(20, &[0.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = KernelCpdDetector::new(KernelKind::Linear).min_size(5);
        Detector::fit(&mut d, &s).unwrap();
        let r = d.predict_n_bkps(0).unwrap();
        assert_eq!(r.bkps, vec![40]);
    }

    #[test]
    fn predict_before_fit_errors() {
        let d = KernelCpdDetector::new(KernelKind::Linear);
        assert!(d.predict_n_bkps(1).is_err());
    }
}
