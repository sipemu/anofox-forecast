//! Dynamic-Programming (Dynp) detector.
//!
//! Exact O(K · n²) algorithm that finds the segmentation minimising the
//! total cost for a *fixed* number of changepoints. Equivalent to
//! `ruptures.detection.Dynp`.

use crate::changepoint::detector::{Cost, Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Exact dynamic-programming detector — supports `predict_n_bkps`.
///
/// Time complexity: O(K · n²). Space: O(K · n).
///
/// For very long series prefer [`PeltDetector`](super::pelt::PeltDetector)
/// with a penalty.
#[derive(Debug, Clone)]
pub struct DynpDetector<C: Cost> {
    cost: C,
    min_size: usize,
    jump: usize,
    n: Option<usize>,
}

impl<C: Cost> DynpDetector<C> {
    /// Construct with the given cost. Defaults: `min_size = max(2, cost.min_size())`, `jump = 1`.
    pub fn new(cost: C) -> Self {
        let min_size = cost.min_size().max(2);
        Self {
            cost,
            min_size,
            jump: 1,
            n: None,
        }
    }

    pub fn min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size.max(self.cost.min_size()).max(1);
        self
    }

    pub fn jump(mut self, jump: usize) -> Self {
        self.jump = jump.max(1);
        self
    }

    pub fn cost(&self) -> &C {
        &self.cost
    }
}

impl<C: Cost> Detector for DynpDetector<C> {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.cost.fit(signal)?;
        self.n = Some(signal.n());
        Ok(())
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        // Dynp doesn't natively use penalty mode; sweep K = 0..=K_max
        // and pick the K minimising total_cost(K) + pen·K.
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("DynpDetector".into()),
        })?;
        let min_size = self.min_size.max(self.cost.min_size());
        let k_max = n
            .saturating_sub(min_size)
            .checked_div(min_size)
            .unwrap_or(0);

        let mut best_cost = f64::INFINITY;
        let mut best_result: Option<DetectorResult> = None;
        for k in 0..=k_max {
            let r = self.predict_n_bkps(k)?;
            let cost = self.compute_total_cost(&r)? + pen * k as f64;
            if cost < best_cost {
                best_cost = cost;
                best_result = Some(r);
            }
        }
        best_result.ok_or_else(|| {
            ForecastError::ComputationError("Dynp: penalty sweep yielded no result".into())
        })
    }

    fn predict_n_bkps(&self, n_bkps: usize) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("DynpDetector".into()),
        })?;
        if n == 0 {
            return Ok(DetectorResult { bkps: vec![0] });
        }
        let min_size = self.min_size.max(self.cost.min_size());
        let segments_needed = n_bkps + 1;
        if n < segments_needed * min_size {
            return Err(ForecastError::InvalidParameter(format!(
                "Dynp: signal of length {} cannot host {} segments of min size {}",
                n, segments_needed, min_size
            )));
        }
        if n_bkps == 0 {
            return Ok(DetectorResult { bkps: vec![n] });
        }

        // F[k][t] = min cost of segmenting signal[0..t] into k+1 segments
        // ending at t. F[0][t] = cost(0, t) for valid t.
        let mut f = vec![vec![f64::INFINITY; n + 1]; segments_needed];
        let mut prev = vec![vec![0usize; n + 1]; segments_needed];

        // Base case: one segment from 0 to t.
        for t in min_size..=n {
            f[0][t] = self.cost.error(0, t)?;
        }

        // Recurrence.
        for k in 1..segments_needed {
            // F[k][t] = min over s in [k*min_size, t - min_size] of F[k-1][s] + cost(s, t)
            let s_lo = k * min_size;
            for t in (k + 1) * min_size..=n {
                let mut best = f64::INFINITY;
                let mut best_s = 0usize;
                let s_hi = t - min_size;
                let mut s = s_lo;
                while s <= s_hi {
                    if f[k - 1][s].is_finite() {
                        let c = self.cost.error(s, t)?;
                        let total = f[k - 1][s] + c;
                        if total < best {
                            best = total;
                            best_s = s;
                        }
                    }
                    s = (s + self.jump).min(s_hi + 1);
                    if s > s_hi {
                        break;
                    }
                }
                f[k][t] = best;
                prev[k][t] = best_s;
            }
        }

        if !f[n_bkps][n].is_finite() {
            return Err(ForecastError::ComputationError(format!(
                "Dynp: no valid segmentation into {} segments for signal of length {}",
                segments_needed, n
            )));
        }

        // Backtrack.
        let mut bkps = vec![n];
        let mut t = n;
        for k in (1..segments_needed).rev() {
            t = prev[k][t];
            bkps.push(t);
        }
        bkps.reverse();
        // bkps now starts with 0 (the very first segment start) — drop it.
        if bkps.first() == Some(&0) {
            bkps.remove(0);
        }
        Ok(DetectorResult { bkps })
    }

    fn name(&self) -> &str {
        "Dynp"
    }
}

impl<C: Cost> DynpDetector<C> {
    fn compute_total_cost(&self, r: &DetectorResult) -> Result<f64> {
        let mut total = 0.0;
        let mut start = 0usize;
        for &end in &r.bkps {
            if end > start {
                total += self.cost.error(start, end)?;
            }
            start = end;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::changepoint::costs::CostL2;

    fn level_shift(n_per: usize, levels: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(n_per * levels.len());
        for &lvl in levels {
            out.extend(std::iter::repeat(lvl).take(n_per));
        }
        out
    }

    #[test]
    fn dynp_single_changepoint() {
        let series = level_shift(30, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = DynpDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        assert_eq!(r.bkps[0], 30);
        assert_eq!(*r.bkps.last().unwrap(), 60);
    }

    #[test]
    fn dynp_three_segments() {
        let series = level_shift(20, &[0.0, 5.0, 0.0]);
        let s = Signal::univariate(&series);
        let mut d = DynpDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(2).unwrap();
        assert_eq!(r.n_changepoints(), 2);
        assert_eq!(r.bkps, vec![20, 40, 60]);
    }

    #[test]
    fn dynp_zero_changepoints_is_single_segment() {
        let series = level_shift(20, &[3.0, 7.0]);
        let s = Signal::univariate(&series);
        let mut d = DynpDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(0).unwrap();
        assert_eq!(r.bkps, vec![40]);
    }

    #[test]
    fn dynp_too_many_bkps_errors() {
        let series = vec![0.0; 10];
        let s = Signal::univariate(&series);
        let mut d = DynpDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        // 10 / 5 = 2 segments max → 1 bkp max
        assert!(d.predict_n_bkps(5).is_err());
    }

    #[test]
    fn dynp_penalty_mode_picks_best_k() {
        let series = level_shift(20, &[0.0, 10.0, 20.0]);
        let s = Signal::univariate(&series);
        let mut d = DynpDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        // Low pen → 2 CPs; high pen → 0 CPs.
        let r_low = d.predict_pen(0.01).unwrap();
        let r_high = d.predict_pen(1e6).unwrap();
        assert!(r_low.n_changepoints() >= 2);
        assert_eq!(r_high.n_changepoints(), 0);
    }
}
