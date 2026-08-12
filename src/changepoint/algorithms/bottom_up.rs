//! Bottom-Up (agglomerative) detector.
//!
//! Starts with all candidate breakpoints (at every `jump` position) and
//! iteratively *removes* the breakpoint whose removal costs the least
//! (cheapest merge). Mirrors `ruptures.detection.BottomUp`.

use crate::changepoint::detector::{Cost, Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Bottom-up agglomerative detector.
///
/// Supports `predict_n_bkps(K)` and `predict_pen(pen)` (stops merging
/// when the merge cost falls above `pen`).
#[derive(Debug, Clone)]
pub struct BottomUpDetector<C: Cost> {
    cost: C,
    min_size: usize,
    jump: usize,
    n: Option<usize>,
}

impl<C: Cost> BottomUpDetector<C> {
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

impl<C: Cost> BottomUpDetector<C> {
    /// Initial set of breakpoints: every `min_size` positions, then add `n`.
    fn initial_bkps(&self, n: usize) -> Vec<usize> {
        let min_size = self.min_size.max(self.cost.min_size());
        if n < 2 * min_size {
            return vec![n];
        }
        let step = self.jump.max(min_size);
        let mut bkps = Vec::new();
        let mut b = step;
        while b < n - min_size + 1 {
            bkps.push(b);
            b += step;
        }
        bkps.push(n);
        bkps
    }

    /// Cost of removing breakpoint at index i in the bkps list:
    /// `cost(left ∪ right) - (cost(left) + cost(right))`.
    fn merge_cost(&self, bkps: &[usize], i: usize) -> Result<f64> {
        // Segments around bkps[i] are:
        //   left  = [prev, bkps[i])
        //   right = [bkps[i], next)
        // After merge: [prev, next).
        let prev = if i == 0 { 0 } else { bkps[i - 1] };
        let curr = bkps[i];
        let next = bkps[i + 1];
        let left = self.cost.error(prev, curr)?;
        let right = self.cost.error(curr, next)?;
        let merged = self.cost.error(prev, next)?;
        Ok(merged - (left + right))
    }
}

impl<C: Cost> Detector for BottomUpDetector<C> {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.cost.fit(signal)?;
        self.n = Some(signal.n());
        Ok(())
    }

    fn predict_n_bkps(&self, n_bkps: usize) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("BottomUpDetector".into()),
        })?;
        let mut bkps = self.initial_bkps(n);
        let internal = bkps.len().saturating_sub(1);
        if n_bkps >= internal {
            return Ok(DetectorResult { bkps });
        }
        let target_internal = n_bkps;

        // Iteratively remove the cheapest internal breakpoint, leaving the
        // terminal `n` in place. Stops once internal count reaches target.
        while bkps.len() - 1 > target_internal {
            if bkps.len() <= 1 {
                break;
            }
            let mut best_i = 0usize;
            let mut best_cost = f64::INFINITY;
            // We can remove any breakpoint except the last (the terminal n).
            for i in 0..(bkps.len() - 1) {
                let c = self.merge_cost(&bkps, i)?;
                if c < best_cost {
                    best_cost = c;
                    best_i = i;
                }
            }
            bkps.remove(best_i);
        }
        Ok(DetectorResult { bkps })
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("BottomUpDetector".into()),
        })?;
        let mut bkps = self.initial_bkps(n);
        loop {
            if bkps.len() <= 2 {
                break;
            }
            let mut best_i = 0usize;
            let mut best_cost = f64::INFINITY;
            for i in 0..(bkps.len() - 1) {
                let c = self.merge_cost(&bkps, i)?;
                if c < best_cost {
                    best_cost = c;
                    best_i = i;
                }
            }
            // Merging is "worth it" while merge_cost ≤ pen.
            if best_cost > pen {
                break;
            }
            bkps.remove(best_i);
        }
        Ok(DetectorResult { bkps })
    }

    fn name(&self) -> &str {
        "BottomUp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::changepoint::costs::CostL2;

    fn level_shift(n_per: usize, levels: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(n_per * levels.len());
        for &lvl in levels {
            out.extend(std::iter::repeat_n(lvl, n_per));
        }
        out
    }

    #[test]
    fn bottom_up_single_changepoint() {
        let series = level_shift(30, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = BottomUpDetector::new(CostL2::new()).min_size(5).jump(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        // Bottom-up locates the CP at one of the candidate positions.
        let cp = r.bkps[0];
        assert!((25..=35).contains(&cp));
    }

    #[test]
    fn bottom_up_zero_bkps_collapses() {
        let series = level_shift(20, &[3.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = BottomUpDetector::new(CostL2::new()).min_size(5).jump(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(0).unwrap();
        assert_eq!(r.bkps, vec![40]);
    }

    #[test]
    fn bottom_up_penalty_stops_when_merges_dont_help() {
        let series = level_shift(20, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = BottomUpDetector::new(CostL2::new()).min_size(5).jump(5);
        d.fit(&s).unwrap();
        // Huge pen → no removals; very low pen → only large changepoints survive.
        let r_high_pen = d.predict_pen(0.0).unwrap();
        assert!(r_high_pen.n_changepoints() >= 1);
    }
}
