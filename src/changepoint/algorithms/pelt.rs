//! Trait-based PELT (Pruned Exact Linear Time) detector.
//!
//! Exact dynamic-programming changepoint detection with a pruning rule
//! that achieves average O(n) runtime under mild conditions. Equivalent
//! to `ruptures.detection.Pelt`.

use crate::changepoint::detector::{Cost, Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// PELT detector parameterised by a [`Cost`].
///
/// Use the builder methods [`min_size`](Self::min_size) and
/// [`jump`](Self::jump) to mirror ruptures' `Pelt(min_size, jump)`
/// constructor.
#[derive(Debug, Clone)]
pub struct PeltDetector<C: Cost> {
    cost: C,
    min_size: usize,
    jump: usize,
    n: Option<usize>,
}

impl<C: Cost> PeltDetector<C> {
    /// Construct a new detector with the given cost. Defaults:
    /// `min_size = max(2, cost.min_size())`, `jump = 1`.
    pub fn new(cost: C) -> Self {
        let min_size = cost.min_size().max(2);
        Self {
            cost,
            min_size,
            jump: 1,
            n: None,
        }
    }

    /// Set the minimum segment length.
    pub fn min_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size.max(self.cost.min_size()).max(1);
        self
    }

    /// Set the step size between candidate breakpoints. `jump = 1`
    /// considers every position; larger values trade resolution for
    /// speed.
    pub fn jump(mut self, jump: usize) -> Self {
        self.jump = jump.max(1);
        self
    }

    /// Borrow the internal cost.
    pub fn cost(&self) -> &C {
        &self.cost
    }
}

impl<C: Cost> Detector for PeltDetector<C> {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.cost.fit(signal)?;
        self.n = Some(signal.n());
        Ok(())
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("PeltDetector".into()),
        })?;
        if n == 0 {
            return Ok(DetectorResult { bkps: vec![0] });
        }
        let min_size = self.min_size.max(self.cost.min_size());
        if n < 2 * min_size {
            // Not enough data for any internal changepoints.
            return Ok(DetectorResult { bkps: vec![n] });
        }

        let mut f = vec![f64::INFINITY; n + 1];
        f[0] = -pen;
        let mut prev = vec![0usize; n + 1];
        let mut admissible: Vec<usize> = vec![0];

        let mut t = min_size;
        loop {
            let mut best = f64::INFINITY;
            let mut best_s = 0usize;
            let mut costs: Vec<(usize, f64)> = Vec::with_capacity(admissible.len());

            for &s in &admissible {
                if t < s + min_size {
                    continue;
                }
                let c = self.cost.error(s, t)?;
                let total = f[s] + c + pen;
                if total < best {
                    best = total;
                    best_s = s;
                }
                costs.push((s, c));
            }

            // Defensive: if no admissible s could score the segment,
            // bail out (shouldn't happen given the n >= 2*min_size guard).
            if !best.is_finite() {
                return Err(ForecastError::ComputationError(format!(
                    "PELT: no admissible predecessor for t = {}",
                    t
                )));
            }

            f[t] = best;
            prev[t] = best_s;

            // Pruning: drop s if f[s] + c(s, t) > f[t].
            let mut next_admissible: Vec<usize> = Vec::with_capacity(admissible.len() + 1);
            let mut idx = 0usize;
            for &s in &admissible {
                if t < s + min_size {
                    // Couldn't be scored against t — keep for future iterations.
                    next_admissible.push(s);
                    continue;
                }
                // Already paired with a cost in `costs` at index `idx`.
                let (s_check, c) = costs[idx];
                debug_assert_eq!(s_check, s);
                idx += 1;
                if f[s] + c <= f[t] {
                    next_admissible.push(s);
                }
            }
            next_admissible.push(t);
            admissible = next_admissible;

            if t == n {
                break;
            }
            t = (t + self.jump).min(n);
        }

        // Backtrack.
        let mut bkps = Vec::new();
        let mut t = n;
        while t > 0 {
            bkps.push(t);
            t = prev[t];
        }
        bkps.reverse();
        Ok(DetectorResult { bkps })
    }

    fn name(&self) -> &str {
        "Pelt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::changepoint::costs::CostL2;
    use approx::assert_relative_eq;

    fn make_level_shift(n_per: usize, levels: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(n_per * levels.len());
        for &lvl in levels {
            out.extend(std::iter::repeat(lvl).take(n_per));
        }
        out
    }

    #[test]
    fn pelt_detects_single_level_shift() {
        let series = make_level_shift(50, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = PeltDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_pen(2.0).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        assert_eq!(r.bkps.last(), Some(&100));
        assert_relative_eq!(r.bkps[0] as f64, 50.0, epsilon = 1.0);
    }

    #[test]
    fn pelt_detects_three_segments() {
        let series = make_level_shift(40, &[0.0, 5.0, 0.0]);
        let s = Signal::univariate(&series);
        let mut d = PeltDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_pen(1.0).unwrap();
        assert_eq!(r.n_changepoints(), 2);
        // CPs should be near 40 and 80.
        assert!(r.bkps[0] >= 38 && r.bkps[0] <= 42);
        assert!(r.bkps[1] >= 78 && r.bkps[1] <= 82);
        assert_eq!(*r.bkps.last().unwrap(), 120);
    }

    #[test]
    fn pelt_no_changepoint_when_flat() {
        let series = vec![3.0; 100];
        let s = Signal::univariate(&series);
        let mut d = PeltDetector::new(CostL2::new());
        d.fit(&s).unwrap();
        let r = d.predict_pen(1.0).unwrap();
        assert_eq!(r.n_changepoints(), 0);
        assert_eq!(r.bkps, vec![100]);
    }

    #[test]
    fn pelt_fit_predict_pen_shortcut() {
        let series = make_level_shift(30, &[1.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = PeltDetector::new(CostL2::new()).min_size(5);
        let r = d.fit_predict_pen(&s, 2.0).unwrap();
        assert_eq!(r.n_changepoints(), 1);
    }

    #[test]
    fn predict_before_fit_errors() {
        let d = PeltDetector::new(CostL2::new());
        let err = d.predict_pen(1.0).unwrap_err();
        assert!(matches!(err, ForecastError::FitRequired { .. }));
    }

    #[test]
    fn small_series_returns_no_changepoints() {
        let series = vec![1.0, 2.0, 3.0];
        let s = Signal::univariate(&series);
        let mut d = PeltDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_pen(1.0).unwrap();
        assert_eq!(r.bkps, vec![3]);
    }
}
