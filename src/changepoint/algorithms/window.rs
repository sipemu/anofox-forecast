//! Sliding-window detector.
//!
//! For each candidate centre `t`, compares the cost of the left and
//! right windows against the cost of the merged window:
//!
//! ```text
//! score(t) = cost(t - w, t + w) − [cost(t - w, t) + cost(t, t + w)]
//! ```
//!
//! Peaks in `score` indicate likely changepoints. Mirrors
//! `ruptures.detection.Window`.

use crate::changepoint::detector::{Cost, Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Sliding-window detector.
///
/// Supports `predict_n_bkps(K)` (top-K non-conflicting peaks) and
/// `predict_pen(pen)` (all peaks above the threshold).
#[derive(Debug, Clone)]
pub struct WindowDetector<C: Cost> {
    cost: C,
    min_size: usize,
    jump: usize,
    width: usize,
    n: Option<usize>,
    scores: Vec<(usize, f64)>,
}

impl<C: Cost> WindowDetector<C> {
    /// Construct with the given cost. `width` is the half-window length.
    pub fn new(cost: C) -> Self {
        let min_size = cost.min_size().max(2);
        Self {
            cost,
            min_size,
            jump: 5,
            width: 100,
            n: None,
            scores: Vec::new(),
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

    /// Half-window length on each side of the candidate centre.
    pub fn width(mut self, width: usize) -> Self {
        self.width = width.max(self.cost.min_size()).max(1);
        self
    }

    pub fn cost(&self) -> &C {
        &self.cost
    }
}

impl<C: Cost> Detector for WindowDetector<C> {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.cost.fit(signal)?;
        let n = signal.n();
        self.n = Some(n);
        self.scores.clear();
        let width = self.width.max(self.cost.min_size()).max(1);
        if n < 2 * width {
            return Ok(());
        }
        let mut t = width;
        while t + width <= n {
            let left = self.cost.error(t - width, t)?;
            let right = self.cost.error(t, t + width)?;
            let merged = self.cost.error(t - width, t + width)?;
            let score = merged - (left + right);
            self.scores.push((t, score));
            t += self.jump;
        }
        Ok(())
    }

    fn predict_n_bkps(&self, n_bkps: usize) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("WindowDetector".into()),
        })?;
        if n_bkps == 0 || self.scores.is_empty() {
            return Ok(DetectorResult { bkps: vec![n] });
        }
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let min_gap = self.width.max(self.min_size);
        let mut chosen: Vec<usize> = Vec::with_capacity(n_bkps);
        for (t, _score) in &sorted {
            if chosen
                .iter()
                .any(|&c| (*t as isize - c as isize).abs() < min_gap as isize)
            {
                continue;
            }
            chosen.push(*t);
            if chosen.len() == n_bkps {
                break;
            }
        }
        chosen.sort_unstable();
        chosen.push(n);
        Ok(DetectorResult { bkps: chosen })
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("WindowDetector".into()),
        })?;
        let min_gap = self.width.max(self.min_size);
        // Greedily pick peaks in descending score order subject to pen and min_gap.
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut chosen: Vec<usize> = Vec::new();
        for (t, score) in &sorted {
            if *score <= pen {
                break;
            }
            if chosen
                .iter()
                .any(|&c| (*t as isize - c as isize).abs() < min_gap as isize)
            {
                continue;
            }
            chosen.push(*t);
        }
        chosen.sort_unstable();
        chosen.push(n);
        Ok(DetectorResult { bkps: chosen })
    }

    fn name(&self) -> &str {
        "Window"
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
    fn window_locates_level_shift_within_tolerance() {
        let series = level_shift(50, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = WindowDetector::new(CostL2::new())
            .width(20)
            .min_size(5)
            .jump(2);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        let cp = r.bkps[0];
        assert!((45..=55).contains(&cp));
    }

    #[test]
    fn window_zero_bkps_returns_terminal() {
        let series = level_shift(30, &[0.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = WindowDetector::new(CostL2::new())
            .width(10)
            .min_size(5)
            .jump(2);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(0).unwrap();
        assert_eq!(r.bkps, vec![60]);
    }

    #[test]
    fn window_high_pen_returns_no_bkps() {
        let series = level_shift(30, &[0.0, 5.0]);
        let s = Signal::univariate(&series);
        let mut d = WindowDetector::new(CostL2::new())
            .width(10)
            .min_size(5)
            .jump(2);
        d.fit(&s).unwrap();
        let r = d.predict_pen(1e9).unwrap();
        assert_eq!(r.bkps, vec![60]);
    }
}
