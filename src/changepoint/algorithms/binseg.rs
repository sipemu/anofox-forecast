//! Binary Segmentation (Binseg) detector.
//!
//! Greedy: find the single best split, then recurse on both halves.
//! Approximate (does not always find the global optimum) but fast —
//! O(n log n) on most inputs. Mirrors `ruptures.detection.Binseg`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::changepoint::detector::{Cost, Detector, DetectorResult};
use crate::changepoint::signal::Signal;
use crate::error::{ForecastError, Result};

/// Binary-segmentation detector.
///
/// Supports both `predict_n_bkps(K)` and `predict_pen(pen)` (greedy
/// growth until the marginal gain falls below `pen`).
#[derive(Debug, Clone)]
pub struct BinsegDetector<C: Cost> {
    cost: C,
    min_size: usize,
    jump: usize,
    n: Option<usize>,
}

impl<C: Cost> BinsegDetector<C> {
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

/// One candidate split. `gain = parent_cost - (left_cost + right_cost)`.
/// A larger gain is a better split.
#[derive(Debug, Clone, Copy)]
struct Split {
    gain: f64,
    parent_start: usize,
    parent_end: usize,
    bkp: usize,
}

impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for Split {}
impl PartialOrd for Split {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Split {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap; we want the largest gain at the top.
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(Ordering::Equal)
            // Tie-break on parent_start for determinism.
            .then(other.parent_start.cmp(&self.parent_start))
    }
}

impl<C: Cost> BinsegDetector<C> {
    /// Find the best internal split of segment `[start, end)`.
    ///
    /// Returns `None` when no valid split exists (segment too short).
    fn best_split(&self, start: usize, end: usize) -> Result<Option<Split>> {
        let min_size = self.min_size.max(self.cost.min_size());
        if end < start + 2 * min_size {
            return Ok(None);
        }
        let parent = self.cost.error(start, end)?;
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_bkp = 0usize;
        let mut b = start + min_size;
        while b + min_size <= end {
            let left = self.cost.error(start, b)?;
            let right = self.cost.error(b, end)?;
            let gain = parent - (left + right);
            if gain > best_gain {
                best_gain = gain;
                best_bkp = b;
            }
            b = (b + self.jump).min(end - min_size);
            if b == end - min_size {
                // Score one more then break.
                let left = self.cost.error(start, b)?;
                let right = self.cost.error(b, end)?;
                let gain = parent - (left + right);
                if gain > best_gain {
                    best_gain = gain;
                    best_bkp = b;
                }
                break;
            }
        }
        if !best_gain.is_finite() {
            return Ok(None);
        }
        Ok(Some(Split {
            gain: best_gain,
            parent_start: start,
            parent_end: end,
            bkp: best_bkp,
        }))
    }
}

impl<C: Cost> Detector for BinsegDetector<C> {
    fn fit(&mut self, signal: &Signal) -> Result<()> {
        self.cost.fit(signal)?;
        self.n = Some(signal.n());
        Ok(())
    }

    fn predict_n_bkps(&self, n_bkps: usize) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("BinsegDetector".into()),
        })?;
        if n_bkps == 0 || n == 0 {
            return Ok(DetectorResult {
                bkps: if n == 0 { vec![0] } else { vec![n] },
            });
        }

        let mut heap = BinaryHeap::new();
        if let Some(s) = self.best_split(0, n)? {
            heap.push(s);
        }

        let mut bkps: Vec<usize> = Vec::with_capacity(n_bkps + 1);
        bkps.push(n);

        for _ in 0..n_bkps {
            let s = match heap.pop() {
                Some(s) if s.gain.is_finite() => s,
                _ => break,
            };
            bkps.push(s.bkp);
            // Add sub-splits on the two child segments.
            if let Some(left) = self.best_split(s.parent_start, s.bkp)? {
                heap.push(left);
            }
            if let Some(right) = self.best_split(s.bkp, s.parent_end)? {
                heap.push(right);
            }
        }

        bkps.sort_unstable();
        bkps.dedup();
        Ok(DetectorResult { bkps })
    }

    fn predict_pen(&self, pen: f64) -> Result<DetectorResult> {
        let n = self.n.ok_or(ForecastError::FitRequired {
            model: Some("BinsegDetector".into()),
        })?;
        if n == 0 {
            return Ok(DetectorResult { bkps: vec![0] });
        }

        let mut heap = BinaryHeap::new();
        if let Some(s) = self.best_split(0, n)? {
            heap.push(s);
        }
        let mut bkps: Vec<usize> = vec![n];

        while let Some(s) = heap.pop() {
            if !s.gain.is_finite() || s.gain <= pen {
                break;
            }
            bkps.push(s.bkp);
            if let Some(left) = self.best_split(s.parent_start, s.bkp)? {
                if left.gain > pen {
                    heap.push(left);
                }
            }
            if let Some(right) = self.best_split(s.bkp, s.parent_end)? {
                if right.gain > pen {
                    heap.push(right);
                }
            }
        }
        bkps.sort_unstable();
        bkps.dedup();
        Ok(DetectorResult { bkps })
    }

    fn name(&self) -> &str {
        "Binseg"
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
    fn binseg_single_changepoint() {
        let series = level_shift(30, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = BinsegDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(1).unwrap();
        assert_eq!(r.n_changepoints(), 1);
        assert_eq!(r.bkps[0], 30);
    }

    #[test]
    fn binseg_three_segments_via_n_bkps() {
        let series = level_shift(25, &[0.0, 5.0, 0.0]);
        let s = Signal::univariate(&series);
        let mut d = BinsegDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        let r = d.predict_n_bkps(2).unwrap();
        assert_eq!(r.n_changepoints(), 2);
        assert_eq!(r.bkps, vec![25, 50, 75]);
    }

    #[test]
    fn binseg_penalty_stops_early() {
        let series = level_shift(20, &[0.0, 10.0]);
        let s = Signal::univariate(&series);
        let mut d = BinsegDetector::new(CostL2::new()).min_size(5);
        d.fit(&s).unwrap();
        // Huge pen → no CPs.
        let r = d.predict_pen(1e9).unwrap();
        assert_eq!(r.n_changepoints(), 0);
    }
}
