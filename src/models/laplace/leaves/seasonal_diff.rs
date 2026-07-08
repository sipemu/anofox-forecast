//! Seasonal-difference wrapper — port of skaters' `seasonal_difference`.
//!
//! ```text
//!   y'_t = y_t - y_{t - period}       ← inner leaf sees this
//! ```
//!
//! On predict, shifts each horizon's Gaussian by the appropriate anchor
//! from the running buffer (or, for horizons past `period`, by the
//! anchor recovered at horizon `h - period`).
//!
//! Skaters composes this with EMA at `period ∈ {7, 12, 24}` × `α ∈
//! {0.05, 0.1}` in its depth-2 pool — 6 candidates. The point is to
//! give the softmax pool a candidate that models the residual **after
//! removing an s-lag seasonal**, without paying for a full seasonal
//! decomposition.
//!
//! PR #4 of #180.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;

pub struct SeasonalDifferenceWrapper {
    inner: Box<dyn Leaf + Send>,
    period: usize,
    // Rolling buffer of the last `2 * period` raw observations.
    buffer: Vec<f64>,
    label: String,
}

impl SeasonalDifferenceWrapper {
    /// `period` = lag `s` for `y_t - y_{t-s}`. Skaters ships `{7, 12, 24}`
    /// composed with EMA at `α ∈ {0.05, 0.1}`.
    pub fn new(inner: Box<dyn Leaf + Send>, period: usize) -> Self {
        let period = period.max(1);
        let label = format!("{}@sd{period}", inner.name());
        Self {
            inner,
            period,
            buffer: Vec::with_capacity(2 * period + 1),
            label,
        }
    }
}

impl Leaf for SeasonalDifferenceWrapper {
    fn name(&self) -> &'static str {
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let inner = self.inner.predict(horizon);
        let buf = &self.buffer;
        let s = self.period;
        // For h in 0..horizon, the anchor at prediction step h+1 is:
        //   - if h < s: the observation at buf[len - s + h]
        //   - else: the recovered mean at horizon h - s
        let mut recovered_means: Vec<f64> = Vec::with_capacity(horizon);
        let mut out: Vec<Gaussian> = Vec::with_capacity(horizon);
        for h in 0..horizon {
            let anchor = if h < s {
                let buf_idx = buf.len().saturating_sub(s).saturating_add(h);
                if buf_idx < buf.len() {
                    buf[buf_idx]
                } else {
                    0.0
                }
            } else {
                recovered_means[h - s]
            };
            let g = inner[h];
            let mean_out = g.mean + anchor;
            recovered_means.push(mean_out);
            out.push(Gaussian::new(mean_out, g.std));
        }
        out
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        let y_prime = if self.buffer.len() >= self.period {
            let idx = self.buffer.len() - self.period;
            y - self.buffer[idx]
        } else {
            0.0
        };
        self.buffer.push(y);
        // Keep the buffer bounded — 2× period gives room for the
        // inverse recovery at h < period without unbounded growth.
        if self.buffer.len() > 2 * self.period {
            self.buffer.remove(0);
        }
        self.inner.observe(y_prime);
    }
}

#[cfg(test)]
mod tests {
    use super::super::EmaLeaf;
    use super::*;

    #[test]
    fn recovers_period_7_pattern() {
        // Deterministic weekly pattern: y_t = day_of_week * 10.
        let mut w = SeasonalDifferenceWrapper::new(Box::new(EmaLeaf::new(0.5)), 7);
        for t in 0..200 {
            let dow = (t % 7) as f64;
            w.observe(dow * 10.0);
        }
        // Next expected observation: continuing the pattern, dow=200%7=4 → 40.
        let g = w.predict(1)[0];
        assert!(
            (g.mean - 40.0).abs() < 5.0,
            "period-7 anchor prediction {} not near 40",
            g.mean
        );
    }

    #[test]
    fn multi_horizon_uses_correct_anchors() {
        // Same weekly pattern. Predicting 8 steps ahead should
        // recover the next weekly value.
        let mut w = SeasonalDifferenceWrapper::new(Box::new(EmaLeaf::new(0.5)), 7);
        for t in 0..100 {
            let dow = (t % 7) as f64;
            w.observe(dow * 10.0);
        }
        let g = w.predict(8);
        // At h=8, the anchor is the h=1 recovered mean (h - s = 1).
        // The differenced-space forecast is ≈ 0 (steady periodic),
        // so g[7].mean should be close to g[0].mean.
        assert!(
            (g[7].mean - g[0].mean).abs() < 5.0,
            "multi-horizon anchor recovery failed: g[0]={}, g[7]={}",
            g[0].mean,
            g[7].mean
        );
    }

    #[test]
    fn nan_is_ignored() {
        let mut w = SeasonalDifferenceWrapper::new(Box::new(EmaLeaf::new(0.1)), 7);
        for t in 0..20 {
            w.observe(t as f64);
        }
        let before_len = w.buffer.len();
        w.observe(f64::NAN);
        w.observe(f64::INFINITY);
        assert_eq!(w.buffer.len(), before_len);
    }
}
