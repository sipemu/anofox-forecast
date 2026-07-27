//! Seasonal-Holt leaf — per-phase level + trend.
//!
//! Extension of [`super::SeasonalEmaLeaf`] (which tracks level per
//! phase) to a **Holt-style level + trend per phase**. For a series
//! with period `p`, each phase `k ∈ 0..p` maintains its own
//! `(level, trend)` pair. Trend is the level-change from one
//! occurrence of phase `k` to the next (one cycle later).
//!
//! Prediction for h-step is
//! `level[p_t] + cycles_ahead · trend[p_t]` where `p_t = (phase_step + h - 1) mod p`
//! and `cycles_ahead` is the number of full cycles from the last
//! observation of `p_t` to the h-step target.
//!
//! Motivation (skaters #113 idea 2): the fixed [`super::SeasonalEmaLeaf`]
//! is level-only. On trending seasonal series (real M-competition
//! monthly / tourism panels), each phase drifts across cycles — the
//! same month next year is not the same level as this year. A per-phase
//! Holt captures that within-phase trend natively, without needing a
//! separate Holt leaf to add trend on top of a seasonal base.
//!
//! Under symmetric noise around a stationary seasonal signal, the
//! trend converges to zero and the leaf degenerates to seasonal-EMA
//! (with a slight over-parametrization penalty from tracking the
//! zero-mean trend estimate).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonalHoltLeaf {
    period: usize,
    alpha_level: f64,
    alpha_trend: f64,
    phase_level: Vec<f64>,
    phase_trend: Vec<f64>,
    /// Times this phase has been observed. Values ≥ 2 mean a trend has
    /// been computed (needs 2 observations for a delta).
    phase_seen_count: Vec<usize>,
    /// Value of `self.n` (post-increment) at the time we last observed
    /// this phase. Used to compute `cycles_ahead` in prediction.
    /// 0 = never observed.
    phase_last_seen_n: Vec<usize>,
    phase_step: usize,
    global_ema: f64,
    n: usize,
    /// Running residual-variance state (Welford).
    ss: f64,
    mean_resid: f64,
}

impl SeasonalHoltLeaf {
    pub fn new(period: usize, alpha_level: f64, alpha_trend: f64) -> Self {
        let period = period.max(1);
        Self {
            period,
            alpha_level: alpha_level.clamp(1e-3, 1.0 - 1e-3),
            alpha_trend: alpha_trend.clamp(1e-3, 1.0 - 1e-3),
            phase_level: vec![0.0; period],
            phase_trend: vec![0.0; period],
            phase_seen_count: vec![0; period],
            phase_last_seen_n: vec![0; period],
            phase_step: 0,
            global_ema: 0.0,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    pub fn period(&self) -> usize {
        self.period
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }

    /// Point forecast for the observation that will occur at
    /// `absolute_step = self.n + h - 1` (h is 1-indexed).
    fn forecast_mean(&self, h: usize) -> f64 {
        let phase = (self.phase_step + h - 1) % self.period;
        let seen = self.phase_seen_count[phase];
        if seen == 0 {
            return self.global_ema;
        }
        let level = self.phase_level[phase];
        if seen == 1 {
            return level;
        }
        let last_seen = self.phase_last_seen_n[phase];
        // Cycles from the last observation of `phase` to the h-step target.
        // Absolute step of last observation: last_seen - 1.
        // Absolute step of h-step target: self.n + h - 1.
        // Δ = self.n + h - last_seen. Should be a positive multiple of period.
        let cycles = (self.n + h).saturating_sub(last_seen) / self.period;
        level + (cycles as f64) * self.phase_trend[phase]
    }
}

impl Leaf for SeasonalHoltLeaf {
    fn name(&self) -> &'static str {
        "seasonal_holt"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let base = self.sigma();
        (1..=horizon)
            .map(|h| Gaussian::new(self.forecast_mean(h), base * (h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            // Advance phase without touching state — matches SeasonalEmaLeaf
            // NaN behaviour (implicit: only updates on finite y).
            self.phase_step = (self.phase_step + 1) % self.period;
            return;
        }
        let phase = self.phase_step;
        // 1-step forecast from PRE-observation state — this is what the
        // leaf would have emitted. Residual = y - this.
        let predicted = self.forecast_mean(1);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        let seen = self.phase_seen_count[phase];
        if seen == 0 {
            // First observation of this phase: seed level, no trend yet.
            self.phase_level[phase] = y;
            self.phase_trend[phase] = 0.0;
        } else if seen == 1 {
            // Second observation — trend defined for the first time.
            let prev_level = self.phase_level[phase];
            let observed_trend = y - prev_level;
            let new_level = self.alpha_level * y + (1.0 - self.alpha_level) * prev_level;
            let new_trend = self.alpha_trend * observed_trend; // (1 - α_t) * 0
            self.phase_level[phase] = new_level;
            self.phase_trend[phase] = new_trend;
        } else {
            // Normal per-phase Holt update.
            let prev_level = self.phase_level[phase];
            let prev_trend = self.phase_trend[phase];
            let one_cycle_forecast = prev_level + prev_trend;
            let new_level = self.alpha_level * y + (1.0 - self.alpha_level) * one_cycle_forecast;
            let observed_trend = new_level - prev_level;
            let new_trend =
                self.alpha_trend * observed_trend + (1.0 - self.alpha_trend) * prev_trend;
            self.phase_level[phase] = new_level;
            self.phase_trend[phase] = new_trend;
        }

        self.phase_seen_count[phase] += 1;
        self.phase_last_seen_n[phase] = self.n;

        if self.n == 1 {
            self.global_ema = y;
        } else {
            self.global_ema = self.alpha_level * y + (1.0 - self.alpha_level) * self.global_ema;
        }

        self.phase_step = (self.phase_step + 1) % self.period;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_onto_stationary_cycle() {
        // Repeat [10, 20, 30] — trend should converge to 0, level to the phase mean.
        let cycle = [10.0, 20.0, 30.0];
        let mut leaf = SeasonalHoltLeaf::new(3, 0.3, 0.1);
        for _ in 0..200 {
            for &y in &cycle {
                leaf.observe(y);
            }
        }
        let preds = leaf.predict(3);
        assert!(
            (preds[0].mean - 10.0).abs() < 0.5,
            "h=1 → {}",
            preds[0].mean
        );
        assert!(
            (preds[1].mean - 20.0).abs() < 0.5,
            "h=2 → {}",
            preds[1].mean
        );
        assert!(
            (preds[2].mean - 30.0).abs() < 0.5,
            "h=3 → {}",
            preds[2].mean
        );
    }

    #[test]
    fn captures_within_phase_trend() {
        // Series with a linear year-over-year trend at each phase:
        // phase 0 grows by +1 per cycle, phase 1 by +2, phase 2 by +3.
        let mut leaf = SeasonalHoltLeaf::new(3, 0.5, 0.5);
        let n_cycles = 30usize;
        for c in 0..n_cycles {
            let cf = c as f64;
            leaf.observe(10.0 + cf); // phase 0
            leaf.observe(20.0 + 2.0 * cf); // phase 1
            leaf.observe(30.0 + 3.0 * cf); // phase 2
        }
        // Predict the next cycle (h=1..=3). Expected values continue the trend.
        let preds = leaf.predict(3);
        let expected = [
            10.0 + (n_cycles as f64),
            20.0 + 2.0 * (n_cycles as f64),
            30.0 + 3.0 * (n_cycles as f64),
        ];
        for (h, (p, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert!(
                (p.mean - e).abs() < 2.0,
                "h={}: predicted {} vs expected {}",
                h + 1,
                p.mean,
                e
            );
        }
    }

    #[test]
    fn multi_cycle_ahead_projects_trend_linearly() {
        // Same construction but predict 2 cycles ahead — level should
        // continue linearly with 2 trend steps.
        let mut leaf = SeasonalHoltLeaf::new(3, 0.5, 0.5);
        for c in 0..30 {
            let cf = c as f64;
            leaf.observe(10.0 + cf);
            leaf.observe(20.0 + 2.0 * cf);
            leaf.observe(30.0 + 3.0 * cf);
        }
        // 2 cycles ahead = h=4, 5, 6.
        let preds = leaf.predict(6);
        // At h=4 (phase 0, cycle 31): expect ≈ 10 + 31 = 41
        // At h=5 (phase 1, cycle 31): expect ≈ 20 + 2·31 = 82
        // At h=6 (phase 2, cycle 31): expect ≈ 30 + 3·31 = 123
        let expected_far = [41.0, 82.0, 123.0];
        for (i, e) in expected_far.iter().enumerate() {
            let h = 4 + i;
            assert!(
                (preds[h - 1].mean - e).abs() < 4.0,
                "h={}: predicted {} vs expected {}",
                h,
                preds[h - 1].mean,
                e
            );
        }
    }

    #[test]
    fn period_one_degenerates_reasonably() {
        // Period 1 → all observations are the same phase → this reduces
        // to Holt over the raw series.
        let mut leaf = SeasonalHoltLeaf::new(1, 0.3, 0.1);
        for y in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            leaf.observe(y);
        }
        let preds = leaf.predict(2);
        // Both horizons should give finite predictions; and since the
        // trend is captured, h=2 should be somewhat higher than h=1.
        for p in &preds {
            assert!(p.mean.is_finite());
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }

    #[test]
    fn cold_start_unseen_phase_uses_global_ema() {
        let mut leaf = SeasonalHoltLeaf::new(4, 0.4, 0.2);
        leaf.observe(5.0);
        leaf.observe(7.0);
        // Phases 2 and 3 have never been observed; forecast for those
        // horizons should fall back to the global EMA (not NaN or 0).
        let preds = leaf.predict(4);
        for (i, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}: not finite", i + 1);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }
}
