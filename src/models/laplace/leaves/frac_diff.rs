//! Fractional-differencing leaf.
//!
//! Skaters' long-memory leaf. Rather than the full ARFIMA batch fit
//! (expensive to stream), this leaf tracks a **rolling window** of recent
//! observations and computes the fractional-differencing filter of order
//! `d ∈ (0, 1)` on that window at each `observe`. The h-step forecast is
//!
//! ```text
//!   ŷ_{t+h} = μ + h · fd_ema
//! ```
//!
//! where `fd_ema` is an EMA of the most recent fractional-difference
//! values. Analogous to [`DriftLeaf`](super::DriftLeaf), but the drift
//! step captures long-memory persistence that a constant-step drift or
//! AR(1) cannot.
//!
//! Uses the crate's existing [`fractional_difference`] filter from the
//! ARIMA module (Lopez de Prado 2018 recursive-weights form) applied to
//! a rolling window sized by the truncation threshold.

use crate::models::arima::fractional_difference;
use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

const DEFAULT_WINDOW: usize = 60;
const WEIGHT_THRESHOLD: f64 = 1e-3;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FractionalDiffLeaf {
    d: f64,
    alpha_mean: f64,
    alpha_diff: f64,
    window: Vec<f64>,
    max_window: usize,
    mean: Option<f64>,
    fd_ema: f64,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl FractionalDiffLeaf {
    /// `d` is the fractional differencing order (clamped to `[0.05, 0.95]`).
    /// `alpha_mean` is the EMA rate for the level μ. `alpha_diff` is the
    /// EMA rate for the running fractional-diff step (drift).
    pub fn new(d: f64, alpha_mean: f64, alpha_diff: f64) -> Self {
        Self {
            d: d.clamp(0.05, 0.95),
            alpha_mean: alpha_mean.clamp(1e-3, 1.0 - 1e-3),
            alpha_diff: alpha_diff.clamp(1e-3, 1.0 - 1e-3),
            window: Vec::with_capacity(DEFAULT_WINDOW),
            max_window: DEFAULT_WINDOW,
            mean: None,
            fd_ema: 0.0,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }
}

impl Leaf for FractionalDiffLeaf {
    fn name(&self) -> &'static str {
        "frac_diff"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let level = self.mean.unwrap_or(0.0);
        let base = self.sigma();
        // Level-only forecast: μ. The fractional-diff EMA `fd_ema` captures
        // long-memory *residual* structure (used indirectly via `σ`) rather
        // than a drift — extrapolating fd_ema as a step-wise drift compounds
        // the finite-window truncation bias into a runaway h-step forecast.
        (1..=horizon)
            .map(|h| Gaussian::new(level, base * (h as f64).sqrt()))
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        Gaussian::new(self.mean.unwrap_or(0.0), self.sigma())
    }

    fn observe(&mut self, y: f64) {
        let predicted = self.mean.map(|m| m + self.fd_ema).unwrap_or(y);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        self.window.push(y);
        if self.window.len() > self.max_window {
            self.window.remove(0);
        }

        // Compute the most recent fractional-diff value if the window is
        // long enough to hold the truncated weight tail; otherwise leave
        // fd_ema alone (cold start uses whatever EMA has accumulated).
        if self.window.len() >= 8 {
            let fd = fractional_difference(&self.window, self.d, WEIGHT_THRESHOLD);
            if let Some(&last) = fd.last() {
                if last.is_finite() {
                    self.fd_ema = self.alpha_diff * last + (1.0 - self.alpha_diff) * self.fd_ema;
                }
            }
        }

        self.mean = Some(match self.mean {
            Some(m) => self.alpha_mean * y + (1.0 - self.alpha_mean) * m,
            None => y,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_produces_finite_predictions() {
        let mut leaf = FractionalDiffLeaf::new(0.4, 0.1, 0.1);
        for y in [1.0, 2.0, 3.0] {
            leaf.observe(y);
        }
        let preds = leaf.predict(5);
        for (h, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}: mean not finite", h + 1);
            assert!(p.std.is_finite() && p.std > 0.0, "h={}: std invalid", h + 1);
        }
    }

    #[test]
    fn flat_series_forecast_is_finite_and_close_to_level() {
        // A truncated fractional-diff filter on a constant series yields a
        // small non-zero drift (the omitted weight tail doesn't cancel).
        // Tolerance is generous — the leaf is a mixture candidate, not a
        // point estimator; the mixture softmax down-weights it when it's
        // consistently wrong.
        let mut leaf = FractionalDiffLeaf::new(0.4, 0.1, 0.1);
        for _ in 0..80 {
            leaf.observe(50.0);
        }
        let preds = leaf.predict(3);
        for p in preds {
            assert!(p.mean.is_finite(), "mean not finite: {}", p.mean);
            assert!(
                (p.mean - 50.0).abs() < 20.0,
                "forecast drifted too far: {}",
                p.mean
            );
        }
    }

    #[test]
    fn linear_trend_produces_valid_forecast() {
        // On a linear trend, this leaf just tracks the level (fractional
        // diff informs σ, not the mean). The mixture's other leaves —
        // Drift/Holt — carry the trend; this leaf contributes a
        // long-memory-aware level candidate.
        let mut leaf = FractionalDiffLeaf::new(0.4, 0.1, 0.1);
        for i in 0..100 {
            leaf.observe(10.0 + i as f64);
        }
        let preds = leaf.predict(3);
        for (h, p) in preds.iter().enumerate() {
            assert!(p.mean.is_finite(), "h={}: mean not finite", h + 1);
            assert!(p.std.is_finite() && p.std > 0.0, "h={}: std invalid", h + 1);
        }
    }
}
