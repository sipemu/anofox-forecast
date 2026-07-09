//! Exponentially-weighted level leaf.
//!
//! Point forecast at any horizon = current EMA of `y`. Predictive std
//! grows with horizon as `σ · √h` (random-walk on the level).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmaLeaf {
    alpha: f64,
    level: Option<f64>,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl EmaLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            level: None,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
        }
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        let var = self.ss / (self.n as f64 - 1.0);
        var.sqrt().max(1e-9)
    }
}

impl Leaf for EmaLeaf {
    fn name(&self) -> &'static str {
        "ema"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let level = self.level.unwrap_or(0.0);
        let base = self.sigma();
        (1..=horizon)
            .map(|h| Gaussian::new(level, base * (h as f64).sqrt()))
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        Gaussian::new(self.level.unwrap_or(0.0), self.sigma())
    }

    fn observe(&mut self, y: f64) {
        let predicted = self.level.unwrap_or(y);
        let resid = y - predicted;
        // Welford update on residuals.
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);
        self.level = Some(match self.level {
            Some(l) => self.alpha * y + (1.0 - self.alpha) * l,
            None => y,
        });
    }
}
