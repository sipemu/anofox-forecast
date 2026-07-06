//! Drift leaf — level plus a smoothed step size.
//!
//! Point forecast at horizon `h`: `level + h · drift`, where `drift` is an
//! EMA of first differences. Predictive std grows as `σ · √h`.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

pub struct DriftLeaf {
    alpha: f64,
    level: Option<f64>,
    drift: f64,
    prev: Option<f64>,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl DriftLeaf {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            level: None,
            drift: 0.0,
            prev: None,
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

impl Leaf for DriftLeaf {
    fn name(&self) -> &'static str {
        "drift"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let level = self.level.unwrap_or(0.0);
        let base = self.sigma();
        (1..=horizon)
            .map(|h| Gaussian::new(level + h as f64 * self.drift, base * (h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let predicted = self.level.map(|l| l + self.drift).unwrap_or(y);
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);

        if let Some(prev) = self.prev {
            let step = y - prev;
            self.drift = self.alpha * step + (1.0 - self.alpha) * self.drift;
        }
        self.level = Some(match self.level {
            Some(l) => self.alpha * y + (1.0 - self.alpha) * l,
            None => y,
        });
        self.prev = Some(y);
    }
}
