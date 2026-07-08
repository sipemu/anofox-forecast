//! Drift leaf — level plus a smoothed step size.
//!
//! Point forecast at horizon `h`: `level + h · drift`, where `drift` is an
//! EMA of first differences. Predictive std grows as `σ · √h`.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Batch-initialize the drift estimate from an OLS fit `y ~ α + β·t`
    /// (yearly Trick 1). Sets `level` to the last training value and
    /// `drift` to the fitted slope β, so the streaming updates start
    /// from a good linear-trend estimate rather than converging from
    /// zero.
    ///
    /// Only useful when the series looks trending. Callers should
    /// check `is_trending(values)` before choosing this over
    /// [`Self::new`].
    pub fn from_batch(alpha: f64, values: &[f64]) -> Self {
        let mut leaf = Self::new(alpha);
        if values.len() < 5 {
            return leaf;
        }
        let (_, beta) = ols_slope(values);
        leaf.level = values.last().copied();
        leaf.drift = beta;
        leaf.prev = values.last().copied();
        leaf
    }
}

/// Simple OLS fit `y ~ α + β·t` for equally-spaced values.
/// Returns `(α, β)`. Used by [`DriftLeaf::from_batch`] and
/// [`crate::models::laplace::leaves::HoltLeaf::from_batch`].
pub(crate) fn ols_slope(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n < 2 {
        return (values.first().copied().unwrap_or(0.0), 0.0);
    }
    let n_f = n as f64;
    let mean_t = (n_f - 1.0) / 2.0;
    let mean_y: f64 = values.iter().sum::<f64>() / n_f;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let dt = i as f64 - mean_t;
        num += dt * (y - mean_y);
        den += dt * dt;
    }
    let beta = if den > 1e-12 { num / den } else { 0.0 };
    let alpha = mean_y - beta * mean_t;
    (alpha, beta)
}

impl DriftLeaf {
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

    #[inline]
    fn predict_one(&self) -> Gaussian {
        Gaussian::new(self.level.unwrap_or(0.0) + self.drift, self.sigma())
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
