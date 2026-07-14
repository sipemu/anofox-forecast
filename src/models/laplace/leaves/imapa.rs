//! IMAPA (Multiple Aggregation Prediction Algorithm) leaf.
//!
//! IMAPA (Petropoulos & Kourentzes, 2015) is a meta-ensemble of ADIDA
//! forecasters at multiple aggregation levels. Each level captures a
//! different seasonal / demand cadence; averaging their predictions
//! reduces the choice of a single k (which ADIDA requires).
//!
//! On the M4 competition intermittent panel IMAPA has slightly better
//! MASE than any single ADIDA. In our ensemble it provides a
//! multi-scale intermittent prediction whose per-scale weights are
//! implicit (uniform across levels).
//!
//! Default k grid: `{1, 2, 3, 4, 6, 8, 12}` (Petropoulos & Kourentzes'
//! choice — covers daily-to-monthly cadences without redundancy).

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

use super::AdidaLeaf;

const DEFAULT_KS: &[usize] = &[1, 2, 3, 4, 6, 8, 12];

pub struct ImapaLeaf {
    adidas: Vec<AdidaLeaf>,
    n: usize,
    ss: f64,
    mean_resid: f64,
}

impl ImapaLeaf {
    /// `alpha`: SES rate used by every ADIDA level.
    pub fn new(alpha: f64) -> Self {
        Self::with_grid(alpha, DEFAULT_KS)
    }

    /// Custom aggregation grid.
    pub fn with_grid(alpha: f64, ks: &[usize]) -> Self {
        let adidas = ks.iter().map(|&k| AdidaLeaf::new(alpha, k)).collect();
        Self {
            adidas,
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

    fn point(&self) -> f64 {
        if self.adidas.is_empty() {
            return 0.0;
        }
        // Uniform average across levels — simplest IMAPA formulation.
        let sum: f64 = self.adidas.iter().map(|a| a.predict(1)[0].mean).sum();
        sum / self.adidas.len() as f64
    }
}

impl Leaf for ImapaLeaf {
    fn name(&self) -> &'static str {
        "imapa"
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        let point = self.point();
        let base = self.sigma();
        (1..=horizon)
            .map(|h| Gaussian::new(point, base * (h as f64).sqrt()))
            .collect()
    }

    fn observe(&mut self, y: f64) {
        let predicted = self.point();
        let resid = y - predicted;
        self.n += 1;
        let delta = resid - self.mean_resid;
        self.mean_resid += delta / self.n as f64;
        self.ss += delta * (resid - self.mean_resid);
        // Broadcast to every ADIDA level.
        for a in self.adidas.iter_mut() {
            a.observe(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imapa_produces_finite_predictions() {
        let mut imapa = ImapaLeaf::new(0.3);
        for _ in 0..80 {
            imapa.observe(10.0);
            for _ in 0..6 {
                imapa.observe(0.0);
            }
        }
        let preds = imapa.predict(4);
        for p in preds {
            assert!(p.mean.is_finite() && p.mean >= 0.0);
            assert!(p.std.is_finite() && p.std > 0.0);
        }
    }

    #[test]
    fn imapa_forecast_between_adida_extremes() {
        // On a steady demand pattern the IMAPA forecast should lie
        // between the highest and lowest single-level ADIDA forecasts.
        let mut imapa = ImapaLeaf::new(0.3);
        for _ in 0..80 {
            imapa.observe(10.0);
            for _ in 0..6 {
                imapa.observe(0.0);
            }
        }
        let imapa_point = imapa.predict(1)[0].mean;
        // Extract per-level forecasts:
        let level_points: Vec<f64> = imapa.adidas.iter().map(|a| a.predict(1)[0].mean).collect();
        let lo = level_points.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = level_points
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            imapa_point >= lo - 1e-9 && imapa_point <= hi + 1e-9,
            "imapa {imapa_point} outside [{lo}, {hi}]"
        );
    }
}
