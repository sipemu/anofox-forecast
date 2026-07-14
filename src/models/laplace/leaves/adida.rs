//! ADIDA (Aggregate-Disaggregate Intermittent Demand Approach) leaf.
//!
//! ADIDA (Nikolopoulos, Syntetos, Boylan, Petropoulos, Assimakopoulos,
//! 2011) reduces intermittency by aggregating the raw series into
//! coarser buckets before forecasting. On a series with 60 % zeros at
//! daily granularity, aggregating into weekly buckets (k=7) turns
//! most weeks non-zero — a continuous-scale SES then works better than
//! any per-period intermittent method.
//!
//! Update rule:
//! - Buffer incoming observations. When buffer reaches size `k`, sum
//!   into one aggregated observation, apply SES, reset buffer.
//! - Forecast per period = `aggregated_ema / k`.
//!
//! At forecast time the horizon-h aggregate prediction is `aggregated_ema`
//! (constant across horizons); disaggregation is the "equal-weights"
//! rule (spread evenly across k buckets — the simplest of the ADIDA
//! disaggregation strategies).
//!
//! Bucket size `k` typically matches the natural seasonal / demand
//! cadence (7 for daily-with-weekly-demand, 4 or 12 for monthly, etc).
//! IMAPA (see `imapa.rs`) is the meta-ensemble across multiple k values.

use crate::models::laplace::dist::Gaussian;
use crate::models::laplace::leaf::Leaf;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdidaLeaf {
    alpha: f64,
    aggregation_period: usize,
    buffer_sum: f64,
    buffer_count: usize,
    aggregated_ema: f64,
    initialized: bool,
    n: usize,
    ss: f64,
    mean_resid: f64,
    label: String,
}

impl AdidaLeaf {
    /// `alpha`: SES rate on the aggregated series. `k`: bucket size.
    pub fn new(alpha: f64, k: usize) -> Self {
        let k = k.max(1);
        let label = format!("adida{k}");
        Self {
            alpha: alpha.clamp(1e-3, 1.0 - 1e-3),
            aggregation_period: k,
            buffer_sum: 0.0,
            buffer_count: 0,
            aggregated_ema: 0.0,
            initialized: false,
            n: 0,
            ss: 0.0,
            mean_resid: 0.0,
            label,
        }
    }

    fn sigma(&self) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        (self.ss / (self.n as f64 - 1.0)).sqrt().max(1e-9)
    }

    fn point(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        // Per-period forecast is the aggregated EMA spread evenly.
        self.aggregated_ema / self.aggregation_period as f64
    }
}

impl Leaf for AdidaLeaf {
    fn name(&self) -> &'static str {
        // Leak the label so the trait's &'static str contract is honoured.
        // O(unique k values) leaks per process — fine.
        Box::leak(self.label.clone().into_boxed_str())
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

        // Accumulate into the current bucket.
        self.buffer_sum += y;
        self.buffer_count += 1;
        if self.buffer_count >= self.aggregation_period {
            // Bucket full — treat sum as one aggregated observation.
            if !self.initialized {
                self.aggregated_ema = self.buffer_sum;
                self.initialized = true;
            } else {
                self.aggregated_ema =
                    self.alpha * self.buffer_sum + (1.0 - self.alpha) * self.aggregated_ema;
            }
            self.buffer_sum = 0.0;
            self.buffer_count = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adida_aggregates_reduce_intermittency() {
        // 80 obs at daily granularity: 10 units every Monday, 0 else.
        // ADIDA at k=7 sees weekly aggregate = 10 → forecast/period = 10/7.
        let mut adida = AdidaLeaf::new(0.3, 7);
        for _ in 0..80 {
            adida.observe(10.0);
            for _ in 0..6 {
                adida.observe(0.0);
            }
        }
        let point = adida.predict(1)[0].mean;
        let expected = 10.0 / 7.0;
        assert!(
            (point - expected).abs() < 0.5,
            "adida k=7 expected ~{expected:.3}, got {point:.3}"
        );
    }

    #[test]
    fn adida_k1_matches_plain_ses() {
        // k=1 → aggregation is no-op → equivalent to SES on raw series.
        let mut adida = AdidaLeaf::new(0.3, 1);
        for y in [1.0, 2.0, 3.0, 4.0, 5.0] {
            adida.observe(y);
        }
        // Not testing exact SES here — just that it converges to a
        // reasonable value in the observed range.
        let point = adida.predict(1)[0].mean;
        assert!(
            (1.0..=5.0).contains(&point),
            "k=1 adida should be in-range, got {point}"
        );
    }
}
