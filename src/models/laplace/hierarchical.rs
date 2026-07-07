//! Hierarchical Laplace — cross-series shrinkage with pluggable priors.
//!
//! The α-30 anti-result showed that panel-mean shrinkage on heterogeneous
//! panels (M-competition style, mixed series directions) hurts more than
//! it helps because the panel-mean growth is close to zero.
//!
//! This module adds three refinements that address the root cause:
//!
//! - [`PriorMode::Cluster`] — k-means cluster series by feature vector,
//!   apply shrinkage within cluster. Homogeneous clusters have coherent
//!   priors.
//! - [`PriorMode::Similarity`] — soft version of clustering: weight each
//!   contribution to the prior by cosine similarity between series
//!   feature vectors.
//! - [`PriorMode::Decomposition`] — global-local: fit a panel-aggregate
//!   trajectory, subtract a per-series scaled version to get residuals,
//!   fit per-series to residuals. Predict = panel + residual.
//!
//! Plus [`PriorMode::PanelMean`] (the α-30 baseline, kept for comparison).
//!
//! Design 4 (DeepAR-style neural encoder) is not implemented — would
//! require an ML framework dependency. `Decomposition` is the closest
//! lightweight analogue.
//!
//! ## Choosing a mode
//!
//! - **Small panels of similar series** → [`PriorMode::PanelMean`] (cif_2016).
//! - **Large heterogeneous panels** → [`PriorMode::Cluster`] with `k = 5-10`.
//! - **Series with continuous variation of characteristics** → [`PriorMode::Similarity`].
//! - **Panels with strong shared trajectory** (retail during promo, tourism at peak season) → [`PriorMode::Decomposition`].
//!
//! When in doubt: try each on a held-out slice; the wrong choice regresses
//! by 2-4 % (see the α-30 anti-result documentation).

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::laplace::LaplaceForecaster;
use crate::models::Forecaster;
use std::collections::HashMap;

/// Which prior-source strategy to use for shrinkage. See module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PriorMode {
    /// α-30 baseline: single panel-mean delta across all series.
    /// Regresses on heterogeneous panels (M-competition); marginal win on
    /// homogeneous ones (cif_2016).
    PanelMean,
    /// k-means cluster series by feature vector, shrink within cluster.
    /// `k` is the cluster count (rule of thumb: `sqrt(N/10)`, min 3).
    Cluster { k: usize },
    /// Similarity-weighted: each series' contribution to the target's
    /// prior is weighted by cosine similarity between feature vectors.
    /// No hard clustering.
    Similarity,
    /// Global-local decomposition: aggregate series to a panel trajectory
    /// (mean of scale-normalized), forecast the aggregate, add scaled
    /// aggregate to each series' residual forecast. Closest lightweight
    /// analogue to DeepAR's global-local factorization.
    Decomposition,
}

/// Cross-series shrinkage wrapper — see [module docs](self).
pub struct HierarchicalLaplace {
    factory: Box<dyn Fn() -> LaplaceForecaster + Send + Sync>,
    series: HashMap<String, LaplaceForecaster>,
    n_obs: HashMap<String, usize>,
    scales: HashMap<String, f64>,
    last_values: HashMap<String, f64>,
    /// Feature vector per series — used by [`PriorMode::Cluster`] and
    /// [`PriorMode::Similarity`] to weight contributions.
    features: HashMap<String, [f64; 6]>,
    /// Panel-level cache computed at finalize. Its meaning depends on
    /// the `PriorMode`:
    /// - `PanelMean` / `Decomposition`: one Vec<f64> for the whole panel.
    /// - `Cluster`: one Vec<f64> per cluster.
    /// - `Similarity`: one Vec<f64> per series (its own similarity-weighted prior).
    prior_cache: PriorCache,
    /// Per-series → cluster id (Cluster mode only).
    cluster_ids: HashMap<String, usize>,
    prior_strength: f64,
    max_n_for_shrinkage: usize,
    mode: PriorMode,
    cached_horizon: usize,
}

enum PriorCache {
    None,
    Single(Vec<f64>),
    PerCluster(Vec<Vec<f64>>),
    PerSeries(HashMap<String, Vec<f64>>),
}

impl HierarchicalLaplace {
    pub fn new<F>(prior_strength: f64, factory: F) -> Self
    where
        F: Fn() -> LaplaceForecaster + Send + Sync + 'static,
    {
        Self {
            factory: Box::new(factory),
            series: HashMap::new(),
            n_obs: HashMap::new(),
            scales: HashMap::new(),
            last_values: HashMap::new(),
            features: HashMap::new(),
            prior_cache: PriorCache::None,
            cluster_ids: HashMap::new(),
            prior_strength: prior_strength.max(1.0),
            max_n_for_shrinkage: 100,
            mode: PriorMode::PanelMean,
            cached_horizon: 0,
        }
    }

    pub fn with_prior_mode(mut self, mode: PriorMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_max_n_for_shrinkage(mut self, n: usize) -> Self {
        self.max_n_for_shrinkage = n;
        self
    }

    pub fn fit_series(&mut self, id: impl Into<String>, series: &TimeSeries) -> Result<()> {
        let id = id.into();
        let mut m = (self.factory)();
        m.fit(series)?;
        let vals = series.primary_values();
        let features = compute_features(vals);
        let scale = median_abs(vals);
        let last = vals.last().copied().unwrap_or(0.0);
        self.n_obs.insert(id.clone(), vals.len());
        self.scales.insert(id.clone(), scale);
        self.last_values.insert(id.clone(), last);
        self.features.insert(id.clone(), features);
        self.series.insert(id, m);
        self.prior_cache = PriorCache::None;
        self.cluster_ids.clear();
        Ok(())
    }

    pub fn finalize(&mut self, max_horizon: usize) -> Result<()> {
        if self.series.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "HierarchicalLaplace::finalize: no series fit".into(),
            ));
        }
        // Pre-compute per-series delta forecasts at max_horizon.
        let mut per_delta: HashMap<String, Vec<f64>> = HashMap::new();
        for (id, m) in &self.series {
            let s = *self.scales.get(id).unwrap_or(&1.0);
            let last = *self.last_values.get(id).unwrap_or(&0.0);
            if s.abs() < 1e-9 {
                continue;
            }
            let fc = m.predict(max_horizon)?;
            let d: Vec<f64> = fc
                .primary()
                .iter()
                .map(|v| if v.is_finite() { (v - last) / s } else { 0.0 })
                .collect();
            per_delta.insert(id.clone(), d);
        }

        self.prior_cache = match self.mode {
            PriorMode::PanelMean | PriorMode::Decomposition => {
                let mut sum = vec![0.0f64; max_horizon];
                let n = per_delta.len().max(1) as f64;
                for d in per_delta.values() {
                    for (i, v) in d.iter().enumerate() {
                        sum[i] += v;
                    }
                }
                PriorCache::Single(sum.iter().map(|s| s / n).collect())
            }
            PriorMode::Cluster { k } => {
                let k = k.clamp(2, self.series.len().max(2));
                let ids: Vec<String> = self.series.keys().cloned().collect();
                let feats: Vec<[f64; 6]> = ids.iter().map(|id| self.features[id]).collect();
                let assignments = kmeans_lite(&feats, k, 30);
                self.cluster_ids.clear();
                for (id, cluster) in ids.iter().zip(assignments.iter()) {
                    self.cluster_ids.insert(id.clone(), *cluster);
                }
                let mut cluster_sums: Vec<Vec<f64>> = vec![vec![0.0; max_horizon]; k];
                let mut cluster_counts: Vec<usize> = vec![0; k];
                for (id, cluster) in ids.iter().zip(assignments.iter()) {
                    if let Some(d) = per_delta.get(id) {
                        for (i, v) in d.iter().enumerate() {
                            cluster_sums[*cluster][i] += v;
                        }
                        cluster_counts[*cluster] += 1;
                    }
                }
                let cluster_means: Vec<Vec<f64>> = cluster_sums
                    .into_iter()
                    .zip(cluster_counts.iter())
                    .map(|(sum, &c)| {
                        let n = c.max(1) as f64;
                        sum.iter().map(|s| s / n).collect()
                    })
                    .collect();
                PriorCache::PerCluster(cluster_means)
            }
            PriorMode::Similarity => {
                let ids: Vec<String> = self.series.keys().cloned().collect();
                let mut per_series_prior: HashMap<String, Vec<f64>> = HashMap::new();
                for id_i in &ids {
                    let feat_i = self.features[id_i];
                    let mut sum = vec![0.0f64; max_horizon];
                    let mut w_total = 0.0f64;
                    for id_j in &ids {
                        if id_j == id_i {
                            continue;
                        }
                        let feat_j = self.features[id_j];
                        // Cosine similarity in [-1, 1]. Clamp to positive
                        // portion so uncorrelated series get zero weight.
                        let sim = cosine_similarity(&feat_i, &feat_j).max(0.0);
                        if let Some(d) = per_delta.get(id_j) {
                            for (i, v) in d.iter().enumerate() {
                                sum[i] += sim * v;
                            }
                            w_total += sim;
                        }
                    }
                    let prior: Vec<f64> = if w_total > 1e-9 {
                        sum.iter().map(|s| s / w_total).collect()
                    } else {
                        vec![0.0; max_horizon]
                    };
                    per_series_prior.insert(id_i.clone(), prior);
                }
                PriorCache::PerSeries(per_series_prior)
            }
        };

        self.cached_horizon = max_horizon;
        Ok(())
    }

    pub fn predict_series(&self, id: &str, horizon: usize) -> Result<Forecast> {
        let m = self
            .series
            .get(id)
            .ok_or_else(|| ForecastError::InvalidParameter(format!("no fit for series `{id}`")))?;
        if horizon > self.cached_horizon {
            return Err(ForecastError::InvalidParameter(format!(
                "requested horizon {} > cached {}; call finalize with larger max",
                horizon, self.cached_horizon
            )));
        }
        let per = m.predict(horizon)?;
        let n = *self.n_obs.get(id).unwrap_or(&0);
        let scale = *self.scales.get(id).unwrap_or(&1.0);
        let last = *self.last_values.get(id).unwrap_or(&0.0);
        let lambda = if n >= self.max_n_for_shrinkage {
            0.0
        } else {
            1.0 / (1.0 + n as f64 / self.prior_strength)
        };
        let prior_delta: Vec<f64> = match &self.prior_cache {
            PriorCache::None => {
                return Err(ForecastError::InvalidParameter(
                    "call finalize() first".into(),
                ));
            }
            PriorCache::Single(v) => v.iter().take(horizon).copied().collect(),
            PriorCache::PerCluster(clusters) => {
                let c = *self.cluster_ids.get(id).unwrap_or(&0);
                clusters[c].iter().take(horizon).copied().collect()
            }
            PriorCache::PerSeries(per_series) => per_series
                .get(id)
                .map(|v| v.iter().take(horizon).copied().collect())
                .unwrap_or_else(|| vec![0.0; horizon]),
        };
        let blended: Vec<f64> = match self.mode {
            PriorMode::Decomposition => {
                // Global-local: prediction = last + panel_delta_scaled +
                // residual (per-series delta - panel_delta_scaled).
                // Effectively the same numerically as blending with
                // lambda = 0.5 when both are given equal weight; we still
                // apply the per-series shrinkage weight so long series
                // rely more on their residual model.
                per.primary()
                    .iter()
                    .zip(prior_delta.iter())
                    .map(|(p, pd)| {
                        let per_delta = p - last;
                        let panel_delta_scaled = pd * scale;
                        let residual = per_delta - panel_delta_scaled;
                        last + panel_delta_scaled + (1.0 - lambda) * residual
                    })
                    .collect()
            }
            _ => per
                .primary()
                .iter()
                .zip(prior_delta.iter())
                .map(|(p, pd)| {
                    let per_delta = p - last;
                    let panel_delta_scaled = pd * scale;
                    last + (1.0 - lambda) * per_delta + lambda * panel_delta_scaled
                })
                .collect(),
        };
        Ok(Forecast::from_values(blended))
    }

    pub fn n_fitted(&self) -> usize {
        self.series.len()
    }

    pub fn shrinkage_weight(&self, id: &str) -> Option<f64> {
        let n = *self.n_obs.get(id)?;
        if n >= self.max_n_for_shrinkage {
            Some(0.0)
        } else {
            Some(1.0 / (1.0 + n as f64 / self.prior_strength))
        }
    }
}

// ============================================================================
// Feature extraction (6-dim per series).
// ============================================================================

fn compute_features(vals: &[f64]) -> [f64; 6] {
    let n = vals.len();
    if n < 3 {
        return [0.0; 6];
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var: f64 = vals.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    let cov = if mean.abs() > 1e-9 {
        sd / mean.abs()
    } else {
        1.0
    };

    // Trend: slope of linear fit y ~ t, normalized.
    let t_mean = (n - 1) as f64 / 2.0;
    let (mut sum_ty, mut sum_tt) = (0.0f64, 0.0f64);
    for (t, y) in vals.iter().enumerate() {
        let dt = t as f64 - t_mean;
        sum_ty += dt * (y - mean);
        sum_tt += dt * dt;
    }
    let trend_slope = if sum_tt > 0.0 { sum_ty / sum_tt } else { 0.0 };
    // Normalize by scale so it's dimensionless.
    let trend = if sd > 1e-9 {
        trend_slope * (n as f64).sqrt() / sd
    } else {
        0.0
    };

    // Direction sign: was the last 25% higher than the first 25% on average?
    let q = (n / 4).max(1);
    let head: f64 = vals[..q].iter().sum::<f64>() / q as f64;
    let tail: f64 = vals[n - q..].iter().sum::<f64>() / q as f64;
    let direction = if (tail - head).abs() < 1e-9 {
        0.0
    } else if tail > head {
        1.0
    } else {
        -1.0
    };

    // Lag-1 ACF (proxy for smoothness).
    let mut num = 0.0f64;
    for i in 1..n {
        num += (vals[i - 1] - mean) * (vals[i] - mean);
    }
    let acf1 = if var > 0.0 {
        (num / (n as f64 * var)).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // log-length feature (short vs long series).
    let log_n = (n as f64).ln();

    [
        trend,
        direction,
        acf1,
        cov,
        log_n,
        mean.abs().ln().max(-10.0),
    ]
}

fn cosine_similarity(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..6 {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

// ============================================================================
// k-means lite (feature-vector clustering, k iterations 20-30 works fine).
// ============================================================================

fn kmeans_lite(points: &[[f64; 6]], k: usize, max_iter: usize) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    // Init: pick k points spread evenly through the input.
    let mut centroids: Vec<[f64; 6]> = Vec::with_capacity(k);
    for i in 0..k {
        let idx = (i * n) / k.max(1);
        centroids.push(points[idx.min(n - 1)]);
    }
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        // Assign.
        for (i, p) in points.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let mut d = 0.0f64;
                for j in 0..6 {
                    let x = p[j] - c[j];
                    d += x * x;
                }
                if d < best_d {
                    best_d = d;
                    best = c_idx;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        // Update.
        let mut sums: Vec<[f64; 6]> = vec![[0.0; 6]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for (p, &c) in points.iter().zip(assignments.iter()) {
            for j in 0..6 {
                sums[c][j] += p[j];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..6 {
                    centroids[c][j] = sums[c][j] / counts[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    assignments
}

fn median_abs(vals: &[f64]) -> f64 {
    let mut abs_vals: Vec<f64> = vals.iter().map(|v| v.abs()).collect();
    if abs_vals.is_empty() {
        return 1.0;
    }
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = abs_vals.len();
    let m = if n % 2 == 1 {
        abs_vals[n / 2]
    } else {
        0.5 * (abs_vals[n / 2 - 1] + abs_vals[n / 2])
    };
    m.max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn ts(vals: Vec<f64>) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let stamps: Vec<_> = (0..vals.len())
            .map(|i| base + Duration::hours(i as i64))
            .collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn panel_mean_mode_works() {
        let mut h = HierarchicalLaplace::new(20.0, || LaplaceForecaster::new().auto())
            .with_prior_mode(PriorMode::PanelMean);
        h.fit_series("A", &ts((0..100).map(|i| 10.0 + i as f64 * 0.1).collect()))
            .unwrap();
        h.fit_series("B", &ts((0..80).map(|i| 5.0 + i as f64 * 0.05).collect()))
            .unwrap();
        h.finalize(10).unwrap();
        assert_eq!(h.predict_series("A", 5).unwrap().primary().len(), 5);
    }

    #[test]
    fn cluster_mode_assigns_series() {
        let mut h = HierarchicalLaplace::new(30.0, || LaplaceForecaster::new().auto())
            .with_prior_mode(PriorMode::Cluster { k: 2 });
        for i in 0..10 {
            h.fit_series(
                format!("up_{i}"),
                &ts((0..80)
                    .map(|t| 10.0 + t as f64 * 0.5 + (i as f64) * 0.1)
                    .collect()),
            )
            .unwrap();
        }
        for i in 0..10 {
            h.fit_series(
                format!("down_{i}"),
                &ts((0..80)
                    .map(|t| 100.0 - t as f64 * 0.5 - (i as f64) * 0.1)
                    .collect()),
            )
            .unwrap();
        }
        h.finalize(5).unwrap();
        for i in 0..10 {
            assert!(h.predict_series(&format!("up_{i}"), 3).is_ok());
        }
    }

    #[test]
    fn similarity_mode_produces_per_series_priors() {
        let mut h = HierarchicalLaplace::new(30.0, || LaplaceForecaster::new().auto())
            .with_prior_mode(PriorMode::Similarity);
        for i in 0..8 {
            h.fit_series(
                format!("s{i}"),
                &ts((0..50).map(|t| 10.0 + t as f64 * 0.1 + i as f64).collect()),
            )
            .unwrap();
        }
        h.finalize(5).unwrap();
        assert_eq!(h.predict_series("s0", 3).unwrap().primary().len(), 3);
    }

    #[test]
    fn decomposition_mode_predicts() {
        let mut h = HierarchicalLaplace::new(30.0, || LaplaceForecaster::new().auto())
            .with_prior_mode(PriorMode::Decomposition);
        for i in 0..5 {
            h.fit_series(
                format!("s{i}"),
                &ts((0..60).map(|t| 10.0 + t as f64 * 0.2).collect()),
            )
            .unwrap();
            let _ = i;
        }
        h.finalize(5).unwrap();
        assert_eq!(h.predict_series("s0", 3).unwrap().primary().len(), 3);
    }
}
