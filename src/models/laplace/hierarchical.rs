//! Hierarchical Laplace — cross-series Empirical-Bayes shrinkage.
//!
//! Wraps a panel of independent [`LaplaceForecaster`] instances with a
//! **panel-level prior** and blends each series' per-series forecast with
//! that prior. Short-history series (few observations) get pulled strongly
//! toward the panel; long series stay mostly on their own trajectory.
//!
//! ```text
//!   ŷ_series(h) = (1 − λ_n) · per_series(h) + λ_n · panel(h) · scale_series
//!
//!   λ_n = 1 / (1 + n_series / prior_strength)
//! ```
//!
//! Where:
//! - `per_series(h)` is the per-series [`LaplaceForecaster::predict`] mean.
//! - `panel(h)` is a **scale-normalised** average of every series' forecast
//!   (cached at [`Self::finalize`] time — `O(N)` per fit, `O(1)` per predict).
//! - `scale_series` is the per-series median absolute value used to
//!   rescale the normalized panel forecast into the series' original units.
//! - `n_series` is the training length of the specific series being
//!   forecast; `prior_strength` controls how quickly the shrinkage decays
//!   with `n`.
//!
//! ## Why this helps
//!
//! Classical demand-forecasting panels (M3-yearly, M1-yearly, tourism_yearly,
//! nn5) have short histories (30-100 obs per series) and dozens-to-hundreds
//! of similar series. The per-series streaming leaves need dozens of
//! observations to warm up; on such short series the estimates are noisy.
//! Bayesian hierarchical modelling routinely improves 10-20 % MASE on
//! these panels by pooling information — [`HierarchicalLaplace`] is the
//! streaming-friendly equivalent.
//!
//! ## Complexity
//!
//! - `fit_series`: same as per-series `LaplaceForecaster::fit` — O(n) per series.
//! - `finalize`: O(N · H) — one prediction call per series at max horizon.
//! - `predict`: O(H) after finalize — a two-term linear blend.
//!
//! `finalize` must be called after all series are fit and before the
//! first `predict`. It caches the panel forecast; further `fit_series`
//! calls invalidate the cache.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::laplace::LaplaceForecaster;
use crate::models::Forecaster;
use std::collections::HashMap;

/// Cross-series Empirical-Bayes wrapper — see [module docs](self).
///
/// **Shrinkage target: the delta from the last observed value.** The panel
/// prior is a scale-normalized mean of `fc_i(h) − y_i(last)` across all
/// fitted series. Blending is done in *change* space, not level space —
/// keeps a rising series rising even when the panel is flat, only pulling
/// the *slope* toward the panel mean.
pub struct HierarchicalLaplace {
    factory: Box<dyn Fn() -> LaplaceForecaster + Send + Sync>,
    series: HashMap<String, LaplaceForecaster>,
    /// Number of training observations per series — sets shrinkage weight.
    n_obs: HashMap<String, usize>,
    /// Per-series scale (median absolute value) used to normalize deltas.
    scales: HashMap<String, f64>,
    /// Per-series last-observed value — the anchor for delta shrinkage.
    last_values: HashMap<String, f64>,
    /// Prior strength `κ` in `λ = 1/(1 + n/κ)`. Higher = more shrinkage.
    prior_strength: f64,
    /// Series with more than this many training observations skip
    /// shrinkage entirely (long series don't need cross-series help and
    /// pulling them in the wrong direction hurts).
    max_n_for_shrinkage: usize,
    /// Cached scale-normalized panel delta forecast up to `max_horizon`.
    panel_delta_cache: Option<Vec<f64>>,
    /// Horizon `panel_delta_cache` was computed for.
    cached_horizon: usize,
}

impl HierarchicalLaplace {
    /// Build with a factory that produces fresh per-series forecasters,
    /// plus a shrinkage `prior_strength` (higher → more panel influence).
    /// Default rule of thumb: pick `prior_strength` ≈ median series length.
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
            prior_strength: prior_strength.max(1.0),
            max_n_for_shrinkage: 100,
            panel_delta_cache: None,
            cached_horizon: 0,
        }
    }

    /// Override the "no-shrinkage" cutoff. Series with more than this many
    /// training observations skip the panel prior entirely. Default 100.
    pub fn with_max_n_for_shrinkage(mut self, n: usize) -> Self {
        self.max_n_for_shrinkage = n;
        self
    }

    /// Fit one series identified by `id`. Invalidates the panel cache.
    pub fn fit_series(&mut self, id: impl Into<String>, series: &TimeSeries) -> Result<()> {
        let id = id.into();
        let mut m = (self.factory)();
        m.fit(series)?;
        let vals = series.primary_values();
        let scale = median_abs(vals);
        let last = vals.last().copied().unwrap_or(0.0);
        self.n_obs.insert(id.clone(), vals.len());
        self.scales.insert(id.clone(), scale);
        self.last_values.insert(id.clone(), last);
        self.series.insert(id, m);
        self.panel_delta_cache = None;
        Ok(())
    }

    /// Bulk-fit `(id, series)` pairs; returns the count of successful fits.
    pub fn fit_panel<'a, I>(&mut self, panel: I) -> usize
    where
        I: IntoIterator<Item = (String, &'a TimeSeries)>,
    {
        let mut ok = 0;
        for (id, ts) in panel {
            if self.fit_series(id, ts).is_ok() {
                ok += 1;
            }
        }
        ok
    }

    /// Compute and cache the **normalized panel delta forecast** up to
    /// `max_horizon`. For each series, the delta from its last observed
    /// value is normalized by scale and averaged across series. This
    /// captures the panel's typical growth/decline *shape* independently
    /// of absolute levels.
    pub fn finalize(&mut self, max_horizon: usize) -> Result<()> {
        if self.series.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "HierarchicalLaplace::finalize: no series fit".into(),
            ));
        }
        let mut delta_sum = vec![0.0f64; max_horizon];
        let mut count = 0usize;
        for (id, m) in &self.series {
            let s = *self.scales.get(id).unwrap_or(&1.0);
            let last = *self.last_values.get(id).unwrap_or(&0.0);
            if s.abs() < 1e-9 {
                continue;
            }
            let fc = m.predict(max_horizon)?;
            for (i, v) in fc.primary().iter().enumerate() {
                if v.is_finite() {
                    delta_sum[i] += (v - last) / s;
                }
            }
            count += 1;
        }
        if count == 0 {
            return Err(ForecastError::InvalidParameter(
                "HierarchicalLaplace::finalize: no series had usable scale".into(),
            ));
        }
        let panel_delta: Vec<f64> = delta_sum.iter().map(|s| s / count as f64).collect();
        self.panel_delta_cache = Some(panel_delta);
        self.cached_horizon = max_horizon;
        Ok(())
    }

    /// Level-space point forecast, blending the series' own predicted
    /// delta from its last observation with the panel-mean delta.
    ///
    /// `blended_fc(h) = last + (1-λ) · (fc_series(h) - last) + λ · panel_delta(h) · scale`
    ///
    /// where `λ = 1/(1 + n/κ)` when `n < max_n_for_shrinkage`, else `λ = 0`
    /// (long series skip shrinkage entirely).
    pub fn predict_series(&self, id: &str, horizon: usize) -> Result<Forecast> {
        let m = self
            .series
            .get(id)
            .ok_or_else(|| ForecastError::InvalidParameter(format!("no fit for series `{id}`")))?;
        let panel_delta = self.panel_delta_cache.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "HierarchicalLaplace::predict_series: call finalize() first".into(),
            )
        })?;
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
        // Long series skip shrinkage. Only short series get pulled toward
        // the panel prior.
        let lambda = if n >= self.max_n_for_shrinkage {
            0.0
        } else {
            1.0 / (1.0 + n as f64 / self.prior_strength)
        };
        let blended: Vec<f64> = per
            .primary()
            .iter()
            .zip(panel_delta.iter().take(horizon))
            .map(|(p, pd)| {
                let per_delta = p - last;
                let panel_delta_scaled = pd * scale;
                last + (1.0 - lambda) * per_delta + lambda * panel_delta_scaled
            })
            .collect();
        Ok(Forecast::from_values(blended))
    }

    /// Number of fitted series in the panel.
    pub fn n_fitted(&self) -> usize {
        self.series.len()
    }

    /// The shrinkage weight `λ` that would be applied to series `id`.
    /// Useful for observability / debugging.
    pub fn shrinkage_weight(&self, id: &str) -> Option<f64> {
        let n = *self.n_obs.get(id)?;
        if n >= self.max_n_for_shrinkage {
            Some(0.0)
        } else {
            Some(1.0 / (1.0 + n as f64 / self.prior_strength))
        }
    }
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
    fn fit_then_finalize_then_predict() {
        let mut h = HierarchicalLaplace::new(20.0, || LaplaceForecaster::new().auto());
        h.fit_series("A", &ts((0..100).map(|i| 10.0 + i as f64 * 0.1).collect()))
            .unwrap();
        h.fit_series("B", &ts((0..80).map(|i| 5.0 + i as f64 * 0.05).collect()))
            .unwrap();
        h.finalize(10).unwrap();
        let fc = h.predict_series("A", 5).unwrap();
        assert_eq!(fc.primary().len(), 5);
        for v in fc.primary() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn predict_without_finalize_errors() {
        let mut h = HierarchicalLaplace::new(20.0, || LaplaceForecaster::new().auto());
        h.fit_series("A", &ts(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
            .unwrap();
        assert!(h.predict_series("A", 3).is_err());
    }

    #[test]
    fn short_history_gets_more_shrinkage() {
        let mut h = HierarchicalLaplace::new(50.0, || LaplaceForecaster::new().auto());
        h.fit_series("short", &ts(vec![10.0; 20])).unwrap();
        h.fit_series("long", &ts(vec![10.0; 500])).unwrap();
        let ws = h.shrinkage_weight("short").unwrap();
        let wl = h.shrinkage_weight("long").unwrap();
        assert!(
            ws > wl * 5.0,
            "short-history should get >5x shrinkage than long-history: {} vs {}",
            ws,
            wl
        );
    }
}
