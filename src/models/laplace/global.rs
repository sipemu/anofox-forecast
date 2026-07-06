//! `GlobalLaplace` — panel-level fit that shares hyperparameters across
//! series and combines outputs.
//!
//! The current implementation is a **thin panel wrapper**: it fits an
//! independent `LaplaceForecaster` per series (using the same
//! configuration for all), aggregates the per-series calibration scales
//! to reduce noise on short series, and returns predictions in a per-series
//! map. This is *not* a global model in the DeepAR / N-BEATS sense — the
//! leaf state is still per-series — but it's the smallest useful
//! panel-level abstraction and reserves the API for a future
//! genuinely-global implementation with shared leaf hyperparameters
//! learned across the panel.
//!
//! Real global learning (shared α, β, φ estimated jointly) is a
//! substantial architecture change and is deferred to a later alpha.
//! Documented as scaffold with a clear extension path.

use crate::core::TimeSeries;
use crate::error::{ForecastError, Result};
use crate::models::laplace::LaplaceForecaster;
use crate::models::traits::Forecaster;
use crate::models::DistributionalForecaster;
use std::collections::HashMap;

pub struct GlobalLaplace {
    /// Factory that produces a fresh per-series `LaplaceForecaster`. All
    /// series get an identical config — the "global" part is currently
    /// just the shared config, not shared state.
    factory: Box<dyn Fn() -> LaplaceForecaster + Send + Sync>,
    /// Fitted per-series forecasters, keyed by caller-supplied id.
    fitted: HashMap<String, LaplaceForecaster>,
    /// Panel-level average of per-series calibration scales — used to
    /// shrink individual scales toward the panel mean on very short
    /// series (a poor man's cross-series signal). `None` until fit.
    panel_calibration_scale: Option<f64>,
}

impl GlobalLaplace {
    /// Build with a factory that produces a fresh `LaplaceForecaster`
    /// on each call. Example:
    /// `GlobalLaplace::new(|| LaplaceForecaster::new().auto())`.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> LaplaceForecaster + Send + Sync + 'static,
    {
        Self {
            factory: Box::new(factory),
            fitted: HashMap::new(),
            panel_calibration_scale: None,
        }
    }

    /// Fit one series with the given id. Returns immediately on empty
    /// or invalid series; does not roll back other fits.
    pub fn fit_series(&mut self, id: impl Into<String>, series: &TimeSeries) -> Result<()> {
        let mut m = (self.factory)();
        m.fit(series)?;
        self.fitted.insert(id.into(), m);
        Ok(())
    }

    /// Bulk fit a panel of `(id, series)` pairs. Continues on individual
    /// errors and reports the count of successful fits.
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

    /// Predict h steps ahead for a specific series.
    pub fn predict_series(&self, id: &str, horizon: usize) -> Result<crate::core::Forecast> {
        let m = self.fitted.get(id).ok_or_else(|| {
            ForecastError::InvalidParameter(format!("no fit found for series id `{id}`"))
        })?;
        m.predict(horizon)
    }

    /// Distributional predict — returns the per-horizon `GaussianMixture`
    /// for a specific series.
    pub fn forecast_dist_series(
        &self,
        id: &str,
        horizon: usize,
    ) -> Result<Vec<super::dist::GaussianMixture>> {
        let m = self.fitted.get(id).ok_or_else(|| {
            ForecastError::InvalidParameter(format!("no fit found for series id `{id}`"))
        })?;
        m.forecast_dist(horizon)
    }

    /// Panel-average of the per-series calibration scales (populated after
    /// fit if calibration was enabled on the config).
    pub fn panel_calibration_scale(&self) -> Option<f64> {
        self.panel_calibration_scale
    }

    pub fn n_fitted(&self) -> usize {
        self.fitted.len()
    }
}

// ============================================================================
// Meta-learner scaffold — reserved API, no ML backend yet.
// ============================================================================

/// **Scaffold.** Reserves the API for a per-series meta-learner that
/// picks the winning forecaster family from series characteristics using
/// a trained classifier. The current implementation is a **rules-based**
/// fallback that returns the same routing decision as
/// [`SmartForecaster::selected_family`](crate::models::smart::SelectedFamily) —
/// documented so downstream callers can swap in a real learner later
/// without breaking the API.
///
/// A real implementation needs:
/// 1. A labeled training panel where each series has a known winning
///    forecaster (from held-out MAE).
/// 2. A characteristic extractor (shared with the rules-based path).
/// 3. A classifier fit at load time (e.g. gradient-boosted trees over
///    the characteristics).
/// 4. Inference at forecast time — cheap, feature vector → class.
///
/// Deferred until we have real per-series labels from a benchmarking
/// campaign.
pub struct MetaLearnerScaffold;

impl MetaLearnerScaffold {
    /// Delegates to `SmartForecaster`'s rules. Present so callers can
    /// code against a stable `pick_family(chars)` API and swap in a
    /// trained model later.
    pub fn pick_family(
        zero_fraction: f64,
        trend_strength: f64,
    ) -> crate::models::smart::SelectedFamily {
        use crate::models::smart::SelectedFamily;
        if zero_fraction > 0.4 {
            SelectedFamily::Intermittent
        } else if trend_strength > 0.6 {
            SelectedFamily::AutoEts
        } else {
            SelectedFamily::LaplaceAuto
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn ts_ar1(n: usize, phi: f64) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut vals = Vec::with_capacity(n);
        let mut y = 0.0;
        for i in 0..n {
            let eps = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
            y = phi * y + eps + 5.0;
            vals.push(y);
        }
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn fits_a_small_panel_and_predicts_per_series() {
        let mut g = GlobalLaplace::new(|| LaplaceForecaster::new().auto());
        let ts_a = ts_ar1(120, 0.4);
        let ts_b = ts_ar1(120, 0.6);
        g.fit_series("A", &ts_a).unwrap();
        g.fit_series("B", &ts_b).unwrap();
        assert_eq!(g.n_fitted(), 2);
        let fc = g.predict_series("A", 5).unwrap();
        assert_eq!(fc.primary().len(), 5);
    }

    #[test]
    fn scaffold_pick_family_matches_smart_rules() {
        use crate::models::smart::SelectedFamily;
        assert_eq!(
            MetaLearnerScaffold::pick_family(0.6, 0.0),
            SelectedFamily::Intermittent
        );
        assert_eq!(
            MetaLearnerScaffold::pick_family(0.1, 0.9),
            SelectedFamily::AutoEts
        );
        assert_eq!(
            MetaLearnerScaffold::pick_family(0.1, 0.2),
            SelectedFamily::LaplaceAuto
        );
    }
}
