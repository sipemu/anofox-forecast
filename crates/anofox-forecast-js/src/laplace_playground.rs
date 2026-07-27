//! WASM bindings for the streaming Laplace playground.
//!
//! Exposes a `LaplacePlayground` struct that mirrors the streaming
//! semantics of skaters' playground: build once, `.observe(y)` per new
//! observation, `.forecast(h)` returns per-horizon quantile bands.
//!
//! The recipe is picked up-front by `laplace::recommended_for` from
//! the initial warm-up series (or the user-supplied `period`), matching
//! the "when to use Laplace" logic in the router.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::{
    recipe_for, recommended_for, DistributionalForecaster, RecipeKind,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use wasm_bindgen::prelude::*;

/// Streaming Laplace forecaster with per-horizon quantile output.
///
/// Lifecycle:
///  1. Construct with the initial warm-up series and a horizon.
///  2. Call `observe(y)` for each new observation.
///  3. Call `forecast(h)` to get an h-step quantile matrix.
#[wasm_bindgen]
pub struct LaplacePlayground {
    inner: Box<dyn DistributionalForecaster>,
    recipe: RecipeKind,
    period: Option<usize>,
    horizon: usize,
    values: Vec<f64>,
}

#[wasm_bindgen]
impl LaplacePlayground {
    /// Construct from an initial warm-up series (needed for the router
    /// to inspect data-shape) and an optional seasonal period.
    /// `period_or_zero` = 0 means "no known period".
    ///
    /// Returns an error string if the series is too short (< 3 obs)
    /// or the initial fit fails.
    #[wasm_bindgen(constructor)]
    pub fn new(
        warmup_values: Vec<f64>,
        period_or_zero: usize,
        horizon: usize,
    ) -> Result<LaplacePlayground, String> {
        if warmup_values.len() < 3 {
            return Err(format!(
                "LaplacePlayground requires ≥ 3 warm-up observations, got {}",
                warmup_values.len()
            ));
        }
        let period = if period_or_zero >= 2 {
            Some(period_or_zero)
        } else {
            None
        };
        let series = build_series(&warmup_values)?;
        let recipe = recipe_for(&series, period);
        let mut inner = recommended_for(&series, horizon.max(1), period);
        <dyn DistributionalForecaster as Forecaster>::fit(&mut *inner, &series)
            .map_err(|e| format!("initial fit failed: {}", e))?;
        Ok(LaplacePlayground {
            inner,
            recipe,
            period,
            horizon: horizon.max(1),
            values: warmup_values,
        })
    }

    /// Which router recipe was picked (short label suitable for a UI
    /// tag). One of: "short_history" / "retail_count_aid" /
    /// "heavy_tailed_crps" / "continuous_multiscale_3sh" /
    /// "continuous_plain_skaters".
    pub fn recipe(&self) -> String {
        self.recipe.name().to_string()
    }

    /// Number of observations seen so far (includes warm-up).
    pub fn n_observed(&self) -> usize {
        self.values.len()
    }

    /// Ingest one new observation. Re-fits the underlying model on the
    /// full history so far (streaming leaves absorb the new sample
    /// natively; the batch-fit call keeps the API uniform across all
    /// recipe branches — some recipes wrap a state-space model that
    /// needs a re-fit).
    pub fn observe(&mut self, y: f64) -> Result<(), String> {
        if !y.is_finite() {
            return Err("non-finite observation".into());
        }
        self.values.push(y);
        let series = build_series(&self.values)?;
        // Re-decide the recipe on the *current* window so shape shifts
        // (e.g. RW → structural break → trending) get routed correctly.
        let new_recipe = recipe_for(&series, self.period);
        if new_recipe != self.recipe {
            self.inner = recommended_for(&series, self.horizon, self.period);
            self.recipe = new_recipe;
        }
        <dyn DistributionalForecaster as Forecaster>::fit(&mut *self.inner, &series)
            .map_err(|e| format!("re-fit failed: {}", e))?;
        Ok(())
    }

    /// Return a per-horizon quantile matrix as a flat `Float64Array`
    /// of length `horizon * q_levels.len()`, laid out as
    /// `[q0_h1, q1_h1, ..., q0_h2, q1_h2, ...]`. Also included: the
    /// mean is at column index `q_levels.len()` for each horizon
    /// (so total width = `q_levels.len() + 1`).
    ///
    /// `q_levels` is typically `[0.1, 0.25, 0.5, 0.75, 0.9]`.
    pub fn forecast(&self, horizon: usize, q_levels: Vec<f64>) -> Result<Vec<f64>, String> {
        let h = horizon.max(1);
        let mixtures = self
            .inner
            .forecast_dist(h)
            .map_err(|e| format!("forecast failed: {}", e))?;
        let n_q = q_levels.len();
        let width = n_q + 1;
        let mut out = Vec::with_capacity(h * width);
        for g in &mixtures {
            for &q in &q_levels {
                let q = q.clamp(1e-6, 1.0 - 1e-6);
                out.push(g.quantile(q));
            }
            out.push(g.mean());
        }
        Ok(out)
    }
}

fn build_series(values: &[f64]) -> Result<TimeSeries, String> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::seconds(3600 * i as i64))
        .collect();
    TimeSeries::univariate(stamps, values.to_vec()).map_err(|e| format!("bad series: {}", e))
}
