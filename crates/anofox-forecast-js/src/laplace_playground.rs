//! WASM bindings for the streaming Laplace playground.
//!
//! Exposes a `LaplacePlayground` struct that mirrors the streaming
//! semantics of skaters' playground: build once, `.observe(y)` per new
//! observation, `.forecast(h)` returns per-horizon quantile bands.
//!
//! The recipe is picked up-front by `laplace::recommended_for` from
//! the initial warm-up series (or the user-supplied `period`), matching
//! the "when to use Laplace" logic in the router.

use anofox_forecast::anomaly::{MahalanobisConfig, MahalanobisDetector};
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::{
    recipe_for, recommended_for, DistributionalForecaster, LaplaceForecaster, MultiScaleLaplace,
    RecipeKind,
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
    label: String,
    variant: String,       // "" = router-picked; else fixed variant name
    periods: Vec<usize>,   // canonical: [] = none, [P] = single, [P1, P2, ...] = multi
    horizon: usize,
    values: Vec<f64>,
}

#[wasm_bindgen]
impl LaplacePlayground {
    /// Construct from an initial warm-up series.
    ///
    /// * `warmup_values`   — at least 3 observations of history.
    /// * `periods_csv`     — comma-separated seasonal periods (e.g. `""` for
    ///   none, `"24"` for daily, `"24,168"` for daily + weekly).
    /// * `horizon`         — max forecast horizon.
    /// * `variant`         — recipe override; `""` uses `recommended_for`
    ///   (data-shape router). Other supported strings:
    ///   `"multiscale_3sh"` (MS + tuned knobs + 3-α SH pool — the fev-27
    ///   winner), `"skaters"`, `"auto"`, `"auto_aid"`. Unknown → router.
    #[wasm_bindgen(constructor)]
    pub fn new(
        warmup_values: Vec<f64>,
        periods_csv: String,
        horizon: usize,
        variant: String,
    ) -> Result<LaplacePlayground, String> {
        if warmup_values.len() < 3 {
            return Err(format!(
                "LaplacePlayground requires ≥ 3 warm-up observations, got {}",
                warmup_values.len()
            ));
        }
        let periods = parse_periods(&periods_csv);
        let series = build_series(&warmup_values)?;
        let horizon = horizon.max(1);
        let (mut inner, label) = build_variant(&variant, &periods, horizon, &series);
        <dyn DistributionalForecaster as Forecaster>::fit(&mut *inner, &series)
            .map_err(|e| format!("initial fit failed: {}", e))?;
        Ok(LaplacePlayground {
            inner,
            label,
            variant,
            periods,
            horizon,
            values: warmup_values,
        })
    }

    /// Recipe / variant label for UI display. For the router variant
    /// this reflects the current `RecipeKind`; for explicit variants
    /// it's the variant name.
    pub fn recipe(&self) -> String {
        self.label.clone()
    }

    /// Number of observations seen so far (includes warm-up).
    pub fn n_observed(&self) -> usize {
        self.values.len()
    }

    /// Ingest one new observation. Re-fits the underlying model on the
    /// full history so far.
    pub fn observe(&mut self, y: f64) -> Result<(), String> {
        if !y.is_finite() {
            return Err("non-finite observation".into());
        }
        self.values.push(y);
        let series = build_series(&self.values)?;
        // Rebuild the inner forecaster each observe — variants can flip
        // (router path) and multi-scale needs a full re-fit anyway.
        let (mut inner, label) =
            build_variant(&self.variant, &self.periods, self.horizon, &series);
        <dyn DistributionalForecaster as Forecaster>::fit(&mut *inner, &series)
            .map_err(|e| format!("re-fit failed: {}", e))?;
        self.inner = inner;
        self.label = label;
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

    /// Surprise score: `-log p(y | current 1-step predictive mixture)`.
    /// Higher = more anomalous. Under a well-calibrated Gaussian
    /// predictive this is roughly `0.5 · z² + const`, so a threshold
    /// near 5 corresponds to ~3σ, near 8.5 to ~4σ.
    ///
    /// Returns `NaN` if the mixture is empty or the predictive density
    /// is exactly zero at `y` (extremely deep tail).
    pub fn surprise(&self, y: f64) -> f64 {
        if !y.is_finite() {
            return f64::NAN;
        }
        let mixtures = match self.inner.forecast_dist(1) {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };
        if mixtures.is_empty() {
            return f64::NAN;
        }
        let lp = mixtures[0].logpdf(y);
        if lp.is_finite() {
            -lp
        } else {
            f64::INFINITY
        }
    }

    /// Two-sided tail probability of `y` under the current 1-step
    /// predictive mixture: `2 · min(F(y), 1 - F(y))`. Returns a value
    /// in `[0, 1]`; smaller = further into the tail = more anomalous.
    /// Complementary to [`Self::surprise`] — this is the "PIT-style"
    /// surprise index that doesn't depend on the log-scale.
    pub fn tail_probability(&self, y: f64) -> f64 {
        if !y.is_finite() {
            return f64::NAN;
        }
        let mixtures = match self.inner.forecast_dist(1) {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };
        if mixtures.is_empty() {
            return f64::NAN;
        }
        let f = mixtures[0].cdf(y).clamp(0.0, 1.0);
        2.0 * f.min(1.0 - f)
    }
}

fn build_series(values: &[f64]) -> Result<TimeSeries, String> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::seconds(3600 * i as i64))
        .collect();
    TimeSeries::univariate(stamps, values.to_vec()).map_err(|e| format!("bad series: {}", e))
}

fn parse_periods(csv: &str) -> Vec<usize> {
    let mut out: Vec<usize> = csv
        .split(',')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .filter(|p| *p >= 2)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Build the inner forecaster + display label for a given variant.
///
/// Supported variants:
///   ""              — router (recommended_for). Label = RecipeKind.name().
///   "multiscale_3sh" — MS + scH + sw=10 + η=0.20 + 3-α SH pool at each
///                       supplied period. This is the fev-27 winner.
///   "skaters"       — .skaters() + scH(H) + sw=10 + η=0.20, plus
///                       .auto_with_seasonal_period(P) for each period
///                       and a SH pool at each.
///   "auto"          — .auto() (+ auto_with_seasonal_period for each P).
///   "auto_aid"      — .auto_aid() (+ auto_with_seasonal_period for each P).
///   anything else   — router (safe fallback).
fn build_variant(
    variant: &str,
    periods: &[usize],
    horizon: usize,
    warmup: &TimeSeries,
) -> (Box<dyn DistributionalForecaster>, String) {
    let single_period = periods.first().copied();
    let multi_period = periods.len() > 1;

    // Neither the router nor MultiScaleLaplace natively support multiple
    // seasonal periods (both take a single period). When the caller
    // supplies multi-period (e.g. 24,168 for hourly-with-weekly), we
    // downgrade to the `skaters` variant, which iterates over all
    // supplied periods and adds `.auto_with_seasonal_period(p)` +
    // 3α-SH per period. The label surfaces the substitution so the UI
    // is honest about what got fit.
    match variant {
        "multiscale_3sh" if multi_period => {
            let (inner, _) = build_variant("skaters", periods, horizon, warmup);
            (
                inner,
                format!(
                    "multi-period requested → skaters + 3α-SH per period=[{}] (MS is single-period only)",
                    periods_label(periods)
                ),
            )
        }
        "" if multi_period => {
            let (inner, _) = build_variant("skaters", periods, horizon, warmup);
            (
                inner,
                format!(
                    "router → multi-period → skaters + 3α-SH per period=[{}]",
                    periods_label(periods)
                ),
            )
        }
        "multiscale_3sh" => {
            let mut m = MultiScaleLaplace::skaters(horizon)
                .with_scoring_horizon()
                .with_scoring_window(10)
                .with_learning_rate(0.20)
                .with_seasonal_holt(0.3, 0.1)
                .with_seasonal_holt(0.5, 0.2)
                .with_seasonal_holt(0.7, 0.3);
            if let Some(p) = single_period {
                m = m.with_period(p);
            }
            let label = format!(
                "MS + scH + sw10 + η=0.20 + 3α-SH  (period={})",
                periods_label(periods)
            );
            (Box::new(m), label)
        }
        "skaters" => {
            let mut f = LaplaceForecaster::new()
                .skaters()
                .with_scoring_horizon(horizon)
                .with_scoring_window(10)
                .learning_rate(0.20);
            for &p in periods {
                f = f.auto_with_seasonal_period(p);
                f = f
                    .with_seasonal_holt(p, 0.3, 0.1)
                    .with_seasonal_holt(p, 0.5, 0.2)
                    .with_seasonal_holt(p, 0.7, 0.3);
            }
            let label = format!(
                ".skaters() + scH + sw10 + η=0.20 + 3α-SH  (periods=[{}])",
                periods_label(periods)
            );
            (Box::new(f), label)
        }
        "auto" => {
            let mut f = LaplaceForecaster::new().auto();
            for &p in periods {
                f = f.auto_with_seasonal_period(p);
            }
            (
                Box::new(f),
                format!(".auto()  (periods=[{}])", periods_label(periods)),
            )
        }
        "auto_aid" => {
            let mut f = LaplaceForecaster::new().auto_aid();
            for &p in periods {
                f = f.auto_with_seasonal_period(p);
            }
            (
                Box::new(f),
                format!(".auto_aid()  (periods=[{}])", periods_label(periods)),
            )
        }
        _ => {
            // Router path, single (or no) period. Uses recommended_for
            // which takes an Option<usize>.
            let recipe = recipe_for(warmup, single_period);
            (
                recommended_for(warmup, horizon, single_period),
                format!("router → {}", recipe.name()),
            )
        }
    }
}

fn periods_label(periods: &[usize]) -> String {
    if periods.is_empty() {
        "—".into()
    } else {
        periods
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Wrapper around the crate's [`MahalanobisDetector`] — the parade-z /
/// Mahalanobis anomaly detector ported from timemachines' `mahalanobis`
/// head (also documented in skaters issue #88). Emits per-tick
/// `d²` (Mahalanobis distance over the k-horizon z-vector), a
/// calibrated `p_value` under the empirical null, and a `run` counter
/// of consecutive above-guard ticks (1-2 = point outlier, growing run
/// = changepoint).
///
/// Prefer this over `LaplacePlayground::surprise` when the score
/// needs to reflect JOINT tail behaviour across multiple horizons, or
/// when the null distribution shouldn't be assumed Gaussian.
#[wasm_bindgen]
pub struct MahalanobisPlayground {
    inner: MahalanobisDetector,
    k: usize,
    n_observed: usize,
    label: String,
}

#[wasm_bindgen]
impl MahalanobisPlayground {
    /// Fit a Laplace base forecaster on the warm-up series, wrap it in
    /// a `MahalanobisDetector` with the requested horizon `k`.
    ///
    /// * `warmup_values`   — at least `k + 3` observations.
    /// * `k`               — parade horizon (clamped to `[1, 32]`).
    /// * `periods_csv`     — comma-separated seasonal periods. If given,
    ///   each period is passed to `.auto_with_seasonal_period(P)` and
    ///   a 3-α SH pool is added at that period. Critical for NAB-style
    ///   real data with strong daily/weekly cycles — without it, the
    ///   base forecaster misses seasonality and every peak-hour tick
    ///   reads as an anomaly.
    #[wasm_bindgen(constructor)]
    pub fn new(
        warmup_values: Vec<f64>,
        k: usize,
        periods_csv: String,
    ) -> Result<MahalanobisPlayground, String> {
        let k = k.clamp(1, 32);
        if warmup_values.len() < k + 3 {
            return Err(format!(
                "MahalanobisPlayground requires ≥ k+3 = {} warm-up observations, got {}",
                k + 3,
                warmup_values.len()
            ));
        }
        let periods = parse_periods(&periods_csv);
        let series = build_series(&warmup_values)?;
        let cfg = MahalanobisConfig::new(k);

        // Build a base tuned for the observed shape. `.skaters()` on its
        // own is a plain pool with no seasonal awareness; feeding it the
        // period(s) unlocks the seasonal-EMA / seasonal-Holt leaves,
        // which massively improves the residual quality the parade sees.
        let mut base = LaplaceForecaster::new()
            .skaters()
            .with_scoring_horizon(k)
            .with_scoring_window(10)
            .learning_rate(0.20);
        for &p in &periods {
            base = base
                .auto_with_seasonal_period(p)
                .with_seasonal_holt(p, 0.3, 0.1)
                .with_seasonal_holt(p, 0.5, 0.2)
                .with_seasonal_holt(p, 0.7, 0.3);
        }
        let inner = MahalanobisDetector::fit_and_wrap(base, &series, cfg)
            .map_err(|e| format!("fit_and_wrap failed: {}", e))?;
        let label = if periods.is_empty() {
            format!("Mahalanobis(k={}, no seasonality)", k)
        } else {
            format!("Mahalanobis(k={}, period={})", k, periods_label(&periods))
        };
        Ok(MahalanobisPlayground {
            inner,
            k,
            n_observed: warmup_values.len(),
            label,
        })
    }

    /// Human-readable label for the UI recipe tag.
    pub fn label(&self) -> String {
        self.label.clone()
    }

    /// Ingest one new observation and update the detector's internal
    /// z-vector history + scatter estimate. Cheap; the parade updates
    /// incrementally.
    pub fn observe(&mut self, y: f64) -> Result<(), String> {
        if !y.is_finite() {
            return Err("non-finite observation".into());
        }
        self.inner
            .observe(y)
            .map_err(|e| format!("observe failed: {}", e))?;
        self.n_observed += 1;
        Ok(())
    }

    /// Score of the last observation as a flat `[d2, p_value, run, ok]`
    /// where:
    ///  - `d2`      = Mahalanobis distance over the current z-vector,
    ///                or `NaN` while the parade is still warming up.
    ///  - `p_value` = calibrated tail p-value under the empirical null,
    ///                in (0, 1); `NaN` while warming up.
    ///  - `run`     = consecutive ticks above the guard threshold.
    ///  - `ok`      = 1.0 if the parade is warm (d2 & p_value valid),
    ///                else 0.0.
    ///
    /// Returned as an array so the JS caller can access it as one
    /// contiguous Float64Array.
    pub fn last_score(&self) -> Vec<f64> {
        let s = self.inner.state();
        let d2 = s.d2.unwrap_or(f64::NAN);
        let p = s.p_value.unwrap_or(f64::NAN);
        let run = s.run as f64;
        let ok = if s.d2.is_some() && s.p_value.is_some() {
            1.0
        } else {
            0.0
        };
        vec![d2, p, run, ok]
    }

    /// Horizon `k` this detector was configured with.
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n_observed(&self) -> usize {
        self.n_observed
    }
}
