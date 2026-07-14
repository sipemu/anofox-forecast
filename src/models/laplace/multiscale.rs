//! `MultiScaleLaplace` — port of skaters' `multiscale` wrapper.
//!
//! For horizon `k`, runs decimated copies of the underlying
//! `LaplaceForecaster` at strides `{1, ⌈√k⌉, k}`. Each scale `s`
//! sees every s-th observation; its "one-step" prediction corresponds
//! to `s` real steps of the raw series.
//!
//! At forecast time h ∈ 1..=horizon, every eligible scale (`s ≤ h`)
//! contributes its `⌈h / s⌉`-step prediction; the per-horizon mixture
//! is a softmax blend across scales, weighted by each scale's mean
//! training log-likelihood. Ports the full behaviour of skaters'
//! `multiscale.py::_skater`.
//!
//! `Forecaster::predict` returns just the mixture means (the same
//! largest-eligible-stride pick for backwards compat with the earlier
//! port). `DistributionalForecaster::forecast_dist` does the proper
//! per-horizon softmax blend across scales.
//!
//! Post-#180 addition — fev-27 follow-up.

use super::dist::GaussianMixture;
use super::forecaster::LaplaceForecaster;
use super::DistributionalForecaster;
use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::traits::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// A stack of `LaplaceForecaster` instances at decimated strides.
pub struct MultiScaleLaplace {
    /// (stride, forecaster) pairs. Sorted by stride ascending.
    scales: Vec<(usize, LaplaceForecaster)>,
    /// Horizon this stack was configured for; determines the strides.
    max_horizon: usize,
    /// Optional period hint. When set (via [`Self::with_period`]) the
    /// strides include the exact period rather than just skaters' `⌈√k⌉`
    /// — a period-aligned decimated forecaster preserves the seasonal
    /// cycle exactly, whereas `⌈√7⌉ = 3` on m4_hourly period=24 misaligns.
    period_hint: Option<usize>,
    /// Per-scale training-set mean log-likelihood — softmax weights
    /// across scales at forecast_dist time. Recomputed each fit.
    scale_scores: Vec<f64>,
    /// If Some(w), pass `.with_scoring_window(w)` to each scale's sub.
    scoring_window: Option<usize>,
    /// If true, pass `.with_scoring_horizon(coarse_h)` to each scale's
    /// sub, where `coarse_h = ceil(max_horizon / s)` for scale s.
    enable_scoring_horizon: bool,
}

/// Scale set for the multi-scale wrapper. Combines skaters' `{1, ⌈√k⌉, k}`
/// with an optional period-aligned stride so seasonal signals decimate
/// coherently.
///
/// Strides are trimmed to those giving `≥ min_samples` decimated
/// observations — the streaming leaves need warmup, so a stride that
/// leaves us with 10 observations is worse than falling back to a
/// smaller stride.
fn default_scales(
    horizon: usize,
    n_train: usize,
    min_samples: usize,
    period: Option<usize>,
) -> Vec<usize> {
    let mut out = vec![1usize];
    // When a period is known, skip the ⌈√k⌉ stride entirely: on
    // seasonal panels (m4_hourly p=24 H=48 → sqrt=7) a stride coprime
    // with the period aliases the seasonal signal and destroys the
    // decimated sub-forecaster. Use only period-aligned strides.
    // Measured on fev-27: including sqrt(H) with period set regresses
    // m4_hourly +64 %, tourism_monthly +50 %. Excluding it recovers.
    let candidates: Vec<usize> = if let Some(p) = period {
        vec![p, horizon]
    } else {
        let sqrt_k = (horizon as f64).sqrt().ceil() as usize;
        vec![sqrt_k, horizon]
    };
    for s in candidates {
        if s > 1 && s <= horizon && n_train / s >= min_samples && !out.contains(&s) {
            out.push(s);
        }
    }
    out.sort();
    out.dedup();
    out
}

impl MultiScaleLaplace {
    /// Build a stack around a fresh `.skaters()` base, sized for the
    /// given max horizon.
    pub fn skaters(max_horizon: usize) -> Self {
        Self {
            scales: Vec::new(),
            max_horizon,
            period_hint: None,
            scale_scores: Vec::new(),
            scoring_window: None,
            enable_scoring_horizon: false,
        }
    }

    /// Pass `.with_scoring_window(w)` to each scale's sub-forecaster.
    pub fn with_scoring_window(mut self, w: usize) -> Self {
        self.scoring_window = Some(w);
        self
    }

    /// Pass `.with_scoring_horizon(coarse_h)` to each scale's sub,
    /// where `coarse_h` is that scale's own coarse target horizon.
    pub fn with_scoring_horizon(mut self) -> Self {
        self.enable_scoring_horizon = true;
        self
    }

    /// Add a period-aligned decimated forecaster. The period stride
    /// preserves seasonal cycles exactly (unlike skaters' `⌈√k⌉` which
    /// misaligns for non-square-integer periods).
    pub fn with_period(mut self, period: usize) -> Self {
        self.period_hint = Some(period);
        self
    }

    /// Which strides this stack is currently configured with (after
    /// `fit()` has trimmed those below the min-samples threshold).
    pub fn strides(&self) -> Vec<usize> {
        self.scales.iter().map(|(s, _)| *s).collect()
    }

    /// Decimate a value slice by stride, keeping every s-th value.
    fn decimate(values: &[f64], stride: usize) -> Vec<f64> {
        if stride <= 1 {
            values.to_vec()
        } else {
            values.iter().step_by(stride).copied().collect()
        }
    }

    /// Build a `TimeSeries` at the given stride from the source
    /// timestamps + values (stride-1 case = pass-through).
    fn decimated_ts(source: &TimeSeries, stride: usize) -> Result<TimeSeries> {
        let values = Self::decimate(source.primary_values(), stride);
        // Preserve stride-scaled temporal spacing so downstream code
        // that inspects `.timestamps()` sees a consistent gap.
        let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
        let stamps: Vec<_> = (0..values.len())
            .map(|i| base + Duration::hours((i * stride.max(1)) as i64))
            .collect();
        TimeSeries::univariate(stamps, values)
    }
}

impl Forecaster for MultiScaleLaplace {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let n = series.primary_values().len();
        // Streaming leaves need enough decimated obs to converge. Empirical
        // fev-27 tuning:
        //   threshold=100  → degenerates to scale-1 on all but m4_daily/m5
        //                    (safe but multiscale contributes almost nothing)
        //   threshold=30   → m4_hourly's stride 24 (29 obs) activates and
        //                    regresses -55 % because the sub can't fit ~30
        //                    leaves on 29 obs
        //   threshold=50   → drops m4_hourly's borderline decimation, keeps
        //                    m4_daily/m5/tourism_yearly benefits
        let strides = default_scales(self.max_horizon, n, 50, self.period_hint);
        self.scales.clear();
        self.scale_scores.clear();
        self.scales.reserve(strides.len());
        self.scale_scores.reserve(strides.len());
        for s in strides {
            let ts = Self::decimated_ts(series, s)?;
            let mut f = LaplaceForecaster::new().skaters();
            // Pass the period hint through — only meaningful on scale 1
            // (the fine-clock forecaster). At scale > 1 the decimated
            // clock's "period" is `period_hint / s`, which doesn't
            // correspond to any of our leaf periods when `period % s`
            // is nonzero.
            if s == 1 {
                if let Some(p) = self.period_hint {
                    if p >= 2 {
                        f = f.auto_with_seasonal_period(p);
                    }
                }
            }
            // Scoring knobs (v0.15.3) — pass through per scale's own
            // coarse horizon. At scale s the target horizon in coarse
            // steps is `ceil(max_horizon / s)`.
            if self.enable_scoring_horizon {
                let coarse_h = self.max_horizon.div_ceil(s).max(1);
                f = f.with_scoring_horizon(coarse_h);
            }
            if let Some(w) = self.scoring_window {
                f = f.with_scoring_window(w);
            }
            f.fit(&ts)?;
            // Mean training log-likelihood at this scale (average
            // 1-step LL over its residuals). Higher = tighter fit; used
            // as the softmax weight for this scale at forecast_dist.
            let residuals = f.residuals().unwrap_or(&[]);
            let sigma = {
                let n_r = residuals.len().max(1) as f64;
                let mean: f64 = residuals.iter().sum::<f64>() / n_r;
                let var: f64 = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_r;
                var.sqrt().max(1e-9)
            };
            // Mean N(0, σ) log-density: -0.5*ln(2πσ²) - 0.5*(residual/σ)²
            //  (we drop the shared constant; only relative scores matter)
            let ll = if residuals.is_empty() {
                f64::NEG_INFINITY
            } else {
                let n_r = residuals.len() as f64;
                let mut acc = 0.0;
                let two_pi_var = 2.0 * std::f64::consts::PI * sigma * sigma;
                let log_c = -0.5 * two_pi_var.ln();
                for &r in residuals {
                    acc += log_c - 0.5 * (r / sigma).powi(2);
                }
                acc / n_r
            };
            self.scales.push((s, f));
            self.scale_scores.push(ll);
        }
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.scales.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("MultiScaleLaplace".into()),
            });
        }
        let mut means = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            // Largest eligible stride: `s` such that `s ≤ h` and
            // the corresponding forecaster's `⌈h/s⌉`-step prediction
            // is well-defined.
            let (s, f) = self
                .scales
                .iter()
                .filter(|(s, _)| *s <= h)
                .max_by_key(|(s, _)| *s)
                .unwrap_or(&self.scales[0]);
            let steps = h.div_ceil(*s);
            let fc = f.predict(steps)?;
            let p = fc.primary();
            if p.len() >= steps {
                means.push(p[steps - 1]);
            } else {
                means.push(0.0);
            }
        }
        Ok(Forecast::from_values(means))
    }

    fn name(&self) -> &str {
        "MultiScaleLaplace"
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        // Fitted values come from the stride-1 forecaster if present.
        self.scales
            .iter()
            .find(|(s, _)| *s == 1)
            .and_then(|(_, f)| f.fitted_values())
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.scales
            .iter()
            .find(|(s, _)| *s == 1)
            .and_then(|(_, f)| f.residuals())
    }
}

impl DistributionalForecaster for MultiScaleLaplace {
    fn forecast_dist(&self, horizon: usize) -> Result<Vec<GaussianMixture>> {
        if self.scales.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("MultiScaleLaplace".into()),
            });
        }
        // Pre-compute each scale's max coarse horizon needed:
        //   scale s needs to serve h ∈ 1..=horizon → coarse step ⌈h/s⌉
        //   max coarse = ⌈horizon / s⌉
        let per_scale_dists: Vec<Vec<GaussianMixture>> = self
            .scales
            .iter()
            .map(|(s, f)| {
                let coarse = horizon.div_ceil(*s);
                f.forecast_dist(coarse).unwrap_or_default()
            })
            .collect();
        // Blend per fine horizon h ∈ 1..=horizon:
        //   for each eligible scale s (s ≤ h), take dists[⌈h/s⌉ - 1]
        //   weight = exp(score_s - max_score)
        //   mixture = weighted concat of components (GaussianMixture::new
        //             re-normalises)
        let max_score = self
            .scale_scores
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let mut out = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let mut comps: Vec<(f64, super::dist::Gaussian)> = Vec::new();
            for (i, (s, _)) in self.scales.iter().enumerate() {
                if *s > h {
                    continue;
                }
                let coarse_idx = h.div_ceil(*s).saturating_sub(1);
                if coarse_idx >= per_scale_dists[i].len() {
                    continue;
                }
                let scale_w = if max_score.is_finite() && self.scale_scores[i].is_finite() {
                    (self.scale_scores[i] - max_score).exp()
                } else {
                    1.0
                };
                let mixture = &per_scale_dists[i][coarse_idx];
                for (w, g) in &mixture.components {
                    comps.push((scale_w * w, *g));
                }
            }
            out.push(GaussianMixture::new(comps));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{Duration, TimeZone, Utc};

    fn periodic_ts(n: usize, period: usize) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                let phase = (i % period) as f64 / period as f64;
                100.0
                    + 30.0 * (2.0 * std::f64::consts::PI * phase).sin()
                    + ((i as f64 * 12.9898).sin() * 43758.5453).fract()
            })
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn strides_include_1_and_sqrt_k_and_k() {
        let strides = default_scales(48, 1000, 10, None);
        assert_eq!(strides, vec![1, 7, 48]);
    }

    #[test]
    fn strides_dropped_when_too_few_samples() {
        // 20 obs, stride 48 → 0 samples → dropped.
        let strides = default_scales(48, 20, 5, None);
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn strides_include_period_when_hint_given() {
        // With period set, the ⌈√k⌉ stride is dropped (would alias
        // the seasonal signal on m4_hourly-like panels).
        let strides = default_scales(48, 1000, 10, Some(24));
        assert_eq!(strides, vec![1, 24, 48]);
    }

    #[test]
    fn fit_predict_produces_horizon_values() {
        let ts = periodic_ts(700, 24);
        let mut m = MultiScaleLaplace::skaters(48);
        m.fit(&ts).unwrap();
        let fc = m.predict(48).unwrap();
        assert_eq!(fc.primary().len(), 48);
        // Values should be finite (not NaN / inf).
        for v in fc.primary() {
            assert!(v.is_finite(), "non-finite forecast value: {v}");
        }
    }

    #[test]
    fn predict_before_fit_errors() {
        let m = MultiScaleLaplace::skaters(10);
        assert!(m.predict(5).is_err());
    }
}
