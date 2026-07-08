//! `MultiScaleLaplace` — port of skaters' `multiscale` wrapper.
//!
//! For horizon `k`, runs decimated copies of the underlying
//! `LaplaceForecaster` at strides `{1, ⌈√k⌉, k}`. Each scale `s`
//! sees every s-th observation; its "one-step" prediction corresponds
//! to `s` real steps of the raw series.
//!
//! At forecast time h ∈ 1..=horizon, we pick the **largest eligible
//! stride** `s ≤ h` and take that forecaster's `⌈h / s⌉`-step
//! prediction. This is a simplification of skaters' likelihood-blend
//! across eligible scales, but captures the main win: long-horizon
//! seasonal forecasts get a decimated forecaster whose native step is
//! aligned with the target horizon, sidestepping the softmax's
//! one-step-scored fine-scale candidates that flat-line at h ≫ 1.
//!
//! Post-#180 addition — fev-27 follow-up.

use super::forecaster::LaplaceForecaster;
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
    let sqrt_k = (horizon as f64).sqrt().ceil() as usize;
    let candidates = [sqrt_k, period.unwrap_or(0), horizon];
    for &s in &candidates {
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
        }
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
        // Streaming leaves need enough decimated obs to converge.
        // Empirical fev-27 tuning: at 50 samples the decimated
        // stride hurts tourism_monthly (short M-competition series
        // → decimated forecaster produces garbage). At 100 it's a
        // safe win on m4_hourly (700 obs) and neutral elsewhere.
        let strides = default_scales(self.max_horizon, n, 100, self.period_hint);
        self.scales.clear();
        self.scales.reserve(strides.len());
        for s in strides {
            let ts = Self::decimated_ts(series, s)?;
            let mut f = LaplaceForecaster::new().skaters();
            f.fit(&ts)?;
            self.scales.push((s, f));
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
        let strides = default_scales(48, 1000, 10, Some(24));
        assert_eq!(strides, vec![1, 7, 24, 48]);
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
