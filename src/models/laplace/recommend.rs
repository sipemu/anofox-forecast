//! Data-shape → recipe router for the Laplace family.
//!
//! Encodes the decision logic distilled from the 2026-07-14 → 2026-07-21
//! sweep sequence. Given a `TimeSeries`, horizon, and optional seasonal
//! period, [`recommended_for`] returns a pre-configured Laplace-family
//! forecaster; [`recipe_for`] returns which branch was chosen (for
//! logging / introspection).
//!
//! # Decision table
//!
//! | # | Property | Threshold | Route |
//! |---|---|---|---|
//! | 1 | Short history | `N < 60` | `.auto()` (with warning — classical Theta/ETS often better outside this crate) |
//! | 2 | Count-like | integer-valued fraction `> 0.95` AND zero-fraction `> 0.30` | `.auto_aid().auto_with_seasonal_period(P)` (no MultiScale — measured +8.7 % regression on M5) |
//! | 3 | Heavy-tailed | excess kurtosis `> 5` | `.skaters().with_terminal_crps()` |
//! | 4 | Continuous seasonal + long | `period ≥ 2` AND `N ≥ 60` | `MultiScaleLaplace + scH + sw=10 + η=0.20 + 3α-SH pool` (the 2026-07-21 fev-27 winner) |
//! | 5 | Continuous fallback | else | `.skaters() + scH(P) + sw=10 + η=0.20` |
//!
//! The router is Laplace-scoped by design: for `N < 60` a caller may
//! prefer `crate::models::theta::AutoTheta` or `crate::models::exponential::AutoETS`,
//! both of which live outside the Laplace family. See
//! [`SmartForecaster`](crate::models::SmartForecaster) for a cross-family router.

use crate::core::TimeSeries;
use crate::models::laplace::multiscale::MultiScaleLaplace;
use crate::models::laplace::{DistributionalForecaster, LaplaceForecaster};

/// Which recipe the router picked, for logging / introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeKind {
    /// N < 60. Laplace family runs but classical Theta/ETS is usually
    /// better here; caller should consider switching.
    ShortHistory,
    /// Count-like: integer-valued with high zero-fraction (M5 SKUs,
    /// intermittent demand, retail).
    RetailCountAid,
    /// Excess kurtosis > 5 (financial returns, extreme-event panels).
    HeavyTailedCrps,
    /// Continuous seasonal + `N ≥ 60` + period supplied. The 2026-07-21
    /// fev-27 winner (rank ~6, geomean 1.4149 ex outliers).
    ContinuousMultiScale,
    /// Continuous but missing period or `N < 50/scale`. Plain skaters
    /// with the same tuned knobs, no MultiScale wrapping.
    ContinuousPlainSkaters,
}

impl RecipeKind {
    /// Short human-readable name — safe to log without extra formatting.
    pub fn name(self) -> &'static str {
        match self {
            RecipeKind::ShortHistory => "short_history",
            RecipeKind::RetailCountAid => "retail_count_aid",
            RecipeKind::HeavyTailedCrps => "heavy_tailed_crps",
            RecipeKind::ContinuousMultiScale => "continuous_multiscale_3sh",
            RecipeKind::ContinuousPlainSkaters => "continuous_plain_skaters",
        }
    }
}

/// Decide which recipe fits the series without building anything. Cheap;
/// pair with [`recommended_for`] when you also want a fit-ready forecaster.
pub fn recipe_for(series: &TimeSeries, period: Option<usize>) -> RecipeKind {
    let values = series.primary_values();
    let n = values.len();
    if n < 60 {
        return RecipeKind::ShortHistory;
    }
    if is_count_like(values) {
        return RecipeKind::RetailCountAid;
    }
    if is_heavy_tailed(values) {
        return RecipeKind::HeavyTailedCrps;
    }
    match period {
        Some(p) if p >= 2 && n >= 60 => RecipeKind::ContinuousMultiScale,
        _ => RecipeKind::ContinuousPlainSkaters,
    }
}

/// Build a pre-configured Laplace-family forecaster keyed to the series'
/// shape. Returns `Box<dyn DistributionalForecaster>` — call `.fit()` and
/// `.forecast_dist()` on the result.
///
/// `period` is the seasonal period the caller knows (or `None`).
/// Auto-detection of period is deliberately out of scope — see the
/// module note in [`super`] on why.
pub fn recommended_for(
    series: &TimeSeries,
    horizon: usize,
    period: Option<usize>,
) -> Box<dyn DistributionalForecaster> {
    match recipe_for(series, period) {
        RecipeKind::ShortHistory => {
            // Laplace-family fallback. Caller was warned via docs that
            // AutoTheta / AutoETS often beat this below N=60.
            let mut f = LaplaceForecaster::new().auto();
            if let Some(p) = period {
                if p >= 2 {
                    f = f.auto_with_seasonal_period(p);
                }
            }
            Box::new(f)
        }
        RecipeKind::RetailCountAid => {
            let mut f = LaplaceForecaster::new().auto_aid();
            if let Some(p) = period {
                if p >= 2 {
                    f = f.auto_with_seasonal_period(p);
                }
            }
            Box::new(f)
        }
        RecipeKind::HeavyTailedCrps => {
            let mut f = LaplaceForecaster::new().skaters().with_terminal_crps();
            if let Some(p) = period {
                if p >= 2 {
                    f = f.auto_with_seasonal_period(p);
                }
            }
            Box::new(f)
        }
        RecipeKind::ContinuousMultiScale => {
            let mut m = MultiScaleLaplace::skaters(horizon)
                .with_scoring_horizon()
                .with_scoring_window(10)
                .with_learning_rate(0.20)
                .with_seasonal_holt(0.3, 0.1)
                .with_seasonal_holt(0.5, 0.2)
                .with_seasonal_holt(0.7, 0.3);
            if let Some(p) = period {
                if p >= 2 {
                    m = m.with_period(p);
                }
            }
            Box::new(m)
        }
        RecipeKind::ContinuousPlainSkaters => {
            let mut f = LaplaceForecaster::new()
                .skaters()
                .with_scoring_horizon(horizon)
                .with_scoring_window(10)
                .learning_rate(0.20);
            if let Some(p) = period {
                if p >= 2 {
                    f = f.auto_with_seasonal_period(p);
                }
            }
            Box::new(f)
        }
    }
}

fn is_count_like(values: &[f64]) -> bool {
    if values.is_empty() {
        return false;
    }
    let n = values.len() as f64;
    let integer_frac = values
        .iter()
        .filter(|v| v.is_finite() && v.fract() == 0.0)
        .count() as f64
        / n;
    let zero_frac = values.iter().filter(|v| **v == 0.0).count() as f64 / n;
    integer_frac > 0.95 && zero_frac > 0.30
}

fn is_heavy_tailed(values: &[f64]) -> bool {
    if values.len() < 30 {
        return false;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if m2 < 1e-9 {
        return false;
    }
    let m4 = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let excess_kurt = m4 / (m2 * m2) - 3.0;
    excess_kurt > 5.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn ts(values: Vec<f64>) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let stamps: Vec<_> = (0..values.len())
            .map(|i| base + Duration::days(i as i64))
            .collect();
        TimeSeries::univariate(stamps, values).unwrap()
    }

    #[test]
    fn short_history_below_60() {
        let series = ts((0..50).map(|i| i as f64).collect());
        assert_eq!(recipe_for(&series, Some(12)), RecipeKind::ShortHistory);
    }

    #[test]
    fn retail_count_pattern() {
        // 100 obs, mostly integers, ~50 % zeros → M5-like.
        let mut vals = Vec::with_capacity(100);
        for i in 0..100 {
            vals.push(if i % 2 == 0 { 0.0 } else { (i % 5) as f64 });
        }
        let series = ts(vals);
        assert_eq!(recipe_for(&series, Some(7)), RecipeKind::RetailCountAid);
    }

    #[test]
    fn continuous_seasonal_with_period() {
        // Smooth sine + trend, N=200, period=12 given.
        let vals: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.02).sin() * 10.0 + i as f64 * 0.05)
            .collect();
        let series = ts(vals);
        assert_eq!(
            recipe_for(&series, Some(12)),
            RecipeKind::ContinuousMultiScale
        );
    }

    #[test]
    fn continuous_without_period_falls_back_to_plain() {
        let vals: Vec<f64> = (0..200).map(|i| (i as f64 * 0.02).sin() * 10.0).collect();
        let series = ts(vals);
        assert_eq!(
            recipe_for(&series, None),
            RecipeKind::ContinuousPlainSkaters
        );
    }

    #[test]
    fn heavy_tailed_triggers_crps_branch() {
        // Cauchy-like: mix of small values with occasional large spikes.
        let mut vals: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        vals[20] = 100.0;
        vals[40] = -80.0;
        vals[60] = 120.0;
        let series = ts(vals);
        assert_eq!(recipe_for(&series, None), RecipeKind::HeavyTailedCrps);
    }

    #[test]
    fn recommended_for_returns_fit_ready_forecaster() {
        let vals: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.02).sin() * 10.0 + 50.0)
            .collect();
        let series = ts(vals);
        let mut f = recommended_for(&series, 12, Some(12));
        <dyn DistributionalForecaster as crate::models::Forecaster>::fit(&mut *f, &series).unwrap();
        let dist = f.forecast_dist(12).unwrap();
        assert_eq!(dist.len(), 12);
        for g in &dist {
            assert!(g.mean().is_finite());
        }
    }

    #[test]
    fn recipe_names_are_unique() {
        let all = [
            RecipeKind::ShortHistory,
            RecipeKind::RetailCountAid,
            RecipeKind::HeavyTailedCrps,
            RecipeKind::ContinuousMultiScale,
            RecipeKind::ContinuousPlainSkaters,
        ];
        let names: std::collections::HashSet<_> = all.iter().map(|k| k.name()).collect();
        assert_eq!(names.len(), all.len());
    }
}
