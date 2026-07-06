//! `SmartForecaster` — feature-based routing across forecaster families.
//!
//! Inspects the training series at `fit()`, computes cheap characteristics
//! (zero fraction, trend strength, seasonality strength, `acf1`, coefficient
//! of variation), and picks the model family that the residual-slicing
//! evidence says wins on that segment. Routes to one of:
//!
//! * `LaplaceForecaster::new().with_intermittent_defaults().non_negative()` —
//!   for zero-inflated series (Croston beats AR-family EMA leaves on
//!   high-zero-fraction).
//! * `AutoETS` — for strongly-trending series (its damped-trend specs
//!   handle sustained trend better than any of our leaves).
//! * `LaplaceForecaster::new().auto()` — for everything else, delegating to
//!   the α-10 per-leaf auto-selector.
//!
//! Rules are deliberately conservative and evidence-derived. Wrong routes
//! only cost users the target family's fit; there's no expensive CV inside.
//! Callers that want CV-based selection should use [`AutoForecast`](super::auto_forecast::AutoForecast)
//! (which α-18 will extend to include `LaplaceForecaster`).

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::exponential::AutoETS;
use crate::models::traits::{validate_series_complete, FittedParams, Forecaster};
use crate::utils::ols::OLSResult;
use std::collections::HashMap;

#[cfg(feature = "distributional")]
use crate::models::laplace::LaplaceForecaster;

/// Family that `SmartForecaster` routed to.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectedFamily {
    #[cfg(feature = "distributional")]
    Intermittent,
    AutoEts,
    #[cfg(feature = "distributional")]
    LaplaceAuto,
}

pub struct SmartForecaster {
    inner: Option<Box<dyn Forecaster + Send>>,
    selected: Option<SelectedFamily>,
}

impl SmartForecaster {
    pub fn new() -> Self {
        Self {
            inner: None,
            selected: None,
        }
    }

    /// The family the router picked. `None` before `fit()` is called.
    pub fn selected_family(&self) -> Option<&SelectedFamily> {
        self.selected.as_ref()
    }
}

impl Default for SmartForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the routing characteristics on the training window.
fn routing_characteristics(train: &[f64]) -> RoutingChars {
    let n = train.len();
    if n < 2 {
        return RoutingChars::default();
    }
    let mean_y: f64 = train.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = train.iter().map(|y| (y - mean_y).powi(2)).sum();
    let zero_fraction = train.iter().filter(|&&y| y.abs() < 1e-9).count() as f64 / n as f64;

    // Trend strength.
    let t_mean = (n - 1) as f64 / 2.0;
    let (mut sum_ty, mut sum_tt) = (0.0, 0.0);
    for (t, y) in train.iter().enumerate() {
        let dt = t as f64 - t_mean;
        sum_ty += dt * (y - mean_y);
        sum_tt += dt * dt;
    }
    let slope = if sum_tt > 0.0 { sum_ty / sum_tt } else { 0.0 };
    let intercept = mean_y - slope * t_mean;
    let ss_res_trend: f64 = train
        .iter()
        .enumerate()
        .map(|(t, y)| (y - (intercept + slope * t as f64)).powi(2))
        .sum();
    let trend_strength = if ss_tot > 0.0 {
        (1.0 - ss_res_trend / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    RoutingChars {
        zero_fraction,
        trend_strength,
        mean_y,
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct RoutingChars {
    zero_fraction: f64,
    trend_strength: f64,
    mean_y: f64,
}

impl Forecaster for SmartForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "SmartForecaster requires at least one observation".into(),
            ));
        }
        let chars = routing_characteristics(values);

        // Rules (α-20 revision — derived from full-M5 loss analysis):
        //
        // - zero_fraction > 0.6 → highly intermittent; Croston wins on
        //   these even at low counts.
        // - low-count noise-dominated (mean < 3 && zero_fraction ∈ [0.2, 0.6])
        //   → AutoETS. Our Gaussian-mixture tails misbehave on integer
        //   counts near zero; AutoETS's damped model puts mass on realistic
        //   regions.
        // - trend_strength > 0.6 → sustained trend; AutoETS wins.
        // - Otherwise → LaplaceForecaster::auto() covers the residual
        //   space. `.auto()` now also handles zero-inflated seasonal via
        //   the seasonal-Croston leaf (see α-20 changes in forecaster.rs).
        let selected: SelectedFamily = if chars.zero_fraction > 0.6 {
            #[cfg(feature = "distributional")]
            {
                SelectedFamily::Intermittent
            }
            #[cfg(not(feature = "distributional"))]
            {
                SelectedFamily::AutoEts
            }
        } else if chars.mean_y < 3.0 && chars.zero_fraction > 0.2 && chars.zero_fraction <= 0.6 {
            SelectedFamily::AutoEts
        } else if chars.trend_strength > 0.6 {
            SelectedFamily::AutoEts
        } else {
            #[cfg(feature = "distributional")]
            {
                SelectedFamily::LaplaceAuto
            }
            #[cfg(not(feature = "distributional"))]
            {
                SelectedFamily::AutoEts
            }
        };

        let mut inner: Box<dyn Forecaster + Send> = match &selected {
            SelectedFamily::AutoEts => Box::new(AutoETS::new()),
            #[cfg(feature = "distributional")]
            SelectedFamily::Intermittent => Box::new(
                LaplaceForecaster::new()
                    .with_intermittent_defaults()
                    .non_negative(),
            ),
            #[cfg(feature = "distributional")]
            SelectedFamily::LaplaceAuto => Box::new(LaplaceForecaster::new().auto()),
        };
        inner.fit(series)?;
        self.inner = Some(inner);
        self.selected = Some(selected);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        match &self.inner {
            Some(m) => m.predict(horizon),
            None => Err(ForecastError::FitRequired {
                model: Some("SmartForecaster".into()),
            }),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        match &self.inner {
            Some(m) => m.predict_with_intervals(horizon, level),
            None => Err(ForecastError::FitRequired {
                model: Some("SmartForecaster".into()),
            }),
        }
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.inner.as_ref().and_then(|m| m.fitted_values())
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.inner.as_ref().and_then(|m| m.residuals())
    }

    fn training_values(&self) -> Result<&[f64]> {
        match &self.inner {
            Some(m) => m.training_values(),
            None => Err(ForecastError::FitRequired {
                model: Some("SmartForecaster".into()),
            }),
        }
    }

    fn fitted_params(&self) -> Option<FittedParams> {
        self.inner.as_ref().and_then(|m| m.fitted_params())
    }

    fn training_regressors(&self) -> Option<&HashMap<String, Vec<f64>>> {
        self.inner.as_ref().and_then(|m| m.training_regressors())
    }

    fn exog_coefficients(&self) -> Option<&OLSResult> {
        self.inner.as_ref().and_then(|m| m.exog_coefficients())
    }

    fn name(&self) -> &str {
        "SmartForecaster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn make_ts(vals: Vec<f64>) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let stamps: Vec<_> = (0..vals.len())
            .map(|i| base + Duration::hours(i as i64))
            .collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[cfg(feature = "distributional")]
    #[test]
    fn intermittent_series_routes_to_intermittent() {
        // Sparse series — 60% zeros → should route to intermittent.
        let mut vals = vec![0.0; 120];
        for i in (0..120).step_by(3) {
            vals[i] = 5.0;
        }
        let ts = make_ts(vals);
        let mut f = SmartForecaster::new();
        f.fit(&ts).unwrap();
        assert_eq!(f.selected_family(), Some(&SelectedFamily::Intermittent));
    }

    #[cfg(feature = "distributional")]
    #[test]
    fn steady_moderate_series_routes_to_laplace_auto() {
        let vals: Vec<f64> = (0..200)
            .map(|i| 50.0 + (i as f64 * 0.05).sin() * 5.0)
            .collect();
        let ts = make_ts(vals);
        let mut f = SmartForecaster::new();
        f.fit(&ts).unwrap();
        assert_eq!(f.selected_family(), Some(&SelectedFamily::LaplaceAuto));
    }

    #[test]
    fn strong_linear_trend_routes_to_ets() {
        let vals: Vec<f64> = (0..200).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = make_ts(vals);
        let mut f = SmartForecaster::new();
        f.fit(&ts).unwrap();
        assert_eq!(f.selected_family(), Some(&SelectedFamily::AutoEts));
    }
}
