//! `SmartForecaster` — AID-driven single-family Laplace commit.
//!
//! Given the AID demand classification of the training series
//! (`(demand_type, distribution)`), picks **one** Laplace configuration
//! that commits to that family — no leaf-mixture soup, no cross-family
//! delegation. Complements
//! [`LaplaceForecaster::auto_aid`](crate::models::laplace::LaplaceForecaster::auto_aid),
//! which uses AID to *add* a distribution-family leaf to the full
//! mixture. Smart *replaces* the mixture with a slimmed-down
//! single-family setup.
//!
//! Requires the default `postprocess` feature (for AID) and the
//! `distributional` feature (for the Laplace shell).
//!
//! # Design boundary — demand data only
//!
//! `SmartForecaster` is a **demand-forecasting** tool. Empirical
//! cross-panel evidence (see the
//! [`laplace` module docs](crate::models::laplace) for the full table):
//!
//! - **M5 full 30k (retail counts)**: Smart matches
//!   [`LaplaceForecaster::auto_aid`] within ~0.5 pp median MAE while
//!   being ~2× faster (single-family commit is cheaper than the full
//!   leaf mixture).
//! - **M3 monthly (macroeconomic)**: Smart regresses **~14 %** median
//!   MAE vs. plain
//!   [`LaplaceForecaster::auto`](crate::models::laplace::LaplaceForecaster::auto).
//!   Committing to a single distribution family is exactly wrong for
//!   smooth continuous economic data — the leaf-mixture soup that
//!   `auto()` runs is doing important work there.
//!
//! **Do not reach for `SmartForecaster` on non-demand panels.** On
//! economic / financial / continuous time series, use plain
//! [`LaplaceForecaster::new().auto()`](crate::models::laplace::LaplaceForecaster::auto)
//! or route via an upstream `AutoTheta` / `AutoETS` in application code.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::traits::{validate_series_complete, FittedParams, Forecaster};
use crate::utils::ols::OLSResult;
use std::collections::HashMap;

#[cfg(all(feature = "distributional", feature = "postprocess"))]
use crate::models::laplace::LaplaceForecaster;
#[cfg(feature = "postprocess")]
use crate::validation::aid::AidAnalyzer;
#[cfg(feature = "postprocess")]
use anofox_regression::solvers::{DemandDistribution, DemandType};

/// AID-derived label describing what family Smart committed to.
///
/// Enum-shaped rather than carrying the raw `AidSummary` so callers can
/// pattern-match cheaply. Reachable via
/// [`SmartForecaster::selected_family`].
#[derive(Debug, Clone, PartialEq)]
pub enum SelectedFamily {
    /// AID: `Intermittent + Poisson | Geometric`. Small-count
    /// intermittent demand.
    IntermittentPoisson,
    /// AID: `Intermittent + NegativeBinomial`. Overdispersed intermittent
    /// counts — the retail-SKU norm.
    IntermittentNegBinomial,
    /// AID: `Intermittent + RectifiedNormal`. Continuous demand with a
    /// point mass at zero.
    IntermittentRectifiedNormal,
    /// AID: `Intermittent + LogNormal | Gamma`. Positive skewed with
    /// zero clusters.
    IntermittentPositive,
    /// AID: `Regular + Poisson | Geometric | NegativeBinomial`. Count
    /// data without heavy zero-inflation.
    RegularCount,
    /// AID: `Regular + LogNormal | Gamma`. Positive skewed.
    RegularPositive,
    /// AID: `Regular + Normal`. Falls through to
    /// [`LaplaceForecaster::auto`](crate::models::laplace::LaplaceForecaster::auto).
    RegularNormal,
    /// AID was unavailable (feature off) — used the classical
    /// `LaplaceForecaster::auto()` fallback.
    Fallback,
}

pub struct SmartForecaster {
    inner: Option<Box<dyn Forecaster + Send>>,
    selected: Option<SelectedFamily>,
    seasonal_period: usize,
}

impl SmartForecaster {
    pub fn new() -> Self {
        Self {
            inner: None,
            selected: None,
            seasonal_period: 7,
        }
    }

    /// Override the seasonal period used when the AID family lands on
    /// something with a seasonality component (default 7, weekly).
    pub fn with_seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = period.max(2);
        self
    }

    /// The family Smart committed to. `None` before `fit()` is called.
    pub fn selected_family(&self) -> Option<&SelectedFamily> {
        self.selected.as_ref()
    }
}

impl Default for SmartForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "distributional", feature = "postprocess"))]
fn build_from_aid(
    demand_type: DemandType,
    distribution: DemandDistribution,
    seasonal_period: usize,
) -> (Box<dyn Forecaster + Send>, SelectedFamily) {
    match (demand_type, distribution) {
        (DemandType::Intermittent, DemandDistribution::Poisson)
        | (DemandType::Intermittent, DemandDistribution::Geometric) => {
            let m = LaplaceForecaster::new()
                .with_poisson_defaults()
                .with_seasonal_intermittent_defaults(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::IntermittentPoisson)
        }
        (DemandType::Intermittent, DemandDistribution::NegativeBinomial) => {
            let m = LaplaceForecaster::new()
                .with_negative_binomial_defaults()
                .with_seasonal_intermittent_defaults(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::IntermittentNegBinomial)
        }
        (DemandType::Intermittent, DemandDistribution::RectifiedNormal) => {
            let m = LaplaceForecaster::new()
                .with_rectified_normal_defaults()
                .with_seasonal_intermittent_defaults(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::IntermittentRectifiedNormal)
        }
        (DemandType::Intermittent, DemandDistribution::LogNormal) => {
            let m = LaplaceForecaster::new()
                .with_lognormal_defaults()
                .with_seasonal_intermittent_defaults(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::IntermittentPositive)
        }
        (DemandType::Intermittent, DemandDistribution::Gamma) => {
            let m = LaplaceForecaster::new()
                .with_gamma_defaults()
                .with_seasonal_intermittent_defaults(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::IntermittentPositive)
        }
        (DemandType::Intermittent, DemandDistribution::Normal) => {
            // Rare — regular normal shouldn't be intermittent per AID's rules,
            // but if it happens fall through to the intermittent leaf +
            // classical auto for the Gaussian branch.
            let m = LaplaceForecaster::new()
                .with_intermittent_defaults()
                .non_negative()
                .auto();
            (Box::new(m), SelectedFamily::IntermittentPositive)
        }
        (DemandType::Regular, DemandDistribution::Poisson)
        | (DemandType::Regular, DemandDistribution::Geometric) => {
            let m = LaplaceForecaster::new()
                .with_poisson_defaults()
                .with_seasonal(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::RegularCount)
        }
        (DemandType::Regular, DemandDistribution::NegativeBinomial) => {
            let m = LaplaceForecaster::new()
                .with_negative_binomial_defaults()
                .with_seasonal(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::RegularCount)
        }
        (DemandType::Regular, DemandDistribution::LogNormal) => {
            let m = LaplaceForecaster::new()
                .with_lognormal_defaults()
                .with_seasonal(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::RegularPositive)
        }
        (DemandType::Regular, DemandDistribution::Gamma) => {
            let m = LaplaceForecaster::new()
                .with_gamma_defaults()
                .with_seasonal(seasonal_period)
                .non_negative();
            (Box::new(m), SelectedFamily::RegularPositive)
        }
        (DemandType::Regular, DemandDistribution::RectifiedNormal) => {
            let m = LaplaceForecaster::new()
                .with_rectified_normal_defaults()
                .non_negative();
            (Box::new(m), SelectedFamily::RegularPositive)
        }
        (DemandType::Regular, DemandDistribution::Normal) => {
            let m = LaplaceForecaster::new().auto();
            (Box::new(m), SelectedFamily::RegularNormal)
        }
    }
}

impl Forecaster for SmartForecaster {
    #[allow(unused_variables)] // seasonal_period unused when features off
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "SmartForecaster requires at least one observation".into(),
            ));
        }

        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        let (mut inner, selected) = {
            let aid = AidAnalyzer::new().analyze(values);
            let summary = aid.summary();
            build_from_aid(
                summary.demand_type,
                summary.distribution,
                self.seasonal_period,
            )
        };

        #[cfg(not(all(feature = "distributional", feature = "postprocess")))]
        let (mut inner, selected): (Box<dyn Forecaster + Send>, SelectedFamily) = {
            // Without AID + Laplace, fall back to a classical baseline —
            // AutoTheta (not ETS, per project directive).
            (
                Box::new(crate::models::theta::AutoTheta::new()),
                SelectedFamily::Fallback,
            )
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

    #[cfg(all(feature = "distributional", feature = "postprocess"))]
    #[test]
    fn intermittent_count_series_routes_to_intermittent_family() {
        let mut vals = vec![0.0; 200];
        for i in (0..200).step_by(3) {
            vals[i] = 2.0;
        }
        let ts = make_ts(vals);
        let mut f = SmartForecaster::new();
        f.fit(&ts).unwrap();
        assert!(matches!(
            f.selected_family(),
            Some(SelectedFamily::IntermittentPoisson)
                | Some(SelectedFamily::IntermittentNegBinomial)
                | Some(SelectedFamily::IntermittentRectifiedNormal)
                | Some(SelectedFamily::IntermittentPositive)
        ));
        let fc = f.predict(10).unwrap();
        for v in fc.primary() {
            assert!(*v >= 0.0);
        }
    }

    #[cfg(all(feature = "distributional", feature = "postprocess"))]
    #[test]
    fn continuous_regular_normal_falls_through_to_auto() {
        let vals: Vec<f64> = (0..200)
            .map(|i| 50.0 + (i as f64 * 0.05).sin() * 5.0)
            .collect();
        let ts = make_ts(vals);
        let mut f = SmartForecaster::new();
        f.fit(&ts).unwrap();
        // Continuous smooth data → Regular family — could be Normal or
        // one of the positive variants depending on AID's IC choice.
        assert!(matches!(
            f.selected_family(),
            Some(SelectedFamily::RegularNormal)
                | Some(SelectedFamily::RegularPositive)
                | Some(SelectedFamily::RegularCount)
        ));
    }
}
