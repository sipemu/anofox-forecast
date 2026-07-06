//! `LaplaceForecaster` — online distributional shell over EMA / drift / AR(1).
//!
//! Alpha surface (behind the `distributional` feature). Inspired by
//! [`microprediction/skaters`](https://github.com/microprediction/skaters):
//! streaming leaves, likelihood-weighted mixture, per-horizon
//! [`GaussianMixture`] output. Only the shell
//! and three cheap leaves are implemented — no CRPS terminal, no
//! seasonal / OU / fractional-differencing / Yeo-Johnson leaves.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::inspect::{Explanation, Inspectable, LaplaceExplanation};
use crate::models::traits::{validate_series_complete, Forecaster};

use super::dist::GaussianMixture;
use super::ensemble::{blend_horizon, softmax};
use super::leaf::Leaf;
use super::leaves::{Ar1Leaf, DriftLeaf, EmaLeaf, SeasonalEmaLeaf};
use super::DistributionalForecaster;

/// Distributional forecaster returning a `GaussianMixture` per horizon.
///
/// Wraps three streaming leaves (EMA, drift, AR(1)) and mixes them by
/// cumulative one-step log-likelihood. Optionally includes a seasonal-EMA
/// leaf when a period is supplied via [`Self::with_seasonal`] — pass the
/// period explicitly (no auto-detection).
pub struct LaplaceForecaster {
    ema_alpha: f64,
    drift_alpha: f64,
    ar_alpha_mean: f64,
    seasonal_period: Option<usize>,
    seasonal_alpha: f64,

    leaves: Vec<Box<dyn Leaf + Send>>,
    cum_log_liks: Vec<f64>,
    n_obs: usize,

    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    training_values: Vec<f64>,
}

impl LaplaceForecaster {
    /// Default configuration: EMA α=0.2, drift α=0.1, AR(1) mean α=0.1;
    /// no seasonal leaf.
    pub fn new() -> Self {
        Self::with_alphas(0.2, 0.1, 0.1)
    }

    pub fn with_alphas(ema_alpha: f64, drift_alpha: f64, ar_alpha_mean: f64) -> Self {
        Self {
            ema_alpha,
            drift_alpha,
            ar_alpha_mean,
            seasonal_period: None,
            seasonal_alpha: 0.15,
            leaves: Vec::new(),
            cum_log_liks: Vec::new(),
            n_obs: 0,
            fitted_values: Vec::new(),
            residuals: Vec::new(),
            training_values: Vec::new(),
        }
    }

    /// Add a seasonal-EMA leaf with the caller-supplied period. A period
    /// of 0 or 1 is treated as "no seasonal leaf" — no runtime error.
    pub fn with_seasonal(mut self, period: usize) -> Self {
        if period >= 2 {
            self.seasonal_period = Some(period);
        }
        self
    }

    /// Override the smoothing rate for the seasonal-EMA leaf. Only meaningful
    /// after `with_seasonal(period)` has been called. Clamped by the leaf.
    pub fn seasonal_alpha(mut self, alpha: f64) -> Self {
        self.seasonal_alpha = alpha;
        self
    }

    fn init_leaves(&mut self) {
        let mut leaves: Vec<Box<dyn Leaf + Send>> = vec![
            Box::new(EmaLeaf::new(self.ema_alpha)),
            Box::new(DriftLeaf::new(self.drift_alpha)),
            Box::new(Ar1Leaf::new(self.ar_alpha_mean)),
        ];
        if let Some(p) = self.seasonal_period {
            leaves.push(Box::new(SeasonalEmaLeaf::new(p, self.seasonal_alpha)));
        }
        self.cum_log_liks = vec![0.0; leaves.len()];
        self.leaves = leaves;
    }

    fn weights(&self) -> Vec<f64> {
        softmax(&self.cum_log_liks)
    }

    fn per_leaf_horizons(&self, horizon: usize) -> Vec<Vec<super::dist::Gaussian>> {
        self.leaves.iter().map(|l| l.predict(horizon)).collect()
    }
}

impl Default for LaplaceForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for LaplaceForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "LaplaceForecaster requires at least one observation".into(),
            ));
        }

        self.init_leaves();
        self.training_values = values.to_vec();
        self.fitted_values = Vec::with_capacity(values.len());
        self.residuals = Vec::with_capacity(values.len());
        self.n_obs = 0;

        for &y in values {
            // 1-step predictions from each leaf, before observing y.
            let per_leaf: Vec<super::dist::Gaussian> =
                self.leaves.iter().map(|l| l.predict(1)[0]).collect();
            let weights = self.weights();

            let mixture =
                GaussianMixture::new(weights.iter().zip(per_leaf.iter()).map(|(w, g)| (*w, *g)));
            let fitted = if mixture.is_empty() {
                y
            } else {
                mixture.mean()
            };
            self.fitted_values.push(fitted);
            self.residuals.push(y - fitted);

            // Score each leaf on this y, then absorb.
            for (i, leaf) in self.leaves.iter_mut().enumerate() {
                let g = per_leaf[i];
                let lp = g.logpdf(y);
                if lp.is_finite() {
                    self.cum_log_liks[i] += lp;
                }
                leaf.observe(y);
            }
            self.n_obs += 1;
        }
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }
        let mixtures = self.forecast_dist(horizon)?;
        let points: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
        Ok(Forecast::from_values(points))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if !(0.0..1.0).contains(&level) {
            return Err(ForecastError::InvalidParameter(format!(
                "confidence level must be in [0, 1), got {level}"
            )));
        }
        let mixtures = self.forecast_dist(horizon)?;
        let alpha = 1.0 - level;
        let lo_p = alpha / 2.0;
        let hi_p = 1.0 - alpha / 2.0;
        let points: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
        let lower: Vec<f64> = mixtures.iter().map(|m| m.quantile(lo_p)).collect();
        let upper: Vec<f64> = mixtures.iter().map(|m| m.quantile(hi_p)).collect();
        Ok(Forecast::from_values_with_intervals(points, lower, upper))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        if self.fitted_values.is_empty() {
            None
        } else {
            Some(&self.fitted_values)
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        if self.residuals.is_empty() {
            None
        } else {
            Some(&self.residuals)
        }
    }

    fn training_values(&self) -> Result<&[f64]> {
        if self.training_values.is_empty() {
            Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            })
        } else {
            Ok(&self.training_values)
        }
    }

    fn name(&self) -> &str {
        "LaplaceForecaster"
    }

    fn explanation(&self) -> Result<Explanation> {
        <Self as Inspectable>::explanation(self)
    }
}

impl Inspectable for LaplaceForecaster {
    fn explanation(&self) -> Result<Explanation> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        let horizon = 8;
        let mixtures = self.forecast_dist(horizon)?;
        let weights = self.weights();
        let names = self.leaves.iter().map(|l| l.name().to_string()).collect();
        Ok(Explanation::Laplace(LaplaceExplanation {
            horizon_dists: mixtures,
            leaf_weights: weights,
            leaf_names: names,
            fitted_values: self.fitted_values.clone(),
            residuals: self.residuals.clone(),
        }))
    }
}

impl DistributionalForecaster for LaplaceForecaster {
    fn forecast_dist(&self, horizon: usize) -> Result<Vec<GaussianMixture>> {
        if self.leaves.is_empty() {
            return Err(ForecastError::FitRequired {
                model: Some("LaplaceForecaster".into()),
            });
        }
        if horizon == 0 {
            return Ok(Vec::new());
        }
        let weights = self.weights();
        let per_leaf = self.per_leaf_horizons(horizon);
        Ok((0..horizon)
            .map(|h| blend_horizon(&weights, &per_leaf, h))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use chrono::{Duration, TimeZone, Utc};

    fn ts_ar1(n: usize, phi: f64) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut vals = Vec::with_capacity(n);
        let mut y = 0.0;
        for i in 0..n {
            let eps = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
            y = phi * y + eps;
            vals.push(y);
        }
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn fit_and_forecast_dist_returns_mixture_per_horizon() {
        let ts = ts_ar1(200, 0.6);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(5).unwrap();
        assert_eq!(dists.len(), 5);
        for d in &dists {
            assert_eq!(d.components.len(), 3);
            let ws: f64 = d.components.iter().map(|(w, _)| w).sum();
            assert!((ws - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn predict_matches_mixture_means() {
        let ts = ts_ar1(150, 0.5);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(3).unwrap();
        let fc = f.predict(3).unwrap();
        let means: Vec<f64> = dists.iter().map(|m| m.mean()).collect();
        assert_eq!(fc.primary(), means.as_slice());
    }

    #[test]
    fn predict_before_fit_errors() {
        let f = LaplaceForecaster::new();
        assert!(matches!(
            f.predict(1),
            Err(ForecastError::FitRequired { .. })
        ));
        assert!(matches!(
            f.forecast_dist(1),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn intervals_are_ordered() {
        let ts = ts_ar1(120, 0.4);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        let fc = f.predict_with_intervals(3, 0.90).unwrap();
        let lower = fc.lower_series(0).unwrap();
        let upper = fc.upper_series(0).unwrap();
        let point = fc.primary();
        for i in 0..3 {
            assert!(lower[i] <= point[i] && point[i] <= upper[i]);
        }
    }

    #[test]
    fn explanation_after_fit_matches_leaf_names() {
        let ts = ts_ar1(80, 0.5);
        let mut f = LaplaceForecaster::new();
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1"]);
                assert_eq!(e.leaf_weights.len(), 3);
                assert!(!e.fitted_values.is_empty());
                assert_eq!(e.fitted_values.len(), e.residuals.len());
                assert_eq!(e.horizon_dists.len(), 8);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    fn ts_seasonal(n: usize, period: usize) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin()
                    + 50.0
            })
            .collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn with_seasonal_adds_seasonal_leaf_and_helps_periodic_series() {
        let ts = ts_seasonal(240, 12);
        let mut plain = LaplaceForecaster::new();
        let mut seasonal = LaplaceForecaster::new().with_seasonal(12);
        plain.fit(&ts).unwrap();
        seasonal.fit(&ts).unwrap();

        match Inspectable::explanation(&seasonal).unwrap() {
            Explanation::Laplace(e) => {
                assert_eq!(e.leaf_names, vec!["ema", "drift", "ar1", "seasonal_ema"]);
                assert_eq!(e.leaf_weights.len(), 4);
            }
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }

        // On a pure periodic series the seasonal fitted residual should be
        // smaller than the plain fitted residual (mean absolute residual).
        let plain_mae: f64 = plain
            .residuals()
            .unwrap()
            .iter()
            .map(|r| r.abs())
            .sum::<f64>()
            / plain.residuals().unwrap().len() as f64;
        let seasonal_mae: f64 = seasonal
            .residuals()
            .unwrap()
            .iter()
            .map(|r| r.abs())
            .sum::<f64>()
            / seasonal.residuals().unwrap().len() as f64;
        assert!(
            seasonal_mae < plain_mae,
            "seasonal MAR ({}) should beat plain MAR ({}) on a pure periodic series",
            seasonal_mae,
            plain_mae
        );
    }

    #[test]
    fn with_seasonal_period_lt_2_is_a_no_op() {
        let ts = ts_ar1(100, 0.4);
        let mut f = LaplaceForecaster::new().with_seasonal(1);
        f.fit(&ts).unwrap();
        match Inspectable::explanation(&f).unwrap() {
            Explanation::Laplace(e) => assert_eq!(e.leaf_names.len(), 3),
            other => panic!("expected Explanation::Laplace, got {other:?}"),
        }
    }

    #[test]
    fn explanation_before_fit_errors() {
        let f = LaplaceForecaster::new();
        assert!(matches!(
            Inspectable::explanation(&f),
            Err(ForecastError::FitRequired { .. })
        ));
    }
}
