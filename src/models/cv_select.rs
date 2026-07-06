//! `CvSelectForecaster` — per-series CV-based model selection.
//!
//! Given a slate of candidate forecasters, fits each on the earlier portion
//! of the training series, scores on a held-out final window, and picks the
//! winner by MAE. The winner is refit on the full training series and used
//! for all downstream predictions.
//!
//! Complements [`SmartForecaster`](super::smart::SmartForecaster) which
//! routes by *series characteristics* (fast, no extra fits). Use `CvSelect`
//! when accuracy matters more than fit-time compute.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::traits::{validate_series_complete, FittedParams, Forecaster};
use crate::utils::ols::OLSResult;
use std::collections::HashMap;

/// One candidate model — a name and a factory that produces a fresh boxed
/// forecaster on demand. Factories return owned instances so the selector
/// can fit them multiple times (once for the CV score, again for the
/// final refit on full data).
pub struct Candidate {
    pub name: String,
    pub factory: Box<dyn Fn() -> Box<dyn Forecaster + Send> + Send + Sync>,
}

impl Candidate {
    pub fn new<F>(name: impl Into<String>, factory: F) -> Self
    where
        F: Fn() -> Box<dyn Forecaster + Send> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            factory: Box::new(factory),
        }
    }
}

pub struct CvSelectForecaster {
    candidates: Vec<Candidate>,
    /// Number of observations reserved as the holdout window. Defaults to
    /// `max(14, series_len / 10)`.
    holdout: Option<usize>,
    winner_name: Option<String>,
    winner_score: Option<f64>,
    inner: Option<Box<dyn Forecaster + Send>>,
}

impl CvSelectForecaster {
    pub fn new(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates,
            holdout: None,
            winner_name: None,
            winner_score: None,
            inner: None,
        }
    }

    /// Override the holdout window length. Must be at least 3.
    pub fn with_holdout(mut self, holdout: usize) -> Self {
        self.holdout = Some(holdout.max(3));
        self
    }

    pub fn winner_name(&self) -> Option<&str> {
        self.winner_name.as_deref()
    }

    pub fn winner_score(&self) -> Option<f64> {
        self.winner_score
    }
}

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    if pred.is_empty() {
        return f64::INFINITY;
    }
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

impl Forecaster for CvSelectForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        if self.candidates.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "CvSelectForecaster requires at least one candidate".into(),
            ));
        }
        let values = series.primary_values();
        let n = values.len();
        let holdout = self.holdout.unwrap_or_else(|| (n / 10).max(14));
        if n <= holdout + 5 {
            return Err(ForecastError::InvalidParameter(format!(
                "series too short for CV: n={}, holdout={}",
                n, holdout
            )));
        }

        // Split off the training portion.
        let train_len = n - holdout;
        let train_values = values[..train_len].to_vec();
        let holdout_values = &values[train_len..];
        // Rebuild a TimeSeries from the training values.
        let stamps = series.timestamps().to_vec();
        let train_stamps = stamps[..train_len].to_vec();
        let train_ts = TimeSeries::univariate(train_stamps, train_values.clone())?;

        // Score each candidate on the holdout window.
        let mut best_idx = 0usize;
        let mut best_score = f64::INFINITY;
        for (i, cand) in self.candidates.iter().enumerate() {
            let mut m = (cand.factory)();
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let Ok(fc) = m.predict(holdout) else {
                continue;
            };
            let pred = fc.primary();
            if pred.len() != holdout_values.len() {
                continue;
            }
            let score = mae(pred, holdout_values);
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }
        if !best_score.is_finite() {
            return Err(ForecastError::InvalidParameter(
                "no candidate produced a finite CV score".into(),
            ));
        }
        self.winner_name = Some(self.candidates[best_idx].name.clone());
        self.winner_score = Some(best_score);

        // Refit the winner on the FULL series.
        let mut winner = (self.candidates[best_idx].factory)();
        winner.fit(series)?;
        self.inner = Some(winner);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        match &self.inner {
            Some(m) => m.predict(horizon),
            None => Err(ForecastError::FitRequired {
                model: Some("CvSelectForecaster".into()),
            }),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        match &self.inner {
            Some(m) => m.predict_with_intervals(horizon, level),
            None => Err(ForecastError::FitRequired {
                model: Some("CvSelectForecaster".into()),
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
                model: Some("CvSelectForecaster".into()),
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
        "CvSelectForecaster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::exponential::AutoETS;
    use crate::models::theta::AutoTheta;
    use chrono::{Duration, TimeZone, Utc};

    fn ts_linear(n: usize) -> TimeSeries {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let vals: Vec<f64> = (0..n).map(|i| 10.0 + 2.0 * i as f64).collect();
        let stamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
        TimeSeries::univariate(stamps, vals).unwrap()
    }

    #[test]
    fn selects_a_winner_from_a_two_candidate_slate() {
        let candidates = vec![
            Candidate::new("AutoETS", || Box::new(AutoETS::new())),
            Candidate::new("AutoTheta", || Box::new(AutoTheta::new())),
        ];
        let ts = ts_linear(200);
        let mut sel = CvSelectForecaster::new(candidates).with_holdout(20);
        sel.fit(&ts).unwrap();
        let winner = sel.winner_name().unwrap();
        assert!(
            winner == "AutoETS" || winner == "AutoTheta",
            "unexpected winner: {}",
            winner
        );
        let fc = sel.predict(10).unwrap();
        assert_eq!(fc.primary().len(), 10);
    }
}
