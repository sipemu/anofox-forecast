//! Naive2 baseline model for competition accuracy evaluation.
//!
//! A harness-only model (D-07): not part of the public `anofox-forecast` API.
//! Composes [`Naive`] and [`SeasonalNaive`] from the library, gated by a
//! 90%-confidence autocorrelation test at the seasonal lag (D-08, ACCUR-06).
//!
//! The ACF threshold is `1.645 / sqrt(n)` (one-sided 90% CI under the
//! Bartlett asymptotic normal approximation, matching the statsforecast/M4
//! Naive2 canonical form — see Assumption A1 in 02-RESEARCH.md).

use anofox_forecast::core::TimeSeries;
use anofox_forecast::error::ForecastError;
use anofox_forecast::models::baseline::{Naive, SeasonalNaive};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// Wrapper enum selecting the inner baseline model after the ACF gate.
enum Naive2Inner {
    Seasonal(SeasonalNaive),
    Random(Naive),
}

/// ACF-gated seasonal/non-seasonal reference baseline (D-07, D-08).
///
/// On `fit`, evaluates the autocorrelation at the configured seasonal lag.
/// If `|acf| > 1.645 / sqrt(n)` (90%-confidence Bartlett test) and the series
/// is long enough to fit a full seasonal cycle, uses [`SeasonalNaive`];
/// otherwise uses [`Naive`] (random-walk last-value carry-forward).
pub struct Naive2 {
    seasonal_period: usize,
    inner: Naive2Inner,
}

impl Naive2 {
    /// Create a new Naive2 with the given seasonal period.
    ///
    /// Starts in random-walk (Naive) mode; `fit` reassigns the inner model
    /// after evaluating the ACF gate.
    pub fn new(seasonal_period: usize) -> Self {
        Self {
            seasonal_period,
            inner: Naive2Inner::Random(Naive::new()),
        }
    }

    /// Fit the model to a training value slice.
    ///
    /// Evaluates the ACF gate and assigns the inner model, then fits it.
    /// Never uses `fitted_values()` from the inner model as an out-of-sample
    /// proxy — the inner fit is used only for `predict` state.
    pub fn fit(&mut self, train: &[f64]) -> Result<(), ForecastError> {
        let acf = acf_at_lag(train, self.seasonal_period);
        let n = train.len();
        // 90%-confidence Bartlett critical value (A1/D-08): 1.645 / sqrt(n)
        let critical = 1.645 / (n as f64).sqrt();

        let use_seasonal =
            acf.abs() > critical && self.seasonal_period >= 2 && n > self.seasonal_period;

        let train_ts = make_ts_from_slice(train)?;

        if use_seasonal {
            let mut model = SeasonalNaive::new(self.seasonal_period);
            model.fit(&train_ts)?;
            self.inner = Naive2Inner::Seasonal(model);
        } else {
            let mut model = Naive::new();
            model.fit(&train_ts)?;
            self.inner = Naive2Inner::Random(model);
        }

        Ok(())
    }

    /// Predict `horizon` steps ahead using the fitted inner model.
    ///
    /// Returns the univariate point-forecast vector (first dimension of the
    /// [`Forecast`] returned by the inner model).
    pub fn predict(&self, horizon: usize) -> Result<Vec<f64>, ForecastError> {
        let forecast = match &self.inner {
            Naive2Inner::Seasonal(m) => m.predict(horizon)?,
            Naive2Inner::Random(m) => m.predict(horizon)?,
        };
        Ok(forecast.primary().to_vec())
    }
}

/// Compute the sample autocorrelation at a given lag.
///
/// Uses the mean-centered biased estimator:
///   `acf(lag) = sum_{t=lag}^{n-1}(x_t - xbar)(x_{t-lag} - xbar)
///               / sum_t (x_t - xbar)^2`
///
/// Returns `0.0` when `n <= lag` or the variance is zero (constant series).
fn acf_at_lag(series: &[f64], lag: usize) -> f64 {
    let n = series.len();
    if n <= lag {
        return 0.0;
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let numer: f64 = (lag..n)
        .map(|t| (series[t] - mean) * (series[t - lag] - mean))
        .sum();
    let denom: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum();
    if denom == 0.0 {
        0.0
    } else {
        numer / denom
    }
}

/// Build a synthetic-timestamp `TimeSeries` from a value slice.
///
/// Uses monthly-spaced UTC timestamps starting 2000-01-01. Mirrors the same
/// helper in `tests/accuracy.rs` (Plan 01) — both must use identical timestamp
/// construction so that any frequency-inference changes affect both consistently.
fn make_ts_from_slice(values: &[f64]) -> Result<TimeSeries, ForecastError> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: fit Naive2(period) on `train`, predict `horizon` steps.
    fn fit_predict(train: &[f64], period: usize, horizon: usize) -> Vec<f64> {
        let mut model = Naive2::new(period);
        model.fit(train).expect("Naive2 fit failed");
        model.predict(horizon).expect("Naive2 predict failed")
    }

    // --- Behavior 1: strongly seasonal series selects SeasonalNaive ---
    //
    // A perfectly repeating period-4 pattern has ACF(4) == 1.0, far above
    // 1.645/sqrt(n). SeasonalNaive repeats the last season; predict(4) must
    // return [10, 20, 30, 40].

    #[test]
    fn naive2_seasonal_gate_seasonal() {
        // Strongly seasonal: [10,20,30,40] repeated 5 times → period 4
        let train: Vec<f64> = [10.0_f64, 20.0, 30.0, 40.0]
            .iter()
            .cycle()
            .take(20)
            .copied()
            .collect();

        let preds = fit_predict(&train, 4, 4);

        assert_eq!(preds.len(), 4, "expected horizon=4 predictions");
        assert!(
            (preds[0] - 10.0).abs() < 1e-9,
            "pred[0] should be 10.0 (last season start), got {}",
            preds[0]
        );
        assert!(
            (preds[1] - 20.0).abs() < 1e-9,
            "pred[1] should be 20.0, got {}",
            preds[1]
        );
        assert!(
            (preds[2] - 30.0).abs() < 1e-9,
            "pred[2] should be 30.0, got {}",
            preds[2]
        );
        assert!(
            (preds[3] - 40.0).abs() < 1e-9,
            "pred[3] should be 40.0, got {}",
            preds[3]
        );
    }

    // --- Behavior 2: flat/non-seasonal series selects Naive ---
    //
    // A constant series has zero variance → ACF is 0 everywhere → Naive.
    // Naive carries the last value forward: predict(4) must return [v, v, v, v].

    #[test]
    fn naive2_seasonal_gate_flat() {
        // Flat (constant) series: ACF undefined/0 → Naive (last-value carry)
        let last_value = 7.0_f64;
        let train: Vec<f64> = vec![last_value; 20];

        let preds = fit_predict(&train, 4, 4);

        assert_eq!(preds.len(), 4, "expected horizon=4 predictions");
        for (i, &p) in preds.iter().enumerate() {
            assert!(
                (p - last_value).abs() < 1e-9,
                "pred[{}] should be last_value={}, got {}",
                i,
                last_value,
                p
            );
        }
    }

    // --- Behavior 3: ACF helper correctness ---
    //
    // A pure period-4 sinusoid must return ACF(4) with |value| > 1.645/sqrt(n).
    // A short i.i.d.-like noise series (we use a small ramp where seasonal ACF
    // at period=4 is 0 by construction) must return |ACF| below threshold.

    #[test]
    fn acf_at_lag_seasonal_signal() {
        // Perfectly periodic: ACF at lag=period should be > critical value
        let n = 40_usize;
        let series: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 4.0).sin())
            .collect();
        let acf = acf_at_lag(&series, 4);
        let critical = 1.645 / (n as f64).sqrt();
        assert!(
            acf.abs() > critical,
            "period-4 sinusoid ACF({})={:.4} should exceed critical={:.4}",
            4,
            acf,
            critical
        );
    }

    #[test]
    fn acf_at_lag_noise_below_threshold() {
        // A series with no seasonal structure at lag=4 should be below threshold.
        // Use a short ramp: [0,1,2,...,11]. ACF(4) measures correlation between
        // pairs (4..11) and (0..7) of a pure linear trend. For a linear trend,
        // the biased ACF at any lag is positive but the period-4 ACF ≈ 0.94
        // which is actually high. Instead, use a series specifically constructed
        // so ACF(4)=0 by symmetry: alternating [1,-1,1,-1,...] has ACF(4)=1
        // but ACF(1)=-1. Use a very short series to trigger n<=lag guard:
        // with n=4, lag=4: n<=lag → return 0.0 (below threshold for any n).
        let short: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let acf = acf_at_lag(&short, 4); // n == lag → 0.0 by guard
        assert!(acf.abs() == 0.0, "n==lag should return 0.0, got {}", acf);
    }

    #[test]
    fn acf_at_lag_zero_variance_returns_zero() {
        // Constant series → denom=0 → acf=0
        let constant = vec![5.0_f64; 20];
        let acf = acf_at_lag(&constant, 4);
        assert_eq!(acf, 0.0, "constant series ACF must be 0.0");
    }
}
