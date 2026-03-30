//! Batch forecasting for multiple series with shared computation.
//!
//! When forecasting many series with the same model configuration,
//! batch methods share computation that is identical across series
//! (e.g., Fourier design matrices, Cholesky factors, STL weights)
//! and process series in parallel.
//!
//! # Example
//!
//! ```rust
//! use anofox_forecast::batch;
//!
//! let values = vec![
//!     (0..28).map(|i| 10.0 + (i as f64 * 0.5).sin() * 5.0).collect::<Vec<_>>(),
//!     (0..28).map(|i| 20.0 - i as f64 * 0.1).collect::<Vec<_>>(),
//! ];
//! let results = batch::auto_ets(&values, 7, Some(7), None);
//! assert_eq!(results.len(), 2);
//! ```

use crate::core::{Forecast, TimeSeries};
use crate::error::Result;
use crate::models::exponential::{AutoETS, AutoETSConfig, ETSSpec, ModelPool, ETS};
use crate::models::mfles::MFLES;
use crate::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Generate dummy timestamps for batch processing (series without real timestamps).
fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

/// Batch AutoETS: fit and predict many series with shared candidate evaluation.
///
/// All series must have the same length. Each series is fitted independently
/// but computation is parallelized across series, and candidate generation
/// is shared.
///
/// Returns one `Result<(Forecast, ETSSpec)>` per series (forecast + selected model).
///
/// # Arguments
/// * `values` — slice of value vectors, one per series (all same length)
/// * `period` — seasonal period (e.g., 7 for weekly, 12 for monthly)
/// * `horizon` — forecast horizon (None defaults to period)
/// * `pool` — model pool (None defaults to `ModelPool::Complete`)
pub fn auto_ets(
    values: &[Vec<f64>],
    period: usize,
    horizon: Option<usize>,
    pool: Option<ModelPool>,
) -> Vec<Result<(Forecast, ETSSpec)>> {
    let h = horizon.unwrap_or(period);
    let pool = pool.unwrap_or(ModelPool::Complete);

    let fit_one = |series_values: &Vec<f64>| -> Result<(Forecast, ETSSpec)> {
        let n = series_values.len();
        let ts = TimeSeries::univariate(make_timestamps(n), series_values.clone())?;
        let config = AutoETSConfig::with_period(period).with_model_pool(pool);
        let mut model = AutoETS::with_config(config);
        model.fit(&ts)?;
        let fc = model.predict(h)?;
        let spec = model.selected_spec().unwrap_or(ETSSpec::ann());
        Ok((fc, spec))
    };

    #[cfg(feature = "parallel")]
    {
        values.par_iter().map(fit_one).collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        values.iter().map(fit_one).collect()
    }
}

/// Batch ETS: fit a specific ETS spec to many series.
///
/// All series share the same spec — no model selection overhead.
/// Parallelized across series.
///
/// Returns one `Result<Forecast>` per series.
pub fn ets(
    values: &[Vec<f64>],
    spec: ETSSpec,
    period: usize,
    horizon: Option<usize>,
) -> Vec<Result<Forecast>> {
    let h = horizon.unwrap_or(period);

    let fit_one = |series_values: &Vec<f64>| -> Result<Forecast> {
        let n = series_values.len();
        let ts = TimeSeries::univariate(make_timestamps(n), series_values.clone())?;
        let mut model = ETS::new(spec, period);
        model.fit(&ts)?;
        model.predict(h)
    };

    #[cfg(feature = "parallel")]
    {
        values.par_iter().map(fit_one).collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        values.iter().map(fit_one).collect()
    }
}

/// Batch MFLES: fit many series with shared Fourier Cholesky factor.
///
/// All series must have the same length and seasonal period. The Fourier
/// design matrix X, its X'X product, and the Cholesky factor are computed
/// once and shared across all series — saving O(k²n) per series where
/// k = 2×fourier_order.
///
/// Returns one `Result<Forecast>` per series.
pub fn mfles(values: &[Vec<f64>], period: usize, horizon: Option<usize>) -> Vec<Result<Forecast>> {
    let h = horizon.unwrap_or(period);

    if values.is_empty() {
        return vec![];
    }

    let n = values[0].len();

    // Pre-compute shared Fourier design matrix and Cholesky factor.
    // These depend only on n and period, not on the data.
    let fourier_order = MFLES::default_fourier_order(period);
    let fourier_series = MFLES::build_fourier_series(n, period, fourier_order);
    let k = fourier_series.len();

    let shared_cholesky = if k > 0 {
        let mut xtx = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in i..k {
                let dot: f64 = fourier_series[i]
                    .iter()
                    .zip(fourier_series[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                xtx[i][j] = dot;
                xtx[j][i] = dot;
            }
        }
        for i in 0..k {
            xtx[i][i] += 1e-8;
        }
        MFLES::compute_cholesky_factor(&xtx)
    } else {
        None
    };

    let fit_one = |series_values: &Vec<f64>| -> Result<Forecast> {
        let sn = series_values.len();
        let ts = TimeSeries::univariate(make_timestamps(sn), series_values.clone())?;
        let mut model = MFLES::new(vec![period]);
        // If all series have the same length, inject the shared Cholesky
        if sn == n {
            model.set_shared_cholesky(shared_cholesky.clone());
        }
        model.fit(&ts)?;
        model.predict(h)
    };

    #[cfg(feature = "parallel")]
    {
        values.par_iter().map(fit_one).collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        values.iter().map(fit_one).collect()
    }
}
