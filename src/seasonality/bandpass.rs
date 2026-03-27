//! Band-pass filters for business-cycle extraction.
//!
//! Provides two classic frequency-domain filters that isolate cyclical
//! components within a specified period range:
//!
//! - **Christiano-Fitzgerald (CF)**: asymmetric, time-varying weights that
//!   preserve the full sample length. See Christiano & Fitzgerald (2003),
//!   "The Band Pass Filter", *International Economic Review*.
//!
//! - **Baxter-King (BK)**: symmetric, fixed-length weights with a lead/lag
//!   truncation parameter `k`, losing `2k` observations. See Baxter & King
//!   (1999), "Measuring Business Cycles", *Review of Economics and Statistics*.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::bandpass::{cf_filter, bk_filter};
//!
//! // Quarterly GDP with NBER business-cycle band (6-32 quarters)
//! let data: Vec<f64> = (0..120)
//!     .map(|i| 100.0 + 0.5 * i as f64 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
//!     .collect();
//!
//! let cf = cf_filter(&data, 6, 32, true).unwrap();
//! assert_eq!(cf.cycle.len(), data.len());
//!
//! let bk = bk_filter(&data, 6, 32, 12).unwrap();
//! assert_eq!(bk.cycle.len(), data.len()); // padded with NaN at ends
//! ```

use crate::error::{ForecastError, Result};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a band-pass filter decomposition.
///
/// `cycle` contains the extracted cyclical component within the specified
/// period band.  `trend` holds everything outside that band (low-frequency
/// trend plus high-frequency noise).
#[derive(Debug, Clone)]
pub struct CycleDecomposition {
    /// Extracted cycle component (frequencies within the passband).
    pub cycle: Vec<f64>,
    /// Residual: original series minus cycle (trend + noise).
    pub trend: Vec<f64>,
    /// Lower bound of the passband in periods (e.g., 6 quarters).
    pub low_period: usize,
    /// Upper bound of the passband in periods (e.g., 32 quarters).
    pub high_period: usize,
}

// ---------------------------------------------------------------------------
// Ideal band-pass weights
// ---------------------------------------------------------------------------

/// Compute the ideal band-pass filter weight at lag `j`.
///
/// B_j = [sin(j*b) - sin(j*a)] / (pi * j)  for j != 0
/// B_0 = (b - a) / pi
///
/// where a = 2*pi / high_period,  b = 2*pi / low_period.
#[inline]
fn ideal_bp_weight(j: i64, a: f64, b: f64) -> f64 {
    if j == 0 {
        (b - a) / PI
    } else {
        let jf = j as f64;
        ((jf * b).sin() - (jf * a).sin()) / (PI * jf)
    }
}

// ---------------------------------------------------------------------------
// Christiano-Fitzgerald band-pass filter
// ---------------------------------------------------------------------------

/// Apply the Christiano-Fitzgerald (CF) asymmetric band-pass filter.
///
/// The CF filter computes time-varying, asymmetric weights for each
/// observation so that no data is lost at the endpoints. It approximates
/// the ideal infinite band-pass filter under the assumption that the
/// series follows a random walk (or, when `drift` is `true`, a random
/// walk with drift).
///
/// # Arguments
///
/// * `series` - Input time series.
/// * `low_period` - Minimum period of the passband (e.g., 6 for NBER).
/// * `high_period` - Maximum period of the passband (e.g., 32 for NBER).
/// * `drift` - If `true`, the filter assumes a random walk with drift;
///   otherwise it assumes a pure random walk.
///
/// # Errors
///
/// Returns `InvalidParameter` if `low_period >= high_period` or either is
/// less than 2.  Returns `InsufficientData` if the series is shorter than
/// `high_period`.
pub fn cf_filter(
    series: &[f64],
    low_period: usize,
    high_period: usize,
    drift: bool,
) -> Result<CycleDecomposition> {
    validate_params(series, low_period, high_period)?;

    let n = series.len();
    let a = 2.0 * PI / high_period as f64; // low frequency cut-off
    let b = 2.0 * PI / low_period as f64; // high frequency cut-off

    // Pre-compute ideal weights B_j for j = 0 .. n-1
    let b_full: Vec<f64> = (0..n as i64).map(|j| ideal_bp_weight(j, a, b)).collect();

    // Remove linear drift from the data if requested
    let y: Vec<f64> = if drift {
        let x0 = series[0];
        let xn = series[n - 1];
        let slope = (xn - x0) / (n - 1) as f64;
        series
            .iter()
            .enumerate()
            .map(|(i, &v)| v - x0 - slope * i as f64)
            .collect()
    } else {
        series.to_vec()
    };

    let mut cycle = vec![0.0; n];

    // For each time point t, construct the CF filter weights.
    // The CF approximation: for observation t with p observations before it
    // and q observations after it, the weights are the ideal weights for
    // lags -p .. q, with the constraint that they sum to zero.
    for t in 0..n {
        let p = t; // observations before t
        let q = n - 1 - t; // observations after t

        // Build the raw ideal weights for lags -p .. q
        // Index in the weight vector: lag + p (so index 0 = lag -p, index p = lag 0)
        let total = p + q + 1; // = n
        let mut w = vec![0.0; total];
        for k in 0..total {
            let lag = k as i64 - p as i64;
            w[k] = b_full[lag.unsigned_abs() as usize];
        }

        // CF constraint: adjust the endpoint weights so that the filter
        // weights sum to zero.  The adjustment is applied to the first and
        // last weights symmetrically.
        let sum: f64 = w.iter().sum();
        // Distribute the excess equally to the two endpoints
        if total >= 2 {
            w[0] -= sum / 2.0;
            w[total - 1] -= sum / 2.0;
        } else {
            w[0] -= sum;
        }

        // Apply weights: cycle[t] = sum_k w[k] * y[t - p + k]
        let start = t as i64 - p as i64; // = 0 always, but keep generic
        let mut val = 0.0;
        for k in 0..total {
            let idx = (start + k as i64) as usize;
            val += w[k] * y[idx];
        }
        cycle[t] = val;
    }

    // If we removed drift, add it back to the trend
    let trend: Vec<f64> = series
        .iter()
        .zip(cycle.iter())
        .map(|(&s, &c)| s - c)
        .collect();

    Ok(CycleDecomposition {
        cycle,
        trend,
        low_period,
        high_period,
    })
}

// ---------------------------------------------------------------------------
// Baxter-King band-pass filter
// ---------------------------------------------------------------------------

/// Apply the Baxter-King (BK) symmetric band-pass filter.
///
/// The BK filter uses a fixed symmetric kernel of `2k + 1` weights derived
/// from the ideal band-pass filter, adjusted so they sum to zero. Because
/// the kernel has length `2k + 1`, the first and last `k` observations
/// cannot be computed and are set to `f64::NAN`.
///
/// # Arguments
///
/// * `series` - Input time series.
/// * `low_period` - Minimum period of the passband.
/// * `high_period` - Maximum period of the passband.
/// * `k` - Lead/lag truncation length. Typical values: 12 for quarterly
///   data, 36 for monthly data.
///
/// # Errors
///
/// Returns `InvalidParameter` if `low_period >= high_period`, either is
/// less than 2, or `k` is zero.  Returns `InsufficientData` if the series
/// has fewer than `2k + 1` observations.
pub fn bk_filter(
    series: &[f64],
    low_period: usize,
    high_period: usize,
    k: usize,
) -> Result<CycleDecomposition> {
    validate_params(series, low_period, high_period)?;

    if k == 0 {
        return Err(ForecastError::InvalidParameter(
            "k (truncation lag) must be >= 1".to_string(),
        ));
    }

    let n = series.len();
    if n < 2 * k + 1 {
        return Err(ForecastError::InsufficientData {
            needed: 2 * k + 1,
            got: n,
            hint: Some(format!(
                "BK filter with k={k} requires at least {} observations",
                2 * k + 1
            )),
        });
    }

    let a = 2.0 * PI / high_period as f64;
    let b = 2.0 * PI / low_period as f64;

    // Ideal weights for lags 0 .. k
    let ideal: Vec<f64> = (0..=k as i64).map(|j| ideal_bp_weight(j, a, b)).collect();

    // Sum of all 2k+1 ideal weights (using symmetry)
    let ideal_sum = ideal[0] + 2.0 * ideal[1..].iter().sum::<f64>();

    // Adjustment: distribute the sum equally across all 2k+1 weights
    let adjustment = ideal_sum / (2 * k + 1) as f64;

    // Adjusted weights: a_j = B_j - adjustment
    let mut a_weights: Vec<f64> = ideal.iter().map(|&w| w - adjustment).collect();

    // The weights are symmetric: a_{-j} = a_j.  We store only j = 0..k.
    // Verify sum is ~0 (it will be by construction).
    debug_assert!({
        let s = a_weights[0] + 2.0 * a_weights[1..].iter().sum::<f64>();
        s.abs() < 1e-12
    });

    let _ = &mut a_weights; // suppress unused-mut if debug_assert is compiled out

    // Apply the symmetric filter
    let mut cycle = vec![f64::NAN; n];
    for t in k..(n - k) {
        let mut val = a_weights[0] * series[t];
        for j in 1..=k {
            val += a_weights[j] * (series[t - j] + series[t + j]);
        }
        cycle[t] = val;
    }

    let trend: Vec<f64> = series
        .iter()
        .zip(cycle.iter())
        .map(|(&s, &c)| if c.is_nan() { f64::NAN } else { s - c })
        .collect();

    Ok(CycleDecomposition {
        cycle,
        trend,
        low_period,
        high_period,
    })
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

fn validate_params(series: &[f64], low_period: usize, high_period: usize) -> Result<()> {
    if low_period < 2 {
        return Err(ForecastError::InvalidParameter(
            "low_period must be >= 2".to_string(),
        ));
    }
    if high_period < 2 {
        return Err(ForecastError::InvalidParameter(
            "high_period must be >= 2".to_string(),
        ));
    }
    if low_period >= high_period {
        return Err(ForecastError::InvalidParameter(format!(
            "low_period ({low_period}) must be less than high_period ({high_period})"
        )));
    }
    if series.is_empty() {
        return Err(ForecastError::EmptyData);
    }
    if series.len() < high_period {
        return Err(ForecastError::InsufficientData {
            needed: high_period,
            got: series.len(),
            hint: Some("series must be at least as long as high_period".to_string()),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave: A * sin(2*pi*t / period + phase).
    fn sine_wave(n: usize, period: f64, amplitude: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|t| amplitude * (2.0 * PI * t as f64 / period + phase).sin())
            .collect()
    }

    /// Mean absolute value of a slice.
    fn mean_abs(xs: &[f64]) -> f64 {
        let finite: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            return 0.0;
        }
        finite.iter().map(|x| x.abs()).sum::<f64>() / finite.len() as f64
    }

    /// RMS error between two slices (ignoring NaN positions).
    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let pairs: Vec<(f64, f64)> = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| (x, y))
            .collect();
        if pairs.is_empty() {
            return 0.0;
        }
        let mse: f64 = pairs.iter().map(|(x, y)| (x - y).powi(2)).sum::<f64>() / pairs.len() as f64;
        mse.sqrt()
    }

    // -----------------------------------------------------------------------
    // CF filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn cf_passband_sine_recovered() {
        // Sine with period 16, passband 6..32 => should be recovered
        let n = 200;
        let period = 16.0;
        let data = sine_wave(n, period, 5.0, 0.0);
        let result = cf_filter(&data, 6, 32, false).unwrap();

        assert_eq!(result.cycle.len(), n);
        // Skip some edge observations where filter ramp-up occurs
        let interior_rmse = rmse(&result.cycle[30..170], &data[30..170]);
        assert!(
            interior_rmse < 0.5,
            "CF should recover passband sine, RMSE={interior_rmse}"
        );
    }

    #[test]
    fn cf_stopband_sine_suppressed() {
        // Sine with period 3 (outside passband 6..32) => cycle should be near zero
        let n = 200;
        let data = sine_wave(n, 3.0, 5.0, 0.0);
        let result = cf_filter(&data, 6, 32, false).unwrap();

        let cycle_power = mean_abs(&result.cycle[20..180]);
        assert!(
            cycle_power < 0.5,
            "CF should suppress out-of-band sine, mean|cycle|={cycle_power}"
        );
    }

    #[test]
    fn cf_preserves_series_length() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = cf_filter(&data, 6, 32, true).unwrap();
        assert_eq!(result.cycle.len(), data.len());
        assert_eq!(result.trend.len(), data.len());
    }

    #[test]
    fn cf_trend_plus_cycle_equals_original() {
        let n = 150;
        let data: Vec<f64> = (0..n)
            .map(|i| 2.0 * i as f64 + 4.0 * (2.0 * PI * i as f64 / 20.0).sin())
            .collect();
        let result = cf_filter(&data, 6, 32, true).unwrap();
        for i in 0..n {
            let recon = result.trend[i] + result.cycle[i];
            assert!(
                (recon - data[i]).abs() < 1e-10,
                "trend + cycle should equal original at i={i}"
            );
        }
    }

    #[test]
    fn cf_drift_removes_linear_trend() {
        // Pure linear trend with no cycle in passband
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| 50.0 + 3.0 * i as f64).collect();
        let result = cf_filter(&data, 6, 32, true).unwrap();
        let cycle_power = mean_abs(&result.cycle);
        assert!(
            cycle_power < 1e-10,
            "CF with drift should produce zero cycle for pure linear trend, got {cycle_power}"
        );
    }

    // -----------------------------------------------------------------------
    // BK filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn bk_passband_sine_recovered() {
        let n = 200;
        let period = 16.0;
        let k = 12;
        let data = sine_wave(n, period, 5.0, 0.0);
        let result = bk_filter(&data, 6, 32, k).unwrap();

        // Interior (non-NaN) portion
        let interior_cycle: Vec<f64> = result.cycle[k..n - k].to_vec();
        let interior_data: Vec<f64> = data[k..n - k].to_vec();
        let err = rmse(&interior_cycle, &interior_data);
        assert!(err < 0.3, "BK should recover passband sine, RMSE={err}");
    }

    #[test]
    fn bk_stopband_sine_suppressed() {
        let n = 200;
        let k = 12;
        let data = sine_wave(n, 3.0, 5.0, 0.0);
        let result = bk_filter(&data, 6, 32, k).unwrap();

        let interior = &result.cycle[k..n - k];
        let power = mean_abs(interior);
        assert!(
            power < 0.3,
            "BK should suppress out-of-band sine, mean|cycle|={power}"
        );
    }

    #[test]
    fn bk_loses_2k_observations() {
        let n = 100;
        let k = 12;
        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let result = bk_filter(&data, 6, 32, k).unwrap();

        assert_eq!(result.cycle.len(), n);

        // First k and last k should be NaN
        for i in 0..k {
            assert!(result.cycle[i].is_nan(), "cycle[{i}] should be NaN");
            assert!(result.trend[i].is_nan(), "trend[{i}] should be NaN");
        }
        for i in (n - k)..n {
            assert!(result.cycle[i].is_nan(), "cycle[{i}] should be NaN");
            assert!(result.trend[i].is_nan(), "trend[{i}] should be NaN");
        }

        // Interior should be finite
        for i in k..(n - k) {
            assert!(result.cycle[i].is_finite(), "cycle[{i}] should be finite");
        }

        // Count of NaN values should be exactly 2k
        let nan_count = result.cycle.iter().filter(|x| x.is_nan()).count();
        assert_eq!(nan_count, 2 * k);
    }

    #[test]
    fn bk_weights_sum_to_zero() {
        // Verify indirectly: a constant series should yield zero cycle
        let n = 100;
        let k = 12;
        let data = vec![42.0; n];
        let result = bk_filter(&data, 6, 32, k).unwrap();

        for i in k..(n - k) {
            assert!(
                result.cycle[i].abs() < 1e-12,
                "BK cycle on constant series should be zero at i={i}, got {}",
                result.cycle[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Mixed signal tests
    // -----------------------------------------------------------------------

    #[test]
    fn cf_isolates_cycle_from_mixed_signal() {
        let n = 300;
        // trend + cycle (period 20) + noise (period 3)
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let cycle = 8.0 * (2.0 * PI * i as f64 / 20.0).sin();
                let noise = 2.0 * (2.0 * PI * i as f64 / 3.0).sin();
                trend + cycle + noise
            })
            .collect();

        let true_cycle: Vec<f64> = (0..n)
            .map(|i| 8.0 * (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        let result = cf_filter(&data, 6, 32, true).unwrap();

        // Check interior (away from endpoints)
        let err = rmse(&result.cycle[40..260], &true_cycle[40..260]);
        assert!(
            err < 1.5,
            "CF should isolate cycle from mixed signal, RMSE={err}"
        );
    }

    #[test]
    fn bk_isolates_cycle_from_mixed_signal() {
        let n = 300;
        let k = 12;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let cycle = 8.0 * (2.0 * PI * i as f64 / 20.0).sin();
                let noise = 2.0 * (2.0 * PI * i as f64 / 3.0).sin();
                trend + cycle + noise
            })
            .collect();

        let true_cycle: Vec<f64> = (0..n)
            .map(|i| 8.0 * (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        let result = bk_filter(&data, 6, 32, k).unwrap();

        let err = rmse(&result.cycle[k..n - k], &true_cycle[k..n - k]);
        assert!(
            err < 1.0,
            "BK should isolate cycle from mixed signal, RMSE={err}"
        );
    }

    // -----------------------------------------------------------------------
    // Edge-case / error tests
    // -----------------------------------------------------------------------

    #[test]
    fn error_series_shorter_than_high_period() {
        let data = vec![1.0; 10];
        let result = cf_filter(&data, 6, 32, false);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForecastError::InsufficientData { needed, got, .. } => {
                assert_eq!(needed, 32);
                assert_eq!(got, 10);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn error_low_period_ge_high_period() {
        let data = vec![1.0; 100];
        assert!(cf_filter(&data, 32, 32, false).is_err());
        assert!(cf_filter(&data, 40, 32, false).is_err());
        assert!(bk_filter(&data, 32, 32, 12).is_err());
        assert!(bk_filter(&data, 40, 32, 12).is_err());
    }

    #[test]
    fn error_empty_series() {
        let data: Vec<f64> = vec![];
        assert!(matches!(
            cf_filter(&data, 6, 32, false),
            Err(ForecastError::EmptyData)
        ));
        assert!(matches!(
            bk_filter(&data, 6, 32, 12),
            Err(ForecastError::EmptyData)
        ));
    }

    #[test]
    fn error_bk_k_zero() {
        let data = vec![1.0; 100];
        assert!(matches!(
            bk_filter(&data, 6, 32, 0),
            Err(ForecastError::InvalidParameter(_))
        ));
    }

    #[test]
    fn error_bk_insufficient_for_k() {
        // Series is long enough for high_period (32) but too short for 2*k+1=25
        // Use low_period=6, high_period=8, k=12 so high_period check passes
        // but 2*12+1=25 > 20 fails.
        let data = vec![1.0; 20];
        let result = bk_filter(&data, 6, 8, 12);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForecastError::InsufficientData { needed, got, .. } => {
                assert_eq!(needed, 25);
                assert_eq!(got, 20);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn error_period_less_than_2() {
        let data = vec![1.0; 100];
        assert!(cf_filter(&data, 1, 32, false).is_err());
        assert!(cf_filter(&data, 6, 1, false).is_err());
        assert!(bk_filter(&data, 1, 32, 12).is_err());
    }

    #[test]
    fn cf_result_fields() {
        let data: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 16.0).sin())
            .collect();
        let result = cf_filter(&data, 6, 32, false).unwrap();
        assert_eq!(result.low_period, 6);
        assert_eq!(result.high_period, 32);
    }

    #[test]
    fn bk_result_fields() {
        let data: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 16.0).sin())
            .collect();
        let result = bk_filter(&data, 6, 32, 12).unwrap();
        assert_eq!(result.low_period, 6);
        assert_eq!(result.high_period, 32);
    }

    #[test]
    fn bk_trend_plus_cycle_equals_original_interior() {
        let n = 200;
        let k = 12;
        let data: Vec<f64> = (0..n)
            .map(|i| 2.0 * i as f64 + 4.0 * (2.0 * PI * i as f64 / 20.0).sin())
            .collect();
        let result = bk_filter(&data, 6, 32, k).unwrap();
        for i in k..(n - k) {
            let recon = result.trend[i] + result.cycle[i];
            assert!(
                (recon - data[i]).abs() < 1e-10,
                "trend + cycle should equal original at i={i}"
            );
        }
    }
}
