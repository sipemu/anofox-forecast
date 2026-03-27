//! Hamilton (2018) regression-based filter for trend-cycle decomposition.
//!
//! Instead of the Hodrick-Prescott filter, Hamilton proposes a simple OLS
//! regression of y_t on its own lags at horizon h:
//!
//! ```text
//! y_t = beta_0 + beta_1 * y_{t-h} + beta_2 * y_{t-h-1} + ... + beta_p * y_{t-h-p+1} + e_t
//! ```
//!
//! - **Fitted values** = trend component
//! - **Residuals** = cycle + irregular component
//!
//! The filter has no endpoint problem (unlike HP) and is a well-defined
//! regression that can be tested and interpreted. The default parameters
//! for quarterly data are h = 8, p = 4 (a 2-year lookahead with 4 lags).
//!
//! # Reference
//!
//! Hamilton, J. D. (2018). "Why You Should Never Use the Hodrick-Prescott
//! Filter." *Review of Economics and Statistics*, 100(5), 831-843.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::hamilton::{hamilton_filter, hamilton_quarterly};
//!
//! let data: Vec<f64> = (0..200).map(|i| {
//!     let t = i as f64;
//!     0.5 * t + (t * 0.3).sin() * 2.0
//! }).collect();
//!
//! // Custom parameters
//! let decomp = hamilton_filter(&data, 8, 4).unwrap();
//! assert_eq!(decomp.offset, 11); // h + p - 1
//!
//! // Convenience: quarterly defaults (h=8, p=4)
//! let decomp_q = hamilton_quarterly(&data).unwrap();
//! ```

use crate::error::{ForecastError, Result};

/// Result of the Hamilton filter decomposition.
#[derive(Debug, Clone)]
pub struct HamiltonDecomposition {
    /// Residuals from the regression (cycle + irregular component).
    pub cycle: Vec<f64>,
    /// Fitted values from the regression (trend component).
    pub trend: Vec<f64>,
    /// Lookahead parameter (horizon).
    pub h: usize,
    /// Number of lags used in the regression.
    pub p: usize,
    /// R-squared of the regression.
    pub r_squared: f64,
    /// Number of observations lost from the start (h + p - 1).
    pub offset: usize,
}

/// Apply the Hamilton (2018) filter for trend-cycle decomposition.
///
/// Regresses `y_t` on `y_{t-h}, y_{t-h-1}, ..., y_{t-h-p+1}` using OLS
/// with an intercept. Fitted values form the trend and residuals form the
/// cycle component.
///
/// # Parameters
///
/// - `series`: the input time series.
/// - `h`: lookahead (horizon) parameter. For quarterly data, 8 (2 years).
/// - `p`: number of lagged regressors. Typically 4.
///
/// # Errors
///
/// - `InvalidParameter` if `h == 0` or `p == 0`.
/// - `InsufficientData` if `series.len() < h + p`.
pub fn hamilton_filter(series: &[f64], h: usize, p: usize) -> Result<HamiltonDecomposition> {
    if h == 0 {
        return Err(ForecastError::InvalidParameter(
            "h (horizon) must be >= 1".to_string(),
        ));
    }
    if p == 0 {
        return Err(ForecastError::InvalidParameter(
            "p (number of lags) must be >= 1".to_string(),
        ));
    }

    let n = series.len();
    let offset = h + p - 1;
    let min_obs = offset + 1; // need at least one usable observation
    if n < min_obs {
        return Err(ForecastError::InsufficientData {
            needed: min_obs,
            got: n,
            hint: Some(format!(
                "Hamilton filter with h={h}, p={p} requires at least h+p = {} observations",
                min_obs
            )),
        });
    }

    // Number of regression observations.
    let n_obs = n - offset;
    // Number of regressors including intercept.
    let k = p + 1;

    // Build X matrix (n_obs x k) in column-major order and y vector.
    // For observation index i (0-based), the dependent variable is series[offset + i]
    // and the regressors are:
    //   intercept = 1.0
    //   series[offset + i - h]
    //   series[offset + i - h - 1]
    //   ...
    //   series[offset + i - h - (p-1)]
    let mut x = vec![0.0; n_obs * k]; // column-major
    let mut y = vec![0.0; n_obs];

    for i in 0..n_obs {
        let t = offset + i;
        y[i] = series[t];
        // Column 0: intercept
        x[i] = 1.0;
        // Columns 1..=p: lags
        for j in 0..p {
            x[(j + 1) * n_obs + i] = series[t - h - j];
        }
    }

    // Solve OLS via Householder QR decomposition on X directly.
    // This avoids forming X'X (which squares the condition number) and
    // handles rank-deficient / near-collinear regressors gracefully.
    let beta = qr_least_squares(n_obs, k, &x, &y)?;

    // Compute fitted values and residuals.
    let mut trend = vec![0.0; n_obs];
    let mut cycle = vec![0.0; n_obs];

    for i in 0..n_obs {
        let mut fitted = 0.0;
        for col in 0..k {
            fitted += beta[col] * x[col * n_obs + i];
        }
        trend[i] = fitted;
        cycle[i] = y[i] - fitted;
    }

    // Compute R-squared.
    let y_mean = y.iter().sum::<f64>() / n_obs as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = cycle.iter().map(|&e| e.powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        // All y values are identical; trend is perfect, cycle is zero.
        1.0
    };

    Ok(HamiltonDecomposition {
        cycle,
        trend,
        h,
        p,
        r_squared,
        offset,
    })
}

/// Hamilton filter with default parameters for quarterly data (h = 8, p = 4).
pub fn hamilton_quarterly(series: &[f64]) -> Result<HamiltonDecomposition> {
    hamilton_filter(series, 8, 4)
}

/// Hamilton filter with default parameters for monthly data (h = 24, p = 4).
pub fn hamilton_monthly(series: &[f64]) -> Result<HamiltonDecomposition> {
    hamilton_filter(series, 24, 4)
}

/// Hamilton filter with default parameters for annual data (h = 2, p = 4).
pub fn hamilton_annual(series: &[f64]) -> Result<HamiltonDecomposition> {
    hamilton_filter(series, 2, 4)
}

/// Solve the least-squares problem min ||Xb - y||^2 via Householder QR with
/// column pivoting.
///
/// `x` is n x k in column-major order, `y` is length n.
/// Returns the coefficient vector b of length k.
///
/// Column pivoting ensures numerical stability even when columns of X are
/// nearly collinear (e.g., a pure linear trend produces rank-2 design matrix
/// with k=5 columns).
fn qr_least_squares(n: usize, k: usize, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    // Work on a mutable copy of X (column-major) and y.
    let mut q = x.to_vec(); // n x k column-major
    let mut rhs = y.to_vec(); // length n

    // Column norms (for pivoting).
    let mut col_norms: Vec<f64> = (0..k)
        .map(|j| {
            let mut s = 0.0;
            for i in 0..n {
                s += q[j * n + i] * q[j * n + i];
            }
            s
        })
        .collect();

    // Pivot permutation.
    let mut perm: Vec<usize> = (0..k).collect();

    // Determine effective rank via threshold.
    let max_col_norm = col_norms.iter().cloned().fold(0.0_f64, f64::max).sqrt();
    let tol = 1e-12 * (n as f64).sqrt() * max_col_norm;
    let mut rank = k;

    // Householder QR with column pivoting.
    for step in 0..k {
        // Find column with largest remaining norm.
        let mut best_col = step;
        let mut best_norm = col_norms[step];
        for j in (step + 1)..k {
            if col_norms[j] > best_norm {
                best_norm = col_norms[j];
                best_col = j;
            }
        }

        // Swap columns if needed.
        if best_col != step {
            for i in 0..n {
                q.swap(step * n + i, best_col * n + i);
            }
            col_norms.swap(step, best_col);
            perm.swap(step, best_col);
        }

        // Compute Householder vector for column `step`, rows step..n.
        let mut norm_sq = 0.0;
        for i in step..n {
            norm_sq += q[step * n + i] * q[step * n + i];
        }
        let norm_val = norm_sq.sqrt();

        if norm_val < tol {
            // Remaining columns are numerically zero; stop here.
            rank = step;
            break;
        }

        let alpha = if q[step * n + step] >= 0.0 {
            -norm_val
        } else {
            norm_val
        };

        // v = x[step:] with v[0] -= alpha; store in-place in q column.
        q[step * n + step] -= alpha;
        let v_norm_sq = {
            let mut s = 0.0;
            for i in step..n {
                s += q[step * n + i] * q[step * n + i];
            }
            s
        };

        if v_norm_sq < 1e-30 {
            rank = step;
            break;
        }

        let tau = 2.0 / v_norm_sq;

        // Apply Householder reflector to remaining columns of Q.
        for j in (step + 1)..k {
            let mut dot = 0.0;
            for i in step..n {
                dot += q[step * n + i] * q[j * n + i];
            }
            let factor = tau * dot;
            for i in step..n {
                q[j * n + i] -= factor * q[step * n + i];
            }
        }

        // Apply Householder reflector to rhs.
        {
            let mut dot = 0.0;
            for i in step..n {
                dot += q[step * n + i] * rhs[i];
            }
            let factor = tau * dot;
            for i in step..n {
                rhs[i] -= factor * q[step * n + i];
            }
        }

        // Store R diagonal.
        q[step * n + step] = alpha;

        // Update column norms for remaining columns.
        for j in (step + 1)..k {
            col_norms[j] -= q[j * n + step] * q[j * n + step];
            if col_norms[j] < 0.0 {
                col_norms[j] = 0.0;
            }
        }
    }

    if rank == 0 {
        return Err(ForecastError::ComputationError(
            "Hamilton filter: design matrix has no non-zero columns".to_string(),
        ));
    }

    // Back-substitution on the upper-triangular R (rank x rank).
    // R is stored in the upper-left of q: R[i][j] = q[j * n + i] for i <= j < rank.
    let mut beta_piv = vec![0.0; k];
    for i in (0..rank).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..rank {
            sum -= q[j * n + i] * beta_piv[j];
        }
        beta_piv[i] = sum / q[i * n + i];
    }
    // Columns beyond rank get zero coefficients (already set).

    // Un-pivot.
    let mut beta = vec![0.0; k];
    for j in 0..k {
        beta[perm[j]] = beta_piv[j];
    }

    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_linear_trend_cycle_near_zero() {
        // y_t = 2 + 0.5*t  (pure linear trend, no cycle)
        // The design matrix is rank-deficient (intercept + lags span only 2D),
        // but QR with column pivoting handles this gracefully.
        let n = 200;
        let series: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();

        let decomp = hamilton_filter(&series, 8, 4).unwrap();

        // Cycle (residuals) should be nearly zero for a pure linear trend.
        let max_cycle = decomp.cycle.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_cycle < 1e-6,
            "max cycle = {max_cycle}, expected near zero for linear trend"
        );

        // Trend (fitted) should match the original series at the valid indices.
        for (i, &t) in decomp.trend.iter().enumerate() {
            let expected = series[decomp.offset + i];
            assert!(
                (t - expected).abs() < 1e-6,
                "trend[{i}] = {t}, expected {expected}"
            );
        }
    }

    #[test]
    fn trend_plus_irregular_cycle() {
        // y_t = 0.3*t + cycle_t where cycle_t is deterministic but not
        // representable as a linear combination of a few lags.
        // We use a sum of many incommensurate sinusoids to create a pseudo-
        // irregular component that the regression cannot fully capture.
        let n = 400;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                let trend = 0.3 * t;
                // Sum of sinusoids with irrational-ish frequency ratios
                let cycle = 3.0 * (t * 0.7).sin()
                    + 2.0 * (t * 1.1).cos()
                    + 1.5 * (t * 2.3).sin()
                    + 1.0 * (t * 3.7).cos()
                    + 0.8 * (t * 5.1).sin();
                trend + cycle
            })
            .collect();

        let decomp = hamilton_filter(&series, 8, 4).unwrap();

        // The cycle component should have meaningful variation.
        let cycle_var: f64 = {
            let mean = decomp.cycle.iter().sum::<f64>() / decomp.cycle.len() as f64;
            decomp.cycle.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / decomp.cycle.len() as f64
        };
        assert!(
            cycle_var > 0.5,
            "cycle variance = {cycle_var}, expected > 0.5 for irregular component"
        );

        // R-squared should be less than 1 (regression cannot fully explain the data).
        assert!(
            decomp.r_squared < 1.0,
            "R-squared = {}, expected < 1.0",
            decomp.r_squared
        );
    }

    #[test]
    fn r_squared_high_for_trend_dominated() {
        // Strong trend with tiny noise.
        let n = 200;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                10.0 + 2.0 * t + 0.001 * (t * 1.7).sin()
            })
            .collect();

        let decomp = hamilton_filter(&series, 8, 4).unwrap();
        assert!(
            decomp.r_squared > 0.999,
            "R-squared = {}, expected > 0.999",
            decomp.r_squared
        );
    }

    #[test]
    fn offset_equals_h_plus_p_minus_1() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let d1 = hamilton_filter(&series, 8, 4).unwrap();
        assert_eq!(d1.offset, 8 + 4 - 1);
        assert_eq!(d1.offset, 11);

        let d2 = hamilton_filter(&series, 24, 4).unwrap();
        assert_eq!(d2.offset, 24 + 4 - 1);
        assert_eq!(d2.offset, 27);

        let d3 = hamilton_filter(&series, 2, 4).unwrap();
        assert_eq!(d3.offset, 2 + 4 - 1);
        assert_eq!(d3.offset, 5);
    }

    #[test]
    fn h_zero_errors() {
        let series = vec![1.0; 50];
        let err = hamilton_filter(&series, 0, 4).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }

    #[test]
    fn p_zero_errors() {
        let series = vec![1.0; 50];
        let err = hamilton_filter(&series, 8, 0).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }

    #[test]
    fn series_too_short_errors() {
        // h=8, p=4 => need at least 12 observations
        let series = vec![1.0; 11];
        let err = hamilton_filter(&series, 8, 4).unwrap_err();
        assert!(matches!(
            err,
            ForecastError::InsufficientData {
                needed: 12,
                got: 11,
                ..
            }
        ));
    }

    #[test]
    fn output_lengths_correct() {
        let n = 150;
        let series: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

        let decomp = hamilton_filter(&series, 8, 4).unwrap();
        let expected_len = n - decomp.offset;
        assert_eq!(decomp.trend.len(), expected_len);
        assert_eq!(decomp.cycle.len(), expected_len);
    }

    #[test]
    fn convenience_quarterly() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let decomp = hamilton_quarterly(&series).unwrap();
        assert_eq!(decomp.h, 8);
        assert_eq!(decomp.p, 4);
    }

    #[test]
    fn convenience_monthly() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let decomp = hamilton_monthly(&series).unwrap();
        assert_eq!(decomp.h, 24);
        assert_eq!(decomp.p, 4);
    }

    #[test]
    fn convenience_annual() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let decomp = hamilton_annual(&series).unwrap();
        assert_eq!(decomp.h, 2);
        assert_eq!(decomp.p, 4);
    }

    #[test]
    fn trend_plus_cycle_equals_original() {
        // trend[i] + cycle[i] should reconstruct series[offset + i]
        let n = 200;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                3.0 + 0.7 * t + 4.0 * (t * 0.2).sin()
            })
            .collect();

        let decomp = hamilton_filter(&series, 8, 4).unwrap();
        for i in 0..decomp.trend.len() {
            let reconstructed = decomp.trend[i] + decomp.cycle[i];
            let original = series[decomp.offset + i];
            assert!(
                (reconstructed - original).abs() < 1e-10,
                "reconstruction mismatch at {i}: {reconstructed} vs {original}"
            );
        }
    }
}
