//! Cross-series (panel) feature aggregations.
//!
//! Given a panel of `N` time series each of length `T` (e.g. SKUs in a
//! retail dataset, sensors in an IoT fleet), these helpers compute
//! per-timestamp aggregates across the cross-section. The resulting
//! `N × T` matrix is suitable for use as an additional regressor on each
//! per-series forecaster (e.g. via `RegressionFeatures::with_exog_lags`
//! or just by appending to the calendar regressors map).
//!
//! # Leakage
//!
//! The `exclude_self` flag controls whether each series sees its own
//! contemporaneous value in its aggregate column:
//!
//! - `exclude_self = false` — every series receives the same
//!   `Vec<f64>` (the cross-section aggregate including itself).
//!   Cheap, but technically uses self at time `t` to predict self at
//!   `t+h` if used naively as exog.
//! - `exclude_self = true` — each series receives a leave-one-out
//!   aggregate excluding its own value at each timestamp. Leakage-safe.
//!
//! [`PanelAggregator::Rank`] always operates over the full cross-section
//! (the rank of each series among its peers); `exclude_self` is ignored.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::features::panel::{panel_aggregate, PanelAggregator};
//!
//! // Three series of length 4
//! let panel = vec![
//!     vec![1.0, 2.0, 3.0, 4.0],
//!     vec![10.0, 20.0, 30.0, 40.0],
//!     vec![100.0, 200.0, 300.0, 400.0],
//! ];
//!
//! // Cross-sectional mean at each timestamp (same for all series)
//! let means = panel_aggregate(&panel, PanelAggregator::Mean, false).unwrap();
//! assert_eq!(means[0], vec![37.0, 74.0, 111.0, 148.0]);
//! ```

use crate::error::{ForecastError, Result};

/// Cross-section aggregation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelAggregator {
    /// Mean across series at each timestamp.
    Mean,
    /// Median across series at each timestamp.
    Median,
    /// Sample standard deviation (Bessel's correction) across series.
    Std,
    /// Fractional rank of each series within the cross-section in `[0, 1]`.
    /// 0 = smallest, 1 = largest. Ties get the average rank.
    Rank,
}

/// Compute a per-timestamp cross-series aggregate for a panel.
///
/// # Arguments
/// - `values` — panel of `N` series, each of length `T`. All series must
///   have identical length.
/// - `kind` — aggregator to compute.
/// - `exclude_self` — if `true`, each series receives a leave-one-out
///   aggregate (excluding its own value at each timestamp). Ignored for
///   [`PanelAggregator::Rank`].
///
/// # Returns
/// `N × T` matrix where row `i` is the aggregate column for series `i`.
/// When `exclude_self = false` (and kind != Rank), all rows are identical.
///
/// # Errors
/// - [`ForecastError::EmptyData`] when the panel is empty.
/// - [`ForecastError::DimensionMismatch`] when series lengths differ.
/// - [`ForecastError::InsufficientData`] when `exclude_self = true` and
///   `N < 2`, or when `kind = Std` (with or without exclude_self) and
///   the contributing population has fewer than 2 elements.
pub fn panel_aggregate(
    values: &[Vec<f64>],
    kind: PanelAggregator,
    exclude_self: bool,
) -> Result<Vec<Vec<f64>>> {
    let n = values.len();
    if n == 0 {
        return Err(ForecastError::EmptyData);
    }
    let t = values[0].len();
    for v in values.iter() {
        if v.len() != t {
            return Err(ForecastError::DimensionMismatch {
                expected: t,
                got: v.len(),
            });
        }
    }
    if exclude_self && n < 2 {
        return Err(ForecastError::InsufficientData {
            needed: 2,
            got: n,
            hint: Some("exclude_self requires ≥ 2 series".into()),
        });
    }

    let result = match kind {
        PanelAggregator::Rank => rank_per_series(values, t),
        _ => {
            if exclude_self {
                loo_aggregate(values, kind, n, t)?
            } else {
                let common = full_aggregate(values, kind, n, t)?;
                vec![common; n]
            }
        }
    };
    Ok(result)
}

/// Convenience: cross-sectional mean per timestamp.
pub fn panel_mean(values: &[Vec<f64>], exclude_self: bool) -> Result<Vec<Vec<f64>>> {
    panel_aggregate(values, PanelAggregator::Mean, exclude_self)
}

/// Convenience: cross-sectional median per timestamp.
pub fn panel_median(values: &[Vec<f64>], exclude_self: bool) -> Result<Vec<Vec<f64>>> {
    panel_aggregate(values, PanelAggregator::Median, exclude_self)
}

/// Convenience: cross-sectional sample std per timestamp.
pub fn panel_std(values: &[Vec<f64>], exclude_self: bool) -> Result<Vec<Vec<f64>>> {
    panel_aggregate(values, PanelAggregator::Std, exclude_self)
}

/// Convenience: fractional rank in `[0, 1]` of each series within the
/// cross-section at each timestamp.
pub fn panel_rank(values: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    panel_aggregate(values, PanelAggregator::Rank, false)
}

fn full_aggregate(
    values: &[Vec<f64>],
    kind: PanelAggregator,
    n: usize,
    t: usize,
) -> Result<Vec<f64>> {
    if matches!(kind, PanelAggregator::Std) && n < 2 {
        return Err(ForecastError::InsufficientData {
            needed: 2,
            got: n,
            hint: Some("Std needs ≥ 2 series for sample variance".into()),
        });
    }
    let mut out = vec![0.0; t];
    let mut col = vec![0.0_f64; n];
    for j in 0..t {
        for (i, series) in values.iter().enumerate() {
            col[i] = series[j];
        }
        out[j] = compute_kind(&col, kind);
    }
    Ok(out)
}

fn loo_aggregate(
    values: &[Vec<f64>],
    kind: PanelAggregator,
    n: usize,
    t: usize,
) -> Result<Vec<Vec<f64>>> {
    if matches!(kind, PanelAggregator::Std) && n < 3 {
        return Err(ForecastError::InsufficientData {
            needed: 3,
            got: n,
            hint: Some("Std with exclude_self needs ≥ 3 series".into()),
        });
    }
    let mut out = vec![vec![0.0_f64; t]; n];
    let mut buf = vec![0.0_f64; n - 1];
    for j in 0..t {
        for i in 0..n {
            // Fill buffer with all values except series i.
            let mut k = 0;
            for (q, series) in values.iter().enumerate() {
                if q != i {
                    buf[k] = series[j];
                    k += 1;
                }
            }
            out[i][j] = compute_kind(&buf, kind);
        }
    }
    Ok(out)
}

fn rank_per_series(values: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
    let n = values.len();
    let mut out = vec![vec![0.0_f64; t]; n];
    if n <= 1 {
        // Single series → fractional rank is 0.5 by convention.
        if n == 1 {
            out[0].iter_mut().for_each(|x| *x = 0.5);
        }
        return out;
    }
    let denom = (n - 1) as f64;
    for j in 0..t {
        for i in 0..n {
            let xi = values[i][j];
            let mut less = 0usize;
            let mut equal = 0usize;
            for series in values.iter() {
                let xj = series[j];
                if xj < xi {
                    less += 1;
                } else if xj == xi {
                    equal += 1;
                }
            }
            // Average rank when ties are present.
            out[i][j] = ((less as f64) + 0.5 * (equal as f64 - 1.0)) / denom;
        }
    }
    out
}

fn compute_kind(xs: &[f64], kind: PanelAggregator) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    match kind {
        PanelAggregator::Mean => xs.iter().sum::<f64>() / n as f64,
        PanelAggregator::Median => {
            let mut v = xs.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            }
        }
        PanelAggregator::Std => {
            if n < 2 {
                0.0
            } else {
                let m = xs.iter().sum::<f64>() / n as f64;
                let var: f64 = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
                var.sqrt()
            }
        }
        PanelAggregator::Rank => unreachable!("Rank handled by rank_per_series"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_panel() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![100.0, 200.0, 300.0, 400.0],
        ]
    }

    #[test]
    fn mean_full_panel_is_identical_per_series() {
        let panel = make_panel();
        let out = panel_mean(&panel, false).unwrap();
        // (1+10+100)/3=37, etc.
        let expected = vec![37.0, 74.0, 111.0, 148.0];
        for row in &out {
            assert_eq!(row, &expected);
        }
    }

    #[test]
    fn mean_exclude_self_loo() {
        let panel = make_panel();
        let out = panel_mean(&panel, true).unwrap();
        // Series 0 LOO at t=0: (10+100)/2 = 55
        assert_relative_eq!(out[0][0], 55.0, epsilon = 1e-12);
        // Series 1 LOO at t=0: (1+100)/2 = 50.5
        assert_relative_eq!(out[1][0], 50.5, epsilon = 1e-12);
        // Series 2 LOO at t=0: (1+10)/2 = 5.5
        assert_relative_eq!(out[2][0], 5.5, epsilon = 1e-12);
    }

    #[test]
    fn median_full_panel() {
        let panel = make_panel();
        let out = panel_median(&panel, false).unwrap();
        // Median of [1, 10, 100] = 10
        assert_relative_eq!(out[0][0], 10.0, epsilon = 1e-12);
        assert_relative_eq!(out[0][1], 20.0, epsilon = 1e-12);
    }

    #[test]
    fn std_requires_at_least_two_series() {
        let single = vec![vec![1.0, 2.0]];
        let err = panel_std(&single, false).unwrap_err();
        assert!(matches!(err, ForecastError::InsufficientData { .. }));
    }

    #[test]
    fn std_full_panel() {
        let panel = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let out = panel_std(&panel, false).unwrap();
        // Sample std of [1, 3] with ddof=1 = sqrt(2) ≈ 1.4142
        assert_relative_eq!(out[0][0], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn rank_recovers_ordering() {
        let panel = make_panel();
        let ranks = panel_rank(&panel).unwrap();
        // At t=0: panel values = [1, 10, 100] — ranks 0, 0.5, 1
        assert_relative_eq!(ranks[0][0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(ranks[1][0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(ranks[2][0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rank_handles_ties() {
        // Three series with two tied at t=0.
        let panel = vec![vec![5.0], vec![5.0], vec![10.0]];
        let ranks = panel_rank(&panel).unwrap();
        // Two tied 5s share fractional ranks: less=0, equal=2 → (0 + 0.5*1)/2 = 0.25 each
        assert_relative_eq!(ranks[0][0], 0.25, epsilon = 1e-12);
        assert_relative_eq!(ranks[1][0], 0.25, epsilon = 1e-12);
        // 10: less=2, equal=1 → (2 + 0.5*0)/2 = 1.0
        assert_relative_eq!(ranks[2][0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let panel = vec![vec![1.0, 2.0], vec![3.0]];
        let err = panel_mean(&panel, false).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }

    #[test]
    fn rejects_empty_panel() {
        let panel: Vec<Vec<f64>> = Vec::new();
        let err = panel_mean(&panel, false).unwrap_err();
        assert!(matches!(err, ForecastError::EmptyData));
    }

    #[test]
    fn loo_requires_at_least_two_series() {
        let single = vec![vec![1.0, 2.0]];
        let err = panel_mean(&single, true).unwrap_err();
        assert!(matches!(err, ForecastError::InsufficientData { .. }));
    }

    #[test]
    fn loo_std_requires_at_least_three_series() {
        let two = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let err = panel_std(&two, true).unwrap_err();
        assert!(matches!(err, ForecastError::InsufficientData { .. }));
    }
}
