//! Differencing utilities for ARIMA models.

/// Apply differencing to a time series.
///
/// # Arguments
/// * `series` - The input series
/// * `d` - Differencing order (number of times to difference)
///
/// # Returns
/// The differenced series.
pub fn difference(series: &[f64], d: usize) -> Vec<f64> {
    if d == 0 || series.is_empty() {
        return series.to_vec();
    }

    let mut result = series.to_vec();
    for _ in 0..d {
        if result.len() <= 1 {
            break;
        }
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

/// Apply seasonal differencing to a time series.
///
/// # Arguments
/// * `series` - The input series
/// * `d` - Seasonal differencing order
/// * `period` - Seasonal period
///
/// # Returns
/// The seasonally differenced series.
pub fn seasonal_difference(series: &[f64], d: usize, period: usize) -> Vec<f64> {
    if d == 0 || period == 0 || series.len() <= period {
        return series.to_vec();
    }

    let mut result = series.to_vec();
    for _ in 0..d {
        if result.len() <= period {
            break;
        }
        result = result
            .iter()
            .skip(period)
            .zip(result.iter())
            .map(|(curr, prev)| curr - prev)
            .collect();
    }
    result
}

/// Integrate (reverse differencing) a differenced series.
///
/// # Arguments
/// * `differenced` - The differenced series
/// * `original` - The original series (needed for initial values)
/// * `d` - Differencing order used
///
/// # Returns
/// The integrated series.
pub fn integrate(differenced: &[f64], original: &[f64], d: usize) -> Vec<f64> {
    if d == 0 || differenced.is_empty() {
        return differenced.to_vec();
    }

    let mut result = differenced.to_vec();

    // We need to reverse the differencing d times
    for level in (0..d).rev() {
        // Get the initial value at this differencing level
        let init_value = if level == 0 {
            *original.last().unwrap_or(&0.0)
        } else {
            // For higher levels, we need the last value of the intermediate difference
            let intermediate = difference(original, level);
            *intermediate.last().unwrap_or(&0.0)
        };

        // Cumulative sum starting from the initial value
        let mut integrated = Vec::with_capacity(result.len());
        let mut cumsum = init_value;
        for &diff in &result {
            cumsum += diff;
            integrated.push(cumsum);
        }
        result = integrated;
    }

    result
}

/// Integrate (reverse seasonal differencing) a seasonally differenced series.
///
/// # Arguments
/// * `differenced` - The seasonally differenced forecast values
/// * `original` - The original series (needed for the last `period * d` values)
/// * `d` - Seasonal differencing order used
/// * `period` - Seasonal period used
///
/// # Returns
/// The integrated series on the original scale.
pub fn seasonal_integrate(
    differenced: &[f64],
    original: &[f64],
    d: usize,
    period: usize,
) -> Vec<f64> {
    if d == 0 || period == 0 || differenced.is_empty() {
        return differenced.to_vec();
    }

    let mut result = differenced.to_vec();

    for level in (0..d).rev() {
        // Get the reference series at this differencing level
        let reference = if level == 0 {
            original.to_vec()
        } else {
            seasonal_difference(original, level, period)
        };

        // Each forecast value: y[t] = diff[t] + y[t - period]
        // We need the last `period` values from the reference to seed integration
        let mut integrated = Vec::with_capacity(result.len());
        let ref_len = reference.len();

        for (i, &diff_val) in result.iter().enumerate() {
            let prev = if i < period {
                // Seed from the tail of the reference series
                let ref_idx = ref_len.wrapping_sub(period).wrapping_add(i);
                if ref_idx < ref_len {
                    reference[ref_idx]
                } else {
                    0.0
                }
            } else {
                integrated[i - period]
            };
            integrated.push(diff_val + prev);
        }
        result = integrated;
    }

    result
}

/// Apply fractional differencing to a time series.
///
/// Uses the binomial series expansion `(1-B)^d` where `B` is the backshift
/// operator. Weights are truncated when `|w_k| < threshold`.
///
/// Fractional differencing (`d ∈ (0, 1)`) removes just enough memory to achieve
/// stationarity while preserving predictive signal — unlike integer differencing
/// which removes all autocorrelation.
///
/// Reference: Lopez de Prado, *Advances in Financial Machine Learning* (2018),
/// Chapter 5: Fractionally Differentiated Features.
///
/// # Arguments
/// * `series` - The input series
/// * `d` - Fractional differencing order (typically 0 < d < 1)
/// * `threshold` - Weight truncation threshold (e.g., 1e-4)
///
/// # Returns
/// The fractionally differenced series (shorter by the number of truncated weights).
pub fn fractional_difference(series: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    if series.is_empty() || d == 0.0 {
        return series.to_vec();
    }

    // Compute weights using the recursive formula:
    // w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
    let weights = fractional_weights(d, series.len(), threshold);
    let k = weights.len();

    if k == 0 {
        return series.to_vec();
    }

    // Apply the filter: y_t^(d) = sum_{j=0}^{k-1} w_j * x_{t-j}
    let n = series.len();
    let mut result = Vec::with_capacity(n.saturating_sub(k - 1));

    for t in (k - 1)..n {
        let mut val = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            val += w * series[t - j];
        }
        result.push(val);
    }

    result
}

/// Compute fractional differencing weights.
///
/// Returns the weight vector `[w_0, w_1, ..., w_K]` where weights are
/// truncated when `|w_k| < threshold`.
///
/// # Arguments
/// * `d` - Fractional differencing order
/// * `max_len` - Maximum number of weights to compute
/// * `threshold` - Truncation threshold for weight magnitude
pub fn fractional_weights(d: f64, max_len: usize, threshold: f64) -> Vec<f64> {
    let mut weights = Vec::with_capacity(max_len.min(1000));
    let mut w = 1.0_f64;
    weights.push(w);

    for k in 1..max_len {
        w *= -(d - k as f64 + 1.0) / k as f64;
        if w.abs() < threshold {
            break;
        }
        weights.push(w);
    }

    weights
}

/// Find the minimum fractional differencing order `d` that makes the series
/// stationary (ADF test p-value < significance level).
///
/// Binary searches `d ∈ [0, 1]` to find the smallest `d` for which the
/// ADF test rejects the null hypothesis of a unit root.
///
/// # Arguments
/// * `series` - The input series
/// * `significance` - ADF p-value threshold (e.g., 0.05 for 5% level)
/// * `threshold` - Weight truncation threshold for fractional differencing
///
/// # Returns
/// `(d, p_value)` — the minimum `d` and the resulting ADF p-value.
/// Returns `d = 1.0` if no fractional order achieves stationarity.
pub fn find_min_fractional_d(series: &[f64], significance: f64, threshold: f64) -> (f64, f64) {
    use crate::validation::stationarity::adf_test;

    // Check if already stationary at d=0
    let adf_0 = adf_test(series, None);
    if adf_0.p_value < significance {
        return (0.0, adf_0.p_value);
    }

    // Binary search for minimum d
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best_d = 1.0;
    let mut best_p = 1.0;

    for _ in 0..20 {
        // 20 iterations gives precision ~1e-6
        let mid = (lo + hi) / 2.0;
        let diffed = fractional_difference(series, mid, threshold);

        if diffed.len() < 10 {
            // Not enough data after truncation — need lower d
            hi = mid;
            continue;
        }

        let adf = adf_test(&diffed, None);

        if adf.p_value < significance {
            best_d = mid;
            best_p = adf.p_value;
            hi = mid; // try smaller d
        } else {
            lo = mid; // need more differencing
        }
    }

    (best_d, best_p)
}

/// Suggest a non-seasonal differencing order using repeated KPSS testing.
///
/// Delegates to [`ndiffs_kpss`] with R `forecast::ndiffs`' defaults
/// (`alpha = 0.05`, `max_d = 2`) — kept as a convenience alias for callers
/// that do not need a custom alpha/max_d.
///
/// # Arguments
/// * `series` - The input series
///
/// # Returns
/// Suggested differencing order (0, 1, or 2).
pub fn suggest_differencing(series: &[f64]) -> usize {
    ndiffs_kpss(series, 0.05, 2)
}

/// STL seasonal-strength test shared by [`nsdiffs_seas`] and
/// `AutoARIMA::has_seasonal_pattern` — reproducing `forecast`'s
/// `seas.heuristic`: `1 - Var(remainder) / Var(seasonal + remainder)`.
/// `None` when there isn't enough data for STL (needs >= 3 full cycles).
pub(crate) fn seasonal_strength(values: &[f64], period: usize) -> Option<f64> {
    if period < 2 || values.len() < 3 * period {
        return None;
    }
    let stl = crate::seasonality::STL::new(period);
    stl.decompose(values).map(|r| r.seasonal_strength())
}

/// Determine the number of seasonal differences required, by R
/// `forecast::nsdiffs(x, test = "seas", max.D)`: constant or `period >=
/// length` series need none; otherwise seasonally difference while the
/// STL seasonal-strength test exceeds `0.64`, `D < max_cap_d`, and the
/// series is still long enough (`>= 2*period`) to retest.
pub fn nsdiffs_seas(series: &[f64], period: usize, max_cap_d: usize) -> usize {
    if period < 2 || series.len() <= period || is_constant_series(series) {
        return 0;
    }

    let mut cur = series.to_vec();
    let mut d = 0usize;
    let mut dodiff = seasonal_strength(&cur, period).is_some_and(|s| s > 0.64);

    while dodiff && d < max_cap_d {
        cur = seasonal_difference(&cur, 1, period);
        d += 1;
        dodiff =
            cur.len() >= 2 * period && seasonal_strength(&cur, period).is_some_and(|s| s > 0.64);
    }

    d
}

/// Determine the number of non-seasonal differences required for
/// stationarity, by repeated KPSS testing — reproducing R
/// `forecast::ndiffs(x, alpha, test = "kpss", type = "level", max.d)`.
///
/// At each step: if the current series is constant, stop (return the
/// current `d`); otherwise compute the KPSS "level" statistic with
/// `use.lag = trunc(3*sqrt(n)/13)` (`forecast::ndiffs`' own default lag,
/// independent of this crate's `validation::kpss_test`'s default lag),
/// map it to a p-value by linear interpolation against the four-point
/// `urca`/`tseries` critical-value table (clamped to `[0.01, 0.10]`), and
/// difference once more while `p < alpha` and `d < max_d`.
///
/// # Arguments
/// * `series` - The input series
/// * `alpha` - Significance level, clamped to `[0.01, 0.10]` like R
/// * `max_d` - Maximum differencing order to try
///
/// # Returns
/// The selected differencing order `d`.
pub fn ndiffs_kpss(series: &[f64], alpha: f64, max_d: usize) -> usize {
    let alpha = alpha.clamp(0.01, 0.10);
    let mut d = 0usize;
    let mut cur = series.to_vec();

    loop {
        if is_constant_series(&cur) {
            break;
        }
        let n = cur.len();
        if n < 4 {
            break;
        }
        let use_lag = ((3.0 * (n as f64).sqrt()) / 13.0).trunc() as usize;
        let stat = kpss_level_statistic(&cur, use_lag);
        let p = ndiffs_kpss_pvalue(stat);

        if p < alpha && d < max_d {
            cur = difference(&cur, 1);
            d += 1;
        } else {
            break;
        }
    }

    d
}

/// KPSS "level" (type = "mu") test statistic, matching
/// `urca::ur.kpss(x, type = "mu", use.lag = lag)`'s `@teststat`.
///
/// Demeans the series, forms the partial-sum (cumulative) process of the
/// residuals, and divides the normalized sum-of-squared partial sums by a
/// Bartlett-kernel long-run-variance estimate with weights
/// `1 - j/(lag+1)` for `j = 1..=lag` (`lag = 0` is allowed, in which case
/// the long-run variance reduces to the plain residual variance).
pub(crate) fn kpss_level_statistic(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }

    let mean = x.iter().sum::<f64>() / n as f64;
    let resid: Vec<f64> = x.iter().map(|&v| v - mean).collect();

    let mut cumsum = 0.0;
    let mut numerator = 0.0;
    for &r in &resid {
        cumsum += r;
        numerator += cumsum * cumsum;
    }
    numerator /= (n * n) as f64;

    let mut lrv = resid.iter().map(|r| r * r).sum::<f64>() / n as f64;
    for j in 1..=lag {
        let weight = 1.0 - j as f64 / (lag as f64 + 1.0);
        let gamma: f64 = resid
            .iter()
            .skip(j)
            .zip(resid.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / n as f64;
        lrv += 2.0 * weight * gamma;
    }

    if lrv <= 0.0 {
        return 0.0;
    }

    numerator / lrv
}

/// Map a KPSS "level" statistic to a p-value by linear interpolation
/// against the four-point `tseries`/`urca` critical-value table, matching
/// R `approx(cvals, pvals, xout = stat, rule = 2)` — flat extrapolation
/// beyond the table, so the result is always in `[0.01, 0.10]`.
fn ndiffs_kpss_pvalue(stat: f64) -> f64 {
    const CVALS: [f64; 4] = [0.347, 0.463, 0.574, 0.739];
    const PVALS: [f64; 4] = [0.10, 0.05, 0.025, 0.01];

    if stat <= CVALS[0] {
        return PVALS[0];
    }
    if stat >= CVALS[CVALS.len() - 1] {
        return PVALS[PVALS.len() - 1];
    }
    for i in 0..CVALS.len() - 1 {
        if stat >= CVALS[i] && stat <= CVALS[i + 1] {
            let frac = (stat - CVALS[i]) / (CVALS[i + 1] - CVALS[i]);
            return PVALS[i] + frac * (PVALS[i + 1] - PVALS[i]);
        }
    }
    PVALS[PVALS.len() - 1]
}

/// R's `is.constant()`-equivalent check used by `ndiffs`: true when every
/// value is (numerically) identical.
fn is_constant_series(x: &[f64]) -> bool {
    if x.len() < 2 {
        return true;
    }
    let first = x[0];
    x.iter()
        .all(|&v| (v - first).abs() < 1e-9 * first.abs().max(1.0))
}

/// Calculate variance of a series (test-only helper; production code now
/// delegates differencing-order selection to the KPSS-based `ndiffs_kpss`).
#[cfg(test)]
fn variance(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }
    crate::simd::variance_sample(series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn difference_order_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = difference(&series, 0);
        assert_eq!(result, series);
    }

    #[test]
    fn difference_order_1() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = difference(&series, 1);
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn difference_order_2() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = difference(&series, 2);
        // First diff: [2, 3, 4, 5]
        // Second diff: [1, 1, 1]
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn difference_constant_series() {
        let series = vec![5.0, 5.0, 5.0, 5.0];
        let result = difference(&series, 1);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn difference_empty() {
        let series: Vec<f64> = vec![];
        let result = difference(&series, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn seasonal_difference_basic() {
        // Quarterly data: Q1 values increase by 10 each year
        let series = vec![
            100.0, 120.0, 80.0, 90.0, // Year 1
            110.0, 130.0, 90.0, 100.0, // Year 2
        ];
        let result = seasonal_difference(&series, 1, 4);
        // Each value minus same quarter previous year
        assert_eq!(result, vec![10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn seasonal_difference_no_change() {
        let series = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let result = seasonal_difference(&series, 1, 3);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn seasonal_difference_order_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0];
        let result = seasonal_difference(&series, 0, 2);
        assert_eq!(result, series);
    }

    #[test]
    fn integrate_reverses_difference() {
        let original = vec![10.0, 12.0, 15.0, 19.0, 24.0];
        let _differenced = difference(&original, 1);
        let forecast_diff = vec![6.0, 7.0]; // Forecasted differences
        let integrated = integrate(&forecast_diff, &original, 1);

        // Should continue from last value: 24 + 6 = 30, 30 + 7 = 37
        assert_relative_eq!(integrated[0], 30.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 37.0, epsilon = 1e-10);
    }

    #[test]
    fn integrate_order_2() {
        let original = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let _differenced = difference(&original, 2);
        // differenced = [1, 1, 1]

        let forecast_diff2 = vec![1.0, 1.0]; // Continue the pattern
        let integrated = integrate(&forecast_diff2, &original, 2);

        // The integration should produce reasonable continuation
        assert!(integrated.len() == 2);
    }

    /// `kpss_level_statistic` must equal `urca::ur.kpss`'s `@teststat` to
    /// `1e-8` absolute for every one of the 100 non-seasonal R-reference
    /// fixture series (rw + ar1), using the fixture's own recorded
    /// `use.lag = trunc(3*sqrt(n)/13)`.
    #[test]
    fn kpss_statistic_matches_urca() {
        use serde_json::Value;

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join("r_reference")
            .join("auto_arima_r.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
        let fixture: Value =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("malformed fixture: {}", e));

        let as_f64_vec = |v: &Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };

        let mut checked = 0usize;
        for family in ["rw", "ar1"] {
            let entries = fixture[family].as_array().unwrap();
            for (i, entry) in entries.iter().enumerate() {
                let values = as_f64_vec(&entry["values"]);
                let use_lag = entry["kpss_use_lag"].as_u64().unwrap() as usize;
                let expected_stat = entry["kpss_stat"].as_f64().unwrap();

                let stat = kpss_level_statistic(&values, use_lag);
                assert!(
                    (stat - expected_stat).abs() < 1e-8,
                    "{family}[{i}]: kpss_level_statistic={stat} vs urca={expected_stat}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 100, "expected 100 non-seasonal fixture series");
    }

    #[test]
    fn suggest_differencing_stationary() {
        // White noise-like stationary series
        let series = vec![1.0, 0.5, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1];
        let d = suggest_differencing(&series);
        assert_eq!(d, 0);
    }

    #[test]
    fn suggest_differencing_trend() {
        // Clear upward trend
        let series: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        let d = suggest_differencing(&series);
        assert!(d >= 1);
    }

    #[test]
    fn suggest_differencing_quadratic() {
        // Quadratic trend
        let series: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
        let d = suggest_differencing(&series);
        assert!(d >= 1);
    }

    #[test]
    fn seasonal_integrate_reverses_seasonal_difference() {
        // Quarterly data with year-over-year growth of 10
        let original = vec![
            100.0, 120.0, 80.0, 90.0, // Year 1
            110.0, 130.0, 90.0, 100.0, // Year 2
        ];
        let diffed = seasonal_difference(&original, 1, 4);
        assert_eq!(diffed, vec![10.0, 10.0, 10.0, 10.0]);

        // Forecast: continue the same year-over-year growth
        let forecast_diff = vec![10.0, 10.0, 10.0, 10.0];
        let integrated = seasonal_integrate(&forecast_diff, &original, 1, 4);

        // Year 3 should be year 2 + 10 for each quarter
        assert_relative_eq!(integrated[0], 120.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 140.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[2], 100.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[3], 110.0, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_integrate_order_0_is_identity() {
        let forecast = vec![1.0, 2.0, 3.0];
        let original = vec![10.0, 20.0, 30.0, 40.0];
        let result = seasonal_integrate(&forecast, &original, 0, 4);
        assert_eq!(result, forecast);
    }

    #[test]
    fn seasonal_integrate_empty() {
        let result = seasonal_integrate(&[], &[1.0, 2.0], 1, 2);
        assert!(result.is_empty());
    }

    // ── Fractional differencing tests ──────────────────────────

    #[test]
    fn fractional_weights_d0_is_single_one() {
        let w = fractional_weights(0.0, 100, 1e-4);
        assert_eq!(w.len(), 1);
        assert_relative_eq!(w[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn fractional_weights_d1_matches_integer_diff() {
        // At d=1.0, weights should be [1, -1, 0, 0, ...] (truncated)
        let w = fractional_weights(1.0, 10, 1e-4);
        assert_eq!(w.len(), 2);
        assert_relative_eq!(w[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(w[1], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn fractional_weights_d05_decay() {
        let w = fractional_weights(0.5, 100, 1e-4);
        // Weights should decay toward zero
        assert!(w.len() > 2);
        assert_relative_eq!(w[0], 1.0, epsilon = 1e-10);
        assert!(w[1] < 0.0); // first weight is negative
        for i in 1..w.len() {
            assert!(w[i].abs() <= w[i - 1].abs() + 1e-10, "weights should decay");
        }
    }

    #[test]
    fn fractional_difference_d0_is_identity() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = fractional_difference(&series, 0.0, 1e-4);
        assert_eq!(result, series);
    }

    #[test]
    fn fractional_difference_d1_matches_standard() {
        let series = vec![10.0, 13.0, 17.0, 22.0, 28.0];
        let frac = fractional_difference(&series, 1.0, 1e-4);
        let standard = difference(&series, 1);
        // Should match standard differencing
        assert_eq!(frac.len(), standard.len());
        for (a, b) in frac.iter().zip(standard.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn fractional_difference_d05_shorter_than_input() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = fractional_difference(&series, 0.5, 1e-4);
        // Result is shorter by (weight_count - 1) elements
        assert!(result.len() < series.len());
        assert!(!result.is_empty());
    }

    #[test]
    fn fractional_difference_preserves_stationarity() {
        // A random walk (non-stationary) should become more stationary after frac diff
        let mut rw = vec![0.0; 200];
        for i in 1..200 {
            rw[i] = rw[i - 1] + ((i * 7 + 3) % 11) as f64 * 0.5 - 2.5;
        }
        let diffed = fractional_difference(&rw, 0.5, 1e-4);
        // The variance of the differenced series should be lower than the original
        let var_orig = variance(&rw);
        let var_diff = variance(&diffed);
        assert!(
            var_diff < var_orig,
            "Fractional diff should reduce variance: {} vs {}",
            var_diff,
            var_orig
        );
    }

    #[test]
    fn fractional_difference_empty() {
        let result = fractional_difference(&[], 0.5, 1e-4);
        assert!(result.is_empty());
    }

    #[test]
    fn find_min_d_stationary_series() {
        // Already stationary series should give d ≈ 0
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5)
            .collect();
        let (d, _p) = find_min_fractional_d(&series, 0.05, 1e-4);
        assert!(d < 0.1, "Stationary series should need d ≈ 0, got {}", d);
    }

    #[test]
    fn find_min_d_nonstationary() {
        // Strong upward trend — clearly non-stationary
        let rw: Vec<f64> = (0..300).map(|i| 100.0 + 2.0 * i as f64).collect();
        let (d, _p) = find_min_fractional_d(&rw, 0.05, 1e-4);
        // Should need some differencing (d > 0)
        assert!(d > 0.0, "Trend series should need d > 0, got {}", d);
        assert!(d <= 1.0, "d should be <= 1.0, got {}", d);
    }

    // ── Extended seasonal_integrate correctness tests ────────────

    #[test]
    fn seasonal_integrate_reverses_multi_period_forecast() {
        // Weekly data (period=7) over 3 weeks with growth
        let period = 7;
        let weekly_base = [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 8.0];
        let growth = 5.0;
        let original: Vec<f64> = (0..21)
            .map(|i| weekly_base[i % period] + growth * (i / period) as f64)
            .collect();

        let diffed = seasonal_difference(&original, 1, period);
        // Each diff should be == growth (5.0) for the last two weeks
        assert_eq!(diffed.len(), 14);
        for &d in &diffed {
            assert_relative_eq!(d, growth, epsilon = 1e-10);
        }

        // Forecast 14 steps (2 more weeks) of same growth
        let forecast_diff: Vec<f64> = vec![growth; 14];
        let integrated = seasonal_integrate(&forecast_diff, &original, 1, period);

        // First 7 values should be original week 3 + 5 (= week 4)
        for i in 0..7 {
            let expected = weekly_base[i] + growth * 3.0;
            assert_relative_eq!(integrated[i], expected, epsilon = 1e-10);
        }
        // Next 7 values should be week 5 = week 4 + 5
        for i in 0..7 {
            let expected = weekly_base[i] + growth * 4.0;
            assert_relative_eq!(integrated[7 + i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn seasonal_integrate_horizon_less_than_period() {
        // Period=7 but forecast only 3 steps — all seeds come from original
        let period = 7;
        let original = vec![
            10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 8.0, // Week 1
            15.0, 25.0, 20.0, 30.0, 35.0, 17.0, 13.0, // Week 2
        ];

        // Forecast differences for 3 steps
        let forecast_diff = vec![5.0, 5.0, 5.0];
        let integrated = seasonal_integrate(&forecast_diff, &original, 1, period);

        // When horizon < period, all seeds come from the tail of the original.
        // integrated[0] = diff[0] + original[14 - 7 + 0] = 5 + original[7] = 5 + 15 = 20
        // integrated[1] = diff[1] + original[14 - 7 + 1] = 5 + original[8] = 5 + 25 = 30
        // integrated[2] = diff[2] + original[14 - 7 + 2] = 5 + original[9] = 5 + 20 = 25
        assert_eq!(integrated.len(), 3);
        assert_relative_eq!(integrated[0], 20.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 30.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[2], 25.0, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_integrate_horizon_greater_than_period() {
        // Period=4, forecast 6 steps — last 2 use previously integrated values
        let period = 4;
        let original = vec![
            100.0, 200.0, 150.0, 250.0, // Q1
            110.0, 210.0, 160.0, 260.0, // Q2
        ];

        // Growth of 10 for each seasonal position
        let forecast_diff = vec![10.0; 6];
        let integrated = seasonal_integrate(&forecast_diff, &original, 1, period);

        assert_eq!(integrated.len(), 6);
        // First 4: seeded from tail of original (last 4 values = Q2)
        // integrated[0] = 10 + original[8-4+0] = 10 + 110 = 120
        // integrated[1] = 10 + original[8-4+1] = 10 + 210 = 220
        // integrated[2] = 10 + original[8-4+2] = 10 + 160 = 170
        // integrated[3] = 10 + original[8-4+3] = 10 + 260 = 270
        assert_relative_eq!(integrated[0], 120.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 220.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[2], 170.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[3], 270.0, epsilon = 1e-10);

        // Next 2: use previously integrated values
        // integrated[4] = 10 + integrated[4-4] = 10 + 120 = 130
        // integrated[5] = 10 + integrated[5-4] = 10 + 220 = 230
        assert_relative_eq!(integrated[4], 130.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[5], 230.0, epsilon = 1e-10);
    }

    #[test]
    fn seasonal_integrate_d2_reversal() {
        // Seasonal differencing with d=2, period=4
        let original = vec![
            100.0, 120.0, 80.0, 90.0, // Year 1
            110.0, 130.0, 90.0, 100.0, // Year 2
            125.0, 145.0, 105.0, 115.0, // Year 3
        ];

        let diffed = seasonal_difference(&original, 2, 4);
        // d=2 means: apply seasonal diff twice
        // First diff (d=1): [10, 10, 10, 10, 15, 15, 15, 15]
        // Second diff (d=1 on that): [5, 5, 5, 5]
        assert_eq!(diffed.len(), 4);
        for &d in &diffed {
            assert_relative_eq!(d, 5.0, epsilon = 1e-10);
        }

        // Forecast: same acceleration
        let forecast_diff = vec![5.0; 4];
        let integrated = seasonal_integrate(&forecast_diff, &original, 2, 4);

        // Year 4 should continue the accelerating pattern:
        // Year 3 growth was [15, 15, 15, 15], so year 4 growth should be [20, 20, 20, 20]
        assert_eq!(integrated.len(), 4);
        assert_relative_eq!(integrated[0], 145.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 165.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[2], 125.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[3], 135.0, epsilon = 1e-10);
    }

    #[test]
    fn integrate_reverses_difference_d1_multi_step() {
        // Verify integrate properly reverses d=1 for multi-step forecasts
        let original = vec![10.0, 13.0, 17.0, 22.0, 28.0];
        // Differences: [3, 4, 5, 6]
        // Forecast 4 more differences
        let forecast_diff = vec![7.0, 8.0, 9.0, 10.0];
        let integrated = integrate(&forecast_diff, &original, 1);

        // Should cumsum from last original value (28)
        assert_relative_eq!(integrated[0], 35.0, epsilon = 1e-10); // 28 + 7
        assert_relative_eq!(integrated[1], 43.0, epsilon = 1e-10); // 35 + 8
        assert_relative_eq!(integrated[2], 52.0, epsilon = 1e-10); // 43 + 9
        assert_relative_eq!(integrated[3], 62.0, epsilon = 1e-10); // 52 + 10
    }

    #[test]
    fn difference_then_integrate_roundtrip_d1() {
        // Verify that diff then integrate of the same data recovers it
        let original = vec![5.0, 8.0, 12.0, 17.0, 23.0, 30.0];
        let diffed = difference(&original, 1);
        // diffed = [3, 4, 5, 6, 7]

        // If we "forecast" the same differences, integration should reproduce the original
        let recovered = integrate(&diffed, &[original[0]], 1);
        // Starting from 5: 5+3=8, 8+4=12, 12+5=17, 17+6=23, 23+7=30
        for (i, &v) in recovered.iter().enumerate() {
            assert_relative_eq!(v, original[i + 1], epsilon = 1e-10);
        }
    }

    #[test]
    fn seasonal_difference_then_integrate_roundtrip() {
        // Weekly pattern: verify seasonal diff then integrate recovers scale
        let period = 7;
        let original: Vec<f64> = (0..28)
            .map(|i| {
                let base = [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 8.0];
                base[i % period] + 3.0 * (i / period) as f64
            })
            .collect();

        let diffed = seasonal_difference(&original, 1, period);
        // All diffs should be 3.0
        assert_eq!(diffed.len(), 21);
        for &d in &diffed {
            assert_relative_eq!(d, 3.0, epsilon = 1e-10);
        }

        // Forecast one more week of same growth
        let forecast_diff = vec![3.0; 7];
        let integrated = seasonal_integrate(&forecast_diff, &original, 1, period);

        // Week 5 = week 4 + 3 for each day
        let week4_base = [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 8.0];
        for i in 0..7 {
            let expected = week4_base[i] + 3.0 * 4.0; // 4th growth increment
            assert_relative_eq!(integrated[i], expected, epsilon = 1e-10);
        }
    }
}
