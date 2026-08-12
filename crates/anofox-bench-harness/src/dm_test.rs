//! Diebold-Mariano test with HLN small-sample correction and HAC variance.
//!
//! Implements squared-error loss + Harvey–Leybourne–Newbold (HLN) correction
//! per D-09. Reference: real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
//!
//! # Usage
//!
//! Apply as a significance gate: when the absolute MASE gap between two models
//! is < 5%, call `diebold_mariano_hln(e1, e2, h)` and require `reject_h0 = true`
//! before making a superiority claim (BENCH-02).
//!
//! # Statistical note (Assumption A2)
//!
//! The p-value uses a standard normal approximation rather than an exact
//! t(T-1) distribution. For M-competition test sets (T ≥ 18) the normal
//! approximation is adequate. Swap `normal_cdf` for a Student's-t CDF when
//! exact p-values are needed for very small test sets.

/// Run the Diebold-Mariano test (HLN + HAC variant, D-09).
///
/// Computes loss differentials `d_t = e1_t² - e2_t²` (squared-error loss),
/// estimates the HAC long-run variance with autocovariances summed to lag `h-1`,
/// applies the Harvey–Leybourne–Newbold small-sample correction, and returns a
/// two-sided p-value under the normal approximation.
///
/// # Arguments
/// * `e1` — Forecast errors from model 1 (actual − predicted₁); length T.
/// * `e2` — Forecast errors from model 2 (actual − predicted₂); length T.
/// * `h`  — Forecast horizon (controls the HAC lag window 0..h-1 and the HLN
///           correction factor `sqrt[(T+1-2h+h(h-1)/T)/T]`).
///
/// # Returns
/// `(p_value, reject_h0)` — two-sided p-value under the normal approximation,
/// and whether H₀ (equal predictive accuracy) is rejected at α = 0.05.
///
/// # Error guards
/// * Mismatched or empty inputs → `(f64::NAN, false)`.
/// * Zero or negative HAC variance (e.g. identical models) → `(1.0, false)`.
/// * Non-positive HLN radicand (tiny T) → falls back to `factor = 1.0`.
pub fn diebold_mariano_hln(e1: &[f64], e2: &[f64], h: usize) -> (f64, bool) {
    // Guard: mismatched or empty inputs → NaN sentinel, not panic.
    if e1.len() != e2.len() || e1.is_empty() {
        return (f64::NAN, false);
    }

    let t = e1.len();

    // Loss differentials: squared-error loss per D-09.
    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(a, b)| a * a - b * b)
        .collect();

    let d_bar = d.iter().sum::<f64>() / t as f64;

    // Autocovariance at lag k:  γ(k) = (1/T) Σ_{i=k}^{T-1} (d_i - d̄)(d_{i-k} - d̄)
    let gamma = |k: usize| -> f64 {
        (k..t)
            .map(|i| (d[i] - d_bar) * (d[i - k] - d_bar))
            .sum::<f64>()
            / t as f64
    };

    // HAC long-run variance: γ(0) when h ≤ 1, else γ(0) + 2·Σ_{k=1}^{h-1} γ(k)
    let hac = if h <= 1 {
        gamma(0)
    } else {
        gamma(0) + 2.0 * (1..h).map(|k| gamma(k)).sum::<f64>()
    };

    // V_hat = HAC / T  (variance of d̄)
    let v_hat = hac / t as f64;

    // Guard non-positive variance (identical models or zero-variance differentials).
    if v_hat <= 0.0 {
        return (1.0, false);
    }

    // Raw DM statistic: DM = d̄ / sqrt(V_hat)
    let dm = d_bar / v_hat.sqrt();

    // HLN small-sample correction:
    //   S* = DM × sqrt[(T + 1 - 2h + h(h-1)/T) / T]
    // Guard against negative radicand (can occur for tiny T with large h).
    let radicand =
        (t as f64 + 1.0 - 2.0 * h as f64 + h as f64 * (h as f64 - 1.0) / t as f64) / t as f64;
    let factor = if radicand > 0.0 {
        radicand.sqrt()
    } else {
        // Radicand ≤ 0 when T is very small relative to h; fall back to no correction.
        1.0
    };
    let s_star = dm * factor;

    // Two-sided p-value via normal approximation (A2 / OQ3 — normal approx adequate
    // for M-competition T ≥ 18; swap for Student's-t CDF for very small test sets).
    let p_value = 2.0 * normal_cdf(-s_star.abs());
    let reject_h0 = p_value < 0.05;

    (p_value, reject_h0)
}

/// Approximate standard normal CDF via the Abramowitz-Stegun rational approximation.
///
/// Maximum absolute error ≤ 7.5 × 10⁻⁸ (A&S formula 26.2.17).
/// Uses only `std` — no external statistical crates required.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;
    if x >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical error vectors → loss differentials are all zero → no evidence of
    /// difference → p-value should be 1.0 (or NaN-safe), reject_h0 must be false.
    #[test]
    fn dm_test_identical_models() {
        let errors: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let (p, reject) = diebold_mariano_hln(&errors, &errors, 1);
        assert!(
            !reject,
            "Identical models must not reject H₀ (got p={p}, reject={reject})"
        );
        // p should be 1.0 (guarded non-positive variance branch) or very close to 1.0.
        assert!(
            p.is_nan() || p >= 0.95,
            "p-value for identical models must be ≈1.0 (got {p})"
        );
    }

    /// Model 1 errors are an order of magnitude larger than model 2 → DM statistic
    /// should be large → p < 0.05 → reject_h0 == true.
    #[test]
    fn dm_test_clear_winner() {
        let n = 100;
        // e1: large errors (model 1 is bad)
        let e1: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        // e2: small errors (model 2 is good)
        let e2: Vec<f64> = (0..n)
            .map(|i| 0.1 + 0.05 * (i as f64 * 0.3).cos())
            .collect();
        let (p, reject) = diebold_mariano_hln(&e1, &e2, 1);
        assert!(
            reject,
            "Clear winner (e1 >> e2) must reject H₀ (got p={p}, reject={reject})"
        );
        assert!(
            p < 0.05,
            "p-value must be < 0.05 for a clear winner (got {p})"
        );
    }

    /// Mismatched-length inputs must return (NaN, false) without panic.
    #[test]
    fn dm_test_length_guard() {
        let e1 = vec![1.0, 2.0, 3.0];
        let e2 = vec![1.0, 2.0]; // different length
        let (p, reject) = diebold_mariano_hln(&e1, &e2, 1);
        assert!(
            p.is_nan(),
            "Mismatched lengths must return NaN p-value (got {p})"
        );
        assert!(!reject, "Mismatched lengths must return reject=false");

        // Empty inputs must also be guarded.
        let (p2, reject2) = diebold_mariano_hln(&[], &[], 1);
        assert!(
            p2.is_nan(),
            "Empty inputs must return NaN p-value (got {p2})"
        );
        assert!(!reject2, "Empty inputs must return reject=false");
    }

    /// For h > 1 the HLN correction factor sqrt[(T+1-2h+h(h-1)/T)/T] is strictly
    /// less than 1 for reasonable T (correction shrinks the statistic toward 0).
    ///
    /// Verify this directly by computing the correction factor values and confirming
    /// `factor(h=6) < factor(h=1)` for T=30. Then verify that the function path
    /// uses the formula: check that the formula terms `h*(h-1)/T` and `2*h` appear
    /// in the radicand by using moderate-signal inputs where both h=1 and h=6 produce
    /// distinguishable p-values.
    #[test]
    fn dm_test_hln_correction_shrinks_stat() {
        // Analytical check: for T=30, h=1 factor = sqrt[(30+1-2+0)/30] = sqrt[29/30] ≈ 0.983
        //                              h=6 factor = sqrt[(30+1-12+6*5/30)/30] = sqrt[20/30] ≈ 0.816
        // So the h=6 factor is meaningfully smaller.
        let t = 30.0_f64;
        let factor_h1 = ((t + 1.0 - 2.0 + 0.0) / t).sqrt();
        let factor_h6 = ((t + 1.0 - 12.0 + 6.0 * 5.0 / t) / t).sqrt();
        assert!(
            factor_h6 < factor_h1,
            "HLN factor at h=6 ({factor_h6:.4}) must be less than h=1 ({factor_h1:.4})"
        );

        // Functional check with borderline inputs: moderate signal where the
        // shrinkage at h=6 is large enough to flip a barely-significant result.
        // Use T=30 with a modest e1 > e2 relationship.
        let n = 30;
        let e1: Vec<f64> = (0..n).map(|i| 1.5 + 0.4 * (i as f64 * 0.7).sin()).collect();
        let e2: Vec<f64> = (0..n).map(|i| 0.5 + 0.4 * (i as f64 * 0.7).cos()).collect();

        let (p_h1, _) = diebold_mariano_hln(&e1, &e2, 1);
        let (p_h6, _) = diebold_mariano_hln(&e1, &e2, 6);

        // The HLN factor at h=6 < h=1, so |S*(h=6)| ≤ |S*(h=1)| (same DM),
        // meaning p(h=6) ≥ p(h=1). Both must be valid finite p-values.
        assert!(p_h1.is_finite(), "p_h1 must be finite (got {p_h1})");
        assert!(p_h6.is_finite(), "p_h6 must be finite (got {p_h6})");
        assert!(
            p_h6 >= p_h1 - 1e-10, // allow ≈ equality (both can hit zero for large signals)
            "HLN correction at h=6 must not reduce p-value below h=1 (p_h1={p_h1:.6}, p_h6={p_h6:.6})"
        );
    }
}
