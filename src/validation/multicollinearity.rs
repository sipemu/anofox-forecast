//! Multicollinearity diagnostics: VIF and condition number.
//!
//! When a regression design matrix has highly correlated columns (e.g.
//! `fourier(p, K)` plus `dummy_seasonal(p)`, or many overlapping rolling
//! windows), Ridge / ElasticNet coefficients become unstable and OLS solves
//! near-singular systems. These helpers surface the problem before fit.
//!
//! - [`variance_inflation_factors`] — VIF per column. `VIF[j] = 1 / (1 -
//!   R²_j)` where `R²_j` is the R² of regressing column `j` on the others.
//!   `VIF > 5` is a soft warning, `VIF > 10` is a hard signal that the
//!   column is essentially redundant.
//! - [`condition_number`] — `√(λ_max / λ_min)` of `X'X`. `cond > 30` is the
//!   classic Belsley-Kuh-Welsch threshold for collinearity concern.
//! - [`multicollinearity_report`] — both rolled into a `MulticollinearityReport`
//!   with named columns and threshold flags.
//!
//! Reference: Belsley, Kuh & Welsch (1980), *Regression Diagnostics:
//! Identifying Influential Data and Sources of Collinearity.*

use std::collections::HashMap;

use crate::error::{ForecastError, Result};
use crate::utils::ols::{ols_fit, ols_residuals};

/// Default soft VIF threshold (warn).
pub const VIF_WARN: f64 = 5.0;
/// Default hard VIF threshold (fail).
pub const VIF_FAIL: f64 = 10.0;
/// Default condition-number threshold (Belsley-Kuh-Welsch).
pub const COND_WARN: f64 = 30.0;

/// Per-column severity flag from a multicollinearity report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// VIF below the soft threshold.
    Ok,
    /// VIF between soft and hard thresholds — review.
    Warn,
    /// VIF above the hard threshold — column is essentially redundant.
    Fail,
}

/// Aggregated multicollinearity report for a design matrix.
#[derive(Debug, Clone)]
pub struct MulticollinearityReport {
    /// Per-column results: `(name, vif, severity)`.
    pub columns: Vec<(String, f64, Severity)>,
    /// Condition number `√(λ_max / λ_min)` of `X'X`. `f64::INFINITY` if
    /// the smallest eigenvalue is zero / numerically negligible.
    pub condition_number: f64,
    /// VIF threshold that triggered `Severity::Warn`.
    pub vif_warn: f64,
    /// VIF threshold that triggered `Severity::Fail`.
    pub vif_fail: f64,
    /// Condition-number threshold for collinearity concern.
    pub cond_warn: f64,
}

impl MulticollinearityReport {
    /// Names of columns whose VIF exceeds `vif_fail`.
    pub fn failing(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|(_, _, s)| *s == Severity::Fail)
            .map(|(name, _, _)| name.as_str())
            .collect()
    }

    /// Names of columns whose VIF is between `vif_warn` and `vif_fail`.
    pub fn warning(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|(_, _, s)| *s == Severity::Warn)
            .map(|(name, _, _)| name.as_str())
            .collect()
    }

    /// `true` when condition number exceeds the configured threshold.
    pub fn is_ill_conditioned(&self) -> bool {
        self.condition_number > self.cond_warn
    }
}

/// Compute the variance-inflation factor for each column of a design
/// matrix.
///
/// `columns` is a slice of equal-length `Vec<f64>`s — one per design-matrix
/// column. Returns one VIF per input column in input order. Constant
/// (zero-variance) columns produce `VIF = 1.0` (no inflation but no useful
/// signal either — caller should filter them separately).
///
/// Each VIF is computed by regressing the target column on the others and
/// taking `1 / (1 - R²)`. Numerical instability (e.g. perfectly duplicated
/// columns) yields `f64::INFINITY`.
pub fn variance_inflation_factors(columns: &[Vec<f64>]) -> Result<Vec<f64>> {
    let p = columns.len();
    if p == 0 {
        return Err(ForecastError::EmptyData);
    }
    let n = columns[0].len();
    if n < 2 {
        return Err(ForecastError::InsufficientData {
            needed: 2,
            got: n,
            hint: Some("VIF needs ≥ 2 observations".into()),
        });
    }
    for c in columns.iter() {
        if c.len() != n {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
    }
    if p == 1 {
        // No "others" to regress on → no inflation possible.
        return Ok(vec![1.0]);
    }

    let mut vifs = Vec::with_capacity(p);
    for j in 0..p {
        let y = &columns[j];

        // Constant column → VIF undefined but reported as 1 (no inflation).
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        if tss <= f64::EPSILON {
            vifs.push(1.0);
            continue;
        }

        let mut other_regressors: HashMap<String, Vec<f64>> = HashMap::new();
        for (k, c) in columns.iter().enumerate() {
            if k != j {
                other_regressors.insert(format!("c{}", k), c.clone());
            }
        }

        match ols_fit(y, &other_regressors) {
            Ok(fit) => match ols_residuals(y, &fit, &other_regressors) {
                Ok(res) => {
                    let rss: f64 = res.iter().map(|r| r * r).sum();
                    let r2 = 1.0 - rss / tss;
                    if r2 >= 1.0 - f64::EPSILON {
                        vifs.push(f64::INFINITY);
                    } else {
                        vifs.push(1.0 / (1.0 - r2));
                    }
                }
                Err(_) => vifs.push(f64::INFINITY),
            },
            Err(_) => vifs.push(f64::INFINITY),
        }
    }
    Ok(vifs)
}

/// Compute the condition number of the design matrix as
/// `√(λ_max(X'X) / λ_min(X'X))`.
///
/// Uses power iteration for `λ_max` and inverse iteration via the
/// Cholesky factor of `X'X + ε·I` for `λ_min`. Returns `f64::INFINITY`
/// when `X'X` is numerically singular.
pub fn condition_number(columns: &[Vec<f64>]) -> Result<f64> {
    let p = columns.len();
    if p == 0 {
        return Err(ForecastError::EmptyData);
    }
    let n = columns[0].len();
    if n < 2 {
        return Err(ForecastError::InsufficientData {
            needed: 2,
            got: n,
            hint: Some("condition_number needs ≥ 2 observations".into()),
        });
    }
    for c in columns.iter() {
        if c.len() != n {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
    }

    // Build X'X (p × p, symmetric PSD).
    let mut xtx = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in i..p {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += columns[i][k] * columns[j][k];
            }
            xtx[i][j] = s;
            xtx[j][i] = s;
        }
    }

    let lambda_max = power_iteration_max(&xtx, p);
    let lambda_min = power_iteration_min(&xtx, p);

    if !(lambda_max.is_finite() && lambda_min.is_finite()) || lambda_min <= 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok((lambda_max / lambda_min).sqrt())
}

/// Run the full multicollinearity report on a design matrix.
///
/// `names` must align with `columns` and is echoed in the report.
pub fn multicollinearity_report(
    columns: &[Vec<f64>],
    names: &[String],
) -> Result<MulticollinearityReport> {
    multicollinearity_report_with_thresholds(columns, names, VIF_WARN, VIF_FAIL, COND_WARN)
}

/// Same as [`multicollinearity_report`] with explicit thresholds.
pub fn multicollinearity_report_with_thresholds(
    columns: &[Vec<f64>],
    names: &[String],
    vif_warn: f64,
    vif_fail: f64,
    cond_warn: f64,
) -> Result<MulticollinearityReport> {
    if names.len() != columns.len() {
        return Err(ForecastError::DimensionMismatch {
            expected: columns.len(),
            got: names.len(),
        });
    }
    let vifs = variance_inflation_factors(columns)?;
    let cond = condition_number(columns)?;

    let cols = names
        .iter()
        .zip(vifs.iter())
        .map(|(name, &vif)| {
            let severity = if vif > vif_fail {
                Severity::Fail
            } else if vif > vif_warn {
                Severity::Warn
            } else {
                Severity::Ok
            };
            (name.clone(), vif, severity)
        })
        .collect();

    Ok(MulticollinearityReport {
        columns: cols,
        condition_number: cond,
        vif_warn,
        vif_fail,
        cond_warn,
    })
}

// ── Power iteration for symmetric PSD matrix ────────────────────────────

/// Largest eigenvalue of a symmetric matrix via power iteration.
fn power_iteration_max(a: &[Vec<f64>], p: usize) -> f64 {
    if p == 0 {
        return 0.0;
    }
    let mut v = vec![1.0_f64 / (p as f64).sqrt(); p];
    let mut lambda = 0.0_f64;
    for _ in 0..200 {
        let av = mat_vec(a, &v, p);
        let new_lambda = dot(&v, &av, p);
        let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        v = av.iter().map(|x| x / norm).collect();
        if (new_lambda - lambda).abs() < 1e-10 * new_lambda.abs().max(1.0) {
            return new_lambda;
        }
        lambda = new_lambda;
    }
    lambda
}

/// Smallest eigenvalue of a symmetric PSD matrix via shifted inverse
/// iteration. Uses Cholesky on `A + ε·I` to invert.
fn power_iteration_min(a: &[Vec<f64>], p: usize) -> f64 {
    if p == 0 {
        return 0.0;
    }
    // Shift slightly off zero so Cholesky succeeds even when A is near-singular.
    let mut shifted = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..p {
            shifted[i][j] = a[i][j];
        }
        shifted[i][i] += 1e-10;
    }
    let l = match cholesky(&shifted, p) {
        Some(l) => l,
        None => return 0.0,
    };

    let mut v = vec![1.0_f64 / (p as f64).sqrt(); p];
    let mut lambda = f64::INFINITY;
    for _ in 0..200 {
        // Solve A · w = v   ⇒   w = A^{-1} v   via Cholesky factor L L^T.
        let y = forward_sub(&l, &v, p);
        let w = back_sub(&l, &y, p);
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return 0.0;
        }
        let v_next: Vec<f64> = w.iter().map(|x| x / norm).collect();
        // Rayleigh quotient w.r.t. A on the new direction.
        let av = mat_vec(a, &v_next, p);
        let new_lambda = dot(&v_next, &av, p);
        if (new_lambda - lambda).abs() < 1e-10 * new_lambda.abs().max(1.0) {
            return new_lambda.max(0.0);
        }
        lambda = new_lambda;
        v = v_next;
    }
    lambda.max(0.0)
}

fn mat_vec(a: &[Vec<f64>], v: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = 0.0;
        for j in 0..p {
            s += a[i][j] * v[j];
        }
        out[i] = s;
    }
    out
}

fn dot(a: &[f64], b: &[f64], p: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..p {
        s += a[i] * b[i];
    }
    s
}

fn cholesky(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j] == 0.0 {
                    return None;
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward_sub(l: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        if l[i][i] == 0.0 {
            return vec![f64::INFINITY; n];
        }
        y[i] = sum / l[i][i];
    }
    y
}

fn back_sub(l: &[Vec<f64>], y: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        if l[i][i] == 0.0 {
            return vec![f64::INFINITY; n];
        }
        x[i] = sum / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn vif_orthogonal_columns_near_one() {
        // sin / cos at unrelated frequencies: theoretically uncorrelated → VIF ≈ 1.
        let n = 200;
        let c1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let c2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let c3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).sin()).collect();
        let vifs = variance_inflation_factors(&[c1, c2, c3]).unwrap();
        for v in vifs {
            assert!(
                v < 2.0,
                "orthogonal sinusoids should give VIF near 1, got {}",
                v
            );
        }
    }

    #[test]
    fn vif_perfect_collinearity_is_infinite() {
        // c2 = 2 * c1 → perfectly collinear, R² = 1 → VIF = ∞.
        let c1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let c2: Vec<f64> = c1.iter().map(|x| 2.0 * x).collect();
        let vifs = variance_inflation_factors(&[c1, c2]).unwrap();
        assert!(vifs[0].is_infinite() || vifs[0] > 1e6);
        assert!(vifs[1].is_infinite() || vifs[1] > 1e6);
    }

    #[test]
    fn vif_constant_column_is_one() {
        let c1: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let c2: Vec<f64> = vec![5.0; 30];
        let vifs = variance_inflation_factors(&[c1, c2]).unwrap();
        // Constant column gets VIF=1 by construction (zero-variance fallback).
        assert_relative_eq!(vifs[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn vif_single_column_is_one() {
        let c1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let vifs = variance_inflation_factors(&[c1]).unwrap();
        assert_eq!(vifs, vec![1.0]);
    }

    #[test]
    fn condition_number_orthogonal_low() {
        // Two orthonormal-ish columns → condition number near 1.
        let n = 100;
        let c1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let c2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let cond = condition_number(&[c1, c2]).unwrap();
        assert!(cond < 5.0, "expected near-1 condition, got {}", cond);
    }

    #[test]
    fn condition_number_collinear_huge() {
        // Almost-duplicated column → condition number very large.
        let c1: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let c2: Vec<f64> = c1.iter().map(|x| x + 1e-9).collect();
        let cond = condition_number(&[c1, c2]).unwrap();
        assert!(
            cond > 1e3,
            "expected huge condition number for near-duplicates, got {}",
            cond
        );
    }

    #[test]
    fn report_flags_failing_columns() {
        let c1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let c2: Vec<f64> = c1.iter().map(|x| 2.0 * x + 1e-10).collect();
        let c3: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let names = vec!["x1".into(), "x2".into(), "x3".into()];
        let report = multicollinearity_report(&[c1, c2, c3], &names).unwrap();
        let failing = report.failing();
        assert!(failing.contains(&"x1") || failing.contains(&"x2"));
        // Sinusoid should be safe.
        assert!(!failing.contains(&"x3"));
        assert!(report.is_ill_conditioned());
    }

    #[test]
    fn report_thresholds_respected() {
        let c1: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let c2: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let names = vec!["a".into(), "b".into()];
        let report = multicollinearity_report(&[c1, c2], &names).unwrap();
        for (_, vif, sev) in &report.columns {
            // With near-orthogonal columns, no severe collinearity.
            assert!(*vif < VIF_FAIL);
            assert!(*sev == Severity::Ok || *sev == Severity::Warn);
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let c1 = vec![1.0, 2.0, 3.0];
        let c2 = vec![1.0, 2.0];
        let err = variance_inflation_factors(&[c1, c2]).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }

    #[test]
    fn empty_columns_errors() {
        let err = variance_inflation_factors(&[]).unwrap_err();
        assert!(matches!(err, ForecastError::EmptyData));
    }

    #[test]
    fn names_length_mismatch_errors() {
        let c1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let c2: Vec<f64> = (0..20).map(|i| (i as f64).sqrt()).collect();
        let names = vec!["only_one".into()];
        let err = multicollinearity_report(&[c1, c2], &names).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }
}
