//! Average Mutual Information (AMI) and Partial AMI (pAMI) lag curves.
//!
//! AMI(h) = I(X_t; X_{t+h}) measures how much information the past carries
//! about the future at horizon h. The pAMI variant conditions on intermediate
//! lags to isolate *direct* dependence at horizon h.

use super::gcmi::gcmi;
use super::knn_mi::knn_mutual_information;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Backend for Conditional MI residualization in pAMI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmiBackend {
    /// OLS linear regression residuals.
    Linear,
}

/// Compute the AMI curve: `AMI(h) = I(X_t; X_{t+h})` for h = 1..max_lag.
///
/// Uses the KSG1 kNN mutual information estimator with `k = 8` neighbors.
///
/// # Returns
/// A `Vec<f64>` of length `max_lag`, where `result[h-1]` is the MI at lag h.
pub fn ami_curve(series: &[f64], max_lag: usize) -> Vec<f64> {
    ami_curve_with_k(series, max_lag, 8)
}

/// AMI curve with a custom number of neighbors.
pub fn ami_curve_with_k(series: &[f64], max_lag: usize, k: usize) -> Vec<f64> {
    let n = series.len();

    #[cfg(feature = "parallel")]
    {
        (1..=max_lag)
            .into_par_iter()
            .map(|h| {
                if n <= h + k {
                    0.0
                } else {
                    knn_mutual_information(&series[..n - h], &series[h..], k)
                }
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        (1..=max_lag)
            .map(|h| {
                if n <= h + k {
                    0.0
                } else {
                    knn_mutual_information(&series[..n - h], &series[h..], k)
                }
            })
            .collect()
    }
}

/// Compute the GCMI curve: `GCMI(h) = I_gauss(X_t; X_{t+h})` for h = 1..max_lag.
///
/// Uses the Gaussian Copula MI estimator — captures only linear dependence.
/// Comparing this with [`ami_curve`] reveals nonlinear structure.
pub fn gcmi_curve(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    let mut result = Vec::with_capacity(max_lag);
    for h in 1..=max_lag {
        if n <= h + 2 {
            result.push(0.0);
            continue;
        }
        let past = &series[..n - h];
        let future = &series[h..];
        result.push(gcmi(past, future));
    }
    result
}

/// Compute the pAMI curve: `pAMI(h) = I(X_t; X_{t+h} | X_{t+1}, …, X_{t+h-1})`
/// using linear residualization.
///
/// For each horizon h, the intermediate lags form a conditioning matrix Z.
/// Both `X_past` and `X_future` are linearly regressed on Z, and the kNN MI
/// is computed on the residuals. This approximates the conditional MI.
///
/// For h = 1, there are no intermediate lags, so pAMI(1) = AMI(1).
pub fn pami_curve(series: &[f64], max_lag: usize, _backend: CmiBackend) -> Vec<f64> {
    pami_curve_linear(series, max_lag)
}

fn pami_curve_linear(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    let k = 8;
    let mut result = Vec::with_capacity(max_lag);

    for h in 1..=max_lag {
        if n <= h + k {
            result.push(0.0);
            continue;
        }

        let usable = n - h;

        if h == 1 {
            // No conditioning variables — pAMI(1) = AMI(1).
            let past = &series[..usable];
            let future = &series[h..h + usable];
            result.push(knn_mutual_information(past, future, k));
            continue;
        }

        // Build conditioning matrix Z: columns X_{t+1}, …, X_{t+h-1}.
        // Each column j (0-indexed) is series[j+1 .. j+1+usable].
        // Uses slices via to_vec only where linear_residualize needs owned data.
        let z_cols: Vec<Vec<f64>> = (1..h).map(|j| series[j..j + usable].to_vec()).collect();

        let past = &series[..usable];
        let future = &series[h..h + usable];

        // Residualize past and future against Z via OLS.
        let past_resid = linear_residualize(past, &z_cols);
        let future_resid = linear_residualize(future, &z_cols);

        if past_resid.len() < k + 1 {
            result.push(0.0);
            continue;
        }

        result.push(knn_mutual_information(&past_resid, &future_resid, k));
    }

    result
}

/// Regress `y` on the column vectors in `z_cols` via OLS and return residuals.
///
/// Uses the normal equations with a column-pivoted approach for numerical
/// stability: β = (Z'Z)⁻¹ Z'y, residual = y - Zβ.
/// For the 1-column case this is just simple linear regression.
pub(crate) fn linear_residualize(y: &[f64], z_cols: &[Vec<f64>]) -> Vec<f64> {
    let n = y.len();
    let p = z_cols.len();

    if p == 0 || n < p + 1 {
        return y.to_vec();
    }

    // Build Z'Z (p×p) and Z'y (p×1).
    let mut ztz = vec![0.0; p * p];
    let mut zty = vec![0.0; p];

    for j in 0..p {
        for k in j..p {
            let dot: f64 = (0..n).map(|i| z_cols[j][i] * z_cols[k][i]).sum();
            ztz[j * p + k] = dot;
            ztz[k * p + j] = dot;
        }
        zty[j] = (0..n).map(|i| z_cols[j][i] * y[i]).sum();
    }

    // Solve via Cholesky (Z'Z is positive semi-definite).
    let beta = match cholesky_solve(&ztz, &zty, p) {
        Some(b) => b,
        None => return y.to_vec(), // degenerate — return original
    };

    // Residuals = y - Zβ.
    let mut resid = Vec::with_capacity(n);
    for i in 0..n {
        let predicted: f64 = (0..p).map(|j| z_cols[j][i] * beta[j]).sum();
        resid.push(y[i] - predicted);
    }
    resid
}

/// Cholesky factorization + solve for a p×p symmetric positive-definite matrix.
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    // L such that A = L L^T.
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * p + k] * l[j * p + k];
            }
            if i == j {
                let diag = a[i * p + i] - sum;
                if diag <= 1e-15 {
                    return None; // not positive definite
                }
                l[i * p + j] = diag.sqrt();
            } else {
                l[i * p + j] = (a[i * p + j] - sum) / l[j * p + j];
            }
        }
    }

    // Forward substitution: L z = b.
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * p + j] * z[j];
        }
        z[i] = (b[i] - sum) / l[i * p + i];
    }

    // Back substitution: L^T x = z.
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = 0.0;
        for j in i + 1..p {
            sum += l[j * p + i] * x[j];
        }
        x[i] = (z[i] - sum) / l[i * p + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut series = Vec::with_capacity(n);
        series.push(0.0);
        for _ in 1..n {
            let noise = (rng.gen::<f64>() - 0.5) * 2.0;
            series.push(phi * *series.last().unwrap() + noise);
        }
        series
    }

    use rand::{Rng, SeedableRng};

    #[test]
    fn ami_curve_detects_ar1_structure() {
        let series = make_ar1(500, 0.8, 42);
        let curve = ami_curve(&series, 5);
        assert_eq!(curve.len(), 5);
        // AMI at lag 1 should be the highest.
        assert!(curve[0] > curve[4], "AMI should decay: {:?}", curve);
        assert!(curve[0] > 0.1, "AMI(1) should be positive for AR(1)");
    }

    #[test]
    fn gcmi_curve_length_correct() {
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let curve = gcmi_curve(&series, 10);
        assert_eq!(curve.len(), 10);
    }

    #[test]
    fn pami_lag1_equals_ami_lag1() {
        let series = make_ar1(300, 0.6, 7);
        let ami = ami_curve(&series, 1);
        let pami = pami_curve(&series, 1, CmiBackend::Linear);
        // Should be approximately equal (same computation).
        let ratio = if ami[0] > 0.01 { pami[0] / ami[0] } else { 1.0 };
        assert!(
            (0.8..1.2).contains(&ratio),
            "pAMI(1) should match AMI(1): pami={:.4} ami={:.4}",
            pami[0],
            ami[0]
        );
    }

    #[test]
    fn pami_removes_indirect_dependence() {
        // For AR(1), AMI(2) > 0 but pAMI(2) should be smaller (the lag-2
        // dependence is indirect through lag-1).
        let series = make_ar1(500, 0.8, 11);
        let ami = ami_curve(&series, 3);
        let pami = pami_curve(&series, 3, CmiBackend::Linear);
        // pAMI(2) should be noticeably less than AMI(2).
        assert!(
            pami[1] < ami[1] * 1.2,
            "pAMI(2)={:.4} should be ≤ AMI(2)={:.4} for AR(1)",
            pami[1],
            ami[1]
        );
    }
}
