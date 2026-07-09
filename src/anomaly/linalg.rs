//! Small dense linear algebra on flat row-major `k × k` matrices.
//!
//! `k` is small (≤ 32 in practice — the forecast horizon), so:
//! - No allocation for pivots or scratch that outlasts a single call.
//! - No dependency on `nalgebra` / `faer`.
//! - All operations either return `Result` or explicitly document
//!   the degenerate cases they handle by regularization (Cholesky
//!   jitter, `top_eig` zero-norm bail).
//!
//! Ports the algorithms from
//! `microprediction/timemachines/heads/mahalanobis.py`:
//! `_cholesky`, `_mahal2`, `_top_eig`, `_top_factors`, `_solve_sym`.

/// Lower Cholesky factor of a (near-)positive-definite `n × n`
/// symmetric matrix. Adds `jitter` to the diagonal if the pivot ever
/// drops below it — the caller decides how much slack is acceptable.
///
/// Returns the flat row-major lower-triangular factor `L` such that
/// `A ≈ L Lᵀ`. Upper-triangle entries of `L` are zero.
pub fn cholesky(a: &[f64], n: usize, jitter: f64) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for t in 0..j {
                s -= l[i * n + t] * l[j * n + t];
            }
            if i == j {
                l[i * n + i] = if s > jitter { s.sqrt() } else { jitter.sqrt() };
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    l
}

/// `‖L⁻¹ v‖²` by forward substitution: computes `vᵀ (L Lᵀ)⁻¹ v` for a
/// Cholesky-factored positive-definite matrix. This is the Mahalanobis
/// distance's squared form.
pub fn mahal2(l: &[f64], v: &[f64], n: usize) -> f64 {
    debug_assert_eq!(l.len(), n * n);
    debug_assert_eq!(v.len(), n);
    let mut w = vec![0.0f64; n];
    let mut d2 = 0.0;
    for i in 0..n {
        let mut s = v[i];
        for t in 0..i {
            s -= l[i * n + t] * w[t];
        }
        let wi = s / l[i * n + i];
        w[i] = wi;
        d2 += wi * wi;
    }
    d2
}

/// Leading eigenpair `(λ, v)` of a symmetric flat matrix by power
/// iteration. Deterministic start (uniform with a small index tilt to
/// break ties) so the whole detector stays reproducible.
///
/// Returns `(0.0, v0)` when the power iteration converges to a zero
/// vector — happens on a zero matrix, which the caller should handle.
pub fn top_eig(s: &[f64], n: usize, iters: usize) -> (f64, Vec<f64>) {
    debug_assert_eq!(s.len(), n * n);
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 1e-3 * i as f64).collect();
    let mut norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    let mut lam = 0.0;
    for _ in 0..iters {
        let w: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| s[i * n + j] * v[j]).sum::<f64>())
            .collect();
        norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return (0.0, v);
        }
        let w_normed: Vec<f64> = w.iter().map(|x| x / norm).collect();
        lam = (0..n)
            .map(|i| w_normed[i] * (0..n).map(|j| s[i * n + j] * w_normed[j]).sum::<f64>())
            .sum::<f64>();
        v = w_normed;
    }
    (lam.max(0.0), v)
}

/// Up to `r` leading eigenpairs by power iteration with deflation.
/// Stops early once an eigenvalue drops below 1 % of the mean diagonal
/// — factors below that carry no usable structure.
pub fn top_factors(s: &[f64], n: usize, r: usize) -> Vec<(f64, Vec<f64>)> {
    debug_assert_eq!(s.len(), n * n);
    let mut work = s.to_vec();
    let mean_diag = (0..n).map(|i| s[i * n + i]).sum::<f64>() / n as f64;
    let cutoff = 0.01 * mean_diag;
    let mut out = Vec::with_capacity(r);
    for _ in 0..r {
        let (lam, v) = top_eig(&work, n, 60);
        if lam <= cutoff {
            break;
        }
        // Deflate: work -= lam * v vᵀ
        for i in 0..n {
            for j in 0..n {
                work[i * n + j] -= lam * v[i] * v[j];
            }
        }
        out.push((lam, v));
    }
    out
}

/// Solve `A x = b` for a small symmetric positive-definite `A`
/// (flat row-major, size `n × n`).
///
/// Uses Cholesky with a small jitter, then forward + back substitution.
pub fn solve_sym(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    let l = cholesky(a, n, 1e-12);
    let mut y = vec![0.0f64; n];
    // Forward: L y = b
    for i in 0..n {
        let mut s = b[i];
        for t in 0..i {
            s -= l[i * n + t] * y[t];
        }
        y[i] = s / l[i * n + i];
    }
    // Back: Lᵀ x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for t in i + 1..n {
            s -= l[t * n + i] * x[t];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_of_identity_is_identity() {
        let n = 3;
        let a = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let l = cholesky(&a, n, 1e-12);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((l[i * n + j] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_reconstruction() {
        // Known SPD matrix
        let n = 3;
        let a = vec![
            4.0, 2.0, 1.0, //
            2.0, 5.0, 3.0, //
            1.0, 3.0, 6.0,
        ];
        let l = cholesky(&a, n, 1e-12);
        // Check L·Lᵀ ≈ A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for t in 0..n {
                    sum += l[i * n + t] * l[j * n + t];
                }
                assert!(
                    (sum - a[i * n + j]).abs() < 1e-10,
                    "L·Lᵀ[{i},{j}] = {sum}, A = {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn mahal2_of_standard_normal() {
        // For A = I, mahal² = v·v.
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0];
        let l = cholesky(&a, n, 1e-12);
        let d2 = mahal2(&l, &v, n);
        assert!((d2 - 14.0).abs() < 1e-10, "expected 14, got {d2}");
    }

    #[test]
    fn top_eig_identity_returns_unit_eigenvalue() {
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (lam, _) = top_eig(&a, n, 50);
        assert!((lam - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_eig_recovers_dominant_direction() {
        // Rank-1: A = 5·v vᵀ with v = (1,0,0)/‖·‖.
        let n = 3;
        let a = vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (lam, vec) = top_eig(&a, n, 100);
        assert!((lam - 5.0).abs() < 1e-6);
        // Eigenvector should be (±1, 0, 0)
        assert!(vec[0].abs() > 0.99);
    }

    #[test]
    fn top_factors_finds_all_significant_and_stops() {
        // A has two significant components + one negligible.
        let n = 3;
        // Diagonal (5, 3, 0.01) — top_factors should return 2 factors.
        let a = vec![
            5.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, //
            0.0, 0.0, 0.01,
        ];
        let facs = top_factors(&a, n, 3);
        // Mean diagonal = 2.67; cutoff = 0.0267. 0.01 < cutoff → drop.
        assert_eq!(facs.len(), 2);
        assert!((facs[0].0 - 5.0).abs() < 1e-3);
        assert!((facs[1].0 - 3.0).abs() < 1e-3);
    }

    #[test]
    fn solve_sym_recovers_x() {
        // A = [[4,1],[1,3]], b = [1,2] → x = [1/11, 7/11]
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = solve_sym(&a, &b, 2);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    }
}
