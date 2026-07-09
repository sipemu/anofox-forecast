//! Chi-square tail probability + quantile.
//!
//! Ports the Abramowitz & Stegun 6.5 series/continued-fraction split
//! from `microprediction/timemachines/heads/mahalanobis.py`. Fractional
//! degrees of freedom supported (needed for Satterthwaite matching).

const EPS: f64 = 3e-12;
const ITMAX: usize = 300;

/// Series expansion for the regularized lower incomplete gamma
/// `P(a, x) = γ(a, x) / Γ(a)` at `x < a + 1`.
fn gser(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut total = 1.0 / a;
    let mut term = total;
    for _ in 0..ITMAX {
        ap += 1.0;
        term *= x / ap;
        total += term;
        if term.abs() < total.abs() * EPS {
            break;
        }
    }
    total * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Continued-fraction expansion for the regularized upper incomplete
/// gamma `Q(a, x) = Γ(a, x) / Γ(a)` at `x >= a + 1`.
fn gcf(a: f64, x: f64) -> f64 {
    let fpmin = f64::MIN_POSITIVE * 1e12;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..ITMAX {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let de = d * c;
        h *= de;
        if (de - 1.0).abs() < EPS {
            break;
        }
    }
    h * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Survival function `P(X > x)` of a chi-square with `dof` degrees of
/// freedom (fractional dof allowed): `Q(dof/2, x/2)`.
pub fn chi2_sf(x: f64, dof: f64) -> f64 {
    debug_assert!(dof > 0.0);
    if x <= 0.0 {
        return 1.0;
    }
    let a = 0.5 * dof;
    let xx = 0.5 * x;
    if xx < a + 1.0 {
        1.0 - gser(a, xx)
    } else {
        gcf(a, xx)
    }
}

/// Quantile of the chi-square distribution via bisection on the survival
/// function. `p ∈ (0, 1)` is the LOWER-tail probability, so the returned
/// value satisfies `P(X ≤ result) = p`.
pub fn chi2_ppf(p: f64, dof: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    debug_assert!(dof > 0.0);
    let tol = 1e-10;
    let mut lo = 0.0;
    let mut hi = dof + 40.0 * (2.0 * dof).sqrt() + 100.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if 1.0 - chi2_sf(mid, dof) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < tol {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Log-Gamma function. Lanczos approximation with the same coefficients
/// as `libm::lgamma` (Cody & Hillstrom). Wraps the intrinsic when we
/// have it; falls back to a hand-rolled Lanczos for portability.
#[inline]
fn lgamma(x: f64) -> f64 {
    // libm's lgamma is a plain math function — no `errno` issue for our
    // inputs (a >= 0.25 always here).
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    {
        extern "C" {
            fn lgamma(x: f64) -> f64;
        }
        // SAFETY: pure math function, no side effects.
        unsafe { lgamma(x) }
    }
    #[cfg(not(any(target_family = "unix", target_family = "windows")))]
    {
        lgamma_lanczos(x)
    }
}

#[cfg(not(any(target_family = "unix", target_family = "windows")))]
fn lgamma_lanczos(x: f64) -> f64 {
    // Coefficients from Cody & Hillstrom (1967); accurate to ~14 digits
    // for x > 0.5. For x <= 0.5 use the reflection formula.
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma_lanczos(1.0 - x);
    }
    let x = x - 1.0;
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut sum = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.stats.chi2.sf(x, dof).
    #[test]
    fn sf_matches_scipy_reference() {
        // chi2.sf(3.84, 1) = 0.05004352... (the 5% critical value)
        assert!((chi2_sf(3.84, 1.0) - 0.05004352).abs() < 1e-6);
        // chi2.sf(5.991, 2) ≈ 0.05
        assert!((chi2_sf(5.991, 2.0) - 0.05).abs() < 1e-3);
        // chi2.sf(18.307, 10) ≈ 0.05
        assert!((chi2_sf(18.307, 10.0) - 0.05).abs() < 1e-3);
        // Fractional dof: matches Python's mahalanobis.chi2_sf(2.0, 1.5).
        // Line-by-line port; integer-dof cases above verify the algorithm.
        let sf_frac = chi2_sf(2.0, 1.5);
        assert!(sf_frac > 0.0 && sf_frac < 1.0);
        // Reference from the Python line-by-line port on same inputs:
        // (verified by executing the Python source directly)
        assert!(
            (sf_frac - 0.260020).abs() < 1e-4,
            "chi2_sf(2, 1.5) = {sf_frac}, expected ~0.260",
        );
    }

    #[test]
    fn ppf_inverse_of_sf() {
        for &dof in &[1.0, 2.0, 5.0, 10.0, 3.7] {
            for &p in &[0.5, 0.9, 0.95, 0.99, 0.999] {
                let x = chi2_ppf(p, dof);
                let recovered = 1.0 - chi2_sf(x, dof);
                assert!(
                    (recovered - p).abs() < 1e-6,
                    "dof={dof} p={p}: got {recovered}",
                );
            }
        }
    }

    #[test]
    fn sf_at_zero_is_one() {
        for &dof in &[1.0, 2.0, 10.0, 3.5] {
            assert_eq!(chi2_sf(0.0, dof), 1.0);
            assert_eq!(chi2_sf(-1.0, dof), 1.0);
        }
    }

    #[test]
    fn sf_at_infinity_is_zero() {
        assert!(chi2_sf(1e6, 5.0) < 1e-100);
    }
}
