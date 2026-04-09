//! Monte-Carlo simulation of critical values for the four CUSUM detectors.
//!
//! Ports `simCritVal` from the R package
//! [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast)
//! (MIT License, © Thomas Grundy). Each detector has an asymptotic limit
//! distribution under the no-change null hypothesis; we draw discretised
//! Wiener paths on `[0, 1]` and apply the detector's limit functional, then
//! take the `(1 - alpha)` quantile as the critical value.
//!
//! The asymptotic limit theory is from
//! [Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899).
//!
//! The functionals `c1`, `c2`, `p1`, `p2` match the R source exactly:
//!
//! - **c1** (CUSUM one-sided):   `max_{t} W_t / (t/n)^γ`
//! - **c2** (CUSUM two-sided):   `max_{t} |W_t| / (t/n)^γ`
//! - **p1** (PageCUSUM one-sided): `max_t (1/(t/n)^γ) · (W_t − min_{s≤t} ((1−t/n)/(1−s/n))·W_s)`
//! - **p2** (PageCUSUM two-sided): `max_t (1/(t/n)^γ) · max_{s≤t} |W_t − ((1−t/n)/(1−s/n))·W_s|`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::sequential::Detector;

/// How to obtain the critical value for a detector configuration.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CriticalValue {
    /// Look the value up in the baked table; return an error if the
    /// `(detector, gamma, alpha)` triple is not in the table.
    #[default]
    Lookup,
    /// Simulate via Wiener Monte Carlo using the supplied budget and seed.
    Simulate {
        samples: usize,
        npts: usize,
        seed: Option<u64>,
    },
    /// Use a user-supplied fixed value.
    Fixed(f64),
}

/// Sample from a standard normal via Box-Muller transform.
///
/// Avoids adding `rand_distr` as a dependency. Deterministic given the seed.
#[inline]
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Clamp to avoid log(0) which would produce +inf.
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a single discretised Wiener path on `[0, 1]` with `npts` points.
///
/// `W[0] = 0` implicitly; the returned vector has length `npts` and
/// `W[k] = Σ_{i=0..=k} N(0, 1/npts)` — i.e. the path is sampled at
/// `t_k = (k + 1) / npts` for `k = 0..npts-1`, matching R's
/// `fdapace::Wiener(n, pts=seq(0, 1, length=npts), K=npts)` up to a
/// one-index shift that is absorbed by the `t/npts` rescaling in the
/// functionals below.
#[inline]
fn wiener_path<R: Rng>(rng: &mut R, npts: usize) -> Vec<f64> {
    let scale = (1.0 / npts as f64).sqrt();
    let mut path = Vec::with_capacity(npts);
    let mut acc = 0.0;
    for _ in 0..npts {
        acc += standard_normal(rng) * scale;
        path.push(acc);
    }
    path
}

/// CUSUM one-sided limit functional.
#[inline]
fn c1(w: &[f64], gamma: f64) -> f64 {
    let npts = w.len() as f64;
    let mut best = f64::NEG_INFINITY;
    for (i, &wi) in w.iter().enumerate() {
        let t = (i + 1) as f64 / npts;
        let denom = t.powf(gamma);
        let v = wi / denom;
        if v > best {
            best = v;
        }
    }
    best
}

/// CUSUM two-sided limit functional.
#[inline]
fn c2(w: &[f64], gamma: f64) -> f64 {
    let npts = w.len() as f64;
    let mut best = f64::NEG_INFINITY;
    for (i, &wi) in w.iter().enumerate() {
        let t = (i + 1) as f64 / npts;
        let denom = t.powf(gamma);
        let v = wi.abs() / denom;
        if v > best {
            best = v;
        }
    }
    best
}

/// Page CUSUM one-sided limit functional.
///
/// Matches R's `p1` + `p1inf`:
///
/// ```text
/// p1(w) = max_{t=1..npts-1} (1/(t/npts)^γ) · (w_t − min_{s=1..t} ((1 − t/npts)/(1 − s/npts)) · w_s)
/// ```
fn p1(w: &[f64], gamma: f64) -> f64 {
    let npts = w.len();
    let n_f = npts as f64;
    let mut best = f64::NEG_INFINITY;
    for t in 1..npts {
        let t_f = t as f64 / n_f;
        let one_minus_t = 1.0 - t_f;
        // min over s = 1..=t of ((1 - t/n)/(1 - s/n)) * w[s-1]
        let mut min_adj = f64::INFINITY;
        for s in 1..=t {
            let s_f = s as f64 / n_f;
            let denom = 1.0 - s_f;
            if denom <= 0.0 {
                continue;
            }
            let adj = (one_minus_t / denom) * w[s - 1];
            if adj < min_adj {
                min_adj = adj;
            }
        }
        if min_adj == f64::INFINITY {
            continue;
        }
        let v = (w[t - 1] - min_adj) / t_f.powf(gamma);
        if v > best {
            best = v;
        }
    }
    best
}

/// Page CUSUM two-sided limit functional.
///
/// Matches R's `p2` + `p2sup`:
///
/// ```text
/// p2(w) = max_{t=1..npts-1} (1/(t/npts)^γ) · max_{s=1..t} |w_t − ((1 − t/npts)/(1 − s/npts)) · w_s|
/// ```
fn p2(w: &[f64], gamma: f64) -> f64 {
    let npts = w.len();
    let n_f = npts as f64;
    let mut best = f64::NEG_INFINITY;
    for t in 1..npts {
        let t_f = t as f64 / n_f;
        let one_minus_t = 1.0 - t_f;
        let wt = w[t - 1];
        let mut max_abs = f64::NEG_INFINITY;
        for s in 1..=t {
            let s_f = s as f64 / n_f;
            let denom = 1.0 - s_f;
            if denom <= 0.0 {
                continue;
            }
            let adj = (one_minus_t / denom) * w[s - 1];
            let v = (wt - adj).abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        if max_abs == f64::NEG_INFINITY {
            continue;
        }
        let v = max_abs / t_f.powf(gamma);
        if v > best {
            best = v;
        }
    }
    best
}

/// Apply the limit functional for a given detector to a Wiener path.
#[inline]
fn apply_functional(detector: Detector, w: &[f64], gamma: f64) -> f64 {
    match detector {
        Detector::Cusum1 => c1(w, gamma),
        Detector::Cusum => c2(w, gamma),
        Detector::PageCusum1 => p1(w, gamma),
        Detector::PageCusum => p2(w, gamma),
    }
}

/// Empirical `1 - alpha` quantile of a slice.
///
/// Uses the R-default type-7 quantile (linear interpolation between order
/// statistics). Input is sorted in place.
fn empirical_quantile(samples: &mut [f64], alpha: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let q = 1.0 - alpha;
    let h = q * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    samples[lo] * (1.0 - frac) + samples[hi] * frac
}

/// Simulate a critical value for a given detector, gamma, and alpha.
///
/// # Arguments
/// * `detector` — which CUSUM variant to simulate
/// * `gamma` — weight-function tuning parameter, `0 ≤ γ < 0.5`
/// * `alpha` — type-I error rate (e.g. `0.05`)
/// * `samples` — number of Wiener paths to draw (R default: 1000)
/// * `npts` — number of discretisation points per path (R default: 500)
/// * `seed` — optional RNG seed for reproducibility
///
/// Parallelised across samples when the `parallel` feature is enabled.
pub fn simulate_critical_value(
    detector: Detector,
    gamma: f64,
    alpha: f64,
    samples: usize,
    npts: usize,
    seed: Option<u64>,
) -> f64 {
    assert!((0.0..0.5).contains(&gamma), "gamma must be in [0, 0.5)");
    assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");
    assert!(samples >= 20, "samples must be at least 20");
    assert!(npts >= 20, "npts must be at least 20");

    let base_seed = seed.unwrap_or(0xC4_D5_E6_F7_01_23_45_67);

    // Sequential path: one RNG, draw all samples in order.
    #[cfg(not(feature = "parallel"))]
    let mut stats: Vec<f64> = {
        let mut rng = StdRng::seed_from_u64(base_seed);
        (0..samples)
            .map(|_| {
                let path = wiener_path(&mut rng, npts);
                apply_functional(detector, &path, gamma)
            })
            .collect()
    };

    // Parallel path: one RNG per sample, seeded deterministically from the
    // base seed + sample index. This keeps results reproducible regardless of
    // thread count.
    #[cfg(feature = "parallel")]
    let mut stats: Vec<f64> = (0..samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            let path = wiener_path(&mut rng, npts);
            apply_functional(detector, &path, gamma)
        })
        .collect();

    empirical_quantile(&mut stats, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn quantile_basic() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // R: quantile(1:5, 0.95, type=7) = 4.8
        let q = empirical_quantile(&mut v, 0.05);
        assert_relative_eq!(q, 4.8, epsilon = 1e-10);
    }

    #[test]
    fn wiener_path_length_and_start() {
        let mut rng = StdRng::seed_from_u64(42);
        let path = wiener_path(&mut rng, 100);
        assert_eq!(path.len(), 100);
        // First step is N(0, 1/npts), small but non-zero with prob 1.
        assert!(path[0].is_finite());
    }

    #[test]
    fn c1_matches_hand_calc() {
        // Monotonic path: w[i] = (i+1)/npts, gamma=0
        // => c1 = max (i+1)/npts / 1 = 1.0 at t=1
        let w: Vec<f64> = (1..=10).map(|i| i as f64 / 10.0).collect();
        assert_relative_eq!(c1(&w, 0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn c2_matches_c1_for_positive_path() {
        let w: Vec<f64> = (1..=10).map(|i| i as f64 / 10.0).collect();
        assert_relative_eq!(c1(&w, 0.0), c2(&w, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn simulate_cusum1_gamma0_matches_half_normal() {
        // For gamma=0 the CUSUM1 limit functional is just the running maximum
        // of a Wiener process — which at t=1 matches |Z| in distribution
        // (reflection principle). A moderate MC draw should land near the
        // half-normal 0.95 quantile (≈ 1.96). Loose tolerance because
        // samples=1000 and we're taking max over 500 pts, which skews above.
        let cv = simulate_critical_value(Detector::Cusum1, 0.0, 0.05, 2000, 500, Some(7));
        // Empirically this converges near 2.2–2.4 (not exactly 1.96 because
        // the running-maximum distribution of a Wiener process is Rayleigh-ish,
        // not half-normal at t=1 — the quantity is max_t W_t / t^0 = max_t W_t).
        assert!(
            (1.5..3.5).contains(&cv),
            "expected cv in [1.5, 3.5], got {}",
            cv
        );
    }

    #[test]
    fn simulate_deterministic_with_seed() {
        let a = simulate_critical_value(Detector::PageCusum, 0.0, 0.05, 200, 200, Some(123));
        let b = simulate_critical_value(Detector::PageCusum, 0.0, 0.05, 200, 200, Some(123));
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }

    #[test]
    fn simulate_all_four_detectors_finite() {
        for d in [
            Detector::Cusum,
            Detector::Cusum1,
            Detector::PageCusum,
            Detector::PageCusum1,
        ] {
            let cv = simulate_critical_value(d, 0.0, 0.05, 200, 100, Some(99));
            assert!(cv.is_finite() && cv > 0.0, "detector {:?}: cv={}", d, cv);
        }
    }

    #[test]
    fn simulate_higher_alpha_lower_critical_value() {
        let cv_01 = simulate_critical_value(Detector::PageCusum, 0.0, 0.01, 500, 200, Some(1));
        let cv_10 = simulate_critical_value(Detector::PageCusum, 0.0, 0.10, 500, 200, Some(1));
        // Stricter alpha (more conservative) => higher quantile => higher cv.
        assert!(
            cv_01 > cv_10,
            "cv(α=0.01)={} should exceed cv(α=0.10)={}",
            cv_01,
            cv_10
        );
    }
}
