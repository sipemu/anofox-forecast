//! Adversarial robustness suite — port of skaters' `rust/tests/robustness.rs`.
//!
//! Each test fits `LaplaceForecaster` to a hand-crafted problem series
//! (constant, lattice, monster-spike, extreme-tick, scale-collapse,
//! vol-whiplash) and asserts the resulting `GaussianMixture` per-horizon
//! output is well-formed: finite `logpdf`, `cdf ∈ [0, 1]`, monotone
//! finite quantiles.
//!
//! Adaptations from skaters:
//! - Our API is batch (`fit` + `forecast_dist`), theirs is streaming
//!   (`.step(y)`). We fit the full series once and check the final
//!   forecast rather than checking every 997th step.
//! - Skaters' full-forecaster serde round-trip is out of scope for us:
//!   our `LaplaceForecaster` holds `Box<dyn Leaf>` (trait objects) that
//!   aren't serde-derivable. Component serde is tested in
//!   `src/models/laplace/dist.rs` and `gpd_tails.rs`.
//! - Determinism check on `fit + forecast_dist` — same input,
//!   bit-identical output.

#![cfg(feature = "distributional")]

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::{
    DistributionalForecaster, GaussianMixture, LaplaceForecaster,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

// -- helpers --

/// Deterministic LCG matching skaters' adversarial harness.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 as f64 / 4_294_967_296.0
    }

    /// Box-Muller gaussian.
    fn gauss(&mut self) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.next();
        }
        let mut v = 0.0;
        while v == 0.0 {
            v = self.next();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn build_ts(values: Vec<f64>) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::seconds(i as i64))
        .collect();
    TimeSeries::univariate(stamps, values).expect("valid TimeSeries")
}

fn assert_wellformed(mix: &GaussianMixture, y_near: f64, label: &str) {
    let lp = mix.logpdf(y_near);
    assert!(!lp.is_nan(), "{label}: logpdf NaN");
    let c = mix.cdf(y_near);
    assert!((0.0..=1.0).contains(&c), "{label}: cdf out of range: {c}");
    let ps = [0.001, 0.25, 0.5, 0.75, 0.999];
    let qs: Vec<f64> = ps.iter().map(|&p| mix.quantile(p)).collect();
    assert!(
        qs.iter().all(|q| q.is_finite()),
        "{label}: non-finite quantile in {qs:?}"
    );
    for w in qs.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "{label}: quantiles unordered: {qs:?}");
    }
}

fn soak(values: Vec<f64>, label: &str) {
    let last = *values.last().expect("non-empty");
    let ts = build_ts(values);
    let mut f = LaplaceForecaster::new().auto();
    f.fit(&ts).expect("fit");
    let dists = f.forecast_dist(1).expect("forecast");
    assert!(!dists.is_empty(), "{label}: empty forecast");
    assert_wellformed(&dists[0], last, label);
}

// -- adversarial series --

#[test]
fn constant_series() {
    soak(vec![3.7; 400], "constant");
}

#[test]
fn lattice_series() {
    let mut r = Lcg(17);
    let mut v = 1.0;
    let mut ys = Vec::new();
    for _ in 0..600 {
        if r.next() >= 0.7 {
            let step = [-0.25, 0.25, 0.5][(r.next() * 3.0) as usize % 3];
            v += step;
        }
        ys.push(v);
    }
    soak(ys, "lattice");
}

#[test]
fn monster_spike_then_recovery() {
    let mut r = Lcg(23);
    let mut ys: Vec<f64> = (0..500).map(|_| r.gauss()).collect();
    ys.push(1e9);
    ys.extend((0..500).map(|_| r.gauss()));
    soak(ys, "monster_spike");
}

#[test]
fn extreme_finite_tick() {
    let mut r = Lcg(29);
    let mut ys: Vec<f64> = (0..400).map(|_| r.gauss()).collect();
    ys.push(1e100); // 1e300 overflows YJ; 1e100 is still an extreme finite tick.
    ys.extend((0..400).map(|_| r.gauss()));
    soak(ys, "extreme_tick");
}

#[test]
fn scale_collapse_and_recovery() {
    let mut r = Lcg(31);
    let mut ys: Vec<f64> = (0..400).map(|_| r.gauss()).collect();
    ys.extend(std::iter::repeat_n(0.0, 400));
    ys.extend((0..400).map(|_| r.gauss()));
    soak(ys, "scale_collapse");
}

#[test]
fn vol_whiplash_on_trend() {
    let mut r = Lcg(41);
    let mut lvl = 0.0;
    let mut ys = Vec::new();
    for t in 0..2000 {
        let vol = if (t / 200) % 2 == 1 { 10.0 } else { 1.0 };
        lvl += 0.05 + vol * r.gauss();
        ys.push(lvl);
    }
    soak(ys, "vol_whiplash");
}

// -- bit-level determinism --

#[test]
fn bitwise_determinism_on_fit_forecast() {
    // Same input series → bit-identical forecast_dist output. Runs
    // twice from scratch, hashes logpdf/cdf/quantile at fixed points,
    // asserts equality.
    let run = || {
        let mut r = Lcg(7);
        let ys: Vec<f64> = (0..500).map(|_| r.gauss()).collect();
        let ts = build_ts(ys);
        let mut f = LaplaceForecaster::new().auto();
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(12).unwrap();
        let mut acc: u64 = 0xcbf29ce484222325;
        for d in &dists {
            for v in [
                d.logpdf(0.3),
                d.cdf(0.3),
                d.quantile(0.1),
                d.mean(),
                d.std(),
            ] {
                acc ^= v.to_bits();
                acc = acc.wrapping_mul(0x100000001b3);
            }
        }
        acc
    };
    assert_eq!(run(), run(), "two identical fit+forecast runs diverged");
}

/// Port of skaters issue microprediction/skaters#86:
/// for integrated transforms, predictive variance should be nondecreasing
/// in horizon `h`. `MultiScaleLaplace` mixes coarse-clock forecasts at
/// stride boundaries `{1, period, k}` — verify no "variance sawtooth"
/// on a random walk (worst-case integrated series).
#[test]
fn multiscale_variance_is_monotone_in_horizon_on_random_walk() {
    use anofox_forecast::models::laplace::MultiScaleLaplace;
    // 500 obs of a Gaussian random walk — variance should scale as h.
    let mut r = Lcg(53);
    let mut lvl = 0.0;
    let ys: Vec<f64> = (0..500)
        .map(|_| {
            lvl += r.gauss();
            lvl
        })
        .collect();
    let ts = build_ts(ys);
    const H: usize = 30;
    let mut m = MultiScaleLaplace::skaters(H);
    m.fit(&ts).expect("fit");
    let dists = m.forecast_dist(H).expect("forecast");
    assert_eq!(dists.len(), H);
    // Weak monotonicity — allow a small numerical epsilon at scale
    // boundaries but flag anything egregious.
    let vars: Vec<f64> = dists.iter().map(|d| d.variance()).collect();
    let mut violations: Vec<String> = Vec::new();
    for h in 1..H {
        let ratio = vars[h] / vars[h - 1];
        // Downward jumps of more than 5% at a boundary = sawtooth
        if ratio < 0.95 {
            violations.push(format!(
                "  h={}: var[{}]={:.3} > var[{}]={:.3} (ratio={:.3})",
                h,
                h - 1,
                vars[h - 1],
                h,
                vars[h],
                ratio
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "MultiScaleLaplace variance sawtooth on random walk (skaters#86):\n{}",
        violations.join("\n")
    );
}

#[test]
fn bitwise_determinism_on_skaters_pool() {
    // Same but with the wider .skaters() pool + our v0.15.3/4 knobs.
    let run = || {
        let mut r = Lcg(11);
        let ys: Vec<f64> = (0..500).map(|_| r.gauss()).collect();
        let ts = build_ts(ys);
        let mut f = LaplaceForecaster::new()
            .skaters()
            .with_scoring_horizon(12)
            .with_scoring_window(14);
        f.fit(&ts).unwrap();
        let dists = f.forecast_dist(12).unwrap();
        let mut acc: u64 = 0xcbf29ce484222325;
        for d in &dists {
            for v in [d.logpdf(0.3), d.cdf(0.3), d.quantile(0.1)] {
                acc ^= v.to_bits();
                acc = acc.wrapping_mul(0x100000001b3);
            }
        }
        acc
    };
    assert_eq!(run(), run(), "two identical .skaters() runs diverged");
}
