//! Diagnostic: seasonality detection on N=48 monthly data.
//!
//! Synthesizes 4 years of monthly data (period=12) with varying
//! signal-to-noise ratios, then reports:
//! - What `detect_seasonal_period` picks
//! - What `auto_characteristics` says about seasonality strength
//! - Which auto toggles fire in `.auto()`
//! - Point forecast for the next 12 months vs the truth
//!
//! Run:
//!   cargo run --release --features distributional \
//!     --example monthly_48_seasonal_diagnostic

use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use anofox_forecast::prelude::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

const N: usize = 48;
const H: usize = 12;
const PERIOD: usize = 12;

fn synth(seasonal_amp: f64, noise_amp: f64) -> Vec<f64> {
    (0..N)
        .map(|i| {
            let seed = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
            let noise = noise_amp * (2.0 * (u - 0.5));
            let phase = (i % PERIOD) as f64 * std::f64::consts::PI * 2.0 / PERIOD as f64;
            10.0 + seasonal_amp * phase.sin() + noise
        })
        .collect()
}

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let n = pred.len().min(truth.len());
    (0..n).map(|i| (pred[i] - truth[i]).abs()).sum::<f64>() / n as f64
}

fn ts(vals: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..vals.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(stamps, vals.to_vec()).unwrap()
}

fn run(label: &str, seasonal_amp: f64, noise_amp: f64) {
    println!("\n=== {label} ===");
    println!("Signal-to-noise: seasonal amp {seasonal_amp:.1} / noise ±{noise_amp:.1}");

    let vals = synth(seasonal_amp, noise_amp);
    let train = ts(&vals);

    // Direct diagnostic: what does detect_seasonal_period pick?
    // Since it's pub(crate), we can't call it from an example. Instead
    // we mirror its computation here to explain what the auto path saw.
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n as f64;
    println!("Mean {mean:.2}, var {var:.3}");
    for p in [4usize, 7, 12, 24] {
        if p >= n / 2 {
            println!("  ACF({p:2}) = SKIPPED (p >= n/2 = {})", n / 2);
            continue;
        }
        let mut cov = 0.0;
        for i in p..n {
            cov += (vals[i] - mean) * (vals[i - p] - mean);
        }
        let acf = cov / ((n - p) as f64 * var);
        let flag = if acf.abs() > 0.35 {
            "✓ passes 0.35 gate"
        } else {
            "✗ below threshold"
        };
        println!("  ACF({p:2}) = {:+.3}  {flag}", acf);
    }

    // Configuration 1: .auto() with no explicit period.
    // (Defaults to auto_seasonal_period=7 which is wrong for monthly)
    let mut m1 = LaplaceForecaster::new().auto();
    m1.fit(&train).expect("fit fail");
    let f1 = m1.forecast_dist(H).expect("predict fail");
    let means1: Vec<f64> = f1
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();

    // Truth for horizon H: continue the seasonal pattern
    let truth: Vec<f64> = (N..N + H)
        .map(|i| {
            let phase = (i % PERIOD) as f64 * std::f64::consts::PI * 2.0 / PERIOD as f64;
            10.0 + seasonal_amp * phase.sin()
        })
        .collect();

    // Configuration 2: .auto() with explicit period=12.
    let mut m2 = LaplaceForecaster::new()
        .auto()
        .auto_with_seasonal_period(PERIOD);
    m2.fit(&train).expect("fit fail");
    let f2 = m2.forecast_dist(H).expect("predict fail");
    let means2: Vec<f64> = f2
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();

    // Configuration 3: explicit period + OPT-IN batch init (the fix).
    let mut m3 = LaplaceForecaster::new()
        .auto()
        .auto_with_seasonal_period(PERIOD)
        .with_seasonal_batch_init();
    m3.fit(&train).expect("fit fail");
    let f3 = m3.forecast_dist(H).expect("predict fail");
    let means3: Vec<f64> = f3
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();

    println!(
        "  MAE .auto()                                      : {:.3}",
        mae(&means1, &truth)
    );
    println!(
        "  MAE .auto().auto_with_seasonal_period(12)        : {:.3}",
        mae(&means2, &truth)
    );
    println!("  MAE .auto().auto_with_seasonal_period(12)         ");
    println!(
        "         .with_seasonal_batch_init() (FIX)         : {:.3}",
        mae(&means3, &truth)
    );

    // Peak / trough capture — the question aggregate MAE hides.
    let range = |xs: &[f64]| -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in xs {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    };
    let (t_lo, t_hi) = range(&truth);
    let (a_lo, a_hi) = range(&means1);
    let (b_lo, b_hi) = range(&means2);
    let (c_lo, c_hi) = range(&means3);
    println!(
        "  Truth  H=12 range     : lo={:+.2}  hi={:+.2}  swing={:.2}",
        t_lo,
        t_hi,
        t_hi - t_lo
    );
    println!(
        "  auto()                : lo={:+.2}  hi={:+.2}  swing={:.2}  (captures {:.0}% of truth swing)",
        a_lo,
        a_hi,
        a_hi - a_lo,
        100.0 * (a_hi - a_lo) / (t_hi - t_lo).max(1e-9)
    );
    println!(
        "  auto(p)               : lo={:+.2}  hi={:+.2}  swing={:.2}  (captures {:.0}% of truth swing)",
        b_lo,
        b_hi,
        b_hi - b_lo,
        100.0 * (b_hi - b_lo) / (t_hi - t_lo).max(1e-9)
    );
    println!(
        "  FIX                   : lo={:+.2}  hi={:+.2}  swing={:.2}  (captures {:.0}% of truth swing)",
        c_lo,
        c_hi,
        c_hi - c_lo,
        100.0 * (c_hi - c_lo) / (t_hi - t_lo).max(1e-9)
    );

    // Full H=12 cycle so peak + trough are both visible.
    println!(
        "  Truth  full-12: {}",
        truth
            .iter()
            .map(|v| format!("{v:+.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  auto() full-12: {}",
        means1
            .iter()
            .map(|v| format!("{v:+.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  FIX    full-12: {}",
        means3
            .iter()
            .map(|v| format!("{v:+.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Also run at H=24 to check the "H=24 straight line" symptom.
    let mut m1b = LaplaceForecaster::new().auto();
    let mut m3b = LaplaceForecaster::new()
        .auto()
        .auto_with_seasonal_period(PERIOD)
        .with_seasonal_batch_init();
    m1b.fit(&train).expect("fit fail");
    m3b.fit(&train).expect("fit fail");
    let means1_24: Vec<f64> = m1b
        .forecast_dist(24)
        .expect("predict fail")
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();
    let means3_24: Vec<f64> = m3b
        .forecast_dist(24)
        .expect("predict fail")
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();
    let truth24: Vec<f64> = (N..N + 24)
        .map(|i| {
            let phase = (i % PERIOD) as f64 * std::f64::consts::PI * 2.0 / PERIOD as f64;
            10.0 + seasonal_amp * phase.sin()
        })
        .collect();
    let (a24_lo, a24_hi) = range(&means1_24);
    let (c24_lo, c24_hi) = range(&means3_24);
    let (t24_lo, t24_hi) = range(&truth24);
    println!(
        "  --- H=24 (two cycles) ---   truth swing {:.2}  |  auto() swing {:.2} ({:.0}%)  |  FIX swing {:.2} ({:.0}%)",
        t24_hi - t24_lo,
        a24_hi - a24_lo,
        100.0 * (a24_hi - a24_lo) / (t24_hi - t24_lo).max(1e-9),
        c24_hi - c24_lo,
        100.0 * (c24_hi - c24_lo) / (t24_hi - t24_lo).max(1e-9),
    );
}

/// Variable seasonal strength: amplitude scales linearly across cycles.
/// `start_amp` is year-1 amplitude, `end_amp` is year-4 amplitude.
/// Truth for horizon H uses the year-5 amplitude extrapolation.
fn synth_variable(start_amp: f64, end_amp: f64, noise_amp: f64) -> Vec<f64> {
    (0..N)
        .map(|i| {
            let seed = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
            let noise = noise_amp * (2.0 * (u - 0.5));
            let phase = (i % PERIOD) as f64 * std::f64::consts::PI * 2.0 / PERIOD as f64;
            let cycle_idx = (i / PERIOD) as f64;
            let n_cycles = (N / PERIOD) as f64 - 1.0;
            let t = if n_cycles > 0.0 {
                cycle_idx / n_cycles
            } else {
                0.0
            };
            let amp = start_amp + t * (end_amp - start_amp);
            10.0 + amp * phase.sin() + noise
        })
        .collect()
}

fn run_variable(label: &str, start_amp: f64, end_amp: f64, noise_amp: f64) {
    println!("\n=== {label} ===");
    println!("Amplitude cycle-1: {start_amp:.1}, cycle-4: {end_amp:.1}, noise ±{noise_amp:.1}");

    let vals = synth_variable(start_amp, end_amp, noise_amp);
    let train = ts(&vals);

    // Truth at forecast time uses the SAME amplitude as year 4 (last
    // observed), i.e. "what happens next month assuming trend levels
    // off" — the standard forecasting question.
    let truth: Vec<f64> = (N..N + 12)
        .map(|i| {
            let phase = (i % PERIOD) as f64 * std::f64::consts::PI * 2.0 / PERIOD as f64;
            10.0 + end_amp * phase.sin()
        })
        .collect();

    #[allow(clippy::type_complexity)]
    let cfgs: [(&str, Box<dyn Fn() -> LaplaceForecaster>); 2] = [
        (
            ".auto().auto_with_seasonal_period(12)",
            Box::new(|| {
                LaplaceForecaster::new()
                    .auto()
                    .auto_with_seasonal_period(PERIOD)
            }),
        ),
        (
            "  + .with_seasonal_batch_init() (FIX)",
            Box::new(|| {
                LaplaceForecaster::new()
                    .auto()
                    .auto_with_seasonal_period(PERIOD)
                    .with_seasonal_batch_init()
            }),
        ),
    ];
    let (t_lo, t_hi) = truth
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });
    println!(
        "  Truth swing (year-4 pattern): {:.2} (peak {:+.2}, trough {:+.2})",
        t_hi - t_lo,
        t_hi,
        t_lo
    );
    for (name, make) in cfgs.iter() {
        let mut m = make();
        m.fit(&train).expect("fit fail");
        let means: Vec<f64> = m
            .forecast_dist(12)
            .expect("predict fail")
            .iter()
            .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
            .collect();
        let (lo, hi) = means
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
                (acc.0.min(v), acc.1.max(v))
            });
        let m_mae = mae(&means, &truth);
        println!(
            "  {name:<40} MAE={m_mae:.3}  swing={:.2} ({:.0}% of truth)",
            hi - lo,
            100.0 * (hi - lo) / (t_hi - t_lo).max(1e-9)
        );
    }
}

fn main() {
    println!("N={N} obs (4 years of monthly), forecasting H={H} months, period={PERIOD}");
    println!("Threshold reminder: detect_seasonal_period uses |ACF| > 0.35 gate.");

    run("STRONG seasonal, low noise", 3.0, 0.2);
    run("MEDIUM seasonal, medium noise", 2.0, 1.0);
    run("WEAK seasonal, high noise", 1.0, 2.0);
    run("SEASONAL DOMINATES noise", 5.0, 0.5);

    println!("\n\n### VARIABLE SEASONAL STRENGTH ###");
    println!("Amplitude changes linearly across the 4-year training window.");
    println!("Truth for the forecast horizon uses the year-4 (most recent) amplitude,");
    println!("which is what most callers actually want (\"assume today's pattern continues\").");
    run_variable("Growing amplitude (retail expanding)", 1.0, 4.0, 0.3);
    run_variable("Shrinking amplitude (dampening cycles)", 4.0, 1.0, 0.3);
    run_variable("Anomalous LAST cycle (COVID-like)", 3.0, 0.5, 0.3);
    run_variable("Stable amplitude (control)", 3.0, 3.0, 0.3);
}
