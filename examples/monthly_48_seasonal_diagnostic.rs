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

    // Show truth vs each forecast for the first 6 horizons
    println!(
        "  Truth  head-6: {}",
        truth
            .iter()
            .take(6)
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  auto() head-6: {}",
        means1
            .iter()
            .take(6)
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  auto(p) head-6: {}",
        means2
            .iter()
            .take(6)
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  FIX     head-6: {}",
        means3
            .iter()
            .take(6)
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn main() {
    println!("N={N} obs (4 years of monthly), forecasting H={H} months, period={PERIOD}");
    println!("Threshold reminder: detect_seasonal_period uses |ACF| > 0.35 gate.");

    run("STRONG seasonal, low noise", 3.0, 0.2);
    run("MEDIUM seasonal, medium noise", 2.0, 1.0);
    run("WEAK seasonal, high noise", 1.0, 2.0);
    run("SEASONAL DOMINATES noise", 5.0, 0.5);
}
