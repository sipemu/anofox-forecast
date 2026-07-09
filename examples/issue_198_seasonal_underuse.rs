//! Reproduces issue #198: LaplaceForecaster `.auto()` / `.skaters()`
//! collapse to a near-flat line on noisy established seasonal series
//! even though the seasonal-EMA leaf IS in the pool (post #195 fix).
//!
//! The user's DuckDB SQL synthetic ported to Rust:
//! - 60 monthly points (5 years)
//! - Base 1000, seasonal factors 0.24-2.25 (July peak, ~9× swing)
//! - ±15% monthly noise + ±7.5% per-year amplitude jitter
//! - Fully deterministic (hash-based noise)
//!
//! **Since v0.15.2, `.with_seasonal(p)` defaults batch init on**,
//! so all four rows below now recover ~8.9× swing. Kept as a
//! regression guard: if you see 2× again, the fix regressed.
//!
//! Run:
//!   cargo run --release --features distributional \
//!     --example issue_198_seasonal_underuse

use anofox_forecast::models::inspect::{Explanation, Inspectable};
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use anofox_forecast::prelude::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

// Retail-shape seasonal factors matching the SQL:
// (1,0.25),(2,0.30),(3,0.55),(4,1.05),(5,1.65),(6,2.15),
// (7,2.25),(8,1.85),(9,1.15),(10,0.60),(11,0.32),(12,0.24)
const SEAS: [f64; 12] = [
    0.25, 0.30, 0.55, 1.05, 1.65, 2.15, 2.25, 1.85, 1.15, 0.60, 0.32, 0.24,
];
const BASE: f64 = 1000.0;
const N: usize = 60;
const H: usize = 12;

fn synth() -> Vec<f64> {
    (0..N)
        .map(|k| {
            let mo = ((k % 12) + 1) as u64; // 1..=12
            let yr = (k / 12) as u64;
            let mult = SEAS[(mo - 1) as usize];
            // Deterministic hash noise matching the SQL:
            //  (1 + 0.30 * (((k*2654435761)%1000)/1000.0 - 0.5))
            //  (1 + 0.15 * (((yr*40503)%100)/100.0 - 0.5))
            let n_monthly = 1.0
                + 0.30 * ((((k as u64).wrapping_mul(2_654_435_761) % 1000) as f64 / 1000.0) - 0.5);
            let n_yearly = 1.0 + 0.15 * (((yr.wrapping_mul(40503) % 100) as f64 / 100.0) - 0.5);
            let _ = mo;
            (BASE * mult * n_monthly * n_yearly).max(0.0).round()
        })
        .collect()
}

fn run(label: &str, mut model: LaplaceForecaster, ts: &TimeSeries) {
    model.fit(ts).expect("fit fail");
    let dists = model.forecast_dist(H).expect("predict fail");
    let means: Vec<f64> = dists
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();
    let peak = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough = means.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "\n{label}:\n  peak={peak:.0}  trough={trough:.0}  ratio={:.2}x  swing={:.0}",
        peak / trough.max(1.0),
        peak - trough,
    );
    print!("  forecast: ");
    for m in &means {
        print!("{m:.0} ");
    }
    println!();
    if let Ok(Explanation::Laplace(ex)) = Inspectable::explanation(&model) {
        let mut w: Vec<(&str, f64)> = ex
            .leaf_names
            .iter()
            .map(String::as_str)
            .zip(ex.leaf_weights.iter().copied())
            .collect();
        w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        print!("  softmax top-5: ");
        for (name, w) in w.iter().take(5) {
            print!("{name}={w:.3} ");
        }
        println!();
        // Also report seasonal_ema's weight specifically.
        let sw: f64 = w
            .iter()
            .filter(|(n, _)| n.starts_with("seasonal_ema"))
            .map(|(_, w)| *w)
            .sum();
        println!("  seasonal_ema weight: {sw:.4}");
    }
}

fn main() {
    let vals = synth();
    let base = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..N)
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    let ts = TimeSeries::univariate(stamps.clone(), vals.clone()).unwrap();

    let peak_train = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough_train = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "Training window: N={N}, range [{trough_train:.0}, {peak_train:.0}], ratio {:.2}x",
        peak_train / trough_train.max(1.0)
    );
    print!("Training head: ");
    for v in &vals[..24] {
        print!("{v:.0} ");
    }
    println!();

    run(
        ".auto().with_seasonal(12)  (laplace)",
        LaplaceForecaster::new().auto().with_seasonal(12),
        &ts,
    );
    run(
        ".skaters().with_seasonal(12)  (skater)",
        LaplaceForecaster::new().skaters().with_seasonal(12),
        &ts,
    );
    run(
        ".skaters().with_seasonal(12).with_seasonal_batch_init()  (skater_bi)",
        LaplaceForecaster::new()
            .skaters()
            .with_seasonal(12)
            .with_seasonal_batch_init(),
        &ts,
    );
    run(
        ".auto().with_seasonal(12).with_seasonal_batch_init()  (auto_bi)",
        LaplaceForecaster::new()
            .auto()
            .with_seasonal(12)
            .with_seasonal_batch_init(),
        &ts,
    );
}
