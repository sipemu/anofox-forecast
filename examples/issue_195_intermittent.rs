//! Repro for the third pathology on issue #195:
//! intermittent-bursty series like M5's `FOODS_3_444_WI_2` where
//! `.skaters()` produces a flat forecast anchored to the level.
//!
//! Synthetic: 48 months with ~37% zeros, alternating "off" months
//! (0-30) and "burst" months (400-1900).

use anofox_forecast::models::inspect::{Explanation, Inspectable};
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use anofox_forecast::prelude::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

const N: usize = 48;
const H: usize = 12;

fn synth() -> Vec<f64> {
    // Alternating zero-month / spike-month pattern (matches
    // FOODS_3_444_WI_2 shape: 37% zeros, spikes 400-1900).
    (0..N)
        .map(|i| {
            let mut seed =
                0xABCD_1234_5678_9EF0u64 ^ (i as u64).wrapping_mul(6_364_136_223_846_793_005);
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Deterministic: every third month is roughly zero, other
            // months are non-zero bursts. Adds a bit of jitter so the
            // pattern isn't perfectly periodic.
            let u = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
            let force_zero = (i % 3 == 0) || u < 0.10;
            if force_zero {
                return 0.0;
            }
            let n2 = (seed >> 20) & 0xFFF;
            let scale = n2 as f64 / 4095.0;
            400.0 + 1500.0 * scale
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
        "\n{label}:\n  peak={peak:.0}  trough={trough:.0}  swing={:.0}",
        peak - trough
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
    }
}

fn main() {
    let vals = synth();
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..N)
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    let ts = TimeSeries::univariate(stamps, vals.clone()).unwrap();

    let zero_frac = vals.iter().filter(|y| y.abs() < 1e-9).count() as f64 / N as f64;
    println!(
        "Series: N={N}, zero_frac={:.2}, non-zero range [{:.0}, {:.0}]",
        zero_frac,
        vals.iter()
            .cloned()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min),
        vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "Values head: {}",
        vals.iter()
            .take(24)
            .map(|v| format!("{v:.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    run(
        ".skaters().auto_with_seasonal_period(12) (with #195 fixes)",
        LaplaceForecaster::new()
            .skaters()
            .auto_with_seasonal_period(12),
        &ts,
    );
    run(
        ".auto().auto_with_seasonal_period(12) (with #195 fixes)",
        LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(12),
        &ts,
    );
}
