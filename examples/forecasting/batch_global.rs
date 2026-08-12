//! Batch / Global forecasting for many series.
//!
//! Demonstrates `GlobalETS`, `GlobalAutoETS`, `GlobalCroston`, and `GlobalTheta`
//! for processing many series with shared parameters — 28-96x faster than
//! individual fits for seasonal models.
//!
//! Run: cargo run --release --example batch_global

use anofox_forecast::models::exponential::{ETSSpec, GlobalAutoETS, GlobalETS, ModelPool};
use anofox_forecast::models::intermittent::GlobalCroston;
use anofox_forecast::models::theta::GlobalTheta;
use std::time::Instant;

fn main() {
    let n_series = 500;
    let n = 200;
    let period = 12;
    let horizon = 12;

    println!("=== Batch/Global Forecasting Demo ===");
    println!(
        "{} series, {} observations each, period={}\n",
        n_series, n, period
    );

    // Generate synthetic seasonal data
    let series: Vec<Vec<f64>> = (0..n_series)
        .map(|s| {
            (0..n)
                .map(|i| {
                    50.0 + (s as f64) * 3.0
                        + 0.2 * i as f64
                        + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                        + ((i * 7 + s * 13 + 3) % 11) as f64 * 0.5
                        - 2.75
                })
                .collect()
        })
        .collect();

    // --- GlobalETS: fixed spec, shared parameters ---
    println!("1. GlobalETS — ETS(A,N,A) with shared α, γ");
    let start = Instant::now();
    let mut global_ets = GlobalETS::new(ETSSpec::ana(), period);
    global_ets.fit(&series).unwrap();
    let forecasts = global_ets.predict(horizon);
    let elapsed = start.elapsed();
    let (alpha, _, gamma, _) = global_ets.params();
    println!(
        "   Fitted {} series in {:.1}ms ({:.3}ms/series)",
        n_series,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / n_series as f64
    );
    println!("   Shared α={:.4}, γ={:.4}", alpha, gamma.unwrap_or(0.0));
    println!(
        "   Series 0 forecast: [{:.1}, {:.1}, {:.1}, ...]",
        forecasts[0][0], forecasts[0][1], forecasts[0][2]
    );

    // --- GlobalAutoETS: automatic model selection ---
    println!("\n2. GlobalAutoETS — Reduced pool, best spec per series");
    let start = Instant::now();
    let mut auto = GlobalAutoETS::new(period, ModelPool::Reduced);
    auto.fit(&series).unwrap();
    let forecasts = auto.predict(horizon);
    let elapsed = start.elapsed();
    let specs = auto.selected_specs();
    println!(
        "   Fitted {} series in {:.0}ms ({:.2}ms/series)",
        n_series,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / n_series as f64
    );
    // Count spec distribution
    let mut counts = std::collections::HashMap::new();
    for s in &specs {
        *counts.entry(format!("{:?}", s)).or_insert(0usize) += 1;
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
    println!("   Model selection:");
    for (spec, count) in sorted.iter().take(3) {
        println!(
            "     {} ({:.0}%) {}",
            count,
            *count as f64 / n_series as f64 * 100.0,
            spec
        );
    }
    println!(
        "   Series 0 forecast: [{:.1}, {:.1}, {:.1}, ...]",
        forecasts[0][0], forecasts[0][1], forecasts[0][2]
    );

    // --- GlobalCroston: intermittent demand ---
    println!("\n3. GlobalCroston — SBA variant, shared α");
    let intermittent: Vec<Vec<f64>> = (0..n_series)
        .map(|s| {
            (0..n)
                .map(|i| {
                    if ((i * 7 + s * 13 + 5) % 10) < 3 {
                        ((i * 3 + s * 7 + 1) % 8 + 1) as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let mut croston = GlobalCroston::sba();
    croston.fit(&intermittent).unwrap();
    let forecasts = croston.predict(horizon);
    let elapsed = start.elapsed();
    println!(
        "   Fitted {} series in {:.1}ms ({:.3}ms/series)",
        n_series,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / n_series as f64
    );
    println!("   Shared α={:.4}", croston.alpha());
    println!("   Series 0 forecast (flat): {:.2}", forecasts[0][0]);

    // --- GlobalTheta: shared alpha ---
    println!("\n4. GlobalTheta — Standard Theta Method, shared α");
    let start = Instant::now();
    let mut theta = GlobalTheta::new();
    theta.fit(&series).unwrap();
    let forecasts = theta.predict(horizon);
    let elapsed = start.elapsed();
    println!(
        "   Fitted {} series in {:.1}ms ({:.3}ms/series)",
        n_series,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / n_series as f64
    );
    println!("   Shared α={:.4}", theta.alpha());
    println!(
        "   Series 0 forecast: [{:.1}, {:.1}, {:.1}, ...]",
        forecasts[0][0], forecasts[0][1], forecasts[0][2]
    );

    println!("\nDone.");
}
