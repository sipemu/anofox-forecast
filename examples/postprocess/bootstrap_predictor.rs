//! Bootstrap Prediction Intervals (Postprocess)
//!
//! Demonstrates the model-agnostic BootstrapPredictor that generates
//! per-step prediction intervals by resampling forecast residuals.
//!
//! Run with: cargo run --example bootstrap_predictor --features postprocess

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::Forecaster;
use anofox_forecast::postprocess::BootstrapPredictor;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== BootstrapPredictor Example ===\n");

    // Create sample time series
    let n = 100;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    let values: Vec<f64> = (0..n)
        .map(|i| 50.0 + 0.3 * i as f64 + 5.0 * (i as f64 * 0.2).sin())
        .collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    // Fit a model and get fitted values
    let mut model = SimpleExponentialSmoothing::auto();
    model.fit(&ts).unwrap();

    let fitted = model.fitted_values().unwrap().to_vec();
    let actuals = ts.primary_values().to_vec();

    // ── 1. IID Bootstrap ─────────────────────────────────────────────
    println!("--- 1. IID Bootstrap (95% coverage) ---\n");

    let predictor = BootstrapPredictor::new(0.95).n_replicates(1000).seed(42);

    let result = predictor.fit(&fitted, &actuals).unwrap();
    println!("Residuals used: {}", result.residuals().len());

    let point_forecast = model.predict(10).unwrap();
    let intervals = predictor.predict(&result, point_forecast.primary());

    println!(
        "{:<6} {:>10} {:>10} {:>10}",
        "Step", "Lower", "Point", "Upper"
    );
    for h in 0..10 {
        println!(
            "h={:<4} {:>10.2} {:>10.2} {:>10.2}",
            h + 1,
            intervals.lower()[h],
            point_forecast.primary()[h],
            intervals.upper()[h]
        );
    }

    // ── 2. Block Bootstrap ───────────────────────────────────────────
    println!("\n--- 2. Block Bootstrap (block_size=5) ---\n");

    let block_predictor = BootstrapPredictor::new(0.90)
        .n_replicates(500)
        .block_size(5)
        .seed(42);

    let block_result = block_predictor.fit(&fitted, &actuals).unwrap();
    let block_intervals = block_predictor.predict(&block_result, point_forecast.primary());

    println!("{:<6} {:>10} {:>10}", "Step", "Width(IID)", "Width(Block)");
    for h in 0..10 {
        let iid_width = intervals.upper()[h] - intervals.lower()[h];
        let block_width = block_intervals.upper()[h] - block_intervals.lower()[h];
        println!("h={:<4} {:>10.2} {:>10.2}", h + 1, iid_width, block_width);
    }

    println!("\nDone!");
}
