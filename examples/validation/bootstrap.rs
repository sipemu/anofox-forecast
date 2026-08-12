//! Bootstrap Confidence Intervals Example
//!
//! This example demonstrates how to use residual bootstrap methods
//! to generate empirical confidence intervals for forecasts.
//!
//! Run with: cargo run --example bootstrap

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::bootstrap::{bootstrap_forecast, bootstrap_intervals, BootstrapConfig};
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Bootstrap Confidence Intervals Example ===\n");

    // Create sample time series with trend + noise
    let n = 80;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let seasonal = 5.0 * ((i % 12) as f64 * std::f64::consts::PI / 6.0).sin();
            let noise = ((i * 17 + 13) % 11) as f64 - 5.0;
            trend + seasonal + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    println!("Time series: {} observations", ts.len());
    println!(
        "First 10 values: {:?}\n",
        ts.primary_values()[..10]
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // =========================================================================
    // 1. BootstrapConfig
    // =========================================================================
    println!("--- BootstrapConfig ---\n");

    let default_config = BootstrapConfig::default();
    println!("Default config:");
    println!("  n_samples:  {}", default_config.n_samples);
    println!("  block_size: {:?}", default_config.block_size);
    println!("  seed:       {:?}", default_config.seed);

    let custom_config = BootstrapConfig::new(500).with_block_size(5).with_seed(42);
    println!("\nCustom config:");
    println!("  n_samples:  {}", custom_config.n_samples);
    println!("  block_size: {:?}", custom_config.block_size);
    println!("  seed:       {:?}", custom_config.seed);

    // =========================================================================
    // 2. Residual Bootstrap Intervals
    // =========================================================================
    println!("\n--- Residual Bootstrap Intervals ---\n");

    let mut naive = Naive::new();
    naive.fit(&ts).unwrap();

    let config = BootstrapConfig::new(500).with_seed(42);
    let horizon = 10;

    let result = bootstrap_intervals(&naive, &ts, horizon, 0.95, &config).unwrap();

    println!(
        "Naive model - 95% bootstrap intervals ({} samples):",
        result.n_samples
    );
    println!("  Level: {:.0}%", result.level * 100.0);
    println!("\n  {:>4}  {:>10}  {:>10}", "Step", "Lower", "Upper");
    println!("  {:-<30}", "");
    for i in 0..horizon {
        println!(
            "  {:>4}  {:>10.2}  {:>10.2}",
            i + 1,
            result.lower[i],
            result.upper[i]
        );
    }

    // =========================================================================
    // 3. bootstrap_forecast() - Combined Point + Intervals
    // =========================================================================
    println!("\n--- bootstrap_forecast() ---\n");

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&ts).unwrap();

    let config = BootstrapConfig::new(500).with_seed(123);
    let forecast = bootstrap_forecast(&ses, &ts, horizon, 0.95, &config).unwrap();

    println!("SES model - bootstrap forecast with intervals:");
    println!("  Horizon:   {}", forecast.horizon());
    println!("  Has lower: {}", forecast.has_lower());
    println!("  Has upper: {}", forecast.has_upper());

    let point = forecast.primary();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();

    println!(
        "\n  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Step", "Lower", "Point", "Upper", "Width"
    );
    println!("  {:-<50}", "");
    for i in 0..horizon {
        let width = upper[i] - lower[i];
        println!(
            "  {:>4}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            i + 1,
            lower[i],
            point[i],
            upper[i],
            width
        );
    }

    // =========================================================================
    // 4. Reproducibility with Seeds
    // =========================================================================
    println!("\n--- Reproducibility with Seeds ---\n");

    let config_seeded = BootstrapConfig::new(200).with_seed(42);
    let r1 = bootstrap_intervals(&naive, &ts, 5, 0.95, &config_seeded).unwrap();
    let r2 = bootstrap_intervals(&naive, &ts, 5, 0.95, &config_seeded).unwrap();

    println!("Same seed (42) produces identical results:");
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>10}",
        "Step", "Run 1 Lower", "Run 2 Lower", "Match?"
    );
    println!("  {:-<50}", "");
    for i in 0..5 {
        let matches = (r1.lower[i] - r2.lower[i]).abs() < 1e-10;
        println!(
            "  {:>4}  {:>12.4}  {:>12.4}  {:>10}",
            i + 1,
            r1.lower[i],
            r2.lower[i],
            if matches { "yes" } else { "no" }
        );
    }

    // =========================================================================
    // 5. Block Bootstrap vs Residual Bootstrap
    // =========================================================================
    println!("\n--- Block Bootstrap vs Residual Bootstrap ---\n");

    println!("Block bootstrap preserves autocorrelation structure");
    println!("by resampling contiguous blocks of residuals.\n");

    let residual_config = BootstrapConfig::new(300).with_seed(42);
    let block_config = BootstrapConfig::new(300).with_block_size(5).with_seed(42);

    let residual_result = bootstrap_intervals(&naive, &ts, 5, 0.95, &residual_config).unwrap();
    let block_result = bootstrap_intervals(&naive, &ts, 5, 0.95, &block_config).unwrap();

    println!(
        "  {:>4}  {:>18}  {:>18}",
        "Step", "Residual Width", "Block(5) Width"
    );
    println!("  {:-<44}", "");
    for i in 0..5 {
        let rw = residual_result.upper[i] - residual_result.lower[i];
        let bw = block_result.upper[i] - block_result.lower[i];
        println!("  {:>4}  {:>18.4}  {:>18.4}", i + 1, rw, bw);
    }

    // =========================================================================
    // 6. Different Confidence Levels
    // =========================================================================
    println!("\n--- Different Confidence Levels ---\n");

    let config = BootstrapConfig::new(500).with_seed(42);

    for &level in &[0.80, 0.90, 0.95, 0.99] {
        let result = bootstrap_intervals(&ses, &ts, 5, level, &config).unwrap();
        let avg_width: f64 = (0..5)
            .map(|i| result.upper[i] - result.lower[i])
            .sum::<f64>()
            / 5.0;
        println!(
            "  {:.0}% interval: avg width = {:.4}",
            level * 100.0,
            avg_width
        );
    }
    println!("\n  Higher confidence => wider intervals (as expected).");

    // =========================================================================
    // Summary
    // =========================================================================
    println!(
        "
--- Summary ---

Bootstrap methods provide empirical confidence intervals by:
  1. Computing residuals from the fitted model
  2. Resampling residuals (with or without blocks)
  3. Creating synthetic series = fitted + resampled residuals
  4. Re-fitting the model on each synthetic series
  5. Collecting forecast distributions across all samples

Key parameters:
  n_samples:  More samples => smoother intervals (500-1000 typical)
  block_size: None = residual bootstrap, Some(k) = block bootstrap
  seed:       For reproducible results
  level:      Confidence level (0.80, 0.90, 0.95, 0.99)
"
    );

    println!("=== Bootstrap Confidence Intervals Example Complete ===");
}
