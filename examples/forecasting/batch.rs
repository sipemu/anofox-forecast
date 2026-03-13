//! Batch Forecasting example.
//!
//! Demonstrates fitting multiple models to multiple time series
//! and comparing results. Uses the `batch` module for parallel processing
//! when the `parallel` feature is enabled.
//!
//! Run with:
//!   cargo run --example batch
//!   cargo run --example batch --features parallel   # parallel mode

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift, SimpleMovingAverage};
use anofox_forecast::models::batch::{fit_predict_many, fit_registry};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::{ModelRegistry, ModelSpec};
use anofox_forecast::utils::comparison::{compare_registry, ComparisonConfig, ComparisonTable};
use chrono::{Duration, TimeZone, Utc};

/// Generate a synthetic time series with trend + seasonality + noise.
fn make_series(
    name: &str,
    n: usize,
    trend: f64,
    amplitude: f64,
    period: usize,
) -> (String, TimeSeries) {
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let seasonal = amplitude * (2.0 * std::f64::consts::PI * t / period as f64).sin();
            10.0 + trend * t + seasonal + 0.5 * ((i * 7 + 3) % 11) as f64 / 10.0
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    (name.to_string(), ts)
}

fn main() {
    println!("=== Batch Forecasting Example ===");
    #[cfg(feature = "parallel")]
    println!("(parallel mode enabled)\n");
    #[cfg(not(feature = "parallel"))]
    println!("(sequential mode — enable `parallel` feature for rayon)\n");

    // --- 1. Create multiple time series ---
    let series_list = vec![
        make_series("trending", 100, 0.3, 0.0, 12),
        make_series("seasonal", 100, 0.0, 5.0, 12),
        make_series("trend+seasonal", 100, 0.2, 3.0, 12),
        make_series("flat", 100, 0.0, 0.0, 12),
    ];

    // --- 2. Build a model registry ---
    let mut registry = ModelRegistry::new();
    registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
    registry.register(ModelSpec::new(
        "RWD",
        || Box::new(RandomWalkWithDrift::new()),
        true,
    ));
    registry.register(ModelSpec::new(
        "SMA-5",
        || Box::new(SimpleMovingAverage::new(5)),
        false,
    ));
    registry.register(ModelSpec::new(
        "SES",
        || Box::new(SimpleExponentialSmoothing::new(0.3)),
        false,
    ));
    registry.register(ModelSpec::new(
        "Holt",
        || Box::new(HoltLinearTrend::new(0.3, 0.1)),
        true,
    ));

    // --- 3. Compare all models on each series (parallel when feature enabled) ---
    println!("--- Model comparison per series ---\n");
    for (name, ts) in &series_list {
        println!("Series: {}", name);
        match compare_registry(&registry, ts, &ComparisonConfig::default()) {
            Ok(results) => println!("{}\n", ComparisonTable(results)),
            Err(e) => println!("  Error: {}\n", e),
        }
    }

    // --- 4. Batch fit-and-predict: one model across all series (parallel) ---
    let horizon = 6;
    println!("--- Batch fit_predict_many (horizon={}) ---\n", horizon);

    let series_refs: Vec<&TimeSeries> = series_list.iter().map(|(_, ts)| ts).collect();

    // fit_predict_many uses rayon::par_iter when `parallel` is enabled
    let results = fit_predict_many(Naive::new, &series_refs, horizon);
    for (r, (name, _)) in results.iter().zip(&series_list) {
        match &r.forecast {
            Some(f) => {
                let vals: Vec<String> = f
                    .primary()
                    .iter()
                    .map(|v: &f64| format!("{:.2}", v))
                    .collect();
                println!("  {} ({}): [{}]", name, r.model_name, vals.join(", "));
            }
            None => println!(
                "  {}: {}",
                name,
                r.error.as_deref().unwrap_or("unknown error")
            ),
        }
    }
    println!();

    // --- 5. Registry comparison: all models on one series (parallel) ---
    println!("--- fit_registry: all models on one series ---\n");
    let reg_results = fit_registry(&registry, &series_list[0].1);
    for r in &reg_results {
        if let Some(metrics) = &r.metrics {
            println!(
                "  {}: RMSE={:.4}, MAE={:.4}",
                r.model_name, metrics.rmse, metrics.mae
            );
        } else {
            println!(
                "  {}: {}",
                r.model_name,
                r.error.as_deref().unwrap_or("no metrics")
            );
        }
    }
    println!();

    // --- 6. Pick the best model per series ---
    println!("--- Best model per series (by in-sample RMSE) ---\n");
    for (name, ts) in &series_list {
        match compare_registry(&registry, ts, &ComparisonConfig::default()) {
            Ok(results) if !results.is_empty() => {
                let best = &results[0];
                println!(
                    "  {}: {} (RMSE={:.4}, MAE={:.4})",
                    name, best.model_name, best.in_sample.rmse, best.in_sample.mae
                );
            }
            _ => println!("  {}: no model fit successfully", name),
        }
    }
}
