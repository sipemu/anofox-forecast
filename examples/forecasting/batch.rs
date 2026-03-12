//! Batch Forecasting example.
//!
//! Demonstrates fitting multiple models to multiple time series
//! and comparing results.
//!
//! Run with: cargo run --example batch

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift, SimpleMovingAverage};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::{BoxedForecaster, ModelRegistry, ModelSpec};
use anofox_forecast::utils::comparison::{compare_registry, ComparisonConfig, ComparisonTable};
use chrono::{Duration, TimeZone, Utc};

/// Generate a synthetic time series with trend + seasonality + noise.
fn make_series(name: &str, n: usize, trend: f64, amplitude: f64, period: usize) -> (String, TimeSeries) {
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
    println!("=== Batch Forecasting Example ===\n");

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

    let config = ComparisonConfig::default();

    // --- 3. Compare all models on each series ---
    println!("--- Model comparison per series ---\n");
    for (name, ts) in &series_list {
        println!("Series: {}", name);
        match compare_registry(&registry, ts, &config) {
            Ok(results) => println!("{}\n", ComparisonTable(results)),
            Err(e) => println!("  Error: {}\n", e),
        }
    }

    // --- 4. Batch fit-and-predict across all series ---
    let horizon = 6;
    println!("--- Batch forecast (horizon={}) ---\n", horizon);

    let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster>)> = vec![
        ("Naive", Box::new(|| Box::new(Naive::new()))),
        ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
        ("SES", Box::new(|| Box::new(SimpleExponentialSmoothing::new(0.3)))),
        ("Holt", Box::new(|| Box::new(HoltLinearTrend::new(0.3, 0.1)))),
    ];

    for (series_name, ts) in &series_list {
        println!("Series: {}", series_name);
        for (model_name, factory) in &factories {
            let mut model = factory();
            if model.fit(ts).is_err() {
                println!("  {}: fit failed", model_name);
                continue;
            }
            match model.predict(horizon) {
                Ok(forecast) => {
                    let vals: Vec<String> = forecast.primary().iter().map(|v| format!("{:.2}", v)).collect();
                    println!("  {}: [{}]", model_name, vals.join(", "));
                }
                Err(e) => println!("  {}: predict failed: {}", model_name, e),
            }
        }
        println!();
    }

    // --- 5. Pick the best model per series ---
    println!("--- Best model per series (by in-sample RMSE) ---\n");
    for (name, ts) in &series_list {
        match compare_registry(&registry, ts, &config) {
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
