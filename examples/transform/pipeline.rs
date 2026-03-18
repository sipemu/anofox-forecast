//! Transform Pipeline example.
//!
//! Shows how to chain reversible transforms (BoxCox, Difference, Scale, Log)
//! around any Forecaster. The Pipeline itself implements Forecaster, making it
//! composable with cross-validation, ensembles, and batch operations.
//!
//! Run with: cargo run --example pipeline

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_forecast::transform::{
    BoxCoxTransform, DifferenceTransform, LogTransform, Pipeline, ScaleMethod, ScaleTransform,
};
use chrono::{Duration, TimeZone, Utc};

fn make_ts(values: Vec<f64>) -> TimeSeries {
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn main() {
    println!("=== Transform Pipeline Example ===\n");

    // ── 1. Basic pipeline: Difference → Naive ────────────────────────────
    println!("--- 1. Difference(1) → Naive ---\n");

    let values: Vec<f64> = (0..100)
        .map(|i| 50.0 + 0.5 * i as f64 + 3.0 * (i as f64 * 0.2).sin())
        .collect();
    let ts = make_ts(values);

    let mut pipeline = Pipeline::builder()
        .transform(DifferenceTransform::new(1))
        .model(Box::new(Naive::new()))
        .build();

    pipeline.fit(&ts).unwrap();
    let forecast = pipeline.predict(5).unwrap();

    println!("Forecasts (auto inverse-differenced):");
    for (i, val) in forecast.primary().iter().enumerate() {
        println!("  h={}: {:.2}", i + 1, val);
    }

    if let Some(residuals) = pipeline.residuals() {
        let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
        println!("Residual RMS: {:.4}", rms);
    }

    // ── 2. Log → Difference → Naive ─────────────────────────────────────
    println!("\n--- 2. Log → Difference(1) → Naive ---\n");

    // Exponential growth data
    let exp_values: Vec<f64> = (0..100)
        .map(|i| (1.0 + i as f64 * 0.03).exp() * 10.0)
        .collect();
    let ts_exp = make_ts(exp_values.clone());

    let mut pipeline_log = Pipeline::builder()
        .transform(LogTransform::new())
        .transform(DifferenceTransform::new(1))
        .model(Box::new(Naive::new()))
        .build();

    pipeline_log.fit(&ts_exp).unwrap();
    let fc_log = pipeline_log.predict(5).unwrap();

    println!(
        "Last 3 actuals: {:.2}, {:.2}, {:.2}",
        exp_values[97], exp_values[98], exp_values[99]
    );
    println!("Forecasts (inv-differenced, then exp):");
    for (i, val) in fc_log.primary().iter().enumerate() {
        println!("  h={}: {:.2}", i + 1, val);
    }

    // ── 3. BoxCox → AutoETS ──────────────────────────────────────────────
    println!("\n--- 3. BoxCox(auto) → AutoETS ---\n");

    // Heteroscedastic data: variance grows with level
    let hetero_values: Vec<f64> = (0..120)
        .map(|i| {
            let level = 10.0 + 0.5 * i as f64;
            let seasonal = (level * 0.1) * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            level + seasonal
        })
        .collect();
    let ts_hetero = make_ts(hetero_values);

    let mut pipeline_bc = Pipeline::builder()
        .transform(BoxCoxTransform::auto())
        .model(Box::new(AutoETS::non_seasonal()))
        .build();

    pipeline_bc.fit(&ts_hetero).unwrap();
    let fc_bc = pipeline_bc.predict(12).unwrap();

    println!("Pipeline name: {}", pipeline_bc.name());
    println!("Forecasts (auto inv-BoxCox):");
    for (i, val) in fc_bc.primary().iter().enumerate() {
        println!("  h={:>2}: {:.2}", i + 1, val);
    }

    // ── 4. Scale → model ─────────────────────────────────────────────────
    println!("\n--- 4. Scale(standardize) → Naive ---\n");

    let large_values: Vec<f64> = (0..80)
        .map(|i| 10_000.0 + 500.0 * (i as f64 * 0.15).sin())
        .collect();
    let ts_large = make_ts(large_values.clone());

    let mut pipeline_scale = Pipeline::builder()
        .transform(ScaleTransform::new(ScaleMethod::Standardize))
        .model(Box::new(Naive::new()))
        .build();

    pipeline_scale.fit(&ts_large).unwrap();
    let fc_scale = pipeline_scale.predict(5).unwrap();

    println!(
        "Original scale — last value: {:.0}",
        large_values.last().unwrap()
    );
    println!("Forecasts (auto un-scaled to original units):");
    for (i, val) in fc_scale.primary().iter().enumerate() {
        println!("  h={}: {:.2}", i + 1, val);
    }

    // ── 5. Stacking transforms ───────────────────────────────────────────
    println!("\n--- 5. BoxCox → Difference → Scale → Naive ---\n");

    let complex_values: Vec<f64> = (0..150)
        .map(|i| {
            let trend = (1.0 + i as f64 * 0.02).exp();
            let season = trend * 0.1 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            trend + season + 5.0
        })
        .collect();
    let ts_complex = make_ts(complex_values);

    let mut pipeline_stack = Pipeline::builder()
        .transform(BoxCoxTransform::auto())
        .transform(DifferenceTransform::new(1))
        .transform(ScaleTransform::new(ScaleMethod::Standardize))
        .model(Box::new(Naive::new()))
        .build();

    pipeline_stack.fit(&ts_complex).unwrap();
    let fc_stack = pipeline_stack.predict(6).unwrap();

    println!("3 transforms applied → model → 3 inverse transforms");
    println!("Forecasts:");
    for (i, val) in fc_stack.primary().iter().enumerate() {
        println!("  h={}: {:.2}", i + 1, val);
    }

    // ── 6. Pipeline is a Forecaster ──────────────────────────────────────
    println!("\n--- 6. Pipeline implements Forecaster ---\n");
    println!("A Pipeline can be used anywhere a Forecaster is expected:");
    println!("  - cross_validate(ts, &spec)");
    println!("  - compare_models(ts, &specs)");
    println!("  - batch_forecast(series, &specs)");
    println!("  - EnsembleForecaster::with_models(pipelines)");
    println!("\nExample ModelSpec factory:");
    println!("  ModelSpec::new(\"BoxCox+Naive\", || {{");
    println!("      Box::new(Pipeline::builder()");
    println!("          .transform(BoxCoxTransform::auto())");
    println!("          .model(Box::new(Naive::new()))");
    println!("          .build())");
    println!("  }}, false)");

    println!("\n=== Transform Pipeline Example Complete ===");
}
