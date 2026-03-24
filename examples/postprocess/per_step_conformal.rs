//! Per-Horizon-Step Conformal Prediction Intervals
//!
//! Demonstrates ConformalPredictor::fit_per_step() which produces separate
//! interval widths for each forecast horizon step — tighter at h=1, wider
//! at h=12 when error grows with horizon.
//!
//! Run with: cargo run --example per_step_conformal --features postprocess

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::Forecaster;
use anofox_forecast::postprocess::ConformalPredictor;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Per-Step Conformal Prediction Intervals ===\n");

    // Create sample time series with trend + seasonality
    let n = 200;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            50.0 + 0.2 * i as f64
                + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                + ((i * 7 + 3) % 11) as f64 * 0.3
                - 1.5
        })
        .collect();

    let horizon = 7;

    // Simulate cross-validation folds: train on expanding windows, predict horizon steps
    let mut fold_forecasts = Vec::new();
    let mut fold_actuals = Vec::new();

    let fold_origins = [120, 130, 140, 150, 160, 170, 180];
    for &origin in &fold_origins {
        let train_ts =
            TimeSeries::univariate(timestamps[..origin].to_vec(), values[..origin].to_vec())
                .unwrap();

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&train_ts).unwrap();
        let forecast = model.predict(horizon).unwrap();

        fold_forecasts.push(forecast.primary().to_vec());
        fold_actuals.push(values[origin..origin + horizon].to_vec());
    }

    // ── Uniform conformal (single quantile for all steps) ────────────
    println!("--- Uniform Conformal (single half-width) ---\n");

    let cp = ConformalPredictor::split(0.90);
    let all_fc: Vec<f64> = fold_forecasts.iter().flatten().copied().collect();
    let all_ac: Vec<f64> = fold_actuals.iter().flatten().copied().collect();
    let uniform_result = cp.fit(&all_fc, &all_ac).unwrap();
    println!(
        "Uniform half-width: {:.2}\n",
        uniform_result.quantile_value()
    );

    // ── Per-step conformal (different half-width per step) ───────────
    println!("--- Per-Step Conformal (per-horizon half-widths) ---\n");

    let per_step_result = cp.fit_per_step(&fold_forecasts, &fold_actuals).unwrap();

    println!("{:<6} {:>12} {:>12}", "Step", "Half-width", "vs Uniform");
    for h in 0..horizon {
        let hw = per_step_result.half_widths()[h];
        let ratio = hw / uniform_result.quantile_value();
        println!("h={:<4} {:>12.2} {:>11.0}%", h + 1, hw, ratio * 100.0);
    }

    // Apply to a new point forecast
    let train_ts =
        TimeSeries::univariate(timestamps[..190].to_vec(), values[..190].to_vec()).unwrap();
    let mut model = SimpleExponentialSmoothing::auto();
    model.fit(&train_ts).unwrap();
    let forecast = model.predict(horizon).unwrap();

    let intervals = per_step_result.predict_intervals(forecast.primary());

    println!("\n--- Applied to forecast ---\n");
    println!(
        "{:<6} {:>10} {:>10} {:>10}",
        "Step", "Lower", "Point", "Upper"
    );
    for h in 0..horizon {
        println!(
            "h={:<4} {:>10.2} {:>10.2} {:>10.2}",
            h + 1,
            intervals.lower()[h],
            forecast.primary()[h],
            intervals.upper()[h]
        );
    }

    println!("\nDone!");
}
