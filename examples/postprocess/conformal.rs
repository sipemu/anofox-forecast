//! Conformal prediction methods comparison.
//!
//! This example demonstrates different conformal prediction methods:
//! - Split conformal (fastest, uses holdout set)
//! - Cross-validation conformal (uses all data, slower)
//! - Jackknife+ (leave-one-out with finite sample validity)
//!
#![allow(clippy::needless_range_loop)]
//! Run with: cargo run --example postprocess_conformal

use anofox_forecast::postprocess::{ConformalMethod, ConformalPredictor, PointForecasts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Conformal Prediction Methods ===\n");

    // Generate synthetic data with known error distribution
    let n = 200;
    let forecasts: Vec<f64> = (0..n).map(|i| 50.0 + 0.2 * i as f64).collect();

    // Errors with slight heteroscedasticity
    let actuals: Vec<f64> = forecasts
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let noise = 1.5 * ((i as f64 * 0.05).sin()) + 0.5 * ((i as f64 * 0.13).cos());
            f + noise
        })
        .collect();

    let point_forecasts = PointForecasts::from_values(forecasts.clone());

    // Future forecasts to predict
    let future: Vec<f64> = (0..10).map(|i| 90.0 + 0.2 * i as f64).collect();
    let future_forecasts = PointForecasts::from_values(future.clone());

    let coverage = 0.90;

    println!("Training data: {} samples", n);
    println!("Target coverage: {:.0}%\n", coverage * 100.0);

    // Method 1: Split Conformal
    println!("--- Split Conformal (cal_fraction=0.2) ---");
    let split = ConformalPredictor::new(coverage, ConformalMethod::Split { cal_fraction: 0.2 });
    let split_result = split.fit(point_forecasts.values(), &actuals)?;
    let split_intervals = split.predict_values(&split_result, future_forecasts.values());

    print_intervals("Split", &future, &split_intervals);

    // Method 2: Cross-Validation Conformal
    println!("\n--- Cross-Validation Conformal (5 folds) ---");
    let cv = ConformalPredictor::new(coverage, ConformalMethod::CrossVal { n_folds: 5 });
    let cv_result = cv.fit(point_forecasts.values(), &actuals)?;
    let cv_intervals = cv.predict_values(&cv_result, future_forecasts.values());

    print_intervals("CV", &future, &cv_intervals);

    // Method 3: Jackknife+
    println!("\n--- Jackknife+ (leave-one-out) ---");
    let jack = ConformalPredictor::new(coverage, ConformalMethod::JackknifePlus);
    let jack_result = jack.fit(point_forecasts.values(), &actuals)?;
    let jack_intervals = jack.predict_values(&jack_result, future_forecasts.values());

    print_intervals("Jack+", &future, &jack_intervals);

    // Compare interval widths
    println!("\n=== Comparison ===");
    println!(
        "{:<12} {:>15} {:>15}",
        "Method", "Avg Width", "Quantile Value"
    );
    println!("{:-<45}", "");

    let split_width: f64 = split_intervals
        .lower()
        .iter()
        .zip(split_intervals.upper())
        .map(|(l, u)| u - l)
        .sum::<f64>()
        / 10.0;
    let cv_width: f64 = cv_intervals
        .lower()
        .iter()
        .zip(cv_intervals.upper())
        .map(|(l, u)| u - l)
        .sum::<f64>()
        / 10.0;
    let jack_width: f64 = jack_intervals
        .lower()
        .iter()
        .zip(jack_intervals.upper())
        .map(|(l, u)| u - l)
        .sum::<f64>()
        / 10.0;

    println!(
        "{:<12} {:>15.3} {:>15.3}",
        "Split",
        split_width,
        split_result.quantile_value()
    );
    println!(
        "{:<12} {:>15.3} {:>15.3}",
        "CrossVal",
        cv_width,
        cv_result.quantile_value()
    );
    println!(
        "{:<12} {:>15.3} {:>15.3}",
        "Jackknife+",
        jack_width,
        jack_result.quantile_value()
    );

    println!("\nNotes:");
    println!("- Split: Fastest, but uses only fraction of data for calibration");
    println!("- CrossVal: Uses all data, better for small samples");
    println!("- Jackknife+: Most robust, finite-sample coverage guarantee");

    Ok(())
}

fn print_intervals(
    name: &str,
    forecasts: &[f64],
    intervals: &anofox_forecast::postprocess::PredictionIntervals,
) {
    println!(
        "{:<6} {:>10} {:>10} {:>10}",
        "Step", "Forecast", "Lower", "Upper"
    );
    for i in 0..3 {
        println!(
            "{:<6} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            forecasts[i],
            intervals.lower()[i],
            intervals.upper()[i]
        );
    }
    println!("  ... ({} showing first 3 of 10)", name);
}
