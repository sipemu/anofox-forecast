//! Quantile methods comparison.
//!
//! This example compares different methods for generating quantile forecasts:
//! - HistoricalSimulator: Empirical error distribution
//! - NormalPredictor: Gaussian error assumption
//! - IDRPredictor: Isotonic distributional regression
//!
#![allow(clippy::needless_range_loop)]
//! Run with: cargo run --example postprocess_quantile_methods

use anofox_forecast::postprocess::{
    HistoricalSimulator, IDRPredictor, NormalPredictor, PointForecasts,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantile Methods Comparison ===\n");

    // Generate synthetic data with non-Gaussian errors
    let n = 150;
    let forecasts: Vec<f64> = (0..n).map(|i| 100.0 + 0.5 * i as f64).collect();

    // Errors with slight skewness (not perfectly Gaussian)
    let actuals: Vec<f64> = forecasts
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let base_error = 2.0 * ((i as f64 * 0.1).sin());
            let skew = if i % 3 == 0 { 1.5 } else { -0.3 };
            f + base_error + skew * 0.5
        })
        .collect();

    let point_forecasts = PointForecasts::from_values(forecasts.clone());

    // Future forecasts
    let future: Vec<f64> = (0..5).map(|i| 175.0 + 0.5 * i as f64).collect();
    let future_forecasts = PointForecasts::from_values(future.clone());

    // Define quantile levels
    let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];

    println!("Training data: {} samples", n);
    println!("Quantile levels: {:?}\n", quantiles);

    // Method 1: Historical Simulation
    println!("--- Historical Simulator ---");
    println!("Uses empirical distribution of past errors\n");

    let hist_sim = HistoricalSimulator::new(quantiles.clone());
    let hist_result = hist_sim.fit(point_forecasts.values(), &actuals)?;
    let hist_quantiles = hist_sim.predict_values(&hist_result, future_forecasts.values())?;

    print_quantile_table("HistSim", &future, &hist_quantiles, &quantiles);

    // Method 2: Normal Predictor
    println!("\n--- Normal Predictor ---");
    println!("Assumes Gaussian errors (mean=0, learned std)\n");

    let normal = NormalPredictor::new(quantiles.clone());
    let normal_result = normal.fit(point_forecasts.values(), &actuals)?;
    let normal_quantiles = normal.predict_values(&normal_result, future_forecasts.values())?;

    print_quantile_table("Normal", &future, &normal_quantiles, &quantiles);
    println!("  Estimated std: {:.3}", normal_result.std_dev());

    // Method 3: IDR (Isotonic Distributional Regression)
    println!("\n--- IDR Predictor ---");
    println!("Learns monotone relationship between forecast and quantiles\n");

    let idr = IDRPredictor::new(quantiles.clone());
    let idr_result = idr.fit(point_forecasts.values(), &actuals)?;
    let idr_quantiles = idr.predict_values(&idr_result, future_forecasts.values())?;

    print_quantile_table("IDR", &future, &idr_quantiles, &quantiles);

    // Compare median predictions
    println!("\n=== Median (q=0.5) Comparison ===");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}",
        "Forecast", "HistSim", "Normal", "IDR", "Spread"
    );
    println!("{:-<60}", "");

    for i in 0..future.len() {
        let hist_med = hist_quantiles.at_time(i).unwrap()[2]; // q=0.5 is index 2
        let norm_med = normal_quantiles.at_time(i).unwrap()[2];
        let idr_med = idr_quantiles.at_time(i).unwrap()[2];
        let spread = (hist_med - norm_med).abs().max((hist_med - idr_med).abs());

        println!(
            "{:<10.2} {:>12.2} {:>12.2} {:>12.2} {:>12.3}",
            future[i], hist_med, norm_med, idr_med, spread
        );
    }

    // Compare interval widths (10%-90%)
    println!("\n=== 80% Interval Width (q0.1 to q0.9) ===");
    println!(
        "{:<10} {:>12} {:>12} {:>12}",
        "Forecast", "HistSim", "Normal", "IDR"
    );
    println!("{:-<50}", "");

    for i in 0..future.len() {
        let hist_width =
            hist_quantiles.at_time(i).unwrap()[4] - hist_quantiles.at_time(i).unwrap()[0];
        let norm_width =
            normal_quantiles.at_time(i).unwrap()[4] - normal_quantiles.at_time(i).unwrap()[0];
        let idr_width = idr_quantiles.at_time(i).unwrap()[4] - idr_quantiles.at_time(i).unwrap()[0];

        println!(
            "{:<10.2} {:>12.2} {:>12.2} {:>12.2}",
            future[i], hist_width, norm_width, idr_width
        );
    }

    println!("\nNotes:");
    println!("- HistSim: Non-parametric, captures true error shape");
    println!("- Normal: Parametric, efficient if errors are Gaussian");
    println!("- IDR: Adaptive, learns forecast-dependent uncertainty");

    Ok(())
}

fn print_quantile_table(
    name: &str,
    forecasts: &[f64],
    quantiles: &anofox_forecast::postprocess::QuantileForecasts,
    q_levels: &[f64],
) {
    print!("{:<8}", "Fcst");
    for q in q_levels {
        print!("{:>8}", format!("q{:.0}", q * 100.0));
    }
    println!();

    for i in 0..2.min(forecasts.len()) {
        print!("{:<8.1}", forecasts[i]);
        let row = quantiles.at_time(i).unwrap();
        for val in row {
            print!("{:>8.2}", val);
        }
        println!();
    }
    println!("  ... ({} showing first 2)", name);
}
