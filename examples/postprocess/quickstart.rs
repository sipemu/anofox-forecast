//! Quickstart example for postprocessing module.
//!
//! This example demonstrates the simplest path to converting point forecasts
//! into prediction intervals with uncertainty quantification.
//!
//! Run with: cargo run --example postprocess_quickstart

#![allow(clippy::needless_range_loop)]

use anofox_forecast::postprocess::{PointForecasts, PostProcessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Postprocessing Quickstart ===\n");

    // Step 1: Generate synthetic historical data
    // In practice, these would be your model's past forecasts and actual outcomes
    let n_history = 100;
    let historical_forecasts: Vec<f64> = (0..n_history)
        .map(|i| 10.0 + 0.1 * i as f64 + 0.5 * (i as f64 * 0.1).sin())
        .collect();

    // Actuals have some noise around the forecasts (simulating forecast errors)
    let historical_actuals: Vec<f64> = historical_forecasts
        .iter()
        .enumerate()
        .map(|(i, &f)| f + 0.8 * ((i as f64 * 0.3).cos()) + 0.2)
        .collect();

    println!("Historical data: {} forecast-actual pairs", n_history);
    println!(
        "  Sample forecast: {:.2}, actual: {:.2}",
        historical_forecasts[0], historical_actuals[0]
    );

    // Step 2: Create a PostProcessor with 90% coverage conformal prediction
    let processor = PostProcessor::conformal(0.90);

    // Step 3: Train on historical data to learn the error distribution
    let point_forecasts = PointForecasts::from_values(historical_forecasts);
    let trained = processor.train(&point_forecasts, &historical_actuals)?;

    println!("\nTrained postprocessor on historical errors");

    // Step 4: Generate prediction intervals for new forecasts
    let future_forecasts: Vec<f64> = (0..7).map(|i| 20.0 + 0.1 * i as f64).collect();

    let new_forecasts = PointForecasts::from_values(future_forecasts.clone());
    let intervals = processor.predict_intervals(&trained, &new_forecasts)?;

    // Step 5: Display results
    println!("\n7-Day Forecast with 90% Prediction Intervals:");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10}",
        "Day", "Forecast", "Lower", "Upper", "Width"
    );
    println!("{:-<50}", "");

    for i in 0..intervals.len() {
        let forecast = future_forecasts[i];
        let lower = intervals.lower()[i];
        let upper = intervals.upper()[i];
        let width = upper - lower;

        println!(
            "{:<6} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            forecast,
            lower,
            upper,
            width
        );
    }

    println!("\nThe intervals capture the learned forecast uncertainty.");
    println!("With 90% coverage, approximately 90% of future actuals");
    println!("should fall within these bounds.");

    Ok(())
}
