//! QRA (Quantile Regression Averaging) for ensemble combining.
//!
//! This example demonstrates how to combine multiple forecasters using QRA
//! to produce calibrated probabilistic forecasts with learned weights.
//!
//! Run with: cargo run --example postprocess_qra_ensemble

#![allow(clippy::needless_range_loop)]
use anofox_forecast::postprocess::QRAPredictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QRA Ensemble Combining ===\n");

    // Simulate 3 different forecasters with different biases
    let n = 100;
    let actuals: Vec<f64> = (0..n).map(|i| 50.0 + 0.3 * i as f64).collect();

    // Forecaster 1: Slight negative bias (underestimates)
    let forecaster1: Vec<f64> = actuals.iter().map(|&a| a - 1.5).collect();

    // Forecaster 2: Slight positive bias (overestimates)
    let forecaster2: Vec<f64> = actuals.iter().map(|&a| a + 2.0).collect();

    // Forecaster 3: Unbiased but noisier
    let forecaster3: Vec<f64> = actuals
        .iter()
        .enumerate()
        .map(|(i, &a)| a + 3.0 * ((i as f64 * 0.2).sin()))
        .collect();

    // Create forecast matrix: each row is [f1, f2, f3] for one time point
    let forecasts_matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![forecaster1[i], forecaster2[i], forecaster3[i]])
        .collect();

    println!("Training data: {} samples, 3 forecasters", n);
    println!("\nForecaster characteristics:");
    println!("  F1: Negative bias (-1.5)");
    println!("  F2: Positive bias (+2.0)");
    println!("  F3: Unbiased, noisy (±3.0)\n");

    // Calculate simple average error for comparison
    let avg_errors: Vec<f64> = (0..n)
        .map(|i| {
            let avg = (forecaster1[i] + forecaster2[i] + forecaster3[i]) / 3.0;
            (actuals[i] - avg).abs()
        })
        .collect();
    let mae_avg = avg_errors.iter().sum::<f64>() / n as f64;

    println!("Simple average MAE: {:.3}", mae_avg);

    // Define quantile levels
    let quantiles = vec![0.1, 0.5, 0.9];

    // Method 1: Standard QRA (no regularization)
    println!("\n--- Standard QRA ---");
    let qra = QRAPredictor::standard(quantiles.clone());
    let qra_result = qra.fit(&forecasts_matrix, &actuals)?;

    println!("Learned coefficients (per quantile):");
    for (i, q) in quantiles.iter().enumerate() {
        if let Some(coeffs) = qra_result.coefficients(i) {
            println!(
                "  q={:.1}: intercept={:.3}, F1={:.3}, F2={:.3}, F3={:.3}",
                q,
                coeffs.first().unwrap_or(&0.0),
                coeffs.get(1).unwrap_or(&0.0),
                coeffs.get(2).unwrap_or(&0.0),
                coeffs.get(3).unwrap_or(&0.0)
            );
        }
    }

    // Method 2: Lasso QRA (sparse weights)
    println!("\n--- Lasso QRA (lambda=0.1) ---");
    let qra_lasso = QRAPredictor::lasso(quantiles.clone(), 0.1);
    let lasso_result = qra_lasso.fit(&forecasts_matrix, &actuals)?;

    println!("Learned coefficients (per quantile):");
    for (i, q) in quantiles.iter().enumerate() {
        if let Some(coeffs) = lasso_result.coefficients(i) {
            println!(
                "  q={:.1}: intercept={:.3}, F1={:.3}, F2={:.3}, F3={:.3}",
                q,
                coeffs.first().unwrap_or(&0.0),
                coeffs.get(1).unwrap_or(&0.0),
                coeffs.get(2).unwrap_or(&0.0),
                coeffs.get(3).unwrap_or(&0.0)
            );
        }
    }

    // Generate predictions for new data
    let future_actuals: Vec<f64> = (100..110).map(|i| 50.0 + 0.3 * i as f64).collect();
    let future_f1: Vec<f64> = future_actuals.iter().map(|&a| a - 1.5).collect();
    let future_f2: Vec<f64> = future_actuals.iter().map(|&a| a + 2.0).collect();
    let future_f3: Vec<f64> = future_actuals
        .iter()
        .enumerate()
        .map(|(i, &a)| a + 3.0 * (((100 + i) as f64 * 0.2).sin()))
        .collect();

    let future_matrix: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![future_f1[i], future_f2[i], future_f3[i]])
        .collect();

    let qra_quantiles = qra.predict(&qra_result, &future_matrix)?;

    // Display predictions
    println!("\n=== Future Predictions ===");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Step", "Actual", "F1", "F2", "F3", "q10", "q50", "q90"
    );
    println!("{:-<70}", "");

    for i in 0..5 {
        let q_row = qra_quantiles.at_time(i).unwrap();
        println!(
            "{:<6} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            i + 1,
            future_actuals[i],
            future_f1[i],
            future_f2[i],
            future_f3[i],
            q_row[0],
            q_row[1],
            q_row[2]
        );
    }
    println!("  ... (showing first 5 of 10)");

    // Check coverage and interval width
    let mut covered = 0;
    let mut total_width = 0.0;

    for i in 0..10 {
        let q_row = qra_quantiles.at_time(i).unwrap();
        let lower = q_row[0];
        let upper = q_row[2];
        let actual = future_actuals[i];

        if actual >= lower && actual <= upper {
            covered += 1;
        }
        total_width += upper - lower;
    }

    println!("\n=== Evaluation ===");
    println!(
        "80% interval coverage: {}/10 = {:.0}%",
        covered,
        covered as f64 * 10.0
    );
    println!("Average interval width: {:.2}", total_width / 10.0);

    println!("\nNotes:");
    println!("- QRA learns optimal weights to combine forecasters");
    println!("- Different weights per quantile level");
    println!("- Lasso regularization can zero out poor forecasters");

    Ok(())
}
