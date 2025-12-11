//! Accuracy Metrics Example
//!
//! This example demonstrates how to calculate and interpret various
//! forecast accuracy metrics using the anofox-forecast crate.
//!
//! Run with: cargo run --example metrics

use anofox_forecast::utils::metrics::{calculate_metrics, mae, mse, rmse, smape};

fn main() {
    println!("=== Accuracy Metrics Example ===\n");

    // Sample actual and predicted values
    let actual = vec![100.0, 110.0, 120.0, 115.0, 125.0, 130.0, 128.0, 135.0];
    let predicted = vec![102.0, 108.0, 118.0, 117.0, 123.0, 132.0, 126.0, 138.0];

    println!("Actual values:    {:?}", actual);
    println!("Predicted values: {:?}", predicted);
    println!();

    // =========================================================================
    // Calculate all metrics at once
    // =========================================================================
    println!("--- Comprehensive Metrics ---\n");

    match calculate_metrics(&actual, &predicted, None) {
        Ok(metrics) => {
            println!("MAE (Mean Absolute Error):     {:.4}", metrics.mae);
            println!("MSE (Mean Squared Error):      {:.4}", metrics.mse);
            println!("RMSE (Root Mean Squared Error): {:.4}", metrics.rmse);
            println!("SMAPE (Symmetric MAPE):        {:.4}%", metrics.smape);
            println!(
                "R² (Coefficient of Determination): {:.4}",
                metrics.r_squared
            );

            if let Some(mape) = metrics.mape {
                println!("MAPE (Mean Absolute % Error):  {:.4}%", mape);
            }

            if let Some(mase) = metrics.mase {
                println!("MASE (Mean Absolute Scaled Error): {:.4}", mase);
            }
        }
        Err(e) => println!("Error calculating metrics: {}", e),
    }

    // =========================================================================
    // Individual metric functions
    // =========================================================================
    println!("\n--- Individual Metric Functions ---\n");

    println!(
        "mae(&actual, &predicted)   = {:.4}",
        mae(&actual, &predicted)
    );
    println!(
        "mse(&actual, &predicted)   = {:.4}",
        mse(&actual, &predicted)
    );
    println!(
        "rmse(&actual, &predicted)  = {:.4}",
        rmse(&actual, &predicted)
    );
    println!(
        "smape(&actual, &predicted) = {:.4}%",
        smape(&actual, &predicted)
    );

    // =========================================================================
    // Metrics with seasonal period (for MASE)
    // =========================================================================
    println!("\n--- MASE with Seasonal Period ---\n");

    // Seasonal data (period = 4)
    let seasonal_actual = vec![
        10.0, 20.0, 15.0, 25.0, // Season 1
        12.0, 22.0, 17.0, 27.0, // Season 2
        14.0, 24.0, 19.0, 29.0, // Season 3
    ];
    let seasonal_predicted = vec![
        11.0, 19.0, 16.0, 24.0, // Season 1
        13.0, 21.0, 18.0, 26.0, // Season 2
        15.0, 23.0, 20.0, 28.0, // Season 3
    ];

    println!("Seasonal data (period=4):");
    println!("Actual:    {:?}", seasonal_actual);
    println!("Predicted: {:?}", seasonal_predicted);

    match calculate_metrics(&seasonal_actual, &seasonal_predicted, Some(4)) {
        Ok(metrics) => {
            println!("\nMAE:  {:.4}", metrics.mae);
            println!("RMSE: {:.4}", metrics.rmse);
            if let Some(mase) = metrics.mase {
                println!("MASE (seasonal period=4): {:.4}", mase);
                println!("\nInterpretation:");
                if mase < 1.0 {
                    println!("  MASE < 1: Model outperforms seasonal naive baseline");
                } else if mase > 1.0 {
                    println!("  MASE > 1: Seasonal naive baseline is better");
                } else {
                    println!("  MASE = 1: Model equals seasonal naive baseline");
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // Comparing model performance
    // =========================================================================
    println!("\n--- Comparing Multiple Models ---\n");

    let actual = vec![100.0, 105.0, 110.0, 108.0, 115.0, 120.0];

    // Model A: Good predictions
    let model_a = vec![101.0, 104.0, 111.0, 107.0, 116.0, 119.0];

    // Model B: Moderate predictions
    let model_b = vec![98.0, 108.0, 107.0, 112.0, 112.0, 123.0];

    // Model C: Poor predictions
    let model_c = vec![95.0, 115.0, 100.0, 118.0, 105.0, 130.0];

    println!("Actual values: {:?}\n", actual);

    let models = [
        ("Model A", &model_a),
        ("Model B", &model_b),
        ("Model C", &model_c),
    ];

    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>10}",
        "Model", "MAE", "RMSE", "SMAPE%", "R²"
    );
    println!("{}", "-".repeat(48));

    for (name, predictions) in &models {
        if let Ok(m) = calculate_metrics(&actual, predictions, None) {
            println!(
                "{:<10} {:>8.3} {:>8.3} {:>8.2} {:>10.4}",
                name, m.mae, m.rmse, m.smape, m.r_squared
            );
        }
    }

    println!("\nRanking (lower is better for MAE/RMSE/SMAPE, higher for R²):");
    println!("  Best model by MAE: Model A");

    // =========================================================================
    // Handling edge cases
    // =========================================================================
    println!("\n--- Edge Cases ---\n");

    // Perfect predictions
    let perfect_actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let perfect_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    if let Ok(m) = calculate_metrics(&perfect_actual, &perfect_pred, None) {
        println!("Perfect predictions:");
        println!(
            "  MAE = {:.1}, RMSE = {:.1}, R² = {:.1}",
            m.mae, m.rmse, m.r_squared
        );
    }

    // Data with zeros (MAPE undefined)
    let with_zeros_actual = vec![0.0, 1.0, 2.0, 3.0];
    let with_zeros_pred = vec![0.1, 1.1, 2.1, 3.1];

    if let Ok(m) = calculate_metrics(&with_zeros_actual, &with_zeros_pred, None) {
        println!("\nData containing zeros:");
        println!("  MAE = {:.2}, RMSE = {:.2}", m.mae, m.rmse);
        match m.mape {
            Some(mape) => println!("  MAPE = {:.2}%", mape),
            None => println!("  MAPE = undefined (zeros in actual values)"),
        }
        println!("  SMAPE = {:.2}% (handles zeros gracefully)", m.smape);
    }

    // =========================================================================
    // Metric interpretation guide
    // =========================================================================
    println!("\n--- Metric Interpretation Guide ---\n");

    println!("MAE (Mean Absolute Error):");
    println!("  - Average magnitude of errors in original units");
    println!("  - Easy to interpret, robust to outliers");
    println!();

    println!("RMSE (Root Mean Squared Error):");
    println!("  - Penalizes large errors more heavily than MAE");
    println!("  - Same units as the data");
    println!("  - RMSE >= MAE always (equal when all errors are same)");
    println!();

    println!("MAPE (Mean Absolute Percentage Error):");
    println!("  - Scale-independent (percentage)");
    println!("  - Undefined when actual values contain zeros");
    println!("  - Asymmetric: penalizes under-predictions more");
    println!();

    println!("SMAPE (Symmetric MAPE):");
    println!("  - Bounded between 0% and 200%");
    println!("  - Handles zeros better than MAPE");
    println!("  - More symmetric treatment of over/under predictions");
    println!();

    println!("MASE (Mean Absolute Scaled Error):");
    println!("  - Scale-independent, compares to naive forecast");
    println!("  - MASE < 1: better than naive baseline");
    println!("  - MASE > 1: worse than naive baseline");
    println!("  - Good for comparing across different series");
    println!();

    println!("R² (Coefficient of Determination):");
    println!("  - Proportion of variance explained by the model");
    println!("  - R² = 1: perfect predictions");
    println!("  - R² = 0: model equals predicting the mean");
    println!("  - R² < 0: model worse than predicting the mean");
}
