//! Cross-Validation Example
//!
//! This example demonstrates time series cross-validation techniques
//! for robust model evaluation using the anofox-forecast crate.
//!
//! Run with: cargo run --example cross_validation

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, SeasonalNaive, SimpleMovingAverage};
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Time Series Cross-Validation Example ===\n");

    // Create sample time series data
    let n = 60; // 60 data points
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    // Generate data with trend and seasonality (period = 12)
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let seasonal = 10.0 * ((i % 12) as f64 * std::f64::consts::PI / 6.0).sin();
            let noise = ((i * 17 + 13) % 11) as f64 - 5.0;
            trend + seasonal + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    println!("Time series: {} observations", ts.len());
    println!("First 10 values: {:?}", &ts.primary_values()[..10]);
    println!();

    // =========================================================================
    // Expanding Window Cross-Validation
    // =========================================================================
    println!("--- Expanding Window CV ---\n");

    println!("Strategy: Training window grows with each fold");
    println!("  Fold 1: Train on [0..30], Test on [30..31]");
    println!("  Fold 2: Train on [0..31], Test on [31..32]");
    println!("  ...");
    println!();

    let expanding_config = CVConfig::expanding(30, 1) // initial_window=30, horizon=1
        .with_step_size(1);

    match cross_validate(&expanding_config, &ts, Naive::new) {
        Ok(results) => {
            println!("Naive model with expanding window:");
            println!("  Number of folds: {}", results.n_folds);
            println!(
                "  Mean MAE:  {:.4} (±{:.4})",
                results.aggregated.mae, results.aggregated.mae_std
            );
            println!(
                "  Mean RMSE: {:.4} (±{:.4})",
                results.aggregated.rmse, results.aggregated.rmse_std
            );
            println!("  Mean SMAPE: {:.2}%", results.aggregated.smape);
        }
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // Rolling Window Cross-Validation
    // =========================================================================
    println!("\n--- Rolling Window CV ---\n");

    println!("Strategy: Fixed-size training window slides forward");
    println!("  Fold 1: Train on [0..30],  Test on [30..31]");
    println!("  Fold 2: Train on [1..31],  Test on [31..32]");
    println!("  ...");
    println!();

    let rolling_config = CVConfig::rolling(30, 1) // window_size=30, horizon=1
        .with_step_size(1);

    match cross_validate(&rolling_config, &ts, Naive::new) {
        Ok(results) => {
            println!("Naive model with rolling window:");
            println!("  Number of folds: {}", results.n_folds);
            println!(
                "  Mean MAE:  {:.4} (±{:.4})",
                results.aggregated.mae, results.aggregated.mae_std
            );
            println!(
                "  Mean RMSE: {:.4} (±{:.4})",
                results.aggregated.rmse, results.aggregated.rmse_std
            );
            println!("  Mean SMAPE: {:.2}%", results.aggregated.smape);
        }
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // Multi-step Horizon
    // =========================================================================
    println!("\n--- Multi-step Forecast Horizon ---\n");

    let horizon = 5;
    let multi_step_config = CVConfig::expanding(30, horizon).with_step_size(5); // Skip every 5 to reduce computation

    println!("Forecasting {} steps ahead at each fold", horizon);
    println!();

    match cross_validate(&multi_step_config, &ts, Naive::new) {
        Ok(results) => {
            println!("Naive model (h={}):", horizon);
            println!("  Number of folds: {}", results.n_folds);
            println!("  Total predictions: {}", results.predicted_values.len());
            println!("  Mean MAE:  {:.4}", results.aggregated.mae);
            println!("  Mean RMSE: {:.4}", results.aggregated.rmse);
        }
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // Comparing Models with CV
    // =========================================================================
    println!("\n--- Model Comparison via CV ---\n");

    let cv_config = CVConfig::expanding(30, 1)
        .with_step_size(2)
        .with_seasonal_period(12);

    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>8}",
        "Model", "MAE", "RMSE", "SMAPE%", "Folds"
    );
    println!("{}", "-".repeat(62));

    // Naive
    if let Ok(r) = cross_validate(&cv_config, &ts, Naive::new) {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.2} {:>8}",
            "Naive", r.aggregated.mae, r.aggregated.rmse, r.aggregated.smape, r.n_folds
        );
    }

    // Seasonal Naive (period=12)
    if let Ok(r) = cross_validate(&cv_config, &ts, || SeasonalNaive::new(12)) {
        println!(
            "{:<20} {:>10.4} {:>10.4} {:>10.2} {:>8}",
            "Seasonal Naive (12)",
            r.aggregated.mae,
            r.aggregated.rmse,
            r.aggregated.smape,
            r.n_folds
        );
    }

    // SMA with different windows
    for window in [3, 5, 7] {
        if let Ok(r) = cross_validate(&cv_config, &ts, || SimpleMovingAverage::new(window)) {
            println!(
                "{:<20} {:>10.4} {:>10.4} {:>10.2} {:>8}",
                format!("SMA ({})", window),
                r.aggregated.mae,
                r.aggregated.rmse,
                r.aggregated.smape,
                r.n_folds
            );
        }
    }

    // =========================================================================
    // Analyzing Per-Fold Results
    // =========================================================================
    println!("\n--- Per-Fold Analysis ---\n");

    let detailed_config = CVConfig::expanding(40, 1).with_step_size(5);

    match cross_validate(&detailed_config, &ts, Naive::new) {
        Ok(results) => {
            println!("Per-fold MAE values:");
            for (i, metrics) in results.fold_metrics.iter().enumerate() {
                let origin = 40 + i * 5;
                println!(
                    "  Fold {:2} (origin={}): MAE={:.4}, RMSE={:.4}",
                    i + 1,
                    origin,
                    metrics.mae,
                    metrics.rmse
                );
            }

            println!("\nAggregated:");
            println!("  Mean MAE: {:.4}", results.aggregated.mae);
            println!("  Std MAE:  {:.4}", results.aggregated.mae_std);

            // Find best and worst folds
            let (best_idx, best) = results
                .fold_metrics
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.mae.partial_cmp(&b.1.mae).unwrap())
                .unwrap();
            let (worst_idx, worst) = results
                .fold_metrics
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.mae.partial_cmp(&b.1.mae).unwrap())
                .unwrap();

            println!("\n  Best fold:  {} (MAE={:.4})", best_idx + 1, best.mae);
            println!("  Worst fold: {} (MAE={:.4})", worst_idx + 1, worst.mae);
        }
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // CV Configuration Options
    // =========================================================================
    println!("\n--- CV Configuration Guide ---\n");

    println!("CVConfig::expanding(initial_window, horizon)");
    println!("  - Training data grows from initial_window");
    println!("  - Better utilizes all historical data");
    println!("  - Later folds have more training data");
    println!();

    println!("CVConfig::rolling(window_size, horizon)");
    println!("  - Fixed training window size");
    println!("  - Tests model on different time periods equally");
    println!("  - Better for detecting concept drift");
    println!();

    println!(".with_step_size(n)");
    println!("  - Step n observations between folds");
    println!("  - Larger step = fewer folds, faster computation");
    println!("  - Smaller step = more folds, better estimates");
    println!();

    println!(".with_seasonal_period(period)");
    println!("  - Sets seasonal period for MASE calculation");
    println!("  - MASE compares to seasonal naive baseline");

    // =========================================================================
    // Best Practices
    // =========================================================================
    println!("\n--- Best Practices ---\n");

    println!("1. Initial window should be large enough for model fitting");
    println!("   - At least 2x seasonal period for seasonal models");
    println!("   - Consider minimum sample requirements of your model");
    println!();

    println!("2. Horizon should match your actual forecasting needs");
    println!("   - Short-term forecasts: horizon = 1-7");
    println!("   - Long-term forecasts: horizon = 30+");
    println!();

    println!("3. Use expanding window for:");
    println!("   - Stable time series without drift");
    println!("   - When more historical data improves forecasts");
    println!();

    println!("4. Use rolling window for:");
    println!("   - Time series with concept drift");
    println!("   - When recent data is more relevant");
    println!();

    println!("5. Report uncertainty:");
    println!("   - Always report MAE ± std across folds");
    println!("   - Check for high variance across folds");
}
