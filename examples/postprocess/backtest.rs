//! Backtesting for postprocessing methods.
//!
//! This example demonstrates how to use backtesting to evaluate
//! postprocessor performance with rolling/expanding windows and
//! horizon-aware calibration.
//!
//! Run with: cargo run --example postprocess_backtest

use anofox_forecast::postprocess::{BacktestConfig, PointForecasts, PostProcessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Backtesting PostProcessors ===\n");

    // Generate synthetic data with trend and noise
    let n = 200;
    let forecasts: Vec<f64> = (0..n).map(|i| 50.0 + 0.2 * i as f64).collect();
    let actuals: Vec<f64> = forecasts
        .iter()
        .enumerate()
        .map(|(i, &f)| f + 2.0 * ((i as f64 * 0.1).sin()) + 0.5)
        .collect();

    let point_forecasts = PointForecasts::from_values(forecasts);

    println!("Data: {} samples\n", n);

    // === Basic Backtest ===
    println!("--- Basic Backtest (Expanding Window) ---");

    let processor = PostProcessor::conformal(0.90);
    let config = BacktestConfig::new()
        .initial_window(50)
        .step(10)
        .horizon(5)
        .expanding(true);

    let result = processor.backtest(&point_forecasts, &actuals, config)?;

    println!("Configuration:");
    println!("  Initial window: 50");
    println!("  Step: 10");
    println!("  Horizon: 5");
    println!("  Window type: Expanding");
    println!();
    println!("Results:");
    println!("  Number of folds: {}", result.n_folds());
    println!("  Overall coverage: {:.1}%", result.coverage() * 100.0);
    println!("  Average interval width: {:.3}", result.interval_widths());
    println!(
        "  Calibration error (vs 90%): {:.1}%",
        result.calibration_error(0.90) * 100.0
    );

    // Show per-fold summary
    println!("\nPer-fold summary (first 5 folds):");
    println!(
        "{:<6} {:>12} {:>10} {:>12} {:>10}",
        "Fold", "Train Size", "Test Size", "Coverage", "Width"
    );
    println!("{:-<55}", "");

    for fold in result.folds().take(5) {
        println!(
            "{:<6} {:>12} {:>10} {:>11.1}% {:>10.3}",
            fold.fold_idx,
            fold.train_size(),
            fold.test_size(),
            fold.coverage() * 100.0,
            fold.avg_width()
        );
    }
    println!("  ...");

    // === Rolling Window Backtest ===
    println!("\n--- Rolling Window Backtest ---");

    let config_rolling = BacktestConfig::new()
        .initial_window(50)
        .step(10)
        .horizon(5)
        .expanding(false); // Rolling window

    let result_rolling = processor.backtest(&point_forecasts, &actuals, config_rolling)?;

    println!("Results:");
    println!("  Number of folds: {}", result_rolling.n_folds());
    println!(
        "  Overall coverage: {:.1}%",
        result_rolling.coverage() * 100.0
    );
    println!(
        "  Average interval width: {:.3}",
        result_rolling.interval_widths()
    );

    // Verify rolling window sizes are constant
    let fold0 = result_rolling.fold(0).unwrap();
    let fold_last = result_rolling.fold(result_rolling.n_folds() - 1).unwrap();
    println!(
        "  First fold train size: {}, Last fold train size: {}",
        fold0.train_size(),
        fold_last.train_size()
    );

    // === Horizon-Aware Backtest ===
    println!("\n--- Horizon-Aware Backtest ---");

    let config_horizon = BacktestConfig::new()
        .initial_window(50)
        .step(10)
        .horizon(7)
        .expanding(true)
        .horizon_aware(true);

    let result_horizon = processor.backtest(&point_forecasts, &actuals, config_horizon)?;

    println!("Coverage by horizon:");
    let coverage_by_h = result_horizon.coverage_by_horizon();
    let widths_by_h = result_horizon.widths_by_horizon();

    println!("{:<10} {:>12} {:>12}", "Horizon", "Coverage", "Avg Width");
    println!("{:-<36}", "");

    for h in 1..=7 {
        let cov = coverage_by_h.get(&h).unwrap_or(&0.0);
        let width = widths_by_h.get(&h).unwrap_or(&0.0);
        println!("{:<10} {:>11.1}% {:>12.3}", h, cov * 100.0, width);
    }

    println!("\nNote: Coverage/width may vary by horizon due to error growth.");

    // === Train Calibrated Model ===
    println!("\n--- Training Calibrated Model from Backtest ---");

    // Use pooled backtest data to train final model
    let calibrated = result_horizon.calibrated_model(&processor)?;
    println!(
        "Pooled model trained on {} samples",
        result_horizon.pooled_forecasts().len()
    );

    // Predict on new data
    let new_forecasts = PointForecasts::from_values(vec![90.0, 90.5, 91.0, 91.5, 92.0]);
    let intervals = processor.predict_intervals(&calibrated, &new_forecasts)?;

    println!("\nPredictions for new forecasts:");
    println!("{:<10} {:>10} {:>10}", "Forecast", "Lower", "Upper");
    println!("{:-<32}", "");

    for i in 0..5 {
        println!(
            "{:<10.1} {:>10.2} {:>10.2}",
            new_forecasts.values()[i],
            intervals.lower()[i],
            intervals.upper()[i]
        );
    }

    // === Horizon-Specific Models ===
    println!("\n--- Horizon-Specific Calibration ---");

    let horizon_models = result_horizon.calibrated_model_by_horizon(&processor)?;
    println!("Trained {} horizon-specific models", horizon_models.len());
    println!("Horizons: {:?}", horizon_models.horizons());

    // Use horizon-specific models for prediction
    let horizon_intervals =
        processor.predict_intervals_by_horizon(&horizon_models, &new_forecasts)?;

    println!("\nPredictions with horizon-specific calibration:");
    println!(
        "{:<10} {:>10} {:>10} {:>10}",
        "Horizon", "Forecast", "Lower", "Upper"
    );
    println!("{:-<44}", "");

    for i in 0..5 {
        println!(
            "{:<10} {:>10.1} {:>10.2} {:>10.2}",
            i + 1,
            new_forecasts.values()[i],
            horizon_intervals.lower()[i],
            horizon_intervals.upper()[i]
        );
    }

    // === Compare Methods ===
    println!("\n--- Method Comparison via Backtest ---");

    let methods: Vec<(&str, PostProcessor)> = vec![
        ("Conformal", PostProcessor::conformal(0.90)),
        (
            "HistoricalSim",
            PostProcessor::historical_sim(vec![0.05, 0.5, 0.95]),
        ),
        ("Normal", PostProcessor::normal(vec![0.05, 0.5, 0.95])),
    ];

    let config_compare = BacktestConfig::new().initial_window(50).step(10).horizon(5);

    println!(
        "{:<15} {:>12} {:>12} {:>12}",
        "Method", "Coverage", "Width", "Cal Error"
    );
    println!("{:-<54}", "");

    for (name, proc) in &methods {
        let res = proc.backtest(&point_forecasts, &actuals, config_compare.clone())?;
        println!(
            "{:<15} {:>11.1}% {:>12.3} {:>11.1}%",
            name,
            res.coverage() * 100.0,
            res.interval_widths(),
            res.calibration_error(0.90) * 100.0
        );
    }

    println!("\n=== Summary ===");
    println!("- Use BacktestConfig to configure rolling/expanding windows");
    println!("- horizon_aware=true tracks metrics per forecast horizon");
    println!("- calibrated_model() returns pooled model for production");
    println!("- calibrated_model_by_horizon() returns per-horizon models");
    println!("- predict_intervals_by_horizon() applies horizon-specific calibration");

    Ok(())
}
