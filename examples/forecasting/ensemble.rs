//! Ensemble Forecasting example.
//!
//! Run with: cargo run --example ensemble

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift, SimpleMovingAverage};
use anofox_forecast::models::ensemble::{CombinationMethod, Ensemble};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::calculate_metrics;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Ensemble Forecasting Example ===\n");

    println!("Ensemble methods combine multiple models for:");
    println!("  - Improved accuracy");
    println!("  - Reduced model uncertainty");
    println!("  - More robust predictions\n");

    // Generate sample data with trend and noise
    let timestamps: Vec<_> = (0..80)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    let values: Vec<f64> = (0..80)
        .map(|i| {
            20.0 + 0.5 * i as f64  // trend
                + 3.0 * (i as f64 * 0.2).sin()  // cycle
                + 1.5 * (i as f64 * 0.1).cos() // noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
    println!("Data: {} observations\n", ts.len());

    // 1. Simple Mean Ensemble
    println!("--- Mean Ensemble ---");
    println!("Combines forecasts using simple average\n");

    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(SimpleMovingAverage::new(5)),
        Box::new(SimpleExponentialSmoothing::auto()),
    ];

    let mut ensemble_mean = Ensemble::new(models);
    ensemble_mean.fit(&ts).unwrap();

    let mean_forecast = ensemble_mean.predict(5).unwrap();
    println!("Models: Naive, SMA(5), SES");
    println!("Method: {}", ensemble_mean.name());
    println!("Weights: {:?}", ensemble_mean.weights());
    println!("\nForecast:");
    for (i, pred) in mean_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 2. Median Ensemble
    println!("\n--- Median Ensemble ---");
    println!("Uses median to reduce impact of outlier forecasts\n");

    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(SimpleMovingAverage::new(5)),
        Box::new(SimpleExponentialSmoothing::auto()),
        Box::new(RandomWalkWithDrift::new()),
        Box::new(HoltLinearTrend::auto()),
    ];

    let mut ensemble_median = Ensemble::new(models).with_method(CombinationMethod::Median);
    ensemble_median.fit(&ts).unwrap();

    let median_forecast = ensemble_median.predict(5).unwrap();
    println!("Models: Naive, SMA(5), SES, RW+Drift, Holt");
    println!("Method: {}", ensemble_median.name());
    println!("\nForecast:");
    for (i, pred) in median_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 3. Weighted MSE Ensemble
    println!("\n--- Weighted MSE Ensemble ---");
    println!("Weights models by inverse of their in-sample MSE\n");

    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(SimpleMovingAverage::new(5)),
        Box::new(SimpleExponentialSmoothing::auto()),
        Box::new(HoltLinearTrend::auto()),
    ];

    let mut ensemble_weighted = Ensemble::new(models).with_method(CombinationMethod::WeightedMSE);
    ensemble_weighted.fit(&ts).unwrap();

    let weighted_forecast = ensemble_weighted.predict(5).unwrap();
    println!("Models: Naive, SMA(5), SES, Holt");
    println!("Method: {}", ensemble_weighted.name());
    println!("\nWeights (by inverse MSE):");
    let model_names = ["Naive", "SMA(5)", "SES", "Holt"];
    for (name, weight) in model_names.iter().zip(ensemble_weighted.weights().iter()) {
        println!("  {}: {:.4} ({:.1}%)", name, weight, weight * 100.0);
    }
    println!("\nForecast:");
    for (i, pred) in weighted_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 4. Custom Weighted Ensemble
    println!("\n--- Custom Weighted Ensemble ---");
    println!("User-specified weights for domain knowledge\n");

    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(HoltLinearTrend::auto()),
        Box::new(Theta::new()),
    ];

    // Assign more weight to Theta (known to perform well)
    let custom_weights = vec![0.1, 0.3, 0.6];

    let mut ensemble_custom = Ensemble::new(models).with_weights(custom_weights);
    ensemble_custom.fit(&ts).unwrap();

    let custom_forecast = ensemble_custom.predict(5).unwrap();
    println!("Models: Naive, Holt, Theta");
    println!("Custom weights: Naive=10%, Holt=30%, Theta=60%");
    println!("\nForecast:");
    for (i, pred) in custom_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 5. Compare Individual vs Ensemble
    println!("\n--- Individual vs Ensemble Comparison ---");

    // Fit individual models
    let mut naive = Naive::new();
    naive.fit(&ts).unwrap();

    let mut sma = SimpleMovingAverage::new(5);
    sma.fit(&ts).unwrap();

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&ts).unwrap();

    let mut holt = HoltLinearTrend::auto();
    holt.fit(&ts).unwrap();

    let mut theta = Theta::new();
    theta.fit(&ts).unwrap();

    // Get h=5 forecasts
    let naive_fc = naive.predict(5).unwrap().primary()[4];
    let sma_fc = sma.predict(5).unwrap().primary()[4];
    let ses_fc = ses.predict(5).unwrap().primary()[4];
    let holt_fc = holt.predict(5).unwrap().primary()[4];
    let theta_fc = theta.predict(5).unwrap().primary()[4];

    println!("\nh=5 Forecast Comparison:");
    println!("{:<20} {:>15}", "Model", "Forecast");
    println!("{:-<37}", "");
    println!("{:<20} {:>15.4}", "Naive", naive_fc);
    println!("{:<20} {:>15.4}", "SMA(5)", sma_fc);
    println!("{:<20} {:>15.4}", "SES", ses_fc);
    println!("{:<20} {:>15.4}", "Holt", holt_fc);
    println!("{:<20} {:>15.4}", "Theta", theta_fc);
    println!("{:-<37}", "");
    println!(
        "{:<20} {:>15.4}",
        "Ensemble (Mean)",
        mean_forecast.primary()[4]
    );
    println!(
        "{:<20} {:>15.4}",
        "Ensemble (Median)",
        median_forecast.primary()[4]
    );
    println!(
        "{:<20} {:>15.4}",
        "Ensemble (Weighted)",
        weighted_forecast.primary()[4]
    );

    // 6. In-Sample Performance
    println!("\n--- In-Sample Performance ---");

    let model_metrics: Vec<(&str, Option<&[f64]>)> = vec![
        ("Naive", naive.fitted_values()),
        ("SMA(5)", sma.fitted_values()),
        ("SES", ses.fitted_values()),
        ("Holt", holt.fitted_values()),
        ("Ensemble (Mean)", ensemble_mean.fitted_values()),
        ("Ensemble (Weighted)", ensemble_weighted.fitted_values()),
    ];

    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "Model", "MAE", "RMSE", "MAPE"
    );
    println!("{:-<52}", "");

    for (name, fitted_opt) in model_metrics {
        if let Some(fitted) = fitted_opt {
            let valid_pairs: Vec<(f64, f64)> = values
                .iter()
                .zip(fitted.iter())
                .filter(|(_, f)| !f.is_nan())
                .map(|(a, f)| (*a, *f))
                .collect();

            if !valid_pairs.is_empty() {
                let (actual, predicted): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();
                if let Ok(metrics) = calculate_metrics(&actual, &predicted, None) {
                    println!(
                        "{:<20} {:>10.4} {:>10.4} {:>10.2}%",
                        name,
                        metrics.mae,
                        metrics.rmse,
                        metrics.mape.unwrap_or(f64::NAN)
                    );
                }
            }
        }
    }

    // 7. Confidence Intervals
    println!("\n--- Ensemble with Confidence Intervals ---");

    let ensemble_ci = ensemble_weighted.predict_with_intervals(10, 0.95).unwrap();
    let preds = ensemble_ci.primary();
    let lower = ensemble_ci.lower_series(0).unwrap();
    let upper = ensemble_ci.upper_series(0).unwrap();

    println!("\n95% Confidence Intervals:");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12}",
        "h", "Lower", "Forecast", "Upper", "Width"
    );
    println!("{:-<56}", "");
    for i in 0..10 {
        let width = upper[i] - lower[i];
        println!(
            "{:>4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            lower[i],
            preds[i],
            upper[i],
            width
        );
    }

    // 8. Best Practices
    println!("\n--- Best Practices for Ensemble Forecasting ---");
    println!(
        "
1. Model Diversity:
   - Combine structurally different models
   - Mix simple (Naive, SMA) with complex (ARIMA, ETS)
   - Include models that capture different patterns

2. Combination Method:
   - Mean: Simple and robust default
   - Median: Robust to outlier forecasts
   - Weighted: When you have performance data

3. Number of Models:
   - 3-7 models typically work well
   - Too few: Limited diversity benefit
   - Too many: Diminishing returns

4. Model Selection:
   - Include at least one trend model (Holt, ARIMA)
   - Include at least one simple benchmark (Naive, SMA)
   - Consider domain-specific models

5. Updating Weights:
   - Re-estimate weights periodically
   - Use recent performance for weighting
   - Consider time-varying weights
"
    );

    println!("=== Ensemble Forecasting Example Complete ===");
}
