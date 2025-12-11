//! Baseline Forecasting Models example.
//!
//! Run with: cargo run --example baseline

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{
    Naive, RandomWalkWithDrift, SeasonalNaive, SeasonalWindowAverage, SimpleMovingAverage,
};
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::calculate_metrics;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Baseline Forecasting Models Example ===\n");

    // Generate sample data with trend and seasonality
    let timestamps: Vec<_> = (0..48)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    let values: Vec<f64> = (0..48)
        .map(|i| {
            let trend = 0.5 * i as f64;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            20.0 + trend + seasonal
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
    println!(
        "Data: {} observations with trend and seasonality (period=12)\n",
        ts.len()
    );
    println!("Last 5 values: {:?}\n", &values[43..]);

    // 1. Naive Forecast
    println!("--- Naive Forecast ---");
    println!("Method: Use last observed value for all forecasts");

    let mut naive = Naive::new();
    naive.fit(&ts).unwrap();

    let naive_forecast = naive.predict(5).unwrap();
    println!("Forecasts: {:?}", naive_forecast.primary());

    let naive_ci = naive.predict_with_intervals(5, 0.95).unwrap();
    println!("\nWith 95% CI:");
    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "h", "Lower", "Forecast", "Upper"
    );
    for i in 0..5 {
        println!(
            "{:>4} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            naive_ci.lower_series(0).unwrap()[i],
            naive_ci.primary()[i],
            naive_ci.upper_series(0).unwrap()[i]
        );
    }

    // 2. Random Walk with Drift
    println!("\n--- Random Walk with Drift ---");
    println!("Method: Last value + h * average_change");

    let mut rw = RandomWalkWithDrift::new();
    rw.fit(&ts).unwrap();

    println!("Estimated drift: {:.4}", rw.drift().unwrap());

    let rw_forecast = rw.predict(5).unwrap();
    println!("\nForecasts:");
    for (i, pred) in rw_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4} (last + {} * drift)", i + 1, pred, i + 1);
    }

    // 3. Seasonal Naive
    println!("\n--- Seasonal Naive ---");
    println!("Method: Use value from same season last period");

    let mut snaive = SeasonalNaive::new(12);
    snaive.fit(&ts).unwrap();

    let snaive_forecast = snaive.predict(12).unwrap();
    println!("\nForecasts (1 full season):");
    println!("{:>4} {:>12} {:>20}", "h", "Forecast", "From (t-12)");
    for i in 0..12 {
        let source_idx = 48 - 12 + i;
        println!(
            "{:>4} {:>12.4} {:>20.4}",
            i + 1,
            snaive_forecast.primary()[i],
            values[source_idx]
        );
    }

    // 4. Simple Moving Average
    println!("\n--- Simple Moving Average ---");

    for window in [3, 6, 12] {
        let mut sma = SimpleMovingAverage::new(window);
        sma.fit(&ts).unwrap();

        let sma_forecast = sma.predict(3).unwrap();
        println!(
            "SMA({}) - Forecasts: {:.2}, {:.2}, {:.2}",
            window,
            sma_forecast.primary()[0],
            sma_forecast.primary()[1],
            sma_forecast.primary()[2]
        );
    }

    // Show the calculation for SMA(3)
    let last_3_avg = values[45..].iter().sum::<f64>() / 3.0;
    println!(
        "\nSMA(3) calculation: ({:.2} + {:.2} + {:.2}) / 3 = {:.4}",
        values[45], values[46], values[47], last_3_avg
    );

    // 5. Seasonal Window Average
    println!("\n--- Seasonal Window Average ---");
    println!("Method: Average of values from the same season across multiple periods");

    let mut swa = SeasonalWindowAverage::new(12, 3); // period=12, use 3 seasons
    swa.fit(&ts).unwrap();

    let swa_forecast = swa.predict(12).unwrap();
    println!("\nForecasts (averaging same season from last 3 periods):");
    println!("{:>4} {:>12}", "h", "Forecast");
    for i in 0..12 {
        println!("{:>4} {:>12.4}", i + 1, swa_forecast.primary()[i]);
    }

    // 6. In-sample comparison
    println!("\n--- In-Sample Performance Comparison ---");

    // Get fitted values and calculate metrics for each model
    let models: Vec<(&str, Box<dyn Forecaster>)> = vec![
        ("Naive", Box::new(Naive::new())),
        ("RW with Drift", Box::new(RandomWalkWithDrift::new())),
        ("Seasonal Naive", Box::new(SeasonalNaive::new(12))),
        ("SMA(6)", Box::new(SimpleMovingAverage::new(6))),
    ];

    println!("{:<18} {:>10} {:>10} {:>10}", "Model", "MAE", "RMSE", "R²");
    println!("{:-<50}", "");

    for (name, mut model) in models {
        model.fit(&ts).unwrap();

        if let Some(fitted) = model.fitted_values() {
            // Find valid (non-NaN) fitted values
            let valid_pairs: Vec<(f64, f64)> = values
                .iter()
                .zip(fitted.iter())
                .filter(|(_, f)| !f.is_nan())
                .map(|(a, f)| (*a, *f))
                .collect();

            if !valid_pairs.is_empty() {
                let (actual, predicted): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();
                if let Ok(metrics) = calculate_metrics(&actual, &predicted, Some(12)) {
                    println!(
                        "{:<18} {:>10.4} {:>10.4} {:>10.4}",
                        name, metrics.mae, metrics.rmse, metrics.r_squared
                    );
                }
            }
        }
    }

    // 7. Residual analysis for Random Walk with Drift
    println!("\n--- Residual Analysis (Random Walk with Drift) ---");
    if let Some(residuals) = rw.residuals() {
        let valid_residuals: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
        let n = valid_residuals.len();
        let mean = valid_residuals.iter().sum::<f64>() / n as f64;
        let variance = valid_residuals
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        println!("Number of residuals: {}", n);
        println!("Mean: {:.6}", mean);
        println!("Variance: {:.6}", variance);
        println!("Std Dev: {:.6}", variance.sqrt());

        // First 5 and last 5 residuals
        println!(
            "\nFirst 5 residuals: {:?}",
            valid_residuals
                .iter()
                .take(5)
                .map(|r| format!("{:.2}", r))
                .collect::<Vec<_>>()
        );
        println!(
            "Last 5 residuals: {:?}",
            valid_residuals
                .iter()
                .rev()
                .take(5)
                .map(|r| format!("{:.2}", r))
                .collect::<Vec<_>>()
        );
    }

    // 8. When to use each method
    println!("\n--- When to Use Each Baseline Method ---");
    println!(
        "
Naive:             Data with no trend or seasonality
                   Quick benchmark for any forecasting problem

Random Walk:       Data with trend but no seasonality
                   Financial time series, stock prices

Seasonal Naive:    Strong seasonal pattern
                   Monthly/quarterly data with clear cycles

SMA:               Smooth out noise, no trend
                   Stationary data with fluctuations

Seasonal Window:   Seasonal pattern with noise
                   Average across multiple seasons for stability
"
    );

    println!("=== Baseline Models Example Complete ===");
}
