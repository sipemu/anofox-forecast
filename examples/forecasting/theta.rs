//! Theta Method example.
//!
//! Run with: cargo run --example theta

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Theta Method Example ===\n");

    println!("The Theta method won the M3 Forecasting Competition.");
    println!("It decomposes a series using 'theta lines' and combines forecasts.\n");

    // Generate sample data with trend
    let timestamps: Vec<_> = (0..60)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    let values: Vec<f64> = (0..60)
        .map(|i| {
            10.0 + 0.8 * i as f64  // trend
                + 2.0 * (i as f64 * 0.3).sin()  // cyclical
                + 0.5 * (i as f64 * 0.1).cos() // noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
    println!("Data: {} observations with trend\n", ts.len());

    // 1. Standard Theta Method (theta=2)
    println!("--- Standard Theta Method (theta=2) ---");
    let mut theta = Theta::new();
    theta.fit(&ts).unwrap();

    println!("Parameters:");
    println!("  theta: {:.2}", theta.theta());
    println!("  alpha (SES): {:.4}", theta.alpha().unwrap());
    println!("  drift (trend slope): {:.4}", theta.drift().unwrap());

    let forecast = theta.predict(10).unwrap();
    println!("\nForecasts:");
    for (i, pred) in forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 2. Theta with custom parameter
    println!("\n--- Theta with Custom Parameter ---");

    let theta_values = [0.5, 1.0, 1.5, 2.0, 3.0];
    println!(
        "{:<10} {:>15} {:>15}",
        "theta", "h=1 forecast", "h=10 forecast"
    );
    println!("{:-<42}", "");

    for &theta_val in &theta_values {
        let mut model = Theta::with_theta(theta_val);
        model.fit(&ts).unwrap();
        let fc = model.predict(10).unwrap();
        println!(
            "{:<10} {:>15.4} {:>15.4}",
            theta_val,
            fc.primary()[0],
            fc.primary()[9]
        );
    }

    // 3. Theta with fixed alpha
    println!("\n--- Theta with Fixed Alpha ---");

    let alphas = [0.1, 0.3, 0.5, 0.7, 0.9];
    println!(
        "{:<10} {:>15} {:>15}",
        "alpha", "h=1 forecast", "h=10 forecast"
    );
    println!("{:-<42}", "");

    for &alpha in &alphas {
        let mut model = Theta::with_alpha(alpha);
        model.fit(&ts).unwrap();
        let fc = model.predict(10).unwrap();
        println!(
            "{:<10} {:>15.4} {:>15.4}",
            alpha,
            fc.primary()[0],
            fc.primary()[9]
        );
    }

    // 4. Seasonal Theta Method
    println!("\n--- Seasonal Theta Method ---");

    // Generate seasonal data
    let seasonal_values: Vec<f64> = (0..72)
        .map(|i| {
            let trend = 0.3 * i as f64;
            let seasonal = 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            20.0 + trend + seasonal
        })
        .collect();
    let seasonal_timestamps: Vec<_> = (0..72)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();
    let seasonal_ts = TimeSeries::univariate(seasonal_timestamps, seasonal_values).unwrap();

    let mut theta_seasonal = Theta::seasonal(12);
    theta_seasonal.fit(&seasonal_ts).unwrap();

    println!("Seasonal period: 12");
    println!("Optimized alpha: {:.4}", theta_seasonal.alpha().unwrap());
    println!("Drift: {:.4}", theta_seasonal.drift().unwrap());

    let seasonal_forecast = theta_seasonal.predict(12).unwrap();
    println!("\nForecast (1 full season):");
    println!("{:>4} {:>12}", "h", "Forecast");
    for (i, pred) in seasonal_forecast.primary().iter().enumerate() {
        println!("{:>4} {:>12.4}", i + 1, pred);
    }

    // 5. Forecast with confidence intervals
    println!("\n--- Forecast with 95% Confidence Intervals ---");
    let mut theta_ci = Theta::new();
    theta_ci.fit(&ts).unwrap();

    let forecast_ci = theta_ci.predict_with_intervals(10, 0.95).unwrap();
    let preds = forecast_ci.primary();
    let lower = forecast_ci.lower_series(0).unwrap();
    let upper = forecast_ci.upper_series(0).unwrap();

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

    // 6. Understanding the Theta decomposition
    println!("\n--- Understanding Theta Decomposition ---");
    println!(
        "
The Theta method works by:

1. Decomposing the series into 'theta lines':
   Z(theta) = theta * Y + (1-theta) * Linear_Trend

2. For Standard Theta Method (theta=2):
   - Theta=2 doubles the local curvature
   - Theta=0 gives the linear trend

3. Forecasting:
   - Apply SES to the theta line (theta=2)
   - Extract linear trend for drift
   - Combine: forecast = SES_level + drift * h

4. The method works well because:
   - SES captures short-term patterns
   - Linear trend captures long-term direction
   - Combination balances both
"
    );

    // 7. Compare with and without trend
    println!("--- Effect of Trend on Theta Forecast ---");

    // Trending data
    let trend_data: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
    let trend_ts = TimeSeries::univariate(
        (0..50)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
            .collect(),
        trend_data.clone(),
    )
    .unwrap();

    // Flat data
    let flat_data: Vec<f64> = vec![50.0; 50];
    let flat_ts = TimeSeries::univariate(
        (0..50)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
            .collect(),
        flat_data.clone(),
    )
    .unwrap();

    let mut theta_trend = Theta::new();
    theta_trend.fit(&trend_ts).unwrap();

    let mut theta_flat = Theta::new();
    theta_flat.fit(&flat_ts).unwrap();

    println!("\nTrending data (slope=2):");
    println!("  Last value: {:.2}", trend_data.last().unwrap());
    println!("  Drift: {:.4}", theta_trend.drift().unwrap());
    let trend_fc = theta_trend.predict(5).unwrap();
    println!("  h=5 forecast: {:.4}", trend_fc.primary()[4]);

    println!("\nFlat data:");
    println!("  Last value: {:.2}", flat_data.last().unwrap());
    println!("  Drift: {:.6}", theta_flat.drift().unwrap());
    let flat_fc = theta_flat.predict(5).unwrap();
    println!("  h=5 forecast: {:.4}", flat_fc.primary()[4]);

    // 8. Fitted values and residuals
    println!("\n--- Residual Analysis ---");
    if let Some(residuals) = theta.residuals() {
        let valid_residuals: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
        let n = valid_residuals.len();
        let mean = valid_residuals.iter().sum::<f64>() / n as f64;
        let variance = valid_residuals
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        println!("Residuals: {} observations", n);
        println!("Mean: {:.6}", mean);
        println!("Std Dev: {:.6}", variance.sqrt());
    }

    println!("\n=== Theta Method Example Complete ===");
}
