//! Quickstart example demonstrating basic usage of anofox-forecast.
//!
//! Run with: cargo run --example quickstart

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::calculate_metrics;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== anofox-forecast Quickstart ===\n");

    // 1. Create sample time series data
    let timestamps: Vec<_> = (0..100)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    // Generate data with trend and seasonality
    let values: Vec<f64> = (0..100)
        .map(|i| {
            10.0                           // base level
            + 0.5 * i as f64               // linear trend
            + 5.0 * (i as f64 * 0.2).sin() // seasonal pattern
            + 0.5 * (i as f64 * 0.1).cos() // noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
    println!("Created time series with {} observations", ts.len());

    // 2. Fit an ARIMA(1,1,1) model
    println!("\n--- Fitting ARIMA(1,1,1) model ---");
    let mut model = ARIMA::new(1, 1, 1);
    model.fit(&ts).unwrap();

    println!("AR coefficients: {:?}", model.ar_coefficients());
    println!("MA coefficients: {:?}", model.ma_coefficients());
    println!("Intercept: {:.4}", model.intercept());

    if let Some(aic) = model.aic() {
        println!("AIC: {:.2}", aic);
    }
    if let Some(bic) = model.bic() {
        println!("BIC: {:.2}", bic);
    }

    // 3. Generate point forecasts
    println!("\n--- Point Forecast (10 steps ahead) ---");
    let forecast = model.predict(10).unwrap();
    let predictions = forecast.primary();

    for (i, pred) in predictions.iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 4. Generate forecasts with confidence intervals
    println!("\n--- Forecast with 95% Confidence Intervals ---");
    let forecast_ci = model.predict_with_intervals(10, 0.95).unwrap();
    let lower = forecast_ci.lower_series(0).unwrap();
    let upper = forecast_ci.upper_series(0).unwrap();
    let preds = forecast_ci.primary();

    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("{:-<44}", "");
    for i in 0..10 {
        println!(
            "{:>4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            lower[i],
            preds[i],
            upper[i]
        );
    }

    // 5. Calculate in-sample accuracy metrics
    println!("\n--- In-Sample Accuracy Metrics ---");
    if let Some(fitted) = model.fitted_values() {
        // Use valid fitted values (skip NaN at the beginning)
        let valid_start = fitted.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let fitted_valid: Vec<f64> = fitted[valid_start..].to_vec();
        let actual_valid: Vec<f64> = values[valid_start..].to_vec();

        if fitted_valid.len() == actual_valid.len() && !fitted_valid.is_empty() {
            match calculate_metrics(&actual_valid, &fitted_valid, None) {
                Ok(metrics) => {
                    println!("MAE:       {:.4}", metrics.mae);
                    println!("RMSE:      {:.4}", metrics.rmse);
                    println!("SMAPE:     {:.4}%", metrics.smape);
                    if let Some(mape) = metrics.mape {
                        println!("MAPE:      {:.4}%", mape);
                    }
                    println!("R-squared: {:.4}", metrics.r_squared);
                }
                Err(e) => println!("Could not calculate metrics: {}", e),
            }
        }
    }

    // 6. Examine residuals
    println!("\n--- Residual Analysis ---");
    if let Some(residuals) = model.residuals() {
        let valid_residuals: Vec<f64> = residuals.iter().filter(|r| !r.is_nan()).copied().collect();
        let mean_resid = valid_residuals.iter().sum::<f64>() / valid_residuals.len() as f64;
        let var_resid = valid_residuals
            .iter()
            .map(|r| (r - mean_resid).powi(2))
            .sum::<f64>()
            / valid_residuals.len() as f64;

        println!("Number of residuals: {}", valid_residuals.len());
        println!("Mean residual:       {:.6}", mean_resid);
        println!("Residual variance:   {:.6}", var_resid);
        println!("Residual std dev:    {:.6}", var_resid.sqrt());
    }

    println!("\n=== Quickstart Complete ===");
}
