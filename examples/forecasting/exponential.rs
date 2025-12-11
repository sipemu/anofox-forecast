//! Exponential Smoothing Models example.
//!
//! Run with: cargo run --example exponential

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, ETSSeasonalType, ETSSpec, ErrorType, HoltLinearTrend, HoltWinters,
    SeasonalType, SimpleExponentialSmoothing, TrendType, ETS,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Exponential Smoothing Models Example ===\n");

    // Generate sample data with trend and seasonality
    let timestamps: Vec<_> = (0..72)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    let values: Vec<f64> = (0..72)
        .map(|i| {
            let base = 50.0;
            let trend = 0.3 * i as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = 0.5 * ((i as f64 * 0.7).cos());
            base + trend + seasonal + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
    println!("Data: {} observations (6 seasons of period 12)\n", ts.len());

    // 1. Simple Exponential Smoothing
    println!("--- Simple Exponential Smoothing (SES) ---");

    // Fixed alpha
    let mut ses_fixed = SimpleExponentialSmoothing::new(0.3);
    ses_fixed.fit(&ts).unwrap();
    println!("Fixed alpha = 0.3");
    println!("Final level: {:.4}", ses_fixed.level().unwrap());

    // Auto-optimized alpha
    let mut ses_auto = SimpleExponentialSmoothing::auto();
    ses_auto.fit(&ts).unwrap();
    println!("\nOptimized alpha: {:.4}", ses_auto.alpha().unwrap());
    println!("Final level: {:.4}", ses_auto.level().unwrap());

    let ses_forecast = ses_auto.predict(5).unwrap();
    println!("\nSES Forecast (flat):");
    for (i, pred) in ses_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 2. Holt's Linear Trend
    println!("\n--- Holt's Linear Trend ---");

    // Create data without seasonality for Holt
    let trend_values: Vec<f64> = (0..50)
        .map(|i| 10.0 + 2.0 * i as f64 + (i as f64 * 0.2).sin())
        .collect();
    let trend_timestamps: Vec<_> = (0..50)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();
    let trend_ts = TimeSeries::univariate(trend_timestamps, trend_values).unwrap();

    // Auto-optimized
    let mut holt = HoltLinearTrend::auto();
    holt.fit(&trend_ts).unwrap();
    println!("Optimized parameters:");
    println!("  alpha: {:.4}", holt.alpha().unwrap());
    println!("  beta:  {:.4}", holt.beta().unwrap());
    println!(
        "Level: {:.4}, Trend: {:.4}",
        holt.level().unwrap(),
        holt.trend().unwrap()
    );

    let holt_forecast = holt.predict(5).unwrap();
    println!("\nHolt Forecast (trending):");
    for (i, pred) in holt_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // Damped trend
    println!("\n--- Holt's Damped Trend ---");
    let mut holt_damped = HoltLinearTrend::auto_damped();
    holt_damped.fit(&trend_ts).unwrap();
    println!("Damping parameter phi: {:.4}", holt_damped.phi().unwrap());

    let damped_forecast = holt_damped.predict(10).unwrap();
    let undamped_forecast = holt.predict(10).unwrap();

    println!("\nComparison at h=10:");
    println!("  Undamped: {:.4}", undamped_forecast.primary()[9]);
    println!("  Damped:   {:.4}", damped_forecast.primary()[9]);

    // 3. Holt-Winters
    println!("\n--- Holt-Winters (Triple Exponential Smoothing) ---");

    // Additive seasonality
    let mut hw_add = HoltWinters::auto(12, SeasonalType::Additive);
    hw_add.fit(&ts).unwrap();
    println!("Additive seasonality (period=12):");
    println!("  alpha: {:.4}", hw_add.alpha().unwrap());
    println!("  beta:  {:.4}", hw_add.beta().unwrap());
    println!("  gamma: {:.4}", hw_add.gamma().unwrap());

    // Multiplicative seasonality
    // Generate multiplicative seasonal data
    let mult_values: Vec<f64> = (0..72)
        .map(|i| {
            let base = 100.0 + 0.5 * i as f64;
            let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            base * seasonal
        })
        .collect();
    let mult_ts = TimeSeries::univariate(timestamps.clone(), mult_values).unwrap();

    let mut hw_mult = HoltWinters::auto(12, SeasonalType::Multiplicative);
    hw_mult.fit(&mult_ts).unwrap();
    println!("\nMultiplicative seasonality (period=12):");
    println!("  alpha: {:.4}", hw_mult.alpha().unwrap());
    println!("  beta:  {:.4}", hw_mult.beta().unwrap());
    println!("  gamma: {:.4}", hw_mult.gamma().unwrap());

    // Seasonal indices
    println!("\nSeasonal indices (additive):");
    if let Some(seasonals) = hw_add.seasonals() {
        for (i, s) in seasonals.iter().enumerate() {
            print!("  S{}: {:+.2}", i + 1, s);
            if (i + 1) % 4 == 0 {
                println!();
            }
        }
    }

    // Forecast with intervals
    let hw_forecast = hw_add.predict_with_intervals(12, 0.95).unwrap();
    println!("\nHolt-Winters Forecast (1 season ahead):");
    let preds = hw_forecast.primary();
    let lower = hw_forecast.lower_series(0).unwrap();
    let upper = hw_forecast.upper_series(0).unwrap();

    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("{:-<38}", "");
    for i in 0..12 {
        println!(
            "{:>4} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            lower[i],
            preds[i],
            upper[i]
        );
    }

    // 4. ETS State-Space Framework
    println!("\n--- ETS State-Space Framework ---");

    // ETS(A,A,A) - Additive error, trend, and seasonality
    let spec = ETSSpec::new(
        ErrorType::Additive,
        TrendType::Additive,
        ETSSeasonalType::Additive,
    );
    let mut ets = ETS::new(spec, 12);
    ets.fit(&ts).unwrap();
    println!("ETS(A,A,A) with period=12");
    println!("  alpha: {:.4}", ets.alpha().unwrap_or(0.0));
    println!("  beta:  {:.4}", ets.beta().unwrap_or(0.0));
    println!("  gamma: {:.4}", ets.gamma().unwrap_or(0.0));

    // 5. AutoETS - Automatic model selection
    println!("\n--- AutoETS (Automatic Model Selection) ---");

    let config = AutoETSConfig::with_period(12);
    let mut auto_ets = AutoETS::with_config(config);
    auto_ets.fit(&ts).unwrap();

    if let Some(spec) = auto_ets.selected_spec() {
        println!(
            "Selected model: ETS({:?}, {:?}, {:?})",
            spec.error, spec.trend, spec.seasonal
        );
    }

    println!("\nTop 5 models by AIC:");
    for (i, (spec, aic)) in auto_ets.model_scores().iter().take(5).enumerate() {
        println!(
            "  {}. ETS({:?},{:?},{:?}): AIC = {:.2}",
            i + 1,
            spec.error,
            spec.trend,
            spec.seasonal,
            aic
        );
    }

    let auto_forecast = auto_ets.predict(6).unwrap();
    println!("\nAutoETS Forecast:");
    for (i, pred) in auto_forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 6. Model comparison
    println!("\n--- Model Comparison ---");
    println!("{:<25} {:>15}", "Model", "Last Prediction");
    println!("{:-<42}", "");

    let ses_pred = ses_auto.predict(1).unwrap().primary()[0];
    let holt_pred = holt.predict(1).unwrap().primary()[0];
    let hw_pred = hw_add.predict(1).unwrap().primary()[0];
    let auto_pred = auto_ets.predict(1).unwrap().primary()[0];

    println!("{:<25} {:>15.4}", "SES", ses_pred);
    println!("{:<25} {:>15.4}", "Holt Linear", holt_pred);
    println!("{:<25} {:>15.4}", "Holt-Winters", hw_pred);
    println!("{:<25} {:>15.4}", "AutoETS", auto_pred);
    println!(
        "{:<25} {:>15.4}",
        "Actual last value",
        values.last().unwrap()
    );

    println!("\n=== Exponential Smoothing Example Complete ===");
}
