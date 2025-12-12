//! ARIMA and AutoARIMA example.
//!
//! Run with: cargo run --example arima

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{difference, integrate, AutoARIMA, ARIMA};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== ARIMA Models Example ===\n");

    // Generate sample data with trend
    let timestamps: Vec<_> = (0..100)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    // Generate data with trend, seasonality, and meaningful noise
    // Using deterministic pseudo-random noise for reproducibility
    let values: Vec<f64> = (0..100)
        .map(|i| {
            let trend = 10.0 + 0.5 * i as f64;
            let seasonal = 2.0 * (i as f64 * 0.3).sin();
            // Add meaningful noise (deterministic but irregular)
            let noise = ((i * 17 + 7) % 13) as f64 - 6.0; // Range roughly -6 to +6
            trend + seasonal + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
    println!(
        "Data: {} observations with trend and cyclical pattern\n",
        ts.len()
    );

    // 1. ARIMA(1,1,1) - Standard specification
    println!("--- ARIMA(1,1,1) ---");
    let mut arima = ARIMA::new(1, 1, 1);
    arima.fit(&ts).unwrap();

    println!(
        "Specification: ({}, {}, {})",
        arima.spec().p,
        arima.spec().d,
        arima.spec().q
    );
    println!("AR coefficients: {:?}", arima.ar_coefficients());
    println!("MA coefficients: {:?}", arima.ma_coefficients());
    println!("Intercept: {:.4}", arima.intercept());

    if let Some(aic) = arima.aic() {
        println!("AIC: {:.2}", aic);
    }
    if let Some(bic) = arima.bic() {
        println!("BIC: {:.2}", bic);
    }

    let forecast = arima.predict(5).unwrap();
    println!("\nForecasts:");
    for (i, pred) in forecast.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    // 2. AR(1) model
    println!("\n--- AR(1) Model ---");
    let mut ar1 = ARIMA::ar(1);
    ar1.fit(&ts).unwrap();
    println!("AR coefficient: {:.4}", ar1.ar_coefficients()[0]);
    if let Some(aic) = ar1.aic() {
        println!("AIC: {:.2}", aic);
    }

    // 3. MA(1) model
    println!("\n--- MA(1) Model ---");
    let mut ma1 = ARIMA::ma(1);
    ma1.fit(&ts).unwrap();
    println!("MA coefficient: {:.4}", ma1.ma_coefficients()[0]);
    if let Some(aic) = ma1.aic() {
        println!("AIC: {:.2}", aic);
    }

    // 4. AutoARIMA - Automatic order selection
    println!("\n--- AutoARIMA (Stepwise) ---");
    let mut auto_arima = AutoARIMA::new();
    auto_arima.fit(&ts).unwrap();

    if let Some((p, d, q)) = auto_arima.selected_order() {
        println!("Selected order: ARIMA({}, {}, {})", p, d, q);
    }

    println!("\nTop 5 model scores (AIC):");
    for (i, (order, score)) in auto_arima.model_scores().iter().take(5).enumerate() {
        if order.is_seasonal() {
            println!(
                "  {}. SARIMA({},{},{})({},-,{})[{}]: AIC = {:.2}",
                i + 1,
                order.p,
                order.d,
                order.q,
                order.cap_p,
                order.cap_q,
                order.s,
                score
            );
        } else {
            println!(
                "  {}. ARIMA({}, {}, {}): AIC = {:.2}",
                i + 1,
                order.p,
                order.d,
                order.q,
                score
            );
        }
    }

    let auto_forecast = auto_arima.predict_with_intervals(10, 0.95).unwrap();
    println!("\nForecast with 95% CI:");
    let preds = auto_forecast.primary();
    let lower = auto_forecast.lower_series(0).unwrap();
    let upper = auto_forecast.upper_series(0).unwrap();

    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("{:-<38}", "");
    for i in 0..10 {
        println!(
            "{:>4} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            lower[i],
            preds[i],
            upper[i]
        );
    }

    // 5. AutoARIMA model scores
    println!("\n--- AutoARIMA Model Scores ---");
    println!("Models evaluated: {}", auto_arima.model_scores().len());
    println!("\nTop 3 models by AIC:");
    for (i, (order, aic)) in auto_arima.model_scores().iter().take(3).enumerate() {
        println!(
            "  {}. ARIMA({}, {}, {}): AIC = {:.2}",
            i + 1,
            order.p,
            order.d,
            order.q,
            aic
        );
    }

    // 6. Differencing utilities
    println!("\n--- Differencing Utilities ---");
    let original = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    println!("Original series: {:?}", original);

    let diff1 = difference(&original, 1);
    println!("First difference: {:?}", diff1);

    let diff2 = difference(&original, 2);
    println!("Second difference: {:?}", diff2);

    // Integrate back
    let restored = integrate(&diff1, &original, 1);
    println!("Integrated (d=1, 3 steps): {:?}", restored);

    // 7. Model comparison
    println!("\n--- Model Comparison (AIC) ---");
    let models: Vec<(&str, Option<f64>)> = vec![
        ("ARIMA(1,1,1)", arima.aic()),
        ("AR(1)", ar1.aic()),
        ("MA(1)", ma1.aic()),
    ];

    let mut valid_models: Vec<_> = models
        .into_iter()
        .filter_map(|(name, aic)| aic.map(|a| (name, a)))
        .collect();
    valid_models.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (name, aic) in valid_models {
        println!("  {}: AIC = {:.2}", name, aic);
    }

    println!("\n=== ARIMA Example Complete ===");
}
