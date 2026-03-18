//! Exogenous Regressors example.
//!
//! Shows how to use FeatureGenerator and external regressors with
//! forecasting models: fit with regressors, predict with future values,
//! and compare scenario forecasts.
//!
//! Run with: cargo run --example exogenous

use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use anofox_forecast::features::FeatureGenerator;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("=== Exogenous Regressors Example ===\n");

    // ── 1. Create data with known effects ────────────────────────────────
    println!("--- 1. Simulated retail data ---\n");

    let n = 365;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    // Generate features
    let gen = FeatureGenerator::new().fourier(7, 2);
    let fourier_feats = gen.generate(&timestamps);

    // External regressor: temperature
    let temperature: Vec<f64> = (0..n)
        .map(|i| 15.0 + 12.0 * (2.0 * PI * i as f64 / 365.0).cos())
        .collect();

    // Promotion flag: every 30 days
    let promo: Vec<f64> = (0..n)
        .map(|i| if i % 30 == 0 { 1.0 } else { 0.0 })
        .collect();

    // y = 200 + weekly_seasonality + 3*temperature + 40*promo
    let y: Vec<f64> = (0..n)
        .map(|i| {
            200.0
                + 8.0 * fourier_feats["fourier_7_sin1"][i]
                + 4.0 * fourier_feats["fourier_7_cos1"][i]
                - 2.0 * fourier_feats["fourier_7_sin2"][i]
                + 1.0 * fourier_feats["fourier_7_cos2"][i]
                + 3.0 * temperature[i]
                + 40.0 * promo[i]
        })
        .collect();

    println!("Generated {} daily observations:", n);
    println!("  Baseline:    200");
    println!("  Weekly:      Fourier(7, 2)");
    println!("  Temperature: coeff = 3.0");
    println!("  Promotion:   coeff = 40.0");

    // ── 2. Attach regressors to TimeSeries ───────────────────────────────
    println!("\n--- 2. Attaching regressors ---\n");

    let mut ts = TimeSeries::univariate(timestamps.clone(), y.clone()).unwrap();

    // Use FeatureGenerator to attach Fourier terms
    gen.add_to(&mut ts);

    // Manually add external regressors
    let mut cal = ts
        .calendar()
        .cloned()
        .unwrap_or_else(CalendarAnnotations::new);
    cal = cal.with_regressor("temperature".to_string(), temperature.clone());
    cal = cal.with_regressor("promo".to_string(), promo.clone());
    ts.set_calendar(cal);

    println!("Attached {} regressors:", ts.all_regressors().len());
    for name in ts.all_regressors().keys() {
        println!("  {}", name);
    }

    // ── 3. Fit models with exogenous ─────────────────────────────────────
    println!("\n--- 3. Fitting models ---\n");

    let horizon = 14;

    // Prepare future regressors (same for all models)
    let all_ts: Vec<_> = (0..(n + horizon) as i64)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
        .collect();
    let all_fourier = gen.generate(&all_ts);

    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_fourier {
        future_regs.insert(name.clone(), vals[n..].to_vec());
    }
    let future_temp: Vec<f64> = (n..n + horizon)
        .map(|i| 15.0 + 12.0 * (2.0 * PI * i as f64 / 365.0).cos())
        .collect();
    let future_promo: Vec<f64> = (n..n + horizon)
        .map(|i| if i % 30 == 0 { 1.0 } else { 0.0 })
        .collect();
    future_regs.insert("temperature".to_string(), future_temp);
    future_regs.insert("promo".to_string(), future_promo);

    // Model 1: ARIMA(0,0,0)
    let mut arima = ARIMA::new(0, 0, 0);
    arima.fit(&ts).unwrap();
    let fc_arima = arima.predict_with_exog(horizon, &future_regs).unwrap();
    println!(
        "ARIMA(0,0,0): has_exog={}, regressors={:?}",
        arima.has_exog(),
        arima.exog_names().unwrap().len()
    );

    // Model 2: AutoETS
    let mut ets = AutoETS::non_seasonal();
    ets.fit(&ts).unwrap();
    let fc_ets = ets.predict_with_exog(horizon, &future_regs).unwrap();
    println!("AutoETS:      has_exog={}", ets.has_exog());

    // Model 3: Theta
    let mut theta = Theta::new();
    theta.fit(&ts).unwrap();
    let fc_theta = theta.predict_with_exog(horizon, &future_regs).unwrap();
    println!("Theta:        has_exog={}", theta.has_exog());

    // Model 4: MSTL
    let mut mstl = MSTLForecaster::new(vec![7]);
    mstl.fit(&ts).unwrap();
    let fc_mstl = mstl.predict_with_exog(horizon, &future_regs).unwrap();
    println!("MSTL:         has_exog={}", mstl.has_exog());

    // ── 4. Compare forecasts ─────────────────────────────────────────────
    println!("\n--- 4. Forecast comparison ---\n");

    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>10}",
        "h", "ARIMA", "AutoETS", "Theta", "MSTL"
    );
    println!("{:-<48}", "");
    for i in 0..horizon {
        println!(
            "{:>4} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            fc_arima.primary()[i],
            fc_ets.primary()[i],
            fc_theta.primary()[i],
            fc_mstl.primary()[i]
        );
    }

    // ── 5. Scenario analysis ─────────────────────────────────────────────
    println!("\n--- 5. Scenario analysis: temperature impact ---\n");

    let mut future_cold = future_regs.clone();
    future_cold.insert("temperature".to_string(), vec![0.0; horizon]);

    let mut future_hot = future_regs.clone();
    future_hot.insert("temperature".to_string(), vec![30.0; horizon]);

    let fc_cold = arima.predict_with_exog(horizon, &future_cold).unwrap();
    let fc_hot = arima.predict_with_exog(horizon, &future_hot).unwrap();

    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "h", "Cold (0°C)", "Hot (30°C)", "Diff"
    );
    println!("{:-<40}", "");
    for i in 0..5 {
        let diff = fc_hot.primary()[i] - fc_cold.primary()[i];
        println!(
            "{:>4} {:>12.2} {:>12.2} {:>12.2}",
            i + 1,
            fc_cold.primary()[i],
            fc_hot.primary()[i],
            diff
        );
    }
    println!("\nExpected diff ≈ 3.0 × (30 - 0) = 90.0");

    // ── 6. Promotion uplift ──────────────────────────────────────────────
    println!("\n--- 6. Promotion uplift ---\n");

    let mut future_with_promo = future_regs.clone();
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0; // promo on day 8
    future_with_promo.insert("promo".to_string(), promo_col);

    let mut future_no_promo = future_regs;
    future_no_promo.insert("promo".to_string(), vec![0.0; horizon]);

    let fc_promo = arima
        .predict_with_exog(horizon, &future_with_promo)
        .unwrap();
    let fc_no_promo = arima.predict_with_exog(horizon, &future_no_promo).unwrap();

    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "h", "With Promo", "No Promo", "Uplift"
    );
    println!("{:-<40}", "");
    for i in 5..10 {
        let uplift = fc_promo.primary()[i] - fc_no_promo.primary()[i];
        println!(
            "{:>4} {:>12.2} {:>12.2} {:>12.2}",
            i + 1,
            fc_promo.primary()[i],
            fc_no_promo.primary()[i],
            uplift
        );
    }
    println!("\nPromo uplift on day 8 ≈ 40.0 (true coefficient)");

    // ── 7. Error handling ────────────────────────────────────────────────
    println!("\n--- 7. Error handling ---\n");

    // predict() fails when model was fit with exog
    match arima.predict(horizon) {
        Ok(_) => println!("predict() succeeded (unexpected)"),
        Err(e) => println!("predict() correctly errors: {}", e),
    }

    // Missing regressor errors
    let empty_regs: HashMap<String, Vec<f64>> = HashMap::new();
    match arima.predict_with_exog(horizon, &empty_regs) {
        Ok(_) => println!("Missing regressors succeeded (unexpected)"),
        Err(e) => println!("Missing regressors correctly errors: {}", e),
    }

    println!("\n=== Exogenous Regressors Example Complete ===");
}
