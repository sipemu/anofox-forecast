//! Standalone FeatureGenerator example.
//!
//! Shows how to create deterministic regressors from timestamps —
//! Fourier terms, calendar indicators, holiday flags — and attach
//! them to a TimeSeries for use with any exog-supporting model.
//!
//! Run with: cargo run --example feature_generator

use anofox_forecast::core::TimeSeries;
use anofox_forecast::features::FeatureGenerator;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("=== FeatureGenerator Example ===\n");

    // ── 1. Build a generator ─────────────────────────────────────────────
    println!("--- 1. Building a FeatureGenerator ---\n");

    let gen = FeatureGenerator::new()
        .fourier(7, 2) // weekly Fourier terms (4 columns)
        .day_of_week() // Mon–Sat indicators (6 columns, Sun dropped)
        .month_of_year() // Feb–Dec indicators (11 columns, Jan dropped)
        .quarter(); // Q2–Q4 indicators (3 columns, Q1 dropped)

    println!("Feature names ({} total):", gen.feature_names().len());
    for name in gen.feature_names() {
        println!("  {}", name);
    }

    // ── 2. Generate features from timestamps ─────────────────────────────
    println!("\n--- 2. Generating features ---\n");

    let n = 365;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    let features = gen.generate(&timestamps);
    println!(
        "Generated {} features × {} observations\n",
        features.len(),
        n
    );

    // Show a few values
    println!("First 7 days of fourier_7_sin1:");
    let sin1 = &features["fourier_7_sin1"];
    for i in 0..7 {
        println!("  {} → {:.4}", timestamps[i].format("%a %Y-%m-%d"), sin1[i]);
    }

    println!("\nFirst 7 days of dow_mon:");
    let dow_mon = &features["dow_mon"];
    for i in 0..7 {
        println!(
            "  {} → {:.0}",
            timestamps[i].format("%a %Y-%m-%d"),
            dow_mon[i]
        );
    }

    // ── 3. Holiday indicators ────────────────────────────────────────────
    println!("\n--- 3. Holiday indicators ---\n");

    let holidays = vec![
        Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // New Year
        Utc.with_ymd_and_hms(2024, 7, 4, 0, 0, 0).unwrap(), // Independence Day
        Utc.with_ymd_and_hms(2024, 12, 25, 0, 0, 0).unwrap(), // Christmas
    ];

    let gen_holiday = FeatureGenerator::new().holiday("national", holidays);

    let holiday_feats = gen_holiday.generate(&timestamps);
    let col = &holiday_feats["holiday_national"];
    let holiday_days: Vec<_> = timestamps
        .iter()
        .zip(col)
        .filter(|(_, &v)| v > 0.0)
        .map(|(t, _)| t.format("%Y-%m-%d").to_string())
        .collect();
    println!("Holiday flags set on: {:?}", holiday_days);

    // ── 4. Attach to TimeSeries and forecast ─────────────────────────────
    println!("\n--- 4. Forecasting with generated features ---\n");

    // Simulate: y = 100 + 5*sin(weekly) + 20*promo_indicator
    let promo_dates: Vec<_> = (0..n)
        .filter(|i| i % 30 == 0)
        .map(|i| timestamps[i])
        .collect();

    let gen_model = FeatureGenerator::new()
        .fourier(7, 2)
        .holiday("promo", promo_dates.clone());

    let model_feats = gen_model.generate(&timestamps);

    let y: Vec<f64> = (0..n)
        .map(|i| {
            100.0
                + 5.0 * (2.0 * PI * i as f64 / 7.0).sin()
                + 3.0 * (2.0 * PI * i as f64 / 7.0).cos()
                + 20.0 * model_feats["holiday_promo"][i]
        })
        .collect();

    // Attach features to TimeSeries
    let mut ts = TimeSeries::univariate(timestamps.clone(), y).unwrap();
    gen_model.add_to(&mut ts);

    println!(
        "Regressors attached to TimeSeries: {}",
        ts.all_regressors().len()
    );

    // Fit ARIMA(0,0,0) — pure regression to recover effects
    let mut model = ARIMA::new(0, 0, 0);
    model.fit(&ts).unwrap();
    println!("Model fitted with exog: {}", model.has_exog());
    println!("Regressor names: {:?}", model.exog_names().unwrap());

    // ── 5. Generate future features for prediction ───────────────────────
    println!("\n--- 5. Future prediction ---\n");

    let horizon = 14;
    // Continue timestamp sequence
    let all_ts: Vec<_> = (0..(n + horizon) as i64)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
        .collect();
    let all_feats = gen_model.generate(&all_ts);

    // Slice future portion
    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_feats {
        future_regs.insert(name.clone(), vals[n..].to_vec());
    }

    let forecast = model.predict_with_exog(horizon, &future_regs).unwrap();
    println!("{:>4} {:>12}", "h", "Forecast");
    println!("{:-<18}", "");
    for (i, val) in forecast.primary().iter().enumerate() {
        println!("{:>4} {:>12.2}", i + 1, val);
    }

    // ── 6. Scenario analysis: promo ON vs OFF ────────────────────────────
    println!("\n--- 6. Scenario analysis ---\n");

    let mut future_promo_on = future_regs.clone();
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0; // promo on day 8
    future_promo_on.insert("holiday_promo".to_string(), promo_col);

    let mut future_promo_off = future_regs;
    future_promo_off.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let fc_on = model.predict_with_exog(horizon, &future_promo_on).unwrap();
    let fc_off = model.predict_with_exog(horizon, &future_promo_off).unwrap();

    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "h", "Promo ON", "Promo OFF", "Effect"
    );
    println!("{:-<44}", "");
    for i in 0..horizon {
        let diff = fc_on.primary()[i] - fc_off.primary()[i];
        println!(
            "{:>4} {:>12.2} {:>12.2} {:>12.2}",
            i + 1,
            fc_on.primary()[i],
            fc_off.primary()[i],
            diff
        );
    }

    // ── 7. Reuse across multiple models ──────────────────────────────────
    println!("\n--- 7. Reusing the same generator ---\n");
    println!("The same FeatureGenerator can be reused for:");
    println!("  - Multiple time series (same temporal structure)");
    println!("  - Multiple models (ARIMA, ETS, Theta, MSTL, ...)");
    println!("  - Train/test splits (generate features for each split)");
    println!("  - Cross-validation folds");
    println!("\nAll features are deterministic (functions of time only),");
    println!("so they are safe for cross-validation.");

    println!("\n=== FeatureGenerator Example Complete ===");
}
