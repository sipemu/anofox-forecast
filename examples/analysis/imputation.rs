//! Missing Value Imputation example.
//!
//! Demonstrates all imputation strategies available in anofox-forecast,
//! from simple policy-based fills to seasonal-aware methods.
//!
//! Run with: cargo run --example imputation

use anofox_forecast::core::{CalendarAnnotations, MissingValuePolicy, TimeSeries};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Missing Value Imputation Example ===\n");

    // --- 1. Create a time series with missing values ---
    let timestamps: Vec<_> = (0..30)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
        .collect();

    let mut values: Vec<f64> = (0..30)
        .map(|i| 50.0 + 2.0 * i as f64 + 5.0 * (i as f64 * 0.5).sin())
        .collect();

    // Introduce missing values at different positions
    values[0] = f64::NAN; // leading
    values[1] = f64::NAN; // leading
    values[10] = f64::NAN; // interior
    values[11] = f64::NAN; // adjacent interior
    values[20] = f64::INFINITY; // Inf treated as missing
    values[29] = f64::NAN; // trailing

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    // --- 2. Inspect missing values ---
    println!("--- Missing Value Metadata ---");
    println!("Has missing values: {}", ts.has_missing_values());
    println!("Missing count:      {:?}", ts.missing_count());

    let mask = ts.missing_mask();
    let missing_positions: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| i)
        .collect();
    println!("Missing at indices:  {:?}\n", missing_positions);

    // --- 3. Policy-based imputation ---
    println!("--- Policy-Based Imputation ---\n");

    // Forward fill
    let ffill = ts.sanitized(MissingValuePolicy::ForwardFill).unwrap();
    println!(
        "ForwardFill:  leading NaN preserved = {}",
        ffill.primary_values()[0].is_nan()
    );
    println!(
        "              index 10 = {:.2} (carried from index 9)",
        ffill.primary_values()[10]
    );

    // Backward fill
    let bfill = ts.sanitized(MissingValuePolicy::BackwardFill).unwrap();
    println!(
        "BackwardFill: leading NaN filled = {:.2} (from index 2)",
        bfill.primary_values()[0]
    );
    println!(
        "              trailing NaN preserved = {}",
        bfill.primary_values()[29].is_nan()
    );

    // Forward + Backward (handles both edges)
    let fb = ts.imputed_forward_backward();
    println!(
        "Fwd+Bwd:      leading = {:.2}, trailing = {:.2}",
        fb.primary_values()[0],
        fb.primary_values()[29]
    );
    println!("              all finite = {}", !fb.has_missing_values());

    // Fill with mean
    let mean_fill = ts.sanitized(MissingValuePolicy::FillMean).unwrap();
    println!(
        "FillMean:     fill value = {:.2}",
        mean_fill.primary_values()[10]
    );

    // Fill with median
    let median_fill = ts.sanitized(MissingValuePolicy::FillMedian).unwrap();
    println!(
        "FillMedian:   fill value = {:.2}",
        median_fill.primary_values()[10]
    );

    // Linear interpolation
    let interp = ts.sanitized(MissingValuePolicy::Interpolate).unwrap();
    println!(
        "Interpolate:  index 10 = {:.2}, index 11 = {:.2}",
        interp.primary_values()[10],
        interp.primary_values()[11]
    );

    // --- 4. Moving average imputation ---
    println!("\n--- Moving Average Imputation (window=5) ---");
    let ma = ts.imputed_moving_average(5).unwrap();
    println!(
        "Index 10: {:.2} (avg of neighbors in window)",
        ma.primary_values()[10]
    );
    println!(
        "Index 11: {:.2} (multi-pass fills adjacent gaps)",
        ma.primary_values()[11]
    );
    println!("All finite: {}", !ma.has_missing_values());

    // --- 5. Seasonal imputation ---
    println!("\n--- Seasonal Imputation (period=7, weekly) ---");

    let weekly_timestamps: Vec<_> = (0..42)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
        .collect();

    // Weekly pattern: high on weekdays, low on weekends
    let day_effect = [10.0, 8.0, 6.0, 4.0, 2.0, -5.0, -8.0];
    let mut weekly_values: Vec<f64> = (0..42)
        .map(|i| 50.0 + day_effect[i % 7] + 0.5 * i as f64)
        .collect();

    // Remove some Mondays
    weekly_values[7] = f64::NAN;
    weekly_values[21] = f64::NAN;

    let weekly_ts = TimeSeries::univariate(weekly_timestamps, weekly_values).unwrap();
    let seasonal = weekly_ts.imputed_seasonal(7).unwrap();

    println!(
        "Monday (index 7):  {:.2} (median of other Mondays: {:.2}, {:.2}, {:.2}, {:.2})",
        seasonal.primary_values()[7],
        weekly_ts.primary_values()[0],
        weekly_ts.primary_values()[14],
        weekly_ts.primary_values()[28],
        weekly_ts.primary_values()[35],
    );

    // --- 6. Regressor imputation ---
    println!("\n--- Regressor Imputation ---");

    let reg_timestamps: Vec<_> = (0..10)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
        .collect();
    let reg_values: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();

    let calendar = CalendarAnnotations::new()
        .with_regressor(
            "price".to_string(),
            vec![
                10.0,
                f64::NAN,
                12.0,
                f64::NAN,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
            ],
        )
        .with_regressor(
            "promo".to_string(),
            vec![0.0, 1.0, f64::NAN, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        );

    let mut reg_ts = TimeSeries::univariate(reg_timestamps, reg_values).unwrap();
    reg_ts.set_calendar(calendar);

    let imputed = reg_ts
        .with_imputed_regressors(MissingValuePolicy::FillMean)
        .unwrap();
    println!("Price regressor (FillMean):");
    println!("  Before: [10, NaN, 12, NaN, 14, 15, 16, 17, 18, 19]");
    println!(
        "  After:  {:?}",
        imputed
            .regressor("price")
            .unwrap()
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
    );

    println!("Promo regressor (FillMean):");
    println!("  Before: [0, 1, NaN, 1, 0, 0, 1, 0, 1, 0]");
    println!(
        "  After:  {:?}",
        imputed
            .regressor("promo")
            .unwrap()
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // --- 7. End-to-end: impute then forecast ---
    println!("\n--- End-to-End: Impute → Fit → Forecast ---");

    let clean = ts.imputed_forward_backward();
    println!(
        "Imputed series: {} observations, no missing = {}",
        clean.len(),
        !clean.has_missing_values()
    );

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    println!(
        "Naive forecast (5 steps): {:?}",
        forecast
            .primary()
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    println!("\n=== Imputation Example Complete ===");
}
