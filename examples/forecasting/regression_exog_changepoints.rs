//! Regression forecasting with exogenous features, changepoints, and backtesting.
//!
//! Demonstrates:
//! 1. Attaching exogenous regressors to a TimeSeries via CalendarAnnotations
//! 2. Adding changepoint structural features to the regression model
//! 3. Cross-validating regression models with exogenous regressors
//! 4. Running the pipeline with structural break detection
//! 5. Batch partitioning of results by structural-break flag
//!
//! Run with: cargo run --example regression_exog_changepoints

use anofox_forecast::core::{CalendarAnnotations, TimeSeries, TimeSeriesBuilder};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::regression::{
    FeatureSafety, RegressionFeatures, RegressionForecaster,
};
use anofox_forecast::models::{Forecaster, ModelRegistry, ModelSpec};
use anofox_forecast::orchestration::prelude::*;
use anofox_forecast::orchestration::report::PipelineReport;
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;

fn main() {
    println!("=== Regression: Exogenous Features + Changepoints + Backtesting ===\n");

    // ─────────────────────────────────────────────────────────────────
    // Generate synthetic data: sales with temperature effect + level shift
    //
    // - Trend: 0.2/day
    // - Seasonal: weekly cycle
    // - Exogenous: "temperature" drives sales
    // - Structural break: marketing campaign at day 80 shifts baseline +15
    // ─────────────────────────────────────────────────────────────────
    let n = 150;
    let changepoint_idx = 80;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    let temperature: Vec<f64> = (0..n)
        .map(|i| 15.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 365.0).sin())
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.2 * i as f64;
            let seasonal = 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let temp_effect = 0.5 * temperature[i]; // temperature drives sales
            let level_shift = if i >= changepoint_idx { 15.0 } else { 0.0 };
            let noise = 1.5 * ((i as f64 * 0.7).cos());
            trend + seasonal + temp_effect + level_shift + noise
        })
        .collect();

    // Attach temperature as an exogenous regressor
    let cal = CalendarAnnotations::new()
        .with_regressor("temperature".to_string(), temperature.clone());

    let ts = TimeSeriesBuilder::new()
        .timestamps(timestamps)
        .values(values)
        .calendar(cal)
        .build()
        .unwrap();

    println!("Data: {} observations", n);
    println!("  Exogenous: temperature (0.5 coefficient)");
    println!("  Changepoint at index {} (level shift +15)", changepoint_idx);
    println!("  Regressors on series: {:?}\n", ts.all_regressors().keys().collect::<Vec<_>>());

    // ─────────────────────────────────────────────────────────────────
    // 1. Regression with exogenous features (no changepoint awareness)
    // ─────────────────────────────────────────────────────────────────
    println!("─── 1. OLS with exogenous regressor (no changepoint) ───\n");

    let features = RegressionFeatures::new()
        .trend()
        .fourier(7, 3)
        .exog(); // include temperature

    let mut model_no_cp = RegressionForecaster::ols(features);
    model_no_cp.fit(&ts).unwrap();
    println!("  R² = {:.4}", model_no_cp.r_squared().unwrap());
    println!("  Has exog? {}", model_no_cp.has_exog());
    println!("  Exog names: {:?}", model_no_cp.exog_names().unwrap());

    // predict() errors when model was fit with exog — must provide future values
    let err = model_no_cp.predict(7);
    println!("  predict(7) without exog: {}", err.unwrap_err());

    // Provide future temperature values for prediction
    let mut future_regs = HashMap::new();
    future_regs.insert(
        "temperature".to_string(),
        (n..n + 7)
            .map(|i| 15.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 365.0).sin())
            .collect(),
    );
    let forecast = model_no_cp.predict_with_exog(7, &future_regs).unwrap();
    println!("  predict_with_exog(7): {:?}\n", &forecast.primary()[..3]);

    // ─────────────────────────────────────────────────────────────────
    // 2. Regression with exog + changepoint step function
    // ─────────────────────────────────────────────────────────────────
    println!("─── 2. Ridge with exog + changepoint step function ───\n");

    let features = RegressionFeatures::new()
        .trend()
        .fourier(7, 3)
        .with_changepoint_steps(vec![changepoint_idx])
        .exog();

    let mut model_cp = RegressionForecaster::ridge(0.1, features);
    model_cp.fit(&ts).unwrap();
    println!("  R² = {:.4} (vs {:.4} without changepoint)", model_cp.r_squared().unwrap(), model_no_cp.r_squared().unwrap());

    let forecast_cp = model_cp.predict_with_exog(7, &future_regs).unwrap();
    println!("  Forecast: {:?}", &forecast_cp.primary()[..3]);

    // ─────────────────────────────────────────────────────────────────
    // 3. Feature safety audit
    // ─────────────────────────────────────────────────────────────────
    println!("\n─── 3. Feature safety classification ───\n");

    let features = RegressionFeatures::new()
        .trend()
        .fourier(7, 3)
        .with_changepoint_steps(vec![changepoint_idx])
        .exog();
    let classification = features.classify_features(&["temperature".to_string()]);

    println!("  {:30} {:>15}", "Feature", "Safety");
    println!("  {:-<46}", "");
    for (name, safety) in &classification {
        let label = match safety {
            FeatureSafety::Deterministic => "Deterministic",
            FeatureSafety::DataDependent => "DataDependent",
            FeatureSafety::Structural => "Structural",
            FeatureSafety::External => "External",
        };
        println!("  {:30} {:>15}", name, label);
    }
    println!();
    println!("  Deterministic: safe in any CV scheme");
    println!("  DataDependent: re-fit per fold (handled by Forecaster::fit)");
    println!("  Structural:    forward-filled — freeze at last training value");
    println!("  External:      user must supply future values");

    // ─────────────────────────────────────────────────────────────────
    // 4. Cross-validation with exogenous regressors
    //
    // The CV framework now automatically:
    // - Slices regressors along with the series (TimeSeries::slice)
    // - Calls predict_with_exog when the model has exogenous features
    // ─────────────────────────────────────────────────────────────────
    println!("\n─── 4. Cross-validation with exogenous features ───\n");

    let cv_config = CVConfig {
        horizon: 7,
        initial_window: 80,
        step_size: 10,
        seasonal_period: Some(7),
        ..CVConfig::default()
    };

    // Model without exog (trend + Fourier + changepoint only)
    let cv_no_exog = cross_validate(&cv_config, &ts, || {
        RegressionForecaster::ols(
            RegressionFeatures::new()
                .trend()
                .fourier(7, 3)
                .with_changepoint_steps(vec![changepoint_idx])
                .no_exog(),
        )
    })
    .unwrap();

    println!(
        "  Without exog:  MAE={:.2}, RMSE={:.2}",
        cv_no_exog.aggregated.mae, cv_no_exog.aggregated.rmse
    );

    // Model with exog (trend + Fourier + changepoint + temperature)
    let cv_with_exog = cross_validate(&cv_config, &ts, || {
        RegressionForecaster::ols(
            RegressionFeatures::new()
                .trend()
                .fourier(7, 3)
                .with_changepoint_steps(vec![changepoint_idx])
                .exog(),
        )
    })
    .unwrap();

    println!(
        "  With exog:     MAE={:.2}, RMSE={:.2}",
        cv_with_exog.aggregated.mae, cv_with_exog.aggregated.rmse
    );
    println!(
        "  Note: exog accuracy depends on fold placement relative to the changepoint.\n  \
         When a fold straddles the level shift, the step function (frozen from training)\n  \
         may not capture the regime change in the test window."
    );

    // ─────────────────────────────────────────────────────────────────
    // 5. Pipeline with structural break detection
    // ─────────────────────────────────────────────────────────────────
    println!("\n─── 5. Pipeline: structural break detection ───\n");

    // Series WITH a large level shift near the end → PELT detects it in holdout
    let n_break = 120;
    let cp_late = 100; // falls in holdout window (last ~20% = 24 obs)
    let timestamps_b: Vec<_> = (0..n_break)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    let values_b: Vec<f64> = (0..n_break)
        .map(|i| {
            let base = if i < cp_late { 50.0 } else { 120.0 }; // large shift for PELT
            base + 0.1 * i as f64 + 2.0 * ((i as f64 * 0.5).sin())
        })
        .collect();
    let ts_break = TimeSeries::univariate(timestamps_b, values_b).unwrap();

    let mut registry = ModelRegistry::new();
    registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));

    let result = PipelineBuilder::new()
        .profile()
        .registry(registry)
        .with_fallback()
        .build()
        .execute(&ts_break, 7)
        .unwrap();

    println!("  Model: {}", result.model_name);
    println!(
        "  Structural break in holdout? {}",
        result.structural_break_in_holdout
    );

    if result.structural_break_in_holdout {
        println!("  Warning: accuracy metrics may be unreliable!");
    }

    // Show the report (includes Warnings section when break detected)
    let report = PipelineReport::from_result(&result);
    let report_text = format!("{}", report);
    if report_text.contains("Warnings") {
        println!("\n  Report excerpt:");
        for line in report_text.lines() {
            if line.contains("Warning") || line.contains("Structural") || line.contains("unreliable") {
                println!("    {}", line);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 6. Batch: partition results by structural break
    // ─────────────────────────────────────────────────────────────────
    println!("\n─── 6. Batch partition by structural break ───\n");

    use anofox_forecast::models::batch::partition_by_structural_break;

    // Simulate a batch of pipeline results
    let series_data = vec![
        ("series_A", false), // clean
        ("series_B", true),  // has structural break in holdout
        ("series_C", false), // clean
        ("series_D", true),  // has structural break
        ("series_E", false), // clean
    ];

    // Build minimal PipelineResult-like objects via the pipeline
    // For demonstration, we use the structural_break_in_holdout flag directly
    let mut batch_results = Vec::new();
    for &(name, has_break) in &series_data {
        let mut r = make_dummy_result(name);
        r.structural_break_in_holdout = has_break;
        batch_results.push(r);
    }

    let (clean_idx, flagged_idx) = partition_by_structural_break(&batch_results);
    println!("  Total series: {}", batch_results.len());
    println!("  Clean (use for aggregated metrics): {:?}", clean_idx);
    println!("  Flagged (exclude from summaries):   {:?}", flagged_idx);

    println!("\n  Clean series:");
    for &i in &clean_idx {
        println!("    [{}] {}", i, series_data[i].0);
    }
    println!("  Flagged series:");
    for &i in &flagged_idx {
        println!("    [{}] {} — structural break in holdout", i, series_data[i].0);
    }

    println!("\n=== Example Complete ===");
}

/// Create a minimal PipelineResult for batch partitioning demonstration.
fn make_dummy_result(name: &str) -> anofox_forecast::orchestration::pipeline::PipelineResult {
    use anofox_forecast::core::Forecast;
    use anofox_forecast::orchestration::decision_log::DecisionLog;
    anofox_forecast::orchestration::pipeline::PipelineResult {
        forecast: Forecast::from_values(vec![1.0, 2.0, 3.0]),
        model_name: name.to_string(),
        profile: None,
        log: DecisionLog::new(),
        model_metadata: vec![],
        horizon_analysis: None,
        selection_confidence: None,
        model_confidence_set: None,
        quality_floor: None,
        preprocess: None,
        ensemble_weights: None,
        metric_scores: None,
        trend_selection: None,
        seasonal_selection: None,
        structural_break_in_holdout: false,
        holdout_forecasts: None,
        holdout_actual: None,
    }
}
