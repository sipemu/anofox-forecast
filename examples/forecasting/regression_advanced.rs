//! Advanced Regression Forecasting
//!
//! Demonstrates auto-lag selection (BIC/AIC), differencing, and seasonal
//! differencing in RegressionForecaster.
//!
//! Run with: cargo run --example regression_advanced --features postprocess

#[cfg(feature = "postprocess")]
fn main() {
    use anofox_forecast::core::TimeSeries;
    use anofox_forecast::models::regression::{
        LagSelectionCriterion, RegressionFeatures, RegressionForecaster,
    };
    use anofox_forecast::models::Forecaster;
    use chrono::{Duration, TimeZone, Utc};

    println!("=== Advanced Regression Forecasting ===\n");

    let n = 120;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    // AR(2) process with trend
    let mut values = vec![10.0; n];
    for i in 2..n {
        let trend = 0.05 * i as f64;
        let ar = 0.5 * values[i - 1] + 0.3 * values[i - 2];
        let noise = ((i * 7 + 3) % 11) as f64 * 0.1 - 0.55;
        values[i] = trend + ar + noise;
    }
    let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

    // ── 1. Auto-lag selection with BIC ───────────────────────────────
    println!("--- 1. Auto-Lag Selection (BIC) ---\n");

    let mut model =
        RegressionForecaster::ols(RegressionFeatures::new().trend().auto_lags(12).no_exog());
    model.fit(&ts).unwrap();

    let forecast = model.predict(5).unwrap();
    println!("R²: {:.4}", model.r_squared().unwrap());
    println!("Forecast (5 steps): {:?}\n", forecast.primary());

    // ── 2. Auto-lag with AIC ─────────────────────────────────────────
    println!("--- 2. Auto-Lag Selection (AIC) ---\n");

    let mut model_aic = RegressionForecaster::ols(
        RegressionFeatures::new()
            .trend()
            .auto_lags_with(12, LagSelectionCriterion::Aic)
            .no_exog(),
    );
    model_aic.fit(&ts).unwrap();
    println!("R² (AIC): {:.4}\n", model_aic.r_squared().unwrap());

    // ── 3. Differencing (d=1) ────────────────────────────────────────
    println!("--- 3. First Differencing ---\n");

    // Strong linear trend — differencing removes it
    let trend_values: Vec<f64> = (0..n).map(|i| 100.0 + 2.0 * i as f64).collect();
    let trend_ts = TimeSeries::univariate(timestamps.clone(), trend_values).unwrap();

    let mut model_diff = RegressionForecaster::ols(
        RegressionFeatures::new()
            .no_trend()
            .differencing(1)
            .lags(1)
            .no_exog(),
    );
    model_diff.fit(&trend_ts).unwrap();
    let forecast_diff = model_diff.predict(5).unwrap();
    println!("Differenced forecast: {:?}", forecast_diff.primary());
    println!("(Should continue the linear trend ~300+)\n");

    // ── 4. Seasonal differencing ─────────────────────────────────────
    println!("--- 4. Seasonal Differencing ---\n");

    let seasonal_values: Vec<f64> = (0..n)
        .map(|i| 50.0 + 0.1 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
        .collect();
    let seasonal_ts = TimeSeries::univariate(timestamps.clone(), seasonal_values).unwrap();

    let mut model_sdiff = RegressionForecaster::ols(
        RegressionFeatures::new()
            .trend()
            .seasonal_differencing(1, 7)
            .no_exog(),
    );
    model_sdiff.fit(&seasonal_ts).unwrap();
    let forecast_sdiff = model_sdiff.predict(7).unwrap();
    println!("Seasonally differenced forecast (7 steps):");
    for (i, &v) in forecast_sdiff.primary().iter().enumerate() {
        println!("  h={}: {:.2}", i + 1, v);
    }

    // ── 5. Combined: differencing + auto-lags ────────────────────────
    println!("\n--- 5. Differencing + Auto-Lags ---\n");

    let mut model_combo = RegressionForecaster::ols(
        RegressionFeatures::new()
            .no_trend()
            .differencing(1)
            .auto_lags(5)
            .no_exog(),
    );
    model_combo.fit(&ts).unwrap();
    let forecast_combo = model_combo.predict(5).unwrap();
    println!("R²: {:.4}", model_combo.r_squared().unwrap());
    println!("Forecast: {:?}", forecast_combo.primary());

    println!("\nDone!");
}

#[cfg(not(feature = "postprocess"))]
fn main() {
    println!("This example requires the 'postprocess' feature.");
    println!("Run with: cargo run --example regression_advanced --features postprocess");
}
