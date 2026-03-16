//! Regression-based forecasting examples.
//!
//! Demonstrates the multi-backend `RegressionForecaster` with various
//! regression estimators, feature configurations, and structural features.
//!
//! Run with: cargo run --example regression

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::regression::{
    ChangepointEncoding, ChangepointFeature, FeatureSafety, RegressionFeatures,
    RegressionForecaster, TrendType,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::sync::Arc;

fn main() {
    println!("=== Regression Forecaster Examples ===\n");

    // ── Generate sample data: trend + weekly seasonality + noise ─────
    let n = 120;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let seasonal = 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let noise = 2.0 * ((i as f64 * 1.3).cos());
            trend + seasonal + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let horizon = 14;
    println!("Data: {} observations, forecasting {} steps ahead\n", n, horizon);

    // ─────────────────────────────────────────────────────────────────
    // 1. OLS — trend + lags
    // ─────────────────────────────────────────────────────────────────
    println!("--- 1. OLS: trend + AR(3) ---");
    let mut ols = RegressionForecaster::trend_ar(3);
    ols.fit(&ts).unwrap();
    let forecast = ols.predict(horizon).unwrap();
    println!("  R² = {:.4}", ols.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 2. OLS — trend + Fourier seasonality
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 2. OLS: trend + Fourier(period=7, order=3) ---");
    let mut fourier = RegressionForecaster::trend_fourier(7, 3);
    fourier.fit(&ts).unwrap();
    let forecast = fourier.predict(horizon).unwrap();
    println!("  R² = {:.4}", fourier.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 3. Ridge — regularized trend + seasonal
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 3. Ridge (lambda=0.1): trend + Fourier ---");
    let features = RegressionFeatures::new().trend().fourier(7, 3).no_exog();
    let mut ridge = RegressionForecaster::ridge(0.1, features);
    ridge.fit(&ts).unwrap();
    let forecast = ridge.predict(horizon).unwrap();
    println!("  R² = {:.4}", ridge.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 4. Elastic Net — L1 + L2
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 4. Elastic Net (lambda=0.1, alpha=0.5): trend + lags + Fourier ---");
    let features = RegressionFeatures::new()
        .trend()
        .lags(3)
        .fourier(7, 3)
        .no_exog();
    let mut enet = RegressionForecaster::elastic_net(0.1, 0.5, features);
    enet.fit(&ts).unwrap();
    let forecast = enet.predict(horizon).unwrap();
    println!("  R² = {:.4}", enet.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 5. Quantile regression — median forecast
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 5. Quantile (tau=0.5, median): trend + Fourier ---");
    let features = RegressionFeatures::new().trend().fourier(7, 3).no_exog();
    let mut quantile = RegressionForecaster::quantile(0.5, features);
    quantile.fit(&ts).unwrap();
    let forecast = quantile.predict(horizon).unwrap();
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 6. WLS — exponential decay weighting (recent data emphasized)
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 6. WLS (decay=0.97): trend + Fourier ---");
    let features = RegressionFeatures::new().trend().fourier(7, 3).no_exog();
    let mut wls = RegressionForecaster::wls_decay(0.97, features);
    wls.fit(&ts).unwrap();
    let forecast = wls.predict(horizon).unwrap();
    println!("  R² = {:.4}", wls.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 7. RLS — adaptive coefficients
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 7. RLS (forgetting=0.98): trend + Fourier ---");
    let features = RegressionFeatures::new().trend().fourier(7, 3).no_exog();
    let mut rls = RegressionForecaster::rls(0.98, features);
    rls.fit(&ts).unwrap();
    let forecast = rls.predict(horizon).unwrap();
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 8. BLS — bounded coefficients
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 8. BLS (non-negative coefficients): trend only ---");
    let features = RegressionFeatures::new().trend().no_exog();
    let mut bls = RegressionForecaster::nnls(features);
    bls.fit(&ts).unwrap();
    let forecast = bls.predict(horizon).unwrap();
    println!("  R² = {:.4}", bls.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 9. Poisson GLM — count data (trend-only for convergence)
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 9. Poisson GLM: count data ---");
    let count_values: Vec<f64> = (0..n)
        .map(|i| {
            let rate = 10.0 + 0.05 * i as f64;
            rate.round().max(1.0)
        })
        .collect();
    let count_timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    let count_ts = TimeSeries::univariate(count_timestamps, count_values).unwrap();
    let features = RegressionFeatures::new().trend().no_exog();
    let mut poisson = RegressionForecaster::poisson(features);
    poisson.fit(&count_ts).unwrap();
    let forecast = poisson.predict(horizon).unwrap();
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 10. Tweedie GLM — continuous positive data
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 10. Tweedie GLM (var_power=1.5): trend ---");
    let features = RegressionFeatures::new().trend().no_exog();
    let mut tweedie = RegressionForecaster::tweedie(1.5, features);
    tweedie.fit(&ts).unwrap();
    let forecast = tweedie.predict(horizon).unwrap();
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 11. Dynamic Linear Model — time-varying parameters
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 11. Dynamic (AICc, time-varying coefficients): trend + Fourier ---");
    let features = RegressionFeatures::new().trend().fourier(7, 3).no_exog();
    let mut dynamic = RegressionForecaster::dynamic(features);
    dynamic.fit(&ts).unwrap();
    let forecast = dynamic.predict(horizon).unwrap();
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 12. Trend components — Polynomial, Exponential, TheilSen
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 12. Advanced trend components ---");

    let features = RegressionFeatures::new()
        .no_trend() // Don't use raw index — use fitted trend components instead
        .with_trend_component(TrendType::Quadratic)
        .fourier(7, 3)
        .no_exog();
    let mut quad = RegressionForecaster::ols(features);
    quad.fit(&ts).unwrap();
    println!("  Quadratic trend R² = {:.4}", quad.r_squared().unwrap());

    let features = RegressionFeatures::new()
        .no_trend()
        .with_trend_component(TrendType::TheilSen)
        .fourier(7, 3)
        .no_exog();
    let mut theilsen = RegressionForecaster::ols(features);
    theilsen.fit(&ts).unwrap();
    println!("  Theil-Sen trend R² = {:.4}", theilsen.r_squared().unwrap());

    // ─────────────────────────────────────────────────────────────────
    // 13. Dummy seasonality (one-hot encoding)
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 13. Dummy seasonality (period=7) ---");
    let features = RegressionFeatures::new()
        .trend()
        .dummy_seasonal(7)
        .no_exog();
    let mut dummy = RegressionForecaster::ols(features);
    dummy.fit(&ts).unwrap();
    let forecast = dummy.predict(horizon).unwrap();
    println!("  R² = {:.4}", dummy.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 14. Structural features — changepoint step functions
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 14. Structural features: changepoint step functions ---");

    // Simulate a level shift at observation 60
    let shift_values: Vec<f64> = (0..n)
        .map(|i| {
            let base = if i < 60 { 50.0 } else { 70.0 }; // level shift
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            base + seasonal + ((i as f64 * 0.9).cos())
        })
        .collect();
    let shift_timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();
    let shift_ts = TimeSeries::univariate(shift_timestamps, shift_values).unwrap();

    let features = RegressionFeatures::new()
        .trend()
        .fourier(7, 3)
        .with_changepoint_steps(vec![60]) // step function at CP index 60
        .no_exog();
    let mut cp_model = RegressionForecaster::ols(features);
    cp_model.fit(&shift_ts).unwrap();
    let forecast = cp_model.predict(horizon).unwrap();
    println!("  R² = {:.4} (with changepoint at index 60)", cp_model.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 15. Changepoint with regime index encoding
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 15. Changepoint: regime index encoding ---");
    let cp = ChangepointFeature::new(vec![40, 80], ChangepointEncoding::RegimeIndex);
    let features = RegressionFeatures::new()
        .trend()
        .fourier(7, 3)
        .with_structural(Arc::new(cp))
        .no_exog();
    let mut regime_model = RegressionForecaster::ols(features);
    regime_model.fit(&ts).unwrap();
    let forecast = regime_model.predict(horizon).unwrap();
    println!("  R² = {:.4} (2 changepoints, regime index)", regime_model.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    // ─────────────────────────────────────────────────────────────────
    // 16. Feature safety classification
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 16. Feature safety classification ---");
    let cp = ChangepointFeature::step_functions(vec![60]);
    let features = RegressionFeatures::new()
        .trend()
        .lags(2)
        .with_trend_component(TrendType::TheilSen)
        .fourier(7, 3)
        .dummy_seasonal(7)
        .with_structural(Arc::new(cp))
        .no_exog();
    let classification = features.classify_features(&[]);
    println!("  {:30} {:>15}", "Feature", "Safety");
    println!("  {:-<46}", "");
    for (name, safety) in &classification {
        let safety_str = match safety {
            FeatureSafety::Deterministic => "Deterministic",
            FeatureSafety::DataDependent => "DataDependent",
            FeatureSafety::Structural => "Structural",
            FeatureSafety::External => "External",
        };
        println!("  {:30} {:>15}", name, safety_str);
    }

    // ─────────────────────────────────────────────────────────────────
    // 17. Combined: Ridge + trend + seasonal + changepoints
    // ─────────────────────────────────────────────────────────────────
    println!("\n--- 17. Full pipeline: Ridge + trend + Fourier + changepoint ---");
    let features = RegressionFeatures::new()
        .trend()
        .with_trend_component(TrendType::Linear)
        .fourier(7, 3)
        .with_changepoint_steps(vec![60])
        .no_exog();
    let mut full = RegressionForecaster::ridge(0.05, features);
    full.fit(&shift_ts).unwrap();
    let forecast = full.predict(horizon).unwrap();
    println!("  Backend: {}", full.name());
    println!("  R² = {:.4}", full.r_squared().unwrap());
    print_forecast(&forecast.primary()[..5]);

    println!("\n=== Regression Forecaster Examples Complete ===");
}

fn print_forecast(values: &[f64]) {
    print!("  Forecast: [");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.2}", v);
    }
    println!(", ...]");
}
