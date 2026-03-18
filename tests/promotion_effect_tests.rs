//! Integration tests: recover simulated promotion / calendar / external effects
//! using FeatureGenerator + statistical forecasting models + OLS pre-regression.
//!
//! Each test generates artificial data with **known** effects, fits a model with
//! exogenous regressors derived from [`FeatureGenerator`], and verifies that:
//!   1. The model recovers the effect direction and approximate magnitude.
//!   2. Forecasts with the effect "on" differ from forecasts with the effect "off"
//!      by roughly the true effect size.

use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use anofox_forecast::features::FeatureGenerator;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig};
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

// ── helpers ──────────────────────────────────────────────────────────────────

fn daily_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn monthly_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let mut ts = Vec::with_capacity(n);
    let mut year = 2020i32;
    let mut month = 1u32;
    for _ in 0..n {
        ts.push(Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).unwrap());
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
    }
    ts
}

/// Root-mean-square error between two slices.
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    mse.sqrt()
}

/// Mean absolute error between two slices.
fn mae(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

// ── Scenario 1: Promotion effect (holiday indicator) ─────────────────────────
//
// Daily sales = baseline(100) + weekly_seasonality + promotion_bump(+30)
// Promotions happen on specific dates.
// We use FeatureGenerator.holiday() to create the indicator, fit the model, and
// verify the forecast difference between promo-on and promo-off ≈ 30.

fn build_promotion_scenario() -> (TimeSeries, FeatureGenerator, Vec<chrono::DateTime<Utc>>) {
    let n = 365;
    let timestamps = daily_timestamps(n);

    // Promotion dates: every 14 days starting from day 7
    let promo_dates: Vec<_> = (0..)
        .map(|k| 7 + k * 14)
        .take_while(|&d| d < n)
        .map(|d| timestamps[d])
        .collect();

    let gen = FeatureGenerator::new()
        .fourier(7, 2) // weekly seasonality
        .holiday("promo", promo_dates.clone());

    let features = gen.generate(&timestamps);

    // Build y = 100 + weekly_seasonality + 30 * promo_indicator
    let mut y = vec![100.0; n];
    for i in 0..n {
        // Weekly seasonality from Fourier
        y[i] += 8.0 * features["fourier_7_sin1"][i];
        y[i] += 4.0 * features["fourier_7_cos1"][i];
        y[i] -= 2.0 * features["fourier_7_sin2"][i];
        y[i] += 1.0 * features["fourier_7_cos2"][i];
        // Promotion effect
        y[i] += 30.0 * features["holiday_promo"][i];
    }

    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    gen.add_to(&mut ts);

    (ts, gen, promo_dates)
}

#[test]
fn arima_recovers_promotion_effect() {
    let (ts, gen, _) = build_promotion_scenario();
    let horizon = 14;

    let mut model = ARIMA::new(0, 0, 0); // pure regression
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    // Generate future features — but Fourier is index-based so we need to
    // continue from training. We'll use gen.generate on training+future and
    // slice the last `horizon` entries.
    let all_ts = daily_timestamps(365 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_regs.insert(name.clone(), vals[365..].to_vec());
    }
    // Override promo: only day 7 of the future
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0;
    future_regs.insert("holiday_promo".to_string(), promo_col);

    let forecast = model.predict_with_exog(horizon, &future_regs).unwrap();

    // Also predict with promo OFF everywhere
    let mut future_no_promo = future_regs.clone();
    future_no_promo.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let forecast_no_promo = model.predict_with_exog(horizon, &future_no_promo).unwrap();

    // The difference on day 7 should be ≈ 30
    let diff = forecast.primary()[7] - forecast_no_promo.primary()[7];
    assert!(
        (diff - 30.0).abs() < 2.0,
        "ARIMA promotion effect: expected ~30, got {:.2}",
        diff
    );

    // Non-promo days should have ~0 difference
    let diff_other = forecast.primary()[3] - forecast_no_promo.primary()[3];
    assert!(
        diff_other.abs() < 0.1,
        "Non-promo day diff should be ~0, got {:.2}",
        diff_other
    );
}

#[test]
fn auto_ets_recovers_promotion_effect() {
    let (ts, gen, _) = build_promotion_scenario();
    let horizon = 14;

    let mut model = AutoETS::with_config(AutoETSConfig::non_seasonal());
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    let all_ts = daily_timestamps(365 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_regs.insert(name.clone(), vals[365..].to_vec());
    }
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0;
    future_regs.insert("holiday_promo".to_string(), promo_col);

    let mut future_no_promo = future_regs.clone();
    future_no_promo.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let forecast_on = model.predict_with_exog(horizon, &future_regs).unwrap();
    let forecast_off = model.predict_with_exog(horizon, &future_no_promo).unwrap();

    let diff = forecast_on.primary()[7] - forecast_off.primary()[7];
    assert!(
        (diff - 30.0).abs() < 2.0,
        "AutoETS promotion effect: expected ~30, got {:.2}",
        diff
    );
}

#[test]
fn theta_recovers_promotion_effect() {
    let (ts, gen, _) = build_promotion_scenario();
    let horizon = 14;

    let mut model = Theta::new();
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    let all_ts = daily_timestamps(365 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_regs.insert(name.clone(), vals[365..].to_vec());
    }
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0;
    future_regs.insert("holiday_promo".to_string(), promo_col);

    let mut future_no_promo = future_regs.clone();
    future_no_promo.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let forecast_on = model.predict_with_exog(horizon, &future_regs).unwrap();
    let forecast_off = model.predict_with_exog(horizon, &future_no_promo).unwrap();

    let diff = forecast_on.primary()[7] - forecast_off.primary()[7];
    assert!(
        (diff - 30.0).abs() < 2.0,
        "Theta promotion effect: expected ~30, got {:.2}",
        diff
    );
}

#[test]
fn mstl_recovers_promotion_effect() {
    let (ts, gen, _) = build_promotion_scenario();
    let horizon = 14;

    let mut model = MSTLForecaster::new(vec![7]);
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    let all_ts = daily_timestamps(365 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_regs: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_regs.insert(name.clone(), vals[365..].to_vec());
    }
    let mut promo_col = vec![0.0; horizon];
    promo_col[7] = 1.0;
    future_regs.insert("holiday_promo".to_string(), promo_col);

    let mut future_no_promo = future_regs.clone();
    future_no_promo.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let forecast_on = model.predict_with_exog(horizon, &future_regs).unwrap();
    let forecast_off = model.predict_with_exog(horizon, &future_no_promo).unwrap();

    let diff = forecast_on.primary()[7] - forecast_off.primary()[7];
    assert!(
        (diff - 30.0).abs() < 2.0,
        "MSTL promotion effect: expected ~30, got {:.2}",
        diff
    );
}

// ── Scenario 2: Temperature effect (continuous regressor) ────────────────────
//
// Monthly sales = 200 + annual_seasonality + 3.5 * temperature
// Temperature is an external continuous regressor (correlated with seasonality).
// We verify that forecast(temp=30) > forecast(temp=10) by ≈ 3.5 * 20 = 70.

fn build_temperature_scenario() -> (TimeSeries, FeatureGenerator, Vec<f64>) {
    let n = 120; // 10 years monthly
    let timestamps = monthly_timestamps(n);

    let gen = FeatureGenerator::new().fourier(12, 2);
    let features = gen.generate(&timestamps);

    // Temperature: correlated with season but with some variation
    let temperature: Vec<f64> = (0..n)
        .map(|i| 15.0 + 12.0 * (2.0 * PI * i as f64 / 12.0).cos() + (i as f64 * 0.7).sin())
        .collect();

    // y = 200 + fourier seasonality + 3.5 * temperature
    let mut y = vec![200.0; n];
    for i in 0..n {
        y[i] += 10.0 * features["fourier_12_sin1"][i];
        y[i] += 5.0 * features["fourier_12_cos1"][i];
        y[i] -= 3.0 * features["fourier_12_sin2"][i];
        y[i] += 2.0 * features["fourier_12_cos2"][i];
        y[i] += 3.5 * temperature[i];
    }

    // Attach features + temperature as regressors
    let mut cal = CalendarAnnotations::new();
    for (name, vals) in &features {
        cal = cal.with_regressor(name.clone(), vals.clone());
    }
    cal = cal.with_regressor("temperature".to_string(), temperature.clone());

    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    ts.set_calendar(cal);

    (ts, gen, temperature)
}

#[test]
fn arima_recovers_temperature_effect() {
    let (ts, gen, _temp) = build_temperature_scenario();
    let horizon = 12;

    let mut model = ARIMA::new(0, 0, 0);
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    // Generate future features
    let all_ts = monthly_timestamps(120 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_high: HashMap<String, Vec<f64>> = HashMap::new();
    let mut future_low: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_high.insert(name.clone(), vals[120..].to_vec());
        future_low.insert(name.clone(), vals[120..].to_vec());
    }

    // High temperature scenario (30°C) vs low (10°C)
    future_high.insert("temperature".to_string(), vec![30.0; horizon]);
    future_low.insert("temperature".to_string(), vec![10.0; horizon]);

    let fc_high = model.predict_with_exog(horizon, &future_high).unwrap();
    let fc_low = model.predict_with_exog(horizon, &future_low).unwrap();

    // Difference should be ≈ 3.5 * (30 - 10) = 70
    let expected_diff = 3.5 * 20.0;
    for i in 0..horizon {
        let diff = fc_high.primary()[i] - fc_low.primary()[i];
        assert!(
            (diff - expected_diff).abs() < 5.0,
            "ARIMA temp effect at h={}: expected ~{:.0}, got {:.2}",
            i + 1,
            expected_diff,
            diff
        );
    }
}

#[test]
fn auto_ets_recovers_temperature_effect() {
    let (ts, gen, _temp) = build_temperature_scenario();
    let horizon = 12;

    let mut model = AutoETS::with_config(AutoETSConfig::non_seasonal());
    model.fit(&ts).unwrap();

    let all_ts = monthly_timestamps(120 + horizon);
    let all_features = gen.generate(&all_ts);
    let mut future_high: HashMap<String, Vec<f64>> = HashMap::new();
    let mut future_low: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_high.insert(name.clone(), vals[120..].to_vec());
        future_low.insert(name.clone(), vals[120..].to_vec());
    }
    future_high.insert("temperature".to_string(), vec![30.0; horizon]);
    future_low.insert("temperature".to_string(), vec![10.0; horizon]);

    let fc_high = model.predict_with_exog(horizon, &future_high).unwrap();
    let fc_low = model.predict_with_exog(horizon, &future_low).unwrap();

    let expected_diff = 3.5 * 20.0;
    for i in 0..horizon {
        let diff = fc_high.primary()[i] - fc_low.primary()[i];
        assert!(
            (diff - expected_diff).abs() < 5.0,
            "AutoETS temp effect at h={}: expected ~{:.0}, got {:.2}",
            i + 1,
            expected_diff,
            diff
        );
    }
}

// ── Scenario 3: Multiple simultaneous effects ────────────────────────────────
//
// Daily data with:
//   - Weekly seasonality via Fourier
//   - Promotion bump (+25 on specific days)
//   - External regressor "ad_spend" with coefficient 0.5
//   - Day-of-week effect (Saturday boost +15, encoded in DOW indicators)
//
// We verify that the full model produces accurate in-sample fitted values
// and that individual effects are recoverable via predict_with_exog scenarios.

fn build_multi_effect_scenario() -> (TimeSeries, FeatureGenerator, Vec<f64>) {
    let n = 364; // 52 weeks
    let timestamps = daily_timestamps(n);

    // Promos every 30 days
    let promo_dates: Vec<_> = (0..)
        .map(|k| k * 30)
        .take_while(|&d| d < n)
        .map(|d| timestamps[d])
        .collect();

    // Use only DOW indicators for weekly pattern (no Fourier — avoids collinearity)
    let gen = FeatureGenerator::new()
        .day_of_week()
        .holiday("promo", promo_dates);

    let features = gen.generate(&timestamps);

    // External regressor: ad spend
    let ad_spend: Vec<f64> = (0..n)
        .map(|i| 50.0 + 20.0 * (i as f64 * 0.05).sin())
        .collect();

    // Build y
    let mut y = vec![500.0; n];
    for i in 0..n {
        // Day-of-week effects
        y[i] += 10.0 * features["dow_mon"][i];
        y[i] += 8.0 * features["dow_tue"][i];
        y[i] += 6.0 * features["dow_wed"][i];
        y[i] += 4.0 * features["dow_thu"][i];
        y[i] += 12.0 * features["dow_fri"][i];
        y[i] += 15.0 * features["dow_sat"][i];
        // Promotion bump
        y[i] += 25.0 * features["holiday_promo"][i];
        // Ad spend effect
        y[i] += 0.5 * ad_spend[i];
    }

    // Attach all regressors
    let mut cal = CalendarAnnotations::new();
    for (name, vals) in &features {
        cal = cal.with_regressor(name.clone(), vals.clone());
    }
    cal = cal.with_regressor("ad_spend".to_string(), ad_spend.clone());

    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    ts.set_calendar(cal);

    (ts, gen, ad_spend)
}

#[test]
fn arima_multi_effect_residuals_near_zero() {
    let (ts, _, _) = build_multi_effect_scenario();

    let mut model = ARIMA::new(0, 0, 0); // pure regression
    model.fit(&ts).unwrap();

    // With no noise and orthogonal regressors, residuals should be ~0
    let residuals = model.residuals().unwrap();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
    assert!(
        rms < 1.0,
        "ARIMA multi-effect residual RMS should be < 1.0, got {:.4}",
        rms
    );
}

#[test]
fn arima_multi_effect_promo_recovery() {
    let (ts, gen, _ad_spend) = build_multi_effect_scenario();
    let horizon = 30;

    let mut model = ARIMA::new(0, 0, 0);
    model.fit(&ts).unwrap();

    // Generate future features: continue from training
    let n = 364;
    let all_ts = daily_timestamps(n + horizon);
    let all_features = gen.generate(&all_ts);

    let mut future_promo: HashMap<String, Vec<f64>> = HashMap::new();
    let mut future_no_promo: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_promo.insert(name.clone(), vals[n..].to_vec());
        future_no_promo.insert(name.clone(), vals[n..].to_vec());
    }

    // Ad spend for future
    let future_ad: Vec<f64> = (n..n + horizon)
        .map(|i| 50.0 + 20.0 * (i as f64 * 0.05).sin())
        .collect();
    future_promo.insert("ad_spend".to_string(), future_ad.clone());
    future_no_promo.insert("ad_spend".to_string(), future_ad);

    // Override promo: ON at day 15
    let mut promo_on = vec![0.0; horizon];
    promo_on[15] = 1.0;
    future_promo.insert("holiday_promo".to_string(), promo_on);
    future_no_promo.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    let fc_promo = model.predict_with_exog(horizon, &future_promo).unwrap();
    let fc_no_promo = model.predict_with_exog(horizon, &future_no_promo).unwrap();

    let diff = fc_promo.primary()[15] - fc_no_promo.primary()[15];
    assert!(
        (diff - 25.0).abs() < 2.0,
        "Promo effect at day 15: expected ~25, got {:.2}",
        diff
    );
}

#[test]
fn arima_multi_effect_ad_spend_recovery() {
    let (ts, gen, _) = build_multi_effect_scenario();
    let horizon = 12;

    let mut model = ARIMA::new(0, 0, 0);
    model.fit(&ts).unwrap();

    let n = 364;
    let all_ts = daily_timestamps(n + horizon);
    let all_features = gen.generate(&all_ts);

    let mut future_high_ad: HashMap<String, Vec<f64>> = HashMap::new();
    let mut future_low_ad: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, vals) in &all_features {
        future_high_ad.insert(name.clone(), vals[n..].to_vec());
        future_low_ad.insert(name.clone(), vals[n..].to_vec());
    }
    future_high_ad.insert("holiday_promo".to_string(), vec![0.0; horizon]);
    future_low_ad.insert("holiday_promo".to_string(), vec![0.0; horizon]);

    // High ad spend (100) vs low (0) → diff should be ≈ 0.5 * 100 = 50
    future_high_ad.insert("ad_spend".to_string(), vec![100.0; horizon]);
    future_low_ad.insert("ad_spend".to_string(), vec![0.0; horizon]);

    let fc_high = model.predict_with_exog(horizon, &future_high_ad).unwrap();
    let fc_low = model.predict_with_exog(horizon, &future_low_ad).unwrap();

    let expected_diff = 0.5 * 100.0;
    for i in 0..horizon {
        let diff = fc_high.primary()[i] - fc_low.primary()[i];
        assert!(
            (diff - expected_diff).abs() < 3.0,
            "Ad spend effect at h={}: expected ~{:.0}, got {:.2}",
            i + 1,
            expected_diff,
            diff
        );
    }
}

// ── Scenario 4: Model comparison — exog vs no-exog accuracy ──────────────────
//
// Generate data with a strong external effect. Fit model with and without
// the regressor. Verify that the exog-aware model has much better in-sample fit.

#[test]
fn exog_model_outperforms_no_exog() {
    // Use ARIMA(0,0,0) (pure regression) to test that including the regressor
    // gives much better predictions than omitting it.
    // Strategy: train/test split, compare OOS forecast accuracy.
    let n_total = 200;
    let n_train = 170;
    let horizon = n_total - n_train;
    let timestamps = daily_timestamps(n_total);

    let gen = FeatureGenerator::new().fourier(7, 1);
    let all_features = gen.generate(&timestamps);

    // Strong external effect
    let price: Vec<f64> = (0..n_total)
        .map(|i| 10.0 + 5.0 * (i as f64 * 0.03).sin() + (i as f64 * 0.07).cos() * 3.0)
        .collect();

    let mut y = vec![100.0; n_total];
    for i in 0..n_total {
        y[i] += 8.0 * all_features["fourier_7_sin1"][i];
        y[i] += 4.0 * all_features["fourier_7_cos1"][i];
        y[i] += 5.0 * price[i];
    }

    // ── Model WITH price ──
    let train_ts_vec = timestamps[..n_train].to_vec();
    let train_y = y[..n_train].to_vec();
    let mut cal_with = CalendarAnnotations::new();
    for (name, vals) in &all_features {
        cal_with = cal_with.with_regressor(name.clone(), vals[..n_train].to_vec());
    }
    cal_with = cal_with.with_regressor("price".to_string(), price[..n_train].to_vec());
    let mut ts_with = TimeSeries::univariate(train_ts_vec.clone(), train_y.clone()).unwrap();
    ts_with.set_calendar(cal_with);

    let mut model_with = ARIMA::new(0, 0, 0);
    model_with.fit(&ts_with).unwrap();

    let mut future_with: HashMap<String, Vec<f64>> = all_features
        .iter()
        .map(|(k, v)| (k.clone(), v[n_train..].to_vec()))
        .collect();
    future_with.insert("price".to_string(), price[n_train..].to_vec());
    let fc_with = model_with.predict_with_exog(horizon, &future_with).unwrap();
    let mae_with = mae(&y[n_train..], fc_with.primary());

    // ── Model WITHOUT price ──
    let mut cal_no_price = CalendarAnnotations::new();
    for (name, vals) in &all_features {
        cal_no_price = cal_no_price.with_regressor(name.clone(), vals[..n_train].to_vec());
    }
    let mut ts_without = TimeSeries::univariate(train_ts_vec, train_y).unwrap();
    ts_without.set_calendar(cal_no_price);

    let mut model_without = ARIMA::new(0, 0, 0);
    model_without.fit(&ts_without).unwrap();

    let future_no_price: HashMap<String, Vec<f64>> = all_features
        .iter()
        .map(|(k, v)| (k.clone(), v[n_train..].to_vec()))
        .collect();
    let fc_without = model_without
        .predict_with_exog(horizon, &future_no_price)
        .unwrap();
    let mae_without = mae(&y[n_train..], fc_without.primary());

    assert!(
        mae_with < mae_without,
        "Exog model MAE ({:.2}) should be lower than no-exog MAE ({:.2})",
        mae_with,
        mae_without,
    );

    // Price explains a large part of the signal
    assert!(
        mae_with < mae_without * 0.3,
        "Exog model should improve substantially: with={:.2}, without={:.2}",
        mae_with,
        mae_without,
    );
}

// ── Scenario 5: Out-of-sample forecast accuracy with known future effects ────
//
// Split data into train/test. Generate features for both. Verify that the
// model's forecast on the test period (with correct future regressors) matches
// the known true values closely.

#[test]
fn out_of_sample_with_known_future_effects() {
    let n_total = 200;
    let n_train = 170;
    let horizon = n_total - n_train;
    let timestamps = daily_timestamps(n_total);

    let gen = FeatureGenerator::new().fourier(7, 2);
    let all_features = gen.generate(&timestamps);

    // External regressor
    let ext: Vec<f64> = (0..n_total)
        .map(|i| 20.0 + 8.0 * (i as f64 * 0.04).sin())
        .collect();

    // y = 80 + fourier + 2.0 * ext  (no noise → should forecast perfectly)
    let mut y = vec![80.0; n_total];
    for i in 0..n_total {
        y[i] += 6.0 * all_features["fourier_7_sin1"][i];
        y[i] += 3.0 * all_features["fourier_7_cos1"][i];
        y[i] -= 2.0 * all_features["fourier_7_sin2"][i];
        y[i] += 1.0 * all_features["fourier_7_cos2"][i];
        y[i] += 2.0 * ext[i];
    }

    // Train split
    let train_timestamps = timestamps[..n_train].to_vec();
    let train_y = y[..n_train].to_vec();
    let train_ext = ext[..n_train].to_vec();
    let train_features: HashMap<String, Vec<f64>> = all_features
        .iter()
        .map(|(k, v)| (k.clone(), v[..n_train].to_vec()))
        .collect();

    let mut cal = CalendarAnnotations::new();
    for (name, vals) in &train_features {
        cal = cal.with_regressor(name.clone(), vals.clone());
    }
    cal = cal.with_regressor("ext".to_string(), train_ext);

    let mut ts_train = TimeSeries::univariate(train_timestamps, train_y).unwrap();
    ts_train.set_calendar(cal);

    // Fit
    let mut model = ARIMA::new(0, 0, 0);
    model.fit(&ts_train).unwrap();

    // Future regressors
    let mut future_regs: HashMap<String, Vec<f64>> = all_features
        .iter()
        .map(|(k, v)| (k.clone(), v[n_train..].to_vec()))
        .collect();
    future_regs.insert("ext".to_string(), ext[n_train..].to_vec());

    let forecast = model.predict_with_exog(horizon, &future_regs).unwrap();

    let actual_test = &y[n_train..];
    let error = rmse(actual_test, forecast.primary());
    assert!(
        error < 1.0,
        "OOS RMSE with known future regressors should be < 1.0, got {:.4}",
        error,
    );
}

// ── Scenario 6: Naive model with feature-generated regressors ────────────────
//
// Even the Naive model (which uses pre-regression OLS) should recover the
// regressor effect via predict_with_exog.

#[test]
fn naive_recovers_promotion_effect_via_features() {
    let n = 200;
    let timestamps = daily_timestamps(n);

    let promo_dates: Vec<_> = (0..)
        .map(|k| k * 20)
        .take_while(|&d| d < n)
        .map(|d| timestamps[d])
        .collect();

    let gen = FeatureGenerator::new().holiday("sale", promo_dates);
    let features = gen.generate(&timestamps);

    let mut y = vec![50.0; n];
    for i in 0..n {
        y[i] += 40.0 * features["holiday_sale"][i];
    }

    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    gen.add_to(&mut ts);

    let mut model = Naive::new();
    model.fit(&ts).unwrap();
    assert!(model.has_exog());

    let horizon = 20;
    let mut future_on: HashMap<String, Vec<f64>> = HashMap::new();
    let mut sale_col = vec![0.0; horizon];
    sale_col[10] = 1.0;
    future_on.insert("holiday_sale".to_string(), sale_col);

    let mut future_off: HashMap<String, Vec<f64>> = HashMap::new();
    future_off.insert("holiday_sale".to_string(), vec![0.0; horizon]);

    let fc_on = model.predict_with_exog(horizon, &future_on).unwrap();
    let fc_off = model.predict_with_exog(horizon, &future_off).unwrap();

    let diff = fc_on.primary()[10] - fc_off.primary()[10];
    assert!(
        (diff - 40.0).abs() < 2.0,
        "Naive promo effect: expected ~40, got {:.2}",
        diff,
    );
}
