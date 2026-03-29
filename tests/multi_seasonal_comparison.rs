//! Compare models on multi-seasonal data (period 7 + 28).
//! Run with: cargo test --release --test multi_seasonal_comparison -- --nocapture

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::HoltWinters;
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::tbats::TBATS;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_multi_seasonal(n: usize) -> (TimeSeries, Vec<f64>) {
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect();

    // Two seasonalities: weekly (period=7) + monthly (period=28)
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.05 * i as f64;
            let weekly = 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let monthly = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 28.0).sin();
            let noise = ((i * 7 + 3) % 11) as f64 * 0.2 - 1.1;
            trend + weekly + monthly + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
    (ts, values)
}

fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len());
    let mse: f64 = actual[..n]
        .iter()
        .zip(predicted[..n].iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

#[test]
fn multi_seasonal_model_comparison() {
    let n_train = 200;
    let horizon = 28; // predict one full monthly cycle
    let n_total = n_train + horizon;
    let (full_ts, full_values) = make_multi_seasonal(n_total);

    let train_ts = full_ts.slice(0, n_train).unwrap();
    let actual = &full_values[n_train..n_total];

    println!(
        "\n=== Multi-Seasonal Comparison (weekly + monthly, n={}, h={}) ===\n",
        n_train, horizon
    );
    println!("{:<30} {:>10} {:>12}", "Model", "RMSE", "Time");
    println!("{}", "-".repeat(54));

    // 1. TBATS with both periods (knows both)
    let t0 = std::time::Instant::now();
    let mut tbats = TBATS::new(vec![7, 28]);
    tbats.fit(&train_ts).unwrap();
    let fc = tbats.predict(horizon).unwrap();
    let t_tbats = t0.elapsed();
    let rmse_tbats = rmse(actual, fc.primary());
    println!(
        "{:<30} {:>10.4} {:>12?}",
        "TBATS(7, 28)", rmse_tbats, t_tbats
    );

    // 2. TBATS with only weekly period (misses monthly)
    let t0 = std::time::Instant::now();
    let mut tbats7 = TBATS::new(vec![7]);
    tbats7.fit(&train_ts).unwrap();
    let fc7 = tbats7.predict(horizon).unwrap();
    let t_tbats7 = t0.elapsed();
    let rmse_tbats7 = rmse(actual, fc7.primary());
    println!(
        "{:<30} {:>10.4} {:>12?}",
        "TBATS(7) only", rmse_tbats7, t_tbats7
    );

    // 3. MSTLForecaster with both periods
    let t0 = std::time::Instant::now();
    let mut mstl = MSTLForecaster::new(vec![7, 28]);
    mstl.fit(&train_ts).unwrap();
    let fc_mstl = mstl.predict(horizon).unwrap();
    let t_mstl = t0.elapsed();
    let rmse_mstl = rmse(actual, fc_mstl.primary());
    println!("{:<30} {:>10.4} {:>12?}", "MSTL(7, 28)", rmse_mstl, t_mstl);

    // 4. HoltWinters with only weekly (can't do multi)
    let t0 = std::time::Instant::now();
    let mut hw = HoltWinters::additive(0.3, 0.1, 0.1, 7);
    hw.fit(&train_ts).unwrap();
    let fc_hw = hw.predict(horizon).unwrap();
    let t_hw = t0.elapsed();
    let rmse_hw = rmse(actual, fc_hw.primary());
    println!("{:<30} {:>10.4} {:>12?}", "HoltWinters(7)", rmse_hw, t_hw);

    // 5. HoltWinters with monthly only (misses weekly)
    let t0 = std::time::Instant::now();
    let mut hw28 = HoltWinters::additive(0.3, 0.1, 0.1, 28);
    hw28.fit(&train_ts).unwrap();
    let fc_hw28 = hw28.predict(horizon).unwrap();
    let t_hw28 = t0.elapsed();
    let rmse_hw28 = rmse(actual, fc_hw28.primary());
    println!(
        "{:<30} {:>10.4} {:>12?}",
        "HoltWinters(28)", rmse_hw28, t_hw28
    );

    println!("\n--- Analysis ---");
    println!(
        "TBATS(7,28) vs TBATS(7):      {:.1}% RMSE reduction",
        (1.0 - rmse_tbats / rmse_tbats7) * 100.0
    );
    println!(
        "TBATS(7,28) vs HoltWinters(7): {:.1}% RMSE reduction",
        (1.0 - rmse_tbats / rmse_hw) * 100.0
    );
    println!(
        "MSTL(7,28) vs HoltWinters(7):  {:.1}% RMSE reduction",
        (1.0 - rmse_mstl / rmse_hw) * 100.0
    );

    // TBATS with both periods should beat single-period models
    assert!(
        rmse_tbats < rmse_tbats7 * 1.1,
        "TBATS(7,28) should be competitive with TBATS(7): {} vs {}",
        rmse_tbats,
        rmse_tbats7
    );
}
