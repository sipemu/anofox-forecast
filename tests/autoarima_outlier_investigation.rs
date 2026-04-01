//! Investigate AutoARIMA outlier series from M4 benchmark.
//! Compares default (reduced) vs full candidate set.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::time::Instant;

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn mae(pred: &[f64], actual: &[f64]) -> f64 {
    let n = pred.len().min(actual.len());
    pred[..n]
        .iter()
        .zip(&actual[..n])
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / n as f64
}

#[test]
fn investigate_outliers() {
    let content = std::fs::read_to_string("tests/data/m4_outliers.json")
        .expect("Run the Python extraction script first");
    let data: HashMap<String, serde_json::Value> = serde_json::from_str(&content).unwrap();

    let period = 7;
    let horizon = 14;

    println!("\n=== AutoARIMA Outlier Investigation ===");
    println!(
        "{:<8} {:>6} {:>10} {:>10} {:>10}  {:>12} {:>12}  {}",
        "Series",
        "n",
        "MAE_def",
        "MAE_full",
        "Improve%",
        "Order_def",
        "Order_full",
        "Time_def/full"
    );

    for (uid, series_data) in &data {
        let train: Vec<f64> = series_data["train"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let test: Vec<f64> = series_data["test"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let n = train.len();
        let timestamps = make_timestamps(n);
        let ts = TimeSeries::univariate(timestamps, train).unwrap();

        // Default config (v0.5.5: max_cap_p=1, max_cap_q=1)
        let start = Instant::now();
        let config_default = AutoARIMAConfig {
            seasonal_period: period,
            ..Default::default()
        };
        let mut model_default = AutoARIMA::with_config(config_default);
        model_default.fit(&ts).unwrap();
        let fc_default = model_default.predict(horizon).unwrap();
        let time_default = start.elapsed().as_secs_f64() * 1000.0;
        let mae_default = mae(fc_default.primary(), &test);
        let order_default = model_default.selected_order();

        // Full config (max_cap_p=2, max_cap_q=2, max_p=5, max_q=5)
        let start = Instant::now();
        let config_full = AutoARIMAConfig {
            seasonal_period: period,
            max_p: 5,
            max_q: 5,
            max_cap_p: 2,
            max_cap_q: 2,
            ..Default::default()
        };
        let mut model_full = AutoARIMA::with_config(config_full);
        model_full.fit(&ts).unwrap();
        let fc_full = model_full.predict(horizon).unwrap();
        let time_full = start.elapsed().as_secs_f64() * 1000.0;
        let mae_full = mae(fc_full.primary(), &test);
        let order_full = model_full.selected_order();

        let improvement = if mae_default > 0.0 {
            (mae_default - mae_full) / mae_default * 100.0
        } else {
            0.0
        };

        println!(
            "{:<8} {:>6} {:>10.1} {:>10.1} {:>9.1}%  {:>12?} {:>12?}  {:.0}/{:.0}ms",
            uid,
            n,
            mae_default,
            mae_full,
            improvement,
            order_default,
            order_full,
            time_default,
            time_full
        );
    }
}
