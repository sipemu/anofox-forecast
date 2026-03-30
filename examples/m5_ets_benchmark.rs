//! M5 ETS Benchmark: Complete vs Reduced model pool.
//!
//! Compares AutoETS accuracy and speed on M5 retail data (top-1000 series)
//! using the Complete (19 models) and Reduced (8 models) pools from
//! Petropoulos et al. (2023) "Wielding Occam's razor".
//!
//! Run: cargo run --release --all-features --example m5_ets_benchmark

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, ModelPool};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, NaiveDate, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 28; // M5 competition horizon (4 weeks)
const PERIOD: usize = 7; // weekly seasonality
const MIN_NONZERO_FRAC: f64 = 0.3; // skip very intermittent series
const MAX_SERIES: usize = 1000; // full M5 top-1000 dataset

fn main() {
    let path = "validation/data/m5_top1000.csv";
    let content = fs::read_to_string(path).expect("Failed to read M5 CSV");
    let mut lines = content.lines();

    // Parse header
    let header = lines.next().expect("Empty CSV");
    let col_names: Vec<&str> = header.split(',').collect();
    let n_series = col_names.len() - 1; // exclude date column

    // Parse data rows
    let mut dates: Vec<chrono::DateTime<Utc>> = Vec::new();
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); n_series];

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < n_series + 1 {
            continue;
        }

        // Parse date
        if let Ok(nd) = NaiveDate::parse_from_str(fields[0], "%Y-%m-%d") {
            dates.push(Utc.from_utc_datetime(&nd.and_hms_opt(0, 0, 0).unwrap()));
        } else {
            continue;
        }

        // Parse values
        for (j, field) in fields.iter().enumerate().skip(1) {
            let val: f64 = field.parse().unwrap_or(0.0);
            data[j - 1].push(val);
        }
    }

    let n_total = dates.len();
    let n_train = n_total - HORIZON;
    println!(
        "M5 dataset: {} series, {} observations ({} train + {} test)",
        n_series, n_total, n_train, HORIZON
    );

    // Filter to series with sufficient non-zero values
    let mut eligible: Vec<usize> = Vec::new();
    for (i, series) in data.iter().enumerate() {
        let nonzero_frac =
            series[..n_train].iter().filter(|&&v| v > 0.0).count() as f64 / n_train as f64;
        if nonzero_frac >= MIN_NONZERO_FRAC {
            eligible.push(i);
        }
    }
    println!(
        "Eligible series (>{}% nonzero): {} of {}",
        (MIN_NONZERO_FRAC * 100.0) as usize,
        eligible.len(),
        n_series
    );

    let n_eval = eligible.len().min(MAX_SERIES);
    let eval_indices = &eligible[..n_eval];
    println!("Evaluating: {} series\n", n_eval);

    // Run benchmark for both pools
    let pools = [
        ("Complete", ModelPool::Complete),
        ("Reduced", ModelPool::Reduced),
    ];

    let train_dates = dates[..n_train].to_vec();

    for (pool_name, pool) in &pools {
        println!("=== AutoETS pool: {} ===", pool_name);

        let mut total_time_ms = 0.0;
        let mut rmse_values = Vec::new();
        let mut mape_values = Vec::new();
        let mut smape_values = Vec::new();
        let mut n_success = 0usize;
        let mut n_fail = 0usize;
        let mut selected_models: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for (count, &idx) in eval_indices.iter().enumerate() {
            let train_vals = data[idx][..n_train].to_vec();
            let test_vals = &data[idx][n_train..n_train + HORIZON];

            let ts = match TimeSeries::univariate(train_dates.clone(), train_vals) {
                Ok(ts) => ts,
                Err(_) => {
                    n_fail += 1;
                    continue;
                }
            };

            let config = AutoETSConfig::with_period(PERIOD).with_model_pool(*pool);
            let mut model = AutoETS::with_config(config);

            let start = Instant::now();
            let fit_ok = model.fit(&ts).is_ok();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            if !fit_ok {
                n_fail += 1;
                continue;
            }

            total_time_ms += elapsed;

            match model.predict(HORIZON) {
                Ok(fc) => {
                    let preds = fc.primary();
                    n_success += 1;

                    // Track selected model
                    if let Some(spec) = model.selected_spec() {
                        let key = format!("{:?}", spec);
                        *selected_models.entry(key).or_insert(0) += 1;
                    }

                    // RMSE
                    let mse: f64 = preds
                        .iter()
                        .zip(test_vals.iter())
                        .map(|(p, a)| (p - a).powi(2))
                        .sum::<f64>()
                        / HORIZON as f64;
                    rmse_values.push(mse.sqrt());

                    // MAPE (skip zeros in actual)
                    let mape_pairs: Vec<f64> = preds
                        .iter()
                        .zip(test_vals.iter())
                        .filter(|(_, a)| **a > 0.0)
                        .map(|(p, a)| ((p - a) / a).abs())
                        .collect();
                    if !mape_pairs.is_empty() {
                        mape_values.push(mape_pairs.iter().sum::<f64>() / mape_pairs.len() as f64);
                    }

                    // sMAPE
                    let smape_pairs: Vec<f64> = preds
                        .iter()
                        .zip(test_vals.iter())
                        .filter(|(p, a)| **p + **a > 0.0)
                        .map(|(p, a)| 2.0 * (p - a).abs() / (p.abs() + a.abs()))
                        .collect();
                    if !smape_pairs.is_empty() {
                        smape_values
                            .push(smape_pairs.iter().sum::<f64>() / smape_pairs.len() as f64);
                    }
                }
                Err(_) => {
                    n_fail += 1;
                }
            }

            // Progress
            if (count + 1) % 50 == 0 {
                eprintln!(
                    "  [{}/{}] {}: {} success, {} fail, {:.1}ms avg",
                    count + 1,
                    n_eval,
                    pool_name,
                    n_success,
                    n_fail,
                    total_time_ms / n_success as f64
                );
            }
        }

        // Summary statistics
        let avg_rmse = rmse_values.iter().sum::<f64>() / rmse_values.len() as f64;
        let med_rmse = {
            let mut sorted = rmse_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let avg_mape = if mape_values.is_empty() {
            f64::NAN
        } else {
            mape_values.iter().sum::<f64>() / mape_values.len() as f64
        };
        let avg_smape = if smape_values.is_empty() {
            f64::NAN
        } else {
            smape_values.iter().sum::<f64>() / smape_values.len() as f64
        };
        let avg_time = total_time_ms / n_success as f64;
        let total_time_s = total_time_ms / 1000.0;

        println!("  Series: {} success, {} failed", n_success, n_fail);
        println!("  Avg fit time:   {:.2} ms", avg_time);
        println!("  Total time:     {:.2} s", total_time_s);
        println!("  Avg RMSE:       {:.4}", avg_rmse);
        println!("  Median RMSE:    {:.4}", med_rmse);
        println!("  Avg MAPE:       {:.2}%", avg_mape * 100.0);
        println!("  Avg sMAPE:      {:.2}%", avg_smape * 100.0);

        // Top 5 selected models
        let mut model_counts: Vec<_> = selected_models.into_iter().collect();
        model_counts.sort_by(|a, b| b.1.cmp(&a.1));
        println!("  Top selected models:");
        for (model, count) in model_counts.iter().take(5) {
            println!(
                "    {:>4} ({:>5.1}%) {}",
                count,
                *count as f64 / n_success as f64 * 100.0,
                model
            );
        }
        println!();
    }
}
