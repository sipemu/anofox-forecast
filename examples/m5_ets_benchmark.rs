//! M5 ETS Benchmark: Complete vs Reduced model pool.
//!
//! Compares AutoETS accuracy and speed on the full M5 retail dataset (~30,490 series)
//! using the Complete (19 models) and Reduced (8 models) pools from
//! Petropoulos et al. (2023) "Wielding Occam's razor".
//!
//! Run: cargo run --release --all-features --example m5_ets_benchmark
//!
//! Requires: validation/data/m5_full.csv (generated via datasetsforecast)

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, ModelPool};
use anofox_forecast::models::Forecaster;
use chrono::{NaiveDate, TimeZone, Utc};
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const HORIZON: usize = 28; // M5 competition horizon (4 weeks)
const PERIOD: usize = 7; // weekly seasonality
const MIN_NONZERO_FRAC: f64 = 0.0; // include all series

fn main() {
    // Try full dataset first, fall back to top-1000
    let path = if fs::metadata("validation/data/m5_full.csv").is_ok() {
        "validation/data/m5_full.csv"
    } else {
        eprintln!("Full M5 not found, falling back to m5_top1000.csv");
        "validation/data/m5_top1000.csv"
    };

    eprintln!("Loading {}...", path);
    let load_start = Instant::now();
    let content = fs::read_to_string(path).expect("Failed to read M5 CSV");
    let mut lines = content.lines();

    // Parse header
    let header = lines.next().expect("Empty CSV");
    let col_names: Vec<&str> = header.split(',').collect();
    let n_series = col_names.len() - 1;

    // Parse data rows
    let mut dates: Vec<chrono::DateTime<Utc>> = Vec::with_capacity(2000);
    let mut data: Vec<Vec<f64>> = vec![Vec::with_capacity(2000); n_series];

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < n_series + 1 {
            continue;
        }
        if let Ok(nd) = NaiveDate::parse_from_str(fields[0], "%Y-%m-%d") {
            dates.push(Utc.from_utc_datetime(&nd.and_hms_opt(0, 0, 0).unwrap()));
        } else {
            continue;
        }
        for (j, field) in fields.iter().enumerate().skip(1) {
            let val: f64 = field.parse().unwrap_or(0.0);
            data[j - 1].push(val);
        }
    }

    let n_total = dates.len();
    let n_train = n_total - HORIZON;
    let load_secs = load_start.elapsed().as_secs_f64();
    eprintln!("Loaded in {:.1}s\n", load_secs);

    println!(
        "M5 dataset: {} series, {} observations ({} train + {} test)",
        n_series, n_total, n_train, HORIZON
    );

    // Filter to series with sufficient non-zero values
    let eligible: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, series)| {
            let nonzero_frac =
                series[..n_train].iter().filter(|&&v| v > 0.0).count() as f64 / n_train as f64;
            nonzero_frac >= MIN_NONZERO_FRAC
        })
        .map(|(i, _)| i)
        .collect();

    println!(
        "Eligible series (>{}% nonzero): {} of {}",
        (MIN_NONZERO_FRAC * 100.0) as usize,
        eligible.len(),
        n_series
    );
    println!("Evaluating: {} series\n", eligible.len());

    let train_dates = dates[..n_train].to_vec();

    // Run benchmark for both pools
    let pools = [
        ("Complete", ModelPool::Complete),
        ("Reduced", ModelPool::Reduced),
    ];

    for (pool_name, pool) in &pools {
        println!("=== AutoETS pool: {} ===", pool_name);

        let total_time_ms = Mutex::new(0.0f64);
        let rmse_values = Mutex::new(Vec::new());
        let mape_values = Mutex::new(Vec::new());
        let smape_values = Mutex::new(Vec::new());
        let n_success = AtomicUsize::new(0);
        let n_fail = AtomicUsize::new(0);
        let selected_models = Mutex::new(HashMap::<String, usize>::new());
        let progress = AtomicUsize::new(0);

        let pool_start = Instant::now();

        // Process series — use rayon if available
        let process_one = |idx: &usize| {
            let idx = *idx;
            let train_vals = data[idx][..n_train].to_vec();
            let test_vals = &data[idx][n_train..n_train + HORIZON];

            let ts = match TimeSeries::univariate(train_dates.clone(), train_vals) {
                Ok(ts) => ts,
                Err(_) => {
                    n_fail.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };

            let config = AutoETSConfig::with_period(PERIOD).with_model_pool(*pool);
            let mut model = AutoETS::with_config(config);

            let start = Instant::now();
            let fit_ok = model.fit(&ts).is_ok();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            if !fit_ok {
                n_fail.fetch_add(1, Ordering::Relaxed);
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(500) {
                    eprintln!("  [{}/{}]", done, eligible.len());
                }
                return;
            }

            match model.predict(HORIZON) {
                Ok(fc) => {
                    let preds = fc.primary();
                    n_success.fetch_add(1, Ordering::Relaxed);
                    *total_time_ms.lock().unwrap() += elapsed;

                    if let Some(spec) = model.selected_spec() {
                        let key = format!("{:?}", spec);
                        *selected_models.lock().unwrap().entry(key).or_insert(0) += 1;
                    }

                    // RMSE
                    let mse: f64 = preds
                        .iter()
                        .zip(test_vals.iter())
                        .map(|(p, a)| (p - a).powi(2))
                        .sum::<f64>()
                        / HORIZON as f64;
                    rmse_values.lock().unwrap().push(mse.sqrt());

                    // MAPE
                    let mape_pairs: Vec<f64> = preds
                        .iter()
                        .zip(test_vals.iter())
                        .filter(|(_, a)| **a > 0.0)
                        .map(|(p, a)| ((p - a) / a).abs())
                        .collect();
                    if !mape_pairs.is_empty() {
                        mape_values
                            .lock()
                            .unwrap()
                            .push(mape_pairs.iter().sum::<f64>() / mape_pairs.len() as f64);
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
                            .lock()
                            .unwrap()
                            .push(smape_pairs.iter().sum::<f64>() / smape_pairs.len() as f64);
                    }
                }
                Err(_) => {
                    n_fail.fetch_add(1, Ordering::Relaxed);
                }
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(500) {
                let succ = n_success.load(Ordering::Relaxed);
                let fail = n_fail.load(Ordering::Relaxed);
                let wall = pool_start.elapsed().as_secs_f64();
                eprintln!(
                    "  [{}/{}] {} success, {} fail, {:.0}s wall",
                    done,
                    eligible.len(),
                    succ,
                    fail,
                    wall
                );
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            eligible.par_iter().for_each(process_one);
        }

        #[cfg(not(feature = "parallel"))]
        {
            eligible.iter().for_each(process_one);
        }

        let wall_time = pool_start.elapsed().as_secs_f64();
        let succ = n_success.load(Ordering::Relaxed);
        let fail = n_fail.load(Ordering::Relaxed);
        let total_ms = *total_time_ms.lock().unwrap();
        let rmse_vals = rmse_values.into_inner().unwrap();
        let mape_vals = mape_values.into_inner().unwrap();
        let smape_vals = smape_values.into_inner().unwrap();

        let avg_rmse = rmse_vals.iter().sum::<f64>() / rmse_vals.len() as f64;
        let med_rmse = {
            let mut sorted = rmse_vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let avg_mape = if mape_vals.is_empty() {
            f64::NAN
        } else {
            mape_vals.iter().sum::<f64>() / mape_vals.len() as f64
        };
        let avg_smape = if smape_vals.is_empty() {
            f64::NAN
        } else {
            smape_vals.iter().sum::<f64>() / smape_vals.len() as f64
        };

        println!("  Series: {} success, {} failed", succ, fail);
        println!("  Wall-clock time: {:.1} s", wall_time);
        println!("  Avg CPU time:    {:.2} ms/series", total_ms / succ as f64);
        println!("  Total CPU time:  {:.1} s", total_ms / 1000.0);
        println!("  Avg RMSE:        {:.4}", avg_rmse);
        println!("  Median RMSE:     {:.4}", med_rmse);
        println!("  Avg MAPE:        {:.2}%", avg_mape * 100.0);
        println!("  Avg sMAPE:       {:.2}%", avg_smape * 100.0);

        let mut model_counts: Vec<_> = selected_models.into_inner().unwrap().into_iter().collect();
        model_counts.sort_by_key(|b| std::cmp::Reverse(b.1));
        println!("  Top selected models:");
        for (model, count) in model_counts.iter().take(7) {
            println!(
                "    {:>5} ({:>5.1}%) {}",
                count,
                *count as f64 / succ as f64 * 100.0,
                model
            );
        }
        println!();
    }
}
