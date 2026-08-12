//! WAPE (Weighted Absolute Percentage Error) of the Laplace estimators on M5.
//!
//! WAPE per series: `Σ|y_t − ŷ_t| / Σ|y_t|`.
//! Aggregated across the panel two ways:
//! - Per-series WAPE, arithmetic mean across series ("mean WAPE").
//! - Pooled WAPE: `Σ_series Σ_t |y_t − ŷ_t| / Σ_series Σ_t |y_t|` ("panel WAPE").
//!
//! Runs on `validation/data/m5_full.csv` (30,490 series × 1970 daily obs).
//! Uses the last 28 days as test window (M5 competition horizon).
//!
//! Run: `cargo run --release --features distributional --example m5_wape`
//! Configure: `SAMPLE_SIZE=1000` to limit series.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::{Forecaster, LaplaceForecaster, SmartForecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 28;
const MIN_LEN: usize = 60;

fn main() {
    let sample: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let path = std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/m5_full.csv".into());

    eprintln!("Loading {path}...");
    let content = fs::read_to_string(&path).expect("read CSV");
    let mut lines = content.lines();
    let header = lines.next().expect("empty CSV");
    let names: Vec<&str> = header.split(',').skip(1).collect();
    let n_total = names.len();

    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(2000); n_total];
    for line in lines {
        let mut parts = line.split(',');
        parts.next(); // skip date
        for (i, tok) in parts.enumerate() {
            let v: f64 = tok.parse().unwrap_or(0.0);
            cols[i].push(v);
        }
    }

    let mut kept: Vec<Vec<f64>> = cols.into_iter().filter(|v| v.len() >= MIN_LEN).collect();
    kept.truncate(sample);
    let n_series = kept.len();
    eprintln!("Running on {n_series} series (H={HORIZON})");

    // Per-series results — Laplace variants only.
    let mut results = [
        ("Laplace+auto", vec![], vec![], vec![], 0u128),
        ("Laplace+auto_aid", vec![], vec![], vec![], 0u128),
        ("SmartForecaster", vec![], vec![], vec![], 0u128),
    ];
    // Each tuple: (name, per_series_wape, abs_err_sums, y_abs_sums, total_us)

    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

    for (idx, values) in kept.iter().enumerate() {
        if idx % 100 == 0 {
            eprintln!("[{idx}/{n_series}]");
        }
        if values.len() <= HORIZON + 20 {
            continue;
        }
        let split = values.len() - HORIZON;
        let train_v = values[..split].to_vec();
        let test_v: Vec<f64> = values[split..].to_vec();
        let y_abs_sum: f64 = test_v.iter().map(|y| y.abs()).sum();
        if y_abs_sum < 1e-9 {
            // Skip series with all-zero test window — WAPE undefined.
            continue;
        }
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Model 0: Laplace + auto
        let t0 = Instant::now();
        let mut m = LaplaceForecaster::new().auto();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                let p = fc.primary();
                if p.len() == test_v.len() {
                    let abs_err: f64 = p
                        .iter()
                        .zip(test_v.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    results[0].1.push(abs_err / y_abs_sum);
                    results[0].2.push(abs_err);
                    results[0].3.push(y_abs_sum);
                }
            }
        }
        results[0].4 += t0.elapsed().as_micros();

        // Model 1: Laplace + auto_aid
        let t0 = Instant::now();
        let mut m = LaplaceForecaster::new().auto_aid();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                let p = fc.primary();
                if p.len() == test_v.len() {
                    let abs_err: f64 = p
                        .iter()
                        .zip(test_v.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    results[1].1.push(abs_err / y_abs_sum);
                    results[1].2.push(abs_err);
                    results[1].3.push(y_abs_sum);
                }
            }
        }
        results[1].4 += t0.elapsed().as_micros();

        // Model 2: SmartForecaster
        let t0 = Instant::now();
        let mut m = SmartForecaster::new();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                let p = fc.primary();
                if p.len() == test_v.len() {
                    let abs_err: f64 = p
                        .iter()
                        .zip(test_v.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    results[2].1.push(abs_err / y_abs_sum);
                    results[2].2.push(abs_err);
                    results[2].3.push(y_abs_sum);
                }
            }
        }
        results[2].4 += t0.elapsed().as_micros();
    }

    println!("\n=== M5 WAPE ({n_series} series, horizon {HORIZON}) ===");
    println!(
        "{:<20}{:>8}{:>14}{:>14}{:>12}",
        "model", "n", "mean WAPE", "panel WAPE", "fit (s)"
    );
    for r in &results {
        let n = r.1.len();
        let mean_wape = if n > 0 {
            r.1.iter().sum::<f64>() / n as f64
        } else {
            f64::NAN
        };
        let panel_wape: f64 = r.2.iter().sum::<f64>() / r.3.iter().sum::<f64>().max(1e-9);
        let fit_s = r.4 as f64 / 1_000_000.0;
        println!(
            "{:<20}{:>8}{:>14.4}{:>14.4}{:>12.1}",
            r.0, n, mean_wape, panel_wape, fit_s
        );
    }
}
