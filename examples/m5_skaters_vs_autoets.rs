//! M5 200-series bench: `LaplaceForecaster::skaters()` vs `AutoETS`.
//!
//! Same protocol as `m5_wape.rs`: for each series in a 200-sample of
//! M5, train on all-but-last-28 days, predict 28-day horizon, score
//! MAE / WAPE / MASE against the actual test window. Reports wall
//! time for each forecaster.
//!
//! Run: `cargo run --release --features distributional --example m5_skaters_vs_autoets`
//! Env: `SAMPLE_SIZE=200` (default), `HORIZON=28`, `MIN_LEN=60`.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::{Forecaster, LaplaceForecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 28;
const MIN_LEN: usize = 60;

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

fn abs_error_sum(pred: &[f64], truth: &[f64]) -> f64 {
    pred.iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum()
}

/// Seasonal-naive scale for MASE: `mean |y_t - y_{t-period}|` on train.
fn mase_scale(train: &[f64], period: usize) -> f64 {
    let p = if train.len() > period { period } else { 1 };
    if train.len() <= p {
        return 1.0;
    }
    let n = train.len() - p;
    let sum: f64 = (p..train.len())
        .map(|i| (train[i] - train[i - p]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

fn main() {
    let sample: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
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
        parts.next();
        for (i, tok) in parts.enumerate() {
            let v: f64 = tok.trim().parse::<f64>().unwrap_or(f64::NAN);
            cols[i].push(v);
        }
    }

    // Trim leading NaN, reject interior NaN, require enough obs.
    let mut kept: Vec<Vec<f64>> = names
        .iter()
        .zip(cols)
        .filter_map(|(_, v)| {
            let start = v.iter().position(|x| !x.is_nan())?;
            let tail = &v[start..];
            if tail.iter().any(|x| x.is_nan()) {
                return None;
            }
            if tail.len() < MIN_LEN {
                return None;
            }
            Some(tail.to_vec())
        })
        .collect();
    kept.truncate(sample);
    let n_series = kept.len();
    eprintln!("Running on {n_series} series (H={HORIZON})");

    // Result accumulators — per model.
    let mut results = [
        ("AutoETS", vec![], 0.0, 0.0, 0u128),
        ("Laplace + skaters()", vec![], 0.0, 0.0, 0u128),
        ("Laplace + auto()", vec![], 0.0, 0.0, 0u128),
    ];
    // Tuple: (name, per_series_mase, cumulative_abs_err, cumulative_abs_truth, total_us)

    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

    let global_start = Instant::now();
    for (idx, values) in kept.iter().enumerate() {
        if idx % 20 == 0 {
            eprintln!(
                "[{}/{}] — elapsed {:.1}s",
                idx,
                n_series,
                global_start.elapsed().as_secs_f64()
            );
        }
        if values.len() <= HORIZON + 20 {
            continue;
        }
        let split = values.len() - HORIZON;
        let train_v = values[..split].to_vec();
        let test_v: Vec<f64> = values[split..].to_vec();
        let y_abs_sum: f64 = test_v.iter().map(|y| y.abs()).sum();
        if y_abs_sum < 1e-9 {
            continue; // all-zero test — WAPE undefined.
        }
        // MASE scale: use M5 canonical weekly period (7) on daily counts.
        let scale = mase_scale(&train_v, 7);
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Model 0: AutoETS
        let t0 = Instant::now();
        {
            let mut m = AutoETS::new();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(HORIZON) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        let abs_err = abs_error_sum(p, &test_v);
                        results[0].1.push(mae(p, &test_v) / scale);
                        results[0].2 += abs_err;
                        results[0].3 += y_abs_sum;
                    }
                }
            }
        }
        results[0].4 += t0.elapsed().as_micros();

        // Model 1: Laplace + skaters()
        let t0 = Instant::now();
        {
            let mut m = LaplaceForecaster::new().skaters();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(HORIZON) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        let abs_err = abs_error_sum(p, &test_v);
                        results[1].1.push(mae(p, &test_v) / scale);
                        results[1].2 += abs_err;
                        results[1].3 += y_abs_sum;
                    }
                }
            }
        }
        results[1].4 += t0.elapsed().as_micros();

        // Model 2: Laplace + auto() — reference for our lighter selector.
        let t0 = Instant::now();
        {
            let mut m = LaplaceForecaster::new().auto();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(HORIZON) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        let abs_err = abs_error_sum(p, &test_v);
                        results[2].1.push(mae(p, &test_v) / scale);
                        results[2].2 += abs_err;
                        results[2].3 += y_abs_sum;
                    }
                }
            }
        }
        results[2].4 += t0.elapsed().as_micros();
    }

    println!(
        "\n=== M5 {} series × H={} — Laplace variants vs AutoETS ===",
        n_series, HORIZON
    );
    println!(
        "{:<22}{:>8}{:>12}{:>12}{:>12}",
        "model", "n", "mean MASE", "panel WAPE", "fit (s)"
    );
    for r in &results {
        let n = r.1.len();
        let mean_mase = if n > 0 {
            r.1.iter().sum::<f64>() / n as f64
        } else {
            f64::NAN
        };
        let panel_wape = r.2 / r.3.max(1e-9);
        let fit_s = r.4 as f64 / 1_000_000.0;
        println!(
            "{:<22}{:>8}{:>12.4}{:>12.4}{:>12.1}",
            r.0, n, mean_mase, panel_wape, fit_s
        );
    }
}
