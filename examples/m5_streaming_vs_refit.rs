//! Demonstrates the O(1) streaming `observe()` speedup over re-fitting.
//!
//! Two paths on the same 200-series M5 sample:
//!
//! A. **Re-fit path**: for each new observation in the last 500 steps,
//!    call `fit()` on the growing window — the "batch" pattern that
//!    treats every prediction as a fresh problem.
//! B. **Streaming path**: `fit()` once on the initial window, then
//!    `observe(y)` for each new observation. This is the O(N_leaves)
//!    incremental primitive that matches skaters' `f(y, state)` shape.
//!
//! Both produce identical forecasts (verified in the unit test
//! `streaming_observe_matches_batch_fit`); the difference is wall time.
//!
//! Run: `cargo run --release --features distributional --example m5_streaming_vs_refit`
//! Env: `SAMPLE_SIZE=50`, `INIT_WINDOW=1400`, `N_STREAMED=500`.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::{Forecaster, LaplaceForecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

fn main() {
    let sample: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let init_window: usize = std::env::var("INIT_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1400);
    let n_streamed: usize = std::env::var("N_STREAMED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let path = std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/m5_full.csv".into());

    eprintln!(
        "M5 streaming vs re-fit — sample={sample} series, init={init_window}, streamed={n_streamed}"
    );
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
    let mut kept: Vec<Vec<f64>> = names
        .iter()
        .zip(cols)
        .filter_map(|(_, v)| {
            let start = v.iter().position(|x| !x.is_nan())?;
            let tail = &v[start..];
            if tail.iter().any(|x| x.is_nan()) {
                return None;
            }
            if tail.len() < init_window + n_streamed {
                return None;
            }
            Some(tail.to_vec())
        })
        .collect();
    kept.truncate(sample);
    let n_series = kept.len();
    eprintln!("Running on {n_series} series");

    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

    // Path A: re-fit on the growing window every step.
    let t_a = Instant::now();
    for values in kept.iter() {
        for extra in 0..n_streamed {
            let end = init_window + extra;
            let train_v = values[..end].to_vec();
            let stamps: Vec<_> = (0..end).map(|i| base + Duration::days(i as i64)).collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let mut m = LaplaceForecaster::new().skaters();
            let _ = m.fit(&train_ts);
        }
    }
    let a_wall = t_a.elapsed().as_secs_f64();

    // Path B: fit once, then stream. Same total observations processed.
    let t_b = Instant::now();
    for values in kept.iter() {
        let train_v = values[..init_window].to_vec();
        let stamps: Vec<_> = (0..init_window)
            .map(|i| base + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut m = LaplaceForecaster::new().skaters();
        if m.fit(&train_ts).is_err() {
            continue;
        }
        for &y in &values[init_window..init_window + n_streamed] {
            let _ = m.observe(y);
        }
    }
    let b_wall = t_b.elapsed().as_secs_f64();

    let total_updates = n_series * n_streamed;
    println!("\n=== M5 streaming vs re-fit ===");
    println!(
        "Path A (re-fit each step): {:>7.2} s   ({:>7.2} ms per update)",
        a_wall,
        1000.0 * a_wall / total_updates as f64
    );
    println!(
        "Path B (fit once, stream): {:>7.2} s   ({:>7.2} ms per update)",
        b_wall,
        1000.0 * b_wall / total_updates as f64
    );
    println!(
        "\nStreaming speedup: {:.1}×",
        if b_wall > 0.0 { a_wall / b_wall } else { 0.0 }
    );
}
