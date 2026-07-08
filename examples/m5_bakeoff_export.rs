//! Bakeoff exporter: rolling one-step density forecasts on FIRST-DIFFERENCED
//! M5 series. Emits JSONL with mixture parameters that a Python harness can
//! score alongside `skaters.laplace` using identical LL + CRPS formulas.
//!
//! Protocol (matches the skaters bakeoff pattern):
//! - target = one-step change `Δy_t = y_t - y_{t-1}`
//! - burn-in = 300 obs (both forecasters ignore predictions before this)
//! - rolling one-step: at each `t ≥ burn_in`, fit on `changes[0..t]`, predict
//!   density for `changes[t]`, save `(t, actual, components)`.
//!
//! Sampling: to keep runtime feasible, this exporter uses `PRED_STRIDE` to
//! predict every N-th step (default 5). The Python harness applies the same
//! stride so both forecasters score the same held-out points.
//!
//! Run: `cargo run --release --features distributional --example m5_bakeoff_export`
//! Env: `SAMPLE_SIZE=100`, `PRED_STRIDE=5`, `BURN_IN=300`, `MAX_T=1200`.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn main() {
    let sample_size: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let burn_in: usize = std::env::var("BURN_IN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let stride: usize = std::env::var("PRED_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_t: usize = std::env::var("MAX_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1200);
    let path = std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/m5_full.csv".into());
    let out_path = std::env::var("OUT_PATH").unwrap_or_else(|_| "/tmp/bakeoff_rs.jsonl".into());

    eprintln!(
        "M5 bakeoff exporter — sample={sample_size} burn_in={burn_in} stride={stride} max_t={max_t}"
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
            // Preserve NaN so we can trim leading NaN below (many M5
            // SKUs weren't yet introduced at the start of the panel).
            let v: f64 = tok.trim().parse::<f64>().unwrap_or(f64::NAN);
            cols[i].push(v);
        }
    }
    let mut kept: Vec<(String, Vec<f64>)> = names
        .iter()
        .zip(cols)
        .filter_map(|(n, v)| {
            // Skip leading NaN, then require no interior NaN in the
            // remaining tail. Also require enough real observations.
            let start = v.iter().position(|x| !x.is_nan())?;
            let tail = &v[start..];
            if tail.iter().any(|x| x.is_nan()) {
                return None;
            }
            if tail.len() < burn_in + 20 {
                return None;
            }
            Some((n.to_string(), tail.to_vec()))
        })
        .collect();
    kept.truncate(sample_size);
    eprintln!("Running on {} series", kept.len());

    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let mut out = fs::File::create(&out_path).expect("create out file");
    let mut n_preds = 0usize;
    let mut total_fit_us = 0u128;
    let global_start = Instant::now();

    for (idx, (series_id, values)) in kept.iter().enumerate() {
        if idx % 10 == 0 {
            eprintln!(
                "[{}/{}] {} — elapsed {:.1}s",
                idx,
                kept.len(),
                series_id,
                global_start.elapsed().as_secs_f64()
            );
        }
        // First-difference the series into changes.
        let changes: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let end = changes.len().min(max_t);
        // Rolling: at each t ≥ burn_in and (t - burn_in) % stride == 0,
        // fit on changes[0..t] and score density at changes[t].
        let mut t = burn_in;
        while t < end {
            let train_v = changes[..t].to_vec();
            let actual = changes[t];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::hours(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v) {
                Ok(t) => t,
                Err(_) => {
                    t += stride;
                    continue;
                }
            };
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new().auto();
            if m.fit(&train_ts).is_ok() {
                if let Ok(mixtures) = m.forecast_dist(1) {
                    if let Some(mix) = mixtures.first() {
                        // Emit components as JSONL row.
                        let comps: Vec<String> = mix
                            .components
                            .iter()
                            .filter(|(w, _)| w.is_finite() && *w > 0.0)
                            .map(|(w, g)| {
                                format!(
                                    "{{\"w\":{},\"mu\":{},\"sigma\":{}}}",
                                    json_num(*w),
                                    json_num(g.mean),
                                    json_num(g.std.max(1e-9))
                                )
                            })
                            .collect();
                        let line = format!(
                            "{{\"method\":\"laplace_rs\",\"series_id\":\"{}\",\"t\":{},\"actual\":{},\"components\":[{}]}}\n",
                            series_id,
                            t,
                            json_num(actual),
                            comps.join(",")
                        );
                        out.write_all(line.as_bytes()).expect("write");
                        n_preds += 1;
                    }
                }
            }
            total_fit_us += t0.elapsed().as_micros();
            t += stride;
        }
    }
    let wall = global_start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {n_preds} predictions written to {out_path}\n\
         wall: {:.1}s  fit-only: {:.1}s  per-pred: {:.2}ms",
        wall,
        total_fit_us as f64 / 1e6,
        (total_fit_us as f64 / 1000.0) / n_preds.max(1) as f64
    );
}

fn json_num(x: f64) -> String {
    if !x.is_finite() {
        "null".into()
    } else {
        format!("{:.6e}", x)
    }
}
