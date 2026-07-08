//! Continuous-data companion to `m5_bakeoff_export.rs`.
//!
//! Same rolling-one-step LL/CRPS protocol as the M5 bakeoff, but on
//! `fred_md` (macroeconomic monthly series) — continuous, smooth,
//! no discrete-repeat structure. Answers the question: does the WQL
//! blowup on the fev-27 short-history yearly panels come from our
//! port, or from sticky-lattice's design mismatch with continuous data?
//!
//! Run: `cargo run --release --features distributional --example fred_bakeoff_export`
//! Env: `BUILDER={skaters, skaters_no_sticky, auto}`, `SAMPLE_SIZE=50`,
//!      `PRED_STRIDE=5`, `BURN_IN=100`, `MAX_T=500`.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn parse_tsf(path: &str) -> Vec<(String, Vec<f64>)> {
    let bytes = fs::read(path).unwrap_or_default();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut series = Vec::new();
    let mut in_data = false;
    for (idx, line) in content.lines().enumerate() {
        if !in_data {
            if line.trim_start().starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let toks: Vec<&str> = line.split(':').collect();
        if toks.len() < 2 {
            continue;
        }
        let name = toks[0].to_string();
        let vals: Vec<f64> = toks[toks.len() - 1]
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        if !vals.is_empty() {
            series.push((name, vals));
        }
        let _ = idx;
    }
    series
}

fn main() {
    let sample_size: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let burn_in: usize = std::env::var("BURN_IN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let stride: usize = std::env::var("PRED_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_t: usize = std::env::var("MAX_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let path = std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/fred_md.tsf".into());
    let out_path =
        std::env::var("OUT_PATH").unwrap_or_else(|_| "/tmp/bakeoff_fred_rs.jsonl".into());
    let builder = std::env::var("BUILDER").unwrap_or_else(|_| "skaters_no_sticky".into());

    eprintln!(
        "FRED-md bakeoff — builder={builder} sample={sample_size} burn_in={burn_in} stride={stride} max_t={max_t}"
    );
    eprintln!("Loading {path}...");
    let mut series = parse_tsf(&path);
    series.retain(|(_, v)| v.len() >= burn_in + 20);
    series.truncate(sample_size);
    eprintln!("Running on {} series", series.len());

    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let mut out = fs::File::create(&out_path).expect("create out file");
    let mut n_preds = 0usize;
    let mut total_fit_us = 0u128;
    let global_start = Instant::now();

    for (idx, (series_id, values)) in series.iter().enumerate() {
        if idx % 10 == 0 {
            eprintln!(
                "[{}/{}] {series_id} — elapsed {:.1}s",
                idx,
                series.len(),
                global_start.elapsed().as_secs_f64()
            );
        }
        // Forecast the one-step CHANGE (matches the M5 bakeoff protocol).
        let changes: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let end = changes.len().min(max_t);
        let mut t = burn_in;
        while t < end {
            let train_v = changes[..t].to_vec();
            let actual = changes[t];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::days(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v) {
                Ok(t) => t,
                Err(_) => {
                    t += stride;
                    continue;
                }
            };
            let t0 = Instant::now();
            let mut m = match builder.as_str() {
                "skaters" => LaplaceForecaster::new().skaters(),
                "skaters_no_sticky" => LaplaceForecaster::new().skaters().no_sticky(),
                _ => LaplaceForecaster::new().auto(),
            };
            if m.fit(&train_ts).is_ok() {
                if let Ok(mixtures) = m.forecast_dist(1) {
                    if let Some(mix) = mixtures.first() {
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
                            "{{\"method\":\"laplace_rs\",\"series_id\":\"{series_id}\",\"t\":{t},\"actual\":{},\"components\":[{}]}}\n",
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
         wall: {wall:.1}s  fit-only: {:.1}s  per-pred: {:.2}ms",
        total_fit_us as f64 / 1e6,
        (total_fit_us as f64 / 1000.0) / n_preds.max(1) as f64
    );
}

fn json_num(x: f64) -> String {
    if !x.is_finite() {
        "null".into()
    } else {
        format!("{x:.6e}")
    }
}
