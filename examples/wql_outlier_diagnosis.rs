//! Diagnose the m1_yearly / cif_2016 / tourism_yearly WQL outlier
//! pathology. Loads each dataset, fits a few LaplaceForecaster variants
//! per series, and prints the WQL contributions per series so we can
//! see which series drive the aggregate blowup.
//!
//! Run: `cargo run --release --features distributional --example wql_outlier_diagnosis`

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::{DistributionalForecaster, LaplaceForecaster};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::fs;

const WQL_QS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

fn wql(matrix: &[Vec<f64>], y: &[f64]) -> f64 {
    let denom: f64 = y.iter().map(|v| v.abs()).sum::<f64>().max(1e-9);
    let mut total = 0.0;
    for (qi, &q) in WQL_QS.iter().enumerate() {
        for (h, &yy) in y.iter().enumerate() {
            let qhat = matrix[qi][h];
            let e = yy - qhat;
            let pinball = if e > 0.0 { q * e } else { (q - 1.0) * e };
            total += 2.0 * pinball;
        }
    }
    total / denom
}

fn mixture_stats(
    model: &mut LaplaceForecaster,
    _ts: &TimeSeries,
    h: usize,
    y: &[f64],
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let dists = model.forecast_dist(h).unwrap();
    let matrix: Vec<Vec<f64>> = WQL_QS
        .iter()
        .map(|&q| dists.iter().map(|g| g.quantile(q)).collect())
        .collect();
    let w = wql(&matrix, y);
    let means: Vec<f64> = dists.iter().map(|g| g.mean()).collect();
    let stds: Vec<f64> = dists.iter().map(|g| g.std()).collect();
    let q05: Vec<f64> = dists.iter().map(|g| g.quantile(0.05)).collect();
    let q95: Vec<f64> = dists.iter().map(|g| g.quantile(0.95)).collect();
    (w, means, stds, q05, q95)
}

fn load_series(path: &str, min_len: usize) -> Vec<Vec<f64>> {
    // Match examples/fev_benchmark.rs parse_tsf — bytes-as-chars to tolerate
    // non-UTF-8 TSF files.
    let bytes = fs::read(path).unwrap_or_default();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
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
        let vals_str = toks[toks.len() - 1];
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if values.len() >= min_len {
            series.push(values);
        }
    }
    series
}

fn run(dataset: &str, path: &str, horizon: usize) {
    println!("\n==== {dataset} (horizon={horizon}) ====");
    let series = load_series(path, horizon + 10);
    println!("loaded {} series", series.len());
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let mut auto_wqls = Vec::new();
    let mut skaters_wqls = Vec::new();
    let mut nosticky_wqls = Vec::new();
    let mut worst_details: Vec<(
        usize,
        f64,
        f64,
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    )> = Vec::new();
    for (idx, values) in series.iter().enumerate().take(500) {
        let split = values.len() - horizon;
        let train_v = values[..split].to_vec();
        let test_v: Vec<f64> = values[split..].to_vec();
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base + Duration::days(i as i64))
            .collect();
        let train_ts = TimeSeries::univariate(stamps, train_v).unwrap();

        let mut m_auto = LaplaceForecaster::new().auto();
        m_auto.fit(&train_ts).unwrap();
        let (w_auto, means_a, stds_a, _, _) =
            mixture_stats(&mut m_auto, &train_ts, horizon, &test_v);
        auto_wqls.push(w_auto);

        let mut m_skaters = LaplaceForecaster::new().skaters();
        m_skaters.fit(&train_ts).unwrap();
        let (w_skaters, means_s, stds_s, q05_s, q95_s) =
            mixture_stats(&mut m_skaters, &train_ts, horizon, &test_v);
        skaters_wqls.push(w_skaters);

        let mut m_nosticky = LaplaceForecaster::new().skaters().no_sticky();
        m_nosticky.fit(&train_ts).unwrap();
        let (w_nosticky, _, _, _, _) = mixture_stats(&mut m_nosticky, &train_ts, horizon, &test_v);
        nosticky_wqls.push(w_nosticky);

        worst_details.push((
            idx, w_auto, w_skaters, w_nosticky, means_a, means_s, stds_a, stds_s, q05_s, q95_s,
            test_v,
        ));
    }
    // Sort by skaters WQL descending — worst first.
    worst_details.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("\nTop 5 worst .skaters() WQL series (mixture inspection):");
    for row in &worst_details[..worst_details.len().min(5)] {
        let (idx, wa, ws, wn, means_a, means_s, stds_a, stds_s, q05_s, q95_s, test_v) = row;
        println!("\n--- series {idx}: WQL auto={wa:.4}, skaters={ws:.4}, +nosticky={wn:.4} ---");
        for h in 0..test_v.len() {
            let width_a = stds_a[h] * 2.0;
            let width_s = q95_s[h] - q05_s[h];
            println!(
                "  h={:2} y={:>12.2}   auto: μ={:>12.2} σ={:>10.3}    skaters: μ={:>12.2} σ={:>10.3} [q05={:>12.2}, q95={:>12.2}]  width_ratio={:.2}",
                h + 1, test_v[h],
                means_a[h], stds_a[h],
                means_s[h], stds_s[h],
                q05_s[h], q95_s[h],
                if width_a > 0.0 { width_s / width_a } else { 0.0 },
            );
        }
    }
    // Aggregate.
    let sum = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "\nMean per-series WQL: auto={:.4} skaters={:.4} skaters+nosticky={:.4}",
        sum(&auto_wqls),
        sum(&skaters_wqls),
        sum(&nosticky_wqls),
    );
}

fn dump_per_leaf(path: &str, series_idx: usize, horizon: usize) {
    let series = load_series(path, horizon + 10);
    let values = &series[series_idx];
    let split = values.len() - horizon;
    let train_v = values[..split].to_vec();
    let y_scale: f64 = train_v
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..train_v.len())
        .map(|i| base + Duration::days(i as i64))
        .collect();
    let train_ts = TimeSeries::univariate(stamps, train_v).unwrap();
    let mut m = LaplaceForecaster::new().skaters();
    m.fit(&train_ts).unwrap();
    println!("\n=== per-leaf predict_one() dump for {path}[{series_idx}] ===");
    println!("y_scale (max |y|) = {y_scale:.3e}");
    let leaves = m.debug_leaf_predictions();
    let mut rows: Vec<_> = leaves.into_iter().collect();
    // Sort by std descending — divergent leaves at the top.
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!(
        "{:<40} {:>10} {:>14} {:>14} {:>10}",
        "leaf", "weight", "mean", "std", "std/y_scale"
    );
    for (name, mu, s, w) in rows.iter().take(20) {
        let ratio = s / y_scale;
        let flag = if ratio > 1000.0 { " ← DIVERGED" } else { "" };
        println!("{name:<40} {w:>10.6} {mu:>14.2e} {s:>14.2e} {ratio:>10.2e}{flag}");
    }
}

fn main() {
    run("m1_yearly", "validation/data/m1_yearly.tsf", 6);
    run("tourism_yearly", "validation/data/tourism_yearly.tsf", 4);
    run("cif_2016", "validation/data/cif_2016.tsf", 12);
    // Culprit-finding: which leaf emits absurd σ?
    dump_per_leaf("validation/data/cif_2016.tsf", 54, 12);
}
