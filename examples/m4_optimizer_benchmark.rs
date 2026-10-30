//! M4 accuracy + time benchmark for the optimizer evaluation of GH #234
//! (steps 2 and 3).
//!
//! Fits the optimizer-driven models on a deterministic sample of the M4
//! competition series (train/test CSVs as distributed by the M4 organizers,
//! read from `$M4_DIR`, default `~/.datasetsforecast/m4/datasets`) and
//! records, per series and model: holdout sMAPE and MASE (M4 definitions),
//! in-sample MSE of the one-step residuals (a backend-independent proxy for
//! the fit objective) and wall time of `fit` + `predict`.
//!
//! The binary is backend-agnostic; build it once per routing and compare:
//!
//! ```text
//! # legacy Nelder-Mead / L-BFGS (default)
//! cargo run --release --features basin --example m4_optimizer_benchmark -- run legacy.json
//! # basin Nelder-Mead / L-BFGS-B (step 1 routing)
//! RUSTFLAGS="--cfg anofox_basin_optimizer" CARGO_TARGET_DIR=target/basin-route \
//!     cargo run --release --features basin --example m4_optimizer_benchmark -- run basin.json
//! # basin BOBYQA for the bounded smoothing parameters (step 2 routing)
//! RUSTFLAGS="--cfg anofox_basin_bobyqa" CARGO_TARGET_DIR=target/bobyqa \
//!     cargo run --release --features basin --example m4_optimizer_benchmark -- run bobyqa.json
//! cargo run --release --features basin --example m4_optimizer_benchmark -- \
//!     compare legacy.json basin.json
//! ```
//!
//! Sample size: `PER_FREQ` series per frequency (default 300, every k-th
//! series of the file); AutoARIMA runs on the first `ARIMA_PER_FREQ`
//! (default 50) of those, Yearly / Quarterly / Monthly only.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::exponential::{
    AutoETS, HoltLinearTrend, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
};
use anofox_forecast::models::theta::{DynamicOptimizedTheta, OptimizedTheta};
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::optimization::optimizer_backend;
use chrono::{Duration, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

/// (file prefix, horizon, seasonal period) per M4 frequency.
const FREQUENCIES: [(&str, usize, usize); 6] = [
    ("Yearly", 6, 1),
    ("Quarterly", 8, 4),
    ("Monthly", 18, 12),
    ("Weekly", 13, 1),
    ("Daily", 14, 1),
    ("Hourly", 48, 24),
];

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    freq: String,
    id: String,
    model: String,
    ok: bool,
    smape: f64,
    mase: f64,
    mse_in: f64,
    fit_us: f64,
}

#[derive(Serialize, Deserialize)]
struct Run {
    backend: String,
    records: Vec<Record>,
}

fn m4_dir() -> String {
    env::var("M4_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.datasetsforecast/m4/datasets",
            env::var("HOME").unwrap_or_default()
        )
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parse an M4 wide CSV (`"id","v1","v2",...` with ragged rows).
fn read_m4(path: &str) -> Vec<(String, Vec<f64>)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    text.lines()
        .skip(1)
        .map(|line| {
            let mut fields = line.split(',').map(|f| f.trim_matches('"'));
            let id = fields.next().unwrap_or_default().to_string();
            let values = fields
                .filter(|f| !f.is_empty())
                .map(|f| f.parse::<f64>().unwrap())
                .collect();
            (id, values)
        })
        .collect()
}

fn smape(actual: &[f64], fc: &[f64]) -> f64 {
    let s: f64 = actual
        .iter()
        .zip(fc)
        .map(|(a, f)| {
            let d = a.abs() + f.abs();
            if d == 0.0 {
                0.0
            } else {
                2.0 * (a - f).abs() / d
            }
        })
        .sum();
    100.0 * s / actual.len() as f64
}

fn mase(train: &[f64], actual: &[f64], fc: &[f64], m: usize) -> f64 {
    let m = if train.len() > m { m } else { 1 };
    let scale: f64 = train
        .windows(m + 1)
        .map(|w| (w[m] - w[0]).abs())
        .sum::<f64>()
        / (train.len() - m) as f64;
    let mae: f64 = actual
        .iter()
        .zip(fc)
        .map(|(a, f)| (a - f).abs())
        .sum::<f64>()
        / actual.len() as f64;
    if scale > 0.0 {
        mae / scale
    } else {
        f64::NAN
    }
}

fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps = (0..values.len())
        .map(|i| base + Duration::days(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn eval<M: Forecaster>(
    out: &mut Vec<Record>,
    freq: &str,
    id: &str,
    model_name: &str,
    mut model: M,
    train: &[f64],
    test: &[f64],
    m: usize,
) {
    let ts = make_ts(train);
    let start = Instant::now();
    let result = model
        .fit(&ts)
        .and_then(|_| model.predict(test.len()))
        .map(|fc| fc.primary().to_vec());
    let fit_us = start.elapsed().as_secs_f64() * 1e6;
    let rec = match result {
        Ok(fc) if fc.iter().all(|v| v.is_finite()) => {
            let mse_in = model
                .residuals()
                .map(|r| {
                    let r: Vec<f64> = r.iter().copied().filter(|v| v.is_finite()).collect();
                    r.iter().map(|e| e * e).sum::<f64>() / r.len().max(1) as f64
                })
                .unwrap_or(f64::NAN);
            Record {
                freq: freq.into(),
                id: id.into(),
                model: model_name.into(),
                ok: true,
                smape: smape(test, &fc),
                mase: mase(train, test, &fc, m),
                mse_in,
                fit_us,
            }
        }
        _ => Record {
            freq: freq.into(),
            id: id.into(),
            model: model_name.into(),
            ok: false,
            smape: f64::NAN,
            mase: f64::NAN,
            mse_in: f64::NAN,
            fit_us,
        },
    };
    out.push(rec);
}

fn run(path: &str) {
    let per_freq = env_usize("PER_FREQ", 300);
    let arima_per_freq = env_usize("ARIMA_PER_FREQ", 50);
    let dir = m4_dir();
    let mut records = Vec::new();
    for (freq, h, m) in FREQUENCIES {
        let train = read_m4(&format!("{dir}/{freq}-train.csv"));
        let test = read_m4(&format!("{dir}/{freq}-test.csv"));
        let step = (train.len() / per_freq).max(1);
        let picked: Vec<usize> = (0..train.len()).step_by(step).take(per_freq).collect();
        let t0 = Instant::now();
        for (rank, &i) in picked.iter().enumerate() {
            let (id, y) = (&train[i].0, &train[i].1);
            let y_test = &test[i].1[..h];
            assert_eq!(&test[i].0, id, "train/test row mismatch");
            eval(
                &mut records,
                freq,
                id,
                "SES",
                SimpleExponentialSmoothing::auto(),
                y,
                y_test,
                m,
            );
            eval(
                &mut records,
                freq,
                id,
                "Holt",
                HoltLinearTrend::auto(),
                y,
                y_test,
                m,
            );
            if m > 1 && y.len() >= 2 * m {
                eval(
                    &mut records,
                    freq,
                    id,
                    "HoltWinters",
                    HoltWinters::auto(m, SeasonalType::Additive),
                    y,
                    y_test,
                    m,
                );
            }
            let auto_ets = if m > 1 {
                AutoETS::with_period(m)
            } else {
                AutoETS::non_seasonal()
            };
            eval(&mut records, freq, id, "AutoETS", auto_ets, y, y_test, m);
            let otm = if m > 1 {
                OptimizedTheta::seasonal(m)
            } else {
                OptimizedTheta::new()
            };
            eval(&mut records, freq, id, "OptimizedTheta", otm, y, y_test, m);
            let dotm = if m > 1 {
                DynamicOptimizedTheta::seasonal_optimized(m)
            } else {
                DynamicOptimizedTheta::dotm()
            };
            eval(
                &mut records,
                freq,
                id,
                "DynamicOptimizedTheta",
                dotm,
                y,
                y_test,
                m,
            );
            if rank < arima_per_freq && matches!(freq, "Yearly" | "Quarterly" | "Monthly") {
                let cfg = AutoARIMAConfig::default().with_seasonal_period(m);
                eval(
                    &mut records,
                    freq,
                    id,
                    "AutoARIMA",
                    AutoARIMA::with_config(cfg),
                    y,
                    y_test,
                    m,
                );
            }
        }
        eprintln!(
            "{freq}: {} series in {:.1}s",
            picked.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    let run = Run {
        backend: optimizer_label(),
        records,
    };
    fs::write(path, serde_json::to_string(&run).unwrap()).unwrap();
    eprintln!("wrote {path} ({})", run.backend);
}

fn optimizer_label() -> String {
    if cfg!(anofox_basin_bobyqa) {
        format!("{}+bobyqa", optimizer_backend())
    } else {
        optimizer_backend().to_string()
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Paired comparison of run `b` against baseline `a`, per (model, freq) and
/// per model over all frequencies.
fn compare(a_path: &str, b_path: &str) {
    let load = |p: &str| -> Run { serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap() };
    let (a, b) = (load(a_path), load(b_path));
    let key = |r: &Record| (r.model.clone(), r.freq.clone(), r.id.clone());
    let b_map: BTreeMap<_, _> = b.records.iter().map(|r| (key(r), r)).collect();

    // group -> (pairs of (a, b))
    let mut groups: BTreeMap<(String, String), Vec<(&Record, &Record)>> = BTreeMap::new();
    for ra in &a.records {
        if let Some(rb) = b_map.get(&key(ra)) {
            groups
                .entry((ra.model.clone(), ra.freq.clone()))
                .or_default()
                .push((ra, rb));
            groups
                .entry((ra.model.clone(), "ALL".into()))
                .or_default()
                .push((ra, rb));
        }
    }

    println!(
        "baseline A = {} ({a_path}), candidate B = {} ({b_path})\n",
        a.backend, b.backend
    );
    println!(
        "| model | freq | n | fail A/B | sMAPE A | sMAPE B | MASE A | MASE B | in-sample MSE: B better / same / worse | median rel. dMSE | time A s | time B s | B/A |"
    );
    println!("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
    for ((model, freq), pairs) in &groups {
        let both: Vec<_> = pairs.iter().filter(|(x, y)| x.ok && y.ok).collect();
        let fail_a = pairs.iter().filter(|(x, _)| !x.ok).count();
        let fail_b = pairs.iter().filter(|(_, y)| !y.ok).count();
        let col = |f: fn(&Record) -> f64, pick_b: bool| -> Vec<f64> {
            both.iter()
                .map(|(x, y)| f(if pick_b { y } else { x }))
                .filter(|v| v.is_finite())
                .collect()
        };
        let (mut better, mut same, mut worse) = (0, 0, 0);
        let mut rel = Vec::new();
        for (x, y) in &both {
            let d = (y.mse_in - x.mse_in) / x.mse_in.abs().max(1e-300);
            if !d.is_finite() {
                continue;
            }
            rel.push(d);
            if d.abs() <= 1e-9 {
                same += 1;
            } else if d < 0.0 {
                better += 1;
            } else {
                worse += 1;
            }
        }
        let ta: f64 = pairs.iter().map(|(x, _)| x.fit_us).sum::<f64>() / 1e6;
        let tb: f64 = pairs.iter().map(|(_, y)| y.fit_us).sum::<f64>() / 1e6;
        println!(
            "| {model} | {freq} | {} | {fail_a}/{fail_b} | {:.3} | {:.3} | {:.3} | {:.3} | {better} / {same} / {worse} | {:+.2e} | {ta:.2} | {tb:.2} | {:.2} |",
            pairs.len(),
            mean(&col(|r| r.smape, false)),
            mean(&col(|r| r.smape, true)),
            mean(&col(|r| r.mase, false)),
            mean(&col(|r| r.mase, true)),
            median(rel),
            tb / ta.max(1e-12),
        );
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") if args.len() == 3 => run(&args[2]),
        Some("compare") if args.len() == 4 => compare(&args[2], &args[3]),
        _ => {
            eprintln!("usage: m4_optimizer_benchmark run <out.json> | compare <a.json> <b.json>");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
