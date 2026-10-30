//! Repeatable parity/timing probe for GH #234, step 1 (basin-backed
//! optimizer evaluation).
//!
//! This binary does not depend on which `utils::optimization` backend is
//! active: it is built from the public model APIs plus
//! `anofox_forecast::utils::optimization::optimizer_backend()`, which labels
//! each JSON run with the backend it actually exercised. Build and run it
//! once against the legacy routing and once against the basin routing (see
//! `docs/audits/basin-optimizer-parity.md`), then diff the two JSON files
//! with the `compare` subcommand below.
//!
//! ```text
//! cargo run --release --features basin --example optimizer_parity_probe -- run legacy.json
//! RUSTFLAGS="--cfg anofox_basin_optimizer" CARGO_TARGET_DIR=target/basin-route \
//!     cargo run --release --features basin --example optimizer_parity_probe -- run basin.json
//! cargo run --release --features basin --example optimizer_parity_probe -- compare legacy.json basin.json
//! ```

use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig, ARIMA};
use anofox_forecast::models::exponential::{AutoETS, HoltLinearTrend, HoltWinters, SeasonalType};
use anofox_forecast::models::garch::GARCH;
use anofox_forecast::models::intermittent::Croston;
use anofox_forecast::models::tbats::TBATS;
use anofox_forecast::models::theta::{DynamicOptimizedTheta, OptimizedTheta, Theta};
use anofox_forecast::models::Forecaster;
use anofox_forecast::prelude::TimeSeries;
use anofox_forecast::utils::optimization::optimizer_backend;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

/// Minimal deterministic linear-congruential generator. Not cryptographic
/// and not statistically rigorous; exists only so the probe's input series
/// are reproducible across runs and machines without pulling in the `rand`
/// crate as a dev-dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform(0, 1).
    fn next_f64(&mut self) -> f64 {
        // Knuth's MMIX LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller, built from two uniform draws.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn make_timestamps(n: usize) -> Vec<DateTime<Utc>> {
    let base: DateTime<Utc> = "2020-01-01T00:00:00Z".parse().unwrap();
    (0..n).map(|i| base + Duration::days(i as i64)).collect()
}

fn make_series(timestamps: &[DateTime<Utc>], values: Vec<f64>) -> TimeSeries {
    TimeSeries::univariate(timestamps.to_vec(), values)
        .expect("deterministic probe series must be valid")
}

/// 120-point monthly-shaped seasonal series with trend and noise (period 12).
fn seasonal_trend_noise_series() -> TimeSeries {
    let n = 120;
    let mut rng = Lcg::new(1001);
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 0.3 * i as f64;
            let season = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = (rng.next_f64() - 0.5) * 2.0;
            100.0 + trend + season + noise
        })
        .collect();
    make_series(&make_timestamps(n), values)
}

/// 80-point trend-only series (small noise to avoid degenerate zero-residual fits).
fn trend_only_series() -> TimeSeries {
    let n = 80;
    let mut rng = Lcg::new(2002);
    let values: Vec<f64> = (0..n)
        .map(|i| 50.0 + 0.8 * i as f64 + (rng.next_f64() - 0.5) * 0.4)
        .collect();
    make_series(&make_timestamps(n), values)
}

/// 60-point level-only series.
fn level_only_series() -> TimeSeries {
    let n = 60;
    let mut rng = Lcg::new(3003);
    let values: Vec<f64> = (0..n)
        .map(|_| 42.0 + (rng.next_f64() - 0.5) * 0.5)
        .collect();
    make_series(&make_timestamps(n), values)
}

/// 80-point intermittent demand series, about 60% zeros.
fn intermittent_series() -> TimeSeries {
    let n = 80;
    let mut rng = Lcg::new(4004);
    let values: Vec<f64> = (0..n)
        .map(|_| {
            let u = rng.next_f64();
            if u < 0.6 {
                0.0
            } else {
                1.0 + rng.next_f64() * 5.0
            }
        })
        .collect();
    make_series(&make_timestamps(n), values)
}

/// 500-point return series with GARCH(1,1)-style volatility clustering.
fn returns_series() -> TimeSeries {
    let n = 500;
    let mut rng = Lcg::new(5005);
    let mut sigma2: f64 = 1.0;
    let values: Vec<f64> = (0..n)
        .map(|_| {
            let z = rng.next_gaussian();
            let eps = sigma2.sqrt() * z;
            sigma2 = 0.05 + 0.1 * eps * eps + 0.85 * sigma2;
            eps
        })
        .collect();
    make_series(&make_timestamps(n), values)
}

#[derive(Debug, Serialize, Deserialize)]
struct CaseResult {
    name: String,
    ok: bool,
    error: Option<String>,
    forecast: Vec<f64>,
    median_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProbeOutput {
    backend: String,
    cases: Vec<CaseResult>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Run one warm-up fit+predict, then 5 timed fit+predicts on fresh model
/// instances, and report the median time and the last obtained forecast (or
/// error).
fn time_case<M, F>(name: &str, make_model: F, series: &TimeSeries, h: usize) -> CaseResult
where
    F: Fn() -> M,
    M: Forecaster,
{
    {
        let mut warm = make_model();
        let _ = warm.fit_predict(series, h);
    }

    let mut durations_ms = Vec::with_capacity(5);
    let mut forecast: Vec<f64> = Vec::new();
    let mut error: Option<String> = None;

    for _ in 0..5 {
        let mut model = make_model();
        let start = Instant::now();
        let result = model.fit_predict(series, h);
        durations_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        match result {
            Ok(f) => {
                forecast = f.point().first().cloned().unwrap_or_default();
                error = None;
            }
            Err(e) => {
                error = Some(e.to_string());
            }
        }
    }

    CaseResult {
        name: name.to_string(),
        ok: error.is_none(),
        error,
        forecast,
        median_ms: median(durations_ms),
    }
}

fn run_all_cases() -> ProbeOutput {
    let h = 12;
    let seasonal = seasonal_trend_noise_series();
    let trend = trend_only_series();
    let level = level_only_series();
    let intermittent = intermittent_series();
    let returns = returns_series();

    let cases = vec![
        time_case(
            "ses_auto",
            anofox_forecast::models::exponential::SimpleExponentialSmoothing::auto,
            &level,
            h,
        ),
        time_case("holt_linear_trend_auto", HoltLinearTrend::auto, &trend, h),
        time_case(
            "holt_winters_additive_p12",
            || HoltWinters::auto(12, SeasonalType::Additive),
            &seasonal,
            h,
        ),
        time_case("auto_ets_p12", || AutoETS::with_period(12), &seasonal, h),
        time_case("arima_1_1_1", || ARIMA::new(1, 1, 1), &seasonal, h),
        time_case(
            "auto_arima_seasonal_12",
            || AutoARIMA::with_config(AutoARIMAConfig::default().with_seasonal_period(12)),
            &seasonal,
            h,
        ),
        time_case("theta", Theta::new, &seasonal, h),
        time_case("optimized_theta", OptimizedTheta::new, &seasonal, h),
        time_case(
            "dynamic_optimized_theta",
            DynamicOptimizedTheta::dotm,
            &seasonal,
            h,
        ),
        time_case(
            "croston_optimized",
            || Croston::new().optimized(),
            &intermittent,
            h,
        ),
        time_case("garch_1_1", || GARCH::new(1, 1), &returns, h),
        time_case("tbats_p12", || TBATS::new(vec![12]), &seasonal, h),
    ];

    ProbeOutput {
        backend: optimizer_backend().to_string(),
        cases,
    }
}

fn cmd_run(out_path: &str) -> ExitCode {
    let output = run_all_cases();
    let json = serde_json::to_string_pretty(&output).expect("serializable probe output");
    if let Err(e) = fs::write(out_path, json) {
        eprintln!("failed to write {out_path}: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "backend={} cases={} -> {out_path}",
        output.backend,
        output.cases.len()
    );
    ExitCode::SUCCESS
}

fn cmd_compare(a_path: &str, b_path: &str) -> ExitCode {
    let a: ProbeOutput = match fs::read_to_string(a_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    {
        Some(v) => v,
        None => {
            eprintln!("failed to read/parse {a_path}");
            return ExitCode::FAILURE;
        }
    };
    let b: ProbeOutput = match fs::read_to_string(b_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    {
        Some(v) => v,
        None => {
            eprintln!("failed to read/parse {b_path}");
            return ExitCode::FAILURE;
        }
    };

    if a.backend == b.backend {
        eprintln!(
            "both files are labeled backend=\"{}\"; expected one legacy run and one basin-routed run",
            a.backend
        );
        return ExitCode::FAILURE;
    }

    println!(
        "| case | max abs delta | max rel delta | {} ms | {} ms | ratio ({}/{}) |",
        a.backend, b.backend, b.backend, a.backend
    );
    println!("|---|---|---|---|---|---|");

    for a_case in &a.cases {
        let Some(b_case) = b.cases.iter().find(|c| c.name == a_case.name) else {
            println!("| {} | missing in {} | | | | |", a_case.name, b_path);
            continue;
        };

        if !a_case.ok || !b_case.ok {
            println!(
                "| {} | n/a (ok={}/{}) | n/a | {:.3} | {:.3} | {} |",
                a_case.name,
                a_case.ok,
                b_case.ok,
                a_case.median_ms,
                b_case.median_ms,
                a_case
                    .error
                    .as_deref()
                    .or(b_case.error.as_deref())
                    .unwrap_or("error, no message")
            );
            continue;
        }

        let n = a_case.forecast.len().min(b_case.forecast.len());
        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        for i in 0..n {
            let diff = (a_case.forecast[i] - b_case.forecast[i]).abs();
            let rel = diff / a_case.forecast[i].abs().max(1e-12);
            max_abs = max_abs.max(diff);
            max_rel = max_rel.max(rel);
        }

        let ratio = if a_case.median_ms > 0.0 {
            b_case.median_ms / a_case.median_ms
        } else {
            f64::NAN
        };

        println!(
            "| {} | {:.6e} | {:.6e} | {:.3} | {:.3} | {:.3} |",
            a_case.name, max_abs, max_rel, a_case.median_ms, b_case.median_ms, ratio
        );
    }

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => match args.get(2) {
            Some(out_path) => cmd_run(out_path),
            None => {
                eprintln!("usage: optimizer_parity_probe run <out.json>");
                ExitCode::FAILURE
            }
        },
        Some("compare") => match (args.get(2), args.get(3)) {
            (Some(a), Some(b)) => cmd_compare(a, b),
            _ => {
                eprintln!("usage: optimizer_parity_probe compare <a.json> <b.json>");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: optimizer_parity_probe run <out.json> | compare <a.json> <b.json>");
            ExitCode::FAILURE
        }
    }
}
