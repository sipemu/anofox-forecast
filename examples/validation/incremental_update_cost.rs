//! Issue #233 measurement: compact-state size and `update()` cost versus a
//! full refit for the exponential-smoothing family.
//!
//! For each model and history length `n`, a model is fit on `n` points, then
//! a batch of `k` new observations arrives. Two ways to absorb them:
//! - **refit**: `fit()` on all `n + k` points (re-estimates parameters);
//! - **update**: restore from the compact `ModelState` and `update()` with
//!   the `k` points (fixed parameters) -- the stored-aggregate-state path.
//!
//! Reported: median wall time of each (release build), the bincode size of
//! the compact `ModelState` and of the full serialized model.
//!
//! Run with:
//! `cargo run --release --example incremental_update_cost --features serde`

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, ETSSpec, HoltLinearTrend, HoltWinters, SeasonalType,
    SimpleExponentialSmoothing, ETS,
};
use anofox_forecast::models::IncrementalForecaster;
use anofox_forecast::utils::persistence::to_bincode;
use chrono::{Duration, TimeZone, Utc};
use serde::Serialize;
use std::time::Instant;

const PERIOD: usize = 12;
const REPS: usize = 7;

fn series(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|t| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state % 10_000) as f64 / 10_000.0 - 0.5;
            let season = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / PERIOD as f64).sin();
            100.0 + 0.3 * t as f64 + season + 4.0 * noise
        })
        .collect()
}

fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

fn median_us(mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[REPS / 2]
}

fn full_size<M: Serialize>(m: &M) -> String {
    to_bincode(m).unwrap().len().to_string()
}

fn measure<M>(name: &str, make: impl Fn() -> M, full: impl Fn(&M) -> String, n: usize, k: usize)
where
    M: IncrementalForecaster,
{
    let values = series(n + k);
    let all = make_ts(&values);
    let mut base = make();
    base.fit(&make_ts(&values[..n])).unwrap();

    let refit = median_us(|| {
        let mut m = make();
        m.fit(&all).unwrap();
    });
    // The stored-state path: restore from the compact state, then update.
    // Restoring is part of the timing; cloning the restored model is not.
    let state = base.state().unwrap();
    let update = median_us(|| {
        let mut m = M::from_state(state.clone()).unwrap();
        m.update(&values[n..]).unwrap();
        std::hint::black_box(&m);
    });
    let state = to_bincode(&base.state().unwrap()).unwrap().len();
    let full = full(&base);
    println!(
        "| {name} | {n} | {k} | {refit:.1} | {update:.2} | {:.0}x | {state} | {full} |",
        refit / update
    );
}

fn main() {
    println!("| model | n | k | refit us | update us | speed-up | state B | full model B |");
    println!("|---|--:|--:|--:|--:|--:|--:|--:|");
    for n in [120, 1_200, 12_000] {
        for k in [1, 30] {
            measure(
                "SES auto",
                SimpleExponentialSmoothing::auto,
                full_size,
                n,
                k,
            );
            measure("Holt auto", HoltLinearTrend::auto, full_size, n, k);
            measure(
                "HoltWinters auto (add, 12)",
                || HoltWinters::auto(PERIOD, SeasonalType::Additive),
                full_size,
                n,
                k,
            );
            measure(
                "ETS(A,A,A) auto (12)",
                || ETS::new(ETSSpec::aaa(), PERIOD),
                full_size,
                n,
                k,
            );
            if n <= 1_200 {
                // AutoETS is not serde-serializable as a whole.
                measure(
                    "AutoETS (12)",
                    || {
                        AutoETS::with_config(AutoETSConfig {
                            seasonal_period: Some(PERIOD),
                            ..AutoETSConfig::default()
                        })
                    },
                    |_| "n/a".into(),
                    n,
                    k,
                );
            }
        }
    }
}
