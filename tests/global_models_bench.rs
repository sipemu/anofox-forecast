//! Benchmark GlobalAutoETS, GlobalCroston, and STL batch.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, GlobalAutoETS, ModelPool};
use anofox_forecast::models::intermittent::{Croston, GlobalCroston};
use anofox_forecast::models::Forecaster;
use anofox_forecast::seasonality::STL;
use chrono::{Duration, TimeZone, Utc};
use std::time::Instant;

fn ts(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn seasonal_series(n_series: usize, n: usize, period: usize) -> Vec<Vec<f64>> {
    (0..n_series)
        .map(|s| {
            (0..n)
                .map(|i| {
                    50.0 + (s as f64) * 5.0
                        + 0.1 * i as f64
                        + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                        + ((i * 7 + s * 13 + 3) % 11) as f64 * 0.5
                        - 2.75
                })
                .collect()
        })
        .collect()
}

fn intermittent_series(n_series: usize, n: usize) -> Vec<Vec<f64>> {
    (0..n_series)
        .map(|s| {
            (0..n)
                .map(|i| {
                    // ~30% nonzero
                    if ((i * 7 + s * 13 + 5) % 10) < 3 {
                        ((i * 3 + s * 7 + 1) % 8 + 1) as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn bench_global_auto_ets() {
    let period = 12;
    let horizon = 12;
    let n = 200;

    for &n_series in &[100, 500, 1000] {
        let values = seasonal_series(n_series, n, period);
        let timestamps = ts(n);
        let pool = ModelPool::Reduced;

        // Individual AutoETS
        let start = Instant::now();
        for v in &values {
            let t = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut m =
                AutoETS::with_config(AutoETSConfig::with_period(period).with_model_pool(pool));
            m.fit(&t).unwrap();
            m.predict(horizon).unwrap();
        }
        let ind_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GlobalAutoETS
        let start = Instant::now();
        let mut global = GlobalAutoETS::new(period, pool);
        global.fit(&values).unwrap();
        let gfc = global.predict(horizon);
        let global_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "\n=== GlobalAutoETS Reduced: {} series, n={}, p={} ===",
            n_series, n, period
        );
        println!(
            "Individual:    {:.0} ms ({:.2} ms/series)",
            ind_ms,
            ind_ms / n_series as f64
        );
        println!(
            "GlobalAutoETS: {:.0} ms ({:.3} ms/series)",
            global_ms,
            global_ms / n_series as f64
        );
        println!("Speedup:       {:.0}x", ind_ms / global_ms);
        assert_eq!(gfc.len(), n_series);
    }
}

#[test]
fn bench_global_croston() {
    let n = 200;
    let horizon = 12;

    for &n_series in &[100, 500, 2000] {
        let values = intermittent_series(n_series, n);
        let timestamps = ts(n);

        // Individual Croston
        let start = Instant::now();
        for v in &values {
            let t = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut m = Croston::new().sba_optimized();
            let _ = m.fit(&t);
            let _ = m.predict(horizon);
        }
        let ind_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GlobalCroston
        let start = Instant::now();
        let mut global = GlobalCroston::sba();
        global.fit(&values).unwrap();
        let gfc = global.predict(horizon);
        let global_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("\n=== GlobalCroston SBA: {} series, n={} ===", n_series, n);
        println!(
            "Individual:     {:.1} ms ({:.3} ms/series)",
            ind_ms,
            ind_ms / n_series as f64
        );
        println!(
            "GlobalCroston:  {:.1} ms ({:.3} ms/series)",
            global_ms,
            global_ms / n_series as f64
        );
        println!("Speedup:        {:.1}x", ind_ms / global_ms);
        println!("Global α:       {:.4}", global.alpha());
        assert_eq!(gfc.len(), n_series);
    }
}

#[test]
fn bench_stl_batch() {
    let period = 12;
    let n = 500;

    for &n_series in &[100, 500, 2000] {
        let values = seasonal_series(n_series, n, period);
        let stl = STL::new(period);

        // Individual decompose
        let start = Instant::now();
        for v in &values {
            stl.decompose(v);
        }
        let ind_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Batch decompose
        let refs: Vec<&[f64]> = values.iter().map(|v| v.as_slice()).collect();
        let start = Instant::now();
        let results = stl.decompose_batch(&refs);
        let batch_ms = start.elapsed().as_secs_f64() * 1000.0;

        let n_ok = results.iter().filter(|r| r.is_some()).count();

        println!(
            "\n=== STL batch: {} series, n={}, p={} ===",
            n_series, n, period
        );
        println!(
            "Individual: {:.1} ms ({:.3} ms/series)",
            ind_ms,
            ind_ms / n_series as f64
        );
        println!(
            "Batch:      {:.1} ms ({:.3} ms/series)",
            batch_ms,
            batch_ms / n_series as f64
        );
        println!("Speedup:    {:.2}x", ind_ms / batch_ms);
        assert_eq!(n_ok, n_series);
    }
}
