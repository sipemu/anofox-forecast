//! Batch API validation: correctness and performance comparison.

use anofox_forecast::batch;
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, ETSSeasonalType, ETSSpec, ErrorType, GlobalETS, ModelPool, TrendType,
    ETS,
};
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::time::Instant;

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn make_series(n_series: usize, n: usize, period: usize) -> Vec<Vec<f64>> {
    (0..n_series)
        .map(|s| {
            (0..n)
                .map(|i| {
                    let base = 50.0 + (s as f64) * 5.0;
                    let trend = 0.1 * (s as f64 + 1.0) * i as f64;
                    let seasonal =
                        8.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                    let noise = ((i * 7 + s * 13 + 3) % 11) as f64 * 0.5 - 2.75;
                    base + trend + seasonal + noise
                })
                .collect()
        })
        .collect()
}

// ====== Correctness: batch == individual ======

#[test]
fn batch_mfles_matches_individual() {
    let period = 12;
    let horizon = 12;
    let n = 120;
    let values = make_series(10, n, period);
    let timestamps = make_timestamps(n);

    // Individual fits
    let individual: Vec<Vec<f64>> = values
        .iter()
        .map(|v| {
            let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut model = MFLES::new(vec![period]);
            model.fit(&ts).unwrap();
            model.predict(horizon).unwrap().primary().to_vec()
        })
        .collect();

    // Batch fit
    let batch_results = batch::mfles(&values, period, Some(horizon));

    // Compare
    for (i, (ind, batch_res)) in individual.iter().zip(batch_results.iter()).enumerate() {
        let batch_fc = batch_res.as_ref().expect("batch should succeed");
        let batch_vals = batch_fc.primary();
        for h in 0..horizon {
            assert!(
                (ind[h] - batch_vals[h]).abs() < 1e-10,
                "Series {} h={}: individual={:.6} batch={:.6} diff={:.2e}",
                i,
                h,
                ind[h],
                batch_vals[h],
                (ind[h] - batch_vals[h]).abs()
            );
        }
    }
}

#[test]
fn batch_auto_ets_matches_individual() {
    let period = 7;
    let horizon = 7;
    let n = 56;
    let values = make_series(5, n, period);
    let timestamps = make_timestamps(n);

    // Individual fits
    let individual: Vec<(Vec<f64>, String)> = values
        .iter()
        .map(|v| {
            let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let config = AutoETSConfig::with_period(period).with_model_pool(ModelPool::Reduced);
            let mut model = AutoETS::with_config(config);
            model.fit(&ts).unwrap();
            let fc = model.predict(horizon).unwrap();
            let spec = format!("{:?}", model.selected_spec());
            (fc.primary().to_vec(), spec)
        })
        .collect();

    // Batch fit
    let batch_results = batch::auto_ets(&values, period, Some(horizon), Some(ModelPool::Reduced));

    for (i, (ind, batch_res)) in individual.iter().zip(batch_results.iter()).enumerate() {
        let (batch_fc, batch_spec) = batch_res.as_ref().expect("batch should succeed");
        let batch_vals = batch_fc.primary();
        let batch_spec_str = format!("{:?}", Some(*batch_spec));

        // Same model selected
        assert_eq!(
            ind.1, batch_spec_str,
            "Series {}: individual selected {} but batch selected {}",
            i, ind.1, batch_spec_str
        );

        // Same forecasts
        for (h, v) in batch_vals.iter().enumerate().take(horizon) {
            assert!(
                (ind.0[h] - v).abs() < 1e-10,
                "Series {} h={}: individual={:.6} batch={:.6}",
                i,
                h,
                ind.0[h],
                batch_vals[h]
            );
        }
    }
}

#[test]
fn batch_ets_matches_individual() {
    let period = 7;
    let horizon = 7;
    let n = 56;
    let spec = ETSSpec::ann();
    let values = make_series(5, n, period);
    let timestamps = make_timestamps(n);

    let individual: Vec<Vec<f64>> = values
        .iter()
        .map(|v| {
            let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut model = ETS::new(spec, period);
            model.fit(&ts).unwrap();
            model.predict(horizon).unwrap().primary().to_vec()
        })
        .collect();

    let batch_results = batch::ets(&values, spec, period, Some(horizon));

    for (i, (ind, batch_res)) in individual.iter().zip(batch_results.iter()).enumerate() {
        let batch_fc = batch_res.as_ref().expect("batch should succeed");
        let batch_vals = batch_fc.primary();
        for h in 0..horizon {
            assert!(
                (ind[h] - batch_vals[h]).abs() < 1e-10,
                "Series {} h={}: individual={:.6} batch={:.6}",
                i,
                h,
                ind[h],
                batch_vals[h]
            );
        }
    }
}

// ====== Performance benchmark ======

#[test]
fn benchmark_mfles_batch_vs_individual() {
    let period = 12;
    let horizon = 12;
    let n = 200;
    let n_series = 100;
    let values = make_series(n_series, n, period);
    let timestamps = make_timestamps(n);

    // Warmup
    {
        let ts = TimeSeries::univariate(timestamps.clone(), values[0].clone()).unwrap();
        let mut m = MFLES::new(vec![period]);
        m.fit(&ts).unwrap();
    }

    // Individual fits (sequential)
    let start = Instant::now();
    let mut individual_forecasts = Vec::with_capacity(n_series);
    for v in &values {
        let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
        let mut model = MFLES::new(vec![period]);
        model.fit(&ts).unwrap();
        individual_forecasts.push(model.predict(horizon).unwrap());
    }
    let individual_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Batch fits (sequential — no rayon, measures shared Cholesky benefit)
    let start = Instant::now();
    let batch_forecasts = batch::mfles(&values, period, Some(horizon));
    let batch_ms = start.elapsed().as_secs_f64() * 1000.0;

    let n_ok = batch_forecasts.iter().filter(|r| r.is_ok()).count();

    println!(
        "\n=== MFLES: {} series, n={}, period={} ===",
        n_series, n, period
    );
    println!(
        "Individual (sequential): {:.1} ms ({:.2} ms/series)",
        individual_ms,
        individual_ms / n_series as f64
    );
    println!(
        "Batch (sequential):      {:.1} ms ({:.2} ms/series)",
        batch_ms,
        batch_ms / n_series as f64
    );
    println!("Speedup:                 {:.2}x", individual_ms / batch_ms);
    println!("Success: {}/{}", n_ok, n_series);

    // Verify same results
    for (i, (ind, batch_res)) in individual_forecasts
        .iter()
        .zip(batch_forecasts.iter())
        .enumerate()
    {
        if let Ok(batch_fc) = batch_res {
            let max_diff: f64 = ind
                .primary()
                .iter()
                .zip(batch_fc.primary().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(max_diff < 1e-10, "Series {} max diff: {:.2e}", i, max_diff);
        }
    }
}

#[test]
fn benchmark_auto_ets_batch_vs_individual() {
    let period = 7;
    let horizon = 7;
    let n = 100;
    let n_series = 20;
    let values = make_series(n_series, n, period);
    let timestamps = make_timestamps(n);
    let pool = ModelPool::Reduced;

    // Warmup
    {
        let ts = TimeSeries::univariate(timestamps.clone(), values[0].clone()).unwrap();
        let mut m = AutoETS::with_config(AutoETSConfig::with_period(period).with_model_pool(pool));
        m.fit(&ts).unwrap();
    }

    // Individual
    let start = Instant::now();
    for v in &values {
        let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
        let mut m = AutoETS::with_config(AutoETSConfig::with_period(period).with_model_pool(pool));
        m.fit(&ts).unwrap();
        m.predict(horizon).unwrap();
    }
    let individual_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Batch
    let start = Instant::now();
    let batch_results = batch::auto_ets(&values, period, Some(horizon), Some(pool));
    let batch_ms = start.elapsed().as_secs_f64() * 1000.0;

    let n_ok = batch_results.iter().filter(|r| r.is_ok()).count();

    println!(
        "\n=== AutoETS Reduced: {} series, n={}, period={} ===",
        n_series, n, period
    );
    println!(
        "Individual (sequential): {:.1} ms ({:.2} ms/series)",
        individual_ms,
        individual_ms / n_series as f64
    );
    println!(
        "Batch (sequential):      {:.1} ms ({:.2} ms/series)",
        batch_ms,
        batch_ms / n_series as f64
    );
    println!("Speedup:                 {:.2}x", individual_ms / batch_ms);
    println!("Success: {}/{}", n_ok, n_series);
}

// ====== GlobalETS benchmarks ======

#[test]
fn benchmark_global_ets_ann_vs_individual() {
    let n = 200;
    let horizon = 12;
    let period = 1; // non-seasonal for ANN

    for &n_series in &[50, 200, 1000] {
        let values = make_series(n_series, n, 12); // data has seasonality but model is ANN
        let timestamps = make_timestamps(n);

        // Individual ETS(A,N,N) fits
        let start = Instant::now();
        let mut ind_fc: Vec<Vec<f64>> = Vec::with_capacity(n_series);
        for v in &values {
            let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut model = ETS::new(ETSSpec::ann(), period);
            model.fit(&ts).unwrap();
            ind_fc.push(model.predict(horizon).unwrap().primary().to_vec());
        }
        let individual_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GlobalETS(A,N,N) — shared α
        let start = Instant::now();
        let mut global = GlobalETS::new(ETSSpec::ann(), period);
        global.fit(&values).unwrap();
        let _global_fc = global.predict(horizon);
        let global_ms = start.elapsed().as_secs_f64() * 1000.0;

        let (alpha, _, _, _) = global.params();

        println!("\n=== GlobalETS ANN: {} series, n={} ===", n_series, n);
        println!(
            "Individual:  {:.1} ms ({:.3} ms/series)",
            individual_ms,
            individual_ms / n_series as f64
        );
        println!(
            "GlobalETS:   {:.1} ms ({:.3} ms/series)",
            global_ms,
            global_ms / n_series as f64
        );
        println!("Speedup:     {:.1}x", individual_ms / global_ms);
        println!("Global α:    {:.4}", alpha);
    }
}

#[test]
fn benchmark_global_ets_aana_vs_individual() {
    let period = 12;
    let n = 200;
    let horizon = 12;

    for &n_series in &[50, 200, 1000] {
        let values = make_series(n_series, n, period);
        let timestamps = make_timestamps(n);
        let spec = ETSSpec::new(
            ErrorType::Additive,
            TrendType::None,
            ETSSeasonalType::Additive,
        );

        // Individual ETS(A,N,A) fits
        let start = Instant::now();
        for v in &values {
            let ts = TimeSeries::univariate(timestamps.clone(), v.clone()).unwrap();
            let mut model = ETS::new(spec, period);
            model.fit(&ts).unwrap();
            model.predict(horizon).unwrap();
        }
        let individual_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GlobalETS(A,N,A) — shared α, γ
        let start = Instant::now();
        let mut global = GlobalETS::new(spec, period);
        global.fit(&values).unwrap();
        let global_fc = global.predict(horizon);
        let global_ms = start.elapsed().as_secs_f64() * 1000.0;

        let (alpha, _, gamma, _) = global.params();

        println!(
            "\n=== GlobalETS ANA: {} series, n={}, period={} ===",
            n_series, n, period
        );
        println!(
            "Individual:  {:.1} ms ({:.3} ms/series)",
            individual_ms,
            individual_ms / n_series as f64
        );
        println!(
            "GlobalETS:   {:.1} ms ({:.3} ms/series)",
            global_ms,
            global_ms / n_series as f64
        );
        println!("Speedup:     {:.1}x", individual_ms / global_ms);
        println!("Global α:    {:.4}, γ: {:.4}", alpha, gamma.unwrap_or(0.0));
        assert_eq!(global_fc.len(), n_series);
    }
}
