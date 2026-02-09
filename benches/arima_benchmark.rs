use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig, ARIMA, SARIMA};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn generate_ar1_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 42;
    let mut series = vec![0.0; n];
    series[0] = 1.0;
    for i in 1..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
        series[i] = 0.7 * series[i - 1] + noise;
    }
    series
}

fn generate_seasonal_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 123;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
        let trend = 0.05 * i as f64;
        series.push(trend + seasonal + noise);
    }
    series
}

fn bench_arima_fit(c: &mut Criterion) {
    let values = generate_ar1_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("arima_110_fit_n200", |b| {
        b.iter(|| {
            let mut model = ARIMA::new(1, 1, 0);
            model.fit(&ts).unwrap();
        })
    });

    c.bench_function("arima_111_fit_n200", |b| {
        b.iter(|| {
            let mut model = ARIMA::new(1, 1, 1);
            model.fit(&ts).unwrap();
        })
    });

    let values_500 = generate_ar1_series(500);
    let ts_500 = TimeSeries::univariate(make_timestamps(500), values_500).unwrap();

    c.bench_function("arima_111_fit_n500", |b| {
        b.iter(|| {
            let mut model = ARIMA::new(1, 1, 1);
            model.fit(&ts_500).unwrap();
        })
    });
}

fn bench_sarima_fit(c: &mut Criterion) {
    let values = generate_seasonal_series(120);
    let ts = TimeSeries::univariate(make_timestamps(120), values).unwrap();

    c.bench_function("sarima_110_110_12_fit_n120", |b| {
        b.iter(|| {
            let mut model = SARIMA::new(1, 1, 0, 1, 1, 0, 12);
            model.fit(&ts).unwrap();
        })
    });

    c.bench_function("sarima_111_111_12_fit_n120", |b| {
        b.iter(|| {
            let mut model = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
            model.fit(&ts).unwrap();
        })
    });
}

fn bench_auto_arima(c: &mut Criterion) {
    let values = generate_ar1_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("auto_arima_stepwise_n200", |b| {
        b.iter(|| {
            let mut model = AutoARIMA::new();
            model.fit(&ts).unwrap();
        })
    });

    c.bench_function("auto_arima_true_stepwise_n200", |b| {
        b.iter(|| {
            let config = AutoARIMAConfig::default().with_true_stepwise();
            let mut model = AutoARIMA::with_config(config);
            model.fit(&ts).unwrap();
        })
    });

    let values_500 = generate_ar1_series(500);
    let ts_500 = TimeSeries::univariate(make_timestamps(500), values_500).unwrap();

    c.bench_function("auto_arima_stepwise_n500", |b| {
        b.iter(|| {
            let mut model = AutoARIMA::new();
            model.fit(&ts_500).unwrap();
        })
    });
}

criterion_group!(benches, bench_arima_fit, bench_sarima_fit, bench_auto_arima);
criterion_main!(benches);
