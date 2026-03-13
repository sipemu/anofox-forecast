use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use criterion::{criterion_group, criterion_main, Criterion};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}
fn generate_standard_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 314;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
        let trend = 20.0 + 0.1 * i as f64;
        let seasonal = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
        series.push(trend + seasonal + noise);
    }
    series
}

fn bench_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_fit_predict_n200");
    let horizon = 12;
    let values = generate_standard_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut m = Naive::new();
            m.fit(&ts).unwrap();
            m.predict(horizon).unwrap();
        })
    });
    group.bench_function("ses", |b| {
        b.iter(|| {
            let mut m = SimpleExponentialSmoothing::auto();
            m.fit(&ts).unwrap();
            m.predict(horizon).unwrap();
        })
    });
    group.bench_function("theta", |b| {
        b.iter(|| {
            let mut m = Theta::new();
            m.fit(&ts).unwrap();
            m.predict(horizon).unwrap();
        })
    });
    group.bench_function("auto_ets_additive", |b| {
        b.iter(|| {
            let config = AutoETSConfig::with_period(12).additive_only();
            let mut m = AutoETS::with_config(config);
            m.fit(&ts).unwrap();
            m.predict(horizon).unwrap();
        })
    });
    group.bench_function("arima_111", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(&ts).unwrap();
            m.predict(horizon).unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, bench_model_comparison);
criterion_main!(benches);
