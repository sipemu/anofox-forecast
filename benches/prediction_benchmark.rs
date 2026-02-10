use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{ARIMA, SARIMA};
use anofox_forecast::models::exponential::{HoltWinters, SimpleExponentialSmoothing};
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
        series.push(trend + seasonal + noise + 20.0);
    }
    series
}

fn bench_arima_predict(c: &mut Criterion) {
    let values = generate_ar1_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    let mut model = ARIMA::new(1, 1, 0);
    model.fit(&ts).unwrap();

    c.bench_function("arima_110_predict_h10", |b| {
        b.iter(|| {
            model.predict(10).unwrap();
        })
    });

    c.bench_function("arima_110_predict_h50", |b| {
        b.iter(|| {
            model.predict(50).unwrap();
        })
    });
}

fn bench_sarima_predict(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    let mut model = SARIMA::new(1, 1, 0, 1, 1, 0, 12);
    model.fit(&ts).unwrap();

    c.bench_function("sarima_110_110_12_predict_h24", |b| {
        b.iter(|| {
            model.predict(24).unwrap();
        })
    });

    c.bench_function("sarima_110_110_12_predict_h50", |b| {
        b.iter(|| {
            model.predict(50).unwrap();
        })
    });
}

fn bench_ets_predict(c: &mut Criterion) {
    let values = generate_ar1_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&ts).unwrap();

    c.bench_function("ses_predict_h10", |b| {
        b.iter(|| {
            ses.predict(10).unwrap();
        })
    });

    c.bench_function("ses_predict_h50", |b| {
        b.iter(|| {
            ses.predict(50).unwrap();
        })
    });

    let values_seasonal = generate_seasonal_series(200);
    let ts_seasonal = TimeSeries::univariate(make_timestamps(200), values_seasonal).unwrap();

    let mut hw = HoltWinters::additive(0.3, 0.1, 0.1, 12);
    hw.fit(&ts_seasonal).unwrap();

    c.bench_function("hw_additive_predict_h24", |b| {
        b.iter(|| {
            hw.predict(24).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_arima_predict,
    bench_sarima_predict,
    bench_ets_predict
);
criterion_main!(benches);
