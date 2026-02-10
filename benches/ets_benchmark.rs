use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, HoltLinearTrend, HoltWinters, SimpleExponentialSmoothing,
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn generate_seasonal_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 123;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
        let trend = 0.05 * i as f64;
        series.push(trend + seasonal + noise + 20.0); // +20 to keep values positive for multiplicative
    }
    series
}

fn bench_ses_fit(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("ses_auto_fit_n200", |b| {
        b.iter(|| {
            let mut model = SimpleExponentialSmoothing::auto();
            model.fit(&ts).unwrap();
        })
    });

    let values_500 = generate_seasonal_series(500);
    let ts_500 = TimeSeries::univariate(make_timestamps(500), values_500).unwrap();

    c.bench_function("ses_auto_fit_n500", |b| {
        b.iter(|| {
            let mut model = SimpleExponentialSmoothing::auto();
            model.fit(&ts_500).unwrap();
        })
    });
}

fn bench_holt_fit(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("holt_auto_fit_n200", |b| {
        b.iter(|| {
            let mut model = HoltLinearTrend::auto();
            model.fit(&ts).unwrap();
        })
    });

    let values_500 = generate_seasonal_series(500);
    let ts_500 = TimeSeries::univariate(make_timestamps(500), values_500).unwrap();

    c.bench_function("holt_auto_fit_n500", |b| {
        b.iter(|| {
            let mut model = HoltLinearTrend::auto();
            model.fit(&ts_500).unwrap();
        })
    });
}

fn bench_holt_winters_fit(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("hw_additive_fit_n200_p12", |b| {
        b.iter(|| {
            let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 12);
            model.fit(&ts).unwrap();
        })
    });

    c.bench_function("hw_auto_fit_n200_p12", |b| {
        b.iter(|| {
            let mut model = HoltWinters::auto(
                12,
                anofox_forecast::models::exponential::SeasonalType::Additive,
            );
            model.fit(&ts).unwrap();
        })
    });
}

fn bench_auto_ets(c: &mut Criterion) {
    let values = generate_seasonal_series(200);
    let ts = TimeSeries::univariate(make_timestamps(200), values).unwrap();

    c.bench_function("auto_ets_n200_p12", |b| {
        b.iter(|| {
            let mut model = AutoETS::with_period(12);
            model.fit(&ts).unwrap();
        })
    });

    let values_500 = generate_seasonal_series(500);
    let ts_500 = TimeSeries::univariate(make_timestamps(500), values_500).unwrap();

    c.bench_function("auto_ets_n500_p12", |b| {
        b.iter(|| {
            let mut model = AutoETS::with_period(12);
            model.fit(&ts_500).unwrap();
        })
    });

    c.bench_function("auto_ets_additive_n200", |b| {
        b.iter(|| {
            let config = AutoETSConfig::with_period(12).additive_only();
            let mut model = AutoETS::with_config(config);
            model.fit(&ts).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_ses_fit,
    bench_holt_fit,
    bench_holt_winters_fit,
    bench_auto_ets
);
criterion_main!(benches);
