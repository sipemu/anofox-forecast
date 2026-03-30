//! Comprehensive performance benchmarks including TBATS/AutoTBATS.
//!
//! Run with: cargo bench --bench comprehensive_benchmark --all-features

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::{SimpleExponentialSmoothing, ETS};
use anofox_forecast::models::tbats::{AutoTBATS, TBATS};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn make_series(n: usize) -> TimeSeries {
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| {
            50.0 + 0.3 * i as f64
                + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                + ((i * 7 + 3) % 11) as f64 * 0.3
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn bench_models(c: &mut Criterion) {
    let ts = make_series(100);

    let mut group = c.benchmark_group("fit_predict_n100");
    group.sample_size(30);

    group.bench_function("Naive", |b| {
        b.iter(|| {
            let mut m = Naive::new();
            m.fit(black_box(&ts)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("SES", |b| {
        b.iter(|| {
            let mut m = SimpleExponentialSmoothing::auto();
            m.fit(black_box(&ts)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("Theta", |b| {
        b.iter(|| {
            let mut m = Theta::new();
            m.fit(black_box(&ts)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("ARIMA_1_1_1", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("ETS_AAN", |b| {
        b.iter(|| {
            let mut m = ETS::default();
            m.fit(black_box(&ts)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.finish();
}

fn bench_tbats(c: &mut Criterion) {
    let ts_200 = make_series(200);
    let ts_500 = make_series(500);

    let mut group = c.benchmark_group("tbats");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    group.bench_function("TBATS_p7_n200", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = TBATS::new(vec![7]);
            m.fit(black_box(&ts_200)).unwrap();
            m.predict(7).unwrap();
        })
    });

    group.bench_function("TBATS_p7_n500", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = TBATS::new(vec![7]);
            m.fit(black_box(&ts_500)).unwrap();
            m.predict(7).unwrap();
        })
    });

    group.bench_function("AutoTBATS_p7_n200", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = AutoTBATS::new(vec![7]);
            m.fit(black_box(&ts_200)).unwrap();
            m.predict(7).unwrap();
        })
    });

    group.finish();
}

fn bench_auto_arima(c: &mut Criterion) {
    use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig, ARIMA};

    let ts_100 = make_series(100);
    let ts_200 = make_series(200);
    let ts_500 = make_series(500);

    let mut group = c.benchmark_group("auto_arima");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    group.bench_function("ARIMA_1_1_1_n100", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("AutoARIMA_default_n100", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = AutoARIMA::new();
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("AutoARIMA_default_n200", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = AutoARIMA::new();
            m.fit(black_box(&ts_200)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("AutoARIMA_default_n500", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = AutoARIMA::new();
            m.fit(black_box(&ts_500)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("AutoARIMA_stepwise_n200", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let config = AutoARIMAConfig::default().with_true_stepwise();
            let mut m = AutoARIMA::with_config(config);
            m.fit(black_box(&ts_200)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("AutoARIMA_seasonal12_n200", |b: &mut criterion::Bencher| {
        b.iter(|| {
            let mut m = AutoARIMA::seasonal(12);
            m.fit(black_box(&ts_200)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches2, bench_models, bench_tbats, bench_auto_arima);
criterion_main!(benches2);
