use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, SimpleMovingAverage};
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn generate_trend_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 42;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
        let trend = 20.0 + 0.3 * i as f64;
        series.push(trend + noise);
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

fn bench_cv_expanding_folds(c: &mut Criterion) {
    let values = generate_trend_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    c.bench_function("cv_expanding_naive_5folds_n100", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(50, 1).with_step_size(10);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_expanding_naive_10folds_n100", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(50, 1).with_step_size(5);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_expanding_naive_25folds_n100", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(50, 1).with_step_size(2);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });
}

fn bench_cv_rolling_folds(c: &mut Criterion) {
    let values = generate_trend_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    c.bench_function("cv_rolling_naive_5folds_n100", |b| {
        b.iter(|| {
            let config = CVConfig::rolling(50, 1).with_step_size(10);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_rolling_naive_10folds_n100", |b| {
        b.iter(|| {
            let config = CVConfig::rolling(50, 1).with_step_size(5);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });
}

fn bench_cv_model_complexity(c: &mut Criterion) {
    let values = generate_trend_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    let config = CVConfig::expanding(50, 1).with_step_size(5);

    c.bench_function("cv_expanding_naive_n100", |b| {
        b.iter(|| {
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_expanding_sma5_n100", |b| {
        b.iter(|| {
            cross_validate(&config, &ts, || SimpleMovingAverage::new(5)).unwrap();
        })
    });

    c.bench_function("cv_expanding_ses_n100", |b| {
        b.iter(|| {
            cross_validate(&config, &ts, SimpleExponentialSmoothing::auto).unwrap();
        })
    });
}

fn bench_cv_horizon(c: &mut Criterion) {
    let values = generate_seasonal_series(120);
    let ts = TimeSeries::univariate(make_timestamps(120), values).unwrap();

    c.bench_function("cv_expanding_naive_h1_n120", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(60, 1).with_step_size(5);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_expanding_naive_h5_n120", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(60, 5).with_step_size(5);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });

    c.bench_function("cv_expanding_naive_h10_n120", |b| {
        b.iter(|| {
            let config = CVConfig::expanding(60, 10).with_step_size(5);
            cross_validate(&config, &ts, Naive::new).unwrap();
        })
    });
}

fn bench_cv_series_length(c: &mut Criterion) {
    let config = CVConfig::expanding(50, 1).with_step_size(5);

    let values_100 = generate_trend_series(100);
    let ts_100 = TimeSeries::univariate(make_timestamps(100), values_100).unwrap();

    c.bench_function("cv_expanding_sma5_n100", |b| {
        b.iter(|| {
            cross_validate(&config, &ts_100, || SimpleMovingAverage::new(5)).unwrap();
        })
    });

    let values_200 = generate_trend_series(200);
    let ts_200 = TimeSeries::univariate(make_timestamps(200), values_200).unwrap();

    c.bench_function("cv_expanding_sma5_n200", |b| {
        b.iter(|| {
            cross_validate(&config, &ts_200, || SimpleMovingAverage::new(5)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_cv_expanding_folds,
    bench_cv_rolling_folds,
    bench_cv_model_complexity,
    bench_cv_horizon,
    bench_cv_series_length
);
criterion_main!(benches);
