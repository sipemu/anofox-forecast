use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{Naive, SimpleMovingAverage};
use anofox_forecast::models::ensemble::{
    AutoEnsemble, AutoEnsembleConfig, CombinationMethod, Ensemble,
};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
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
        let cycle = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin();
        series.push(trend + cycle + noise);
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

fn bench_ensemble_mean_fit(c: &mut Criterion) {
    let values = generate_trend_series(120);
    let ts = TimeSeries::univariate(make_timestamps(120), values).unwrap();

    c.bench_function("ensemble_mean_3models_fit_n120", |b| {
        b.iter(|| {
            let models: Vec<Box<dyn Forecaster>> = vec![
                Box::new(Naive::new()),
                Box::new(SimpleMovingAverage::new(5)),
                Box::new(SimpleExponentialSmoothing::auto()),
            ];
            let mut ensemble = Ensemble::new(models);
            ensemble.fit(&ts).unwrap();
        })
    });

    c.bench_function("ensemble_mean_5models_fit_n120", |b| {
        b.iter(|| {
            let models: Vec<Box<dyn Forecaster>> = vec![
                Box::new(Naive::new()),
                Box::new(SimpleMovingAverage::new(5)),
                Box::new(SimpleExponentialSmoothing::auto()),
                Box::new(HoltLinearTrend::auto()),
                Box::new(Theta::new()),
            ];
            let mut ensemble = Ensemble::new(models);
            ensemble.fit(&ts).unwrap();
        })
    });
}

fn bench_ensemble_weighted_fit(c: &mut Criterion) {
    let values = generate_trend_series(120);
    let ts = TimeSeries::univariate(make_timestamps(120), values).unwrap();

    c.bench_function("ensemble_weighted_mse_fit_n120", |b| {
        b.iter(|| {
            let models: Vec<Box<dyn Forecaster>> = vec![
                Box::new(Naive::new()),
                Box::new(SimpleMovingAverage::new(5)),
                Box::new(SimpleExponentialSmoothing::auto()),
                Box::new(HoltLinearTrend::auto()),
            ];
            let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::WeightedMSE);
            ensemble.fit(&ts).unwrap();
        })
    });

    c.bench_function("ensemble_median_fit_n120", |b| {
        b.iter(|| {
            let models: Vec<Box<dyn Forecaster>> = vec![
                Box::new(Naive::new()),
                Box::new(SimpleMovingAverage::new(5)),
                Box::new(SimpleExponentialSmoothing::auto()),
                Box::new(HoltLinearTrend::auto()),
            ];
            let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::Median);
            ensemble.fit(&ts).unwrap();
        })
    });
}

fn bench_ensemble_predict(c: &mut Criterion) {
    let values = generate_trend_series(120);
    let ts = TimeSeries::univariate(make_timestamps(120), values).unwrap();

    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(SimpleMovingAverage::new(5)),
        Box::new(SimpleExponentialSmoothing::auto()),
        Box::new(HoltLinearTrend::auto()),
    ];
    let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::WeightedMSE);
    ensemble.fit(&ts).unwrap();

    c.bench_function("ensemble_predict_h10", |b| {
        b.iter(|| {
            ensemble.predict(10).unwrap();
        })
    });

    c.bench_function("ensemble_predict_with_intervals_h10", |b| {
        b.iter(|| {
            ensemble.predict_with_intervals(10, 0.95).unwrap();
        })
    });
}

fn bench_auto_ensemble(c: &mut Criterion) {
    let values = generate_trend_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    c.bench_function("auto_ensemble_default_n100", |b| {
        b.iter(|| {
            let mut model = AutoEnsemble::new();
            model.fit(&ts).unwrap();
        })
    });

    c.bench_function("auto_ensemble_top2_n100", |b| {
        b.iter(|| {
            let config = AutoEnsembleConfig::default().with_top_k(2);
            let mut model = AutoEnsemble::with_config(config);
            model.fit(&ts).unwrap();
        })
    });

    let values_seasonal = generate_seasonal_series(120);
    let ts_seasonal = TimeSeries::univariate(make_timestamps(120), values_seasonal).unwrap();

    c.bench_function("auto_ensemble_seasonal_n120_p12", |b| {
        b.iter(|| {
            let mut model = AutoEnsemble::seasonal(12);
            model.fit(&ts_seasonal).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_ensemble_mean_fit,
    bench_ensemble_weighted_fit,
    bench_ensemble_predict,
    bench_auto_ensemble
);
criterion_main!(benches);
