//! Comprehensive performance benchmarks for all major features.
//!
//! Run with: cargo bench --bench comprehensive_benchmark --all-features

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::{SimpleExponentialSmoothing, ETS};
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

fn bench_model_fit_predict(c: &mut Criterion) {
    let ts_100 = make_series(100);
    let ts_500 = make_series(500);
    let ts_1000 = make_series(1000);

    let mut group = c.benchmark_group("model_fit_predict_n100");
    group.sample_size(30);

    group.bench_function("Naive", |b| {
        b.iter(|| {
            let mut m = Naive::new();
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("SES", |b| {
        b.iter(|| {
            let mut m = SimpleExponentialSmoothing::auto();
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("Theta", |b| {
        b.iter(|| {
            let mut m = Theta::new();
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("ARIMA(1,1,1)", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("ETS(AAN)", |b| {
        b.iter(|| {
            let mut m = ETS::default();
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.finish();

    // Scaling benchmarks
    let mut group = c.benchmark_group("arima_scaling");
    group.sample_size(20);

    group.bench_function("n=100", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts_100)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("n=500", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts_500)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.bench_function("n=1000", |b| {
        b.iter(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(black_box(&ts_1000)).unwrap();
            m.predict(12).unwrap();
        })
    });

    group.finish();
}

fn bench_decomposition(c: &mut Criterion) {
    use anofox_forecast::seasonality::StlBuilder;

    let ts = make_series(365);
    let values = ts.primary_values();

    let mut group = c.benchmark_group("decomposition");
    group.sample_size(30);

    group.bench_function("STL_period7_n365", |b| {
        b.iter(|| {
            StlBuilder::new(7).decompose(black_box(values)).unwrap();
        })
    });

    group.finish();
}

fn bench_features(c: &mut Criterion) {
    use anofox_forecast::features::{autocorrelation, mean, skewness, variance};

    let values: Vec<f64> = (0..1000)
        .map(|i| 50.0 + 10.0 * (i as f64 * 0.1).sin())
        .collect();

    let mut group = c.benchmark_group("features_n1000");

    group.bench_function("mean", |b| b.iter(|| mean(black_box(&values))));

    group.bench_function("variance", |b| b.iter(|| variance(black_box(&values))));

    group.bench_function("skewness", |b| b.iter(|| skewness(black_box(&values))));

    group.bench_function("autocorrelation_lag1", |b| {
        b.iter(|| autocorrelation(black_box(&values), 1))
    });

    group.finish();
}

fn bench_differencing(c: &mut Criterion) {
    use anofox_forecast::models::arima::{
        difference, find_min_fractional_d, fractional_difference,
    };

    let values: Vec<f64> = (0..1000).map(|i| 100.0 + 2.0 * i as f64).collect();

    let mut group = c.benchmark_group("differencing_n1000");

    group.bench_function("integer_d1", |b| {
        b.iter(|| difference(black_box(&values), 1))
    });

    group.bench_function("fractional_d0.5", |b| {
        b.iter(|| fractional_difference(black_box(&values), 0.5, 1e-4))
    });

    group.bench_function("find_min_d", |b| {
        b.iter(|| find_min_fractional_d(black_box(&values), 0.05, 1e-4))
    });

    group.finish();
}

fn bench_cycle_filters(c: &mut Criterion) {
    use anofox_forecast::seasonality::{bk_filter, cf_filter, hamilton_filter};

    // Quarterly GDP-like data (80 quarters = 20 years)
    let values: Vec<f64> = (0..80)
        .map(|i| {
            1000.0 + 5.0 * i as f64 // trend
                + 50.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin() // 6-year cycle
                + ((i * 7 + 3) % 11) as f64 * 2.0 // noise
        })
        .collect();

    let mut group = c.benchmark_group("cycle_filters_n80");
    group.sample_size(50);

    group.bench_function("cf_filter_6_32", |b| {
        b.iter(|| cf_filter(black_box(&values), 6, 32, true).unwrap())
    });

    group.bench_function("bk_filter_6_32_k12", |b| {
        b.iter(|| bk_filter(black_box(&values), 6, 32, 12).unwrap())
    });

    group.bench_function("hamilton_h8_p4", |b| {
        b.iter(|| hamilton_filter(black_box(&values), 8, 4).unwrap())
    });

    group.finish();
}

fn bench_postprocess(c: &mut Criterion) {
    use anofox_forecast::postprocess::{BootstrapPredictor, ConformalPredictor};

    let forecasts: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let actuals: Vec<f64> = (0..200)
        .map(|i| i as f64 + ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5)
        .collect();
    let point: Vec<f64> = (200..212).map(|i| i as f64).collect();

    let mut group = c.benchmark_group("postprocess");
    group.sample_size(30);

    group.bench_function("conformal_split_fit", |b| {
        let cp = ConformalPredictor::split(0.95);
        b.iter(|| cp.fit(black_box(&forecasts), black_box(&actuals)).unwrap())
    });

    group.bench_function("conformal_predict", |b| {
        let cp = ConformalPredictor::split(0.95);
        let result = cp.fit(&forecasts, &actuals).unwrap();
        b.iter(|| cp.predict_values(black_box(&result), black_box(&point)))
    });

    group.bench_function("bootstrap_1000_replicates", |b| {
        let bp = BootstrapPredictor::new(0.95).n_replicates(1000).seed(42);
        let result = bp.fit(&forecasts, &actuals).unwrap();
        b.iter(|| bp.predict(black_box(&result), black_box(&point)))
    });

    group.bench_function("bootstrap_quantiles_5_levels", |b| {
        let bp = BootstrapPredictor::new(0.95).n_replicates(1000).seed(42);
        let result = bp.fit(&forecasts, &actuals).unwrap();
        let levels = vec![0.10, 0.25, 0.50, 0.75, 0.90];
        b.iter(|| bp.predict_quantiles(black_box(&result), black_box(&point), black_box(&levels)))
    });

    group.finish();
}

fn bench_cross_validation(c: &mut Criterion) {
    use anofox_forecast::utils::cross_validation::CvFoldGenerator;

    let mut group = c.benchmark_group("cv_fold_generation");

    group.bench_function("5_folds_n365", |b| {
        let gen = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(7)
            .min_initial_window(30);
        b.iter(|| gen.generate(black_box(365)).unwrap())
    });

    group.bench_function("20_folds_n1000", |b| {
        let gen = CvFoldGenerator::new()
            .n_folds(20)
            .horizon(12)
            .min_initial_window(50);
        b.iter(|| gen.generate(black_box(1000)).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_model_fit_predict,
    bench_decomposition,
    bench_features,
    bench_differencing,
    bench_cycle_filters,
    bench_postprocess,
    bench_cross_validation,
);
criterion_main!(benches);
