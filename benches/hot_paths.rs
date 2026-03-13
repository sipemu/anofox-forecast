use anofox_forecast::core::{Forecast, TimeSeries};
use chrono::{Duration, TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn bench_simd_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_sum");
    for &size in &[100, 1000, 10000] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| black_box(d.iter().sum::<f64>()))
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| black_box(anofox_forecast::simd::sum(d)))
        });
    }
    group.finish();
}

fn bench_simd_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_mean");
    for &size in &[100, 1000, 10000] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| black_box(d.iter().sum::<f64>() / d.len() as f64))
        });
        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| black_box(anofox_forecast::simd::mean(d)))
        });
    }
    group.finish();
}

fn bench_simd_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_dot");
    for &size in &[100, 1000, 10000] {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let bv: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.001).collect();
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &bv),
            |bench, (a, b)| {
                bench.iter(|| black_box(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &bv),
            |bench, (a, b)| bench.iter(|| black_box(anofox_forecast::simd::dot(a, b))),
        );
    }
    group.finish();
}

fn bench_forecast_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecast_construction");
    group.bench_function("from_values_h100", |b| {
        let vals: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        b.iter(|| {
            black_box(Forecast::from_values(vals.clone()));
        })
    });
    group.bench_function("from_values_with_intervals_h100", |b| {
        let point: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let lower: Vec<f64> = point.iter().map(|v| v - 2.0).collect();
        let upper: Vec<f64> = point.iter().map(|v| v + 2.0).collect();
        b.iter(|| {
            black_box(Forecast::from_values_with_intervals(
                point.clone(),
                lower.clone(),
                upper.clone(),
            ));
        })
    });
    group.bench_function("multivariate_3dim_h50", |b| {
        b.iter(|| {
            let mut forecast = Forecast::with_dimensions(3);
            for dim in 0..3 {
                forecast
                    .series_mut(dim)
                    .extend((0..50).map(|i| (dim * 50 + i) as f64));
            }
            black_box(forecast);
        })
    });
    group.finish();
}

fn bench_timeseries_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeseries_construction");
    for &n in &[100, 1000, 10000] {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 10.0).collect();
        group.bench_with_input(
            BenchmarkId::new("univariate", n),
            &(timestamps.clone(), values.clone()),
            |b, (ts, vals)| {
                b.iter(|| {
                    black_box(TimeSeries::univariate(ts.clone(), vals.clone()).unwrap());
                })
            },
        );
    }
    group.finish();
}

fn bench_timeseries_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeseries_slice");
    let n = 10000;
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 10.0).collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    for &window in &[100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::new("window", window), &ts, |b, ts| {
            b.iter(|| {
                black_box(ts.slice(0, window).unwrap());
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_sum,
    bench_simd_mean,
    bench_simd_dot,
    bench_forecast_construction,
    bench_timeseries_construction,
    bench_timeseries_slice
);
criterion_main!(benches);
