//! SIMD benchmarks comparing scalar vs Trueno-accelerated implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_sum(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("sum");

    for size in sizes {
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

fn bench_sum_of_squares(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("sum_of_squares");

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| black_box(d.iter().map(|x| x * x).sum::<f64>()))
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| black_box(anofox_forecast::simd::sum_of_squares(d)))
        });
    }
    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("dot");

    for size in sizes {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(anofox_forecast::simd::dot(a, b))),
        );
    }
    group.finish();
}

fn bench_mean(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("mean");

    for size in sizes {
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

fn bench_variance(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("variance");

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| {
                let m = d.iter().sum::<f64>() / d.len() as f64;
                black_box(d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / d.len() as f64)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| black_box(anofox_forecast::simd::variance(d)))
        });
    }
    group.finish();
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("euclidean_distance");

    for size in sizes {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f64>()
                            .sqrt(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(anofox_forecast::simd::squared_distance(a, b).sqrt()))
            },
        );
    }
    group.finish();
}

fn bench_l1_distance(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let mut group = c.benchmark_group("l1_distance");

    for size in sizes {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    black_box(
                        a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x - y).abs())
                            .sum::<f64>(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(anofox_forecast::simd::l1_distance(a, b))),
        );
    }
    group.finish();
}

fn bench_zscore(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];
    let mut group = c.benchmark_group("zscore");

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, d| {
            b.iter(|| {
                let m = d.iter().sum::<f64>() / d.len() as f64;
                let v = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / d.len() as f64;
                let s = v.sqrt();
                black_box(d.iter().map(|x| (x - m) / s).collect::<Vec<f64>>())
            })
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &data, |b, d| {
            b.iter(|| black_box(anofox_forecast::simd::zscore(d)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_sum_of_squares,
    bench_dot,
    bench_mean,
    bench_variance,
    bench_euclidean_distance,
    bench_l1_distance,
    bench_zscore,
);
criterion_main!(benches);
