use anofox_forecast::detection::welch_periodogram;
use anofox_forecast::features::{agg_autocorrelation, autocorrelation};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_seasonal_signal(n: usize, period: usize) -> Vec<f64> {
    let mut rng_state: u64 = 789;
    (0..n)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            let trend = 0.02 * i as f64;
            trend + seasonal + noise + 20.0
        })
        .collect()
}

fn bench_autocorrelation(c: &mut Criterion) {
    let mut group = c.benchmark_group("autocorrelation");
    for &n in &[500, 2000] {
        let signal = generate_seasonal_signal(n, 12);
        let max_lag = n / 4;
        group.bench_with_input(BenchmarkId::new("single_lag/n", n), &signal, |b, sig| {
            b.iter(|| {
                autocorrelation(sig, 12);
            })
        });
        group.bench_with_input(BenchmarkId::new("agg_mean/n", n), &signal, |b, sig| {
            b.iter(|| {
                agg_autocorrelation(sig, max_lag, "mean");
            })
        });
    }
    group.finish();
}

fn bench_welch_periodogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("welch_periodogram");
    for &n in &[500, 2000] {
        let signal = generate_seasonal_signal(n, 12);
        let window = 128.min(n);
        group.bench_with_input(BenchmarkId::new("n", n), &signal, |b, sig| {
            b.iter(|| {
                welch_periodogram(sig, window, 0.5);
            })
        });
    }
    group.finish();
}

fn bench_welch_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("welch_window_size");
    let signal = generate_seasonal_signal(2000, 12);
    for &window in &[64, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::new("w", window), &signal, |b, sig| {
            b.iter(|| {
                welch_periodogram(sig, window, 0.5);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_autocorrelation,
    bench_welch_periodogram,
    bench_welch_window_sizes
);
criterion_main!(benches);
