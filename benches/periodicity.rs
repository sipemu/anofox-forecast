//! Benchmarks for periodicity detection algorithms.

use anofox_forecast::detection::{
    detect_period, detect_period_ensemble, ACFPeriodicityDetector, Autoperiod, CFDAutoperiod,
    FFTPeriodicityDetector, PeriodicityDetector, SAZED,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_sine(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
        .collect()
}

fn generate_multi_sine(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 52.0).sin()
        })
        .collect()
}

fn bench_detectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("periodicity_detectors");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let signal = generate_sine(*size, 12);

        group.bench_with_input(BenchmarkId::new("ACF", size), size, |b, _| {
            let detector = ACFPeriodicityDetector::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });

        group.bench_with_input(BenchmarkId::new("FFT", size), size, |b, _| {
            let detector = FFTPeriodicityDetector::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });

        group.bench_with_input(BenchmarkId::new("Autoperiod", size), size, |b, _| {
            let detector = Autoperiod::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });

        group.bench_with_input(BenchmarkId::new("CFDAutoperiod", size), size, |b, _| {
            let detector = CFDAutoperiod::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });

        group.bench_with_input(BenchmarkId::new("SAZED", size), size, |b, _| {
            let detector = SAZED::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });
    }

    group.finish();
}

fn bench_convenience_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("convenience_functions");

    let signal = generate_multi_sine(365);

    group.bench_function("detect_period", |b| {
        b.iter(|| detect_period(black_box(&signal)))
    });

    group.bench_function("detect_period_ensemble", |b| {
        b.iter(|| detect_period_ensemble(black_box(&signal)))
    });

    group.finish();
}

fn bench_fft_vs_acf(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vs_acf_scaling");

    for size in [256, 512, 1024, 2048, 4096].iter() {
        let signal = generate_sine(*size, 12);

        group.bench_with_input(BenchmarkId::new("FFT_only", size), size, |b, _| {
            let detector = FFTPeriodicityDetector::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });

        group.bench_with_input(BenchmarkId::new("ACF_only", size), size, |b, _| {
            let detector = ACFPeriodicityDetector::default();
            b.iter(|| detector.detect(black_box(&signal)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_detectors,
    bench_convenience_functions,
    bench_fft_vs_acf
);
criterion_main!(benches);
