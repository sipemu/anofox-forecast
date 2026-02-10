use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::seasonality::STL;

fn generate_seasonal_series(n: usize, period: usize) -> Vec<f64> {
    let mut rng_state: u64 = 123;
    let mut series = Vec::with_capacity(n);
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
        let trend = 0.05 * i as f64;
        series.push(trend + seasonal + noise + 20.0);
    }
    series
}

fn bench_stl_decompose(c: &mut Criterion) {
    let values_200 = generate_seasonal_series(200, 12);
    c.bench_function("stl_decompose_n200_p12", |b| {
        let stl = STL::new(12);
        b.iter(|| {
            stl.decompose(&values_200).unwrap();
        })
    });

    let values_500 = generate_seasonal_series(500, 12);
    c.bench_function("stl_decompose_n500_p12", |b| {
        let stl = STL::new(12);
        b.iter(|| {
            stl.decompose(&values_500).unwrap();
        })
    });

    let values_1000 = generate_seasonal_series(1000, 52);
    c.bench_function("stl_decompose_n1000_p52", |b| {
        let stl = STL::new(52);
        b.iter(|| {
            stl.decompose(&values_1000).unwrap();
        })
    });
}

fn bench_stl_robust(c: &mut Criterion) {
    let values_200 = generate_seasonal_series(200, 12);
    c.bench_function("stl_robust_n200_p12", |b| {
        let stl = STL::new(12).robust();
        b.iter(|| {
            stl.decompose(&values_200).unwrap();
        })
    });
}

criterion_group!(benches, bench_stl_decompose, bench_stl_robust);
criterion_main!(benches);
