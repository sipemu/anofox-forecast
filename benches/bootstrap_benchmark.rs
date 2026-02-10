use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::bootstrap::{bootstrap_forecast, BootstrapConfig};
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn generate_ar1_series(n: usize) -> Vec<f64> {
    let mut rng_state: u64 = 42;
    let mut series = vec![0.0; n];
    series[0] = 1.0;
    for i in 1..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
        series[i] = 0.7 * series[i - 1] + noise;
    }
    series
}

fn bench_bootstrap_residual(c: &mut Criterion) {
    let values = generate_ar1_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    let mut model = ARIMA::new(1, 1, 0);
    model.fit(&ts).unwrap();

    let config = BootstrapConfig::new(100).with_seed(42);

    c.bench_function("bootstrap_residual_n100_s100_h10", |b| {
        b.iter(|| {
            bootstrap_forecast(&model, &ts, 10, 0.95, &config).unwrap();
        })
    });
}

fn bench_bootstrap_block(c: &mut Criterion) {
    let values = generate_ar1_series(100);
    let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

    let mut model = ARIMA::new(1, 1, 0);
    model.fit(&ts).unwrap();

    let config = BootstrapConfig::new(100).with_block_size(10).with_seed(42);

    c.bench_function("bootstrap_block_n100_s100_h10", |b| {
        b.iter(|| {
            bootstrap_forecast(&model, &ts, 10, 0.95, &config).unwrap();
        })
    });
}

criterion_group!(benches, bench_bootstrap_residual, bench_bootstrap_block);
criterion_main!(benches);
