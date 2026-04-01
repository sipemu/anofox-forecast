use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::Forecaster;
use anofox_forecast::prelude::TimeSeries;
use chrono::{DateTime, Duration, Utc};
use std::time::Instant;

fn main() {
    let n = 2358;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = i as f64 * 0.05;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let noise = ((i * 7 + 13) % 100) as f64 / 25.0 - 2.0;
            10.0 + trend + seasonal + noise
        })
        .collect();

    let base: DateTime<Utc> = "2020-01-01T00:00:00Z".parse().unwrap();
    let dates: Vec<DateTime<Utc>> = (0..n).map(|i| base + Duration::days(i as i64)).collect();
    let ts = TimeSeries::univariate(dates, values).unwrap();

    // Warm up
    {
        let config = AutoARIMAConfig::default().with_seasonal_period(7);
        let mut model = AutoARIMA::with_config(config);
        let _ = model.fit(&ts);
    }

    let start = Instant::now();
    let config = AutoARIMAConfig::default().with_seasonal_period(7);
    let mut model = AutoARIMA::with_config(config);
    model.fit(&ts).unwrap();
    let elapsed = start.elapsed();

    println!("AutoARIMA n={} p=7: {:?}", n, elapsed);
}
