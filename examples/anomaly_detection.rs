//! Streaming anomaly detection example.
//!
//! Simulates a stable Gaussian stream with:
//! - An **injected point outlier** at tick 700 (12 σ);
//! - A **structural level shift** starting at tick 900 (mean jumps by 5 σ).
//!
//! Wraps [`LaplaceForecaster`] in a [`MahalanobisDetector`] and prints
//! flagged ticks: tick idx, y, d², p-value, run.
//!
//! Run:
//! ```
//! cargo run --release --features anomaly --example anomaly_detection
//! ```

use anofox_forecast::anomaly::{MahalanobisConfig, MahalanobisDetector};
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::prelude::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

const N_TOTAL: usize = 1200;
const N_TRAIN: usize = 300;
const SPIKE_TICK: usize = 700;
const SHIFT_START: usize = 900;

fn synthetic_stream() -> Vec<f64> {
    (0..N_TOTAL)
        .map(|i| {
            // Deterministic Box-Muller via LCG.
            let seed = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u1 = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
            let seed2 = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u2 = ((seed2 >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mut y = z; // N(0, 1) baseline
            if i == SPIKE_TICK {
                y = 12.0; // 12-sigma spike
            }
            if i >= SHIFT_START {
                y += 5.0; // level shift
            }
            y
        })
        .collect()
}

fn main() {
    let vals = synthetic_stream();
    let base_ts = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..N_TRAIN)
        .map(|i| base_ts + Duration::hours(i as i64))
        .collect();
    let train = TimeSeries::univariate(stamps, vals[..N_TRAIN].to_vec()).unwrap();

    let cfg = MahalanobisConfig::new(8);
    let mut det = MahalanobisDetector::fit_and_wrap(LaplaceForecaster::new().auto(), &train, cfg)
        .expect("fit failed");

    println!("t\ty\t\td^2\t\tp-value\t\trun\tnote");
    println!("----\t--------\t--------\t--------\t----\t----");

    let mut n_flagged = 0;
    for i in N_TRAIN..N_TOTAL {
        det.observe(vals[i]).expect("observe failed");
        let out = det.state();
        let (d2, p) = match (out.d2, out.p_value) {
            (Some(a), Some(b)) => (a, b),
            _ => continue, // warmup
        };
        let flag = p < 0.001;
        let note = match i {
            SPIKE_TICK => "SPIKE",
            SHIFT_START => "SHIFT",
            _ => "",
        };
        if flag || !note.is_empty() {
            println!(
                "{}\t{:+.3}\t{:.2}\t\t{:.2e}\t{}\t{}",
                i, vals[i], d2, p, out.run, note
            );
            if flag {
                n_flagged += 1;
            }
        }
    }
    println!("\n{} ticks flagged with p < 0.001", n_flagged);
    println!(
        "Injected: point outlier at t={}, level shift starting t={}",
        SPIKE_TICK, SHIFT_START
    );
}
