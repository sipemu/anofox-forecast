//! Reproduces issue #195: LaplaceForecaster over-damps seasonality on
//! amplitude-declining series.
//!
//! Ports the user's DuckDB SQL to a plain Rust synthetic. 37 months
//! (Jun 2023 → Jun 2026) with strong 2023-2025 seasonal factors
//! (0.27 → 1.70) and amplitude halved (×0.45) in 2026.
//!
//! Reports softmax weights per leaf name at end of fit so we can see
//! which leaves are winning after the 2026 amplitude drop.
//!
//! Run:
//!   cargo run --release --features distributional \
//!     --example issue_195_amplitude_decline

use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::{DistributionalForecaster, Forecaster};
use anofox_forecast::prelude::TimeSeries;
use chrono::{DateTime, TimeZone, Utc};

const BASE: f64 = 5000.0;
const NOISE: f64 = 0.12;
// Retail-shape seasonal factors, Jan..Dec.
const SEAS: [f64; 12] = [
    0.50, 0.55, 1.70, 1.65, 1.30, 1.00, 0.80, 0.50, 0.40, 0.37, 0.32, 0.27,
];
// 2026 amplitude scale — factor applied to (f - 1). 1.0 = full swing;
// 0.45 = amplitude halved.
const AMP_2026: f64 = 0.45;

/// Deterministic noise per index — Wichura-like LCG.
fn noise_at(i: usize, seed_base: u64) -> f64 {
    let mut seed = seed_base ^ (i as u64).wrapping_mul(6_364_136_223_846_793_005);
    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
    2.0 * u - 1.0
}

fn synth() -> (Vec<DateTime<Utc>>, Vec<f64>) {
    // 37 obs: Jun 2023 (7 months) + 2024 (12) + 2025 (12) + Jan–Jun 2026 (6).
    let start = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
    let mut stamps = Vec::with_capacity(37);
    let mut vals = Vec::with_capacity(37);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..37 {
        let d = add_months(start, i);
        let month_idx = ((d.month() - 1) as usize) % 12;
        let year = d.year();
        let amp = if year == 2026 { AMP_2026 } else { 1.0 };
        // Deterministic uniform → symmetric noise in [-1, 1].
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = ((seed >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0 - 1e-12);
        let noise = 2.0 * u - 1.0;
        let y = (BASE * (1.0 + amp * (SEAS[month_idx] - 1.0)) * (1.0 + NOISE * noise)).max(0.0);
        stamps.push(d);
        vals.push(y);
    }
    (stamps, vals)
}

fn add_months(dt: DateTime<Utc>, n: i64) -> DateTime<Utc> {
    let total = dt.year() as i64 * 12 + (dt.month() as i64 - 1) + n;
    let year = (total.div_euclid(12)) as i32;
    let month = (total.rem_euclid(12) + 1) as u32;
    Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).unwrap()
}

use chrono::Datelike;

fn run(label: &str, mut model: LaplaceForecaster, ts: &TimeSeries) {
    use anofox_forecast::models::inspect::{Explanation, Inspectable};
    model.fit(ts).expect("fit fail");
    let f = model.forecast_dist(12).expect("predict fail");
    let means: Vec<f64> = f
        .iter()
        .map(|mix| mix.components.iter().map(|(w, g)| w * g.mean).sum::<f64>())
        .collect();
    let peak = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough = means.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "\n{label}:\n  peak={peak:.0}  trough={trough:.0}  ratio={:.2}x  swing={:.0}",
        peak / trough.max(1.0),
        peak - trough,
    );
    print!("  forecast: ");
    for m in &means {
        print!("{:.0} ", m);
    }
    println!();
    // Which leaves are winning the softmax? Print top-5 by weight.
    if let Ok(Explanation::Laplace(ex)) = Inspectable::explanation(&model) {
        let mut weighted: Vec<(&str, f64)> = ex
            .leaf_names
            .iter()
            .map(|s| s.as_str())
            .zip(ex.leaf_weights.iter().copied())
            .collect();
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        print!("  softmax top-5: ");
        for (name, w) in weighted.iter().take(5) {
            print!("{name}={w:.3} ");
        }
        println!();
    }
}

fn main() {
    let (stamps, vals) = synth();
    let ts = TimeSeries::univariate(stamps.clone(), vals.clone()).unwrap();

    println!("=== Actuals (last 12 months of training, Jul 2025 → Jun 2026) ===");
    for i in 25..37 {
        let d = stamps[i];
        println!("  {}-{:02}: {:.0}", d.year(), d.month(), vals[i]);
    }
    let last12 = &vals[25..37];
    let peak = last12.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough = last12.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "  actual peak/trough over last 12: {:.0} / {:.0} → ratio {:.2}x",
        peak,
        trough,
        peak / trough.max(1.0)
    );

    // Recent-cycle only (2026 partial): what ratio SHOULD the forecast reflect?
    let last6 = &vals[31..37]; // Jan-Jun 2026
    let peak2026 = last6.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough2026 = last6.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "  actual 2026 (partial): peak {:.0} / trough {:.0} → ratio {:.2}x (target for forecast)",
        peak2026,
        trough2026,
        peak2026 / trough2026.max(1.0)
    );

    run(
        ".auto() (baseline, no batch init)",
        LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(12),
        &ts,
    );
    run(
        ".auto() + with_seasonal_batch_init() (0.15 fix)",
        LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(12)
            .with_seasonal_batch_init(),
        &ts,
    );
    run(
        ".skaters() (no batch init)",
        LaplaceForecaster::new()
            .skaters()
            .auto_with_seasonal_period(12),
        &ts,
    );
    run(
        ".skaters() + with_seasonal_batch_init()",
        LaplaceForecaster::new()
            .skaters()
            .auto_with_seasonal_period(12)
            .with_seasonal_batch_init(),
        &ts,
    );
    // Prototype fix: drop the level-forecast families (plain diff-EMA,
    // Yeo-Johnson-wrapped diff-EMA) from the pool. These are level
    // trackers with excellent 1-step but flat multi-step forecasts.
    run(
        ".skaters() minus diff-EMA family (prototype fix)",
        LaplaceForecaster::new()
            .skaters()
            .auto_with_seasonal_period(12)
            .with_seasonal_batch_init()
            .with_diff_ema(&[])
            .with_yj_coord(&[]),
        &ts,
    );
    // KEY: .auto_with_seasonal_period(p) only sets a FALLBACK for
    // .auto()'s ACF detection. It does NOT add a seasonal-EMA leaf
    // to .skaters()'s pool. To force one, use .with_seasonal(p).
    run(
        ".skaters().with_seasonal(12).with_seasonal_batch_init()",
        LaplaceForecaster::new()
            .skaters()
            .with_seasonal(12)
            .with_seasonal_batch_init(),
        &ts,
    );

    // ---- Extended regression panel: increasing / phase-shift / other ----
    println!("\n\n=== EXTENDED PANEL ===");
    #[allow(clippy::type_complexity)]
    let scenarios: &[(&str, Box<dyn Fn(usize) -> f64>, f64)] = &[
        // (label, function returning the value at index i, target ratio the
        //  next-cycle forecast should aim for)
        (
            "declining amplitude (issue #195 shape)",
            Box::new(|i| {
                let month = i % 12;
                let year_off = i / 12;
                let amp = if year_off >= 3 { 0.45 } else { 1.0 };
                let n = NOISE * noise_at(i, 0x1_deca);
                (BASE * (1.0 + amp * (SEAS[month] - 1.0)) * (1.0 + n)).max(0.0)
            }),
            // 2026 partial swing ≈ 1.99×
            1.99,
        ),
        (
            "increasing amplitude (retail expanding)",
            Box::new(|i| {
                let month = i % 12;
                let year_off = i / 12;
                // amp grows 0.4 → 1.4 over 4 cycles.
                let amp = 0.4 + 0.33 * (year_off.min(3) as f64);
                let n = NOISE * noise_at(i, 0x2_1CE);
                (BASE * (1.0 + amp * (SEAS[month] - 1.0)) * (1.0 + n)).max(0.0)
            }),
            // With amp=1.4 at year 4, peak factor 1.7 → 3979; trough 0.27 → -30 → 0 (clamp).
            // Effective ratio ~5-10 depending on noise. Wide target.
            5.0,
        ),
        (
            "phase-shifted seasonality (peak moves March -> May)",
            Box::new(|i| {
                let month = (i % 12) as i32;
                let year_off = i / 12;
                // Shift peak month by 2 months for year 3+.
                let shift = if year_off >= 3 { 2 } else { 0 };
                let idx = ((month - shift).rem_euclid(12)) as usize;
                let n = NOISE * noise_at(i, 0x3_5F1);
                (BASE * (1.0 + 1.0 * (SEAS[idx] - 1.0)) * (1.0 + n)).max(0.0)
            }),
            // Full swing preserved; ratio same as original ~6.3×.
            6.0,
        ),
        (
            "constant amplitude control (no change)",
            Box::new(|i| {
                let month = i % 12;
                let n = NOISE * noise_at(i, 0x4_C047);
                (BASE * (1.0 + 1.0 * (SEAS[month] - 1.0)) * (1.0 + n)).max(0.0)
            }),
            // Full 8500/1350 ≈ 6.3× peak/trough.
            6.3,
        ),
        (
            "additive drift + seasonal (upward trend + fixed amp)",
            Box::new(|i| {
                let month = i % 12;
                // linear drift +100 per month
                let drift = 100.0 * i as f64;
                let n = NOISE * noise_at(i, 0x5_D71F);
                (BASE + drift + BASE * (SEAS[month] - 1.0) * 1.0 * (1.0 + 0.3 * n)).max(0.0)
            }),
            5.0,
        ),
        (
            "recent regime change (year 4 is anomalous / near-flat)",
            Box::new(|i| {
                let month = i % 12;
                let year_off = i / 12;
                // Full seasonal for years 1-3, then amplitude → 0.1 for year 4.
                let amp = if year_off >= 3 { 0.1 } else { 1.0 };
                let n = NOISE * noise_at(i, 0x6_A20);
                (BASE * (1.0 + amp * (SEAS[month] - 1.0)) * (1.0 + n)).max(0.0)
            }),
            // year 4 has near-zero swing; forecast should be near-flat.
            1.2,
        ),
    ];

    for (label, f, target_ratio) in scenarios.iter() {
        println!("\n\n### {label} ###");
        // Build a 37-obs series with the same schedule as the main synthetic.
        let mut vals2 = Vec::with_capacity(37);
        let start = Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap();
        let mut stamps2 = Vec::with_capacity(37);
        // Use the SAME index -> month mapping the main synth uses so seasonal
        // phase alignment is consistent (Jun 2023 = index 0 = SEAS[5]).
        for i in 0..37 {
            let d = add_months(start, i as i64);
            let month_idx = (d.month() - 1) as usize;
            // Map (year_off, month_idx) into a scenario-friendly cycle index.
            // For simplicity feed the scenario a linear index that wraps
            // month_idx correctly:
            let year_off = ((d.year() - 2023) as usize).saturating_sub(0);
            let scenario_i = year_off * 12 + month_idx;
            vals2.push(f(scenario_i));
            stamps2.push(d);
        }
        let ts2 = TimeSeries::univariate(stamps2, vals2.clone()).unwrap();
        // Recent-year target for comparison (assumes user wants next-cycle
        // shape to match the recent one).
        let last12 = &vals2[25..37];
        let peak = last12.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let trough = last12
            .iter()
            .cloned()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        println!(
            "  actuals last-12 peak/trough: {:.0}/{:.0} = {:.2}x   target ≈ {:.2}x",
            peak,
            trough,
            peak / trough.max(1.0),
            target_ratio,
        );
        run(
            "  .skaters().auto_with_seasonal_period(12) (FIX A)",
            LaplaceForecaster::new()
                .skaters()
                .auto_with_seasonal_period(12),
            &ts2,
        );
        run(
            "  .skaters().auto_with_seasonal_period(12).with_seasonal_batch_init()",
            LaplaceForecaster::new()
                .skaters()
                .auto_with_seasonal_period(12)
                .with_seasonal_batch_init(),
            &ts2,
        );
    }
}
