//! M4 Daily accuracy regression tests for `AutoARIMA` (issue #64).
//!
//! Runs `AutoARIMA::seasonal(7)` on 10 representative M4-Daily series
//! and asserts forecast accuracy stays inside bounded multipliers of
//! the statsforecast (Python reference) baseline. Catches the kind of
//! quality regression introduced in v0.5.6 (mean MAE gap blew from
//! +12% to +37%) — the existing unit tests didn't, because they only
//! validate model contracts, not real-world accuracy.
//!
//! ## Data
//!
//! `tests/data/m4_outliers.json` carries the training + test split
//! for 10 series picked from the M4 Daily panel. The set covers the
//! length distribution (short / medium / long) and includes known
//! outlier series where anofox was historically fragile:
//!
//! - Short (n < 200): D2085, D4047
//! - Medium (n 200–1000): D2172, D2178, D1648
//! - Long (n > 1000): D2277, D2283, D2300, D2304, D2305
//!
//! ## Reference baseline
//!
//! `STATSFORECAST_MAE` is the per-series MAE produced by
//! `statsforecast 2.0.3`'s `AutoARIMA` with `seasonal_period=7`, on the
//! same train/test split (14-step horizon). Embedded inline so the
//! assertion is reproducible without an external dependency.
//!
//! ## Three layered assertions (all active in CI)
//!
//! 1. **Aggregate** (`auto_arima_m4_daily_aggregate_within_baseline`):
//!    `mean(anofox_mae) / mean(SF_mae) ≤ AGGREGATE_TOLERANCE`. Catches
//!    the v0.5.6-shaped distribution-wide regression that motivated
//!    this test.
//!
//! 2. **Catastrophic** (`auto_arima_m4_daily_no_series_catastrophic`):
//!    `max_series_mae ≤ SF_mae × CATASTROPHIC_MULTIPLIER`. Hard wall
//!    against silent quality cliffs — no series can blow up by 10×
//!    and stay green.
//!
//! 3. **Per-series 2× tolerance**
//!    (`auto_arima_m4_daily_per_series_within_2x_baseline`). All 10
//!    series currently land below 1.75× the SF baseline. `PER_SERIES_EXEMPTIONS`
//!    provides a structured escape hatch for data-quality artifacts —
//!    none are needed as of v0.9.3.

use std::collections::HashMap;

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use serde_json::Value;

/// Statsforecast 2.0.3 baseline MAE per series on the M4-Daily test
/// split (14-step horizon, `seasonal_period=7`).
///
/// These values were regenerated in v0.9.3 by actually running
/// `statsforecast 2.0.3` against the same train/test splits — the
/// hand-copied numbers in issue #64 had several errors, most notably
/// D2085 (69.2 → 257.1) and D2172 (3148 → 2670). The v0.9.2 D2085
/// "data-quality exemption" was a consequence of the wrong reference;
/// with the correct number anofox is actually slightly better than
/// SF on that series.
///
/// Regeneration script: see PR #130. Reproduce with
/// `pip install statsforecast==2.0.3`, fit `AutoARIMA(season_length=7)`
/// per UID, predict h=14, MAE vs test actuals.
const STATSFORECAST_MAE: &[(&str, f64)] = &[
    ("D2085", 257.1),
    ("D4047", 141.5),
    ("D2172", 2_670.2),
    ("D2178", 3_523.6),
    ("D1648", 183.6),
    ("D2283", 224.4),
    ("D2300", 260.4),
    ("D2277", 198.0),
    ("D2304", 169.0),
    ("D2305", 133.8),
];

/// Aggregate tolerance — `mean(anofox_mae) / mean(SF_mae)` must stay
/// below this.
///
/// After regenerating the SF baseline (v0.9.3) the current ratio is
/// 1.222×, with D2172 (anofox 4628 vs SF 2670) contributing ~25% of
/// the aggregate gap on its own. The remaining 75% is small
/// per-series differences (1–10%) across the other nine series.
///
/// 1.30× gives ~8pp future-regression headroom without being so
/// loose that a meaningful distribution-wide quality drop slips
/// through. v0.5.6's +37% regression would still land at ~1.55× and
/// trip the gate.
const AGGREGATE_TOLERANCE: f64 = 1.30;

/// Per-series tolerance multiplier — anofox MAE must stay below
/// `SF_mae × PER_SERIES_TOLERANCE`. Calibrated against the corrected
/// SF baseline (v0.9.3): all 10 series currently land below 1.75×,
/// with D2172 the worst at 1.73×. The historic 8× catastrophic
/// failure mode (D4047 before #128) and 2.66× (D2172 before any
/// hardening) would both fire.
const PER_SERIES_TOLERANCE: f64 = 2.0;

/// Hard wall on per-series MAE — no series may exceed
/// `SF_mae × CATASTROPHIC_MULTIPLIER` regardless of other tolerances.
const CATASTROPHIC_MULTIPLIER: f64 = 10.0;

const HORIZON: usize = 14;
const SEASONAL_PERIOD: usize = 7;

/// Per-series MAE pulled from a single AutoARIMA fit + 14-step
/// predict. Used by all three assertion tests so the fixture is read
/// once per test.
fn measure_per_series_mae() -> Vec<(String, f64, f64)> {
    let content = std::fs::read_to_string("tests/data/m4_outliers.json")
        .expect("M4 fixture missing — tests/data/m4_outliers.json should be checked in");
    let data: HashMap<String, Value> = serde_json::from_str(&content).expect("parse fixture JSON");

    let mut rows = Vec::with_capacity(STATSFORECAST_MAE.len());
    for (uid, sf_mae) in STATSFORECAST_MAE {
        let series = data
            .get(*uid)
            .unwrap_or_else(|| panic!("fixture missing series {}", uid));
        let train: Vec<f64> = series["train"]
            .as_array()
            .expect("train is array")
            .iter()
            .map(|v| v.as_f64().expect("train value is float"))
            .collect();
        let test: Vec<f64> = series["test"]
            .as_array()
            .expect("test is array")
            .iter()
            .map(|v| v.as_f64().expect("test value is float"))
            .collect();
        assert!(
            test.len() >= HORIZON,
            "{}: test slice too short ({})",
            uid,
            test.len()
        );

        let n = train.len();
        let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..n).map(|i| base + Duration::days(i as i64)).collect();
        let ts = TimeSeries::univariate(timestamps, train).unwrap();

        let mut model = AutoARIMA::seasonal(SEASONAL_PERIOD);
        model
            .fit(&ts)
            .unwrap_or_else(|e| panic!("{} fit failed: {:?}", uid, e));
        let forecast = model
            .predict(HORIZON)
            .unwrap_or_else(|e| panic!("{} predict failed: {:?}", uid, e));
        let pred = forecast.primary();

        let mae = pred
            .iter()
            .take(HORIZON)
            .zip(test.iter().take(HORIZON))
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>()
            / HORIZON as f64;

        rows.push((uid.to_string(), mae, *sf_mae));
    }
    rows
}

fn print_table(rows: &[(String, f64, f64)]) {
    eprintln!("\n=== AutoARIMA M4-Daily accuracy table ===");
    eprintln!(
        "{:<8} {:>12} {:>12} {:>10}",
        "Series", "Anofox MAE", "SF MAE", "Ratio"
    );
    let n = rows.len() as f64;
    let mean_anofox = rows.iter().map(|(_, m, _)| *m).sum::<f64>() / n;
    let mean_sf = rows.iter().map(|(_, _, m)| *m).sum::<f64>() / n;
    for (uid, anofox, sf) in rows {
        eprintln!(
            "{:<8} {:>12.2} {:>12.2} {:>10.3}×",
            uid,
            anofox,
            sf,
            anofox / sf
        );
    }
    eprintln!(
        "{:<8} {:>12.2} {:>12.2} {:>10.3}×",
        "MEAN",
        mean_anofox,
        mean_sf,
        mean_anofox / mean_sf
    );
}

#[test]
fn auto_arima_m4_daily_aggregate_within_baseline() {
    let rows = measure_per_series_mae();
    let n = rows.len() as f64;
    let mean_anofox = rows.iter().map(|(_, m, _)| *m).sum::<f64>() / n;
    let mean_sf = rows.iter().map(|(_, _, m)| *m).sum::<f64>() / n;
    let ratio = mean_anofox / mean_sf;
    if ratio > AGGREGATE_TOLERANCE {
        print_table(&rows);
        panic!(
            "aggregate: mean(anofox)={:.2} / mean(SF)={:.2} = {:.3}× exceeds tolerance {:.2}×",
            mean_anofox, mean_sf, ratio, AGGREGATE_TOLERANCE
        );
    }
}

#[test]
fn auto_arima_m4_daily_no_series_catastrophic() {
    let rows = measure_per_series_mae();
    let mut catastrophic: Vec<String> = Vec::new();
    for (uid, anofox, sf) in &rows {
        let limit = sf * CATASTROPHIC_MULTIPLIER;
        if *anofox > limit {
            catastrophic.push(format!(
                "{}: MAE = {:.2} > {:.2} ({:.0}× SF baseline)",
                uid, anofox, limit, CATASTROPHIC_MULTIPLIER
            ));
        }
    }
    if !catastrophic.is_empty() {
        print_table(&rows);
        panic!(
            "\n{} series catastrophically exceeded {}× SF baseline:\n  - {}",
            catastrophic.len(),
            CATASTROPHIC_MULTIPLIER,
            catastrophic.join("\n  - ")
        );
    }
}

/// No per-series exemptions are needed as of v0.9.3. The earlier
/// D2085 exemption was a consequence of the wrong SF baseline (69.2
/// instead of 257.1); with the corrected reference D2085 lands at
/// ~0.99× — better than SF.
const PER_SERIES_EXEMPTIONS: &[(&str, &str)] = &[];

#[test]
fn auto_arima_m4_daily_per_series_within_2x_baseline() {
    let rows = measure_per_series_mae();
    let exempt: std::collections::HashMap<&str, &str> =
        PER_SERIES_EXEMPTIONS.iter().copied().collect();
    let mut breaches: Vec<String> = Vec::new();
    for (uid, anofox, sf) in &rows {
        let limit = sf * PER_SERIES_TOLERANCE;
        if *anofox > limit {
            if let Some(reason) = exempt.get(uid.as_str()) {
                eprintln!(
                    "{}: MAE = {:.2} > {:.2} ({:.2}× SF baseline) — exempt: {}",
                    uid,
                    anofox,
                    limit,
                    anofox / sf,
                    reason,
                );
                continue;
            }
            breaches.push(format!(
                "{}: MAE = {:.2} > {:.2} ({:.2}× SF baseline)",
                uid,
                anofox,
                limit,
                anofox / sf
            ));
        }
    }
    if !breaches.is_empty() {
        print_table(&rows);
        panic!(
            "\n{} series exceed {}× SF tolerance:\n  - {}",
            breaches.len(),
            PER_SERIES_TOLERANCE,
            breaches.join("\n  - ")
        );
    }
}
