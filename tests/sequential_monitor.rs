//! Integration tests for the sequential monitor module.
//!
//! These exercise the public API end-to-end: fitting a forecaster, monitoring
//! its residuals (in-sample and CV), and verifying the detector fires when an
//! injected drift is introduced.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::Forecaster;
use anofox_forecast::monitor::{
    monitor_forecaster, monitor_forecaster_cv, CriticalValue, Detector, ForecastErrorType,
    SequentialConfig, SequentialDetector,
};
use chrono::{TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| {
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::hours(i as i64)
        })
        .collect()
}

fn normal_sample(rng: &mut StdRng, mean: f64, sd: f64) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen();
    mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// End-to-end: fit Naive on a level series, then inject a level shift
// ---------------------------------------------------------------------------

#[test]
fn monitor_forecaster_in_sample_residuals_round_trip() {
    let mut rng = StdRng::seed_from_u64(99);
    // Pure noise around a constant level — Naive's in-sample residuals are
    // just the first differences, which under iid noise are approximately
    // zero-mean.
    let n = 400;
    let values: Vec<f64> = (0..n)
        .map(|_| 10.0 + normal_sample(&mut rng, 0.0, 1.0))
        .collect();
    let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

    let mut model = Naive::new();
    model.fit(&ts).unwrap();

    let cfg = SequentialConfig::new(150)
        .detector(Detector::PageCusum)
        .error_type(ForecastErrorType::Both);
    let detector = monitor_forecaster(&model, cfg).unwrap();

    // Stationary noise — the detector should not fire under the null.
    assert!(
        !detector.has_detected(),
        "false alarm at τ={:?}",
        detector.first_detection()
    );
    assert!(!detector.cusum().is_empty());
}

#[test]
fn online_streaming_workflow_detects_injected_shift() {
    let mut rng = StdRng::seed_from_u64(42);

    // Phase 1: 300 stable observations (training + early monitoring).
    let stable: Vec<f64> = (0..300)
        .map(|_| normal_sample(&mut rng, 0.0, 1.0))
        .collect();

    let cfg = SequentialConfig::new(200)
        .detector(Detector::PageCusum)
        .error_type(ForecastErrorType::Raw)
        .critical_value(CriticalValue::Lookup);

    let mut detector = SequentialDetector::fit(&stable, cfg).unwrap();
    assert!(!detector.has_detected(), "false alarm during stable phase");

    // Phase 2: stream in 50 more iid observations one by one — still nothing.
    for _ in 0..50 {
        let next = normal_sample(&mut rng, 0.0, 1.0);
        detector.update(&[next]).unwrap();
    }
    assert!(
        !detector.has_detected(),
        "false alarm during quiet streaming phase, τ={:?}",
        detector.first_detection()
    );

    // Phase 3: drift kicks in — feed N(2, 1) errors.
    let mut detected_at = None;
    for i in 0..200 {
        let next = normal_sample(&mut rng, 2.0, 1.0);
        detector.update(&[next]).unwrap();
        if detector.has_detected() && detected_at.is_none() {
            detected_at = Some(i);
            break;
        }
    }
    assert!(
        detected_at.is_some(),
        "detector failed to fire after 200 drifted observations"
    );
    let i = detected_at.unwrap();
    assert!(
        i < 100,
        "detector fired too late ({} updates after drift)",
        i
    );
}

#[test]
fn monitor_forecaster_cv_calibrates_via_rolling_origin() {
    // Build a long, mostly-stationary series and verify monitor_forecaster_cv
    // produces a SequentialDetector whose CUSUM history has the expected
    // length and does not spuriously fire.
    let mut rng = StdRng::seed_from_u64(2024);
    let n = 400;
    let values: Vec<f64> = (0..n)
        .map(|_| 10.0 + normal_sample(&mut rng, 0.0, 1.0))
        .collect();
    let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

    let cfg = SequentialConfig::new(50)
        .detector(Detector::PageCusum)
        .error_type(ForecastErrorType::Raw)
        .critical_value(CriticalValue::Lookup);

    let detector = monitor_forecaster_cv(Naive::new, &ts, cfg, 100, 1).unwrap();

    // CV with initial_train_size=100, horizon=1, step_size=1 produces
    // n - 100 = 300 windows, each yielding one prediction. The detector then
    // takes the first 50 as training and monitors the remaining 250.
    assert_eq!(detector.cusum().len() + 50, 300);
    // Stationary noise: should rarely if ever fire under PageCusum α=0.05.
    // With seed 2024 the path is quiet — we accept "no detection" or
    // "detection in the second half" but flag a near-immediate trip as a bug.
    if let Some(tau) = detector.tau() {
        assert!(
            tau > 50,
            "implausibly early trip on stationary CV residuals: τ={}",
            tau
        );
    }
}

#[test]
fn variance_shift_only_caught_by_squared_stream() {
    let mut rng = StdRng::seed_from_u64(13);
    // Stable mean, stable variance up to index 300; afterwards the variance
    // triples while the mean stays at 0.
    let mut errors: Vec<f64> = (0..300)
        .map(|_| normal_sample(&mut rng, 0.0, 1.0))
        .collect();
    errors.extend((0..150).map(|_| normal_sample(&mut rng, 0.0, 3.0)));

    let cfg = SequentialConfig::new(200)
        .error_type(ForecastErrorType::Both)
        .detector(Detector::PageCusum)
        .critical_value(CriticalValue::Lookup);

    let detector = SequentialDetector::fit(&errors, cfg).unwrap();
    assert!(
        detector.tau_squared().is_some(),
        "squared stream should fire on variance shift"
    );
    // The raw stream may or may not fire because the mean shift is zero.
    // The "Both" mode's first_detection should match the squared stream
    // (or be earlier, if the raw stream got lucky).
    let first = detector.first_detection().unwrap();
    let sq = detector.tau_squared().unwrap();
    assert!(first <= sq);
}

#[test]
fn detector_serializes_state_for_external_persistence() {
    // Even without serde we should be able to clone the detector and
    // continue updating it on the clone independently. This proves the state
    // is fully self-contained.
    let mut rng = StdRng::seed_from_u64(5);
    let errors: Vec<f64> = (0..200)
        .map(|_| normal_sample(&mut rng, 0.0, 1.0))
        .collect();
    let cfg = SequentialConfig::new(100)
        .error_type(ForecastErrorType::Raw)
        .detector(Detector::PageCusum)
        .critical_value(CriticalValue::Fixed(2.5));

    let detector = SequentialDetector::fit(&errors, cfg).unwrap();
    let mut clone_a = detector.clone();
    let mut clone_b = detector.clone();

    let next: Vec<f64> = (0..20).map(|_| normal_sample(&mut rng, 0.0, 1.0)).collect();
    clone_a.update(&next).unwrap();
    clone_b.update(&next).unwrap();

    assert_eq!(clone_a.cusum(), clone_b.cusum());
    assert_eq!(clone_a.threshold(), clone_b.threshold());
}
