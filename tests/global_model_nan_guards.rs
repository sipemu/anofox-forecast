//! NaN/Inf guard-path tests for GlobalETS (V-01), GlobalAutoETS (V-01 auto-wrapper),
//! and GlobalCroston (V-02).
//!
//! GlobalTheta guard tests (V-03) live in tests/global_theta_smoke.rs.
//! Each test also exercises the happy path to confirm the guard does not
//! fire on valid finite input.

use anofox_forecast::error::ForecastError;
use anofox_forecast::models::exponential::{ETSSpec, GlobalAutoETS, GlobalETS, ModelPool};
use anofox_forecast::models::intermittent::GlobalCroston;

// ─────────────────────────────────────────────────────────────────────────────
// GlobalETS — V-01 guard
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn global_ets_nan_guard() {
    // Panel where the second series contains NaN
    let series = vec![
        vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0],
        vec![20.0, f64::NAN, 21.0, 23.0, 22.0, 24.0, 23.0, 25.0],
    ];
    let mut model = GlobalETS::new(ETSSpec::ann(), 1);
    let err = model.fit(&series).expect_err("fit must fail on NaN input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_ets_inf_guard() {
    // Panel where the first series contains Inf
    let series = vec![vec![
        10.0,
        f64::INFINITY,
        11.0,
        13.0,
        12.0,
        14.0,
        13.0,
        15.0,
    ]];
    let mut model = GlobalETS::new(ETSSpec::ann(), 1);
    let err = model.fit(&series).expect_err("fit must fail on Inf input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_ets_valid_panel_guard_does_not_fire() {
    // All-finite panel must fit successfully (guard must not fire on valid data)
    let series = vec![
        vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0],
        vec![20.0, 22.0, 21.0, 23.0, 22.0, 24.0, 23.0, 25.0],
    ];
    let mut model = GlobalETS::new(ETSSpec::ann(), 1);
    model.fit(&series).expect("fit must succeed on valid panel");
    let forecasts = model.predict(4);
    assert_eq!(forecasts.len(), 2, "must produce one forecast per series");
}

// ─────────────────────────────────────────────────────────────────────────────
// GlobalAutoETS — V-01 auto-wrapper guard (CR-01 regression tests)
//
// Without the up-front guard added by CR-01, NaN input causes every candidate
// spec's inner GlobalETS::fit to return Err (V-01 fires), the loop swallows
// all errors with `continue`, and GlobalAutoETS::fit unconditionally sets
// self.fitted = true and returns Ok(()) — emitting silent zero forecasts.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn global_auto_ets_nan_guard() {
    // Regression for CR-01: NaN in any series must propagate as Err, not silent Ok.
    let series = vec![
        vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0],
        vec![20.0, f64::NAN, 21.0, 23.0, 22.0, 24.0, 23.0, 25.0],
    ];
    let mut model = GlobalAutoETS::new(1, ModelPool::Reduced);
    let err = model
        .fit(&series)
        .expect_err("fit must fail on NaN input (CR-01)");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_auto_ets_inf_guard() {
    // Regression for CR-01: Inf must also be caught up-front.
    let series = vec![vec![
        10.0,
        f64::INFINITY,
        11.0,
        13.0,
        12.0,
        14.0,
        13.0,
        15.0,
    ]];
    let mut model = GlobalAutoETS::new(1, ModelPool::Reduced);
    let err = model
        .fit(&series)
        .expect_err("fit must fail on Inf input (CR-01)");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_auto_ets_valid_panel_guard_does_not_fire() {
    // Guard must not fire on valid finite input — confirm happy path still works.
    let series = vec![
        vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0],
        vec![20.0, 22.0, 21.0, 23.0, 22.0, 24.0, 23.0, 25.0],
    ];
    let mut model = GlobalAutoETS::new(1, ModelPool::Reduced);
    model
        .fit(&series)
        .expect("fit must succeed on valid finite panel");
    let forecasts = model.predict(4);
    assert_eq!(forecasts.len(), 2, "must produce one forecast per series");
}

// ─────────────────────────────────────────────────────────────────────────────
// GlobalCroston — V-02 guard
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn global_croston_nan_guard() {
    // Intermittent panel where one series contains NaN
    let series = vec![
        vec![0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0],
        vec![
            0.0,
            f64::NAN,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            4.0,
        ],
    ];
    let mut model = GlobalCroston::new();
    let err = model.fit(&series).expect_err("fit must fail on NaN input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_croston_inf_guard() {
    // Intermittent panel where one series contains Inf
    let series = vec![vec![
        0.0,
        f64::INFINITY,
        3.0,
        0.0,
        0.0,
        2.0,
        0.0,
        4.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]];
    let mut model = GlobalCroston::new();
    let err = model.fit(&series).expect_err("fit must fail on Inf input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_croston_valid_panel_guard_does_not_fire() {
    // All-finite intermittent panel must fit successfully
    let series = vec![
        vec![0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0],
    ];
    let mut model = GlobalCroston::new();
    model.fit(&series).expect("fit must succeed on valid panel");
    let forecasts = model.predict(4);
    assert_eq!(forecasts.len(), 2, "must produce one forecast per series");
}
