//! Smoke tests for GlobalTheta — closes G-01 (0% coverage void).
//!
//! Exercises: new(), with_theta(), alpha(), fit(), predict(), and both
//! guard paths (V-03 NaN guard, pre-existing empty-input guard).

use anofox_forecast::error::ForecastError;
use anofox_forecast::models::theta::GlobalTheta;

#[test]
fn global_theta_basic_fit_predict() {
    // Panel of two finite series of length 50
    let series: Vec<Vec<f64>> = vec![
        (0..50).map(|i| 10.0 + 0.3 * i as f64).collect(),
        (0..50).map(|i| 20.0 - 0.1 * i as f64).collect(),
    ];
    let mut model = GlobalTheta::new();
    model.fit(&series).expect("fit must succeed on valid data");

    let alpha = model.alpha();
    assert!(
        alpha > 0.0 && alpha < 1.0,
        "alpha must be in (0,1), got {}",
        alpha
    );

    let forecasts = model.predict(10);
    assert_eq!(forecasts.len(), 2, "must produce one forecast per series");
    for fc in &forecasts {
        assert_eq!(fc.len(), 10, "each forecast must have horizon=10 steps");
        for &v in fc {
            assert!(v.is_finite(), "all forecast values must be finite");
        }
    }
}

#[test]
fn global_theta_nan_guard() {
    // Panel where the second series contains NaN (V-03 guard path)
    let series = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.0, f64::NAN, 3.0, 4.0, 5.0],
    ];
    let mut model = GlobalTheta::new();
    let err = model.fit(&series).expect_err("fit must fail on NaN input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_theta_inf_guard() {
    // Single series with an Inf element (V-03 guard path)
    let series = vec![vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0]];
    let mut model = GlobalTheta::new();
    let err = model.fit(&series).expect_err("fit must fail on Inf input");
    assert!(
        matches!(err, ForecastError::InvalidParameter(_)),
        "expected InvalidParameter, got {:?}",
        err
    );
}

#[test]
fn global_theta_empty_input_guard() {
    // Pre-existing empty-slice guard still holds
    let mut model = GlobalTheta::new();
    let err = model.fit(&[]).expect_err("fit must fail on empty slice");
    assert!(
        matches!(err, ForecastError::InsufficientData { .. }),
        "expected InsufficientData, got {:?}",
        err
    );
}

#[test]
fn global_theta_with_theta_constructor() {
    // with_theta() constructor and fit/predict on a single series
    let series: Vec<Vec<f64>> = vec![(0..30).map(|i| i as f64).collect()];
    let mut model = GlobalTheta::with_theta(1.5);
    model.fit(&series).expect("fit must succeed");
    let forecasts = model.predict(5);
    assert_eq!(forecasts.len(), 1);
    assert_eq!(forecasts[0].len(), 5);
}
