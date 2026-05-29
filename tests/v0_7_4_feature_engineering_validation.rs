//! End-to-end validation for v0.7.4 feature-engineering additions.
//!
//! Each unit test in the crate covers a single new feature; this
//! integration test wires several of them together on one synthetic
//! series to confirm they compose without breaking.
//!
//! Features exercised:
//! - Rolling-statistic extensions (Quantile, ZScore, Slope, Rank)
//! - EventDistanceFeature
//! - Derived exog: Lag, Polynomial, Interaction, Categorical (OneHot)
//! - Yeo-Johnson transform (round-trip)
//! - Cross-series panel aggregations
//! - Multicollinearity diagnostic

use std::collections::HashMap;

use anofox_forecast::core::{CalendarAnnotations, TimeSeries, TimeSeriesBuilder};
use anofox_forecast::features::panel::{panel_aggregate, panel_mean, PanelAggregator};
use anofox_forecast::models::regression::{
    CategoricalStrategy, EventDistanceMode, RegressionFeatures, RegressionForecaster,
};
use anofox_forecast::models::Forecaster;
use anofox_forecast::transform::{InverseMode, Transform, YeoJohnsonTransform};
use anofox_forecast::validation::{multicollinearity_report, variance_inflation_factors};
use approx::assert_relative_eq;
use chrono::{Duration, TimeZone, Utc};

fn timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

/// Build a series with a richer-than-usual structure so the model has to
/// use the new feature kinds to fit. The series is deterministic (no
/// randomness) so we can make exact-ish assertions on a holdout.
fn synth_series(n: usize) -> (TimeSeries, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Three exogenous signals:
    //   price: smooth ramp + small bump
    //   promo: 0/1 indicator (every 7th day = 1)
    //   region: 0/1/2 cycling category code
    let price: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.02).collect();
    let promo: Vec<f64> = (0..n).map(|i| if i % 7 == 0 { 1.0 } else { 0.0 }).collect();
    let region: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();

    // Target: deterministic combination of region OneHot baseline,
    // a polynomial-of-price term, and a promo×price interaction. We
    // deliberately omit a standalone `promo` term so the data is fully
    // spanned by {intercept, region_eq_1, region_eq_2, price_pow2,
    // promo_x_price} — the columns the model will actually build.
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let r = region[i] as i64;
            let region_offset = match r {
                0 => 1.0,
                1 => 5.0,
                _ => 9.0,
            };
            // Polynomial-of-price effect: 0.4 * price^2
            let poly = 0.4 * price[i] * price[i];
            // Interaction: 0.7 * promo * price
            let inter = 0.7 * promo[i] * price[i];
            region_offset + poly + inter
        })
        .collect();

    let cal = CalendarAnnotations::new()
        .with_regressor("price".to_string(), price.clone())
        .with_regressor("promo".to_string(), promo.clone())
        .with_regressor("region".to_string(), region.clone());

    let ts = TimeSeriesBuilder::new()
        .timestamps(timestamps(n))
        .values(target.clone())
        .calendar(cal)
        .build()
        .unwrap();

    (ts, price, promo, region)
}

#[test]
fn rolling_extensions_zscore_slope_quantile_rank_fit_predict_works() {
    // Strong upward AR(1)-ish trend so Slope picks up signal.
    let n = 80;
    let mut values = vec![0.0_f64; n];
    values[0] = 1.0;
    for i in 1..n {
        let bump = ((i * 11 + 5) % 17) as f64 * 0.01 - 0.085;
        values[i] = 0.95 * values[i - 1] + 0.2 + bump;
    }
    let ts = TimeSeries::univariate(timestamps(n), values).unwrap();

    let mut model = RegressionForecaster::ols(
        RegressionFeatures::new()
            .no_trend()
            .no_exog()
            .with_rolling_zscore(10)
            .unwrap()
            .with_rolling_slope(10)
            .unwrap()
            .with_rolling_quantile(10, 0.75)
            .unwrap()
            .with_rolling_rank(10)
            .unwrap(),
    );
    model.fit(&ts).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
    for &v in forecast.primary() {
        assert!(v.is_finite(), "forecasts should be finite, got {}", v);
    }

    // Inspect column names so we know the new RollingStatKinds materialize.
    let names = model.features().feature_names(&[]);
    assert!(names.iter().any(|n| n.contains("zscore")));
    assert!(names.iter().any(|n| n.contains("slope")));
    assert!(names.iter().any(|n| n.contains("q0.75")));
    assert!(names.iter().any(|n| n.contains("rank")));
}

#[test]
fn event_distance_plus_categorical_plus_polynomial_compose() {
    let n = 90;
    let (ts, price, promo, _region) = synth_series(n);

    let model = RegressionFeatures::new()
        .no_trend()
        .no_exog()
        // Categorical region with three pre-declared codes. drop_first=false
        // so all three dummy columns appear — they span the region constants
        // without relying on an implicit intercept.
        .with_categorical(
            "region",
            vec![0, 1, 2],
            CategoricalStrategy::OneHot { drop_first: false },
        )
        .unwrap()
        // Polynomial of price to capture the 0.4 · p² term.
        .with_exog_polynomial("price", 2)
        .unwrap()
        // Interaction to capture the 0.7 · promo · price term.
        .with_exog_interaction("promo", "price")
        // Event distance from "promotion days" — every 7 steps. We use
        // `Since` only, not `Both`, because steps_since + steps_until is a
        // constant (= 7 here) and three OneHot dummies also sum to 1 —
        // including both event columns produces an unidentified rank-2
        // collinearity that the OLS solver collapses on.
        .with_event_distance(
            (0..(n + 14)).filter(|i| i % 7 == 0).collect(),
            EventDistanceMode::Since,
        );

    let mut forecaster = RegressionForecaster::ols(model);
    forecaster.fit(&ts).unwrap();
    let r2 = forecaster.r_squared().unwrap_or(0.0);
    assert!(
        r2 > 0.95,
        "expected R² > 0.95 with composed features, got {}",
        r2
    );

    // Predict 14 steps ahead using the same generators.
    let mut future_regs = HashMap::new();
    future_regs.insert(
        "price".to_string(),
        (n..n + 14).map(|i| 1.0 + (i as f64) * 0.02).collect(),
    );
    future_regs.insert(
        "promo".to_string(),
        (n..n + 14)
            .map(|i| if i % 7 == 0 { 1.0 } else { 0.0 })
            .collect(),
    );
    future_regs.insert(
        "region".to_string(),
        (n..n + 14).map(|i| (i % 3) as f64).collect(),
    );

    let forecast = forecaster.predict_with_exog(14, &future_regs).unwrap();
    let preds = forecast.primary();
    assert_eq!(preds.len(), 14);

    // Spot-check first step: i = 90 → region=0 (1.0), promo=0, price=2.8 → ≈ 1 + 0.4·2.8² ≈ 1 + 3.136 = 4.136
    let p0 = 1.0 + (n as f64) * 0.02;
    let expected_step0 = 1.0 + 0.4 * p0 * p0;
    assert_relative_eq!(preds[0], expected_step0, epsilon = 0.05);

    // Reference the generated synth inputs to silence unused-warning.
    assert_eq!(price.len(), n);
    assert_eq!(promo.len(), n);
}

#[test]
fn exog_lags_round_trip_with_future_regressors() {
    let n = 60;
    let rain: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| if i >= 3 { 2.0 * rain[i - 3] } else { 0.0 })
        .collect();

    let cal = CalendarAnnotations::new().with_regressor("rain".to_string(), rain.clone());
    let ts = TimeSeriesBuilder::new()
        .timestamps(timestamps(n))
        .values(y)
        .calendar(cal)
        .build()
        .unwrap();

    let mut model = RegressionForecaster::ols(
        RegressionFeatures::new()
            .no_trend()
            .no_exog()
            .with_exog_lags("rain", &[3]),
    );
    model.fit(&ts).unwrap();

    let future_rain: Vec<f64> = (n..n + 5).map(|i| ((i as f64) * 0.2).sin()).collect();
    let mut fr = HashMap::new();
    fr.insert("rain".to_string(), future_rain.clone());
    let forecast = model.predict_with_exog(5, &fr).unwrap();
    // Step h=3 should ≈ 2 * future_rain[0].
    assert_relative_eq!(forecast.primary()[3], 2.0 * future_rain[0], epsilon = 0.05);
}

#[test]
fn yeo_johnson_transforms_zero_inclusive_series_and_round_trips() {
    // BoxCox would error on this; Yeo-Johnson handles it.
    let values: Vec<f64> = (0..40)
        .map(|i| if i % 4 == 0 { 0.0 } else { (i as f64).sqrt() })
        .collect();

    let mut t = YeoJohnsonTransform::auto();
    let transformed = t.fit_transform(&values).unwrap();
    assert_eq!(transformed.len(), values.len());

    let recovered = t.inverse(&transformed, InverseMode::Predict).unwrap();
    for (a, b) in values.iter().zip(recovered.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-8);
    }

    assert!(t.fitted_lambda().is_some());
}

#[test]
fn panel_aggregations_produce_expected_shapes_and_values() {
    let panel: Vec<Vec<f64>> = vec![
        (0..10).map(|i| i as f64).collect(),
        (0..10).map(|i| 10.0 + i as f64).collect(),
        (0..10).map(|i| 100.0 + i as f64).collect(),
    ];

    // Common mean: each output row equals the cross-section mean, repeated.
    let common = panel_mean(&panel, false).unwrap();
    assert_eq!(common.len(), 3);
    assert_eq!(common[0], common[1]);
    assert_eq!(common[0], common[2]);
    assert_relative_eq!(common[0][0], (0.0 + 10.0 + 100.0) / 3.0, epsilon = 1e-12);

    // LOO mean for series 0 at t=0: (10 + 100) / 2 = 55.
    let loo = panel_mean(&panel, true).unwrap();
    assert_relative_eq!(loo[0][0], 55.0, epsilon = 1e-12);

    // Rank: series 0 < series 1 < series 2 everywhere.
    let ranks = panel_aggregate(&panel, PanelAggregator::Rank, false).unwrap();
    assert_relative_eq!(ranks[0][0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(ranks[1][0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(ranks[2][0], 1.0, epsilon = 1e-12);
}

#[test]
fn multicollinearity_detects_redundant_columns() {
    // c0 and c1 are perfectly collinear; c2 is orthogonal-ish (sin/cos).
    let n = 50;
    let c0: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let c1: Vec<f64> = c0.iter().map(|x| 2.0 * x + 1.0).collect();
    let c2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();

    let vifs = variance_inflation_factors(&[c0.clone(), c1.clone(), c2.clone()]).unwrap();
    assert!(vifs[0].is_infinite() || vifs[0] > 1e6);
    assert!(vifs[1].is_infinite() || vifs[1] > 1e6);
    assert!(vifs[2] < 5.0);

    let report =
        multicollinearity_report(&[c0, c1, c2], &["c0".into(), "c1".into(), "c2".into()]).unwrap();
    let failing = report.failing();
    assert!(failing.contains(&"c0") || failing.contains(&"c1"));
    assert!(!failing.contains(&"c2"));
    assert!(report.is_ill_conditioned());
}
