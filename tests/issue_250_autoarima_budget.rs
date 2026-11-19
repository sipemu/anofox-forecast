//! Issue #250: callers can bound the cost of one seasonal `AutoARIMA` fit.
//!
//! Long seasonal periods (m = 30 on 550 weekly points) take over a minute
//! per fit, almost all of it in the exact-likelihood refits after the CSS
//! search. `with_time_budget` stops the search and the refits once the
//! budget is spent and returns the best model found so far;
//! `with_max_models` is R's `nmodels` cap on the stepwise search.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::time::Instant;

const DATA: &str = include_str!("data/m4_weekly_w118_last550.csv");

fn weekly_series() -> TimeSeries {
    let y: Vec<f64> = DATA.lines().map(|l| l.trim().parse().unwrap()).collect();
    let start = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let times = (0..y.len())
        .map(|i| start + Duration::days(7 * i as i64))
        .collect();
    TimeSeries::univariate(times, y).unwrap()
}

#[test]
fn defaults_match_r() {
    let config = AutoARIMAConfig::default();
    assert_eq!(config.max_models, 94);
    assert_eq!(config.time_budget, None);
}

#[test]
fn time_budget_bounds_long_seasonal_fit() {
    let budget = std::time::Duration::from_secs(2);
    let config = AutoARIMAConfig::default()
        .with_seasonal_period(30)
        .with_time_budget(budget);
    let mut model = AutoARIMA::with_config(config);
    let t = Instant::now();
    model.fit(&weekly_series()).unwrap();
    let elapsed = t.elapsed();
    // Unbudgeted this fit takes over a minute (release build). Allow for
    // one optimiser iteration plus the final CSS fit past the budget.
    assert!(
        elapsed < budget + std::time::Duration::from_secs(10),
        "fit took {elapsed:?} with a {budget:?} budget"
    );
    let order = model.selected_full_order().unwrap();
    assert_eq!(order.s, 30);
    let fc = model.predict(12).unwrap();
    assert!(fc.primary().iter().all(|v| v.is_finite()));
}

#[test]
fn max_models_caps_stepwise_search() {
    let config = AutoARIMAConfig::default()
        .with_seasonal_period(30)
        .with_max_models(6)
        .with_time_budget(std::time::Duration::from_secs(5));
    let mut model = AutoARIMA::with_config(config);
    model.fit(&weekly_series()).unwrap();
    assert!(
        model.model_scores().len() <= 6,
        "visited {} models with max_models = 6",
        model.model_scores().len()
    );
}
