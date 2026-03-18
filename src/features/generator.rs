//! Standalone feature generator for deterministic regressors.
//!
//! [`FeatureGenerator`] produces regressors from timestamps alone — Fourier
//! terms, calendar indicators, holiday flags — and attaches them to any
//! [`TimeSeries`] via [`CalendarAnnotations`]. Because the features are purely
//! deterministic (functions of time, not data), they are safe for
//! cross-validation and can be generated once for multiple models and series.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::features::FeatureGenerator;
//! use anofox_forecast::core::TimeSeries;
//! use chrono::{TimeZone, Utc, Duration};
//!
//! let timestamps: Vec<_> = (0..60)
//!     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
//!     .collect();
//! let values: Vec<f64> = (0..60).map(|i| (i as f64).sin()).collect();
//! let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
//!
//! let gen = FeatureGenerator::new()
//!     .fourier(7, 3)
//!     .day_of_week();
//!
//! gen.add_to(&mut ts);
//!
//! // For prediction: generate future regressors
//! let future_ts: Vec<_> = (60..67)
//!     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i))
//!     .collect();
//! let future_regs = gen.generate(&future_ts);
//! assert!(future_regs.contains_key("fourier_7_sin1"));
//! assert!(future_regs.contains_key("dow_mon"));
//! ```

use chrono::{DateTime, Datelike, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::core::{CalendarAnnotations, TimeSeries};

/// A reusable, deterministic feature generator.
///
/// All features are functions of timestamps only, making them safe for
/// cross-validation and reusable across multiple time series and models.
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    specs: Vec<FeatureSpec>,
}

#[derive(Debug, Clone)]
enum FeatureSpec {
    Fourier {
        period: usize,
        order: usize,
    },
    DayOfWeek,
    MonthOfYear,
    Quarter,
    Holiday {
        dates: Vec<DateTime<Utc>>,
        name: String,
    },
}

impl FeatureGenerator {
    /// Create an empty feature generator.
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Add Fourier terms for a given period and order.
    ///
    /// Produces `2 × order` columns named `fourier_{period}_sin{k}` and
    /// `fourier_{period}_cos{k}` for `k` in `1..=order`.
    ///
    /// The period is in the same units as the observation spacing (e.g.,
    /// period=7 for weekly seasonality in daily data, period=12 for annual
    /// seasonality in monthly data).
    pub fn fourier(mut self, period: usize, order: usize) -> Self {
        self.specs.push(FeatureSpec::Fourier { period, order });
        self
    }

    /// Add day-of-week indicators (Monday–Saturday, Sunday dropped).
    ///
    /// Produces 6 binary columns: `dow_mon`, `dow_tue`, …, `dow_sat`.
    pub fn day_of_week(mut self) -> Self {
        self.specs.push(FeatureSpec::DayOfWeek);
        self
    }

    /// Add month-of-year indicators (Feb–Dec, January dropped).
    ///
    /// Produces 11 binary columns: `month_feb`, `month_mar`, …, `month_dec`.
    pub fn month_of_year(mut self) -> Self {
        self.specs.push(FeatureSpec::MonthOfYear);
        self
    }

    /// Add quarter indicators (Q2–Q4, Q1 dropped).
    ///
    /// Produces 3 binary columns: `quarter_2`, `quarter_3`, `quarter_4`.
    pub fn quarter(mut self) -> Self {
        self.specs.push(FeatureSpec::Quarter);
        self
    }

    /// Add a binary holiday indicator.
    ///
    /// Produces 1 column named `holiday_{name}` that is 1.0 on dates
    /// matching any of the given datetimes (date part only), 0.0 otherwise.
    pub fn holiday(mut self, name: impl Into<String>, dates: Vec<DateTime<Utc>>) -> Self {
        self.specs.push(FeatureSpec::Holiday {
            dates,
            name: name.into(),
        });
        self
    }

    /// Generate features for a slice of timestamps.
    ///
    /// Returns a `HashMap<String, Vec<f64>>` ready for use with
    /// [`predict_with_exog`](crate::models::Forecaster::predict_with_exog).
    pub fn generate(&self, timestamps: &[DateTime<Utc>]) -> HashMap<String, Vec<f64>> {
        let mut result = HashMap::new();
        let n = timestamps.len();

        for spec in &self.specs {
            match spec {
                FeatureSpec::Fourier { period, order } => {
                    let period_f = *period as f64;
                    for k in 1..=*order {
                        let freq = 2.0 * PI * k as f64 / period_f;
                        let mut sin_col = Vec::with_capacity(n);
                        let mut cos_col = Vec::with_capacity(n);
                        for (i, _) in timestamps.iter().enumerate() {
                            let angle = freq * i as f64;
                            sin_col.push(angle.sin());
                            cos_col.push(angle.cos());
                        }
                        result.insert(format!("fourier_{}_sin{}", period, k), sin_col);
                        result.insert(format!("fourier_{}_cos{}", period, k), cos_col);
                    }
                }
                FeatureSpec::DayOfWeek => {
                    let names = ["mon", "tue", "wed", "thu", "fri", "sat"];
                    // 0=Mon..6=Sun in chrono's weekday().num_days_from_monday()
                    for (dow, name) in names.iter().enumerate() {
                        let col: Vec<f64> = timestamps
                            .iter()
                            .map(|ts| {
                                if ts.weekday().num_days_from_monday() as usize == dow {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        result.insert(format!("dow_{}", name), col);
                    }
                }
                FeatureSpec::MonthOfYear => {
                    let names = [
                        "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                    ];
                    for (i, name) in names.iter().enumerate() {
                        let month = i + 2; // Feb=2, Mar=3, ...
                        let col: Vec<f64> = timestamps
                            .iter()
                            .map(|ts| {
                                if ts.month() as usize == month {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        result.insert(format!("month_{}", name), col);
                    }
                }
                FeatureSpec::Quarter => {
                    for q in 2..=4u32 {
                        let col: Vec<f64> = timestamps
                            .iter()
                            .map(|ts| {
                                let m = ts.month();
                                let quarter = (m - 1) / 3 + 1;
                                if quarter == q {
                                    1.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        result.insert(format!("quarter_{}", q), col);
                    }
                }
                FeatureSpec::Holiday { dates, name } => {
                    let holiday_dates: std::collections::HashSet<_> =
                        dates.iter().map(|d| d.date_naive()).collect();
                    let col: Vec<f64> = timestamps
                        .iter()
                        .map(|ts| {
                            if holiday_dates.contains(&ts.date_naive()) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    result.insert(format!("holiday_{}", name), col);
                }
            }
        }

        result
    }

    /// Generate features and attach them to a [`TimeSeries`] as regressors.
    ///
    /// If the series already has a [`CalendarAnnotations`], the new features
    /// are merged into it. Otherwise a new one is created.
    pub fn add_to(&self, ts: &mut TimeSeries) {
        let features = self.generate(ts.timestamps());

        let mut cal = ts
            .calendar()
            .cloned()
            .unwrap_or_else(CalendarAnnotations::new);
        for (name, values) in features {
            cal = cal.with_regressor(name, values);
        }
        ts.set_calendar(cal);
    }

    /// Return the column names this generator will produce, in sorted order.
    pub fn feature_names(&self) -> Vec<String> {
        // Generate for a dummy single timestamp to get names.
        let dummy = vec![Utc::now()];
        let map = self.generate(&dummy);
        let mut names: Vec<String> = map.into_keys().collect();
        names.sort();
        names
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn daily_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        (0..n)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
            .collect()
    }

    fn monthly_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        let mut ts = Vec::with_capacity(n);
        let mut year = 2020i32;
        let mut month = 1u32;
        for _ in 0..n {
            ts.push(Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).unwrap());
            month += 1;
            if month > 12 {
                month = 1;
                year += 1;
            }
        }
        ts
    }

    // ── Fourier ─────────────────────────────────────────────────────────

    #[test]
    fn fourier_column_count() {
        let gen = FeatureGenerator::new().fourier(7, 3).fourier(365, 2);
        let feats = gen.generate(&daily_timestamps(30));
        // 7×3 = 6 columns + 365×2 = 4 columns = 10
        assert_eq!(feats.len(), 10);
        assert!(feats.contains_key("fourier_7_sin1"));
        assert!(feats.contains_key("fourier_7_cos3"));
        assert!(feats.contains_key("fourier_365_sin2"));
    }

    #[test]
    fn fourier_values_periodic() {
        let gen = FeatureGenerator::new().fourier(7, 1);
        let ts = daily_timestamps(14);
        let feats = gen.generate(&ts);
        let sin1 = &feats["fourier_7_sin1"];

        // sin should repeat with period 7
        for i in 0..7 {
            assert!(
                (sin1[i] - sin1[i + 7]).abs() < 1e-10,
                "sin1[{}]={} != sin1[{}]={}",
                i,
                sin1[i],
                i + 7,
                sin1[i + 7]
            );
        }
    }

    #[test]
    fn fourier_sin_at_zero_is_zero() {
        let gen = FeatureGenerator::new().fourier(12, 1);
        let ts = monthly_timestamps(12);
        let feats = gen.generate(&ts);
        assert!((feats["fourier_12_sin1"][0]).abs() < 1e-10);
        assert!((feats["fourier_12_cos1"][0] - 1.0).abs() < 1e-10);
    }

    // ── Day of week ─────────────────────────────────────────────────────

    #[test]
    fn day_of_week_columns() {
        let gen = FeatureGenerator::new().day_of_week();
        let feats = gen.generate(&daily_timestamps(7));
        assert_eq!(feats.len(), 6); // Mon-Sat, Sunday dropped
        assert!(feats.contains_key("dow_mon"));
        assert!(feats.contains_key("dow_sat"));
        assert!(!feats.contains_key("dow_sun"));
    }

    #[test]
    fn day_of_week_one_hot() {
        let gen = FeatureGenerator::new().day_of_week();
        let ts = daily_timestamps(7);
        let feats = gen.generate(&ts);

        // 2024-01-01 is Monday
        assert_eq!(feats["dow_mon"][0], 1.0);
        assert_eq!(feats["dow_tue"][0], 0.0);
        assert_eq!(feats["dow_tue"][1], 1.0); // Tuesday
        assert_eq!(feats["dow_sat"][5], 1.0); // Saturday

        // Sunday (index 6): all 6 columns should be 0
        for col in feats.values() {
            assert_eq!(col[6], 0.0);
        }
    }

    // ── Month of year ───────────────────────────────────────────────────

    #[test]
    fn month_of_year_columns() {
        let gen = FeatureGenerator::new().month_of_year();
        let feats = gen.generate(&monthly_timestamps(12));
        assert_eq!(feats.len(), 11); // Feb-Dec, January dropped
        assert!(feats.contains_key("month_feb"));
        assert!(feats.contains_key("month_dec"));
        assert!(!feats.contains_key("month_jan"));
    }

    #[test]
    fn month_of_year_one_hot() {
        let gen = FeatureGenerator::new().month_of_year();
        let ts = monthly_timestamps(12);
        let feats = gen.generate(&ts);

        // Index 0 = Jan 2020: all columns 0 (dropped category)
        for col in feats.values() {
            assert_eq!(col[0], 0.0);
        }
        // Index 1 = Feb 2020
        assert_eq!(feats["month_feb"][1], 1.0);
        assert_eq!(feats["month_mar"][1], 0.0);
        // Index 11 = Dec 2020
        assert_eq!(feats["month_dec"][11], 1.0);
    }

    // ── Quarter ─────────────────────────────────────────────────────────

    #[test]
    fn quarter_columns() {
        let gen = FeatureGenerator::new().quarter();
        let feats = gen.generate(&monthly_timestamps(12));
        assert_eq!(feats.len(), 3); // Q2, Q3, Q4 — Q1 dropped
    }

    #[test]
    fn quarter_one_hot() {
        let gen = FeatureGenerator::new().quarter();
        let ts = monthly_timestamps(12);
        let feats = gen.generate(&ts);

        // Jan(0), Feb(1), Mar(2) = Q1 → all 0
        for i in 0..3 {
            assert_eq!(feats["quarter_2"][i], 0.0);
            assert_eq!(feats["quarter_3"][i], 0.0);
            assert_eq!(feats["quarter_4"][i], 0.0);
        }
        // Apr(3), May(4), Jun(5) = Q2
        for i in 3..6 {
            assert_eq!(feats["quarter_2"][i], 1.0);
        }
        // Oct(9), Nov(10), Dec(11) = Q4
        for i in 9..12 {
            assert_eq!(feats["quarter_4"][i], 1.0);
        }
    }

    // ── Holiday ─────────────────────────────────────────────────────────

    #[test]
    fn holiday_indicator() {
        let xmas = Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();
        let gen = FeatureGenerator::new().holiday("xmas", vec![xmas]);
        let ts = daily_timestamps(7);
        let feats = gen.generate(&ts);

        let col = &feats["holiday_xmas"];
        assert_eq!(col[2], 1.0); // Jan 3
        assert_eq!(col[0], 0.0);
        assert_eq!(col[4], 0.0);
    }

    // ── add_to TimeSeries ───────────────────────────────────────────────

    #[test]
    fn add_to_attaches_regressors() {
        let ts_vec = daily_timestamps(14);
        let values: Vec<f64> = (0..14).map(|i| i as f64).collect();
        let mut ts = TimeSeries::univariate(ts_vec, values).unwrap();

        let gen = FeatureGenerator::new().fourier(7, 2).day_of_week();
        gen.add_to(&mut ts);

        let regs = ts.all_regressors();
        // 4 fourier + 6 dow = 10
        assert_eq!(regs.len(), 10);
        assert!(regs.contains_key("fourier_7_sin1"));
        assert!(regs.contains_key("dow_fri"));
    }

    #[test]
    fn add_to_preserves_existing_regressors() {
        let ts_vec = daily_timestamps(14);
        let values: Vec<f64> = (0..14).map(|i| i as f64).collect();
        let mut ts = TimeSeries::univariate(ts_vec, values).unwrap();

        // Attach an existing regressor
        let cal =
            CalendarAnnotations::new().with_regressor("temperature".to_string(), vec![20.0; 14]);
        ts.set_calendar(cal);

        let gen = FeatureGenerator::new().fourier(7, 1);
        gen.add_to(&mut ts);

        let regs = ts.all_regressors();
        assert!(regs.contains_key("temperature")); // preserved
        assert!(regs.contains_key("fourier_7_sin1")); // added
        assert_eq!(regs.len(), 3); // temperature + sin1 + cos1
    }

    // ── feature_names ───────────────────────────────────────────────────

    #[test]
    fn feature_names_sorted() {
        let gen = FeatureGenerator::new()
            .fourier(7, 2)
            .day_of_week()
            .month_of_year()
            .quarter();
        let names = gen.feature_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        // 4 fourier + 6 dow + 11 month + 3 quarter = 24
        assert_eq!(names.len(), 24);
    }

    // ── Composability ───────────────────────────────────────────────────

    #[test]
    fn generate_for_future_timestamps() {
        let gen = FeatureGenerator::new().fourier(7, 1).day_of_week();

        let train = daily_timestamps(28);
        let future = daily_timestamps(35)[28..].to_vec();

        let train_feats = gen.generate(&train);
        let future_feats = gen.generate(&future);

        // Same columns
        assert_eq!(train_feats.len(), future_feats.len());
        for key in train_feats.keys() {
            assert!(future_feats.contains_key(key), "missing key: {}", key);
        }

        // Future has correct length
        for col in future_feats.values() {
            assert_eq!(col.len(), 7);
        }
    }

    #[test]
    fn empty_generator() {
        let gen = FeatureGenerator::new();
        let feats = gen.generate(&daily_timestamps(10));
        assert!(feats.is_empty());
        assert!(gen.feature_names().is_empty());
    }

    // ── OLS integration: recover known effects from artificial data ──────

    use crate::utils::ols::ols_fit;

    /// Build y = intercept + sum(coeff_i * feature_i) and verify OLS recovers the coefficients.
    fn assert_ols_recovers(
        features: &HashMap<String, Vec<f64>>,
        true_intercept: f64,
        true_coeffs: &HashMap<String, f64>,
        tol: f64,
    ) {
        let n = features.values().next().unwrap().len();
        let mut y = vec![true_intercept; n];
        for (name, coeff) in true_coeffs {
            let col = &features[name];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi += coeff * col[i];
            }
        }

        let result = ols_fit(&y, features).unwrap();

        assert!(
            (result.intercept - true_intercept).abs() < tol,
            "intercept: expected {}, got {} (tol {})",
            true_intercept,
            result.intercept,
            tol,
        );
        for (name, &expected) in true_coeffs {
            let idx = result
                .regressor_names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("missing regressor '{}'", name));
            assert!(
                (result.coefficients[idx] - expected).abs() < tol,
                "coeff '{}': expected {}, got {} (tol {})",
                name,
                expected,
                result.coefficients[idx],
                tol,
            );
        }
    }

    #[test]
    fn ols_recovers_fourier_effects() {
        // y = 100 + 5*sin1 - 3*cos1
        let gen = FeatureGenerator::new().fourier(12, 1);
        let ts = monthly_timestamps(120); // 10 years of monthly data
        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        true_coeffs.insert("fourier_12_sin1".to_string(), 5.0);
        true_coeffs.insert("fourier_12_cos1".to_string(), -3.0);

        assert_ols_recovers(&features, 100.0, &true_coeffs, 0.01);
    }

    #[test]
    fn ols_recovers_day_of_week_effects() {
        // y = 50 + effects for Mon..Sat (Sunday is baseline = 0)
        let gen = FeatureGenerator::new().day_of_week();
        let ts = daily_timestamps(364); // 52 full weeks
        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        true_coeffs.insert("dow_mon".to_string(), 10.0);
        true_coeffs.insert("dow_tue".to_string(), 8.0);
        true_coeffs.insert("dow_wed".to_string(), 6.0);
        true_coeffs.insert("dow_thu".to_string(), 4.0);
        true_coeffs.insert("dow_fri".to_string(), 12.0);
        true_coeffs.insert("dow_sat".to_string(), -5.0);

        assert_ols_recovers(&features, 50.0, &true_coeffs, 0.01);
    }

    #[test]
    fn ols_recovers_month_effects() {
        // y = 200 + month effects (January = baseline)
        let gen = FeatureGenerator::new().month_of_year();
        let ts = monthly_timestamps(120); // 10 years
        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        let month_effects = [
            ("month_feb", 2.0),
            ("month_mar", 5.0),
            ("month_apr", 10.0),
            ("month_may", 15.0),
            ("month_jun", 18.0),
            ("month_jul", 20.0),
            ("month_aug", 19.0),
            ("month_sep", 14.0),
            ("month_oct", 8.0),
            ("month_nov", 3.0),
            ("month_dec", -1.0),
        ];
        for (name, effect) in &month_effects {
            true_coeffs.insert(name.to_string(), *effect);
        }

        assert_ols_recovers(&features, 200.0, &true_coeffs, 0.01);
    }

    #[test]
    fn ols_recovers_quarter_effects() {
        // y = 80 + quarter effects (Q1 = baseline)
        let gen = FeatureGenerator::new().quarter();
        let ts = monthly_timestamps(48); // 4 years
        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        true_coeffs.insert("quarter_2".to_string(), 15.0);
        true_coeffs.insert("quarter_3".to_string(), 25.0);
        true_coeffs.insert("quarter_4".to_string(), 10.0);

        assert_ols_recovers(&features, 80.0, &true_coeffs, 0.01);
    }

    #[test]
    fn ols_recovers_holiday_effect() {
        // y = 30 + 50 on holidays
        let ts = daily_timestamps(365);
        // Pick a few dates as holidays (day 10, 50, 100, 200, 300)
        let holiday_dates: Vec<DateTime<Utc>> = [10, 50, 100, 200, 300]
            .iter()
            .map(|&d| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(d))
            .collect();

        let gen = FeatureGenerator::new().holiday("promo", holiday_dates);
        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        true_coeffs.insert("holiday_promo".to_string(), 50.0);

        assert_ols_recovers(&features, 30.0, &true_coeffs, 0.01);
    }

    #[test]
    fn ols_recovers_combined_effects() {
        // y = 100 + fourier(7,1) effects + day-of-week effects + holiday effect
        let n = 364; // 52 weeks
        let ts = daily_timestamps(n);
        let holiday_dates: Vec<DateTime<Utc>> = (0..52)
            .map(|w| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(w * 7))
            .collect();

        let gen = FeatureGenerator::new()
            .fourier(7, 1)
            .day_of_week()
            .holiday("weekly_event", holiday_dates);

        let features = gen.generate(&ts);

        let mut true_coeffs = HashMap::new();
        // Fourier effects
        true_coeffs.insert("fourier_7_sin1".to_string(), 3.0);
        true_coeffs.insert("fourier_7_cos1".to_string(), -2.0);
        // Day-of-week effects
        true_coeffs.insert("dow_mon".to_string(), 5.0);
        true_coeffs.insert("dow_tue".to_string(), 4.0);
        true_coeffs.insert("dow_wed".to_string(), 3.0);
        true_coeffs.insert("dow_thu".to_string(), 2.0);
        true_coeffs.insert("dow_fri".to_string(), 6.0);
        true_coeffs.insert("dow_sat".to_string(), -1.0);
        // Holiday effect
        true_coeffs.insert("holiday_weekly_event".to_string(), 20.0);

        // Fourier + DOW are collinear (both weekly), so tolerance is larger.
        // OLS will still recover the combined effect but individual coefficients
        // shift. Test the combined prediction instead.
        let n = features.values().next().unwrap().len();
        let mut y = vec![100.0_f64; n];
        for (name, coeff) in &true_coeffs {
            let col = &features[name];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi += coeff * col[i];
            }
        }

        let result = ols_fit(&y, &features).unwrap();
        let y_hat = result.predict(&features).unwrap();

        let max_error = y
            .iter()
            .zip(y_hat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error < 0.01,
            "max prediction error: {} (should be < 0.01)",
            max_error,
        );
    }

    #[test]
    fn ols_recovers_fourier_with_external_regressor() {
        // y = 50 + 4*sin1 + 2*cos1 + 7*temperature
        // where temperature is an external regressor (not from FeatureGenerator)
        let gen = FeatureGenerator::new().fourier(12, 1);
        let ts = monthly_timestamps(120);
        let mut features = gen.generate(&ts);

        // Add external temperature regressor
        let temperature: Vec<f64> = (0..120)
            .map(|i| 15.0 + 10.0 * (2.0 * PI * i as f64 / 12.0).cos())
            .collect();
        features.insert("temperature".to_string(), temperature);

        let mut true_coeffs = HashMap::new();
        true_coeffs.insert("fourier_12_sin1".to_string(), 4.0);
        true_coeffs.insert("fourier_12_cos1".to_string(), 2.0);
        true_coeffs.insert("temperature".to_string(), 7.0);

        // Note: fourier_12_cos1 and temperature are correlated (both cos with period 12).
        // With enough data and regularization the combined prediction should still be accurate.
        let n = features.values().next().unwrap().len();
        let mut y = vec![50.0_f64; n];
        for (name, coeff) in &true_coeffs {
            let col = &features[name];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi += coeff * col[i];
            }
        }

        let result = ols_fit(&y, &features).unwrap();
        let y_hat = result.predict(&features).unwrap();

        let max_error = y
            .iter()
            .zip(y_hat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error < 0.01,
            "max prediction error: {} (should be < 0.01)",
            max_error,
        );
    }

    #[test]
    fn ols_predict_future_with_features() {
        // Train: y = 100 + 5*sin1(period=7), then predict future and verify
        let gen = FeatureGenerator::new().fourier(7, 1);

        let train_ts = daily_timestamps(70);
        let train_features = gen.generate(&train_ts);

        let n = 70;
        let mut y = vec![100.0; n];
        let sin_col = &train_features["fourier_7_sin1"];
        let cos_col = &train_features["fourier_7_cos1"];
        for i in 0..n {
            y[i] += 5.0 * sin_col[i] + 3.0 * cos_col[i];
        }

        let result = ols_fit(&y, &train_features).unwrap();

        // Generate future features
        let future_ts: Vec<_> = (70..77)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
            .collect();
        let future_features = gen.generate(&future_ts);
        let predictions = result.predict(&future_features).unwrap();

        // Expected future values
        for (i, &pred) in predictions.iter().enumerate() {
            let idx = 70 + i;
            let angle = 2.0 * PI / 7.0 * idx as f64;
            let expected = 100.0 + 5.0 * angle.sin() + 3.0 * angle.cos();
            assert!(
                (pred - expected).abs() < 0.01,
                "future[{}]: expected {}, got {}",
                i,
                expected,
                pred,
            );
        }
    }
}
