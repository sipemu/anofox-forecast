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
    /// Cyclical sin/cos encoding of a time component.
    Cyclical(TimeComponent),
    /// Binary indicator (0/1).
    Binary(BinaryIndicator),
    /// Advanced numeric feature.
    Advanced(AdvancedFeature),
}

/// Time component for cyclical (sin/cos) encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimeComponent {
    /// Month of year (1-12), period=12.
    Month,
    /// Quarter (1-4), period=4.
    Quarter,
    /// Semester (1-2), period=2.
    Semester,
    /// ISO week of year (1-53), period=53.
    WeekOfYear,
    /// Day of week (0-6, Mon=0), period=7.
    DayOfWeek,
    /// Day of month (1-31), period=31.
    DayOfMonth,
    /// Day of year (1-366), period=366.
    DayOfYear,
    /// Hour of day (0-23), period=24.
    Hour,
    /// Minute of hour (0-59), period=60.
    Minute,
    /// Second of minute (0-59), period=60.
    Second,
}

/// Binary (0/1) calendar indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinaryIndicator {
    /// First day of the month.
    MonthStart,
    /// Last day of the month.
    MonthEnd,
    /// First day of the quarter (Jan/Apr/Jul/Oct 1).
    QuarterStart,
    /// Last day of the quarter (Mar/Jun/Sep/Dec last day).
    QuarterEnd,
    /// January 1.
    YearStart,
    /// December 31.
    YearEnd,
    /// Saturday or Sunday.
    Weekend,
}

/// Advanced numeric calendar features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AdvancedFeature {
    /// 1.0 if leap year, 0.0 otherwise.
    LeapYear,
    /// Number of days in the month (28-31).
    DaysInMonth,
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

    /// Add cyclical sin/cos encoding of a time component.
    ///
    /// Produces 2 columns: `{name}_sin` and `{name}_cos` where the angle
    /// is `2*pi*value/period`.
    pub fn cyclical(mut self, component: TimeComponent) -> Self {
        self.specs.push(FeatureSpec::Cyclical(component));
        self
    }

    /// Add a binary (0/1) calendar indicator.
    pub fn binary(mut self, indicator: BinaryIndicator) -> Self {
        self.specs.push(FeatureSpec::Binary(indicator));
        self
    }

    /// Add an advanced numeric calendar feature.
    pub fn advanced(mut self, feature: AdvancedFeature) -> Self {
        self.specs.push(FeatureSpec::Advanced(feature));
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
                FeatureSpec::Cyclical(component) => {
                    use chrono::{Datelike, Timelike};
                    let (name, period) = match component {
                        TimeComponent::Month => ("month", 12.0),
                        TimeComponent::Quarter => ("quarter", 4.0),
                        TimeComponent::Semester => ("semester", 2.0),
                        TimeComponent::WeekOfYear => ("week_of_year", 53.0),
                        TimeComponent::DayOfWeek => ("day_of_week", 7.0),
                        TimeComponent::DayOfMonth => ("day_of_month", 31.0),
                        TimeComponent::DayOfYear => ("day_of_year", 366.0),
                        TimeComponent::Hour => ("hour", 24.0),
                        TimeComponent::Minute => ("minute", 60.0),
                        TimeComponent::Second => ("second", 60.0),
                    };
                    let mut sin_col = Vec::with_capacity(n);
                    let mut cos_col = Vec::with_capacity(n);
                    for ts in timestamps {
                        let value = match component {
                            TimeComponent::Month => ts.month() as f64,
                            TimeComponent::Quarter => ((ts.month() - 1) / 3 + 1) as f64,
                            TimeComponent::Semester => ((ts.month() - 1) / 6 + 1) as f64,
                            TimeComponent::WeekOfYear => ts.iso_week().week() as f64,
                            TimeComponent::DayOfWeek => ts.weekday().num_days_from_monday() as f64,
                            TimeComponent::DayOfMonth => ts.day() as f64,
                            TimeComponent::DayOfYear => ts.ordinal() as f64,
                            TimeComponent::Hour => ts.hour() as f64,
                            TimeComponent::Minute => ts.minute() as f64,
                            TimeComponent::Second => ts.second() as f64,
                        };
                        let angle = 2.0 * PI * value / period;
                        sin_col.push(angle.sin());
                        cos_col.push(angle.cos());
                    }
                    result.insert(format!("{}_sin", name), sin_col);
                    result.insert(format!("{}_cos", name), cos_col);
                }
                FeatureSpec::Binary(indicator) => {
                    use chrono::Datelike;
                    let (name, test_fn): (&str, Box<dyn Fn(&DateTime<Utc>) -> bool>) =
                        match indicator {
                            BinaryIndicator::MonthStart => {
                                ("month_start", Box::new(|ts| ts.day() == 1))
                            }
                            BinaryIndicator::MonthEnd => (
                                "month_end",
                                Box::new(|ts| {
                                    let max_day = crate::core::time_series::days_in_month_pub(
                                        ts.year(),
                                        ts.month(),
                                    );
                                    ts.day() == max_day
                                }),
                            ),
                            BinaryIndicator::QuarterStart => (
                                "quarter_start",
                                Box::new(|ts| {
                                    ts.day() == 1 && matches!(ts.month(), 1 | 4 | 7 | 10)
                                }),
                            ),
                            BinaryIndicator::QuarterEnd => (
                                "quarter_end",
                                Box::new(|ts| {
                                    let m = ts.month();
                                    let max_day =
                                        crate::core::time_series::days_in_month_pub(ts.year(), m);
                                    ts.day() == max_day && matches!(m, 3 | 6 | 9 | 12)
                                }),
                            ),
                            BinaryIndicator::YearStart => (
                                "year_start",
                                Box::new(|ts| ts.month() == 1 && ts.day() == 1),
                            ),
                            BinaryIndicator::YearEnd => (
                                "year_end",
                                Box::new(|ts| ts.month() == 12 && ts.day() == 31),
                            ),
                            BinaryIndicator::Weekend => (
                                "weekend",
                                Box::new(|ts| {
                                    matches!(
                                        ts.weekday(),
                                        chrono::Weekday::Sat | chrono::Weekday::Sun
                                    )
                                }),
                            ),
                        };
                    let col: Vec<f64> = timestamps
                        .iter()
                        .map(|ts| if test_fn(ts) { 1.0 } else { 0.0 })
                        .collect();
                    result.insert(name.to_string(), col);
                }
                FeatureSpec::Advanced(feature) => {
                    use chrono::Datelike;
                    let (name, compute_fn): (&str, Box<dyn Fn(&DateTime<Utc>) -> f64>) =
                        match feature {
                            AdvancedFeature::LeapYear => (
                                "leap_year",
                                Box::new(|ts| {
                                    if crate::core::time_series::is_leap_year_pub(ts.year()) {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }),
                            ),
                            AdvancedFeature::DaysInMonth => (
                                "days_in_month",
                                Box::new(|ts| {
                                    crate::core::time_series::days_in_month_pub(
                                        ts.year(),
                                        ts.month(),
                                    ) as f64
                                }),
                            ),
                        };
                    let col: Vec<f64> = timestamps.iter().map(|ts| compute_fn(ts)).collect();
                    result.insert(name.to_string(), col);
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

    // ── Cyclical encoding correctness ──────────────────────────────────

    #[test]
    fn cyclical_month_january_values() {
        // January: value=1, period=12, angle = 2*pi*1/12
        let ts = vec![Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap()];
        let gen = FeatureGenerator::new().cyclical(TimeComponent::Month);
        let feats = gen.generate(&ts);

        let expected_angle = 2.0 * PI * 1.0 / 12.0;
        let sin_val = feats["month_sin"][0];
        let cos_val = feats["month_cos"][0];
        assert!(
            (sin_val - expected_angle.sin()).abs() < 1e-10,
            "Jan sin: expected {}, got {}",
            expected_angle.sin(),
            sin_val,
        );
        assert!(
            (cos_val - expected_angle.cos()).abs() < 1e-10,
            "Jan cos: expected {}, got {}",
            expected_angle.cos(),
            cos_val,
        );
    }

    #[test]
    fn cyclical_month_december_wraps_near_january() {
        // Dec: value=12, angle = 2*pi*12/12 = 2*pi => same as 0
        // Jan: value=1, angle = 2*pi*1/12
        // The point is that December (12) wraps: sin(2*pi*12/12) = sin(2*pi) ~ 0
        // and January (1) has sin(2*pi*1/12) ~ 0.5, so they are close in the
        // cyclical space compared to, say, July (7).
        let jan = vec![Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap()];
        let dec = vec![Utc.with_ymd_and_hms(2024, 12, 15, 0, 0, 0).unwrap()];
        let jul = vec![Utc.with_ymd_and_hms(2024, 7, 15, 0, 0, 0).unwrap()];
        let gen = FeatureGenerator::new().cyclical(TimeComponent::Month);

        let jan_f = gen.generate(&jan);
        let dec_f = gen.generate(&dec);
        let jul_f = gen.generate(&jul);

        // Euclidean distance in (sin, cos) space
        let dist_dec_jan = ((dec_f["month_sin"][0] - jan_f["month_sin"][0]).powi(2)
            + (dec_f["month_cos"][0] - jan_f["month_cos"][0]).powi(2))
        .sqrt();
        let dist_jul_jan = ((jul_f["month_sin"][0] - jan_f["month_sin"][0]).powi(2)
            + (jul_f["month_cos"][0] - jan_f["month_cos"][0]).powi(2))
        .sqrt();

        assert!(
            dist_dec_jan < dist_jul_jan,
            "Dec should be closer to Jan than Jul in cyclical space: dec-jan={}, jul-jan={}",
            dist_dec_jan,
            dist_jul_jan,
        );
    }

    #[test]
    fn cyclical_day_of_week_monday_differs_from_sunday() {
        // Monday=0, Sunday=6
        // 2024-01-01 is Monday, 2024-01-07 is Sunday
        let monday = vec![Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()];
        let sunday = vec![Utc.with_ymd_and_hms(2024, 1, 7, 0, 0, 0).unwrap()];
        let gen = FeatureGenerator::new().cyclical(TimeComponent::DayOfWeek);

        let mon_f = gen.generate(&monday);
        let sun_f = gen.generate(&sunday);

        // Monday: value=0, angle=0 => sin=0, cos=1
        assert!((mon_f["day_of_week_sin"][0] - 0.0).abs() < 1e-10);
        assert!((mon_f["day_of_week_cos"][0] - 1.0).abs() < 1e-10);

        // Sunday: value=6, angle=2*pi*6/7
        let expected_angle = 2.0 * PI * 6.0 / 7.0;
        assert!(
            (sun_f["day_of_week_sin"][0] - expected_angle.sin()).abs() < 1e-10,
            "Sunday sin: expected {}, got {}",
            expected_angle.sin(),
            sun_f["day_of_week_sin"][0],
        );
        assert!(
            (sun_f["day_of_week_cos"][0] - expected_angle.cos()).abs() < 1e-10,
            "Sunday cos: expected {}, got {}",
            expected_angle.cos(),
            sun_f["day_of_week_cos"][0],
        );

        // They should differ
        assert!(
            (mon_f["day_of_week_sin"][0] - sun_f["day_of_week_sin"][0]).abs() > 1e-5
                || (mon_f["day_of_week_cos"][0] - sun_f["day_of_week_cos"][0]).abs() > 1e-5,
            "Monday and Sunday should have different cyclical encodings"
        );
    }

    #[test]
    fn cyclical_hour_zero_equals_hour_24_wrap() {
        // Hour 0 and hour 24 should produce the same values since
        // sin(2*pi*0/24) = sin(2*pi*24/24) = sin(0) = sin(2*pi) = 0
        // cos(2*pi*0/24) = cos(2*pi*24/24) = cos(0) = cos(2*pi) = 1
        // Note: chrono hours are 0-23, there is no hour=24. But hour=0
        // should equal what hour=24 would be (i.e., wrap around).
        let hour0 = vec![Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()];
        let gen = FeatureGenerator::new().cyclical(TimeComponent::Hour);
        let feats = gen.generate(&hour0);

        // hour=0: angle = 2*pi*0/24 = 0
        // hour=24 would be: angle = 2*pi*24/24 = 2*pi (same as 0)
        let sin_0 = feats["hour_sin"][0];
        let cos_0 = feats["hour_cos"][0];

        // sin(0) = 0, cos(0) = 1 -- same as sin(2*pi), cos(2*pi)
        assert!(
            sin_0.abs() < 1e-10,
            "hour=0 sin should be 0.0, got {}",
            sin_0
        );
        assert!(
            (cos_0 - 1.0).abs() < 1e-10,
            "hour=0 cos should be 1.0, got {}",
            cos_0
        );
    }

    #[test]
    fn cyclical_all_values_in_range() {
        // Generate cyclical features for all components over a full year of
        // hourly data and verify every value lies in [-1, 1].
        let hourly_ts: Vec<DateTime<Utc>> =
            (0..8760) // 365 * 24
                .map(|i| {
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64)
                })
                .collect();

        let gen = FeatureGenerator::new()
            .cyclical(TimeComponent::Month)
            .cyclical(TimeComponent::Quarter)
            .cyclical(TimeComponent::Semester)
            .cyclical(TimeComponent::WeekOfYear)
            .cyclical(TimeComponent::DayOfWeek)
            .cyclical(TimeComponent::DayOfMonth)
            .cyclical(TimeComponent::DayOfYear)
            .cyclical(TimeComponent::Hour)
            .cyclical(TimeComponent::Minute)
            .cyclical(TimeComponent::Second);

        let feats = gen.generate(&hourly_ts);

        for (name, values) in &feats {
            for (i, &v) in values.iter().enumerate() {
                assert!(
                    v >= -1.0 && v <= 1.0,
                    "Feature '{}' at index {} has value {} outside [-1, 1]",
                    name,
                    i,
                    v,
                );
            }
        }
    }

    // ── Binary indicators ──────────────────────────────────────────────

    #[test]
    fn binary_month_start_only_day_1() {
        // Generate daily timestamps for 3 months (Jan-Mar 2024)
        let ts: Vec<DateTime<Utc>> = (0..91)
            .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
            .collect();
        let gen = FeatureGenerator::new().binary(BinaryIndicator::MonthStart);
        let feats = gen.generate(&ts);
        let col = &feats["month_start"];

        // Jan 1 = index 0, Feb 1 = index 31, Mar 1 = index 31+29=60 (2024 is leap)
        let ones: Vec<usize> = col
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            ones,
            vec![0, 31, 60],
            "MonthStart should only fire on day 1"
        );

        // All others should be 0
        let zero_count = col.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 91 - 3);
    }

    #[test]
    fn binary_month_end_various_months() {
        // Test specific month-end dates
        let dates = vec![
            // Jan 31, 2024 (31 days)
            Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap(),
            // Jan 30, 2024 (not end)
            Utc.with_ymd_and_hms(2024, 1, 30, 0, 0, 0).unwrap(),
            // Feb 29, 2024 (leap year)
            Utc.with_ymd_and_hms(2024, 2, 29, 0, 0, 0).unwrap(),
            // Feb 28, 2024 (NOT end in leap year)
            Utc.with_ymd_and_hms(2024, 2, 28, 0, 0, 0).unwrap(),
            // Feb 28, 2023 (end in non-leap year)
            Utc.with_ymd_and_hms(2023, 2, 28, 0, 0, 0).unwrap(),
            // Apr 30, 2024 (30 days)
            Utc.with_ymd_and_hms(2024, 4, 30, 0, 0, 0).unwrap(),
            // Apr 29, 2024 (not end)
            Utc.with_ymd_and_hms(2024, 4, 29, 0, 0, 0).unwrap(),
        ];
        let gen = FeatureGenerator::new().binary(BinaryIndicator::MonthEnd);
        let feats = gen.generate(&dates);
        let col = &feats["month_end"];

        assert_eq!(col[0], 1.0, "Jan 31 should be month end");
        assert_eq!(col[1], 0.0, "Jan 30 should NOT be month end");
        assert_eq!(col[2], 1.0, "Feb 29 (leap) should be month end");
        assert_eq!(col[3], 0.0, "Feb 28 (leap year) should NOT be month end");
        assert_eq!(col[4], 1.0, "Feb 28 (non-leap) should be month end");
        assert_eq!(col[5], 1.0, "Apr 30 should be month end");
        assert_eq!(col[6], 0.0, "Apr 29 should NOT be month end");
    }

    #[test]
    fn binary_quarter_start() {
        // Only Jan 1, Apr 1, Jul 1, Oct 1 should be 1
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // Q1 start
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(), // not
            Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).unwrap(), // not (month start but not quarter)
            Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap(), // Q2 start
            Utc.with_ymd_and_hms(2024, 5, 1, 0, 0, 0).unwrap(), // not
            Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap(), // Q3 start
            Utc.with_ymd_and_hms(2024, 10, 1, 0, 0, 0).unwrap(), // Q4 start
            Utc.with_ymd_and_hms(2024, 10, 2, 0, 0, 0).unwrap(), // not
        ];
        let gen = FeatureGenerator::new().binary(BinaryIndicator::QuarterStart);
        let feats = gen.generate(&dates);
        let col = &feats["quarter_start"];

        assert_eq!(col, &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn binary_quarter_end() {
        // Only Mar 31, Jun 30, Sep 30, Dec 31 should be 1
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 3, 31, 0, 0, 0).unwrap(), // Q1 end
            Utc.with_ymd_and_hms(2024, 3, 30, 0, 0, 0).unwrap(), // not
            Utc.with_ymd_and_hms(2024, 6, 30, 0, 0, 0).unwrap(), // Q2 end
            Utc.with_ymd_and_hms(2024, 6, 29, 0, 0, 0).unwrap(), // not
            Utc.with_ymd_and_hms(2024, 9, 30, 0, 0, 0).unwrap(), // Q3 end
            Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(), // Q4 end
            Utc.with_ymd_and_hms(2024, 12, 30, 0, 0, 0).unwrap(), // not
            Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap(), // month end but not quarter end
        ];
        let gen = FeatureGenerator::new().binary(BinaryIndicator::QuarterEnd);
        let feats = gen.generate(&dates);
        let col = &feats["quarter_end"];

        assert_eq!(col, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn binary_year_start() {
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), // yes
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(), // no
            Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(), // no
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(), // yes
            Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap(), // no (month start, not year)
        ];
        let gen = FeatureGenerator::new().binary(BinaryIndicator::YearStart);
        let feats = gen.generate(&dates);
        let col = &feats["year_start"];

        assert_eq!(col, &[1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn binary_year_end() {
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(), // yes
            Utc.with_ymd_and_hms(2024, 12, 30, 0, 0, 0).unwrap(), // no
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),   // no
            Utc.with_ymd_and_hms(2023, 12, 31, 0, 0, 0).unwrap(), // yes
            Utc.with_ymd_and_hms(2024, 6, 30, 0, 0, 0).unwrap(),  // no (quarter end, not year)
        ];
        let gen = FeatureGenerator::new().binary(BinaryIndicator::YearEnd);
        let feats = gen.generate(&dates);
        let col = &feats["year_end"];

        assert_eq!(col, &[1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn binary_weekend() {
        // 2024-01-01 is Monday
        let ts = daily_timestamps(7);
        let gen = FeatureGenerator::new().binary(BinaryIndicator::Weekend);
        let feats = gen.generate(&ts);
        let col = &feats["weekend"];

        // Mon=0, Tue=0, Wed=0, Thu=0, Fri=0, Sat=1, Sun=1
        assert_eq!(col, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    // ── Advanced features ──────────────────────────────────────────────

    #[test]
    fn advanced_days_in_month() {
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 2, 15, 0, 0, 0).unwrap(), // Feb 2024 (leap) = 29
            Utc.with_ymd_and_hms(2023, 2, 15, 0, 0, 0).unwrap(), // Feb 2023 (non-leap) = 28
            Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(), // Jan = 31
            Utc.with_ymd_and_hms(2024, 4, 20, 0, 0, 0).unwrap(), // Apr = 30
        ];
        let gen = FeatureGenerator::new().advanced(AdvancedFeature::DaysInMonth);
        let feats = gen.generate(&dates);
        let col = &feats["days_in_month"];

        assert_eq!(col[0], 29.0, "Feb 2024 (leap) should have 29 days");
        assert_eq!(col[1], 28.0, "Feb 2023 (non-leap) should have 28 days");
        assert_eq!(col[2], 31.0, "Jan should have 31 days");
        assert_eq!(col[3], 30.0, "Apr should have 30 days");
    }

    #[test]
    fn advanced_leap_year() {
        let dates = vec![
            Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(), // leap (div by 4)
            Utc.with_ymd_and_hms(2023, 6, 1, 0, 0, 0).unwrap(), // not leap
            Utc.with_ymd_and_hms(2000, 6, 1, 0, 0, 0).unwrap(), // leap (div by 400)
            Utc.with_ymd_and_hms(1900, 6, 1, 0, 0, 0).unwrap(), // NOT leap (div by 100 but not 400)
        ];
        let gen = FeatureGenerator::new().advanced(AdvancedFeature::LeapYear);
        let feats = gen.generate(&dates);
        let col = &feats["leap_year"];

        assert_eq!(col[0], 1.0, "2024 should be leap year");
        assert_eq!(col[1], 0.0, "2023 should NOT be leap year");
        assert_eq!(col[2], 1.0, "2000 should be leap year");
        assert_eq!(col[3], 0.0, "1900 should NOT be leap year");
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn empty_timestamp_slice() {
        let empty: Vec<DateTime<Utc>> = vec![];
        let gen = FeatureGenerator::new()
            .cyclical(TimeComponent::Month)
            .cyclical(TimeComponent::Hour)
            .binary(BinaryIndicator::MonthStart)
            .binary(BinaryIndicator::Weekend)
            .advanced(AdvancedFeature::DaysInMonth)
            .advanced(AdvancedFeature::LeapYear)
            .fourier(7, 2)
            .day_of_week()
            .month_of_year();

        let feats = gen.generate(&empty);

        // All columns should exist but be empty
        assert!(
            !feats.is_empty(),
            "Features map should have keys even for empty input"
        );
        for (name, values) in &feats {
            assert!(
                values.is_empty(),
                "Feature '{}' should have 0 elements for empty timestamps, got {}",
                name,
                values.len(),
            );
        }
    }

    #[test]
    fn single_timestamp() {
        let single = vec![Utc.with_ymd_and_hms(2024, 6, 15, 12, 30, 45).unwrap()];
        let gen = FeatureGenerator::new()
            .cyclical(TimeComponent::Month)
            .cyclical(TimeComponent::DayOfWeek)
            .cyclical(TimeComponent::Hour)
            .binary(BinaryIndicator::MonthStart)
            .binary(BinaryIndicator::MonthEnd)
            .binary(BinaryIndicator::Weekend)
            .advanced(AdvancedFeature::DaysInMonth)
            .advanced(AdvancedFeature::LeapYear);

        let feats = gen.generate(&single);

        // All columns should have exactly 1 element
        for (name, values) in &feats {
            assert_eq!(
                values.len(),
                1,
                "Feature '{}' should have 1 element, got {}",
                name,
                values.len(),
            );
        }

        // June 15, 2024 is a Saturday
        assert_eq!(feats["weekend"][0], 1.0, "June 15, 2024 is a Saturday");
        assert_eq!(feats["month_start"][0], 0.0, "Day 15 is not month start");
        assert_eq!(feats["month_end"][0], 0.0, "Day 15 is not month end");
        assert_eq!(feats["days_in_month"][0], 30.0, "June has 30 days");
        assert_eq!(feats["leap_year"][0], 1.0, "2024 is a leap year");

        // Cyclical month: June=6, angle=2*pi*6/12 = pi
        let expected_sin = (2.0 * PI * 6.0 / 12.0).sin();
        let expected_cos = (2.0 * PI * 6.0 / 12.0).cos();
        assert!(
            (feats["month_sin"][0] - expected_sin).abs() < 1e-10,
            "Month sin for June: expected {}, got {}",
            expected_sin,
            feats["month_sin"][0],
        );
        assert!(
            (feats["month_cos"][0] - expected_cos).abs() < 1e-10,
            "Month cos for June: expected {}, got {}",
            expected_cos,
            feats["month_cos"][0],
        );
    }
}
