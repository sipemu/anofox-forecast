//! Levenbach's STI_Class taxonomy.
//!
//! Classifies a monthly (or pseudo-monthly) series into one of six
//! categories by ranking the relative magnitude of the Seasonal /
//! Trend / Irregular mean-squares from a two-way ANOVA without
//! replication on a `years × months` grid.
//!
//! The premise is that method ranking should be performed within
//! class rather than globally — seasonal-dominant classes (`Sit`,
//! `Sti`) typically favour seasonal baselines, trend-dominant ones
//! (`Tsi`, `Tis`) favour persistence (`Naive`), and irregular-dominant
//! classes (`Ist`, `Its`) are essentially noise.
//!
//! # Reference
//!
//! - Levenbach, H. (2025). STI_Class scheme — a classification
//!   framework for model selection. <https://www.linkedin.com/pulse/sticlass-scheme-classification-framework-model-levenbach-phd-cpdf-va7ae/>
//!
//! # Evaluation on M4 monthly
//!
//! On 48,000 M4-Monthly series (last 60 months as a 5 × 12 grid), the
//! best baseline genuinely flips between classes — `SeasonalNaive(12)`
//! wins 67–88 % of seasonal-dominant series, plain `Naive` wins the
//! trend-dominant and irregular-dominant ones, with a 58-pp spread.
//! `SF_trnd` correlates ρ = 0.97 with `STL::trend_strength`; the new
//! information is the categorical class assignment.
//!
//! # Granularity
//!
//! Designed for monthly granularity. On sub-monthly data the scheme
//! tends to assign almost everything to a single trend-dominant
//! class — call this on monthly aggregates only.

/// The six STI_Class categories.
///
/// Variants are named by the ranking of the three mean-squares
/// (Seasonal, Trend, Irregular) from largest to smallest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StiClass {
    /// `S > I > T` — seasonal dominates, irregular second, trend last.
    Sit,
    /// `S > T > I` — seasonal dominates, trend second, irregular last.
    Sti,
    /// `I > S > T` — irregular dominates, seasonal second, trend last.
    Ist,
    /// `T > S > I` — trend dominates, seasonal second, irregular last.
    Tsi,
    /// `I > T > S` — irregular dominates, trend second, seasonal last.
    Its,
    /// `T > I > S` — trend dominates, irregular second, seasonal last.
    Tis,
}

impl std::fmt::Display for StiClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sit => write!(f, "SIT"),
            Self::Sti => write!(f, "STI"),
            Self::Ist => write!(f, "IST"),
            Self::Tsi => write!(f, "TSI"),
            Self::Its => write!(f, "ITS"),
            Self::Tis => write!(f, "TIS"),
        }
    }
}

/// Result of an STI_Class classification.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StiClassResult {
    /// The assigned class.
    pub class: StiClass,
    /// Seasonal strength factor `SS(col) / (SS(col) + SS(err))`.
    /// Closer in spirit to `STL::seasonal_strength` but uses ANOVA's
    /// hard 12-month bucket rather than smoother decomposition (ρ ≈ 0.79
    /// against the STL metric on M4 monthly).
    pub sf_seas: f64,
    /// Trend strength factor `SS(row) / (SS(row) + SS(err))`. Empirically
    /// near-identical to `STL::trend_strength` (ρ ≈ 0.97).
    pub sf_trnd: f64,
    /// Seasonality F-statistic `MS(col) / MS(err)`.
    pub sif: f64,
    /// Trend F-statistic `MS(row) / MS(err)`.
    pub tif: f64,
}

/// Classify a monthly series into one of the six STI_Class categories.
///
/// Arranges the last `years × months_per_year` observations as a
/// `years × months_per_year` matrix (rows = years, columns = months)
/// and runs a two-way ANOVA without replication. The class is assigned
/// by the relative magnitude of the three mean-squares.
///
/// # Arguments
///
/// - `series` — the time series, ordered chronologically. Must contain
///   at least `years * months_per_year` observations; only the last
///   `years * months_per_year` are used.
/// - `months_per_year` — number of columns in the grid (usually 12).
///   Must be ≥ 2.
/// - `years` — number of rows in the grid. Must be ≥ 2.
///
/// # Returns
///
/// `Some(StiClassResult)` on success, or `None` on a degenerate input:
///
/// - too few observations
/// - `months_per_year < 2` or `years < 2` (ANOVA needs ≥ 1 df per source)
/// - any non-finite value in the grid
/// - zero total variance (constant series) or zero MS(err) (perfectly
///   decomposable — ties can't be ranked)
///
/// # Example
///
/// ```
/// use anofox_forecast::forecastability::sti_class::{sti_class, StiClass};
///
/// // Pure seasonal sinusoid at period 12.
/// let n = 60;
/// let series: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin())
///     .collect();
/// let result = sti_class(&series, 12, 5).unwrap();
/// // Seasonal must rank first.
/// assert!(matches!(result.class, StiClass::Sit | StiClass::Sti));
/// ```
pub fn sti_class(series: &[f64], months_per_year: usize, years: usize) -> Option<StiClassResult> {
    if months_per_year < 2 || years < 2 {
        return None;
    }
    let needed = months_per_year.checked_mul(years)?;
    if series.len() < needed {
        return None;
    }

    // Use the last `years * months_per_year` observations.
    let tail = &series[series.len() - needed..];
    if tail.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let m = months_per_year;
    let y = years;
    let n = needed as f64;

    // Grand mean.
    let y_bar: f64 = tail.iter().sum::<f64>() / n;

    // Row (year) means and column (month) means.
    let mut row_mean = vec![0.0_f64; y];
    let mut col_mean = vec![0.0_f64; m];
    for (idx, &v) in tail.iter().enumerate() {
        let row = idx / m;
        let col = idx % m;
        row_mean[row] += v;
        col_mean[col] += v;
    }
    for v in row_mean.iter_mut() {
        *v /= m as f64;
    }
    for v in col_mean.iter_mut() {
        *v /= y as f64;
    }

    // SS components.
    let mut ss_total = 0.0_f64;
    for &v in tail {
        let d = v - y_bar;
        ss_total += d * d;
    }
    let mut ss_row = 0.0_f64;
    for &r in &row_mean {
        let d = r - y_bar;
        ss_row += d * d;
    }
    ss_row *= m as f64;
    let mut ss_col = 0.0_f64;
    for &c in &col_mean {
        let d = c - y_bar;
        ss_col += d * d;
    }
    ss_col *= y as f64;
    let ss_err = ss_total - ss_row - ss_col;

    if ss_total <= 0.0 || !ss_total.is_finite() {
        return None;
    }
    if ss_err <= 0.0 || !ss_err.is_finite() {
        // Floating-point edge: a perfectly decomposable grid (no
        // residual variance) can't be ranked against the other
        // sources. Surface as unclassified.
        return None;
    }

    // Degrees of freedom.
    let df_row = (y - 1) as f64;
    let df_col = (m - 1) as f64;
    let df_err = df_row * df_col;

    // Mean squares.
    let ms_row = ss_row / df_row;
    let ms_col = ss_col / df_col;
    let ms_err = ss_err / df_err;

    // Strength factors.
    let sf_seas = ss_col / (ss_col + ss_err);
    let sf_trnd = ss_row / (ss_row + ss_err);

    // F-statistics.
    let sif = ms_col / ms_err;
    let tif = ms_row / ms_err;

    // Class assignment by ranking (S, T, I) := (ms_col, ms_row, ms_err).
    let class = rank_to_class(ms_col, ms_row, ms_err);

    Some(StiClassResult {
        class,
        sf_seas,
        sf_trnd,
        sif,
        tif,
    })
}

/// Map the ordering of `(seasonal, trend, irregular)` mean-squares to
/// the six-class enum.
fn rank_to_class(s: f64, t: f64, i: f64) -> StiClass {
    if s >= t && s >= i {
        // S is largest
        if t >= i {
            StiClass::Sti
        } else {
            StiClass::Sit
        }
    } else if t >= s && t >= i {
        // T is largest
        if s >= i {
            StiClass::Tsi
        } else {
            StiClass::Tis
        }
    } else {
        // I is largest
        if s >= t {
            StiClass::Ist
        } else {
            StiClass::Its
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: usize = 12;
    const Y: usize = 5;

    #[test]
    fn pure_seasonal_classifies_as_seasonal_dominant() {
        // Strict sinusoid at period 12 — no trend, no noise. Seasonal
        // SS dominates total SS, MS(err) is essentially 0 within
        // float tolerance. The detector should pick a seasonal class.
        let n = M * Y;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * (i % M) as f64 / M as f64).sin()
                    + 0.001 * ((i * 7) % 11) as f64 // tiny dither so MS(err) > 0
            })
            .collect();
        let result = sti_class(&series, M, Y).expect("classify pure seasonal");
        assert!(
            matches!(result.class, StiClass::Sit | StiClass::Sti),
            "expected seasonal-dominant class, got {}",
            result.class,
        );
        assert!(result.sf_seas > 0.9, "SF_seas too low: {}", result.sf_seas);
    }

    #[test]
    fn pure_trend_classifies_as_trend_dominant() {
        // Monotone linear trend — no seasonal, no noise. Trend SS
        // dominates.
        let n = M * Y;
        let series: Vec<f64> = (0..n)
            .map(|i| i as f64 + 0.001 * ((i * 13) % 7) as f64)
            .collect();
        let result = sti_class(&series, M, Y).expect("classify pure trend");
        assert!(
            matches!(result.class, StiClass::Tsi | StiClass::Tis),
            "expected trend-dominant class, got {}",
            result.class,
        );
        assert!(result.sf_trnd > 0.9, "SF_trnd too low: {}", result.sf_trnd);
    }

    #[test]
    fn noise_dominated_signal_has_small_strength_factors() {
        // Under pure white noise the three mean-squares are equal in
        // expectation, so the class assignment is effectively random
        // — any of the 6 classes can come up depending on sample
        // fluctuations (especially with the small dof ratios in a
        // 5×12 or 20×12 grid). What *is* deterministic under noise
        // dominance is that both strength factors stay small.
        let years = 20;
        let n = M * years;
        let mut state: u64 = 0xCAFEBABE_DEADBEEF;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let trend = 0.001 * i as f64;
                let seasonal =
                    0.01 * (2.0 * std::f64::consts::PI * (i % M) as f64 / M as f64).sin();
                100.0 * noise + trend + seasonal
            })
            .collect();
        let result = sti_class(&series, M, years).expect("classify noise-dominated");
        // Both strength factors should be small under noise dominance.
        assert!(
            result.sf_seas < 0.5,
            "noise-dominated SF_seas = {} (expected < 0.5)",
            result.sf_seas
        );
        assert!(
            result.sf_trnd < 0.5,
            "noise-dominated SF_trnd = {} (expected < 0.5)",
            result.sf_trnd
        );
    }

    #[test]
    fn constant_series_returns_none() {
        let series = vec![7.0; M * Y];
        assert!(sti_class(&series, M, Y).is_none());
    }

    #[test]
    fn too_short_returns_none() {
        let series = vec![1.0; M * Y - 1];
        assert!(sti_class(&series, M, Y).is_none());
    }

    #[test]
    fn rejects_degenerate_grid_dims() {
        let series = vec![1.0_f64; 100];
        assert!(sti_class(&series, 1, 5).is_none());
        assert!(sti_class(&series, 12, 1).is_none());
    }

    #[test]
    fn rejects_non_finite_observations() {
        let mut series: Vec<f64> = (0..M * Y).map(|i| i as f64).collect();
        series[5] = f64::NAN;
        assert!(sti_class(&series, M, Y).is_none());
    }

    #[test]
    fn uses_only_last_window_for_classification() {
        // First half noise, second half perfectly seasonal — the
        // detector should classify the *tail* and call it seasonal,
        // independent of the noisy head.
        let n_head = M * Y;
        let n_tail = M * Y;
        let mut series: Vec<f64> = (0..n_head)
            .map(|i| ((i * 17) % 23) as f64 - 11.0) // noisy head
            .collect();
        series.extend((0..n_tail).map(|i| {
            (2.0 * std::f64::consts::PI * (i % M) as f64 / M as f64).sin()
                + 0.001 * ((i * 7) % 11) as f64
        }));
        let result = sti_class(&series, M, Y).expect("classify tail");
        assert!(
            matches!(result.class, StiClass::Sit | StiClass::Sti),
            "tail should drive class, got {}",
            result.class,
        );
    }

    #[test]
    fn strength_factors_in_unit_interval() {
        let n = M * Y;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * (i % M) as f64 / M as f64).sin()
                    + 0.1 * i as f64
                    + 0.05 * ((i * 7) % 11) as f64
            })
            .collect();
        let result = sti_class(&series, M, Y).unwrap();
        assert!((0.0..=1.0).contains(&result.sf_seas));
        assert!((0.0..=1.0).contains(&result.sf_trnd));
    }
}
