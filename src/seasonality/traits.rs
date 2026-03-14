//! Traits for composable seasonality and trend components.
//!
//! Components implementing these traits can be used standalone (fit, predict,
//! get fitted values) or as feature extractors via the `features()` method.
//!
//! The [`Recency`] enum controls how much of the data is used for fitting:
//! recent observations are typically more relevant for forecasting than
//! distant history.

use crate::changepoint::pelt::Pelt;
use crate::changepoint::CostFunction;
use crate::error::Result;

/// Configuration for automatic recency detection via changepoint analysis.
///
/// When `Recency::Auto` is used, PELT changepoint detection runs on the data
/// and the fitting window starts at the last detected changepoint. If no
/// changepoints are found, the fallback fraction is used instead.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoRecencyConfig {
    /// Fraction of data to use when no changepoints are detected (default: 0.3).
    pub fallback_fraction: f64,
    /// PELT penalty (default: BIC = `ln(n)`). Set to `None` for BIC.
    pub penalty: Option<f64>,
    /// Cost function for changepoint detection (default: `LinearTrend`).
    pub cost_fn: CostFunction,
    /// Minimum segment length (default: 5).
    pub min_segment_length: usize,
}

impl Default for AutoRecencyConfig {
    fn default() -> Self {
        Self {
            fallback_fraction: 0.3,
            penalty: None, // BIC = ln(n)
            cost_fn: CostFunction::LinearTrend,
            min_segment_length: 5,
        }
    }
}

/// Controls how much of the data is used for fitting.
///
/// For forecasting, the most recent trend is usually what matters.
/// `Recency` specifies which portion of the data to use for parameter
/// estimation. The fitted values still cover the full series (the portion
/// before the recency window is filled by evaluating the fitted model at
/// earlier indices — backwards extrapolation).
///
/// # Examples
///
/// ```
/// use anofox_forecast::seasonality::Recency;
///
/// // Use only the last 30% of data (default)
/// let r = Recency::default();
/// assert_eq!(r, Recency::Fraction(0.3));
///
/// // Use the last 50 observations
/// let r = Recency::Window(50);
/// let (start, end) = r.resolve(200);
/// assert_eq!((start, end), (150, 200));
///
/// // Use all data
/// let r = Recency::Full;
/// let (start, end) = r.resolve(200);
/// assert_eq!((start, end), (0, 200));
///
/// // Automatic: detect last changepoint via PELT
/// let r = Recency::auto();
/// // resolve_with_data() runs PELT on the actual values
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Recency {
    /// Use the last `N` observations for fitting.
    Window(usize),
    /// Use the last fraction of data (e.g., 0.3 = last 30%). Default: 0.3.
    Fraction(f64),
    /// Use all data.
    Full,
    /// Automatically detect the recency window via changepoint detection.
    ///
    /// Runs PELT on the data and fits from the last detected changepoint.
    /// Falls back to a fraction if no changepoints are found.
    Auto(AutoRecencyConfig),
}

impl Default for Recency {
    fn default() -> Self {
        Recency::Fraction(0.3)
    }
}

impl Recency {
    /// Create an `Auto` recency with default configuration.
    ///
    /// Uses PELT with `LinearTrend` cost, BIC penalty, and falls back to 30%.
    pub fn auto() -> Self {
        Recency::Auto(AutoRecencyConfig::default())
    }

    /// Resolve the recency specification to a `(start, end)` index range.
    ///
    /// Returns the half-open range `[start, end)` within a series of length `n`.
    /// The range always covers at least 3 observations (the minimum for most
    /// regression models).
    ///
    /// **Note**: For `Recency::Auto`, this falls back to the configured
    /// `fallback_fraction` since the actual data is not available. Use
    /// [`resolve_with_data`](Self::resolve_with_data) to run changepoint
    /// detection on the actual values.
    pub fn resolve(&self, n: usize) -> (usize, usize) {
        if n == 0 {
            return (0, 0);
        }
        let min_window = 3.min(n);
        let start = match self {
            Recency::Full => 0,
            Recency::Window(w) => n.saturating_sub(*w),
            Recency::Fraction(f) => {
                let frac = f.clamp(0.0, 1.0);
                let window = (n as f64 * frac).ceil() as usize;
                n.saturating_sub(window.max(min_window))
            }
            Recency::Auto(config) => {
                // Fallback when data is not available
                let frac = config.fallback_fraction.clamp(0.0, 1.0);
                let window = (n as f64 * frac).ceil() as usize;
                n.saturating_sub(window.max(min_window))
            }
        };
        // Ensure at least min_window observations
        let start = start.min(n.saturating_sub(min_window));
        (start, n)
    }

    /// Resolve the recency specification using the actual data values.
    ///
    /// For `Recency::Auto`, this runs PELT changepoint detection and starts
    /// the fitting window at the last detected changepoint. For all other
    /// variants, it delegates to [`resolve`](Self::resolve).
    ///
    /// # Examples
    ///
    /// ```
    /// use anofox_forecast::seasonality::Recency;
    ///
    /// // Regime change: slope +3 then slope -2 at index 180
    /// let values: Vec<f64> = (0..200).map(|i| {
    ///     if i < 180 { 3.0 * i as f64 } else { 3.0 * 180.0 - 2.0 * (i - 180) as f64 }
    /// }).collect();
    ///
    /// let r = Recency::auto();
    /// let (start, _end) = r.resolve_with_data(&values);
    /// // start should be near the changepoint at 180
    /// assert!(start >= 170);
    /// ```
    pub fn resolve_with_data(&self, values: &[f64]) -> (usize, usize) {
        match self {
            Recency::Auto(config) => {
                let n = values.len();
                if n < 6 {
                    return self.resolve(n);
                }
                let penalty = config.penalty.unwrap_or_else(|| (n as f64).ln());
                let result = Pelt::new(config.cost_fn)
                    .penalty(penalty)
                    .min_size(config.min_segment_length)
                    .detect(values);

                if let Some(&last_cp) = result.changepoints.last() {
                    // Ensure at least 3 observations in the window
                    let min_window = 3.min(n);
                    let start = last_cp.min(n.saturating_sub(min_window));
                    (start, n)
                } else {
                    // No changepoints found — fall back
                    Recency::Fraction(config.fallback_fraction).resolve(n)
                }
            }
            _ => self.resolve(values.len()),
        }
    }

    /// Returns details about the last changepoint detection (for diagnostics).
    ///
    /// Only meaningful for `Recency::Auto`. Returns `None` for other variants.
    pub fn detect_changepoints(&self, values: &[f64]) -> Option<Vec<usize>> {
        match self {
            Recency::Auto(config) => {
                let n = values.len();
                if n < 6 {
                    return Some(Vec::new());
                }
                let penalty = config.penalty.unwrap_or_else(|| (n as f64).ln());
                let result = Pelt::new(config.cost_fn)
                    .penalty(penalty)
                    .min_size(config.min_segment_length)
                    .detect(values);
                Some(result.changepoints)
            }
            _ => None,
        }
    }
}

/// A composable seasonal component.
///
/// Implementations model periodic patterns at a given period. Each component
/// can be fitted to data, used to predict future seasonal values, and queried
/// for named features.
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::seasonality::{DummySeasonality, SeasonalComponent};
///
/// let mut comp = DummySeasonality::new();
/// comp.fit_seasonal(&values, 12)?;
/// let fitted = comp.fitted_seasonal();
/// let forecast = comp.predict_seasonal(24);
/// let features = comp.seasonal_features();
/// ```
pub trait SeasonalComponent {
    /// Fit the seasonal component to `values` with the given `period`.
    fn fit_seasonal(&mut self, values: &[f64], period: usize) -> Result<()>;

    /// Return the fitted seasonal values (in-sample, same length as training data).
    fn fitted_seasonal(&self) -> &[f64];

    /// Predict the seasonal component for the next `n_ahead` steps.
    fn predict_seasonal(&self, n_ahead: usize) -> Vec<f64>;

    /// Extract named features from the fitted component.
    ///
    /// Returns `(name, value)` pairs suitable for ML pipelines and the
    /// feature-extraction module. Feature names are prefixed with the
    /// component name (e.g. `"dummy_seasonal_strength"`).
    fn seasonal_features(&self) -> Vec<(&str, f64)>;

    /// Human-readable name of this component.
    fn seasonal_name(&self) -> &str;

    /// Number of free parameters in the fitted model (for IC-based selection).
    fn n_params(&self) -> usize {
        0
    }
}

/// A composable trend component.
///
/// Implementations extract and model the trend in a time series.
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::seasonality::{PiecewiseLinearTrend, TrendComponent};
///
/// let mut comp = PiecewiseLinearTrend::new();
/// comp.fit_trend(&values)?;
/// let fitted = comp.fitted_trend();
/// let forecast = comp.predict_trend(12);
/// let features = comp.trend_features();
/// ```
pub trait TrendComponent {
    /// Fit the trend component to `values`.
    fn fit_trend(&mut self, values: &[f64]) -> Result<()>;

    /// Return the fitted trend values (in-sample, same length as training data).
    fn fitted_trend(&self) -> &[f64];

    /// Predict the trend for the next `n_ahead` steps.
    fn predict_trend(&self, n_ahead: usize) -> Vec<f64>;

    /// Extract named features from the fitted component.
    ///
    /// Returns `(name, value)` pairs suitable for ML pipelines.
    fn trend_features(&self) -> Vec<(&str, f64)>;

    /// Human-readable name of this component.
    fn trend_name(&self) -> &str;

    /// Number of free parameters in the fitted model (for IC-based selection).
    fn n_params(&self) -> usize {
        0
    }
}
