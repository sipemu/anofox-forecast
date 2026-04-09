//! Feature extraction functions for JavaScript.
//!
//! Exposes statistical features from anofox-forecast for use in JS/WASM,
//! including the [`FeatureGenerator`] for deterministic calendar/Fourier
//! regressors and the [`generateFutureTimestamps`] standalone helper.

use wasm_bindgen::prelude::*;

use anofox_forecast::core::{generate_future_timestamps as gen_future_ts, Frequency};
use anofox_forecast::features::{
    autocorrelation as acf_fn, basic, change, complexity, counting, distribution, entropy,
    partial_autocorrelation as pacf_fn, trend, AdvancedFeature, BinaryIndicator,
    FeatureGenerator as InnerFeatureGenerator, TimeComponent,
};

/// Compute the autocorrelation of a time series at a specific lag.
///
/// @param values - Array of numeric values
/// @param lag - Lag value
/// @returns Autocorrelation coefficient at the given lag
#[wasm_bindgen]
pub fn autocorrelation(values: &[f64], lag: usize) -> f64 {
    acf_fn(values, lag)
}

/// Compute the partial autocorrelation of a time series at a specific lag.
///
/// Uses the Durbin-Levinson algorithm.
///
/// @param values - Array of numeric values
/// @param lag - Lag value (must be >= 1)
/// @returns Partial autocorrelation coefficient at the given lag
#[wasm_bindgen(js_name = partialAutocorrelation)]
pub fn partial_autocorrelation(values: &[f64], lag: usize) -> f64 {
    pacf_fn(values, lag)
}

/// Compute the arithmetic mean of a time series.
///
/// @param values - Array of numeric values
/// @returns Arithmetic mean, or NaN if empty
#[wasm_bindgen]
pub fn mean(values: &[f64]) -> f64 {
    basic::mean(values)
}

/// Compute the population variance of a time series.
///
/// Uses population formula (n denominator) matching tsfresh.
///
/// @param values - Array of numeric values
/// @returns Population variance, or NaN if empty
#[wasm_bindgen]
pub fn variance(values: &[f64]) -> f64 {
    basic::variance(values)
}

/// Compute the skewness (third standardized moment) of a time series.
///
/// Measures the asymmetry of the distribution.
///
/// @param values - Array of numeric values
/// @returns Skewness, or NaN if fewer than 3 values
#[wasm_bindgen]
pub fn skewness(values: &[f64]) -> f64 {
    distribution::skewness(values)
}

/// Compute the excess kurtosis (fourth standardized moment) of a time series.
///
/// Measures the "tailedness" of the distribution. A normal distribution has
/// excess kurtosis of 0.
///
/// @param values - Array of numeric values
/// @returns Excess kurtosis, or NaN if fewer than 4 values
#[wasm_bindgen]
pub fn kurtosis(values: &[f64]) -> f64 {
    distribution::kurtosis(values)
}

/// Compute the approximate entropy of a time series.
///
/// Measures the complexity/regularity of a time series, including self-matches.
///
/// @param values - Array of numeric values
/// @param m - Embedding dimension (typically 2)
/// @param r - Tolerance (typically 0.2 * standard deviation)
/// @returns Approximate entropy, or NaN if insufficient data
#[wasm_bindgen(js_name = approximateEntropy)]
pub fn approximate_entropy(values: &[f64], m: usize, r: f64) -> f64 {
    entropy::approximate_entropy(values, m, r)
}

/// Compute the sample entropy of a time series.
///
/// Measures the complexity/regularity of a time series. Lower values indicate
/// more regularity. Unlike approximate entropy, excludes self-matches.
///
/// @param values - Array of numeric values
/// @param m - Embedding dimension (typically 2)
/// @param r - Tolerance (typically 0.2 * standard deviation)
/// @returns Sample entropy, or NaN if insufficient data
#[wasm_bindgen(js_name = sampleEntropy)]
pub fn sample_entropy(values: &[f64], m: usize, r: f64) -> f64 {
    entropy::sample_entropy(values, m, r)
}

// ── FeatureGenerator WASM wrapper ────────────────────────────────────────────

/// Parse a string into a [`TimeComponent`] enum variant.
fn parse_time_component(s: &str) -> Result<TimeComponent, JsError> {
    match s {
        "month" => Ok(TimeComponent::Month),
        "quarter" => Ok(TimeComponent::Quarter),
        "semester" => Ok(TimeComponent::Semester),
        "week_of_year" => Ok(TimeComponent::WeekOfYear),
        "day_of_week" => Ok(TimeComponent::DayOfWeek),
        "day_of_month" => Ok(TimeComponent::DayOfMonth),
        "day_of_year" => Ok(TimeComponent::DayOfYear),
        "hour" => Ok(TimeComponent::Hour),
        "minute" => Ok(TimeComponent::Minute),
        "second" => Ok(TimeComponent::Second),
        _ => Err(JsError::new(&format!(
            "unknown time component: '{}' (expected month, quarter, semester, \
             week_of_year, day_of_week, day_of_month, day_of_year, hour, minute, second)",
            s
        ))),
    }
}

/// Parse a string into a [`BinaryIndicator`] enum variant.
fn parse_binary_indicator(s: &str) -> Result<BinaryIndicator, JsError> {
    match s {
        "weekend" => Ok(BinaryIndicator::Weekend),
        "month_start" => Ok(BinaryIndicator::MonthStart),
        "month_end" => Ok(BinaryIndicator::MonthEnd),
        "quarter_start" => Ok(BinaryIndicator::QuarterStart),
        "quarter_end" => Ok(BinaryIndicator::QuarterEnd),
        "year_start" => Ok(BinaryIndicator::YearStart),
        "year_end" => Ok(BinaryIndicator::YearEnd),
        _ => Err(JsError::new(&format!(
            "unknown binary indicator: '{}' (expected weekend, month_start, month_end, \
             quarter_start, quarter_end, year_start, year_end)",
            s
        ))),
    }
}

/// Parse a string into an [`AdvancedFeature`] enum variant.
fn parse_advanced_feature(s: &str) -> Result<AdvancedFeature, JsError> {
    match s {
        "leap_year" => Ok(AdvancedFeature::LeapYear),
        "days_in_month" => Ok(AdvancedFeature::DaysInMonth),
        _ => Err(JsError::new(&format!(
            "unknown advanced feature: '{}' (expected leap_year, days_in_month)",
            s
        ))),
    }
}

/// Convert an array of millisecond-epoch f64 values into `DateTime<Utc>`.
fn ms_to_datetimes(timestamps_ms: &[f64]) -> Vec<chrono::DateTime<chrono::Utc>> {
    timestamps_ms
        .iter()
        .filter_map(|&ms| {
            let secs = (ms / 1000.0) as i64;
            let nanos = ((ms % 1000.0) * 1_000_000.0) as u32;
            chrono::DateTime::from_timestamp(secs, nanos)
        })
        .collect()
}

/// Deterministic feature generator for timestamp-based regressors.
///
/// Builds Fourier terms, one-hot calendar indicators, cyclical sin/cos
/// encodings, binary flags and advanced numeric features from timestamps
/// alone. Because features are purely deterministic they are safe for
/// cross-validation and can be shared across models.
///
/// Use the builder pattern to configure features, then call `generate()`
/// to produce a JS object mapping feature names to `Float64Array` values.
///
/// @example
/// ```js
/// const gen = new FeatureGenerator();
/// gen.fourier(7, 3);
/// gen.dayOfWeek();
/// gen.cyclical("month");
/// gen.binary("weekend");
/// gen.advanced("days_in_month");
///
/// const features = gen.generate(timestamps); // { fourier_7_sin1: [...], ... }
/// const names = gen.featureNames(); // sorted array of column names
/// ```
#[wasm_bindgen]
pub struct FeatureGenerator {
    inner: InnerFeatureGenerator,
}

#[wasm_bindgen]
impl FeatureGenerator {
    /// Create an empty feature generator.
    #[wasm_bindgen(constructor)]
    pub fn new() -> FeatureGenerator {
        FeatureGenerator {
            inner: InnerFeatureGenerator::new(),
        }
    }

    /// Add Fourier terms for a given period and order.
    ///
    /// Produces `2 * order` columns (`fourier_{period}_sin{k}`,
    /// `fourier_{period}_cos{k}` for k in 1..=order).
    ///
    /// @param period - Period in observation units (e.g. 7 for weekly in daily data)
    /// @param order  - Number of harmonics
    pub fn fourier(&mut self, period: usize, order: usize) {
        self.inner = std::mem::take(&mut self.inner).fourier(period, order);
    }

    /// Add day-of-week one-hot indicators (Mon-Sat, Sunday dropped).
    #[wasm_bindgen(js_name = dayOfWeek)]
    pub fn day_of_week(&mut self) {
        self.inner = std::mem::take(&mut self.inner).day_of_week();
    }

    /// Add month-of-year one-hot indicators (Feb-Dec, January dropped).
    #[wasm_bindgen(js_name = monthOfYear)]
    pub fn month_of_year(&mut self) {
        self.inner = std::mem::take(&mut self.inner).month_of_year();
    }

    /// Add quarter one-hot indicators (Q2-Q4, Q1 dropped).
    pub fn quarter(&mut self) {
        self.inner = std::mem::take(&mut self.inner).quarter();
    }

    /// Add cyclical sin/cos encoding of a time component.
    ///
    /// Produces 2 columns: `{name}_sin` and `{name}_cos`.
    ///
    /// @param component - One of: "month", "quarter", "semester",
    ///   "week_of_year", "day_of_week", "day_of_month", "day_of_year",
    ///   "hour", "minute", "second"
    pub fn cyclical(&mut self, component: &str) -> Result<(), JsError> {
        let tc = parse_time_component(component)?;
        self.inner = std::mem::take(&mut self.inner).cyclical(tc);
        Ok(())
    }

    /// Add a binary (0/1) calendar indicator.
    ///
    /// @param indicator - One of: "weekend", "month_start", "month_end",
    ///   "quarter_start", "quarter_end", "year_start", "year_end"
    pub fn binary(&mut self, indicator: &str) -> Result<(), JsError> {
        let bi = parse_binary_indicator(indicator)?;
        self.inner = std::mem::take(&mut self.inner).binary(bi);
        Ok(())
    }

    /// Add an advanced numeric calendar feature.
    ///
    /// @param feature - One of: "leap_year", "days_in_month"
    pub fn advanced(&mut self, feature: &str) -> Result<(), JsError> {
        let af = parse_advanced_feature(feature)?;
        self.inner = std::mem::take(&mut self.inner).advanced(af);
        Ok(())
    }

    /// Generate features for the given timestamps.
    ///
    /// @param timestamps - Array of timestamps as milliseconds since Unix epoch
    /// @returns Object mapping feature names to Float64Array values
    pub fn generate(&self, timestamps: &[f64]) -> Result<JsValue, JsError> {
        let datetimes = ms_to_datetimes(timestamps);
        let features = self.inner.generate(&datetimes);
        serde_wasm_bindgen::to_value(&features).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Return the sorted list of column names this generator will produce.
    ///
    /// @returns Array of feature name strings
    #[wasm_bindgen(js_name = featureNames)]
    pub fn feature_names(&self) -> Result<JsValue, JsError> {
        let names = self.inner.feature_names();
        serde_wasm_bindgen::to_value(&names).map_err(|e| JsError::new(&e.to_string()))
    }
}

impl Default for FeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ── generateFutureTimestamps standalone function ────────────────────────────

/// Generate future timestamps starting after a given last timestamp.
///
/// @param lastTimestampMs - Last known timestamp as milliseconds since epoch
/// @param frequency - Polars-style frequency string: "1d", "1w", "1mo",
///   "1q", "1y", "1h", "30m", "30s", etc.
/// @param horizon - Number of future timestamps to generate
/// @returns Float64Array of timestamps as milliseconds since epoch
#[wasm_bindgen(js_name = generateFutureTimestamps)]
pub fn generate_future_timestamps(
    last_timestamp_ms: f64,
    frequency: &str,
    horizon: usize,
) -> Result<Vec<f64>, JsError> {
    let secs = (last_timestamp_ms / 1000.0) as i64;
    let nanos = ((last_timestamp_ms % 1000.0) * 1_000_000.0) as u32;
    let last = chrono::DateTime::from_timestamp(secs, nanos)
        .ok_or_else(|| JsError::new("invalid last timestamp"))?;

    let freq = Frequency::parse(frequency).map_err(|e| JsError::new(&e.to_string()))?;

    let future = gen_future_ts(&last, &freq, horizon);
    Ok(future
        .iter()
        .map(|dt| dt.timestamp_millis() as f64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_autocorrelation() {
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let acf1 = autocorrelation(&series, 1);
        assert!(acf1 > 0.8);
    }

    #[wasm_bindgen_test]
    fn test_partial_autocorrelation() {
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pacf1 = partial_autocorrelation(&series, 1);
        assert!(!pacf1.is_nan());
    }

    #[wasm_bindgen_test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&values) - 3.0).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance(&values) - 2.0).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_skewness() {
        // Symmetric distribution should have skewness near 0
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!(skewness(&values).abs() < 0.1);
    }

    #[wasm_bindgen_test]
    fn test_kurtosis() {
        let values: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let k = kurtosis(&values);
        assert!(!k.is_nan());
    }

    #[wasm_bindgen_test]
    fn test_approximate_entropy() {
        let values: Vec<f64> = (0..50).map(|i| (i as f64 * 0.5).sin()).collect();
        let ae = approximate_entropy(&values, 2, 0.2);
        assert!(!ae.is_nan());
    }

    #[wasm_bindgen_test]
    fn test_sample_entropy() {
        let values: Vec<f64> = (0..100)
            .map(|i| ((i % 10) as f64 * std::f64::consts::PI / 5.0).sin())
            .collect();
        let se = sample_entropy(&values, 2, 0.2);
        assert!(!se.is_nan());
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Extended feature library — thin wrappers over `anofox_forecast::features`.
// ─────────────────────────────────────────────────────────────────────────

// ── basic ───────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = absEnergy)]
pub fn abs_energy(values: &[f64]) -> f64 {
    basic::abs_energy(values)
}

#[wasm_bindgen(js_name = absoluteMaximum)]
pub fn absolute_maximum(values: &[f64]) -> f64 {
    basic::absolute_maximum(values)
}

#[wasm_bindgen(js_name = absoluteSumOfChanges)]
pub fn absolute_sum_of_changes(values: &[f64]) -> f64 {
    basic::absolute_sum_of_changes(values)
}

#[wasm_bindgen]
pub fn maximum(values: &[f64]) -> f64 {
    basic::maximum(values)
}

#[wasm_bindgen]
pub fn minimum(values: &[f64]) -> f64 {
    basic::minimum(values)
}

#[wasm_bindgen]
pub fn median(values: &[f64]) -> f64 {
    basic::median(values)
}

#[wasm_bindgen(js_name = meanAbsChange)]
pub fn mean_abs_change(values: &[f64]) -> f64 {
    basic::mean_abs_change(values)
}

#[wasm_bindgen(js_name = meanChange)]
pub fn mean_change(values: &[f64]) -> f64 {
    basic::mean_change(values)
}

#[wasm_bindgen(js_name = meanSecondDerivativeCentral)]
pub fn mean_second_derivative_central(values: &[f64]) -> f64 {
    basic::mean_second_derivative_central(values)
}

#[wasm_bindgen(js_name = meanNAbsoluteMax)]
pub fn mean_n_absolute_max(values: &[f64], n: usize) -> f64 {
    basic::mean_n_absolute_max(values, n)
}

#[wasm_bindgen(js_name = rootMeanSquare)]
pub fn root_mean_square(values: &[f64]) -> f64 {
    basic::root_mean_square(values)
}

#[wasm_bindgen(js_name = standardDeviation)]
pub fn standard_deviation(values: &[f64]) -> f64 {
    basic::standard_deviation(values)
}

#[wasm_bindgen(js_name = sumValues)]
pub fn sum_values(values: &[f64]) -> f64 {
    basic::sum_values(values)
}

#[wasm_bindgen(js_name = varianceSample)]
pub fn variance_sample(values: &[f64]) -> f64 {
    basic::variance_sample(values)
}

// ── distribution ────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn quantile(values: &[f64], q: f64) -> f64 {
    distribution::quantile(values, q)
}

#[wasm_bindgen(js_name = variationCoefficient)]
pub fn variation_coefficient(values: &[f64]) -> f64 {
    distribution::variation_coefficient(values)
}

#[wasm_bindgen(js_name = largeStandardDeviation)]
pub fn large_standard_deviation(values: &[f64], r: f64) -> bool {
    distribution::large_standard_deviation(values, r)
}

#[wasm_bindgen(js_name = varianceLargerThanStandardDeviation)]
pub fn variance_larger_than_standard_deviation(values: &[f64]) -> bool {
    distribution::variance_larger_than_standard_deviation(values)
}

#[wasm_bindgen(js_name = symmetryLooking)]
pub fn symmetry_looking(values: &[f64], r: f64) -> bool {
    distribution::symmetry_looking(values, r)
}

#[wasm_bindgen(js_name = ratioBeyondRSigma)]
pub fn ratio_beyond_r_sigma(values: &[f64], r: f64) -> f64 {
    distribution::ratio_beyond_r_sigma(values, r)
}

// ── entropy ─────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = permutationEntropy)]
pub fn permutation_entropy(values: &[f64], order: usize, delay: usize) -> f64 {
    entropy::permutation_entropy(values, order, delay)
}

#[wasm_bindgen(js_name = permutationEntropyNormalized)]
pub fn permutation_entropy_normalized(values: &[f64], order: usize, delay: usize) -> f64 {
    entropy::permutation_entropy_normalized(values, order, delay)
}

#[wasm_bindgen(js_name = binnedEntropy)]
pub fn binned_entropy(values: &[f64], max_bins: usize) -> f64 {
    entropy::binned_entropy(values, max_bins)
}

#[wasm_bindgen(js_name = fourierEntropy)]
pub fn fourier_entropy(values: &[f64]) -> f64 {
    entropy::fourier_entropy(values)
}

// ── complexity ──────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = cidCe)]
pub fn cid_ce(values: &[f64], normalize: bool) -> f64 {
    complexity::cid_ce(values, normalize)
}

#[wasm_bindgen]
pub fn c3(values: &[f64], lag: usize) -> f64 {
    complexity::c3(values, lag)
}

#[wasm_bindgen(js_name = lempelZivComplexity)]
pub fn lempel_ziv_complexity(values: &[f64], bins: usize) -> f64 {
    complexity::lempel_ziv_complexity(values, bins)
}

#[wasm_bindgen(js_name = lempelZivComplexityBinary)]
pub fn lempel_ziv_complexity_binary(values: &[f64]) -> f64 {
    complexity::lempel_ziv_complexity_binary(values)
}

// ── counting ────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = countAbove)]
pub fn count_above(values: &[f64], threshold: f64) -> usize {
    counting::count_above(values, threshold)
}

#[wasm_bindgen(js_name = countBelow)]
pub fn count_below(values: &[f64], threshold: f64) -> usize {
    counting::count_below(values, threshold)
}

#[wasm_bindgen(js_name = countAboveMean)]
pub fn count_above_mean(values: &[f64]) -> usize {
    counting::count_above_mean(values)
}

#[wasm_bindgen(js_name = countBelowMean)]
pub fn count_below_mean(values: &[f64]) -> usize {
    counting::count_below_mean(values)
}

#[wasm_bindgen(js_name = numberPeaks)]
pub fn number_peaks(values: &[f64], support: usize) -> usize {
    counting::number_peaks(values, support)
}

#[wasm_bindgen(js_name = numberCrossingM)]
pub fn number_crossing_m(values: &[f64], m: f64) -> usize {
    counting::number_crossing_m(values, m)
}

#[wasm_bindgen(js_name = longestStrikeAboveMean)]
pub fn longest_strike_above_mean(values: &[f64]) -> usize {
    counting::longest_strike_above_mean(values)
}

#[wasm_bindgen(js_name = longestStrikeBelowMean)]
pub fn longest_strike_below_mean(values: &[f64]) -> usize {
    counting::longest_strike_below_mean(values)
}

#[wasm_bindgen(js_name = firstLocationOfMaximum)]
pub fn first_location_of_maximum(values: &[f64]) -> f64 {
    counting::first_location_of_maximum(values)
}

#[wasm_bindgen(js_name = firstLocationOfMinimum)]
pub fn first_location_of_minimum(values: &[f64]) -> f64 {
    counting::first_location_of_minimum(values)
}

#[wasm_bindgen(js_name = lastLocationOfMaximum)]
pub fn last_location_of_maximum(values: &[f64]) -> f64 {
    counting::last_location_of_maximum(values)
}

#[wasm_bindgen(js_name = lastLocationOfMinimum)]
pub fn last_location_of_minimum(values: &[f64]) -> f64 {
    counting::last_location_of_minimum(values)
}

#[wasm_bindgen(js_name = hasDuplicate)]
pub fn has_duplicate(values: &[f64]) -> bool {
    counting::has_duplicate(values)
}

#[wasm_bindgen(js_name = hasDuplicateMax)]
pub fn has_duplicate_max(values: &[f64]) -> bool {
    counting::has_duplicate_max(values)
}

#[wasm_bindgen(js_name = hasDuplicateMin)]
pub fn has_duplicate_min(values: &[f64]) -> bool {
    counting::has_duplicate_min(values)
}

#[wasm_bindgen(js_name = indexMassQuantile)]
pub fn index_mass_quantile(values: &[f64], q: f64) -> f64 {
    counting::index_mass_quantile(values, q)
}

#[wasm_bindgen(js_name = valueCount)]
pub fn value_count(values: &[f64], value: f64) -> usize {
    counting::value_count(values, value)
}

#[wasm_bindgen(js_name = rangeCount)]
pub fn range_count(values: &[f64], min: f64, max: f64) -> usize {
    counting::range_count(values, min, max)
}

// ── change ──────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = energyRatioByChunks)]
pub fn energy_ratio_by_chunks(values: &[f64], n_chunks: usize, chunk_index: usize) -> f64 {
    change::energy_ratio_by_chunks(values, n_chunks, chunk_index)
}

#[wasm_bindgen(js_name = percentageOfReoccurringDatapointsToAllDatapoints)]
pub fn percentage_of_reoccurring_datapoints_to_all_datapoints(values: &[f64]) -> f64 {
    change::percentage_of_reoccurring_datapoints_to_all_datapoints(values)
}

#[wasm_bindgen(js_name = percentageOfReoccurringValuesToAllValues)]
pub fn percentage_of_reoccurring_values_to_all_values(values: &[f64]) -> f64 {
    change::percentage_of_reoccurring_values_to_all_values(values)
}

#[wasm_bindgen(js_name = ratioValueNumberToTimeSeriesLength)]
pub fn ratio_value_number_to_time_series_length(values: &[f64]) -> f64 {
    change::ratio_value_number_to_time_series_length(values)
}

#[wasm_bindgen(js_name = sumOfReoccurringDataPoints)]
pub fn sum_of_reoccurring_data_points(values: &[f64]) -> f64 {
    change::sum_of_reoccurring_data_points(values)
}

#[wasm_bindgen(js_name = sumOfReoccurringValues)]
pub fn sum_of_reoccurring_values(values: &[f64]) -> f64 {
    change::sum_of_reoccurring_values(values)
}

// ── autocorrelation (extra) ─────────────────────────────────────────────

#[wasm_bindgen(js_name = aggAutocorrelation)]
pub fn agg_autocorrelation(values: &[f64], max_lag: usize, agg_func: &str) -> f64 {
    anofox_forecast::features::agg_autocorrelation(values, max_lag, agg_func)
}

#[wasm_bindgen(js_name = timeReversalAsymmetryStatistic)]
pub fn time_reversal_asymmetry_statistic(values: &[f64], lag: usize) -> f64 {
    anofox_forecast::features::time_reversal_asymmetry_statistic(values, lag)
}

// ── trend / AR ──────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = aggLinearTrend)]
pub fn agg_linear_trend(values: &[f64], chunk_len: usize, agg_func: &str, attribute: &str) -> f64 {
    trend::agg_linear_trend(values, chunk_len, agg_func, attribute)
}

#[wasm_bindgen(js_name = arCoefficient)]
pub fn ar_coefficient(values: &[f64], k: usize, coeff: usize) -> f64 {
    trend::ar_coefficient(values, k, coeff)
}

#[wasm_bindgen(js_name = arCoefficientYuleWalker)]
pub fn ar_coefficient_yule_walker(values: &[f64], k: usize) -> f64 {
    trend::ar_coefficient_yule_walker(values, k)
}

#[wasm_bindgen(js_name = augmentedDickeyFuller)]
pub fn augmented_dickey_fuller(values: &[f64]) -> f64 {
    trend::augmented_dickey_fuller(values)
}
