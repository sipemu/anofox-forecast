//! Feature factory for batch computation of time series features.
//!
//! Provides a builder-style API to select and compute multiple features at once,
//! returning results as a `HashMap<String, f64>` compatible with `selection::select_features`.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::features::factory::{Feature, FeatureFactory};
//!
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//!
//! // Pick specific features
//! let features = FeatureFactory::new()
//!     .feature(Feature::Mean)
//!     .feature(Feature::Variance)
//!     .feature(Feature::Skewness)
//!     .feature(Feature::Autocorrelation { lag: 1 })
//!     .compute(&series);
//!
//! assert!(features.contains_key("mean"));
//! assert!(features.contains_key("autocorrelation_1"));
//!
//! // Use a preset
//! let all = FeatureFactory::all().compute(&series);
//! assert!(all.len() > 50);
//! ```

use std::collections::HashMap;

/// Specifies a single feature to compute.
#[derive(Debug, Clone)]
pub enum Feature {
    // ── Basic ──
    Mean,
    Median,
    Variance,
    VarianceSample,
    StandardDeviation,
    Minimum,
    Maximum,
    AbsEnergy,
    AbsoluteMaximum,
    AbsoluteSumOfChanges,
    Length,
    MeanAbsChange,
    MeanChange,
    MeanSecondDerivativeCentral,
    MeanNAbsoluteMax { n: usize },
    RootMeanSquare,
    SumValues,

    // ── Distribution ──
    Skewness,
    Kurtosis,
    Quantile { q: f64 },
    LargeStandardDeviation { r: f64 },
    VarianceLargerThanStd,
    VariationCoefficient,
    SymmetryLooking { r: f64 },
    RatioBeyondRSigma { r: f64 },

    // ── Autocorrelation ──
    Autocorrelation { lag: usize },
    PartialAutocorrelation { lag: usize },
    AggAutocorrelation { max_lag: usize, agg_func: String },
    TimeReversalAsymmetry { lag: usize },

    // ── Counting ──
    CountAboveMean,
    CountBelowMean,
    NumberPeaks { support: usize },
    NumberCrossingMean,
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    FirstLocationOfMaximum,
    FirstLocationOfMinimum,
    LastLocationOfMaximum,
    LastLocationOfMinimum,
    HasDuplicate,
    HasDuplicateMax,
    HasDuplicateMin,
    IndexMassQuantile { q: f64 },

    // ── Entropy ──
    SampleEntropy { m: usize, r: f64 },
    ApproximateEntropy { m: usize, r: f64 },
    PermutationEntropy { order: usize, delay: usize },
    BinnedEntropy { max_bins: usize },
    FourierEntropy,

    // ── Complexity ──
    CidCe { normalize: bool },
    C3 { lag: usize },
    LempelZivComplexity { bins: usize },

    // ── Trend ──
    LinearTrendSlope,
    LinearTrendIntercept,
    LinearTrendRSquared,
    LinearTrendPValue,
    AugmentedDickeyFuller,
    ArCoefficient { k: usize, coeff: usize },

    // ── Change ──
    EnergyRatioByChunks { n_chunks: usize, chunk_index: usize },
    PercentageReoccurringDatapoints,
    PercentageReoccurringValues,
    RatioValueNumberToLength,
    SumOfReoccurringDataPoints,
    SumOfReoccurringValues,
}

impl Feature {
    /// Returns the canonical name used as the HashMap key.
    pub fn name(&self) -> String {
        match self {
            // Basic
            Feature::Mean => "mean".into(),
            Feature::Median => "median".into(),
            Feature::Variance => "variance".into(),
            Feature::VarianceSample => "variance_sample".into(),
            Feature::StandardDeviation => "standard_deviation".into(),
            Feature::Minimum => "minimum".into(),
            Feature::Maximum => "maximum".into(),
            Feature::AbsEnergy => "abs_energy".into(),
            Feature::AbsoluteMaximum => "absolute_maximum".into(),
            Feature::AbsoluteSumOfChanges => "absolute_sum_of_changes".into(),
            Feature::Length => "length".into(),
            Feature::MeanAbsChange => "mean_abs_change".into(),
            Feature::MeanChange => "mean_change".into(),
            Feature::MeanSecondDerivativeCentral => "mean_second_derivative_central".into(),
            Feature::MeanNAbsoluteMax { n } => format!("mean_n_absolute_max_{n}"),
            Feature::RootMeanSquare => "root_mean_square".into(),
            Feature::SumValues => "sum_values".into(),

            // Distribution
            Feature::Skewness => "skewness".into(),
            Feature::Kurtosis => "kurtosis".into(),
            Feature::Quantile { q } => format!("quantile_{q}"),
            Feature::LargeStandardDeviation { r } => {
                format!("large_standard_deviation_{r}")
            }
            Feature::VarianceLargerThanStd => "variance_larger_than_std".into(),
            Feature::VariationCoefficient => "variation_coefficient".into(),
            Feature::SymmetryLooking { r } => format!("symmetry_looking_{r}"),
            Feature::RatioBeyondRSigma { r } => format!("ratio_beyond_r_sigma_{r}"),

            // Autocorrelation
            Feature::Autocorrelation { lag } => format!("autocorrelation_{lag}"),
            Feature::PartialAutocorrelation { lag } => {
                format!("partial_autocorrelation_{lag}")
            }
            Feature::AggAutocorrelation { max_lag, agg_func } => {
                format!("agg_autocorrelation_{agg_func}_{max_lag}")
            }
            Feature::TimeReversalAsymmetry { lag } => {
                format!("time_reversal_asymmetry_{lag}")
            }

            // Counting
            Feature::CountAboveMean => "count_above_mean".into(),
            Feature::CountBelowMean => "count_below_mean".into(),
            Feature::NumberPeaks { support } => format!("number_peaks_{support}"),
            Feature::NumberCrossingMean => "number_crossing_mean".into(),
            Feature::LongestStrikeAboveMean => "longest_strike_above_mean".into(),
            Feature::LongestStrikeBelowMean => "longest_strike_below_mean".into(),
            Feature::FirstLocationOfMaximum => "first_location_of_maximum".into(),
            Feature::FirstLocationOfMinimum => "first_location_of_minimum".into(),
            Feature::LastLocationOfMaximum => "last_location_of_maximum".into(),
            Feature::LastLocationOfMinimum => "last_location_of_minimum".into(),
            Feature::HasDuplicate => "has_duplicate".into(),
            Feature::HasDuplicateMax => "has_duplicate_max".into(),
            Feature::HasDuplicateMin => "has_duplicate_min".into(),
            Feature::IndexMassQuantile { q } => format!("index_mass_quantile_{q}"),

            // Entropy
            Feature::SampleEntropy { m, r } => format!("sample_entropy_{m}_{r}"),
            Feature::ApproximateEntropy { m, r } => format!("approximate_entropy_{m}_{r}"),
            Feature::PermutationEntropy { order, delay } => {
                format!("permutation_entropy_{order}_{delay}")
            }
            Feature::BinnedEntropy { max_bins } => format!("binned_entropy_{max_bins}"),
            Feature::FourierEntropy => "fourier_entropy".into(),

            // Complexity
            Feature::CidCe { normalize } => format!("cid_ce_{normalize}"),
            Feature::C3 { lag } => format!("c3_{lag}"),
            Feature::LempelZivComplexity { bins } => {
                format!("lempel_ziv_complexity_{bins}")
            }

            // Trend
            Feature::LinearTrendSlope => "linear_trend_slope".into(),
            Feature::LinearTrendIntercept => "linear_trend_intercept".into(),
            Feature::LinearTrendRSquared => "linear_trend_r_squared".into(),
            Feature::LinearTrendPValue => "linear_trend_p_value".into(),
            Feature::AugmentedDickeyFuller => "augmented_dickey_fuller".into(),
            Feature::ArCoefficient { k, coeff } => format!("ar_coefficient_{k}_{coeff}"),

            // Change
            Feature::EnergyRatioByChunks {
                n_chunks,
                chunk_index,
            } => format!("energy_ratio_by_chunks_{n_chunks}_{chunk_index}"),
            Feature::PercentageReoccurringDatapoints => "percentage_reoccurring_datapoints".into(),
            Feature::PercentageReoccurringValues => "percentage_reoccurring_values".into(),
            Feature::RatioValueNumberToLength => "ratio_value_number_to_length".into(),
            Feature::SumOfReoccurringDataPoints => "sum_of_reoccurring_data_points".into(),
            Feature::SumOfReoccurringValues => "sum_of_reoccurring_values".into(),
        }
    }

    /// Computes this feature on the given series.
    pub fn compute(&self, series: &[f64]) -> f64 {
        use super::*;

        match self {
            // Basic
            Feature::Mean => basic::mean(series),
            Feature::Median => basic::median(series),
            Feature::Variance => basic::variance(series),
            Feature::VarianceSample => basic::variance_sample(series),
            Feature::StandardDeviation => basic::standard_deviation(series),
            Feature::Minimum => basic::minimum(series),
            Feature::Maximum => basic::maximum(series),
            Feature::AbsEnergy => basic::abs_energy(series),
            Feature::AbsoluteMaximum => basic::absolute_maximum(series),
            Feature::AbsoluteSumOfChanges => basic::absolute_sum_of_changes(series),
            Feature::Length => basic::length(series),
            Feature::MeanAbsChange => basic::mean_abs_change(series),
            Feature::MeanChange => basic::mean_change(series),
            Feature::MeanSecondDerivativeCentral => basic::mean_second_derivative_central(series),
            Feature::MeanNAbsoluteMax { n } => basic::mean_n_absolute_max(series, *n),
            Feature::RootMeanSquare => basic::root_mean_square(series),
            Feature::SumValues => basic::sum_values(series),

            // Distribution
            Feature::Skewness => distribution::skewness(series),
            Feature::Kurtosis => distribution::kurtosis(series),
            Feature::Quantile { q } => distribution::quantile(series, *q),
            Feature::LargeStandardDeviation { r } => {
                if distribution::large_standard_deviation(series, *r) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::VarianceLargerThanStd => {
                if distribution::variance_larger_than_standard_deviation(series) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::VariationCoefficient => distribution::variation_coefficient(series),
            Feature::SymmetryLooking { r } => {
                if distribution::symmetry_looking(series, *r) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::RatioBeyondRSigma { r } => distribution::ratio_beyond_r_sigma(series, *r),

            // Autocorrelation
            Feature::Autocorrelation { lag } => autocorrelation::autocorrelation(series, *lag),
            Feature::PartialAutocorrelation { lag } => {
                autocorrelation::partial_autocorrelation(series, *lag)
            }
            Feature::AggAutocorrelation { max_lag, agg_func } => {
                autocorrelation::agg_autocorrelation(series, *max_lag, agg_func)
            }
            Feature::TimeReversalAsymmetry { lag } => {
                autocorrelation::time_reversal_asymmetry_statistic(series, *lag)
            }

            // Counting
            Feature::CountAboveMean => counting::count_above_mean(series) as f64,
            Feature::CountBelowMean => counting::count_below_mean(series) as f64,
            Feature::NumberPeaks { support } => counting::number_peaks(series, *support) as f64,
            Feature::NumberCrossingMean => {
                if series.is_empty() {
                    0.0
                } else {
                    let m = basic::mean(series);
                    counting::number_crossing_m(series, m) as f64
                }
            }
            Feature::LongestStrikeAboveMean => counting::longest_strike_above_mean(series) as f64,
            Feature::LongestStrikeBelowMean => counting::longest_strike_below_mean(series) as f64,
            Feature::FirstLocationOfMaximum => counting::first_location_of_maximum(series),
            Feature::FirstLocationOfMinimum => counting::first_location_of_minimum(series),
            Feature::LastLocationOfMaximum => counting::last_location_of_maximum(series),
            Feature::LastLocationOfMinimum => counting::last_location_of_minimum(series),
            Feature::HasDuplicate => {
                if counting::has_duplicate(series) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::HasDuplicateMax => {
                if counting::has_duplicate_max(series) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::HasDuplicateMin => {
                if counting::has_duplicate_min(series) {
                    1.0
                } else {
                    0.0
                }
            }
            Feature::IndexMassQuantile { q } => counting::index_mass_quantile(series, *q),

            // Entropy
            Feature::SampleEntropy { m, r } => entropy::sample_entropy(series, *m, *r),
            Feature::ApproximateEntropy { m, r } => entropy::approximate_entropy(series, *m, *r),
            Feature::PermutationEntropy { order, delay } => {
                entropy::permutation_entropy(series, *order, *delay)
            }
            Feature::BinnedEntropy { max_bins } => entropy::binned_entropy(series, *max_bins),
            Feature::FourierEntropy => entropy::fourier_entropy(series),

            // Complexity
            Feature::CidCe { normalize } => complexity::cid_ce(series, *normalize),
            Feature::C3 { lag } => complexity::c3(series, *lag),
            Feature::LempelZivComplexity { bins } => {
                complexity::lempel_ziv_complexity(series, *bins)
            }

            // Trend
            Feature::LinearTrendSlope => trend::linear_trend(series).slope,
            Feature::LinearTrendIntercept => trend::linear_trend(series).intercept,
            Feature::LinearTrendRSquared => trend::linear_trend(series).r_squared,
            Feature::LinearTrendPValue => trend::linear_trend(series).p_value,
            Feature::AugmentedDickeyFuller => trend::augmented_dickey_fuller(series),
            Feature::ArCoefficient { k, coeff } => trend::ar_coefficient(series, *k, *coeff),

            // Change
            Feature::EnergyRatioByChunks {
                n_chunks,
                chunk_index,
            } => change::energy_ratio_by_chunks(series, *n_chunks, *chunk_index),
            Feature::PercentageReoccurringDatapoints => {
                change::percentage_of_reoccurring_datapoints_to_all_datapoints(series)
            }
            Feature::PercentageReoccurringValues => {
                change::percentage_of_reoccurring_values_to_all_values(series)
            }
            Feature::RatioValueNumberToLength => {
                change::ratio_value_number_to_time_series_length(series)
            }
            Feature::SumOfReoccurringDataPoints => change::sum_of_reoccurring_data_points(series),
            Feature::SumOfReoccurringValues => change::sum_of_reoccurring_values(series),
        }
    }
}

/// Batch feature computation with builder-style API.
///
/// # Example
///
/// ```
/// use anofox_forecast::features::factory::{Feature, FeatureFactory};
///
/// let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// // Category presets
/// let features = FeatureFactory::new()
///     .basic()
///     .distribution()
///     .compute(&series);
///
/// // Default set (most useful features)
/// let features = FeatureFactory::default_set().compute(&series);
///
/// // Everything
/// let features = FeatureFactory::all().compute(&series);
/// ```
#[derive(Debug, Clone)]
pub struct FeatureFactory {
    features: Vec<Feature>,
}

impl FeatureFactory {
    /// Creates an empty factory. Add features with `.add()` or category methods.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Adds a single feature.
    pub fn feature(mut self, f: Feature) -> Self {
        self.features.push(f);
        self
    }

    /// Adds multiple features at once.
    pub fn features(mut self, features: impl IntoIterator<Item = Feature>) -> Self {
        self.features.extend(features);
        self
    }

    /// Adds all basic statistical features.
    pub fn basic(self) -> Self {
        self.features([
            Feature::Mean,
            Feature::Median,
            Feature::Variance,
            Feature::VarianceSample,
            Feature::StandardDeviation,
            Feature::Minimum,
            Feature::Maximum,
            Feature::AbsEnergy,
            Feature::AbsoluteMaximum,
            Feature::AbsoluteSumOfChanges,
            Feature::Length,
            Feature::MeanAbsChange,
            Feature::MeanChange,
            Feature::MeanSecondDerivativeCentral,
            Feature::MeanNAbsoluteMax { n: 1 },
            Feature::RootMeanSquare,
            Feature::SumValues,
        ])
    }

    /// Adds all distribution features with common default parameters.
    pub fn distribution(self) -> Self {
        self.features([
            Feature::Skewness,
            Feature::Kurtosis,
            Feature::Quantile { q: 0.25 },
            Feature::Quantile { q: 0.5 },
            Feature::Quantile { q: 0.75 },
            Feature::LargeStandardDeviation { r: 0.25 },
            Feature::VarianceLargerThanStd,
            Feature::VariationCoefficient,
            Feature::SymmetryLooking { r: 0.05 },
            Feature::RatioBeyondRSigma { r: 2.0 },
            Feature::RatioBeyondRSigma { r: 2.5 },
        ])
    }

    /// Adds autocorrelation features at lags 1-5 plus aggregated statistics.
    pub fn autocorrelation(self) -> Self {
        self.features([
            Feature::Autocorrelation { lag: 1 },
            Feature::Autocorrelation { lag: 2 },
            Feature::Autocorrelation { lag: 3 },
            Feature::Autocorrelation { lag: 5 },
            Feature::Autocorrelation { lag: 10 },
            Feature::PartialAutocorrelation { lag: 1 },
            Feature::PartialAutocorrelation { lag: 2 },
            Feature::PartialAutocorrelation { lag: 5 },
            Feature::AggAutocorrelation {
                max_lag: 10,
                agg_func: "mean".into(),
            },
            Feature::AggAutocorrelation {
                max_lag: 10,
                agg_func: "var".into(),
            },
            Feature::TimeReversalAsymmetry { lag: 1 },
            Feature::TimeReversalAsymmetry { lag: 2 },
        ])
    }

    /// Adds all counting-based features.
    pub fn counting(self) -> Self {
        self.features([
            Feature::CountAboveMean,
            Feature::CountBelowMean,
            Feature::NumberPeaks { support: 1 },
            Feature::NumberPeaks { support: 3 },
            Feature::NumberCrossingMean,
            Feature::LongestStrikeAboveMean,
            Feature::LongestStrikeBelowMean,
            Feature::FirstLocationOfMaximum,
            Feature::FirstLocationOfMinimum,
            Feature::LastLocationOfMaximum,
            Feature::LastLocationOfMinimum,
            Feature::HasDuplicate,
            Feature::HasDuplicateMax,
            Feature::HasDuplicateMin,
            Feature::IndexMassQuantile { q: 0.5 },
        ])
    }

    /// Adds entropy-based features with common default parameters.
    pub fn entropy(self) -> Self {
        self.features([
            Feature::SampleEntropy { m: 2, r: 0.2 },
            Feature::ApproximateEntropy { m: 2, r: 0.2 },
            Feature::PermutationEntropy { order: 3, delay: 1 },
            Feature::BinnedEntropy { max_bins: 10 },
            Feature::FourierEntropy,
        ])
    }

    /// Adds complexity-based features.
    pub fn complexity(self) -> Self {
        self.features([
            Feature::CidCe { normalize: true },
            Feature::CidCe { normalize: false },
            Feature::C3 { lag: 1 },
            Feature::C3 { lag: 2 },
            Feature::LempelZivComplexity { bins: 10 },
        ])
    }

    /// Adds trend-based features.
    pub fn trend(self) -> Self {
        self.features([
            Feature::LinearTrendSlope,
            Feature::LinearTrendIntercept,
            Feature::LinearTrendRSquared,
            Feature::LinearTrendPValue,
            Feature::AugmentedDickeyFuller,
            Feature::ArCoefficient { k: 1, coeff: 1 },
            Feature::ArCoefficient { k: 2, coeff: 1 },
            Feature::ArCoefficient { k: 2, coeff: 2 },
        ])
    }

    /// Adds change-based features with common default parameters.
    pub fn change(self) -> Self {
        self.features([
            Feature::EnergyRatioByChunks {
                n_chunks: 10,
                chunk_index: 0,
            },
            Feature::EnergyRatioByChunks {
                n_chunks: 10,
                chunk_index: 1,
            },
            Feature::PercentageReoccurringDatapoints,
            Feature::PercentageReoccurringValues,
            Feature::RatioValueNumberToLength,
            Feature::SumOfReoccurringDataPoints,
            Feature::SumOfReoccurringValues,
        ])
    }

    /// Creates a factory with a curated default feature set.
    ///
    /// Includes the most commonly useful features for classification and clustering:
    /// basic statistics, distribution shape, autocorrelation, trend, and complexity.
    pub fn default_set() -> Self {
        Self::new().features([
            // Statistics
            Feature::Mean,
            Feature::Median,
            Feature::Variance,
            Feature::StandardDeviation,
            Feature::Minimum,
            Feature::Maximum,
            Feature::AbsEnergy,
            Feature::AbsoluteSumOfChanges,
            Feature::MeanAbsChange,
            Feature::MeanChange,
            Feature::RootMeanSquare,
            // Distribution
            Feature::Skewness,
            Feature::Kurtosis,
            Feature::Quantile { q: 0.25 },
            Feature::Quantile { q: 0.75 },
            Feature::VariationCoefficient,
            Feature::RatioBeyondRSigma { r: 2.0 },
            // Autocorrelation
            Feature::Autocorrelation { lag: 1 },
            Feature::Autocorrelation { lag: 2 },
            Feature::PartialAutocorrelation { lag: 1 },
            // Counting
            Feature::CountAboveMean,
            Feature::CountBelowMean,
            Feature::NumberPeaks { support: 1 },
            Feature::NumberCrossingMean,
            Feature::LongestStrikeAboveMean,
            Feature::LongestStrikeBelowMean,
            Feature::HasDuplicate,
            // Entropy & complexity
            Feature::BinnedEntropy { max_bins: 10 },
            Feature::CidCe { normalize: true },
            Feature::PermutationEntropy { order: 3, delay: 1 },
            // Trend
            Feature::LinearTrendSlope,
            Feature::LinearTrendRSquared,
            Feature::AugmentedDickeyFuller,
            // Change
            Feature::PercentageReoccurringDatapoints,
            Feature::RatioValueNumberToLength,
        ])
    }

    /// Creates a factory with all features using common default parameters.
    pub fn all() -> Self {
        Self::new()
            .basic()
            .distribution()
            .autocorrelation()
            .counting()
            .entropy()
            .complexity()
            .trend()
            .change()
    }

    /// Returns the number of features configured.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns true if no features are configured.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Returns the list of feature names that will be computed.
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(|f| f.name()).collect()
    }

    /// Computes all configured features on the given series.
    ///
    /// Returns a map of feature name to computed value.
    /// NaN values are included in the output (not filtered).
    pub fn compute(&self, series: &[f64]) -> HashMap<String, f64> {
        let mut result = HashMap::with_capacity(self.features.len());
        for feature in &self.features {
            result.insert(feature.name(), feature.compute(series));
        }
        result
    }
}

impl Default for FeatureFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_series() -> Vec<f64> {
        (0..100)
            .map(|i| (i as f64 * 0.2).sin() * 10.0 + 50.0)
            .collect()
    }

    // ==================== Builder API ====================

    #[test]
    fn empty_factory_produces_empty_map() {
        let result = FeatureFactory::new().compute(&[1.0, 2.0, 3.0]);
        assert!(result.is_empty());
    }

    #[test]
    fn add_single_feature() {
        let result = FeatureFactory::new()
            .feature(Feature::Mean)
            .compute(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result["mean"], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn add_multiple_features() {
        let result = FeatureFactory::new()
            .feature(Feature::Mean)
            .feature(Feature::Variance)
            .feature(Feature::Skewness)
            .compute(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("mean"));
        assert!(result.contains_key("variance"));
        assert!(result.contains_key("skewness"));
    }

    #[test]
    fn add_many_features() {
        let result = FeatureFactory::new()
            .features([Feature::Mean, Feature::Median, Feature::Maximum])
            .compute(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
    }

    // ==================== Category presets ====================

    #[test]
    fn basic_category() {
        let series = test_series();
        let factory = FeatureFactory::new().basic();
        assert_eq!(factory.len(), 17);
        let result = factory.compute(&series);
        assert!(result.contains_key("mean"));
        assert!(result.contains_key("variance"));
        assert!(result.contains_key("standard_deviation"));
        assert!(result.contains_key("minimum"));
        assert!(result.contains_key("maximum"));
    }

    #[test]
    fn distribution_category() {
        let series = test_series();
        let factory = FeatureFactory::new().distribution();
        assert_eq!(factory.len(), 11);
        let result = factory.compute(&series);
        assert!(result.contains_key("skewness"));
        assert!(result.contains_key("kurtosis"));
        assert!(result.contains_key("quantile_0.25"));
    }

    #[test]
    fn autocorrelation_category() {
        let series = test_series();
        let factory = FeatureFactory::new().autocorrelation();
        assert_eq!(factory.len(), 12);
        let result = factory.compute(&series);
        assert!(result.contains_key("autocorrelation_1"));
        assert!(result.contains_key("partial_autocorrelation_1"));
    }

    #[test]
    fn counting_category() {
        let series = test_series();
        let factory = FeatureFactory::new().counting();
        assert_eq!(factory.len(), 15);
        let result = factory.compute(&series);
        assert!(result.contains_key("count_above_mean"));
        assert!(result.contains_key("has_duplicate"));
    }

    #[test]
    fn entropy_category() {
        let series = test_series();
        let factory = FeatureFactory::new().entropy();
        assert_eq!(factory.len(), 5);
        let result = factory.compute(&series);
        assert!(result.contains_key("binned_entropy_10"));
        assert!(result.contains_key("fourier_entropy"));
    }

    #[test]
    fn complexity_category() {
        let series = test_series();
        let factory = FeatureFactory::new().complexity();
        assert_eq!(factory.len(), 5);
        let result = factory.compute(&series);
        assert!(result.contains_key("cid_ce_true"));
        assert!(result.contains_key("c3_1"));
    }

    #[test]
    fn trend_category() {
        let series = test_series();
        let factory = FeatureFactory::new().trend();
        assert_eq!(factory.len(), 8);
        let result = factory.compute(&series);
        assert!(result.contains_key("linear_trend_slope"));
        assert!(result.contains_key("augmented_dickey_fuller"));
    }

    #[test]
    fn change_category() {
        let series = test_series();
        let factory = FeatureFactory::new().change();
        assert_eq!(factory.len(), 7);
        let result = factory.compute(&series);
        assert!(result.contains_key("ratio_value_number_to_length"));
    }

    #[test]
    fn chained_categories() {
        let factory = FeatureFactory::new().basic().distribution();
        assert_eq!(factory.len(), 17 + 11);
    }

    // ==================== Preset sets ====================

    #[test]
    fn default_set() {
        let series = test_series();
        let factory = FeatureFactory::default_set();
        assert_eq!(factory.len(), 35);
        let result = factory.compute(&series);
        assert!(result.contains_key("mean"));
        assert!(result.contains_key("skewness"));
        assert!(result.contains_key("autocorrelation_1"));
        assert!(result.contains_key("linear_trend_slope"));
    }

    #[test]
    fn all_features() {
        let series = test_series();
        let factory = FeatureFactory::all();
        assert!(
            factory.len() >= 70,
            "Expected 70+ features, got {}",
            factory.len()
        );
        let result = factory.compute(&series);
        assert!(result.len() >= 70);
    }

    // ==================== Feature names ====================

    #[test]
    fn feature_names_list() {
        let factory = FeatureFactory::new()
            .feature(Feature::Mean)
            .feature(Feature::Autocorrelation { lag: 3 });
        let names = factory.feature_names();
        assert_eq!(names, vec!["mean", "autocorrelation_3"]);
    }

    #[test]
    fn parametric_feature_names() {
        assert_eq!(Feature::Quantile { q: 0.25 }.name(), "quantile_0.25");
        assert_eq!(
            Feature::SampleEntropy { m: 2, r: 0.2 }.name(),
            "sample_entropy_2_0.2"
        );
        assert_eq!(
            Feature::ArCoefficient { k: 2, coeff: 1 }.name(),
            "ar_coefficient_2_1"
        );
        assert_eq!(
            Feature::EnergyRatioByChunks {
                n_chunks: 10,
                chunk_index: 0
            }
            .name(),
            "energy_ratio_by_chunks_10_0"
        );
    }

    // ==================== Correctness spot checks ====================

    #[test]
    fn computed_values_match_direct_calls() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = FeatureFactory::new()
            .feature(Feature::Mean)
            .feature(Feature::Variance)
            .feature(Feature::Skewness)
            .feature(Feature::Autocorrelation { lag: 1 })
            .feature(Feature::BinnedEntropy { max_bins: 5 })
            .feature(Feature::LinearTrendSlope)
            .compute(&series);

        assert_relative_eq!(
            result["mean"],
            super::super::basic::mean(&series),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result["variance"],
            super::super::basic::variance(&series),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result["skewness"],
            super::super::distribution::skewness(&series),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result["autocorrelation_1"],
            super::super::autocorrelation::autocorrelation(&series, 1),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result["binned_entropy_5"],
            super::super::entropy::binned_entropy(&series, 5),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result["linear_trend_slope"],
            super::super::trend::linear_trend(&series).slope,
            epsilon = 1e-10
        );
    }

    #[test]
    fn bool_features_return_0_or_1() {
        let series = vec![1.0, 2.0, 1.0, 3.0, 4.0];
        let result = FeatureFactory::new()
            .feature(Feature::HasDuplicate)
            .feature(Feature::HasDuplicateMax)
            .feature(Feature::VarianceLargerThanStd)
            .feature(Feature::SymmetryLooking { r: 0.05 })
            .feature(Feature::LargeStandardDeviation { r: 0.25 })
            .compute(&series);

        for (_, &val) in &result {
            assert!(
                val == 0.0 || val == 1.0,
                "Bool feature should be 0 or 1, got {val}"
            );
        }
    }

    #[test]
    fn counting_features_return_integers() {
        let series = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 1.0, 9.0, 4.0, 6.0];
        let result = FeatureFactory::new()
            .feature(Feature::CountAboveMean)
            .feature(Feature::CountBelowMean)
            .feature(Feature::NumberPeaks { support: 1 })
            .feature(Feature::NumberCrossingMean)
            .feature(Feature::LongestStrikeAboveMean)
            .feature(Feature::LongestStrikeBelowMean)
            .compute(&series);

        for (_, &val) in &result {
            assert_relative_eq!(val, val.round(), epsilon = 1e-10);
        }
    }

    #[test]
    fn len_and_is_empty() {
        let empty = FeatureFactory::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let one = FeatureFactory::new().feature(Feature::Mean);
        assert!(!one.is_empty());
        assert_eq!(one.len(), 1);
    }

    // ==================== Edge cases ====================

    #[test]
    fn empty_series() {
        let result = FeatureFactory::default_set().compute(&[]);
        // Should not panic; values will be NaN or 0 depending on the feature
        assert_eq!(result.len(), 35);
    }

    #[test]
    fn single_value_series() {
        let result = FeatureFactory::default_set().compute(&[42.0]);
        assert_eq!(result.len(), 35);
        assert_relative_eq!(result["mean"], 42.0, epsilon = 1e-10);
    }

    #[test]
    fn constant_series() {
        let series = vec![5.0; 50];
        let result = FeatureFactory::new()
            .feature(Feature::Mean)
            .feature(Feature::Variance)
            .feature(Feature::StandardDeviation)
            .compute(&series);
        assert_relative_eq!(result["mean"], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result["variance"], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result["standard_deviation"], 0.0, epsilon = 1e-10);
    }

    // ==================== Integration with selection ====================

    #[test]
    fn output_compatible_with_selection() {
        // FeatureFactory output should work with select_features
        use super::super::selection::{select_features, FeatureSelectionConfig};

        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let series3 = vec![5.0, 5.0, 5.0, 5.0, 5.0];

        let factory = FeatureFactory::new()
            .feature(Feature::Mean)
            .feature(Feature::Variance)
            .feature(Feature::Skewness);

        let r1 = factory.compute(&series1);
        let r2 = factory.compute(&series2);
        let r3 = factory.compute(&series3);

        // Build feature matrix: feature_name -> [val_series1, val_series2, val_series3]
        let mut feature_matrix: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for name in factory.feature_names() {
            feature_matrix.insert(name.clone(), vec![r1[&name], r2[&name], r3[&name]]);
        }

        let config = FeatureSelectionConfig::default();
        let selected = select_features(&feature_matrix, config);
        // At least some features should survive selection
        assert!(!selected.is_empty());
    }
}
