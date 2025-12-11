//! Time series feature extraction.
//!
//! Provides 76+ statistical features for ML pipelines, compatible with tsfresh.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::features::{basic, distribution, counting};
//!
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Basic statistics
//! let m = basic::mean(&series);
//! let v = basic::variance(&series);
//!
//! // Distribution features
//! let sk = distribution::skewness(&series);
//!
//! // Counting features
//! let peaks = counting::number_peaks(&series, 1);
//! ```

pub mod autocorrelation;
pub mod basic;
pub mod change;
pub mod complexity;
pub mod counting;
pub mod distribution;
pub mod entropy;
pub mod trend;

// Re-export commonly used items
pub use basic::{
    abs_energy, absolute_maximum, absolute_sum_of_changes, length, maximum, mean, mean_abs_change,
    mean_change, mean_n_absolute_max, mean_second_derivative_central, median, minimum,
    root_mean_square, standard_deviation, sum_values, variance,
};

pub use distribution::{
    kurtosis, large_standard_deviation, quantile, ratio_beyond_r_sigma, skewness, symmetry_looking,
    variance_larger_than_standard_deviation, variation_coefficient,
};

pub use autocorrelation::{
    agg_autocorrelation, autocorrelation, partial_autocorrelation,
    time_reversal_asymmetry_statistic,
};

pub use counting::{
    count_above, count_above_mean, count_below, count_below_mean, first_location_of_maximum,
    first_location_of_minimum, has_duplicate, has_duplicate_max, has_duplicate_min,
    index_mass_quantile, last_location_of_maximum, last_location_of_minimum,
    longest_strike_above_mean, longest_strike_below_mean, number_crossing_m, number_peaks,
    range_count, value_count,
};

pub use entropy::{
    approximate_entropy, binned_entropy, fourier_entropy, permutation_entropy, sample_entropy,
};

pub use complexity::{c3, cid_ce, lempel_ziv_complexity};

pub use trend::{
    agg_linear_trend, ar_coefficient, augmented_dickey_fuller, linear_trend, LinearTrendResult,
};

pub use change::{
    change_quantiles, energy_ratio_by_chunks,
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    percentage_of_reoccurring_values_to_all_values, ratio_value_number_to_time_series_length,
    sum_of_reoccurring_data_points, sum_of_reoccurring_values,
};
