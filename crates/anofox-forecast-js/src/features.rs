//! Feature extraction functions for JavaScript.
//!
//! Exposes statistical features from anofox-forecast for use in JS/WASM.

use wasm_bindgen::prelude::*;

use anofox_forecast::features::{
    autocorrelation as acf_fn, basic, distribution, entropy, partial_autocorrelation as pacf_fn,
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
