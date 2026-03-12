//! Automated feature selection for time series feature sets.
//!
//! Provides methods to reduce feature dimensionality by removing low-variance,
//! highly correlated, or low-importance features. Designed to work with the
//! 76+ feature extraction functions in this crate.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//! use anofox_forecast::features::selection::{select_features, FeatureSelectionConfig};
//!
//! let mut features = HashMap::new();
//! features.insert("mean".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
//! features.insert("constant".to_string(), vec![5.0, 5.0, 5.0, 5.0]);
//! features.insert("variance".to_string(), vec![1.1, 2.1, 3.1, 4.1]);
//!
//! let config = FeatureSelectionConfig::default();
//! let selected = select_features(&features, config);
//! // "constant" is removed due to zero variance
//! assert!(!selected.contains(&"constant".to_string()));
//! ```

use std::collections::HashMap;

/// A feature paired with its importance score.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Name of the feature.
    pub name: String,
    /// Importance score (higher is more important).
    pub score: f64,
}

/// Configuration for the feature selection pipeline.
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Minimum variance threshold. Features with variance below this are removed.
    pub min_variance: f64,
    /// Maximum allowed Pearson correlation between feature pairs.
    /// When two features exceed this threshold, the second (alphabetically) is removed.
    pub max_correlation: f64,
    /// Maximum number of features to keep. `None` means no limit.
    pub max_features: Option<usize>,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            min_variance: 0.01,
            max_correlation: 0.95,
            max_features: None,
        }
    }
}

/// Methods for selecting subsets of features.
pub struct FeatureSelector;

impl FeatureSelector {
    /// Removes features whose variance falls below `threshold`.
    ///
    /// # Arguments
    /// * `features` - Map of feature name to values across series
    /// * `threshold` - Minimum variance to keep a feature
    ///
    /// # Returns
    /// Sorted list of feature names that pass the variance threshold.
    pub fn variance_threshold(
        features: &HashMap<String, Vec<f64>>,
        threshold: f64,
    ) -> Vec<String> {
        let mut selected: Vec<String> = features
            .iter()
            .filter(|(_, values)| compute_variance(values) >= threshold)
            .map(|(name, _)| name.clone())
            .collect();
        selected.sort();
        selected
    }

    /// Removes highly correlated features, keeping the first of each correlated pair.
    ///
    /// For every pair of features with absolute Pearson correlation above `threshold`,
    /// the feature that comes later in sorted order is removed.
    ///
    /// # Arguments
    /// * `features` - Map of feature name to values across series
    /// * `threshold` - Maximum allowed absolute correlation (e.g. 0.95)
    ///
    /// # Returns
    /// Sorted list of feature names after removing redundant ones.
    pub fn correlation_filter(
        features: &HashMap<String, Vec<f64>>,
        threshold: f64,
    ) -> Vec<String> {
        let mut names: Vec<&String> = features.keys().collect();
        names.sort();

        let mut to_remove = std::collections::HashSet::new();

        for i in 0..names.len() {
            if to_remove.contains(names[i]) {
                continue;
            }
            let vals_i = &features[names[i]];
            for j in (i + 1)..names.len() {
                if to_remove.contains(names[j]) {
                    continue;
                }
                let vals_j = &features[names[j]];
                let corr = pearson_correlation(vals_i, vals_j);
                if corr.abs() > threshold {
                    to_remove.insert(names[j].clone());
                }
            }
        }

        let mut selected: Vec<String> = names
            .into_iter()
            .filter(|n| !to_remove.contains(*n))
            .cloned()
            .collect();
        selected.sort();
        selected
    }

    /// Returns the names of the top-k features by importance score.
    ///
    /// # Arguments
    /// * `importances` - Slice of feature importances
    /// * `k` - Number of top features to select
    ///
    /// # Returns
    /// Names of the top-k features, sorted by descending score.
    pub fn select_top_k(importances: &[FeatureImportance], k: usize) -> Vec<String> {
        let mut sorted: Vec<&FeatureImportance> = importances.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted.iter().take(k).map(|fi| fi.name.clone()).collect()
    }
}

/// Ranks features by discriminative power across series.
///
/// Computes the variance of each feature's values across all series, then
/// normalizes so the scores sum to 1.0. Higher scores indicate features
/// that vary more across series and thus have greater discriminative power.
///
/// # Arguments
/// * `features` - Map of feature name to values across multiple series
///
/// # Returns
/// List of `FeatureImportance` sorted by descending score.
pub fn rank_features(features: &HashMap<String, Vec<f64>>) -> Vec<FeatureImportance> {
    if features.is_empty() {
        return Vec::new();
    }

    let mut raw_scores: Vec<(String, f64)> = features
        .iter()
        .map(|(name, values)| (name.clone(), compute_variance(values)))
        .collect();

    let total: f64 = raw_scores.iter().map(|(_, s)| s).sum();

    let mut importances: Vec<FeatureImportance> = if total > 0.0 {
        raw_scores
            .drain(..)
            .map(|(name, score)| FeatureImportance {
                name,
                score: score / total,
            })
            .collect()
    } else {
        // All features have zero variance; assign equal weight
        let n = raw_scores.len() as f64;
        raw_scores
            .drain(..)
            .map(|(name, _)| FeatureImportance {
                name,
                score: 1.0 / n,
            })
            .collect()
    };

    importances.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    importances
}

/// Convenience function that runs the full feature selection pipeline.
///
/// Pipeline order:
/// 1. Variance threshold filter (removes low-variance features)
/// 2. Correlation filter (removes redundant, highly correlated features)
/// 3. Top-k selection if `max_features` is set (keeps most discriminative)
///
/// # Arguments
/// * `features` - Map of feature name to values across series
/// * `config` - Selection configuration
///
/// # Returns
/// Sorted list of selected feature names.
pub fn select_features(
    features: &HashMap<String, Vec<f64>>,
    config: FeatureSelectionConfig,
) -> Vec<String> {
    // Step 1: Variance threshold
    let after_variance = FeatureSelector::variance_threshold(features, config.min_variance);

    // Build subset map for correlation filtering
    let subset: HashMap<String, Vec<f64>> = after_variance
        .iter()
        .filter_map(|name| features.get(name).map(|v| (name.clone(), v.clone())))
        .collect();

    // Step 2: Correlation filter
    let after_correlation = FeatureSelector::correlation_filter(&subset, config.max_correlation);

    // Step 3: Top-k if max_features is set
    match config.max_features {
        Some(k) if k < after_correlation.len() => {
            // Rank the remaining features and pick top-k
            let remaining: HashMap<String, Vec<f64>> = after_correlation
                .iter()
                .filter_map(|name| features.get(name).map(|v| (name.clone(), v.clone())))
                .collect();
            let importances = rank_features(&remaining);
            let top = FeatureSelector::select_top_k(&importances, k);
            // Return sorted for consistency
            let mut result = top;
            result.sort();
            result
        }
        _ => after_correlation,
    }
}

/// Compute the population variance of a slice.
fn compute_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Compute the Pearson correlation coefficient between two slices.
///
/// Returns 0.0 if either series has zero variance or the lengths differ.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper to build a feature map from name-value pairs.
    fn make_features(entries: &[(&str, Vec<f64>)]) -> HashMap<String, Vec<f64>> {
        entries
            .iter()
            .map(|(name, vals)| (name.to_string(), vals.clone()))
            .collect()
    }

    // ==================== variance_threshold ====================

    #[test]
    fn variance_threshold_removes_constant_features() {
        let features = make_features(&[
            ("varying", vec![1.0, 2.0, 3.0, 4.0]),
            ("constant", vec![5.0, 5.0, 5.0, 5.0]),
            ("low_var", vec![1.0, 1.001, 1.002, 0.999]),
        ]);
        let selected = FeatureSelector::variance_threshold(&features, 0.01);
        assert!(selected.contains(&"varying".to_string()));
        assert!(!selected.contains(&"constant".to_string()));
        assert!(!selected.contains(&"low_var".to_string()));
    }

    #[test]
    fn variance_threshold_keeps_all_above() {
        let features = make_features(&[
            ("a", vec![1.0, 10.0, 20.0]),
            ("b", vec![5.0, 15.0, 25.0]),
        ]);
        let selected = FeatureSelector::variance_threshold(&features, 0.01);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn variance_threshold_empty_map() {
        let features: HashMap<String, Vec<f64>> = HashMap::new();
        let selected = FeatureSelector::variance_threshold(&features, 0.01);
        assert!(selected.is_empty());
    }

    // ==================== correlation_filter ====================

    #[test]
    fn correlation_filter_removes_redundant_features() {
        // b = 2*a + 1, perfectly correlated
        let features = make_features(&[
            ("a", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("b", vec![3.0, 5.0, 7.0, 9.0, 11.0]),
            ("c", vec![5.0, 3.0, 1.0, 4.0, 2.0]),
        ]);
        let selected = FeatureSelector::correlation_filter(&features, 0.95);
        // "a" and "b" are perfectly correlated; "b" (later alphabetically) should be removed
        assert!(selected.contains(&"a".to_string()));
        assert!(!selected.contains(&"b".to_string()));
        assert!(selected.contains(&"c".to_string()));
    }

    #[test]
    fn correlation_filter_keeps_uncorrelated() {
        let features = make_features(&[
            ("x", vec![1.0, 0.0, -1.0, 0.0]),
            ("y", vec![0.0, 1.0, 0.0, -1.0]),
        ]);
        let selected = FeatureSelector::correlation_filter(&features, 0.95);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn correlation_filter_single_feature() {
        let features = make_features(&[("only", vec![1.0, 2.0, 3.0])]);
        let selected = FeatureSelector::correlation_filter(&features, 0.95);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], "only");
    }

    // ==================== select_top_k ====================

    #[test]
    fn select_top_k_returns_correct_count() {
        let importances = vec![
            FeatureImportance {
                name: "a".to_string(),
                score: 0.5,
            },
            FeatureImportance {
                name: "b".to_string(),
                score: 0.8,
            },
            FeatureImportance {
                name: "c".to_string(),
                score: 0.3,
            },
            FeatureImportance {
                name: "d".to_string(),
                score: 0.9,
            },
        ];
        let top2 = FeatureSelector::select_top_k(&importances, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], "d");
        assert_eq!(top2[1], "b");
    }

    #[test]
    fn select_top_k_exceeds_length() {
        let importances = vec![FeatureImportance {
            name: "only".to_string(),
            score: 1.0,
        }];
        let top5 = FeatureSelector::select_top_k(&importances, 5);
        assert_eq!(top5.len(), 1);
    }

    // ==================== rank_features ====================

    #[test]
    fn rank_features_scores_sum_to_one() {
        let features = make_features(&[
            ("a", vec![1.0, 10.0, 3.0]),
            ("b", vec![5.0, 5.0, 5.0]),
            ("c", vec![2.0, 8.0, 4.0]),
        ]);
        let ranked = rank_features(&features);
        // "b" is constant so its variance is 0; non-constant features get scores
        let total: f64 = ranked.iter().map(|r| r.score).sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn rank_features_highest_variance_first() {
        let features = make_features(&[
            ("low_var", vec![1.0, 1.1, 0.9]),
            ("high_var", vec![1.0, 100.0, 50.0]),
        ]);
        let ranked = rank_features(&features);
        assert_eq!(ranked[0].name, "high_var");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn rank_features_empty() {
        let features: HashMap<String, Vec<f64>> = HashMap::new();
        let ranked = rank_features(&features);
        assert!(ranked.is_empty());
    }

    #[test]
    fn rank_features_all_constant() {
        let features = make_features(&[
            ("a", vec![1.0, 1.0, 1.0]),
            ("b", vec![2.0, 2.0, 2.0]),
        ]);
        let ranked = rank_features(&features);
        // All zero variance: equal weight
        for fi in &ranked {
            assert_relative_eq!(fi.score, 0.5, epsilon = 1e-10);
        }
    }

    // ==================== select_features (full pipeline) ====================

    #[test]
    fn select_features_full_pipeline() {
        let features = make_features(&[
            ("varying", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("constant", vec![7.0, 7.0, 7.0, 7.0, 7.0]),
            ("correlated", vec![2.0, 4.0, 6.0, 8.0, 10.0]),    // = 2 * varying
            ("independent", vec![5.0, 3.0, 1.0, 4.0, 2.0]),
        ]);
        let config = FeatureSelectionConfig {
            min_variance: 0.01,
            max_correlation: 0.95,
            max_features: None,
        };
        let selected = select_features(&features, config);
        // "constant" removed by variance filter
        assert!(!selected.contains(&"constant".to_string()));
        // "correlated" removed because it's perfectly correlated with "varying"
        // (alphabetically "correlated" < "varying", so "varying" is removed)
        assert!(selected.contains(&"independent".to_string()));
        // Either "correlated" or "varying" kept, but not both
        let has_corr = selected.contains(&"correlated".to_string());
        let has_vary = selected.contains(&"varying".to_string());
        assert!(has_corr || has_vary);
        assert!(!(has_corr && has_vary));
    }

    #[test]
    fn select_features_with_max_features() {
        let features = make_features(&[
            ("a", vec![1.0, 10.0, 3.0, 7.0]),
            ("b", vec![2.0, 20.0, 5.0, 15.0]),
            ("c", vec![100.0, 1.0, 50.0, 25.0]),
            ("d", vec![3.0, 3.0, 3.0, 3.0]),
        ]);
        let config = FeatureSelectionConfig {
            min_variance: 0.01,
            max_correlation: 0.95,
            max_features: Some(1),
        };
        let selected = select_features(&features, config);
        // "d" removed by variance; "a" and "b" highly correlated so one removed;
        // Then top-1 of the remaining
        assert!(selected.len() <= 1);
    }

    #[test]
    fn select_features_empty_input() {
        let features: HashMap<String, Vec<f64>> = HashMap::new();
        let config = FeatureSelectionConfig::default();
        let selected = select_features(&features, config);
        assert!(selected.is_empty());
    }

    #[test]
    fn select_features_all_identical_values() {
        let features = make_features(&[
            ("a", vec![1.0, 1.0, 1.0]),
            ("b", vec![2.0, 2.0, 2.0]),
            ("c", vec![3.0, 3.0, 3.0]),
        ]);
        let config = FeatureSelectionConfig::default();
        let selected = select_features(&features, config);
        // All constant -> all removed by variance filter
        assert!(selected.is_empty());
    }

    // ==================== internal helpers ====================

    #[test]
    fn pearson_correlation_perfect() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_relative_eq!(pearson_correlation(&a, &b), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn pearson_correlation_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_relative_eq!(pearson_correlation(&a, &b), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn compute_variance_works() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Population variance = 4.0
        assert_relative_eq!(compute_variance(&values), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn compute_variance_empty() {
        assert_relative_eq!(compute_variance(&[]), 0.0, epsilon = 1e-10);
    }
}
