//! Integration with the Automatic Identification of Demand (AID) classifier
//! from the `anofox-regression` crate.
//!
//! Provides two views of the classification result:
//! - **Summary statistics** ([`AidSummary`]): demand type, best-fitting
//!   distribution, fitted parameters, and zero proportion.
//! - **Features** ([`AidFeatures`]): per-observation anomaly labels
//!   (`Vec<AidAnomalyLabel>`) with the same length as the input series.
//!
//! Both are computed from a single [`AidResult`] which wraps the underlying
//! `DemandClassification` from `anofox-regression`.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_forecast::validation::aid::{AidAnalyzer, AidAnomalyLabel};
//!
//! let demand = vec![10.0, 0.0, 0.0, 5.0, 0.0, 12.0, 0.0, 0.0, 8.0, 0.0];
//! let result = AidAnalyzer::new().analyze(&demand);
//!
//! // Summary statistics
//! let summary = result.summary();
//! println!("Demand type: {:?}", summary.demand_type);
//! println!("Distribution: {:?}", summary.distribution);
//! println!("Mean: {:.2}", summary.mean);
//!
//! // Per-observation features
//! let features = result.features();
//! assert_eq!(features.labels.len(), demand.len());
//! for (i, label) in features.labels.iter().enumerate() {
//!     if *label != AidAnomalyLabel::Normal {
//!         println!("Observation {}: {:?}", i, label);
//!     }
//! }
//! ```

use anofox_regression::solvers::{
    AidClassifier, AidClassifierBuilder, AnomalyType, DemandClassification, DemandDistribution,
    DemandType, InformationCriterion,
};
use faer::Col;
use std::collections::HashMap;
use std::fmt;

/// Per-observation anomaly label produced by the AID classifier.
///
/// This is a forecast-friendly relabeling of `anofox_regression::AnomalyType`
/// so that downstream code does not need to depend on the regression crate
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AidAnomalyLabel {
    /// No anomaly — normal demand observation.
    Normal,
    /// Unexpected zero in a region that otherwise has demand (potential stockout).
    Stockout,
    /// Leading zeros indicating a new product lifecycle.
    NewProduct,
    /// Trailing zeros indicating an obsolete/end-of-life product.
    ObsoleteProduct,
    /// Unusually high demand value.
    HighOutlier,
    /// Unusually low (but non-zero) demand value.
    LowOutlier,
}

impl fmt::Display for AidAnomalyLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AidAnomalyLabel::Normal => write!(f, "Normal"),
            AidAnomalyLabel::Stockout => write!(f, "Stockout"),
            AidAnomalyLabel::NewProduct => write!(f, "NewProduct"),
            AidAnomalyLabel::ObsoleteProduct => write!(f, "ObsoleteProduct"),
            AidAnomalyLabel::HighOutlier => write!(f, "HighOutlier"),
            AidAnomalyLabel::LowOutlier => write!(f, "LowOutlier"),
        }
    }
}

impl From<AnomalyType> for AidAnomalyLabel {
    fn from(at: AnomalyType) -> Self {
        match at {
            AnomalyType::None => AidAnomalyLabel::Normal,
            AnomalyType::Stockout => AidAnomalyLabel::Stockout,
            AnomalyType::NewProduct => AidAnomalyLabel::NewProduct,
            AnomalyType::ObsoleteProduct => AidAnomalyLabel::ObsoleteProduct,
            AnomalyType::HighOutlier => AidAnomalyLabel::HighOutlier,
            AnomalyType::LowOutlier => AidAnomalyLabel::LowOutlier,
        }
    }
}

/// Summary statistics from the AID classification.
///
/// Contains the demand type, best-fitting distribution, fitted parameters,
/// and zero proportion — a single-row description of the series.
#[derive(Debug, Clone)]
pub struct AidSummary {
    /// Whether the demand is regular or intermittent.
    pub demand_type: DemandType,
    /// Whether the data contains fractional (non-integer) values.
    pub is_fractional: bool,
    /// Best-fitting demand distribution selected by information criterion.
    pub distribution: DemandDistribution,
    /// Fitted mean of the selected distribution.
    pub mean: f64,
    /// Fitted variance of the selected distribution.
    pub variance: f64,
    /// Shape parameter (Gamma, Negative Binomial), if applicable.
    pub shape: Option<f64>,
    /// Scale parameter, if applicable.
    pub scale: Option<f64>,
    /// Estimated probability of zero demand (intermittent models).
    pub zero_prob: Option<f64>,
    /// Observed proportion of zeros in the series.
    pub zero_proportion: f64,
    /// Number of observations analysed.
    pub n_observations: usize,
    /// Information criterion values for all candidate distributions.
    pub ic_values: HashMap<DemandDistribution, f64>,
}

impl fmt::Display for AidSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AID Summary")?;
        writeln!(f, "===========")?;
        writeln!(f, "Demand type:    {:?}", self.demand_type)?;
        writeln!(f, "Distribution:   {:?}", self.distribution)?;
        writeln!(f, "Mean:           {:.4}", self.mean)?;
        writeln!(f, "Variance:       {:.4}", self.variance)?;
        if let Some(s) = self.shape {
            writeln!(f, "Shape:          {:.4}", s)?;
        }
        if let Some(s) = self.scale {
            writeln!(f, "Scale:          {:.4}", s)?;
        }
        if let Some(zp) = self.zero_prob {
            writeln!(f, "Zero prob:      {:.4}", zp)?;
        }
        writeln!(f, "Zero proportion:{:.4}", self.zero_proportion)?;
        writeln!(f, "N observations: {}", self.n_observations)?;
        writeln!(f, "Fractional:     {}", self.is_fractional)?;
        Ok(())
    }
}

/// Per-observation anomaly features from the AID classification.
///
/// The `labels` vector has the same length as the input time series.
/// Each element describes the anomaly status of the corresponding observation.
#[derive(Debug, Clone)]
pub struct AidFeatures {
    /// One label per observation, same length as the input series.
    pub labels: Vec<AidAnomalyLabel>,
}

impl AidFeatures {
    /// Count occurrences of each anomaly label.
    pub fn label_counts(&self) -> HashMap<AidAnomalyLabel, usize> {
        let mut counts = HashMap::new();
        for &label in &self.labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
    }

    /// Return `true` if any stockout was detected.
    pub fn has_stockouts(&self) -> bool {
        self.labels.contains(&AidAnomalyLabel::Stockout)
    }

    /// Return `true` if the series starts with a new-product pattern.
    pub fn is_new_product(&self) -> bool {
        self.labels.contains(&AidAnomalyLabel::NewProduct)
    }

    /// Return `true` if the series ends with an obsolete-product pattern.
    pub fn is_obsolete_product(&self) -> bool {
        self.labels.contains(&AidAnomalyLabel::ObsoleteProduct)
    }

    /// Return the indices of observations flagged as any anomaly.
    pub fn anomaly_indices(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l != AidAnomalyLabel::Normal)
            .map(|(i, _)| i)
            .collect()
    }
}

impl fmt::Display for AidFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counts = self.label_counts();
        let n_anomalies: usize = counts
            .iter()
            .filter(|(k, _)| **k != AidAnomalyLabel::Normal)
            .map(|(_, v)| *v)
            .sum();
        writeln!(f, "AID Features ({} observations)", self.labels.len())?;
        writeln!(f, "Anomalies: {}", n_anomalies)?;
        for (label, count) in &counts {
            if *label != AidAnomalyLabel::Normal {
                writeln!(f, "  {:?}: {}", label, count)?;
            }
        }
        Ok(())
    }
}

/// Full AID result wrapping the underlying `DemandClassification`.
///
/// Use [`summary()`](AidResult::summary) for aggregate statistics and
/// [`features()`](AidResult::features) for per-observation anomaly labels.
#[derive(Debug, Clone)]
pub struct AidResult {
    inner: DemandClassification,
}

impl AidResult {
    /// Extract summary statistics (single-row view).
    pub fn summary(&self) -> AidSummary {
        AidSummary {
            demand_type: self.inner.demand_type,
            is_fractional: self.inner.is_fractional,
            distribution: self.inner.distribution,
            mean: self.inner.parameters.mean,
            variance: self.inner.parameters.variance,
            shape: self.inner.parameters.shape,
            scale: self.inner.parameters.scale,
            zero_prob: self.inner.parameters.zero_prob,
            zero_proportion: self.inner.zero_proportion,
            n_observations: self.inner.n_observations,
            ic_values: self.inner.ic_values.clone(),
        }
    }

    /// Extract per-observation anomaly features.
    ///
    /// The returned `AidFeatures.labels` vector has the same length as the
    /// input time series. Each element is an [`AidAnomalyLabel`].
    pub fn features(&self) -> AidFeatures {
        AidFeatures {
            labels: self
                .inner
                .anomalies
                .iter()
                .map(|a| AidAnomalyLabel::from(*a))
                .collect(),
        }
    }

    /// Access the underlying `DemandClassification` from `anofox-regression`.
    pub fn raw(&self) -> &DemandClassification {
        &self.inner
    }
}

/// Builder / entry point for running the AID classifier on demand data.
///
/// Wraps [`AidClassifier`] with
/// an API that accepts `&[f64]` (converting to `faer::Col<f64>` internally).
///
/// # Example
///
/// ```rust,ignore
/// use anofox_forecast::validation::aid::AidAnalyzer;
///
/// let data = vec![5.0, 0.0, 0.0, 12.0, 0.0, 8.0, 0.0, 0.0, 3.0, 0.0];
/// let result = AidAnalyzer::new().analyze(&data);
///
/// let summary = result.summary();
/// println!("{}", summary);
///
/// let features = result.features();
/// println!("{}", features);
/// ```
pub struct AidAnalyzer {
    builder: AidClassifierBuilder,
}

impl Default for AidAnalyzer {
    fn default() -> Self {
        Self {
            builder: AidClassifier::builder(),
        }
    }
}

impl AidAnalyzer {
    /// Create an analyzer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the significance level for anomaly detection.
    ///
    /// Clamped to [0.001, 0.5]. Default: 0.05.
    pub fn anomaly_alpha(mut self, alpha: f64) -> Self {
        self.builder = self.builder.anomaly_alpha(alpha);
        self
    }

    /// Set the zero-proportion threshold for intermittent classification.
    ///
    /// Series with a zero proportion above this threshold are classified as
    /// intermittent. Default: 0.3.
    pub fn intermittent_threshold(mut self, threshold: f64) -> Self {
        self.builder = self.builder.intermittent_threshold(threshold);
        self
    }

    /// Enable or disable per-observation anomaly detection.
    ///
    /// Default: `true`.
    pub fn detect_anomalies(mut self, detect: bool) -> Self {
        self.builder = self.builder.detect_anomalies(detect);
        self
    }

    /// Set the information criterion used for distribution selection.
    ///
    /// Default: `AICc`.
    pub fn ic(mut self, criterion: InformationCriterion) -> Self {
        self.builder = self.builder.ic(criterion);
        self
    }

    /// Run the AID classifier on a slice of demand values.
    ///
    /// Converts the input to `faer::Col<f64>` internally.
    pub fn analyze(self, data: &[f64]) -> AidResult {
        let col = Col::from_fn(data.len(), |i| data[i]);
        let classifier = self.builder.build();
        let classification = classifier.classify(&col);
        AidResult {
            inner: classification,
        }
    }

    /// Run the AID classifier on the first dimension of a
    /// [`TimeSeries`](crate::core::TimeSeries).
    pub fn analyze_series(self, ts: &crate::core::TimeSeries) -> crate::Result<AidResult> {
        let values = ts.values(0)?;
        Ok(self.analyze(values))
    }
}

// Re-export the upstream types that users may need for matching.
pub use anofox_regression::solvers::{
    DemandDistribution as AidDistribution, DemandType as AidDemandType,
    InformationCriterion as AidInformationCriterion,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== AidAnalyzer basics ====================

    #[test]
    fn regular_demand_classified_correctly() {
        let data: Vec<f64> = (0..100)
            .map(|i| 10.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();

        assert_eq!(summary.demand_type, DemandType::Regular);
        assert_eq!(summary.n_observations, 100);
        assert!(summary.zero_proportion < 0.01);
    }

    #[test]
    fn intermittent_demand_classified_correctly() {
        // 60% zeros
        let mut data = vec![0.0; 60];
        for i in 0..40 {
            data[i * 3 / 2] = 5.0 + (i as f64) * 0.5;
        }
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();

        assert_eq!(summary.demand_type, DemandType::Intermittent);
    }

    #[test]
    fn summary_fields_populated() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64) + 1.0).collect();
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();

        assert!(summary.mean > 0.0);
        assert!(summary.variance > 0.0);
        assert_eq!(summary.n_observations, 50);
        assert!(!summary.ic_values.is_empty());
    }

    // ==================== Features / anomaly labels ====================

    #[test]
    fn features_length_matches_input() {
        let data = vec![10.0, 0.0, 5.0, 0.0, 8.0, 0.0, 12.0, 0.0, 6.0, 0.0];
        let result = AidAnalyzer::new().analyze(&data);
        let features = result.features();

        assert_eq!(features.labels.len(), data.len());
    }

    #[test]
    fn features_without_anomaly_detection() {
        let data: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let result = AidAnalyzer::new().detect_anomalies(false).analyze(&data);
        let features = result.features();

        assert_eq!(features.labels.len(), data.len());
        // All should be Normal when detection is off
        assert!(features
            .labels
            .iter()
            .all(|l| *l == AidAnomalyLabel::Normal));
    }

    #[test]
    fn label_counts_sum_to_length() {
        let data = vec![0.0, 0.0, 0.0, 5.0, 10.0, 0.0, 0.0, 8.0, 0.0, 0.0];
        let result = AidAnalyzer::new().analyze(&data);
        let features = result.features();
        let counts = features.label_counts();
        let total: usize = counts.values().sum();

        assert_eq!(total, data.len());
    }

    #[test]
    fn anomaly_indices_correct() {
        let data: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let result = AidAnalyzer::new().detect_anomalies(false).analyze(&data);
        let features = result.features();

        assert!(features.anomaly_indices().is_empty());
    }

    // ==================== Builder configuration ====================

    #[test]
    fn custom_threshold_changes_classification() {
        // Series with ~25% zeros — default threshold (0.3) classifies as Regular,
        // lower threshold (0.1) classifies as Intermittent.
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64) + 1.0).collect();
        for i in (0..100).step_by(4) {
            data[i] = 0.0;
        }

        let result_default = AidAnalyzer::new().analyze(&data);
        let result_strict = AidAnalyzer::new()
            .intermittent_threshold(0.1)
            .analyze(&data);

        assert_eq!(result_default.summary().demand_type, DemandType::Regular);
        assert_eq!(
            result_strict.summary().demand_type,
            DemandType::Intermittent
        );
    }

    #[test]
    fn ic_option_accepted() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64) + 1.0).collect();
        let result = AidAnalyzer::new()
            .ic(InformationCriterion::BIC)
            .analyze(&data);
        let summary = result.summary();
        // Should still produce a valid result
        assert!(summary.n_observations == 50);
    }

    // ==================== Display ====================

    #[test]
    fn summary_display_contains_key_fields() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64) + 1.0).collect();
        let result = AidAnalyzer::new().analyze(&data);
        let text = format!("{}", result.summary());

        assert!(text.contains("AID Summary"));
        assert!(text.contains("Demand type:"));
        assert!(text.contains("Distribution:"));
        assert!(text.contains("Mean:"));
    }

    #[test]
    fn features_display_shows_anomaly_count() {
        let data: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let result = AidAnalyzer::new().analyze(&data);
        let text = format!("{}", result.features());

        assert!(text.contains("AID Features"));
        assert!(text.contains("Anomalies:"));
    }

    // ==================== AidAnomalyLabel conversion ====================

    #[test]
    fn anomaly_type_conversion_roundtrip() {
        let pairs = vec![
            (AnomalyType::None, AidAnomalyLabel::Normal),
            (AnomalyType::Stockout, AidAnomalyLabel::Stockout),
            (AnomalyType::NewProduct, AidAnomalyLabel::NewProduct),
            (
                AnomalyType::ObsoleteProduct,
                AidAnomalyLabel::ObsoleteProduct,
            ),
            (AnomalyType::HighOutlier, AidAnomalyLabel::HighOutlier),
            (AnomalyType::LowOutlier, AidAnomalyLabel::LowOutlier),
        ];
        for (src, expected) in pairs {
            assert_eq!(AidAnomalyLabel::from(src), expected);
        }
    }

    #[test]
    fn label_display() {
        assert_eq!(format!("{}", AidAnomalyLabel::Normal), "Normal");
        assert_eq!(format!("{}", AidAnomalyLabel::Stockout), "Stockout");
        assert_eq!(format!("{}", AidAnomalyLabel::HighOutlier), "HighOutlier");
    }

    // ==================== Edge cases ====================

    #[test]
    fn empty_series() {
        let data: Vec<f64> = vec![];
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();
        let features = result.features();

        assert_eq!(summary.n_observations, 0);
        assert!(features.labels.is_empty());
    }

    #[test]
    fn single_observation() {
        let data = vec![42.0];
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();
        let features = result.features();

        assert_eq!(summary.n_observations, 1);
        assert_eq!(features.labels.len(), 1);
    }

    #[test]
    fn all_zeros() {
        let data = vec![0.0; 20];
        let result = AidAnalyzer::new().analyze(&data);
        let summary = result.summary();

        assert_eq!(summary.demand_type, DemandType::Intermittent);
        assert!((summary.zero_proportion - 1.0).abs() < 1e-10);
    }

    #[test]
    fn raw_access() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64) + 1.0).collect();
        let result = AidAnalyzer::new().analyze(&data);
        let raw = result.raw();

        assert_eq!(raw.n_observations, 30);
        assert_eq!(raw.anomalies.len(), 30);
    }

    // ==================== TimeSeries integration ====================

    #[test]
    fn analyze_time_series() {
        use crate::core::TimeSeriesBuilder;
        use chrono::{Duration, Utc};

        let n = 50;
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap();
        let result = AidAnalyzer::new().analyze_series(&ts).unwrap();
        let summary = result.summary();

        assert_eq!(summary.n_observations, n);
        assert_eq!(result.features().labels.len(), n);
    }
}
