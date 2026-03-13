//! Specialized diagnostic tools for intermittent demand series.
//!
//! Implements the Syntetos-Boylan (2005) demand classification framework
//! and computes metrics relevant to intermittent demand forecasting:
//! ADI (Average Demand Interval), CV-squared of non-zero demands,
//! demand classification, coverage rate, bias, and Periods-In-Stock.

use std::fmt;

/// Demand classification following Syntetos-Boylan (2005).
///
/// The classification uses two dimensions:
/// - ADI (Average Demand Interval): mean number of periods between demands
/// - CV squared: squared coefficient of variation of non-zero demand sizes
///
/// Thresholds: ADI = 1.32, CV squared = 0.49.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemandClassification {
    /// ADI < 1.32 and CV squared < 0.49: regular, low-variability demand.
    Smooth,
    /// ADI < 1.32 and CV squared >= 0.49: frequent but variable demand sizes.
    Erratic,
    /// ADI >= 1.32 and CV squared < 0.49: infrequent but consistent demand sizes.
    Intermittent,
    /// ADI >= 1.32 and CV squared >= 0.49: infrequent and variable demand sizes.
    Lumpy,
}

impl fmt::Display for DemandClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DemandClassification::Smooth => write!(f, "Smooth"),
            DemandClassification::Erratic => write!(f, "Erratic"),
            DemandClassification::Intermittent => write!(f, "Intermittent"),
            DemandClassification::Lumpy => write!(f, "Lumpy"),
        }
    }
}

/// Specialized diagnostics for intermittent demand data.
///
/// Provides demand classification, forecast bias analysis, coverage rate
/// evaluation, and Periods-In-Stock (PIS) tracking for intermittent
/// demand models such as Croston, TSB, ADIDA, and IMAPA.
#[derive(Debug, Clone)]
pub struct IntermittentDiagnostics {
    /// Average Demand Interval: mean number of periods between demands.
    pub adi: f64,
    /// Coefficient of Variation squared (CV squared) of non-zero demands.
    pub cv_squared: f64,
    /// Demand classification: Smooth, Erratic, Intermittent, or Lumpy.
    pub classification: DemandClassification,
    /// Fraction of zero observations in the series.
    pub zero_fraction: f64,
    /// Coverage rate: fraction of actuals within prediction intervals.
    /// Only populated when intervals are provided via `with_intervals`.
    pub coverage_rate: Option<f64>,
    /// Bias: mean(forecast - actual) for non-zero periods.
    /// Only populated when a forecast is provided.
    pub bias: f64,
    /// Periods-In-Stock (PIS): cumulative(forecast) - cumulative(actual)
    /// at each time step. Positive values indicate overstock.
    /// Only populated when a forecast is provided.
    pub periods_in_stock: Vec<f64>,
}

/// ADI threshold for the Syntetos-Boylan classification.
const ADI_THRESHOLD: f64 = 1.32;
/// CV squared threshold for the Syntetos-Boylan classification.
const CV2_THRESHOLD: f64 = 0.49;

impl IntermittentDiagnostics {
    /// Classify demand based on ADI and CV squared thresholds.
    fn classify(adi: f64, cv_squared: f64) -> DemandClassification {
        match (adi < ADI_THRESHOLD, cv_squared < CV2_THRESHOLD) {
            (true, true) => DemandClassification::Smooth,
            (true, false) => DemandClassification::Erratic,
            (false, true) => DemandClassification::Intermittent,
            (false, false) => DemandClassification::Lumpy,
        }
    }

    /// Compute ADI from the actual series.
    ///
    /// ADI is defined as n / (number of non-zero observations).
    /// When every period has demand, ADI = 1.0.
    /// When no period has demand, ADI is infinite (we return f64::INFINITY).
    fn compute_adi(actual: &[f64]) -> f64 {
        let nonzero_count = actual.iter().filter(|&&v| v > 0.0).count();
        if nonzero_count == 0 {
            return f64::INFINITY;
        }
        actual.len() as f64 / nonzero_count as f64
    }

    /// Compute CV squared of non-zero demand values.
    ///
    /// CV = std / mean, so CV squared = variance / mean^2.
    /// Returns 0.0 when there are fewer than 2 non-zero observations
    /// or when the mean is zero.
    fn compute_cv_squared(actual: &[f64]) -> f64 {
        let nonzero: Vec<f64> = actual.iter().copied().filter(|&v| v > 0.0).collect();
        let n = nonzero.len();
        if n < 2 {
            return 0.0;
        }
        let mean = nonzero.iter().sum::<f64>() / n as f64;
        if mean.abs() < 1e-30 {
            return 0.0;
        }
        let variance = nonzero.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance / (mean * mean)
    }

    /// Compute the zero fraction of the series.
    fn compute_zero_fraction(actual: &[f64]) -> f64 {
        if actual.is_empty() {
            return 0.0;
        }
        let zero_count = actual.iter().filter(|&&v| v == 0.0).count();
        zero_count as f64 / actual.len() as f64
    }

    /// Create diagnostics from actual demand data only (no forecast).
    ///
    /// Computes ADI, CV squared, classification, and zero fraction.
    /// Bias and PIS are set to defaults (0.0 and empty).
    pub fn from_data(actual: &[f64]) -> Self {
        let adi = Self::compute_adi(actual);
        let cv_squared = Self::compute_cv_squared(actual);
        let classification = Self::classify(adi, cv_squared);
        let zero_fraction = Self::compute_zero_fraction(actual);

        IntermittentDiagnostics {
            adi,
            cv_squared,
            classification,
            zero_fraction,
            coverage_rate: None,
            bias: 0.0,
            periods_in_stock: Vec::new(),
        }
    }

    /// Create diagnostics from actual and forecast data.
    ///
    /// In addition to the data-only metrics, computes:
    /// - Bias: mean(forecast - actual) over non-zero actual periods
    /// - Periods-In-Stock: cumulative(forecast - actual) at each step
    pub fn with_forecast(actual: &[f64], forecast: &[f64]) -> Self {
        let mut diag = Self::from_data(actual);
        let len = actual.len().min(forecast.len());

        // Bias over non-zero actual periods
        let mut bias_sum = 0.0;
        let mut bias_count = 0usize;
        for i in 0..len {
            if actual[i] > 0.0 {
                bias_sum += forecast[i] - actual[i];
                bias_count += 1;
            }
        }
        diag.bias = if bias_count > 0 {
            bias_sum / bias_count as f64
        } else {
            0.0
        };

        // Periods-In-Stock: cumulative forecast minus cumulative actual
        let mut cum_diff = 0.0;
        let mut pis = Vec::with_capacity(len);
        for i in 0..len {
            cum_diff += forecast[i] - actual[i];
            pis.push(cum_diff);
        }
        diag.periods_in_stock = pis;

        diag
    }

    /// Create diagnostics from actual, forecast, and prediction interval data.
    ///
    /// In addition to `with_forecast` metrics, computes:
    /// - Coverage rate: fraction of actuals falling within [lower, upper]
    pub fn with_intervals(actual: &[f64], forecast: &[f64], lower: &[f64], upper: &[f64]) -> Self {
        let mut diag = Self::with_forecast(actual, forecast);
        let len = actual
            .len()
            .min(forecast.len())
            .min(lower.len())
            .min(upper.len());

        if len == 0 {
            diag.coverage_rate = Some(0.0);
            return diag;
        }

        let covered = (0..len)
            .filter(|&i| actual[i] >= lower[i] && actual[i] <= upper[i])
            .count();
        diag.coverage_rate = Some(covered as f64 / len as f64);

        diag
    }

    /// Human-readable summary of the diagnostics.
    pub fn summary(&self) -> String {
        let mut s = String::from("Intermittent Demand Diagnostics\n");
        s.push_str("================================\n");
        s.push_str(&format!("ADI:                {:.4}\n", self.adi));
        s.push_str(&format!("CV squared:         {:.4}\n", self.cv_squared));
        s.push_str(&format!("Classification:     {}\n", self.classification));
        s.push_str(&format!("Zero fraction:      {:.4}\n", self.zero_fraction));
        s.push_str(&format!("Bias (non-zero):    {:.4}\n", self.bias));

        if let Some(cr) = self.coverage_rate {
            s.push_str(&format!("Coverage rate:      {:.4}\n", cr));
        }

        if !self.periods_in_stock.is_empty() {
            if let Some(last) = self.periods_in_stock.last() {
                s.push_str(&format!("Final PIS:          {:.4}\n", last));
            }
        }

        s.push_str(&format!(
            "Recommended model:  {}\n",
            self.recommended_model()
        ));

        s
    }

    /// Recommend which intermittent model to use based on classification.
    ///
    /// Follows Syntetos-Boylan (2005) guidance:
    /// - Smooth: SES or ARIMA (demand is regular enough)
    /// - Erratic: Croston (frequent but variable sizes)
    /// - Intermittent: Croston or SBA (infrequent, consistent sizes)
    /// - Lumpy: TSB (infrequent and variable sizes)
    pub fn recommended_model(&self) -> &'static str {
        match self.classification {
            DemandClassification::Smooth => "SES or ARIMA",
            DemandClassification::Erratic => "Croston",
            DemandClassification::Intermittent => "Croston or SBA",
            DemandClassification::Lumpy => "TSB",
        }
    }
}

impl fmt::Display for IntermittentDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Classification tests ====================

    #[test]
    fn smooth_data_classified_correctly() {
        // All non-zero, low variability: ADI ~ 1.0, CV^2 < 0.49
        let data = vec![10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 10.0, 11.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert_eq!(diag.classification, DemandClassification::Smooth);
        assert!(
            diag.adi < ADI_THRESHOLD,
            "ADI should be < 1.32, got {}",
            diag.adi
        );
        assert!(
            diag.cv_squared < CV2_THRESHOLD,
            "CV^2 should be < 0.49, got {}",
            diag.cv_squared
        );
    }

    #[test]
    fn erratic_data_classified_correctly() {
        // All non-zero (ADI ~ 1.0), high variability in sizes
        let data = vec![1.0, 50.0, 2.0, 80.0, 1.0, 60.0, 3.0, 90.0, 1.0, 70.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert_eq!(diag.classification, DemandClassification::Erratic);
        assert!(
            diag.adi < ADI_THRESHOLD,
            "ADI should be < 1.32, got {}",
            diag.adi
        );
        assert!(
            diag.cv_squared >= CV2_THRESHOLD,
            "CV^2 should be >= 0.49, got {}",
            diag.cv_squared
        );
    }

    #[test]
    fn intermittent_data_classified_correctly() {
        // Many zeros (high ADI), but consistent non-zero sizes (low CV^2)
        // Demand = 10 every ~4 periods -> ADI ~ 4.0, sizes are constant
        let data = vec![
            10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0,
        ];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert_eq!(diag.classification, DemandClassification::Intermittent);
        assert!(
            diag.adi >= ADI_THRESHOLD,
            "ADI should be >= 1.32, got {}",
            diag.adi
        );
        assert!(
            diag.cv_squared < CV2_THRESHOLD,
            "CV^2 should be < 0.49, got {}",
            diag.cv_squared
        );
    }

    #[test]
    fn lumpy_data_classified_correctly() {
        // Many zeros (high ADI) AND highly variable sizes (high CV^2)
        let data = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 80.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert_eq!(diag.classification, DemandClassification::Lumpy);
        assert!(
            diag.adi >= ADI_THRESHOLD,
            "ADI should be >= 1.32, got {}",
            diag.adi
        );
        assert!(
            diag.cv_squared >= CV2_THRESHOLD,
            "CV^2 should be >= 0.49, got {}",
            diag.cv_squared
        );
    }

    // ==================== ADI computation ====================

    #[test]
    fn adi_all_nonzero() {
        // Every period has demand: ADI = n / n = 1.0
        let data = vec![5.0, 3.0, 4.0, 6.0, 2.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert!(
            (diag.adi - 1.0).abs() < 1e-10,
            "ADI should be 1.0, got {}",
            diag.adi
        );
    }

    #[test]
    fn adi_half_zeros() {
        // 5 non-zero in 10 periods: ADI = 10/5 = 2.0
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert!(
            (diag.adi - 2.0).abs() < 1e-10,
            "ADI should be 2.0, got {}",
            diag.adi
        );
    }

    #[test]
    fn adi_known_pattern() {
        // 3 non-zero in 12 periods: ADI = 12/3 = 4.0
        let data = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert!(
            (diag.adi - 4.0).abs() < 1e-10,
            "ADI should be 4.0, got {}",
            diag.adi
        );
    }

    // ==================== CV squared computation ====================

    #[test]
    fn cv_squared_constant_demand() {
        // All non-zero demands identical: CV^2 = 0
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert!(
            diag.cv_squared.abs() < 1e-10,
            "CV^2 should be 0 for constant demand, got {}",
            diag.cv_squared
        );
    }

    #[test]
    fn cv_squared_known_values() {
        // Non-zero values: [2, 4, 6, 8]
        // mean = 5.0
        // sample variance = [(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2] / 3 = (9+1+1+9)/3 = 20/3
        // CV^2 = (20/3) / 25 = 20/75 = 4/15 ~ 0.2667
        let data = vec![2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        let expected = 4.0 / 15.0;
        assert!(
            (diag.cv_squared - expected).abs() < 1e-10,
            "CV^2 should be {:.6}, got {:.6}",
            expected,
            diag.cv_squared
        );
    }

    #[test]
    fn cv_squared_high_variability() {
        // Very different demand sizes should give high CV^2
        let data = vec![1.0, 100.0, 1.0, 100.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert!(
            diag.cv_squared >= CV2_THRESHOLD,
            "CV^2 should be high for variable demands, got {}",
            diag.cv_squared
        );
    }

    // ==================== Recommended model ====================

    #[test]
    fn recommended_model_matches_classification() {
        // Smooth
        let smooth = IntermittentDiagnostics {
            adi: 1.0,
            cv_squared: 0.1,
            classification: DemandClassification::Smooth,
            zero_fraction: 0.0,
            coverage_rate: None,
            bias: 0.0,
            periods_in_stock: Vec::new(),
        };
        assert_eq!(smooth.recommended_model(), "SES or ARIMA");

        // Erratic
        let erratic = IntermittentDiagnostics {
            classification: DemandClassification::Erratic,
            ..smooth.clone()
        };
        assert_eq!(erratic.recommended_model(), "Croston");

        // Intermittent
        let intermittent = IntermittentDiagnostics {
            classification: DemandClassification::Intermittent,
            ..smooth.clone()
        };
        assert_eq!(intermittent.recommended_model(), "Croston or SBA");

        // Lumpy
        let lumpy = IntermittentDiagnostics {
            classification: DemandClassification::Lumpy,
            ..smooth.clone()
        };
        assert_eq!(lumpy.recommended_model(), "TSB");
    }

    // ==================== Coverage rate ====================

    #[test]
    fn coverage_rate_all_covered() {
        let actual = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let forecast = vec![4.0, 1.0, 3.0, 1.0, 4.0];
        let lower = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let upper = vec![10.0, 10.0, 10.0, 10.0, 10.0];

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
        assert!(
            (diag.coverage_rate.unwrap() - 1.0).abs() < 1e-10,
            "All actuals within wide intervals should give coverage 1.0"
        );
    }

    #[test]
    fn coverage_rate_none_covered() {
        let actual = vec![10.0, 20.0, 30.0];
        let forecast = vec![1.0, 1.0, 1.0];
        let lower = vec![0.0, 0.0, 0.0];
        let upper = vec![5.0, 5.0, 5.0];

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
        assert!(
            diag.coverage_rate.unwrap().abs() < 1e-10,
            "No actuals within intervals should give coverage 0.0"
        );
    }

    #[test]
    fn coverage_rate_partial() {
        let actual = vec![5.0, 15.0, 3.0, 20.0];
        let forecast = vec![5.0, 5.0, 5.0, 5.0];
        let lower = vec![0.0, 0.0, 0.0, 0.0];
        let upper = vec![10.0, 10.0, 10.0, 10.0];
        // actual[0]=5 in [0,10] yes; actual[1]=15 in [0,10] no;
        // actual[2]=3 in [0,10] yes; actual[3]=20 in [0,10] no
        // coverage = 2/4 = 0.5

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
        assert!(
            (diag.coverage_rate.unwrap() - 0.5).abs() < 1e-10,
            "Coverage should be 0.5, got {}",
            diag.coverage_rate.unwrap()
        );
    }

    // ==================== PIS computation ====================

    #[test]
    fn pis_perfect_forecast() {
        let actual = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let forecast = vec![5.0, 0.0, 3.0, 0.0, 4.0];

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        assert_eq!(diag.periods_in_stock.len(), 5);
        for &pis in &diag.periods_in_stock {
            assert!(
                pis.abs() < 1e-10,
                "PIS should be 0 for perfect forecast, got {}",
                pis
            );
        }
    }

    #[test]
    fn pis_constant_overforecast() {
        // Forecast always 2 more than actual
        let actual = vec![3.0, 3.0, 3.0, 3.0];
        let forecast = vec![5.0, 5.0, 5.0, 5.0];

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        // PIS accumulates: 2, 4, 6, 8
        assert_eq!(diag.periods_in_stock.len(), 4);
        assert!((diag.periods_in_stock[0] - 2.0).abs() < 1e-10);
        assert!((diag.periods_in_stock[1] - 4.0).abs() < 1e-10);
        assert!((diag.periods_in_stock[2] - 6.0).abs() < 1e-10);
        assert!((diag.periods_in_stock[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn pis_underforecast() {
        // Forecast always 1 less than actual
        let actual = vec![5.0, 5.0, 5.0];
        let forecast = vec![4.0, 4.0, 4.0];

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        // PIS accumulates: -1, -2, -3
        assert!((diag.periods_in_stock[0] - (-1.0)).abs() < 1e-10);
        assert!((diag.periods_in_stock[1] - (-2.0)).abs() < 1e-10);
        assert!((diag.periods_in_stock[2] - (-3.0)).abs() < 1e-10);
    }

    // ==================== Bias computation ====================

    #[test]
    fn bias_perfect_forecast() {
        let actual = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let forecast = vec![5.0, 1.0, 3.0, 1.0, 4.0];
        // Bias is computed only over non-zero actual periods:
        // (5-5) + (3-3) + (4-4) = 0 / 3 = 0

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        assert!(
            diag.bias.abs() < 1e-10,
            "Bias should be 0 for exact match on non-zero periods, got {}",
            diag.bias
        );
    }

    #[test]
    fn bias_overforecast() {
        let actual = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let forecast = vec![7.0, 1.0, 5.0, 1.0, 6.0];
        // Non-zero periods: (7-5)=2, (5-3)=2, (6-4)=2 => mean = 2.0

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        assert!(
            (diag.bias - 2.0).abs() < 1e-10,
            "Bias should be 2.0, got {}",
            diag.bias
        );
    }

    // ==================== Edge cases ====================

    #[test]
    fn edge_case_all_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert!(
            diag.adi.is_infinite(),
            "ADI should be infinite for all zeros"
        );
        assert!(
            diag.cv_squared.abs() < 1e-10,
            "CV^2 should be 0 for all zeros"
        );
        assert!(
            (diag.zero_fraction - 1.0).abs() < 1e-10,
            "Zero fraction should be 1.0"
        );
        // With no non-zero demands, classification goes to Lumpy
        // (ADI = inf >= 1.32, CV^2 = 0 < 0.49 => Intermittent)
        // Actually: inf >= 1.32 is true, 0.0 < 0.49 is true => Intermittent
        assert_eq!(diag.classification, DemandClassification::Intermittent);
    }

    #[test]
    fn edge_case_all_nonzero() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert!(
            (diag.adi - 1.0).abs() < 1e-10,
            "ADI should be 1.0 for all non-zero"
        );
        assert!(
            (diag.zero_fraction - 0.0).abs() < 1e-10,
            "Zero fraction should be 0.0"
        );
        assert_eq!(diag.classification, DemandClassification::Smooth);
    }

    #[test]
    fn edge_case_single_observation_nonzero() {
        let data = vec![5.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert!(
            (diag.adi - 1.0).abs() < 1e-10,
            "ADI should be 1.0 for single non-zero observation"
        );
        // CV^2 should be 0 (< 2 non-zero values)
        assert!(
            diag.cv_squared.abs() < 1e-10,
            "CV^2 should be 0 for single observation"
        );
        assert_eq!(diag.classification, DemandClassification::Smooth);
    }

    #[test]
    fn edge_case_single_observation_zero() {
        let data = vec![0.0];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert!(diag.adi.is_infinite());
        assert!(diag.cv_squared.abs() < 1e-10);
        assert!((diag.zero_fraction - 1.0).abs() < 1e-10);
    }

    #[test]
    fn edge_case_empty_data() {
        let data: Vec<f64> = vec![];
        let diag = IntermittentDiagnostics::from_data(&data);

        assert!(diag.adi.is_infinite());
        assert!(diag.cv_squared.abs() < 1e-10);
        assert!(diag.zero_fraction.abs() < 1e-10);
    }

    // ==================== Zero fraction ====================

    #[test]
    fn zero_fraction_computed_correctly() {
        let data = vec![0.0, 5.0, 0.0, 0.0, 3.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        // 3 zeros out of 5
        assert!(
            (diag.zero_fraction - 0.6).abs() < 1e-10,
            "Zero fraction should be 0.6, got {}",
            diag.zero_fraction
        );
    }

    // ==================== Summary and Display ====================

    #[test]
    fn summary_contains_key_info() {
        let data = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        let s = diag.summary();

        assert!(s.contains("Intermittent Demand Diagnostics"));
        assert!(s.contains("ADI:"));
        assert!(s.contains("CV squared:"));
        assert!(s.contains("Classification:"));
        assert!(s.contains("Zero fraction:"));
        assert!(s.contains("Recommended model:"));
    }

    #[test]
    fn display_matches_summary() {
        let data = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let diag = IntermittentDiagnostics::from_data(&data);
        assert_eq!(format!("{}", diag), diag.summary());
    }

    // ==================== with_intervals includes forecast metrics ====================

    #[test]
    fn with_intervals_includes_bias_and_pis() {
        let actual = vec![5.0, 0.0, 3.0];
        let forecast = vec![6.0, 1.0, 4.0];
        let lower = vec![0.0, 0.0, 0.0];
        let upper = vec![10.0, 10.0, 10.0];

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);

        // bias on non-zero actual: (6-5)=1, (4-3)=1 => 1.0
        assert!(
            (diag.bias - 1.0).abs() < 1e-10,
            "Bias should be 1.0, got {}",
            diag.bias
        );
        // PIS: cum(f-a) = [1, 2, 3]
        assert_eq!(diag.periods_in_stock.len(), 3);
        assert!((diag.periods_in_stock[2] - 3.0).abs() < 1e-10);
        // Coverage: all within [0, 10]
        assert!((diag.coverage_rate.unwrap() - 1.0).abs() < 1e-10);
    }

    // ==================== Classification boundary tests ====================

    #[test]
    fn classification_at_boundaries() {
        // Test exact classification logic
        assert_eq!(
            IntermittentDiagnostics::classify(1.0, 0.2),
            DemandClassification::Smooth
        );
        assert_eq!(
            IntermittentDiagnostics::classify(1.0, 0.5),
            DemandClassification::Erratic
        );
        assert_eq!(
            IntermittentDiagnostics::classify(2.0, 0.2),
            DemandClassification::Intermittent
        );
        assert_eq!(
            IntermittentDiagnostics::classify(2.0, 0.5),
            DemandClassification::Lumpy
        );
    }

    #[test]
    fn classification_at_exact_thresholds() {
        // ADI exactly at 1.32 (>= threshold) and CV^2 exactly at 0.49 (>= threshold)
        assert_eq!(
            IntermittentDiagnostics::classify(ADI_THRESHOLD, CV2_THRESHOLD),
            DemandClassification::Lumpy
        );
        // ADI just below threshold
        assert_eq!(
            IntermittentDiagnostics::classify(ADI_THRESHOLD - 0.001, CV2_THRESHOLD),
            DemandClassification::Erratic
        );
        // CV^2 just below threshold
        assert_eq!(
            IntermittentDiagnostics::classify(ADI_THRESHOLD, CV2_THRESHOLD - 0.001),
            DemandClassification::Intermittent
        );
    }

    // ==================== Coverage with empty intervals ====================

    #[test]
    fn coverage_rate_empty_data() {
        let actual: Vec<f64> = vec![];
        let forecast: Vec<f64> = vec![];
        let lower: Vec<f64> = vec![];
        let upper: Vec<f64> = vec![];

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
        assert!((diag.coverage_rate.unwrap() - 0.0).abs() < 1e-10);
    }

    // ==================== Mismatched lengths ====================

    #[test]
    fn with_forecast_mismatched_lengths() {
        // forecast shorter than actual: only computes over min length
        let actual = vec![5.0, 0.0, 3.0, 0.0, 4.0];
        let forecast = vec![5.0, 0.0, 3.0];

        let diag = IntermittentDiagnostics::with_forecast(&actual, &forecast);
        assert_eq!(diag.periods_in_stock.len(), 3);
    }

    #[test]
    fn with_intervals_mismatched_lengths() {
        let actual = vec![5.0, 3.0, 4.0];
        let forecast = vec![5.0, 3.0];
        let lower = vec![0.0, 0.0, 0.0, 0.0];
        let upper = vec![10.0];

        let diag = IntermittentDiagnostics::with_intervals(&actual, &forecast, &lower, &upper);
        // Min length is 1 (upper)
        assert!(diag.coverage_rate.is_some());
    }
}
