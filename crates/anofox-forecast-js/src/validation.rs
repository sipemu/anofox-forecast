//! Statistical validation tests for JavaScript.
//!
//! Exposes residual diagnostic tests and stationarity tests for
//! validating time series models.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::validation;

// =============================================================================
// Residual diagnostic tests
// =============================================================================

/// Result of the Ljung-Box test, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LjungBoxResultJs {
    /// Test statistic Q.
    statistic: f64,
    /// P-value (approximate).
    p_value: f64,
    /// Number of lags tested.
    lags: usize,
    /// Degrees of freedom.
    df: usize,
    /// Whether residuals appear to be white noise at 5% significance.
    is_white_noise: bool,
}

/// Perform the Ljung-Box test for autocorrelation in residuals.
///
/// Tests the null hypothesis that residuals are independently distributed
/// (white noise). A low p-value suggests significant autocorrelation remains.
///
/// @param residuals - Array of model residuals
/// @param lags - Number of lags to include (default: min(10, n/5))
/// @param fittedParams - Number of fitted model parameters (for df adjustment, default: 0)
/// @returns Object with `statistic`, `pValue`, `lags`, `df`, and `isWhiteNoise`
#[wasm_bindgen(js_name = ljungBox)]
pub fn ljung_box(
    residuals: &[f64],
    lags: Option<usize>,
    fitted_params: Option<usize>,
) -> Result<JsValue, JsError> {
    let result = validation::ljung_box(residuals, lags, fitted_params.unwrap_or(0));

    let js_result = LjungBoxResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        lags: result.lags,
        df: result.df,
        is_white_noise: result.is_white_noise(0.05),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Result of the Durbin-Watson test, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DurbinWatsonResultJs {
    /// Test statistic (0 to 4).
    statistic: f64,
    /// Interpretation: "positiveStrong", "positiveWeak", "none", "negativeWeak", "negativeStrong".
    interpretation: String,
}

/// Perform the Durbin-Watson test for first-order autocorrelation.
///
/// The statistic ranges from 0 to 4:
/// - Near 0: Strong positive autocorrelation
/// - Near 2: No autocorrelation
/// - Near 4: Strong negative autocorrelation
///
/// @param residuals - Array of model residuals
/// @returns Object with `statistic` and `interpretation`
#[wasm_bindgen(js_name = durbinWatson)]
pub fn durbin_watson(residuals: &[f64]) -> Result<JsValue, JsError> {
    let result = validation::durbin_watson(residuals);

    let interpretation = match result.interpretation {
        validation::AutocorrelationType::PositiveStrong => "positiveStrong",
        validation::AutocorrelationType::PositiveWeak => "positiveWeak",
        validation::AutocorrelationType::None => "none",
        validation::AutocorrelationType::NegativeWeak => "negativeWeak",
        validation::AutocorrelationType::NegativeStrong => "negativeStrong",
    };

    let js_result = DurbinWatsonResultJs {
        statistic: result.statistic,
        interpretation: interpretation.to_string(),
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Jarque-Bera test result, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JarqueBeraResultJs {
    /// Test statistic.
    statistic: f64,
    /// P-value (chi-squared with df=2).
    p_value: f64,
    /// Whether residuals appear normally distributed at 5% significance.
    is_normal: bool,
    /// Skewness of the residuals.
    skewness: f64,
    /// Excess kurtosis of the residuals.
    excess_kurtosis: f64,
}

/// Perform the Jarque-Bera test for normality of residuals.
///
/// Tests the null hypothesis that residuals are normally distributed by
/// examining skewness and kurtosis. A high statistic (low p-value) suggests
/// non-normality.
///
/// @param residuals - Array of model residuals
/// @returns Object with `statistic`, `pValue`, `isNormal`, `skewness`, and `excessKurtosis`
#[wasm_bindgen(js_name = jarqueBera)]
pub fn jarque_bera(residuals: &[f64]) -> Result<JsValue, JsError> {
    let result = validation::jarque_bera(residuals);

    let js_result = JarqueBeraResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        is_normal: result.is_normal(0.05),
        skewness: result.skewness,
        excess_kurtosis: result.excess_kurtosis,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Comprehensive residual diagnostics result.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnoseResidualsResultJs {
    /// Ljung-Box test results.
    ljung_box: LjungBoxResultJs,
    /// Durbin-Watson test results.
    durbin_watson: DurbinWatsonResultJs,
    /// Jarque-Bera test results.
    jarque_bera: JarqueBeraResultJs,
    /// Residual mean.
    mean: f64,
    /// Residual variance.
    variance: f64,
    /// Number of residuals.
    n: usize,
    /// Overall assessment: true if residuals pass all tests at 5% significance.
    is_adequate: bool,
}

/// Run comprehensive residual diagnostics.
///
/// Combines Ljung-Box (autocorrelation), Durbin-Watson (first-order autocorrelation),
/// and Jarque-Bera (normality) tests into a single diagnostic report.
///
/// @param residuals - Array of model residuals
/// @param fittedParams - Number of fitted model parameters (default: 0)
/// @returns Object with `ljungBox`, `durbinWatson`, `jarqueBera`, `mean`, `variance`, `n`, and `isAdequate`
#[wasm_bindgen(js_name = diagnoseResiduals)]
pub fn diagnose_residuals(
    residuals: &[f64],
    fitted_params: Option<usize>,
) -> Result<JsValue, JsError> {
    let diag = validation::diagnose_residuals(residuals, fitted_params.unwrap_or(0));

    let lb_interpretation = match diag.durbin_watson.interpretation {
        validation::AutocorrelationType::PositiveStrong => "positiveStrong",
        validation::AutocorrelationType::PositiveWeak => "positiveWeak",
        validation::AutocorrelationType::None => "none",
        validation::AutocorrelationType::NegativeWeak => "negativeWeak",
        validation::AutocorrelationType::NegativeStrong => "negativeStrong",
    };

    let result = DiagnoseResidualsResultJs {
        ljung_box: LjungBoxResultJs {
            statistic: diag.ljung_box.statistic,
            p_value: diag.ljung_box.p_value,
            lags: diag.ljung_box.lags,
            df: diag.ljung_box.df,
            is_white_noise: diag.ljung_box.is_white_noise(0.05),
        },
        durbin_watson: DurbinWatsonResultJs {
            statistic: diag.durbin_watson.statistic,
            interpretation: lb_interpretation.to_string(),
        },
        jarque_bera: JarqueBeraResultJs {
            statistic: diag.jarque_bera.statistic,
            p_value: diag.jarque_bera.p_value,
            is_normal: diag.jarque_bera.is_normal(0.05),
            skewness: diag.jarque_bera.skewness,
            excess_kurtosis: diag.jarque_bera.excess_kurtosis,
        },
        mean: diag.mean,
        variance: diag.variance,
        n: diag.n,
        is_adequate: diag.is_adequate(0.05),
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

// =============================================================================
// Stationarity tests
// =============================================================================

/// Result of a stationarity test, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StationarityResultJs {
    /// Test statistic.
    statistic: f64,
    /// P-value (approximate).
    p_value: f64,
    /// Number of lags used.
    lags: usize,
    /// Whether the series appears stationary.
    is_stationary: bool,
    /// Critical values.
    critical_values: CriticalValuesJs,
}

/// Critical values at common significance levels.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CriticalValuesJs {
    /// Critical value at 1% significance.
    cv_1pct: f64,
    /// Critical value at 5% significance.
    cv_5pct: f64,
    /// Critical value at 10% significance.
    cv_10pct: f64,
}

/// Perform the Augmented Dickey-Fuller (ADF) test for unit root.
///
/// Tests the null hypothesis that the series has a unit root (non-stationary).
/// Rejection (low p-value, negative statistic below critical values) implies
/// stationarity.
///
/// @param values - Array of numeric values
/// @param maxLags - Maximum lags to include (default: (n-1)^(1/3))
/// @returns Object with `statistic`, `pValue`, `lags`, `isStationary`, and `criticalValues`
#[wasm_bindgen(js_name = adfTest)]
pub fn adf_test(values: &[f64], max_lags: Option<usize>) -> Result<JsValue, JsError> {
    let result = validation::adf_test(values, max_lags);

    let js_result = StationarityResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        lags: result.lags,
        is_stationary: result.is_stationary,
        critical_values: CriticalValuesJs {
            cv_1pct: result.critical_values.cv_1pct,
            cv_5pct: result.critical_values.cv_5pct,
            cv_10pct: result.critical_values.cv_10pct,
        },
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Perform the KPSS test for stationarity.
///
/// Tests the null hypothesis that the series is (level) stationary.
/// Rejection (high statistic above critical values) implies non-stationarity.
///
/// Note: ADF and KPSS test opposite null hypotheses:
/// - ADF: H0 = non-stationary, reject = stationary
/// - KPSS: H0 = stationary, reject = non-stationary
///
/// @param values - Array of numeric values
/// @param lags - Number of lags for HAC variance (default: 4*(n/100)^0.25)
/// @returns Object with `statistic`, `pValue`, `lags`, `isStationary`, and `criticalValues`
#[wasm_bindgen(js_name = kpssTest)]
pub fn kpss_test(values: &[f64], lags: Option<usize>) -> Result<JsValue, JsError> {
    let result = validation::kpss_test(values, lags);

    let js_result = StationarityResultJs {
        statistic: result.statistic,
        p_value: result.p_value,
        lags: result.lags,
        is_stationary: result.is_stationary,
        critical_values: CriticalValuesJs {
            cv_1pct: result.critical_values.cv_1pct,
            cv_5pct: result.critical_values.cv_5pct,
            cv_10pct: result.critical_values.cv_10pct,
        },
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_ljung_box() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let result = ljung_box(&residuals, Some(10), None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_durbin_watson() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let result = durbin_watson(&residuals);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_jarque_bera() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let result = jarque_bera(&residuals);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_diagnose_residuals() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let result = diagnose_residuals(&residuals, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_adf_test() {
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let result = adf_test(&series, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_kpss_test() {
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let result = kpss_test(&series, None);
        assert!(result.is_ok());
    }
}
