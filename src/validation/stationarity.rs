//! Stationarity tests for time series.
//!
//! Provides tests to determine if a time series is stationary.
//!
//! The Augmented Dickey-Fuller (ADF) test (`adf_test` / `adf_test_with_options`)
//! regresses `Δy_t` on a deterministic term (none / constant / constant + linear
//! trend), `y_{t-1}` and `lag` lagged differences `Δy_{t-1}..Δy_{t-lag}`, and
//! reports the t-statistic on `y_{t-1}` together with MacKinnon (1994, 2010)
//! p-values and finite-sample critical values — matching `statsmodels.tsa.stattools.adfuller`
//! and `urca::ur.df` to numerical tolerance (see `tests/adf_kpss_reference.rs`).
//! `lags` in the returned [`StationarityResult`] may be `0` when AIC/BIC/t-stat
//! selection picks no augmentation at all.
//!
//! The KPSS test (`kpss_test`) p-value matches `tseries::kpss.test`
//! (`null = "Level"`, `lshort = TRUE`) exactly: linear interpolation within a
//! four-point critical-value table, clamped to `[0.01, 0.10]` — `tseries`
//! never reports a p-value outside that range, and neither does this crate.

use statrs::distribution::{ContinuousCDF, Normal};

/// Result of a stationarity test.
#[derive(Debug, Clone)]
pub struct StationarityResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value (approximate)
    pub p_value: f64,
    /// Number of lags used
    pub lags: usize,
    /// Whether series appears stationary
    pub is_stationary: bool,
    /// Critical values at common significance levels
    pub critical_values: CriticalValues,
}

/// Critical values for stationarity tests.
#[derive(Debug, Clone, Default)]
pub struct CriticalValues {
    /// Critical value at 1% significance
    pub cv_1pct: f64,
    /// Critical value at 5% significance
    pub cv_5pct: f64,
    /// Critical value at 10% significance
    pub cv_10pct: f64,
}

fn nan_result(lags: usize) -> StationarityResult {
    StationarityResult {
        statistic: f64::NAN,
        p_value: f64::NAN,
        lags,
        is_stationary: false,
        critical_values: CriticalValues::default(),
    }
}

// ============================================================================
// ADF regression types
// ============================================================================

/// Deterministic regressors included in the ADF regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdfRegression {
    /// No constant, no trend (`statsmodels` "n").
    NoConstant,
    /// Constant only (`statsmodels` "c"). Default.
    #[default]
    Constant,
    /// Constant and linear trend (`statsmodels` "ct").
    ConstantTrend,
}

impl AdfRegression {
    /// Number of deterministic regressors (`ntrend` in statsmodels).
    fn ntrend(self) -> usize {
        match self {
            AdfRegression::NoConstant => 0,
            AdfRegression::Constant => 1,
            AdfRegression::ConstantTrend => 2,
        }
    }

    /// MacKinnon table key ("n" / "c" / "ct").
    fn table_key(self) -> &'static str {
        match self {
            AdfRegression::NoConstant => "n",
            AdfRegression::Constant => "c",
            AdfRegression::ConstantTrend => "ct",
        }
    }
}

/// Lag-order selection method for the ADF regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdfLagSelection {
    /// Minimize Akaike Information Criterion over lags `0..=max_lags`. Default.
    #[default]
    Aic,
    /// Minimize Bayesian Information Criterion over lags `0..=max_lags`.
    Bic,
    /// Start at `max_lags`, drop the highest lag while its t-statistic is
    /// insignificant at the 5% two-sided normal threshold (statsmodels' "t-stat").
    TStat,
    /// Use `max_lags` as a fixed lag order (no search).
    Fixed,
}

/// Options controlling [`adf_test_with_options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AdfOptions {
    /// Deterministic regressors.
    pub regression: AdfRegression,
    /// Maximum lag order considered (searched methods) or the fixed lag
    /// (`Fixed`). `None` defaults to `floor((n-1)^(1/3))`.
    pub max_lags: Option<usize>,
    /// Lag-order selection method.
    pub lag_selection: AdfLagSelection,
}

impl AdfOptions {
    /// Set the deterministic regression type.
    pub fn with_regression(mut self, regression: AdfRegression) -> Self {
        self.regression = regression;
        self
    }

    /// Set the maximum (or fixed) lag order.
    pub fn with_max_lags(mut self, max_lags: usize) -> Self {
        self.max_lags = Some(max_lags);
        self
    }

    /// Set the lag-order selection method.
    pub fn with_lag_selection(mut self, lag_selection: AdfLagSelection) -> Self {
        self.lag_selection = lag_selection;
        self
    }
}

/// Default lag bound used by statsmodels/urca when the caller does not fix one:
/// `floor((n-1)^(1/3))`.
fn default_max_lag(n: usize) -> usize {
    (((n - 1) as f64).powf(1.0 / 3.0)).floor().max(0.0) as usize
}

/// Clamp a requested max lag the way statsmodels does:
/// `min(n/2 - ntrend - 1, requested)`, floored at 0.
fn clamp_max_lag(n: usize, ntrend: usize, requested: usize) -> usize {
    let bound = (n / 2).saturating_sub(ntrend).saturating_sub(1);
    requested.min(bound)
}

// ============================================================================
// OLS core (Cholesky on the normal equations; no new dependency)
// ============================================================================

/// Fitted OLS result: coefficients, residual sum of squares, sample size,
/// regressor count and per-coefficient standard errors.
struct OlsFit {
    beta: Vec<f64>,
    ssr: f64,
    nobs: usize,
    k: usize,
    se: Vec<f64>,
}

/// Cholesky factor (lower triangular, row-major `k*k`) of a symmetric
/// positive-definite `k*k` matrix, or `None` if not PD.
fn cholesky_factor(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0;
            for m in 0..j {
                sum += l[i * k + m] * l[j * k + m];
            }
            if i == j {
                let diag = a[i * k + i] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[i * k + j] = diag.sqrt();
            } else {
                l[i * k + j] = (a[i * k + j] - sum) / l[j * k + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` given the Cholesky factor `l`.
fn cholesky_solve(l: &[f64], k: usize, b: &[f64]) -> Vec<f64> {
    // Forward: L y = b
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * k + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * k + i];
    }
    // Backward: L' x = y
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..k {
            sum += l[j * k + i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i * k + i];
    }
    x
}

/// Fit `y = X beta + e` by OLS. `x_cols[j]` is the j-th regressor's values
/// (length `nobs`, same order as `y`). Returns `None` if `nobs <= k`, the
/// normal equations are singular, or the residual degrees of freedom are
/// non-positive.
fn ols_fit(y: &[f64], x_cols: &[Vec<f64>]) -> Option<OlsFit> {
    let nobs = y.len();
    let k = x_cols.len();
    if k == 0 || nobs <= k {
        return None;
    }
    for col in x_cols {
        if col.len() != nobs {
            return None;
        }
    }

    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    for i in 0..nobs {
        for a in 0..k {
            let xa = x_cols[a][i];
            xty[a] += xa * y[i];
            for b in 0..=a {
                xtx[a * k + b] += xa * x_cols[b][i];
            }
        }
    }
    // Mirror the lower triangle into the upper triangle.
    for a in 0..k {
        for b in (a + 1)..k {
            xtx[a * k + b] = xtx[b * k + a];
        }
    }

    let l = cholesky_factor(&xtx, k)?;
    let beta = cholesky_solve(&l, k, &xty);

    let mut ssr = 0.0;
    for i in 0..nobs {
        let mut pred = 0.0;
        for a in 0..k {
            pred += beta[a] * x_cols[a][i];
        }
        let resid = y[i] - pred;
        ssr += resid * resid;
    }

    let dof = nobs as i64 - k as i64;
    if dof <= 0 {
        return None;
    }
    let sigma2 = ssr / dof as f64;

    let mut se = vec![0.0; k];
    for j in 0..k {
        let mut e = vec![0.0; k];
        e[j] = 1.0;
        let z = cholesky_solve(&l, k, &e);
        let var_j = sigma2 * z[j];
        se[j] = if var_j.is_finite() && var_j > 0.0 {
            var_j.sqrt()
        } else {
            f64::NAN
        };
    }

    Some(OlsFit {
        beta,
        ssr,
        nobs,
        k,
        se,
    })
}

impl OlsFit {
    fn tvalue(&self, j: usize) -> f64 {
        self.beta[j] / self.se[j]
    }
}

// ============================================================================
// ADF design matrix construction
// ============================================================================

/// Components of the ADF regression over a fixed window of `nobs` rows
/// (the most recent `nobs` observations' first differences).
struct AdfDesign {
    y: Vec<f64>,
    level: Vec<f64>,
    /// `diffs[k-1]` is the column for lag `k` (`Δy_{t-k}`), `k = 1..=maxlag`.
    diffs: Vec<Vec<f64>>,
    trend: Vec<Vec<f64>>,
}

/// Build the ADF regression components for `nobs` rows with up to `maxlag`
/// lagged-difference columns available. `diff[i] = series[i+1] - series[i]`.
fn build_adf_design(
    diff: &[f64],
    series: &[f64],
    regression: AdfRegression,
    maxlag: usize,
    nobs: usize,
) -> AdfDesign {
    let n = series.len();
    let t_start = n - nobs;

    let mut y = Vec::with_capacity(nobs);
    let mut level = Vec::with_capacity(nobs);
    let mut diffs: Vec<Vec<f64>> = (0..maxlag).map(|_| Vec::with_capacity(nobs)).collect();

    for row in 0..nobs {
        let t = t_start + row;
        y.push(diff[t - 1]);
        level.push(series[t - 1]);
        for k in 1..=maxlag {
            diffs[k - 1].push(diff[t - 1 - k]);
        }
    }

    let mut trend: Vec<Vec<f64>> = Vec::new();
    match regression {
        AdfRegression::NoConstant => {}
        AdfRegression::Constant => trend.push(vec![1.0; nobs]),
        AdfRegression::ConstantTrend => {
            trend.push(vec![1.0; nobs]);
            trend.push((1..=nobs).map(|i| i as f64).collect());
        }
    }

    AdfDesign {
        y,
        level,
        diffs,
        trend,
    }
}

/// Columns in statsmodels' autolag-selection order: `[trend..., level, diff_1..diff_lag]`.
fn columns_trend_first(design: &AdfDesign, lag: usize) -> Vec<Vec<f64>> {
    let mut cols = design.trend.clone();
    cols.push(design.level.clone());
    cols.extend(design.diffs[..lag].iter().cloned());
    cols
}

/// Columns in statsmodels' final-regression order: `[level, diff_1..diff_lag, trend...]`.
/// Column 0 is always the level coefficient, regardless of regression type.
fn columns_level_first(design: &AdfDesign, lag: usize) -> Vec<Vec<f64>> {
    let mut cols = Vec::with_capacity(1 + lag + design.trend.len());
    cols.push(design.level.clone());
    cols.extend(design.diffs[..lag].iter().cloned());
    cols.extend(design.trend.clone());
    cols
}

/// `aic = -2*llf + 2*k`, `bic = -2*llf + ln(nobs)*k`,
/// `llf = -nobs/2 * (ln(2π) + ln(ssr/nobs) + 1)` (statsmodels `OLSResults`).
fn information_criterion(fit: &OlsFit, method: AdfLagSelection) -> f64 {
    let nobs = fit.nobs as f64;
    let k = fit.k as f64;
    if fit.ssr <= 0.0 {
        return f64::INFINITY;
    }
    let llf = -nobs / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (fit.ssr / nobs).ln() + 1.0);
    match method {
        AdfLagSelection::Aic => -2.0 * llf + 2.0 * k,
        AdfLagSelection::Bic => -2.0 * llf + nobs.ln() * k,
        _ => f64::INFINITY,
    }
}

/// `t-stat` threshold used by statsmodels' autolag: `norm.ppf(0.95)`.
const T_STAT_THRESHOLD: f64 = 1.6448536269514722;

/// Final ADF regression result: the t-statistic on `y_{t-1}`, its sample size
/// and the number of lagged differences used.
struct AdfFinal {
    statistic: f64,
    nobs: usize,
    lags: usize,
}

/// Run the full ADF procedure (selection + final refit) for the given
/// options, returning `None` if the series is too short for any valid fit.
fn run_adf(series: &[f64], options: &AdfOptions) -> Option<AdfFinal> {
    let n = series.len();
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let ntrend = options.regression.ntrend();

    let chosen_lag = match options.lag_selection {
        AdfLagSelection::Fixed => {
            let requested = options.max_lags.unwrap_or_else(|| default_max_lag(n));
            clamp_max_lag(n, ntrend, requested)
        }
        AdfLagSelection::Aic | AdfLagSelection::Bic | AdfLagSelection::TStat => {
            let requested = options.max_lags.unwrap_or_else(|| default_max_lag(n));
            let maxlag = clamp_max_lag(n, ntrend, requested);
            let nobs_sel = (n as i64 - 1 - maxlag as i64) as usize;
            if (n as i64 - 1 - maxlag as i64) <= 0 {
                return None;
            }
            let design = build_adf_design(&diff, series, options.regression, maxlag, nobs_sel);

            match options.lag_selection {
                AdfLagSelection::Aic | AdfLagSelection::Bic => {
                    let mut best_lag = 0usize;
                    let mut best_ic = f64::INFINITY;
                    for lag in 0..=maxlag {
                        let cols = columns_trend_first(&design, lag);
                        if let Some(fit) = ols_fit(&design.y, &cols) {
                            let ic = information_criterion(&fit, options.lag_selection);
                            if ic < best_ic {
                                best_ic = ic;
                                best_lag = lag;
                            }
                        }
                    }
                    best_lag
                }
                AdfLagSelection::TStat => {
                    let mut best_lag = maxlag;
                    for lag in (0..=maxlag).rev() {
                        let cols = columns_trend_first(&design, lag);
                        if let Some(fit) = ols_fit(&design.y, &cols) {
                            let last = fit.k - 1;
                            let t = fit.tvalue(last);
                            best_lag = lag;
                            if t.is_finite() && t.abs() >= T_STAT_THRESHOLD {
                                break;
                            }
                        } else {
                            best_lag = lag;
                        }
                    }
                    best_lag
                }
                AdfLagSelection::Fixed => unreachable!(),
            }
        }
    };

    let nobs_final = (n as i64 - 1 - chosen_lag as i64) as usize;
    if n < 1 || (n as i64 - 1 - chosen_lag as i64) <= 0 {
        return None;
    }
    let final_design = build_adf_design(&diff, series, options.regression, chosen_lag, nobs_final);
    let final_cols = columns_level_first(&final_design, chosen_lag);
    let fit = ols_fit(&final_design.y, &final_cols)?;
    let statistic = fit.tvalue(0);
    if !statistic.is_finite() {
        return None;
    }

    Some(AdfFinal {
        statistic,
        nobs: fit.nobs,
        lags: chosen_lag,
    })
}

/// Augmented Dickey-Fuller test for unit root (non-stationarity), with
/// explicit control over the deterministic regression and lag selection.
///
/// See the module docs for the regression specification. Statistic, p-value
/// and critical values match `statsmodels.tsa.stattools.adfuller` /
/// `urca::ur.df` to numerical tolerance.
pub fn adf_test_with_options(series: &[f64], options: &AdfOptions) -> StationarityResult {
    let n = series.len();
    if n < 4 {
        return nan_result(0);
    }

    let Some(result) = run_adf(series, options) else {
        return nan_result(0);
    };

    let critical_values = mackinnon_critical_values(options.regression, result.nobs);
    let p_value = mackinnon_p_value(result.statistic, options.regression);
    let is_stationary = result.statistic < critical_values.cv_5pct;

    StationarityResult {
        statistic: result.statistic,
        p_value,
        lags: result.lags,
        is_stationary,
        critical_values,
    }
}

/// Augmented Dickey-Fuller test for unit root (non-stationarity).
///
/// Tests null hypothesis that series has a unit root (non-stationary).
/// Rejection implies stationarity.
///
/// Equivalent to [`adf_test_with_options`] with `AdfOptions::default()`
/// (constant regression, AIC lag selection) and `max_lags` set to the given
/// value (default `floor((n-1)^(1/3))` when `None`).
///
/// # Arguments
/// * `series` - Time series data
/// * `max_lags` - Maximum lags to include (default: (n-1)^(1/3))
///
/// # Returns
/// `StationarityResult` with test statistic and p-value
pub fn adf_test(series: &[f64], max_lags: Option<usize>) -> StationarityResult {
    let mut options = AdfOptions::default();
    if let Some(m) = max_lags {
        options = options.with_max_lags(m);
    }
    adf_test_with_options(series, &options)
}

// ============================================================================
// MacKinnon (1994, 2010) p-values and critical values
// ============================================================================

/// Approximate MacKinnon (1994) p-value for an ADF/cointegration test
/// statistic, `N = 1` (single series believed `I(1)`, i.e. the ADF case).
///
/// Port of `statsmodels.tsa.adfvalues.mackinnonp`. Coefficient tables are
/// statsmodels' own shipped values (see `THIRD_PARTY_NOTICES.md`); matched
/// to the fixture dump in `tests/data/r_reference/adf_statsmodels.json` to
/// 1e-12 (`tests/adf_kpss_reference.rs::mackinnon_tables_match_statsmodels`).
pub fn mackinnon_p_value(statistic: f64, regression: AdfRegression) -> f64 {
    if statistic.is_nan() {
        return f64::NAN;
    }
    let key = regression.table_key();
    let max_stat = TAU_MAX[idx(key)];
    let min_stat = TAU_MIN[idx(key)];
    let star_stat = TAU_STAR[idx(key)];

    if statistic > max_stat {
        return 1.0;
    }
    if statistic < min_stat {
        return 0.0;
    }

    let coef: &[f64] = if statistic <= star_stat {
        &TAU_SMALLP[idx(key)]
    } else {
        &TAU_LARGEP[idx(key)]
    };

    let mut poly = 0.0;
    for &c in coef.iter().rev() {
        poly = poly * statistic + c;
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(poly)
}

/// MacKinnon (2010) finite-sample critical values for the ADF test,
/// `N = 1`, at the given regression's own sample size `nobs`.
///
/// Port of `statsmodels.tsa.adfvalues.mackinnoncrit`.
pub fn mackinnon_critical_values(regression: AdfRegression, nobs: usize) -> CriticalValues {
    let key = idx(regression.table_key());
    let eval = |coef: &[f64; 4]| -> f64 {
        let x = 1.0 / nobs as f64;
        coef[0] + coef[1] * x + coef[2] * x * x + coef[3] * x * x * x
    };
    CriticalValues {
        cv_1pct: eval(&TAU_2010[key][0]),
        cv_5pct: eval(&TAU_2010[key][1]),
        cv_10pct: eval(&TAU_2010[key][2]),
    }
}

fn idx(key: &str) -> usize {
    match key {
        "n" => 0,
        "c" => 1,
        "ct" => 2,
        _ => unreachable!(),
    }
}

// MacKinnon coefficient tables, N = 1 rows only (single series believed
// I(1) — the ADF case). These are statsmodels' own shipped values
// (`statsmodels.tsa.adfvalues`: MacKinnon 1994 p-value surfaces / MacKinnon
// 2010 critical-value surfaces; BSD-3-Clause, see THIRD_PARTY_NOTICES.md),
// dumped programmatically into
// `tests/data/r_reference/adf_statsmodels.json` by
// `validation/reference/python/adf_statsmodels.py` and proven equal to the
// constants below (1e-12) by `mackinnon_tables_match_statsmodels`.
// statsmodels issue #10271 notes transcription-vs-paper discrepancies in a
// few `tau_2010["c"]` cells; this crate reproduces statsmodels' shipped
// values (not the paper) since that is the oracle these tests are proven
// against.
const TAU_MAX: [f64; 3] = [f64::INFINITY, 2.74, 0.7];
const TAU_MIN: [f64; 3] = [-19.04, -18.83, -16.18];
const TAU_STAR: [f64; 3] = [-1.04, -1.61, -2.89];
const TAU_SMALLP: [[f64; 3]; 3] = [
    [0.6344, 1.2378, 0.032496],
    [2.1659, 1.4412, 0.038269],
    [3.2512, 1.6047, 0.049588],
];
const TAU_LARGEP: [[f64; 4]; 3] = [
    [0.4797, 0.93557, -0.06999, 0.033066],
    [1.7339, 0.93202, -0.12745, -0.010368],
    [2.5261, 0.61654, -0.37956, -0.060285],
];
const TAU_2010: [[[f64; 4]; 3]; 3] = [
    [
        [-2.56574, -2.2358, -3.627, 0.0],
        [-1.94100, -0.2686, -3.365, 31.223],
        [-1.61682, 0.2656, -2.714, 25.364],
    ],
    [
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8903, -4.234, -40.040],
        [-2.56677, -1.5384, -2.809, 0.0],
    ],
    [
        [-3.95877, -9.0531, -28.428, -134.155],
        [-3.41049, -4.3904, -9.036, -45.374],
        [-3.12705, -2.5856, -3.925, -22.380],
    ],
];

// ============================================================================
// KPSS
// ============================================================================

/// KPSS test for stationarity.
///
/// Tests null hypothesis that series is (trend) stationary.
/// Rejection implies non-stationarity.
///
/// The p-value matches `tseries::kpss.test(null = "Level", lshort = TRUE)`:
/// linear interpolation within the table `(0.347, 0.10)`, `(0.463, 0.05)`,
/// `(0.574, 0.025)`, `(0.739, 0.01)`, clamped to `[0.01, 0.10]` outside it
/// (see [`kpss_p_value`]).
///
/// # Arguments
/// * `series` - Time series data
/// * `lags` - Number of lags for HAC variance (default: 4*(n/100)^0.25)
///
/// # Returns
/// `StationarityResult` with test statistic and p-value
pub fn kpss_test(series: &[f64], lags: Option<usize>) -> StationarityResult {
    let n = series.len();

    if n < 4 {
        return nan_result(0);
    }

    // Default lag: 4 * (n/100)^0.25
    let lags = lags.unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize);
    let lags = lags.min(n / 2).max(1);

    // Demean the series (level stationarity)
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let residuals: Vec<f64> = series.iter().map(|&x| x - mean).collect();

    // Compute cumulative sum of residuals
    let mut cumsum = vec![0.0; n];
    cumsum[0] = residuals[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + residuals[i];
    }

    // Compute numerator: sum of squared cumulative sums
    let numerator: f64 = cumsum.iter().map(|&s| s * s).sum::<f64>() / (n * n) as f64;

    // Compute HAC variance estimator (Bartlett kernel)
    let mut variance = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    for j in 1..=lags {
        let weight = 1.0 - j as f64 / (lags + 1) as f64;
        let autocovar: f64 = residuals
            .iter()
            .skip(j)
            .zip(residuals.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / n as f64;
        variance += 2.0 * weight * autocovar;
    }

    if variance <= 0.0 {
        return StationarityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags,
            is_stationary: true,
            critical_values: CriticalValues::default(),
        };
    }

    let stat = numerator / variance;

    // Critical values for KPSS level stationarity
    let critical_values = CriticalValues {
        cv_1pct: 0.739,
        cv_5pct: 0.463,
        cv_10pct: 0.347,
    };

    // Approximate p-value
    let p_value = kpss_p_value(stat);

    // Series is stationary if we fail to reject null (stat < critical value)
    let is_stationary = stat < critical_values.cv_5pct;

    StationarityResult {
        statistic: stat,
        p_value,
        lags,
        is_stationary,
        critical_values,
    }
}

/// `tseries::kpss.test`'s four-point table (`null = "Level"`): linear
/// interpolation between these `(statistic, p_value)` pairs, clamped to
/// `[0.01, 0.10]` at both ends (R's `approx(..., rule = 2)` — flat
/// extrapolation, never linear beyond the table). `tseries` never reports a
/// p-value outside this range because the table has no data outside it.
const KPSS_TABLE: [(f64, f64); 4] = [(0.347, 0.10), (0.463, 0.05), (0.574, 0.025), (0.739, 0.01)];

/// P-value for the KPSS test statistic, matching `tseries::kpss.test`
/// (`null = "Level"`, `lshort = TRUE`) exactly: linear interpolation within
/// the four-point critical-value table, clamped to `[0.01, 0.10]` outside
/// it (never extrapolated toward 0 or 1).
pub fn kpss_p_value(stat: f64) -> f64 {
    if stat.is_nan() {
        return f64::NAN;
    }

    if stat <= KPSS_TABLE[0].0 {
        return KPSS_TABLE[0].1;
    }
    if stat >= KPSS_TABLE[KPSS_TABLE.len() - 1].0 {
        return KPSS_TABLE[KPSS_TABLE.len() - 1].1;
    }

    for i in 0..KPSS_TABLE.len() - 1 {
        let (x1, y1) = KPSS_TABLE[i];
        let (x2, y2) = KPSS_TABLE[i + 1];
        if stat >= x1 && stat <= x2 {
            return y1 + (y2 - y1) * (stat - x1) / (x2 - x1);
        }
    }
    unreachable!("stat is within table bounds by the checks above")
}

/// Combined stationarity test using both ADF and KPSS.
///
/// # Returns
/// A tuple of (adf_result, kpss_result, conclusion)
/// where conclusion is:
/// - "stationary" if ADF rejects AND KPSS fails to reject
/// - "non_stationary" if ADF fails to reject AND KPSS rejects
/// - "inconclusive" otherwise
pub fn test_stationarity(series: &[f64]) -> (StationarityResult, StationarityResult, &'static str) {
    let adf = adf_test(series, None);
    let kpss = kpss_test(series, None);

    let conclusion = if adf.is_stationary && kpss.is_stationary {
        "stationary"
    } else if !adf.is_stationary && !kpss.is_stationary {
        "non_stationary"
    } else {
        "inconclusive"
    };

    (adf, kpss, conclusion)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== adf_test ====================

    #[test]
    fn adf_stationary_series() {
        // White noise should be stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        assert!(result.statistic < 0.0); // Should be negative
    }

    #[test]
    fn adf_random_walk() {
        // Random walk should be non-stationary
        let mut series = vec![0.0; 200];
        for i in 1..200 {
            series[i] = series[i - 1] + ((i * 17) % 19) as f64 / 10.0 - 0.9;
        }

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        // Just verify we get a valid result
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn adf_trending_series() {
        // Series with strong trend (add small noise to avoid numerical issues)
        let series: Vec<f64> = (0..200)
            .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.01)
            .collect();

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        // Trending series should fail to reject null
        assert!(!result.is_stationary);
    }

    #[test]
    fn adf_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let result = adf_test(&series, Some(1));

        assert!(result.statistic.is_nan());
    }

    #[test]
    fn adf_empty() {
        let result = adf_test(&[], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn adf_critical_values() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, None);

        assert!(result.critical_values.cv_1pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_10pct);
    }

    // ==================== kpss_test ====================

    #[test]
    fn kpss_stationary_series() {
        // White noise should be stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
        assert!(result.statistic > 0.0);
        // Should fail to reject null (stationary)
        assert!(result.is_stationary);
    }

    #[test]
    fn kpss_trending_series() {
        // Series with strong trend should reject stationarity
        let series: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
        // Should reject null (not stationary)
        assert!(!result.is_stationary);
    }

    #[test]
    fn kpss_random_walk() {
        // Random walk
        let mut series = vec![0.0; 200];
        for i in 1..200 {
            series[i] = series[i - 1] + ((i * 17) % 19) as f64 / 10.0 - 0.9;
        }

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
    }

    #[test]
    fn kpss_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let result = kpss_test(&series, Some(1));

        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_empty() {
        let result = kpss_test(&[], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_critical_values() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, None);

        // KPSS critical values should increase
        assert!(result.critical_values.cv_10pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_1pct);
    }

    // ==================== test_stationarity ====================

    #[test]
    fn combined_test_stationary() {
        // White noise should be conclusively stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let (adf, kpss, conclusion) = test_stationarity(&series);

        assert!(!adf.statistic.is_nan());
        assert!(!kpss.statistic.is_nan());
        // Conclusion depends on specific values
        assert!(conclusion == "stationary" || conclusion == "inconclusive");
    }

    #[test]
    fn combined_test_trending() {
        // Strong trend should be non-stationary (add small noise to avoid numerical issues)
        let series: Vec<f64> = (0..200)
            .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.01)
            .collect();

        let (adf, kpss, conclusion) = test_stationarity(&series);

        assert!(!adf.statistic.is_nan());
        assert!(!kpss.statistic.is_nan());
        // Should be non-stationary or inconclusive
        assert!(conclusion == "non_stationary" || conclusion == "inconclusive");
    }

    #[test]
    fn combined_test_short() {
        let series = vec![1.0, 2.0, 3.0];

        let (adf, kpss, _) = test_stationarity(&series);

        assert!(adf.statistic.is_nan());
        assert!(kpss.statistic.is_nan());
    }

    // ==================== adf_test edge cases ====================

    #[test]
    fn adf_constant_series() {
        // Constant series has zero variance in differences
        let series = vec![5.0; 100];
        let result = adf_test(&series, Some(1));

        // Constant series -> all diffs zero -> NaN or degenerate
        assert!(result.statistic.is_nan() || result.statistic.is_finite());
    }

    #[test]
    fn adf_single_value() {
        let result = adf_test(&[42.0], None);
        assert!(result.statistic.is_nan());
        assert!(!result.is_stationary);
    }

    #[test]
    fn adf_two_values() {
        let result = adf_test(&[1.0, 2.0], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn adf_four_values_minimum() {
        // Exactly 4 values is the minimum for non-NaN
        let series = vec![1.0, 3.0, 2.0, 4.0];
        let result = adf_test(&series, Some(1));
        // Should produce some result (may or may not be NaN depending on regression)
        assert!(result.lags <= 1);
    }

    #[test]
    fn adf_large_lag_clamped() {
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = adf_test(&series, Some(100));
        // max_lags should be clamped to n/2 - ntrend - 1
        assert!(result.lags <= 9);
    }

    #[test]
    fn adf_p_value_range() {
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, None);
        // p-value should be in valid range
        if !result.p_value.is_nan() {
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }
    }

    #[test]
    fn adf_critical_values_ordering() {
        // 1% CV < 5% CV < 10% CV (all negative, becoming less negative)
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, None);
        assert!(result.critical_values.cv_1pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_10pct);
        // All should be negative for ADF
        assert!(result.critical_values.cv_1pct < 0.0);
        assert!(result.critical_values.cv_5pct < 0.0);
        assert!(result.critical_values.cv_10pct < 0.0);
    }

    // ==================== kpss_test edge cases ====================

    #[test]
    fn kpss_constant_series() {
        let series = vec![5.0; 100];
        let result = kpss_test(&series, Some(5));

        // Constant series has zero variance -> NaN or degenerate statistic
        // After demeaning, residuals are all zero
        assert!(result.statistic.is_nan() || result.statistic == 0.0 || result.is_stationary);
    }

    #[test]
    fn kpss_single_value() {
        let result = kpss_test(&[42.0], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_two_values() {
        let result = kpss_test(&[1.0, 2.0], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_lag_selection_default() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, None);
        // Default lag: 4 * (100/100)^0.25 = 4
        assert!(result.lags >= 1);
    }

    #[test]
    fn kpss_critical_values_ordering() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, None);
        // KPSS critical values should be positive and increasing
        assert!(result.critical_values.cv_10pct > 0.0);
        assert!(result.critical_values.cv_10pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_1pct);
    }

    #[test]
    fn kpss_p_value_range() {
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, None);
        if !result.p_value.is_nan() {
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        }
    }

    #[test]
    fn kpss_large_lag_clamped() {
        let series: Vec<f64> = (0..20)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, Some(100));
        // Lags should be clamped to n/2
        assert!(result.lags <= 10);
    }

    // ==================== kpss_p_value internal ====================

    #[test]
    fn kpss_p_value_boundaries() {
        // Bounded like tseries::kpss.test (DIAG-02): p is always in
        // [0.01, 0.10], clamped flat below/above the table, not extrapolated
        // toward 0/1 as the old implementation did.

        // stat < 0.347 -> clamped at p = 0.10 (was p > 0.10 before DIAG-02)
        let p1 = kpss_p_value(0.1);
        assert_eq!(p1, 0.10);

        // stat == 0.347 -> p == 0.10 exactly (table endpoint)
        let p2 = kpss_p_value(0.347);
        assert_eq!(p2, 0.10);

        // stat == 0.463 -> p == 0.05 exactly (table endpoint)
        let p3 = kpss_p_value(0.463);
        assert_eq!(p3, 0.05);

        // stat == 0.739 -> p == 0.01 exactly (table endpoint)
        let p4 = kpss_p_value(0.739);
        assert_eq!(p4, 0.01);

        // stat > 0.739 -> clamped at p = 0.01 (was p < 0.01 before DIAG-02)
        let p5 = kpss_p_value(1.0);
        assert_eq!(p5, 0.01);

        // stat == 0.574 -> p == 0.025 exactly (the table's 2.5% point, which
        // the pre-DIAG-02 implementation skipped entirely)
        let p6 = kpss_p_value(0.574);
        assert_eq!(p6, 0.025);

        // NaN input -> NaN output
        let p_nan = kpss_p_value(f64::NAN);
        assert!(p_nan.is_nan());
    }

    // ==================== test_stationarity edge cases ====================

    #[test]
    fn combined_test_constant() {
        let series = vec![42.0; 100];
        let (adf, kpss, _conclusion) = test_stationarity(&series);

        // Just verify it doesn't panic and returns valid results
        let _ = adf.statistic;
        let _ = kpss.statistic;
        assert!(
            _conclusion == "stationary"
                || _conclusion == "non_stationary"
                || _conclusion == "inconclusive"
        );
    }

    #[test]
    fn combined_test_empty() {
        let (adf, kpss, _) = test_stationarity(&[]);
        assert!(adf.statistic.is_nan());
        assert!(kpss.statistic.is_nan());
    }
}
