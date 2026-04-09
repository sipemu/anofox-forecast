//! Sequential CUSUM detector for monitoring forecast errors.
//!
//! Port of `cptSeqCUSUM` / `updateForecast` from the R package
//! [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast)
//! by Thomas Grundy (Lancaster University), MIT License.
//!
//! The detector consumes a stream of forecast errors, treats the first `m`
//! observations as a training window that defines the "healthy" mean and
//! variance, and then monitors the CUSUM of subsequent errors against a
//! time-varying threshold. The first index where the CUSUM crosses the
//! threshold is the detection time `τ`.
//!
//! See [Fremdt (2014)](https://doi.org/10.1080/02331888.2014.921899) for the
//! Page's CUSUM detector and the asymptotic limit distributions.

use super::sequential_crit::{simulate_critical_value, CriticalValue};
use super::sequential_table::lookup_critical_value;
use crate::core::TimeSeries;
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::cross_validation::{rolling_forecast, RollingForecastConfig};

// ---------------------------------------------------------------------------
// Core enums
// ---------------------------------------------------------------------------

/// Which CUSUM detector to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Detector {
    /// Original CUSUM, two-sided alternative.
    ///
    /// `cusum_k = |Σ_{i=1}^{k} (X_{m+i} − μ̂_m)|`.
    Cusum,
    /// Original CUSUM, one-sided alternative (detects positive drift).
    ///
    /// `cusum_k = Σ_{i=1}^{k} (X_{m+i} − μ̂_m)`.
    Cusum1,
    /// Page's CUSUM, two-sided alternative (recommended default).
    ///
    /// Runs two running maxima and takes the maximum.
    #[default]
    PageCusum,
    /// Page's CUSUM, one-sided alternative.
    PageCusum1,
}

/// Which transformation of the errors to monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForecastErrorType {
    /// Monitor the raw errors only (detects mean changes).
    Raw,
    /// Monitor the centred squared errors only (detects variance changes).
    Squared,
    /// Monitor both streams in parallel (default).
    #[default]
    Both,
}

// ---------------------------------------------------------------------------
// Weight function
// ---------------------------------------------------------------------------

/// Weight function used to rescale the detector threshold over time.
///
/// `w(m, k, γ) = √m · (1 + k/m) · (k / (k + m))^γ` with `0 ≤ γ < 0.5`.
///
/// Ports `weightFun` from R. Larger `γ` pushes detection earlier at the cost
/// of a higher false-alarm rate near the start of monitoring.
#[inline]
pub fn weight(m: usize, k: usize, gamma: f64) -> f64 {
    let m_f = m as f64;
    let k_f = k as f64;
    m_f.sqrt() * (1.0 + k_f / m_f) * (k_f / (k_f + m_f)).powf(gamma)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`SequentialDetector`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SequentialConfig {
    /// Length of the training window (the first `m` errors are assumed
    /// changepoint-free and define the baseline mean/variance).
    pub m: usize,
    /// Which CUSUM detector to use. Default: `PageCusum`.
    pub detector: Detector,
    /// Which error transformation(s) to monitor. Default: `Both`.
    pub error_type: ForecastErrorType,
    /// Weight-function tuning parameter, `0 ≤ γ < 0.5`. Default: `0.0`.
    pub gamma: f64,
    /// Nominal type-I error rate in `(0, 1)`. Default: `0.05`.
    pub alpha: f64,
    /// How to obtain the critical value. Default: `Lookup` (consult the
    /// baked table; fall through to simulation when off-grid).
    pub critical_value: CriticalValue,
    /// Optional override for the training-window variance.
    ///
    /// R uses `var(X[1:m])` which assumes independence. For autocorrelated
    /// residuals, pass a long-run variance estimate via
    /// [`SequentialConfig::with_sigma2`].
    pub sigma2: Option<f64>,
}

impl Default for SequentialConfig {
    fn default() -> Self {
        Self {
            m: 50,
            detector: Detector::default(),
            error_type: ForecastErrorType::default(),
            gamma: 0.0,
            alpha: 0.05,
            critical_value: CriticalValue::default(),
            sigma2: None,
        }
    }
}

impl SequentialConfig {
    /// Create a configuration with training-window length `m`.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            ..Default::default()
        }
    }

    /// Set the training-window length.
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the detector.
    pub fn detector(mut self, detector: Detector) -> Self {
        self.detector = detector;
        self
    }

    /// Set the error transformation.
    pub fn error_type(mut self, error_type: ForecastErrorType) -> Self {
        self.error_type = error_type;
        self
    }

    /// Set the weight-function tuning parameter.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the nominal type-I error rate.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set how the critical value is obtained.
    pub fn critical_value(mut self, cv: CriticalValue) -> Self {
        self.critical_value = cv;
        self
    }

    /// Override the training-window variance.
    ///
    /// Use this when the forecast-error series has non-trivial
    /// autocorrelation and you have a long-run variance estimate
    /// (Newey–West, HAC, etc.).
    pub fn with_sigma2(mut self, sigma2: f64) -> Self {
        self.sigma2 = Some(sigma2);
        self
    }

    fn validate(&self, n: usize) -> Result<()> {
        if self.m < 2 || self.m >= n {
            return Err(ForecastError::InvalidParameter(format!(
                "training-window length m must satisfy 2 ≤ m < n (got m={}, n={})",
                self.m, n
            )));
        }
        if !(0.0..0.5).contains(&self.gamma) {
            return Err(ForecastError::InvalidParameter(format!(
                "gamma must be in [0, 0.5), got {}",
                self.gamma
            )));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(ForecastError::InvalidParameter(format!(
                "alpha must be in [0, 1], got {}",
                self.alpha
            )));
        }
        if let Some(s2) = self.sigma2 {
            if !(s2.is_finite() && s2 > 0.0) {
                return Err(ForecastError::InvalidParameter(format!(
                    "sigma2 override must be a positive finite number, got {}",
                    s2
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Detector state
// ---------------------------------------------------------------------------

/// Per-stream detector state (raw or squared errors).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamState {
    /// Sample mean of the training window.
    pub train_mean: f64,
    /// Sample variance used in the threshold (sample var on training window
    /// unless overridden).
    pub sigma2: f64,
    /// Critical value for the chosen `(detector, gamma, alpha)`.
    pub crit_value: f64,
    /// Running CUSUM value for one-sided detectors, or the "upper" arm of
    /// Page's two-sided detector.
    pub cusum_a: f64,
    /// "Lower" arm of Page's two-sided detector. Unused for the one-sided
    /// variants.
    pub cusum_b: f64,
    /// History of CUSUM values emitted so far (one per monitored error).
    pub cusum: Vec<f64>,
    /// History of threshold values (matches `cusum` in length).
    pub threshold: Vec<f64>,
    /// Index (1-based, relative to the start of monitoring) of the first
    /// exceedance, or `None` if the detector has not fired.
    pub tau: Option<usize>,
}

impl StreamState {
    /// Number of errors this stream has processed (post-training).
    pub fn len(&self) -> usize {
        self.cusum.len()
    }

    /// `true` if no errors have been processed yet.
    pub fn is_empty(&self) -> bool {
        self.cusum.is_empty()
    }
}

/// Sequential CUSUM detector over a stream of forecast errors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SequentialDetector {
    /// Configuration snapshot.
    pub config: SequentialConfig,
    /// Sample mean of the raw training window. Used to re-centre new errors
    /// before feeding them into the squared stream during online updates.
    pub train_mean_raw: f64,
    /// Raw-error stream (present unless `error_type == Squared`).
    pub raw: Option<StreamState>,
    /// Squared-error stream (present if `error_type` is `Squared` or `Both`).
    pub squared: Option<StreamState>,
}

impl SequentialConfig {
    fn resolve_crit_value(&self) -> f64 {
        match self.critical_value {
            CriticalValue::Fixed(v) => v,
            CriticalValue::Lookup => {
                if let Some(v) = lookup_critical_value(self.detector, self.gamma, self.alpha) {
                    v
                } else {
                    // Off-grid: fall through to simulation with a moderate
                    // budget so the user gets a reasonable threshold without
                    // configuring anything.
                    simulate_critical_value(
                        self.detector,
                        self.gamma,
                        self.alpha,
                        2000,
                        500,
                        Some(42),
                    )
                }
            }
            CriticalValue::Simulate {
                samples,
                npts,
                seed,
            } => {
                simulate_critical_value(self.detector, self.gamma, self.alpha, samples, npts, seed)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl SequentialDetector {
    /// Fit the detector on a complete error vector.
    ///
    /// The first `config.m` values define the training window; the remainder
    /// are processed through the detector in order. If the detector crosses
    /// its threshold, `tau` is set to the earliest exceedance index.
    pub fn fit(errors: &[f64], config: SequentialConfig) -> Result<Self> {
        if errors.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        if errors.iter().any(|v| !v.is_finite()) {
            return Err(ForecastError::MissingValues);
        }
        config.validate(errors.len())?;

        let train = &errors[..config.m];
        let train_mean = mean(train);

        // Raw stream
        let raw = if matches!(
            config.error_type,
            ForecastErrorType::Raw | ForecastErrorType::Both
        ) {
            let sigma2 = config
                .sigma2
                .unwrap_or_else(|| sample_variance(train).max(f64::MIN_POSITIVE));
            let crit_value = config.resolve_crit_value();
            let mut state = StreamState {
                train_mean,
                sigma2,
                crit_value,
                cusum_a: 0.0,
                cusum_b: 0.0,
                cusum: Vec::new(),
                threshold: Vec::new(),
                tau: None,
            };
            run_stream(&mut state, &errors[config.m..], 0, &config);
            Some(state)
        } else {
            None
        };

        // Squared stream uses centred squared errors: (e − train_mean)^2.
        let squared = if matches!(
            config.error_type,
            ForecastErrorType::Squared | ForecastErrorType::Both
        ) {
            let sq_train: Vec<f64> = train.iter().map(|&e| (e - train_mean).powi(2)).collect();
            let sq_train_mean = mean(&sq_train);
            let sigma2 = config
                .sigma2
                .unwrap_or_else(|| sample_variance(&sq_train).max(f64::MIN_POSITIVE));
            let crit_value = config.resolve_crit_value();
            let mut state = StreamState {
                train_mean: sq_train_mean,
                sigma2,
                crit_value,
                cusum_a: 0.0,
                cusum_b: 0.0,
                cusum: Vec::new(),
                threshold: Vec::new(),
                tau: None,
            };
            let sq_stream: Vec<f64> = errors[config.m..]
                .iter()
                .map(|&e| (e - train_mean).powi(2))
                .collect();
            run_stream(&mut state, &sq_stream, 0, &config);
            Some(state)
        } else {
            None
        };

        Ok(Self {
            config,
            train_mean_raw: train_mean,
            raw,
            squared,
        })
    }

    /// Feed additional errors into the detector.
    ///
    /// Continues the CUSUM stream from its current state without re-examining
    /// the training window — this is the online path used in production
    /// monitoring. Passing an empty slice is a no-op.
    pub fn update(&mut self, new_errors: &[f64]) -> Result<()> {
        if new_errors.is_empty() {
            return Ok(());
        }
        if new_errors.iter().any(|v| !v.is_finite()) {
            return Err(ForecastError::MissingValues);
        }

        if let Some(state) = self.raw.as_mut() {
            let offset = state.len();
            run_stream(state, new_errors, offset, &self.config);
        }

        if let Some(state) = self.squared.as_mut() {
            let offset = state.len();
            let sq: Vec<f64> = new_errors
                .iter()
                .map(|&e| (e - self.train_mean_raw).powi(2))
                .collect();
            run_stream(state, &sq, offset, &self.config);
        }

        Ok(())
    }

    /// First detection in the raw stream (1-based index relative to monitoring start).
    pub fn tau(&self) -> Option<usize> {
        self.raw.as_ref().and_then(|s| s.tau)
    }

    /// First detection in the squared stream (1-based index relative to monitoring start).
    pub fn tau_squared(&self) -> Option<usize> {
        self.squared.as_ref().and_then(|s| s.tau)
    }

    /// Earliest detection across all active streams.
    pub fn first_detection(&self) -> Option<usize> {
        match (self.tau(), self.tau_squared()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// `true` if any active stream has flagged a changepoint.
    pub fn has_detected(&self) -> bool {
        self.first_detection().is_some()
    }

    /// Full CUSUM history of the raw stream.
    pub fn cusum(&self) -> &[f64] {
        self.raw.as_ref().map(|s| s.cusum.as_slice()).unwrap_or(&[])
    }

    /// Full threshold history of the raw stream.
    pub fn threshold(&self) -> &[f64] {
        self.raw
            .as_ref()
            .map(|s| s.threshold.as_slice())
            .unwrap_or(&[])
    }

    /// Full CUSUM history of the squared stream.
    pub fn cusum_squared(&self) -> &[f64] {
        self.squared
            .as_ref()
            .map(|s| s.cusum.as_slice())
            .unwrap_or(&[])
    }

    /// Full threshold history of the squared stream.
    pub fn threshold_squared(&self) -> &[f64] {
        self.squared
            .as_ref()
            .map(|s| s.threshold.as_slice())
            .unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// Forecaster integration helpers
// ---------------------------------------------------------------------------

/// Monitor a fitted forecaster's **in-sample residuals**.
///
/// Cheap but biased: because the model was trained on the same observations,
/// the sample variance of its fitted residuals understates the true one-step
/// innovation variance, so the detector runs slightly hotter than its nominal
/// `alpha`. Prefer [`monitor_forecaster_cv`] when calibrated false-alarm
/// rates matter.
///
/// Leading non-finite residuals (e.g. `NaN` warmup values that `Naive` and
/// other models emit before they have a prior observation to subtract from)
/// are skipped automatically. An interior `NaN` after the warmup region is
/// still treated as an error.
pub fn monitor_forecaster(
    fc: &dyn Forecaster,
    config: SequentialConfig,
) -> Result<SequentialDetector> {
    let residuals = fc.residuals().ok_or_else(|| ForecastError::FitRequired {
        model: Some(fc.name().to_string()),
    })?;
    let first = residuals
        .iter()
        .position(|v| v.is_finite())
        .ok_or(ForecastError::EmptyData)?;
    SequentialDetector::fit(&residuals[first..], config)
}

/// Monitor a forecaster's **out-of-sample residuals** via rolling-origin CV.
///
/// Uses [`rolling_forecast`] under the hood to produce one-step-ahead
/// forecast errors at every origin, then runs the sequential detector over
/// the concatenated residual stream. This gives the detector an unbiased
/// estimate of the innovation variance and calibrates the nominal `alpha`.
///
/// # Arguments
/// * `factory` — creates a fresh unfitted forecaster for each window
/// * `series` — the full series to evaluate on
/// * `config` — sequential detector configuration (note: `config.m` is the
///   training window *inside the residual stream*, not the CV's
///   `initial_train_size`)
/// * `cv_initial_train_size` — first training-window length for the CV
/// * `cv_horizon` — forecast horizon at each CV origin (typically 1 for
///   one-step residuals, which is what the detector expects)
pub fn monitor_forecaster_cv<F, Factory>(
    factory: Factory,
    series: &TimeSeries,
    config: SequentialConfig,
    cv_initial_train_size: usize,
    cv_horizon: usize,
) -> Result<SequentialDetector>
where
    F: Forecaster + Send,
    Factory: Fn() -> F + Sync,
{
    let cv_config =
        RollingForecastConfig::new(cv_initial_train_size, cv_horizon).step_size(cv_horizon);
    let result = rolling_forecast(series, &cv_config, factory)?;
    let residuals: Vec<f64> = result
        .all_actuals
        .iter()
        .zip(result.all_predictions.iter())
        .map(|(a, p)| a - p)
        .collect();
    SequentialDetector::fit(&residuals, config)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

#[inline]
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Sample variance with Bessel's correction; matches R's `var()`.
#[inline]
fn sample_variance(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let sum_sq: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    sum_sq / (n - 1) as f64
}

/// Run the detector recurrence over `new_errors` for a single stream.
///
/// `offset` is the number of errors already processed by this stream — used
/// to compute the weight-function index `k` for each new observation.
fn run_stream(
    state: &mut StreamState,
    new_errors: &[f64],
    offset: usize,
    config: &SequentialConfig,
) {
    let mean = state.train_mean;
    let m = config.m;
    let gamma = config.gamma;
    let sigma = state.sigma2.sqrt();
    let crit = state.crit_value;
    let scale = crit * sigma;

    state.cusum.reserve(new_errors.len());
    state.threshold.reserve(new_errors.len());

    for (i, &x) in new_errors.iter().enumerate() {
        let k = offset + i + 1; // 1-based index into the monitoring stream
        let value = match config.detector {
            Detector::PageCusum => {
                state.cusum_a = (state.cusum_a + x - mean).max(0.0);
                state.cusum_b = (state.cusum_b - x + mean).max(0.0);
                state.cusum_a.max(state.cusum_b)
            }
            Detector::PageCusum1 => {
                state.cusum_a = (state.cusum_a + x - mean).max(0.0);
                state.cusum_a
            }
            Detector::Cusum => {
                state.cusum_a += x - mean;
                state.cusum_a.abs()
            }
            Detector::Cusum1 => {
                state.cusum_a += x - mean;
                state.cusum_a
            }
        };

        let w = weight(m, k, gamma);
        let thr = w * scale;

        state.cusum.push(value);
        state.threshold.push(thr);

        if state.tau.is_none() && value > thr {
            state.tau = Some(k);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn normal_sample<R: Rng>(rng: &mut R, mean: f64, sd: f64) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.gen();
        mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn weight_at_k_zero_is_sqrt_m() {
        // w(m, 0, γ) = √m · 1 · 0^γ = 0 for γ>0, √m for γ=0
        assert_relative_eq!(weight(100, 0, 0.0), 10.0, epsilon = 1e-10);
        assert_relative_eq!(weight(100, 0, 0.25), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn weight_scaling() {
        // w(m, m, 0) = √m · 2 · 0.5^0 = 2√m
        assert_relative_eq!(weight(100, 100, 0.0), 20.0, epsilon = 1e-10);
    }

    #[test]
    fn fit_empty_errors_rejects() {
        let err = SequentialDetector::fit(&[], SequentialConfig::new(5)).unwrap_err();
        assert_eq!(err, ForecastError::EmptyData);
    }

    #[test]
    fn fit_nan_rejects() {
        let errors = vec![0.0, 1.0, f64::NAN, 0.5, 1.0, 2.0];
        let err = SequentialDetector::fit(&errors, SequentialConfig::new(3)).unwrap_err();
        assert_eq!(err, ForecastError::MissingValues);
    }

    #[test]
    fn fit_rejects_m_too_large() {
        let errors: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let err = SequentialDetector::fit(&errors, SequentialConfig::new(10)).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }

    #[test]
    fn no_change_series_tau_none() {
        let mut rng = rng(1);
        let errors: Vec<f64> = (0..500)
            .map(|_| normal_sample(&mut rng, 0.0, 1.0))
            .collect();
        let cfg = SequentialConfig::new(200)
            .detector(Detector::PageCusum)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Simulate {
                samples: 2000,
                npts: 500,
                seed: Some(1),
            });
        let det = SequentialDetector::fit(&errors, cfg).unwrap();
        // With α=0.05 and n=300 post-training observations, the detector
        // *may* trigger a false alarm ~5% of the time. Seed 1 is benign.
        assert!(
            !det.has_detected(),
            "unexpected detection at τ={:?}",
            det.first_detection()
        );
    }

    #[test]
    fn page_cusum_detects_mean_shift() {
        let mut rng = rng(11);
        let mut errors: Vec<f64> = (0..400)
            .map(|_| normal_sample(&mut rng, 0.0, 1.0))
            .collect();
        errors.extend((0..100).map(|_| normal_sample(&mut rng, 2.0, 1.0)));
        let cfg = SequentialConfig::new(300)
            .detector(Detector::PageCusum)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Simulate {
                samples: 2000,
                npts: 500,
                seed: Some(1),
            });
        let det = SequentialDetector::fit(&errors, cfg).unwrap();
        let tau = det.tau().expect("detector should fire");
        // Shift starts at monitoring index 101 (400 - 300 + 1).
        assert!(
            tau >= 101,
            "tau={} fired before the injected shift (<101)",
            tau
        );
        assert!(tau < 200, "tau={} fired too late after shift", tau);
    }

    #[test]
    fn page_cusum_detects_variance_shift_via_squared_stream() {
        let mut rng = rng(7);
        let mut errors: Vec<f64> = (0..400)
            .map(|_| normal_sample(&mut rng, 0.0, 1.0))
            .collect();
        errors.extend((0..100).map(|_| normal_sample(&mut rng, 0.0, 3.0)));

        let cfg = SequentialConfig::new(300)
            .error_type(ForecastErrorType::Both)
            .critical_value(CriticalValue::Simulate {
                samples: 2000,
                npts: 500,
                seed: Some(1),
            });
        let det = SequentialDetector::fit(&errors, cfg).unwrap();
        let tau_sq = det
            .tau_squared()
            .expect("squared stream should catch variance shift");
        assert!(tau_sq >= 101, "tau_squared fired before shift: {}", tau_sq);
        assert!(tau_sq < 200, "tau_squared fired too late: {}", tau_sq);
    }

    /// The key invariant: online updates must be equivalent to a full fit.
    #[test]
    fn online_update_matches_full_fit_page_cusum() {
        let mut rng = rng(21);
        let mut errors: Vec<f64> = (0..200)
            .map(|_| normal_sample(&mut rng, 0.0, 1.0))
            .collect();
        errors.extend((0..50).map(|_| normal_sample(&mut rng, 1.5, 1.0)));

        let full_cfg = SequentialConfig::new(100)
            .detector(Detector::PageCusum)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Fixed(3.0));

        let full = SequentialDetector::fit(&errors, full_cfg.clone()).unwrap();

        // Online: fit on first 180 then update with remaining 70.
        let mut online = SequentialDetector::fit(&errors[..180], full_cfg).unwrap();
        online.update(&errors[180..]).unwrap();

        let a = full.cusum();
        let b = online.cusum();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(*x, *y, epsilon = 1e-10, max_relative = 1e-10);
            let _ = i;
        }
        assert_eq!(full.tau(), online.tau());
    }

    #[test]
    fn online_update_matches_full_fit_cusum1() {
        let mut rng = rng(33);
        let errors: Vec<f64> = (0..250)
            .map(|_| normal_sample(&mut rng, 0.0, 1.0))
            .collect();

        let cfg = SequentialConfig::new(100)
            .detector(Detector::Cusum1)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Fixed(3.0));

        let full = SequentialDetector::fit(&errors, cfg.clone()).unwrap();
        let mut online = SequentialDetector::fit(&errors[..175], cfg).unwrap();
        online.update(&errors[175..]).unwrap();

        assert_eq!(full.cusum().len(), online.cusum().len());
        for (x, y) in full.cusum().iter().zip(online.cusum()) {
            assert_relative_eq!(*x, *y, epsilon = 1e-10);
        }
    }

    #[test]
    fn fixed_critical_value_used_verbatim() {
        let errors: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let cfg = SequentialConfig::new(30)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Fixed(7.5));
        let det = SequentialDetector::fit(&errors, cfg).unwrap();
        assert_relative_eq!(det.raw.as_ref().unwrap().crit_value, 7.5, epsilon = 1e-12);
    }

    #[test]
    fn sigma2_override_respected() {
        let errors: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let cfg = SequentialConfig::new(30)
            .error_type(ForecastErrorType::Raw)
            .critical_value(CriticalValue::Fixed(1.0))
            .with_sigma2(4.0);
        let det = SequentialDetector::fit(&errors, cfg).unwrap();
        assert_relative_eq!(det.raw.as_ref().unwrap().sigma2, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn sample_variance_matches_r() {
        // R: var(c(1, 2, 3, 4, 5)) = 2.5
        assert_relative_eq!(
            sample_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            2.5,
            epsilon = 1e-12
        );
    }
}
