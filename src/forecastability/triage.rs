//! Forecastability triage: single-call and batch series classification.
//!
//! Wraps [`ForecastabilityFingerprint`] into a decision pipeline that
//! classifies each series into a [`SeriesPattern`] (A–E) and recommends
//! a [`ModelFamily`]. This is the entry point for pre-modeling routing in
//! production orchestration systems.
//!
//! Mirrors `run_triage` / `run_batch_triage` from the Python
//! `dependence-forecastability` package.

use super::fingerprint::ForecastabilityFingerprint;
use super::scorers::{score, Scorer};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Series archetype based on the forecastability fingerprint.
///
/// Matches patterns A–E from the Python `dependence-forecastability`
/// walkthrough notebooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeriesPattern {
    /// **A — White noise**: no exploitable signal. `information_mass ≈ 0`,
    /// `signal_to_noise < 1.5`.
    WhiteNoise,
    /// **B — Linear / AR-like**: strong signal concentrated at short lags,
    /// captured by GCMI. `nonlinear_share < 0.3`, high `directness_ratio`.
    Linear,
    /// **C — Seasonal / periodic**: signal spread across lags at multiples
    /// of a period. `information_structure > 0.6`, moderate horizon.
    Seasonal,
    /// **D — Nonlinear deterministic**: significant signal that surrogates
    /// cannot reproduce. `nonlinear_share > 0.5`, `signal_to_noise > 2`.
    Nonlinear,
    /// **E — Complex / mixed**: combination of linear and nonlinear
    /// components, or long-range dependence.
    Complex,
}

impl std::fmt::Display for SeriesPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WhiteNoise => write!(f, "A: White noise"),
            Self::Linear => write!(f, "B: Linear / AR-like"),
            Self::Seasonal => write!(f, "C: Seasonal / periodic"),
            Self::Nonlinear => write!(f, "D: Nonlinear deterministic"),
            Self::Complex => write!(f, "E: Complex / mixed"),
        }
    }
}

/// Recommended model family based on the series pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Series has no signal — use Naive or skip.
    Skip,
    /// Linear signal → ARIMA, ETS, Theta, linear regression.
    LinearStatistical,
    /// Seasonal structure → SeasonalARIMA, ETS with seasonality, Fourier
    /// regression, MSTL.
    SeasonalStatistical,
    /// Nonlinear signal → MFLES, RegressionForecaster with rolling features,
    /// tree-based models.
    NonlinearML,
    /// Complex signal → ensemble of linear + nonlinear, or AutoForecast
    /// with full candidate pool.
    Ensemble,
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Skip => write!(f, "Skip (Naive)"),
            Self::LinearStatistical => write!(f, "ARIMA / ETS / Theta"),
            Self::SeasonalStatistical => write!(f, "SeasonalARIMA / ETS(seasonal) / MSTL"),
            Self::NonlinearML => write!(f, "MFLES / RegressionForecaster / tree-based"),
            Self::Ensemble => write!(f, "AutoForecast / Ensemble"),
        }
    }
}

// ---------------------------------------------------------------------------
// Triage result
// ---------------------------------------------------------------------------

/// Result of forecastability triage for a single series.
#[derive(Debug, Clone)]
pub struct TriageResult {
    /// Detected series pattern (A–E).
    pub pattern: SeriesPattern,
    /// Recommended model family.
    pub model_family: ModelFamily,
    /// The underlying fingerprint (full detail).
    pub fingerprint: ForecastabilityFingerprint,
    /// Permutation entropy (normalized, 0 = regular, 1 = random).
    pub permutation_entropy: f64,
    /// Spectral predictability (1 − spectral entropy).
    pub spectral_predictability: f64,
}

/// Result of batch triage across multiple series.
#[derive(Debug, Clone)]
pub struct BatchTriageResult {
    /// Per-series triage results.
    pub results: Vec<TriageResult>,
    /// Count of series per pattern.
    pub pattern_counts: [(SeriesPattern, usize); 5],
    /// Count of series per model family.
    pub family_counts: [(ModelFamily, usize); 5],
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the triage pipeline.
#[derive(Debug, Clone)]
pub struct TriageConfig {
    /// Maximum lag to probe. Default: 20.
    pub max_lag: usize,
    /// Number of phase surrogates for significance testing. Default: 50.
    pub n_surrogates: usize,
    /// Significance level. Default: 0.05.
    pub alpha: f64,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for TriageConfig {
    fn default() -> Self {
        Self {
            max_lag: 20,
            n_surrogates: 50,
            alpha: 0.05,
            seed: None,
        }
    }
}

impl TriageConfig {
    pub fn max_lag(mut self, v: usize) -> Self {
        self.max_lag = v;
        self
    }
    pub fn n_surrogates(mut self, v: usize) -> Self {
        self.n_surrogates = v;
        self
    }
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = Some(v);
        self
    }
}

// ---------------------------------------------------------------------------
// Classification logic
// ---------------------------------------------------------------------------

/// Classify the fingerprint into a series pattern.
fn classify_pattern(fp: &ForecastabilityFingerprint, pe: f64) -> SeriesPattern {
    // A — White noise: no significant lags, or very low SNR
    if fp.informative_horizons.is_empty() || fp.signal_to_noise < 1.5 {
        // Double-check: if permutation entropy is very high (> 0.95),
        // the series is likely random even if a stray lag passed.
        if pe > 0.9 || fp.information_mass < 0.01 {
            return SeriesPattern::WhiteNoise;
        }
    }

    // D — Nonlinear: high nonlinear share, good SNR. Check BEFORE seasonal
    // because chaotic systems (e.g. logistic map) can have many significant
    // lags with high information_structure — but the dominant signal is
    // nonlinear, not seasonal.
    if fp.nonlinear_share > 0.5 && fp.signal_to_noise > 2.0 {
        return SeriesPattern::Nonlinear;
    }

    // B — Linear: low nonlinear share, high directness
    if fp.nonlinear_share < 0.3 && fp.directness_ratio > 0.3 {
        return SeriesPattern::Linear;
    }

    // C — Seasonal: signal spread evenly across lags, moderate nonlinear share
    if fp.information_structure > 0.6
        && fp.informative_horizons.len() >= 3
        && fp.nonlinear_share < 0.5
    {
        return SeriesPattern::Seasonal;
    }

    // E — Complex: everything else
    SeriesPattern::Complex
}

/// Map pattern to recommended model family.
fn recommend_family(pattern: SeriesPattern) -> ModelFamily {
    match pattern {
        SeriesPattern::WhiteNoise => ModelFamily::Skip,
        SeriesPattern::Linear => ModelFamily::LinearStatistical,
        SeriesPattern::Seasonal => ModelFamily::SeasonalStatistical,
        SeriesPattern::Nonlinear => ModelFamily::NonlinearML,
        SeriesPattern::Complex => ModelFamily::Ensemble,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run forecastability triage on a single series.
///
/// Computes the fingerprint, classifies the pattern, and recommends a
/// model family — all in one call.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_forecast::forecastability::triage::{run_triage, TriageConfig};
///
/// let result = run_triage(&values, &TriageConfig::default());
/// println!("Pattern: {}", result.pattern);
/// println!("Recommendation: {}", result.model_family);
/// println!("Informative lags: {:?}", result.fingerprint.informative_horizons);
/// ```
pub fn run_triage(series: &[f64], config: &TriageConfig) -> TriageResult {
    let fp = ForecastabilityFingerprint::compute(
        series,
        config.max_lag,
        config.n_surrogates,
        config.alpha,
        config.seed,
    );
    let pe = score(series, Scorer::PermutationEntropy);
    let sp = score(series, Scorer::SpectralPredictability);
    let pattern = classify_pattern(&fp, pe);
    let model_family = recommend_family(pattern);

    TriageResult {
        pattern,
        model_family,
        fingerprint: fp,
        permutation_entropy: pe,
        spectral_predictability: sp,
    }
}

/// Run forecastability triage on a batch of series.
///
/// With the `parallel` feature enabled, series are processed in parallel
/// via rayon.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_forecast::forecastability::triage::{run_batch_triage, TriageConfig};
///
/// let all_series: Vec<Vec<f64>> = load_data();
/// let batch = run_batch_triage(&all_series, &TriageConfig::default());
///
/// for (pattern, count) in &batch.pattern_counts {
///     println!("{}: {} series", pattern, count);
/// }
/// ```
pub fn run_batch_triage(all_series: &[Vec<f64>], config: &TriageConfig) -> BatchTriageResult {
    #[cfg(feature = "parallel")]
    let results: Vec<TriageResult> = all_series
        .par_iter()
        .map(|s| run_triage(s, config))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<TriageResult> = all_series.iter().map(|s| run_triage(s, config)).collect();

    let mut pattern_counts = [
        (SeriesPattern::WhiteNoise, 0),
        (SeriesPattern::Linear, 0),
        (SeriesPattern::Seasonal, 0),
        (SeriesPattern::Nonlinear, 0),
        (SeriesPattern::Complex, 0),
    ];
    let mut family_counts = [
        (ModelFamily::Skip, 0),
        (ModelFamily::LinearStatistical, 0),
        (ModelFamily::SeasonalStatistical, 0),
        (ModelFamily::NonlinearML, 0),
        (ModelFamily::Ensemble, 0),
    ];

    for r in &results {
        for pc in &mut pattern_counts {
            if pc.0 == r.pattern {
                pc.1 += 1;
            }
        }
        for fc in &mut family_counts {
            if fc.0 == r.model_family {
                fc.1 += 1;
            }
        }
    }

    BatchTriageResult {
        results,
        pattern_counts,
        family_counts,
    }
}

/// Screen exogenous candidates: compute transfer entropy from each
/// candidate to the target and rank by TE at lag 1.
///
/// Returns `(candidate_index, te_at_lag1)` sorted descending by TE.
pub fn screen_exogenous(
    target: &[f64],
    candidates: &[Vec<f64>],
    max_lag: usize,
) -> Vec<(usize, f64)> {
    let mut scores: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, cand)| {
            let te = super::transfer_entropy::transfer_entropy_curve(cand, target, max_lag);
            let te_lag1 = te.first().copied().unwrap_or(0.0);
            (i, te_lag1)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = vec![0.0; n];
        for t in 1..n {
            let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.gen();
            x[t] =
                phi * x[t - 1] + (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        x
    }

    fn make_white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }

    fn make_logistic(n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        x[0] = 0.1;
        for t in 1..n {
            x[t] = 3.9 * x[t - 1] * (1.0 - x[t - 1]);
        }
        x
    }

    fn make_seasonal(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                    + 0.3 * (4.0 * std::f64::consts::PI * i as f64 / period as f64).cos()
                    + ((i * 7 + 3) % 11) as f64 * 0.05
            })
            .collect()
    }

    #[test]
    fn triage_white_noise_classifies_a() {
        let series = make_white_noise(500, 42);
        let result = run_triage(&series, &TriageConfig::default().seed(1));
        assert_eq!(
            result.pattern,
            SeriesPattern::WhiteNoise,
            "white noise should be pattern A, got {}",
            result.pattern
        );
        assert_eq!(result.model_family, ModelFamily::Skip);
    }

    #[test]
    fn triage_logistic_map_classifies_nonlinear() {
        let series = make_logistic(1000);
        let result = run_triage(&series, &TriageConfig::default().seed(1));
        assert!(
            result.pattern == SeriesPattern::Nonlinear || result.pattern == SeriesPattern::Complex,
            "logistic map should be pattern D or E, got {}",
            result.pattern
        );
        assert!(
            result.model_family == ModelFamily::NonlinearML
                || result.model_family == ModelFamily::Ensemble,
        );
    }

    #[test]
    fn batch_triage_counts_match() {
        let series = vec![
            make_white_noise(300, 1),
            make_white_noise(300, 2),
            make_logistic(500),
        ];
        let config = TriageConfig::default()
            .max_lag(10)
            .n_surrogates(30)
            .seed(42);
        let batch = run_batch_triage(&series, &config);
        assert_eq!(batch.results.len(), 3);
        let total: usize = batch.pattern_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn screen_exogenous_ranks_driver_first() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 300;
        let driver: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();
        let mut target = vec![0.0; n];
        for t in 1..n {
            target[t] = 0.7 * driver[t - 1] + (rng.gen::<f64>() - 0.5) * 0.5;
        }
        let noise: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();

        let scores = screen_exogenous(&target, &[driver, noise], 3);
        // Driver should rank first (higher TE).
        assert_eq!(scores[0].0, 0, "driver should rank first");
        assert!(
            scores[0].1 > scores[1].1,
            "driver TE ({:.4}) should exceed noise TE ({:.4})",
            scores[0].1,
            scores[1].1
        );
    }
}
