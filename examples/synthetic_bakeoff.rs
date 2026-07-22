//! Synthetic bake-off — 18 archetypes × 30 replicates × 5 models.
//!
//! Answers two questions:
//!  1. Does `laplace::recommended_for(...)` pick the right recipe for
//!     each archetype? (validation of the 2026-07-22 router)
//!  2. Where does our Laplace family beat / lose to classical
//!     (AutoETS, AutoTheta) on clean-signal, single-behaviour panels?
//!     (regression discovery — fev-27 is noisy; synthetic isolates one
//!     axis at a time)
//!
//! Each archetype isolates ONE axis (length, seasonality, trend,
//! variance, jump, distribution, count-ness, multi-seasonal) so the
//! winner tells us something about *why* it won.
//!
//! Run: `SAMPLE_PER=3 cargo run --release --features distributional --example synthetic_bakeoff`  (smoke)
//! Run: `cargo run --release --features distributional --example synthetic_bakeoff`  (full: 30 replicates)

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::laplace::{
    recipe_for, recommended_for, MultiScaleLaplace, RecipeKind,
};
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::DistributionalForecaster;
use anofox_forecast::models::Forecaster;
use anofox_forecast::models::LaplaceForecaster;

use chrono::{Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::time::Instant;

// ---------- Archetype spec ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Archetype {
    PureGaussianNoise,
    ShortGaussian,
    StationarySeasonalShort,
    StationarySeasonalLong,
    SeasonalLinearTrend,
    SeasonalDampedTrend,
    MultiSeasonalHourly,
    LinearTrendOnly,
    RandomWalk,
    MeanRevertingOu,
    LevelShiftMidway,
    VarianceRegimeChange,
    HeavyTailCauchy,
    StudentTDf3,
    PoissonIntermittent,
    PoissonSeasonalRetail,
    EverythingAtOnce,
    HeteroscedasticMultiSeasonal,
}

impl Archetype {
    const ALL: &'static [Archetype] = &[
        Archetype::PureGaussianNoise,
        Archetype::ShortGaussian,
        Archetype::StationarySeasonalShort,
        Archetype::StationarySeasonalLong,
        Archetype::SeasonalLinearTrend,
        Archetype::SeasonalDampedTrend,
        Archetype::MultiSeasonalHourly,
        Archetype::LinearTrendOnly,
        Archetype::RandomWalk,
        Archetype::MeanRevertingOu,
        Archetype::LevelShiftMidway,
        Archetype::VarianceRegimeChange,
        Archetype::HeavyTailCauchy,
        Archetype::StudentTDf3,
        Archetype::PoissonIntermittent,
        Archetype::PoissonSeasonalRetail,
        Archetype::EverythingAtOnce,
        Archetype::HeteroscedasticMultiSeasonal,
    ];

    fn name(self) -> &'static str {
        match self {
            Archetype::PureGaussianNoise => "pure_gaussian_noise",
            Archetype::ShortGaussian => "short_gaussian",
            Archetype::StationarySeasonalShort => "stationary_seasonal_short",
            Archetype::StationarySeasonalLong => "stationary_seasonal_long",
            Archetype::SeasonalLinearTrend => "seasonal_linear_trend",
            Archetype::SeasonalDampedTrend => "seasonal_damped_trend",
            Archetype::MultiSeasonalHourly => "multi_seasonal_hourly",
            Archetype::LinearTrendOnly => "linear_trend_only",
            Archetype::RandomWalk => "random_walk",
            Archetype::MeanRevertingOu => "mean_reverting_ou",
            Archetype::LevelShiftMidway => "level_shift_midway",
            Archetype::VarianceRegimeChange => "variance_regime_change",
            Archetype::HeavyTailCauchy => "heavy_tail_cauchy",
            Archetype::StudentTDf3 => "student_t_df3",
            Archetype::PoissonIntermittent => "poisson_intermittent",
            Archetype::PoissonSeasonalRetail => "poisson_seasonal_retail",
            Archetype::EverythingAtOnce => "everything_at_once",
            Archetype::HeteroscedasticMultiSeasonal => "heteroscedastic_multi_seasonal",
        }
    }

    /// Total series length. Test-window horizon is 20 % of this (capped
    /// at the archetype's canonical H below).
    fn length(self) -> usize {
        match self {
            Archetype::ShortGaussian => 50,
            Archetype::PureGaussianNoise => 200,
            Archetype::StationarySeasonalShort => 100,
            Archetype::StationarySeasonalLong => 500,
            Archetype::SeasonalLinearTrend | Archetype::SeasonalDampedTrend => 300,
            Archetype::MultiSeasonalHourly => 800,
            Archetype::LinearTrendOnly => 200,
            Archetype::RandomWalk | Archetype::MeanRevertingOu => 300,
            Archetype::LevelShiftMidway | Archetype::VarianceRegimeChange => 300,
            Archetype::HeavyTailCauchy | Archetype::StudentTDf3 => 300,
            Archetype::PoissonIntermittent => 200,
            Archetype::PoissonSeasonalRetail => 300,
            Archetype::EverythingAtOnce => 400,
            Archetype::HeteroscedasticMultiSeasonal => 800,
        }
    }

    /// Seasonal period. `None` means non-seasonal — models should NOT be
    /// told a period. `Some(0)` never happens.
    fn period(self) -> Option<usize> {
        match self {
            Archetype::StationarySeasonalShort
            | Archetype::StationarySeasonalLong
            | Archetype::SeasonalLinearTrend
            | Archetype::SeasonalDampedTrend
            | Archetype::EverythingAtOnce => Some(12),
            Archetype::MultiSeasonalHourly | Archetype::HeteroscedasticMultiSeasonal => Some(24),
            Archetype::PoissonSeasonalRetail => Some(7),
            _ => None,
        }
    }

    fn horizon(self) -> usize {
        // Match fev's per-frequency horizons where applicable.
        match self {
            Archetype::ShortGaussian => 6,
            Archetype::StationarySeasonalShort => 12,
            Archetype::StationarySeasonalLong => 18,
            Archetype::SeasonalLinearTrend | Archetype::SeasonalDampedTrend => 18,
            Archetype::MultiSeasonalHourly | Archetype::HeteroscedasticMultiSeasonal => 48,
            Archetype::EverythingAtOnce => 24,
            Archetype::PoissonSeasonalRetail => 14,
            _ => 20,
        }
    }

    /// What the router SHOULD pick, given the archetype's designed shape.
    /// A mismatch is a router bug worth investigating.
    fn expected_recipe(self) -> RecipeKind {
        match self {
            Archetype::ShortGaussian => RecipeKind::ShortHistory,
            Archetype::PoissonIntermittent | Archetype::PoissonSeasonalRetail => {
                RecipeKind::RetailCountAid
            }
            Archetype::HeavyTailCauchy | Archetype::StudentTDf3 => RecipeKind::HeavyTailedCrps,
            // Seasonal + long enough for period activation.
            Archetype::StationarySeasonalShort
            | Archetype::StationarySeasonalLong
            | Archetype::SeasonalLinearTrend
            | Archetype::SeasonalDampedTrend
            | Archetype::MultiSeasonalHourly
            | Archetype::HeteroscedasticMultiSeasonal
            | Archetype::EverythingAtOnce => RecipeKind::ContinuousMultiScale,
            // Non-seasonal continuous.
            _ => RecipeKind::ContinuousPlainSkaters,
        }
    }
}

// ---------- Series generation ----------

fn standard_normal(rng: &mut StdRng) -> f64 {
    // Box-Muller.
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn cauchy(rng: &mut StdRng, scale: f64) -> f64 {
    let u: f64 = rng.gen_range(1e-6..(1.0 - 1e-6));
    scale * (std::f64::consts::PI * (u - 0.5)).tan()
}

fn student_t(rng: &mut StdRng, df: f64) -> f64 {
    // Inverse-CDF via statrs.
    let u: f64 = rng.gen_range(1e-6..(1.0 - 1e-6));
    StudentsT::new(0.0, 1.0, df).unwrap().inverse_cdf(u)
}

fn poisson_sample(rng: &mut StdRng, lambda: f64) -> f64 {
    // Knuth. Only stable for small lambda (< ~30) but that's what we want.
    if lambda <= 0.0 {
        return 0.0;
    }
    let l = (-lambda).exp();
    let mut k: u64 = 0;
    let mut p: f64 = 1.0;
    loop {
        k += 1;
        let u: f64 = rng.gen();
        p *= u;
        if p <= l {
            return (k - 1) as f64;
        }
    }
}

fn generate(archetype: Archetype, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = archetype.length();
    let mut y = Vec::with_capacity(n);
    match archetype {
        Archetype::PureGaussianNoise | Archetype::ShortGaussian => {
            for _ in 0..n {
                y.push(50.0 + standard_normal(&mut rng));
            }
        }
        Archetype::StationarySeasonalShort | Archetype::StationarySeasonalLong => {
            let p = archetype.period().unwrap() as f64;
            for i in 0..n {
                let s = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                y.push(50.0 + s + standard_normal(&mut rng));
            }
        }
        Archetype::SeasonalLinearTrend => {
            let p = 12.0;
            for i in 0..n {
                let s = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                let t = 0.05 * i as f64;
                y.push(50.0 + s + t + standard_normal(&mut rng));
            }
        }
        Archetype::SeasonalDampedTrend => {
            let p = 12.0;
            for i in 0..n {
                let s = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                let t = 10.0 * (1.0 - (-0.01 * i as f64).exp());
                y.push(50.0 + s + t + standard_normal(&mut rng));
            }
        }
        Archetype::MultiSeasonalHourly => {
            for i in 0..n {
                let daily = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin();
                let weekly = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin();
                y.push(50.0 + daily + weekly + standard_normal(&mut rng));
            }
        }
        Archetype::LinearTrendOnly => {
            for i in 0..n {
                y.push(50.0 + 0.1 * i as f64 + standard_normal(&mut rng));
            }
        }
        Archetype::RandomWalk => {
            let mut x = 50.0;
            for _ in 0..n {
                x += standard_normal(&mut rng);
                y.push(x);
            }
        }
        Archetype::MeanRevertingOu => {
            let mu = 50.0;
            let theta = 0.1;
            let sigma = 1.0;
            let dt: f64 = 1.0;
            let mut x = mu;
            for _ in 0..n {
                x += theta * (mu - x) * dt + sigma * dt.sqrt() * standard_normal(&mut rng);
                y.push(x);
            }
        }
        Archetype::LevelShiftMidway => {
            for i in 0..n {
                let base = if i < n / 2 { 50.0 } else { 55.0 };
                y.push(base + 0.5 * standard_normal(&mut rng));
            }
        }
        Archetype::VarianceRegimeChange => {
            for i in 0..n {
                let sigma = if i < n / 2 { 0.5 } else { 3.0 };
                y.push(50.0 + sigma * standard_normal(&mut rng));
            }
        }
        Archetype::HeavyTailCauchy => {
            for _ in 0..n {
                y.push(50.0 + cauchy(&mut rng, 1.0));
            }
        }
        Archetype::StudentTDf3 => {
            for _ in 0..n {
                y.push(50.0 + student_t(&mut rng, 3.0));
            }
        }
        Archetype::PoissonIntermittent => {
            for _ in 0..n {
                y.push(poisson_sample(&mut rng, 0.5));
            }
        }
        Archetype::PoissonSeasonalRetail => {
            // Week-of-day rate: high Fri-Sun, low Tue-Thu.
            let rates = [1.0, 0.8, 0.5, 0.5, 0.8, 3.0, 4.0];
            for i in 0..n {
                let lambda = rates[i % 7];
                y.push(poisson_sample(&mut rng, lambda));
            }
        }
        Archetype::EverythingAtOnce => {
            let p = 12.0;
            for i in 0..n {
                let s = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                let t = 0.05 * i as f64;
                let jump = if i >= n / 2 { 3.0 } else { 0.0 };
                let sigma = 1.0 + 0.005 * i as f64;
                y.push(50.0 + s + t + jump + sigma * standard_normal(&mut rng));
            }
        }
        Archetype::HeteroscedasticMultiSeasonal => {
            for i in 0..n {
                let daily = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin();
                let weekly = 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin();
                let trend = 0.02 * i as f64;
                let sigma = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin().abs();
                y.push(50.0 + daily + weekly + trend + sigma * standard_normal(&mut rng));
            }
        }
    }
    y
}

// ---------- Metrics ----------

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

fn mase_scale(train: &[f64], period: usize) -> f64 {
    let p = if train.len() > period { period } else { 1 };
    if train.len() <= p {
        return 1.0;
    }
    let n = train.len() - p;
    let sum: f64 = (p..train.len())
        .map(|i| (train[i] - train[i - p]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

const WQL_Q: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

fn wql(matrix: &[Vec<f64>], truth: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (qi, &q) in WQL_Q.iter().enumerate() {
        for h in 0..truth.len() {
            let d = truth[h] - matrix[qi][h];
            let loss = 2.0 * ((q * d).max((q - 1.0) * d));
            num += loss;
            den += truth[h].abs();
        }
    }
    if den < 1e-9 {
        f64::NAN
    } else {
        num / den
    }
}

fn gaussian_q(mu: f64, sigma: f64, q: f64) -> f64 {
    mu + sigma * Normal::new(0.0, 1.0).unwrap().inverse_cdf(q)
}

fn geomean(xs: &[f64]) -> f64 {
    let xs: Vec<f64> = xs
        .iter()
        .filter(|x| x.is_finite() && **x > 0.0)
        .copied()
        .collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    let ls: f64 = xs.iter().map(|x| x.ln()).sum();
    (ls / xs.len() as f64).exp()
}

// ---------- Model panel ----------

const MODEL_NAMES: [&str; 5] = [
    "AutoETS",
    "AutoTheta",
    "Lap.auto()",
    "recommended_for",
    "MS+3SH manual",
];
const N_MODELS: usize = 5;

fn run_one_series(
    values: Vec<f64>,
    horizon: usize,
    period: Option<usize>,
) -> [Option<(f64, f64, f64)>; N_MODELS] {
    // Returns [(mase, wql, fit_secs); N_MODELS], Option per model.
    let split = values.len() - horizon;
    if split < 20 {
        return [None; N_MODELS];
    }
    let train_v = values[..split].to_vec();
    let test_v = &values[split..];
    let scale = mase_scale(&train_v, period.unwrap_or(1));
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let stamps: Vec<_> = (0..train_v.len())
        .map(|i| base + Duration::seconds(3600 * i as i64))
        .collect();
    let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
        Ok(t) => t,
        Err(_) => return [None; N_MODELS],
    };

    let mut out: [Option<(f64, f64, f64)>; N_MODELS] = [None; N_MODELS];

    // M0: AutoETS
    {
        let t0 = Instant::now();
        let mut m = match period {
            Some(p) if p >= 2 => AutoETS::with_period(p),
            _ => AutoETS::new(),
        };
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(horizon) {
                let p = fc.primary();
                if p.len() == test_v.len() {
                    let ms = mae(p, test_v) / scale;
                    // Gaussian fallback for WQL.
                    let sigma = mase_scale(&train_v, 1).max(1e-9);
                    let matrix: Vec<Vec<f64>> = WQL_Q
                        .iter()
                        .map(|&q| p.iter().map(|&mu| gaussian_q(mu, sigma, q)).collect())
                        .collect();
                    let w = wql(&matrix, test_v);
                    out[0] = Some((ms, w, t0.elapsed().as_secs_f64()));
                }
            }
        }
    }

    // M1: AutoTheta
    {
        let t0 = Instant::now();
        let mut m = match period {
            Some(p) if p >= 2 => AutoTheta::seasonal(p),
            _ => AutoTheta::new(),
        };
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(horizon) {
                let p = fc.primary();
                if p.len() == test_v.len() {
                    let ms = mae(p, test_v) / scale;
                    let sigma = mase_scale(&train_v, 1).max(1e-9);
                    let matrix: Vec<Vec<f64>> = WQL_Q
                        .iter()
                        .map(|&q| p.iter().map(|&mu| gaussian_q(mu, sigma, q)).collect())
                        .collect();
                    let w = wql(&matrix, test_v);
                    out[1] = Some((ms, w, t0.elapsed().as_secs_f64()));
                }
            }
        }
    }

    // M2: Laplace.auto()
    {
        let t0 = Instant::now();
        let mut m = LaplaceForecaster::new().auto();
        if let Some(p) = period {
            if p >= 2 {
                m = m.auto_with_seasonal_period(p);
            }
        }
        if <LaplaceForecaster as Forecaster>::fit(&mut m, &train_ts).is_ok() {
            if let Ok(mix) =
                <LaplaceForecaster as DistributionalForecaster>::forecast_dist(&m, horizon)
            {
                if mix.len() == test_v.len() {
                    let p_means: Vec<f64> = mix.iter().map(|g| g.mean()).collect();
                    let ms = mae(&p_means, test_v) / scale;
                    let matrix: Vec<Vec<f64>> = WQL_Q
                        .iter()
                        .map(|&q| mix.iter().map(|g| g.quantile(q)).collect())
                        .collect();
                    let w = wql(&matrix, test_v);
                    out[2] = Some((ms, w, t0.elapsed().as_secs_f64()));
                }
            }
        }
    }

    // M3: recommended_for (router)
    {
        let t0 = Instant::now();
        let mut m = recommended_for(&train_ts, horizon, period);
        if <dyn DistributionalForecaster as Forecaster>::fit(&mut *m, &train_ts).is_ok() {
            if let Ok(mix) = m.forecast_dist(horizon) {
                if mix.len() == test_v.len() {
                    let p_means: Vec<f64> = mix.iter().map(|g| g.mean()).collect();
                    let ms = mae(&p_means, test_v) / scale;
                    let matrix: Vec<Vec<f64>> = WQL_Q
                        .iter()
                        .map(|&q| mix.iter().map(|g| g.quantile(q)).collect())
                        .collect();
                    let w = wql(&matrix, test_v);
                    out[3] = Some((ms, w, t0.elapsed().as_secs_f64()));
                }
            }
        }
    }

    // M4: MS+3SH manual — the SOTA opt-in.
    {
        let t0 = Instant::now();
        let mut m = MultiScaleLaplace::skaters(horizon)
            .with_scoring_horizon()
            .with_scoring_window(10)
            .with_learning_rate(0.20)
            .with_seasonal_holt(0.3, 0.1)
            .with_seasonal_holt(0.5, 0.2)
            .with_seasonal_holt(0.7, 0.3);
        if let Some(p) = period {
            if p >= 2 {
                m = m.with_period(p);
            }
        }
        if m.fit(&train_ts).is_ok() {
            if let Ok(mix) = m.forecast_dist(horizon) {
                if mix.len() == test_v.len() {
                    let p_means: Vec<f64> = mix.iter().map(|g| g.mean()).collect();
                    let ms = mae(&p_means, test_v) / scale;
                    let matrix: Vec<Vec<f64>> = WQL_Q
                        .iter()
                        .map(|&q| mix.iter().map(|g| g.quantile(q)).collect())
                        .collect();
                    let w = wql(&matrix, test_v);
                    out[4] = Some((ms, w, t0.elapsed().as_secs_f64()));
                }
            }
        }
    }

    out
}

// ---------- Main ----------

fn main() {
    let n_replicates: usize = std::env::var("SAMPLE_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    eprintln!(
        "synthetic_bakeoff — {} replicates × {} archetypes × {} models",
        n_replicates,
        Archetype::ALL.len(),
        N_MODELS
    );

    let start = Instant::now();
    // Per-archetype accumulators.
    let mut mase_by_arch: Vec<Vec<Vec<f64>>> = Archetype::ALL
        .iter()
        .map(|_| vec![Vec::new(); N_MODELS])
        .collect();
    let mut wql_by_arch: Vec<Vec<Vec<f64>>> = Archetype::ALL
        .iter()
        .map(|_| vec![Vec::new(); N_MODELS])
        .collect();
    let mut fit_by_arch: Vec<Vec<Vec<f64>>> = Archetype::ALL
        .iter()
        .map(|_| vec![Vec::new(); N_MODELS])
        .collect();
    let mut router_picks: Vec<Vec<RecipeKind>> =
        Archetype::ALL.iter().map(|_| Vec::new()).collect();

    for (ai, &arch) in Archetype::ALL.iter().enumerate() {
        let period = arch.period();
        let horizon = arch.horizon();
        eprintln!(
            "  [{}] N={}, H={}, period={:?}, expected recipe={}",
            arch.name(),
            arch.length(),
            horizon,
            period,
            arch.expected_recipe().name()
        );
        for rep in 0..n_replicates {
            let seed = (ai as u64) * 1000 + rep as u64 + 1;
            let values = generate(arch, seed);

            // Track what the router picks for this replicate.
            let train_len = values.len() - horizon;
            let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
            let stamps: Vec<_> = (0..train_len)
                .map(|i| base + Duration::seconds(3600 * i as i64))
                .collect();
            if let Ok(train_ts) = TimeSeries::univariate(stamps, values[..train_len].to_vec()) {
                router_picks[ai].push(recipe_for(&train_ts, period));
            }

            let per_model = run_one_series(values, horizon, period);
            for (mi, res) in per_model.iter().enumerate() {
                if let Some((ms, w, ft)) = res {
                    if ms.is_finite() && *ms > 0.0 {
                        mase_by_arch[ai][mi].push(*ms);
                    }
                    if w.is_finite() && *w > 0.0 {
                        wql_by_arch[ai][mi].push(*w);
                    }
                    fit_by_arch[ai][mi].push(*ft);
                }
            }
        }
    }

    // ---------- Reports ----------

    println!(
        "\n=== MASE per archetype (geomean of {} replicates) ===",
        n_replicates
    );
    print!("{:<34}", "archetype");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    println!();
    for (ai, arch) in Archetype::ALL.iter().enumerate() {
        print!("{:<34}", arch.name());
        for mi in 0..N_MODELS {
            let g = geomean(&mase_by_arch[ai][mi]);
            print!("{:>18.4}", g);
        }
        println!();
    }
    // Overall geomean-of-geomeans across all archetypes.
    print!("{:<34}", "OVERALL geomean");
    for mi in 0..N_MODELS {
        let all: Vec<f64> = mase_by_arch.iter().map(|arch| geomean(&arch[mi])).collect();
        let g = geomean(&all);
        print!("{:>18.4}", g);
    }
    println!();

    println!(
        "\n=== WQL per archetype (mean of {} replicates) ===",
        n_replicates
    );
    print!("{:<34}", "archetype");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    println!();
    for (ai, arch) in Archetype::ALL.iter().enumerate() {
        print!("{:<34}", arch.name());
        for mi in 0..N_MODELS {
            let g = geomean(&wql_by_arch[ai][mi]);
            print!("{:>18.4}", g);
        }
        println!();
    }

    println!("\n=== Router validation (2026-07-22) ===");
    println!(
        "{:<34}{:<28}{:<28}{}",
        "archetype", "expected recipe", "picked (mode)", "match?"
    );
    for (ai, arch) in Archetype::ALL.iter().enumerate() {
        // Mode of picks.
        let picks = &router_picks[ai];
        let mut counts = std::collections::HashMap::new();
        for &p in picks {
            *counts.entry(p).or_insert(0usize) += 1;
        }
        let mode = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| *k)
            .unwrap_or(RecipeKind::ContinuousPlainSkaters);
        let expected = arch.expected_recipe();
        let ok = if mode == expected { "OK" } else { "MISMATCH" };
        println!(
            "{:<34}{:<28}{:<28}{}",
            arch.name(),
            expected.name(),
            mode.name(),
            ok
        );
    }

    println!("\n=== Wins per archetype (MASE) ===");
    print!("{:<34}", "");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    println!();
    let mut wins = vec![0usize; N_MODELS];
    for ai in 0..Archetype::ALL.len() {
        let geos: Vec<f64> = (0..N_MODELS)
            .map(|mi| geomean(&mase_by_arch[ai][mi]))
            .collect();
        let (best_i, _) = geos
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_finite())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &f64::NAN));
        wins[best_i] += 1;
    }
    print!("{:<34}", "wins on MASE");
    for w in &wins {
        print!("{:>18}", w);
    }
    println!();

    println!("\n=== Fit time per archetype (mean seconds) ===");
    print!("{:<34}", "archetype");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    println!();
    for (ai, arch) in Archetype::ALL.iter().enumerate() {
        print!("{:<34}", arch.name());
        for mi in 0..N_MODELS {
            let xs = &fit_by_arch[ai][mi];
            let mean = if xs.is_empty() {
                f64::NAN
            } else {
                xs.iter().sum::<f64>() / xs.len() as f64
            };
            print!("{:>18.4}", mean);
        }
        println!();
    }

    println!("\n[total {:.1}s]", start.elapsed().as_secs_f64());
}
