//! Synthetic bake-off — 41 archetypes × 30 replicates × 6 models.
//!
//! Answers two questions:
//!  1. Does `laplace::recommended_for(...)` pick the right recipe for
//!     each archetype? (validation of the 2026-07-22 router)
//!  2. Where does our Laplace family beat / lose to classical
//!     (AutoETS, AutoTheta)? (which panel shapes should users route
//!     to Laplace vs AutoETS — the "when to use Laplace" cut)
//!
//! Each archetype isolates ONE axis (length, seasonality, trend,
//! variance, jump, distribution, count-ness, multi-seasonal, regime
//! shift, contamination, GARCH clustering, fading structure,
//! tick-grid discreteness). The output segments wins by data-shape
//! `Category` so the "when to use Laplace" story is legible.
//!
//! History: started 2026-07-22 with 18 archetypes; extended
//! 2026-07-23 with 11 more designed to isolate Laplace's actual
//! advantages; extended 2026-07-24 with 12 more covering
//! under-tested axes (skewed marginals, overdispersed counts,
//! multiplicative seasonality, AR(1), piecewise / exponential /
//! S-curve trends, realistic retail + web-traffic + edge cases).
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
use anofox_forecast::models::SmartForecaster;

use chrono::{Duration, TimeZone, Utc};
use rand::distributions::Distribution as _;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ContinuousCDF, Gamma, NegativeBinomial, Normal, StudentsT};
use std::time::Instant;

// ---------- Archetype spec ----------

/// Where the archetype's DGP sits relative to model families' assumptions.
/// Wins-by-category is the useful cut: "AutoETS wins overall" is boring
/// because it lumps its home-turf archetypes with its blind spots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    /// Textbook trend + seasonal + Gaussian noise — AutoETS's home turf.
    AutoETSFavoring,
    /// Non-parametric shapes, regime shifts, heavy tails, count / zero-inflated,
    /// GARCH clustering, tick-grid discreteness — where the streaming
    /// Laplace pool is expected to hold up better than a parametric baseline.
    LaplaceFavoring,
    /// Genuinely close: pure noise, short history, moderate variance shifts.
    /// Neither model has a structural advantage.
    Neutral,
}

impl Category {
    fn name(self) -> &'static str {
        match self {
            Category::AutoETSFavoring => "AutoETS-favoring",
            Category::LaplaceFavoring => "Laplace-favoring",
            Category::Neutral => "Neutral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Archetype {
    // === Original 18 (2026-07-22) ===
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
    // === 2026-07-23 additions — designed to isolate Laplace's actual
    // advantages so the "when to use Laplace" story sharpens. ===
    /// Level-only for first half, linear trend for second half.
    /// AutoETS grid-searches ONE decomposition; Laplace's streaming
    /// softmax re-weights leaves at the regime boundary.
    RegimeShiftFlatToTrend,
    /// Sine + Gaussian noise, but 5 % of observations are Cauchy
    /// contamination (outliers). AutoETS Kalman gets pulled by outliers;
    /// Laplace's Student-t / Cauchy leaves resist.
    ContaminatedSeasonal,
    /// Linear trend with Student-t(df=3) innovations. AutoETS assumes
    /// Gaussian noise → miscalibrates the tails; router should pick
    /// `HeavyTailedCrps`.
    StudentTTrended,
    /// 80 % zeros, 20 % Poisson(λ=5) bursts. Retail-adjacent but far
    /// more intermittent than `poisson_intermittent`. Should trigger
    /// `RetailCountAid`.
    IntermittentBursty,
    /// Gaussian noise with σ = 1 + 3·|sin(2π·t/50)| — variance breathes
    /// with period 50, non-monotonic. AutoETS's ETS variants assume
    /// stationary error scale.
    EvolvingVarianceSmooth,
    /// Sine × (1 − t/N) — seasonality amplitude fades to zero. AutoETS's
    /// fixed seasonal coefficients keep predicting the initial-cycle
    /// amplitude; Laplace's streaming softmax deprecates the seasonal
    /// leaf as it starts losing to non-seasonal ones.
    FadingSeasonality,
    /// Trend + two level shifts + Student-t innovations. Realistic
    /// combined stress: structural break × heavy tail × trend, at once.
    TrendJumpsHeavyTails,
    /// 70 % forced zeros + Poisson(weekly_rate). Extreme
    /// zero-inflation on top of a weekly cycle. Should hit
    /// `RetailCountAid` and let `.auto_aid()`'s ZIP / ZINB leaves earn
    /// their keep.
    ZeroInflatedSeasonal,
    /// GARCH(1,1) volatility clustering. AutoETS assumes constant or
    /// trending σ; `.skaters()`'s GARCH cascade is designed for this.
    GarchVolatilityClustering,
    /// Markov switching between two Gaussians (μ = 45, μ = 55). Bimodal
    /// residual distribution; AutoETS assumes unimodal Gaussian, Laplace
    /// emits a `GaussianMixture` natively.
    BimodalRegimeSwitch,
    /// Random walk rounded to a 0.25 tick grid — discrete-valued
    /// continuous process. Sticky lattice inside `.skaters()` is
    /// designed for exactly this pattern.
    TickGridWalk,
    // === 2026-07-24 additions — cover under-tested distributional /
    // structural / real-world axes. ===
    /// Gamma(α=2, β=1) noise — positive-only, right-skewed continuous.
    /// Realistic for monetary flows, energy demand baselines. AutoETS
    /// assumes Gaussian residuals → miscalibrates quantiles.
    GammaPositiveSkewed,
    /// Lognormal (μ=0, σ=0.5) multiplicative noise on a slow-growing
    /// base. Monetary / demand pattern. Right-tail is heavy.
    LognormalMultiplicative,
    /// Negative-Binomial(r=3, p=0.4) counts — overdispersed relative
    /// to Poisson. Retail / demand where Poisson variance
    /// under-estimates real dispersion. `.auto_aid()`'s NegBin leaf
    /// is designed for exactly this.
    NegbinOverdispersedCounts,
    /// Multiplicative seasonality — seasonal amplitude scales with
    /// level. Common for retail / demand series where absolute
    /// swings grow with volume.
    MultiplicativeSeasonality,
    /// Pure AR(1) with φ=0.9 — highly persistent stationary process,
    /// no trend, no seasonality. `Ar1Leaf` in the pool is designed
    /// for this shape.
    Ar1Persistent,
    /// Piecewise-linear trend — slope changes at t=n/3 and t=2n/3.
    /// AutoETS's damped-trend fits one slope; regime shifts break it.
    PiecewiseLinearTrend,
    /// Exponential growth y = a·exp(b·t) + noise. Compounding,
    /// non-linear trend. AutoETS's multiplicative-error-multiplicative-trend
    /// (MMN) variants target this; a good stress test for AutoETS's
    /// grid vs Laplace's leaf pool.
    ExponentialGrowth,
    /// Logistic S-curve growth — trend saturates. Product-adoption
    /// pattern. AutoETS's damped-trend variants are the classical
    /// candidate; Laplace has to reach for `Ar2Leaf` / `HoltLeaf`
    /// with a damping factor.
    SCurveLogisticGrowth,
    /// Retail-with-promotions — smooth weekly baseline with random
    /// 3-8× spikes on ~5 % of days (promotion days). Realistic SKU
    /// pattern with structural anomalies on top of clean seasonality.
    RetailWithPromotions,
    /// Web-traffic style — daily (24) + weekly (168) seasonality +
    /// occasional 10× release-day spikes on ~2 % of hours.
    WeeklyPlusDailyPlusSpike,
    /// Near-constant series (σ = 0.01). Numerical edge case — tests
    /// that models don't explode when variance is effectively zero.
    NearConstantLowVar,
    /// 99 % zeros with 1 % Poisson(10) spikes. Extreme intermittency
    /// past `intermittent_bursty`. Route → RetailCountAid.
    AllZerosRareSpikes,
}

impl Archetype {
    const ALL: &'static [Archetype] = &[
        // Original 18.
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
        // 2026-07-23 additions.
        Archetype::RegimeShiftFlatToTrend,
        Archetype::ContaminatedSeasonal,
        Archetype::StudentTTrended,
        Archetype::IntermittentBursty,
        Archetype::EvolvingVarianceSmooth,
        Archetype::FadingSeasonality,
        Archetype::TrendJumpsHeavyTails,
        Archetype::ZeroInflatedSeasonal,
        Archetype::GarchVolatilityClustering,
        Archetype::BimodalRegimeSwitch,
        Archetype::TickGridWalk,
        // 2026-07-24 additions.
        Archetype::GammaPositiveSkewed,
        Archetype::LognormalMultiplicative,
        Archetype::NegbinOverdispersedCounts,
        Archetype::MultiplicativeSeasonality,
        Archetype::Ar1Persistent,
        Archetype::PiecewiseLinearTrend,
        Archetype::ExponentialGrowth,
        Archetype::SCurveLogisticGrowth,
        Archetype::RetailWithPromotions,
        Archetype::WeeklyPlusDailyPlusSpike,
        Archetype::NearConstantLowVar,
        Archetype::AllZerosRareSpikes,
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
            Archetype::RegimeShiftFlatToTrend => "regime_shift_flat_to_trend",
            Archetype::ContaminatedSeasonal => "contaminated_seasonal",
            Archetype::StudentTTrended => "student_t_trended",
            Archetype::IntermittentBursty => "intermittent_bursty",
            Archetype::EvolvingVarianceSmooth => "evolving_variance_smooth",
            Archetype::FadingSeasonality => "fading_seasonality",
            Archetype::TrendJumpsHeavyTails => "trend_jumps_heavy_tails",
            Archetype::ZeroInflatedSeasonal => "zero_inflated_seasonal",
            Archetype::GarchVolatilityClustering => "garch_volatility_clustering",
            Archetype::BimodalRegimeSwitch => "bimodal_regime_switch",
            Archetype::TickGridWalk => "tick_grid_walk",
            Archetype::GammaPositiveSkewed => "gamma_positive_skewed",
            Archetype::LognormalMultiplicative => "lognormal_multiplicative",
            Archetype::NegbinOverdispersedCounts => "negbin_overdispersed_counts",
            Archetype::MultiplicativeSeasonality => "multiplicative_seasonality",
            Archetype::Ar1Persistent => "ar1_persistent",
            Archetype::PiecewiseLinearTrend => "piecewise_linear_trend",
            Archetype::ExponentialGrowth => "exponential_growth",
            Archetype::SCurveLogisticGrowth => "s_curve_logistic_growth",
            Archetype::RetailWithPromotions => "retail_with_promotions",
            Archetype::WeeklyPlusDailyPlusSpike => "weekly_plus_daily_plus_spike",
            Archetype::NearConstantLowVar => "near_constant_low_var",
            Archetype::AllZerosRareSpikes => "all_zeros_rare_spikes",
        }
    }

    /// Which "team" the DGP is stacked toward. Wins-by-category is what
    /// makes the "when to use Laplace" story readable.
    fn category(self) -> Category {
        match self {
            // Textbook parametric DGP — AutoETS's home turf.
            Archetype::StationarySeasonalShort
            | Archetype::StationarySeasonalLong
            | Archetype::SeasonalLinearTrend
            | Archetype::SeasonalDampedTrend
            | Archetype::MultiSeasonalHourly
            | Archetype::LinearTrendOnly
            | Archetype::EverythingAtOnce
            | Archetype::HeteroscedasticMultiSeasonal
            | Archetype::MultiplicativeSeasonality
            | Archetype::ExponentialGrowth
            | Archetype::SCurveLogisticGrowth => Category::AutoETSFavoring,
            // Non-parametric / regime / heavy-tail / count / discrete —
            // Laplace's design targets.
            Archetype::RandomWalk
            | Archetype::MeanRevertingOu
            | Archetype::LevelShiftMidway
            | Archetype::HeavyTailCauchy
            | Archetype::StudentTDf3
            | Archetype::PoissonIntermittent
            | Archetype::PoissonSeasonalRetail
            | Archetype::RegimeShiftFlatToTrend
            | Archetype::ContaminatedSeasonal
            | Archetype::StudentTTrended
            | Archetype::IntermittentBursty
            | Archetype::EvolvingVarianceSmooth
            | Archetype::FadingSeasonality
            | Archetype::TrendJumpsHeavyTails
            | Archetype::ZeroInflatedSeasonal
            | Archetype::GarchVolatilityClustering
            | Archetype::BimodalRegimeSwitch
            | Archetype::TickGridWalk
            | Archetype::GammaPositiveSkewed
            | Archetype::LognormalMultiplicative
            | Archetype::NegbinOverdispersedCounts
            | Archetype::Ar1Persistent
            | Archetype::PiecewiseLinearTrend
            | Archetype::RetailWithPromotions
            | Archetype::WeeklyPlusDailyPlusSpike
            | Archetype::AllZerosRareSpikes => Category::LaplaceFavoring,
            // Genuinely ambiguous.
            Archetype::PureGaussianNoise
            | Archetype::ShortGaussian
            | Archetype::VarianceRegimeChange
            | Archetype::NearConstantLowVar => Category::Neutral,
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
            Archetype::RegimeShiftFlatToTrend => 300,
            Archetype::ContaminatedSeasonal => 300,
            Archetype::StudentTTrended => 300,
            Archetype::IntermittentBursty => 250,
            Archetype::EvolvingVarianceSmooth => 300,
            Archetype::FadingSeasonality => 400,
            Archetype::TrendJumpsHeavyTails => 400,
            Archetype::ZeroInflatedSeasonal => 280,
            Archetype::GarchVolatilityClustering => 400,
            Archetype::BimodalRegimeSwitch => 300,
            Archetype::TickGridWalk => 300,
            Archetype::GammaPositiveSkewed => 300,
            Archetype::LognormalMultiplicative => 300,
            Archetype::NegbinOverdispersedCounts => 300,
            Archetype::MultiplicativeSeasonality => 300,
            Archetype::Ar1Persistent => 300,
            Archetype::PiecewiseLinearTrend => 300,
            Archetype::ExponentialGrowth => 250,
            Archetype::SCurveLogisticGrowth => 250,
            Archetype::RetailWithPromotions => 280,
            Archetype::WeeklyPlusDailyPlusSpike => 800,
            Archetype::NearConstantLowVar => 200,
            Archetype::AllZerosRareSpikes => 200,
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
            | Archetype::EverythingAtOnce
            | Archetype::ContaminatedSeasonal
            | Archetype::FadingSeasonality
            | Archetype::MultiplicativeSeasonality => Some(12),
            Archetype::MultiSeasonalHourly
            | Archetype::HeteroscedasticMultiSeasonal
            | Archetype::WeeklyPlusDailyPlusSpike => Some(24),
            Archetype::PoissonSeasonalRetail
            | Archetype::ZeroInflatedSeasonal
            | Archetype::RetailWithPromotions => Some(7),
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
            Archetype::MultiSeasonalHourly
            | Archetype::HeteroscedasticMultiSeasonal
            | Archetype::WeeklyPlusDailyPlusSpike => 48,
            Archetype::EverythingAtOnce => 24,
            Archetype::PoissonSeasonalRetail
            | Archetype::ZeroInflatedSeasonal
            | Archetype::RetailWithPromotions => 14,
            Archetype::ContaminatedSeasonal | Archetype::FadingSeasonality => 18,
            Archetype::MultiplicativeSeasonality => 18,
            _ => 20,
        }
    }

    /// What the router SHOULD pick, given the archetype's designed shape.
    /// A mismatch is a router bug worth investigating.
    ///
    /// Note: `level_shift_midway` and `bimodal_regime_switch` are
    /// designed as "structural break" archetypes, but from a
    /// first-differences standpoint they look heavy-tailed (rare large
    /// residuals). The router's differencing-based heavy-tail check
    /// routes them to `HeavyTailedCrps` — this is arguably correct
    /// behaviour, since CRPS scoring handles both cases robustly.
    fn expected_recipe(self) -> RecipeKind {
        match self {
            Archetype::ShortGaussian => RecipeKind::ShortHistory,
            Archetype::PoissonIntermittent
            | Archetype::PoissonSeasonalRetail
            | Archetype::IntermittentBursty
            | Archetype::ZeroInflatedSeasonal
            | Archetype::NegbinOverdispersedCounts
            | Archetype::AllZerosRareSpikes
            | Archetype::RetailWithPromotions => RecipeKind::RetailCountAid,
            Archetype::HeavyTailCauchy
            | Archetype::StudentTDf3
            | Archetype::StudentTTrended
            | Archetype::TrendJumpsHeavyTails
            | Archetype::GarchVolatilityClustering
            | Archetype::ContaminatedSeasonal
            | Archetype::LevelShiftMidway
            | Archetype::BimodalRegimeSwitch
            | Archetype::LognormalMultiplicative => RecipeKind::HeavyTailedCrps,
            // Seasonal + long enough for period activation.
            Archetype::StationarySeasonalShort
            | Archetype::StationarySeasonalLong
            | Archetype::SeasonalLinearTrend
            | Archetype::SeasonalDampedTrend
            | Archetype::MultiSeasonalHourly
            | Archetype::HeteroscedasticMultiSeasonal
            | Archetype::EverythingAtOnce
            | Archetype::FadingSeasonality
            | Archetype::MultiplicativeSeasonality
            | Archetype::WeeklyPlusDailyPlusSpike => RecipeKind::ContinuousMultiScale,
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

fn gamma_sample(rng: &mut StdRng, shape: f64, rate: f64) -> f64 {
    Gamma::new(shape, rate).unwrap().sample(rng)
}

fn lognormal_sample(rng: &mut StdRng, mu: f64, sigma: f64) -> f64 {
    (mu + sigma * standard_normal(rng)).exp()
}

fn neg_bin_sample(rng: &mut StdRng, r: f64, p: f64) -> f64 {
    NegativeBinomial::new(r, p).unwrap().sample(rng) as f64
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
        // ===== 2026-07-23 additions =====
        Archetype::RegimeShiftFlatToTrend => {
            let mid = n / 2;
            for i in 0..n {
                let base = if i < mid {
                    50.0
                } else {
                    50.0 + 0.15 * (i - mid) as f64
                };
                y.push(base + standard_normal(&mut rng));
            }
        }
        Archetype::ContaminatedSeasonal => {
            let p = 12.0;
            for i in 0..n {
                let base = 50.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                let u: f64 = rng.gen();
                let noise = if u < 0.05 {
                    // 5 % Cauchy contamination: rare, large.
                    cauchy(&mut rng, 3.0)
                } else {
                    standard_normal(&mut rng)
                };
                y.push(base + noise);
            }
        }
        Archetype::StudentTTrended => {
            for i in 0..n {
                y.push(50.0 + 0.05 * i as f64 + student_t(&mut rng, 3.0));
            }
        }
        Archetype::IntermittentBursty => {
            for _ in 0..n {
                let u: f64 = rng.gen();
                if u < 0.2 {
                    y.push(poisson_sample(&mut rng, 5.0));
                } else {
                    y.push(0.0);
                }
            }
        }
        Archetype::EvolvingVarianceSmooth => {
            for i in 0..n {
                let sigma = 1.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin().abs();
                y.push(50.0 + sigma * standard_normal(&mut rng));
            }
        }
        Archetype::FadingSeasonality => {
            let p = 12.0;
            for i in 0..n {
                let fade = 1.0 - i as f64 / n as f64;
                let s = 5.0 * fade * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                y.push(50.0 + s + standard_normal(&mut rng));
            }
        }
        Archetype::TrendJumpsHeavyTails => {
            for i in 0..n {
                let mut base = 50.0 + 0.03 * i as f64;
                if i >= n / 4 {
                    base += 4.0;
                }
                if i >= 3 * n / 4 {
                    base -= 6.0;
                }
                y.push(base + student_t(&mut rng, 3.0));
            }
        }
        Archetype::ZeroInflatedSeasonal => {
            // 70 % forced zeros; the 30 % that fire are Poisson at a weekly rate.
            let rates = [1.5, 1.0, 0.8, 0.8, 1.5, 6.0, 8.0];
            for i in 0..n {
                let u: f64 = rng.gen();
                if u < 0.7 {
                    y.push(0.0);
                } else {
                    y.push(poisson_sample(&mut rng, rates[i % 7]));
                }
            }
        }
        Archetype::GarchVolatilityClustering => {
            // GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            let omega: f64 = 0.05;
            let alpha: f64 = 0.15;
            let beta: f64 = 0.80;
            let mut sigma2: f64 = omega / (1.0 - alpha - beta);
            let mut eps_prev: f64 = 0.0;
            for _ in 0..n {
                sigma2 = omega + alpha * eps_prev * eps_prev + beta * sigma2;
                let eps = sigma2.sqrt() * standard_normal(&mut rng);
                y.push(50.0 + eps);
                eps_prev = eps;
            }
        }
        Archetype::BimodalRegimeSwitch => {
            // Markov switch between (μ=45, σ=1) and (μ=55, σ=1). Switch prob 0.02.
            let mut state: u8 = 0;
            for _ in 0..n {
                let switch: f64 = rng.gen();
                if switch < 0.02 {
                    state = 1 - state;
                }
                let mu = if state == 0 { 45.0 } else { 55.0 };
                y.push(mu + standard_normal(&mut rng));
            }
        }
        Archetype::TickGridWalk => {
            let mut x = 50.0;
            for _ in 0..n {
                x += standard_normal(&mut rng);
                // Snap to the 0.25 tick grid.
                y.push((x * 4.0).round() / 4.0);
            }
        }
        // ===== 2026-07-24 additions =====
        Archetype::GammaPositiveSkewed => {
            // Gamma(α=2, β=1) has mean=2, var=2, right-skewed. Shift so
            // most values sit around ~50.
            for _ in 0..n {
                y.push(48.0 + gamma_sample(&mut rng, 2.0, 1.0));
            }
        }
        Archetype::LognormalMultiplicative => {
            // Slow-growing base level; lognormal multiplicative noise
            // gives a right-skewed distribution with a heavy right tail.
            for i in 0..n {
                let base = 10.0 + 0.02 * i as f64;
                y.push(base * lognormal_sample(&mut rng, 0.0, 0.3));
            }
        }
        Archetype::NegbinOverdispersedCounts => {
            // NegBin(r=3, p=0.4) — mean=r(1-p)/p=4.5, var=r(1-p)/p²=11.25
            // (overdispersed vs Poisson at the same mean).
            for _ in 0..n {
                y.push(neg_bin_sample(&mut rng, 3.0, 0.4));
            }
        }
        Archetype::MultiplicativeSeasonality => {
            // Seasonal amplitude scales with level. Level grows linearly.
            let p = 12.0;
            for i in 0..n {
                let level = 20.0 + 0.05 * i as f64;
                let season = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * i as f64 / p).sin();
                y.push(level * season + standard_normal(&mut rng));
            }
        }
        Archetype::Ar1Persistent => {
            // AR(1) with φ=0.9 and long-run mean 50.
            let phi = 0.9;
            let mu = 50.0;
            let sigma = 1.0;
            let mut x = mu;
            for _ in 0..n {
                x = mu + phi * (x - mu) + sigma * standard_normal(&mut rng);
                y.push(x);
            }
        }
        Archetype::PiecewiseLinearTrend => {
            // Slope 0.10 for first third, -0.05 for middle, 0.15 for last.
            let seg1 = n / 3;
            let seg2 = 2 * n / 3;
            let mut base = 50.0;
            for i in 0..n {
                if i > 0 {
                    let slope = if i < seg1 {
                        0.10
                    } else if i < seg2 {
                        -0.05
                    } else {
                        0.15
                    };
                    base += slope;
                }
                y.push(base + standard_normal(&mut rng));
            }
        }
        Archetype::ExponentialGrowth => {
            // y = a·exp(b·t) + noise. b=0.015 for gentle compounding.
            for i in 0..n {
                y.push(5.0 * (0.015 * i as f64).exp() + standard_normal(&mut rng));
            }
        }
        Archetype::SCurveLogisticGrowth => {
            // Logistic: L / (1 + exp(-k·(t - t0))).
            let capacity = 100.0;
            let k = 0.03;
            let t0 = n as f64 / 2.0;
            for i in 0..n {
                let s = capacity / (1.0 + (-k * (i as f64 - t0)).exp());
                y.push(s + standard_normal(&mut rng));
            }
        }
        Archetype::RetailWithPromotions => {
            // Weekly rate + Poisson counts + 5 % of days get a
            // multiplicative 3-8× promotion spike.
            let rates = [1.2, 1.0, 0.8, 0.8, 1.2, 3.0, 4.0];
            for i in 0..n {
                let base = poisson_sample(&mut rng, rates[i % 7]);
                let u: f64 = rng.gen();
                if u < 0.05 {
                    let mult = 3.0 + rng.gen::<f64>() * 5.0;
                    y.push((base * mult).round());
                } else {
                    y.push(base);
                }
            }
        }
        Archetype::WeeklyPlusDailyPlusSpike => {
            // Continuous-valued web-traffic pattern: daily + weekly
            // cycles plus rare 10× release spikes.
            for i in 0..n {
                let daily = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin();
                let weekly = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin();
                let base = 50.0 + daily + weekly + standard_normal(&mut rng);
                let u: f64 = rng.gen();
                let final_val = if u < 0.02 {
                    base + 30.0 // 10× the daily amplitude
                } else {
                    base
                };
                y.push(final_val);
            }
        }
        Archetype::NearConstantLowVar => {
            // σ=0.01 — numerical edge case; MASE denominator (naive-1
            // error) will be tiny so absolute MASE values inflate,
            // but that's a fair test of numerical robustness.
            for _ in 0..n {
                y.push(50.0 + 0.01 * standard_normal(&mut rng));
            }
        }
        Archetype::AllZerosRareSpikes => {
            // 99 % zeros, 1 % Poisson(10) spikes.
            for _ in 0..n {
                let u: f64 = rng.gen();
                if u < 0.01 {
                    y.push(poisson_sample(&mut rng, 10.0));
                } else {
                    y.push(0.0);
                }
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

const MODEL_NAMES: [&str; 6] = [
    "AutoETS",
    "AutoTheta",
    "Lap.auto()",
    "recommended_for",
    "MS+3SH manual",
    "SmartForecaster",
];
const N_MODELS: usize = 6;

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

    // M5: SmartForecaster — the 2026-07-24 cross-family router.
    // Point-only via Forecaster trait; WQL via Gaussian fallback at
    // scale = naive-1 σ, matching the AutoETS / AutoTheta convention.
    {
        let t0 = Instant::now();
        let mut m = if let Some(p) = period {
            SmartForecaster::new().with_seasonal_period(p.max(2))
        } else {
            SmartForecaster::new()
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
                    out[5] = Some((ms, w, t0.elapsed().as_secs_f64()));
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
    // Per-archetype winner, kept for the segmented report below.
    let mut winner_per_arch = Vec::with_capacity(Archetype::ALL.len());
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
        winner_per_arch.push(best_i);
    }
    print!("{:<34}", "wins on MASE");
    for w in &wins {
        print!("{:>18}", w);
    }
    println!();

    // ---- Segmented by Category (the "when to use Laplace" table) ----
    println!("\n=== Wins by category (MASE) — the \"when to use Laplace\" cut ===");
    print!("{:<34}", "category");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    print!("{:>10}", "count");
    println!();
    for cat in [
        Category::LaplaceFavoring,
        Category::AutoETSFavoring,
        Category::Neutral,
    ] {
        let mut w = vec![0usize; N_MODELS];
        let mut n = 0usize;
        for (ai, arch) in Archetype::ALL.iter().enumerate() {
            if arch.category() == cat {
                w[winner_per_arch[ai]] += 1;
                n += 1;
            }
        }
        print!("{:<34}", cat.name());
        for wi in &w {
            print!("{:>18}", wi);
        }
        print!("{:>10}", n);
        println!();
    }

    // Per-category geomean of MASE (geomean across archetypes in that category).
    println!("\n=== Geomean MASE by category ===");
    print!("{:<34}", "category");
    for m in MODEL_NAMES.iter() {
        print!("{:>18}", m);
    }
    println!();
    for cat in [
        Category::LaplaceFavoring,
        Category::AutoETSFavoring,
        Category::Neutral,
    ] {
        print!("{:<34}", cat.name());
        for mi in 0..N_MODELS {
            let per_arch: Vec<f64> = Archetype::ALL
                .iter()
                .enumerate()
                .filter(|(_, a)| a.category() == cat)
                .map(|(ai, _)| geomean(&mase_by_arch[ai][mi]))
                .collect();
            let g = geomean(&per_arch);
            print!("{:>18.4}", g);
        }
        println!();
    }

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
