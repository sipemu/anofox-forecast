//! Enum-dispatch `LeafEnum` — devirtualization for the fit hot path.
//!
//! Replaces `Box<dyn Leaf + Send>` in the softmax pool with a
//! `LeafEnum` variant so the compiler can inline `predict_one` /
//! `observe` across all leaf types.
//!
//! The wrapper leaves (`YjWrappedLeaf`, `GarchWrappedLeaf`,
//! `PowerTransformWrapper`, `SlowStandardizeWrapper`,
//! `SeasonalDifferenceWrapper`, `StandardizeWrapper`) contain a
//! `Box<LeafEnum>` inner so composition stays flexible.
//!
//! Item 3 of #180 (perf audit).

use super::dist::Gaussian;
use super::leaf::Leaf;
use super::leaves::{
    Ar1Leaf, Ar2Leaf, BetaLeaf, DiscreteUniformLeaf, DriftLeaf, EmaLeaf, FractionalDiffLeaf,
    GammaLeaf, HoltLeaf, IntermittentLeaf, LogNormalLeaf, MultiplicativeSeasonalLeaf,
    NegativeBinomialLeaf, OuLeaf, PoissonLeaf, RectifiedNormalLeaf, SeasonalEmaLeaf,
    SeasonalIntermittentLeaf, SkewNormalLeaf, StudentTLeaf, ThetaLeaf, TweedieLeaf,
    ZeroInflatedNegativeBinomialLeaf, ZeroInflatedPoissonLeaf,
};

/// Concrete leaf variant for the fit-loop pool.
///
/// Excludes the six wrapper leaves (which hold a `Box<LeafEnum>` inner
/// and reach via [`Wrapper`]) — they're a separate enum below to keep
/// this one non-recursive and stack-sized.
pub enum LeafEnum {
    Ema(EmaLeaf),
    Drift(DriftLeaf),
    Ar1(Ar1Leaf),
    Ar2(Ar2Leaf),
    Holt(HoltLeaf),
    Theta(ThetaLeaf),
    Ou(OuLeaf),
    FracDiff(FractionalDiffLeaf),
    SeasonalEma(SeasonalEmaLeaf),
    SeasonalIntermittent(SeasonalIntermittentLeaf),
    SeasonalMult(MultiplicativeSeasonalLeaf),
    Intermittent(IntermittentLeaf),
    Poisson(PoissonLeaf),
    NegativeBinomial(NegativeBinomialLeaf),
    LogNormal(LogNormalLeaf),
    Gamma(GammaLeaf),
    RectifiedNormal(RectifiedNormalLeaf),
    StudentT(StudentTLeaf),
    Beta(BetaLeaf),
    Tweedie(TweedieLeaf),
    SkewNormal(SkewNormalLeaf),
    DiscreteUniform(DiscreteUniformLeaf),
    Zip(ZeroInflatedPoissonLeaf),
    Zinb(ZeroInflatedNegativeBinomialLeaf),
    /// Escape hatch for wrapped leaves — kept as trait-object to allow
    /// composed leaves (YJ / GARCH / PowerTransform / Standardize /
    /// SlowStandardize / SeasonalDiff wrappers) without recursing the
    /// enum. The virtual-dispatch cost is amortized over the wrapper's
    /// inner-leaf work.
    Wrapped(Box<dyn Leaf + Send>),
}

macro_rules! dispatch {
    ($self:ident, $method:ident $(, $arg:expr)*) => {
        match $self {
            LeafEnum::Ema(l) => l.$method($($arg),*),
            LeafEnum::Drift(l) => l.$method($($arg),*),
            LeafEnum::Ar1(l) => l.$method($($arg),*),
            LeafEnum::Ar2(l) => l.$method($($arg),*),
            LeafEnum::Holt(l) => l.$method($($arg),*),
            LeafEnum::Theta(l) => l.$method($($arg),*),
            LeafEnum::Ou(l) => l.$method($($arg),*),
            LeafEnum::FracDiff(l) => l.$method($($arg),*),
            LeafEnum::SeasonalEma(l) => l.$method($($arg),*),
            LeafEnum::SeasonalIntermittent(l) => l.$method($($arg),*),
            LeafEnum::SeasonalMult(l) => l.$method($($arg),*),
            LeafEnum::Intermittent(l) => l.$method($($arg),*),
            LeafEnum::Poisson(l) => l.$method($($arg),*),
            LeafEnum::NegativeBinomial(l) => l.$method($($arg),*),
            LeafEnum::LogNormal(l) => l.$method($($arg),*),
            LeafEnum::Gamma(l) => l.$method($($arg),*),
            LeafEnum::RectifiedNormal(l) => l.$method($($arg),*),
            LeafEnum::StudentT(l) => l.$method($($arg),*),
            LeafEnum::Beta(l) => l.$method($($arg),*),
            LeafEnum::Tweedie(l) => l.$method($($arg),*),
            LeafEnum::SkewNormal(l) => l.$method($($arg),*),
            LeafEnum::DiscreteUniform(l) => l.$method($($arg),*),
            LeafEnum::Zip(l) => l.$method($($arg),*),
            LeafEnum::Zinb(l) => l.$method($($arg),*),
            LeafEnum::Wrapped(l) => l.$method($($arg),*),
        }
    };
}

impl LeafEnum {
    /// Inlined equivalent of `Leaf::predict_one` via enum-match.
    #[inline]
    pub fn predict_one(&self) -> Gaussian {
        dispatch!(self, predict_one)
    }

    /// Inlined equivalent of `Leaf::observe` via enum-match.
    #[inline]
    pub fn observe(&mut self, y: f64) {
        dispatch!(self, observe, y)
    }

    /// Inlined equivalent of `Leaf::predict` via enum-match. Falls
    /// through to a `Vec<Gaussian>` allocation; only used off the fit
    /// hot path (forecast_dist multi-horizon).
    pub fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        dispatch!(self, predict, horizon)
    }

    pub fn name(&self) -> &'static str {
        dispatch!(self, name)
    }
}

/// Also implement the `Leaf` trait on `LeafEnum` for API compatibility
/// (wrappers hold `Box<LeafEnum>` and need trait access).
impl Leaf for LeafEnum {
    fn name(&self) -> &'static str {
        LeafEnum::name(self)
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        LeafEnum::predict(self, horizon)
    }

    fn predict_one(&self) -> Gaussian {
        LeafEnum::predict_one(self)
    }

    fn observe(&mut self, y: f64) {
        LeafEnum::observe(self, y)
    }
}
