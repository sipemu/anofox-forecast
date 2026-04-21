//! Pre-modeling forecastability analysis via information-theoretic dependence
//! measures.
//!
//! Determines whether a time series has exploitable predictive structure
//! *before* running expensive model search. The methods here answer:
//!
//! - **"Is there any predictive signal?"** — via [`ami_curve`] (Average
//!   Mutual Information at each horizon lag) and significance testing with
//!   [`phase_surrogates`].
//! - **"Is the signal linear or nonlinear?"** — via comparison of AMI with
//!   [`gcmi_curve`] (Gaussian Copula MI, which only captures linear
//!   dependence) and the [`ForecastabilityFingerprint::nonlinear_share`]
//!   summary.
//! - **"Does X predict Y beyond Y's own history?"** — via
//!   [`transfer_entropy_curve`] (conditional MI).
//! - **"How deep does the dependence go?"** — via [`pami_curve`] (Partial
//!   AMI, conditioning on intermediate lags).
//! - **"How chaotic is the system?"** — via [`largest_lyapunov_exponent`]
//!   (Rosenstein et al. 1993).
//!
//! # Attribution
//!
//! The algorithms and API design are inspired by the Python package
//! [`dependence-forecastability`](https://github.com/AdamKrysztopa/dependence-forecastability)
//! by Adam Krysztopa, released under the MIT License.
//!
//! # References
//!
//! - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating
//!   mutual information. *Physical Review E*, 69(6), 066138.
//! - Ince, R. A. A., et al. (2017). A statistical framework for
//!   neuroimaging data analysis based on mutual information estimated via a
//!   Gaussian copula. *Human Brain Mapping*, 38(3), 1541–1573.
//! - Szekely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007). Measuring and
//!   testing dependence by correlation of distances. *The Annals of
//!   Statistics*, 35(6), 2769–2794.
//! - Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A
//!   practical method for calculating largest Lyapunov exponents from small
//!   data sets. *Physica D*, 65(1–2), 117–134.
//! - Bandt, C. & Pompe, B. (2002). Permutation entropy: A natural complexity
//!   measure for time series. *Physical Review Letters*, 88(17), 174102.

pub mod distance_correlation;
pub mod fft_complex;
pub mod gcmi;
pub mod knn_mi;
pub mod surrogates;

// Phase 2
pub mod ami;
pub mod lag_correlations;
pub mod transfer_entropy;

// Phase 3
pub mod fingerprint;
pub mod lyapunov;
pub mod scorers;
pub mod theoretical;

// Triage pipeline
pub mod triage;

pub use ami::{ami_curve, gcmi_curve, pami_curve, CmiBackend};
pub use distance_correlation::distance_correlation;
pub use fingerprint::ForecastabilityFingerprint;
pub use gcmi::gcmi;
pub use knn_mi::knn_mutual_information;
pub use lag_correlations::{kendall_curve, pearson_curve, spearman_curve};
pub use lyapunov::largest_lyapunov_exponent;
pub use scorers::{score, Scorer};
pub use surrogates::{phase_surrogates, significance_bands, SignificanceBands};
pub use theoretical::ar1_theoretical_ami;
pub use transfer_entropy::transfer_entropy_curve;
pub use triage::{
    run_batch_triage, run_triage, screen_exogenous, BatchTriageResult, ModelFamily, SeriesPattern,
    TriageConfig, TriageResult,
};
