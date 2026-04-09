//! Baked-in critical value lookup table for sequential CUSUM detectors.
//!
//! Mirrors `critValTable` / `sysdata.rda` from the R package. Values are
//! pre-simulated via [`simulate_critical_value`](super::sequential_crit::simulate_critical_value)
//! with a fixed seed so the table is reproducible. Regenerate with:
//!
//! ```bash
//! cargo test --features parallel --release \
//!   -- monitor::sequential_table::regenerate_crit_val_table --ignored --nocapture
//! ```
//!
//! The grid matches the R package one-for-one:
//! `4 detectors × 19 gammas (0.000..0.450 step 0.025) × 3 alphas (0.01, 0.05, 0.10) = 228 entries`.

use super::sequential::Detector;

/// Tolerance for matching `gamma` and `alpha` arguments to baked grid points.
const GRID_TOL: f64 = 1e-9;

/// Gamma grid: 0.000, 0.025, …, 0.450 (19 points).
pub const GAMMAS: [f64; 19] = [
    0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300,
    0.325, 0.350, 0.375, 0.400, 0.425, 0.450,
];

/// Alpha grid: 0.01, 0.05, 0.10.
pub const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];

/// One entry in the baked lookup table.
///
/// Stored as a flat `(detector, gamma, alpha, value)` tuple to make the
/// constant compact and reviewable in git.
#[derive(Debug, Clone, Copy)]
pub struct CritValEntry {
    pub detector: Detector,
    pub gamma: f64,
    pub alpha: f64,
    pub value: f64,
}

/// Look up a baked critical value.
///
/// Returns `None` when the `(detector, gamma, alpha)` triple is not on the
/// pre-simulated grid. Callers should fall back to [`simulate_critical_value`](
/// super::sequential_crit::simulate_critical_value) in that case.
pub fn lookup_critical_value(detector: Detector, gamma: f64, alpha: f64) -> Option<f64> {
    CRIT_VAL_TABLE
        .iter()
        .find(|e| {
            e.detector == detector
                && (e.gamma - gamma).abs() < GRID_TOL
                && (e.alpha - alpha).abs() < GRID_TOL
        })
        .map(|e| e.value)
}

/// Check whether the supplied `(gamma, alpha)` lies on the baked grid.
pub fn is_on_grid(gamma: f64, alpha: f64) -> bool {
    GAMMAS.iter().any(|g| (g - gamma).abs() < GRID_TOL)
        && ALPHAS.iter().any(|a| (a - alpha).abs() < GRID_TOL)
}

/// Pre-simulated critical values (4 × 19 × 3 = 228 entries).
///
/// Generated with:
/// - `samples = 10_000`
/// - `npts   = 500`
/// - `seed   = 0xC4_D5_E6_F7_01_23_45_67 + sample_index`  (deterministic)
///
/// These values are the `(1 - alpha)` quantile of the detector's asymptotic
/// limit distribution. The sampling error at 10 000 draws is roughly
/// `±1 %` of the value — stable enough for a production threshold.
#[rustfmt::skip]
pub const CRIT_VAL_TABLE: &[CritValEntry] = &[
    // Detector::Cusum
    CritValEntry { detector: Detector::Cusum, gamma: 0.000, alpha: 0.01, value: 2.809992 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.000, alpha: 0.05, value: 2.223624 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.000, alpha: 0.10, value: 1.930590 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.025, alpha: 0.01, value: 2.823215 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.025, alpha: 0.05, value: 2.234038 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.025, alpha: 0.10, value: 1.941959 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.050, alpha: 0.01, value: 2.830044 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.050, alpha: 0.05, value: 2.241880 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.050, alpha: 0.10, value: 1.952797 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.075, alpha: 0.01, value: 2.835973 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.075, alpha: 0.05, value: 2.254782 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.075, alpha: 0.10, value: 1.965503 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.100, alpha: 0.01, value: 2.845726 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.100, alpha: 0.05, value: 2.265130 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.100, alpha: 0.10, value: 1.979764 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.125, alpha: 0.01, value: 2.853901 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.125, alpha: 0.05, value: 2.277119 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.125, alpha: 0.10, value: 1.993097 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.150, alpha: 0.01, value: 2.860843 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.150, alpha: 0.05, value: 2.287916 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.150, alpha: 0.10, value: 2.005267 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.175, alpha: 0.01, value: 2.880422 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.175, alpha: 0.05, value: 2.302299 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.175, alpha: 0.10, value: 2.018311 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.200, alpha: 0.01, value: 2.889192 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.200, alpha: 0.05, value: 2.316567 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.200, alpha: 0.10, value: 2.036821 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.225, alpha: 0.01, value: 2.903944 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.225, alpha: 0.05, value: 2.328326 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.225, alpha: 0.10, value: 2.058847 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.250, alpha: 0.01, value: 2.917767 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.250, alpha: 0.05, value: 2.353567 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.250, alpha: 0.10, value: 2.084002 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.275, alpha: 0.01, value: 2.932577 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.275, alpha: 0.05, value: 2.380403 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.275, alpha: 0.10, value: 2.113654 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.300, alpha: 0.01, value: 2.959861 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.300, alpha: 0.05, value: 2.411698 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.300, alpha: 0.10, value: 2.137363 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.325, alpha: 0.01, value: 2.978728 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.325, alpha: 0.05, value: 2.442953 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.325, alpha: 0.10, value: 2.173130 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.350, alpha: 0.01, value: 2.999316 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.350, alpha: 0.05, value: 2.476427 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.350, alpha: 0.10, value: 2.213566 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.375, alpha: 0.01, value: 3.029436 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.375, alpha: 0.05, value: 2.520173 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.375, alpha: 0.10, value: 2.258153 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.400, alpha: 0.01, value: 3.076537 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.400, alpha: 0.05, value: 2.582274 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.400, alpha: 0.10, value: 2.308306 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.425, alpha: 0.01, value: 3.142402 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.425, alpha: 0.05, value: 2.648712 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.425, alpha: 0.10, value: 2.387210 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.450, alpha: 0.01, value: 3.229189 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.450, alpha: 0.05, value: 2.744508 },
    CritValEntry { detector: Detector::Cusum, gamma: 0.450, alpha: 0.10, value: 2.476353 },
    // Detector::Cusum1
    CritValEntry { detector: Detector::Cusum1, gamma: 0.000, alpha: 0.01, value: 2.557434 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.000, alpha: 0.05, value: 1.949008 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.000, alpha: 0.10, value: 1.629245 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.025, alpha: 0.01, value: 2.562692 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.025, alpha: 0.05, value: 1.959428 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.025, alpha: 0.10, value: 1.636465 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.050, alpha: 0.01, value: 2.565633 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.050, alpha: 0.05, value: 1.974754 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.050, alpha: 0.10, value: 1.648053 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.075, alpha: 0.01, value: 2.580003 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.075, alpha: 0.05, value: 1.990606 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.075, alpha: 0.10, value: 1.658633 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.100, alpha: 0.01, value: 2.588480 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.100, alpha: 0.05, value: 2.003389 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.100, alpha: 0.10, value: 1.671713 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.125, alpha: 0.01, value: 2.604464 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.125, alpha: 0.05, value: 2.016254 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.125, alpha: 0.10, value: 1.688953 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.150, alpha: 0.01, value: 2.613093 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.150, alpha: 0.05, value: 2.028137 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.150, alpha: 0.10, value: 1.705132 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.175, alpha: 0.01, value: 2.629558 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.175, alpha: 0.05, value: 2.041188 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.175, alpha: 0.10, value: 1.725708 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.200, alpha: 0.01, value: 2.641446 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.200, alpha: 0.05, value: 2.060159 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.200, alpha: 0.10, value: 1.745887 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.225, alpha: 0.01, value: 2.667202 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.225, alpha: 0.05, value: 2.077727 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.225, alpha: 0.10, value: 1.766523 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.250, alpha: 0.01, value: 2.677713 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.250, alpha: 0.05, value: 2.101448 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.250, alpha: 0.10, value: 1.792548 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.275, alpha: 0.01, value: 2.711548 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.275, alpha: 0.05, value: 2.123433 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.275, alpha: 0.10, value: 1.814621 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.300, alpha: 0.01, value: 2.743940 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.300, alpha: 0.05, value: 2.156085 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.300, alpha: 0.10, value: 1.847494 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.325, alpha: 0.01, value: 2.766227 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.325, alpha: 0.05, value: 2.188154 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.325, alpha: 0.10, value: 1.879594 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.350, alpha: 0.01, value: 2.781231 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.350, alpha: 0.05, value: 2.229946 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.350, alpha: 0.10, value: 1.927265 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.375, alpha: 0.01, value: 2.822995 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.375, alpha: 0.05, value: 2.279362 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.375, alpha: 0.10, value: 1.980029 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.400, alpha: 0.01, value: 2.887342 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.400, alpha: 0.05, value: 2.323187 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.400, alpha: 0.10, value: 2.041025 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.425, alpha: 0.01, value: 2.945943 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.425, alpha: 0.05, value: 2.403335 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.425, alpha: 0.10, value: 2.118307 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.450, alpha: 0.01, value: 3.046129 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.450, alpha: 0.05, value: 2.508631 },
    CritValEntry { detector: Detector::Cusum1, gamma: 0.450, alpha: 0.10, value: 2.210169 },
    // Detector::PageCusum
    CritValEntry { detector: Detector::PageCusum, gamma: 0.000, alpha: 0.01, value: 2.823533 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.000, alpha: 0.05, value: 2.245956 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.000, alpha: 0.10, value: 1.968711 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.025, alpha: 0.01, value: 2.831755 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.025, alpha: 0.05, value: 2.254301 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.025, alpha: 0.10, value: 1.981024 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.050, alpha: 0.01, value: 2.834382 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.050, alpha: 0.05, value: 2.266954 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.050, alpha: 0.10, value: 1.997644 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.075, alpha: 0.01, value: 2.845679 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.075, alpha: 0.05, value: 2.279477 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.075, alpha: 0.10, value: 2.010216 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.100, alpha: 0.01, value: 2.850280 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.100, alpha: 0.05, value: 2.289404 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.100, alpha: 0.10, value: 2.021147 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.125, alpha: 0.01, value: 2.858898 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.125, alpha: 0.05, value: 2.303490 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.125, alpha: 0.10, value: 2.037110 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.150, alpha: 0.01, value: 2.874187 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.150, alpha: 0.05, value: 2.319240 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.150, alpha: 0.10, value: 2.054169 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.175, alpha: 0.01, value: 2.890749 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.175, alpha: 0.05, value: 2.333393 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.175, alpha: 0.10, value: 2.069218 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.200, alpha: 0.01, value: 2.909486 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.200, alpha: 0.05, value: 2.351291 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.200, alpha: 0.10, value: 2.089332 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.225, alpha: 0.01, value: 2.916087 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.225, alpha: 0.05, value: 2.372665 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.225, alpha: 0.10, value: 2.113759 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.250, alpha: 0.01, value: 2.933600 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.250, alpha: 0.05, value: 2.392806 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.250, alpha: 0.10, value: 2.138268 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.275, alpha: 0.01, value: 2.961618 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.275, alpha: 0.05, value: 2.421875 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.275, alpha: 0.10, value: 2.166825 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.300, alpha: 0.01, value: 2.986050 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.300, alpha: 0.05, value: 2.458042 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.300, alpha: 0.10, value: 2.199335 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.325, alpha: 0.01, value: 3.013405 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.325, alpha: 0.05, value: 2.495254 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.325, alpha: 0.10, value: 2.238961 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.350, alpha: 0.01, value: 3.036801 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.350, alpha: 0.05, value: 2.532802 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.350, alpha: 0.10, value: 2.281130 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.375, alpha: 0.01, value: 3.073598 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.375, alpha: 0.05, value: 2.579637 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.375, alpha: 0.10, value: 2.333758 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.400, alpha: 0.01, value: 3.102192 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.400, alpha: 0.05, value: 2.636177 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.400, alpha: 0.10, value: 2.388688 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.425, alpha: 0.01, value: 3.166442 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.425, alpha: 0.05, value: 2.709183 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.425, alpha: 0.10, value: 2.466160 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.450, alpha: 0.01, value: 3.272203 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.450, alpha: 0.05, value: 2.795235 },
    CritValEntry { detector: Detector::PageCusum, gamma: 0.450, alpha: 0.10, value: 2.556915 },
    // Detector::PageCusum1
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.000, alpha: 0.01, value: 2.566663 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.000, alpha: 0.05, value: 1.981608 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.000, alpha: 0.10, value: 1.670021 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.025, alpha: 0.01, value: 2.567186 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.025, alpha: 0.05, value: 2.001139 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.025, alpha: 0.10, value: 1.684825 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.050, alpha: 0.01, value: 2.575794 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.050, alpha: 0.05, value: 2.015299 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.050, alpha: 0.10, value: 1.700025 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.075, alpha: 0.01, value: 2.593607 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.075, alpha: 0.05, value: 2.025995 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.075, alpha: 0.10, value: 1.715577 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.100, alpha: 0.01, value: 2.603822 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.100, alpha: 0.05, value: 2.040229 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.100, alpha: 0.10, value: 1.732670 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.125, alpha: 0.01, value: 2.621233 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.125, alpha: 0.05, value: 2.058358 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.125, alpha: 0.10, value: 1.750871 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.150, alpha: 0.01, value: 2.636016 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.150, alpha: 0.05, value: 2.075932 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.150, alpha: 0.10, value: 1.766534 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.175, alpha: 0.01, value: 2.659927 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.175, alpha: 0.05, value: 2.091768 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.175, alpha: 0.10, value: 1.788910 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.200, alpha: 0.01, value: 2.670606 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.200, alpha: 0.05, value: 2.111925 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.200, alpha: 0.10, value: 1.809369 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.225, alpha: 0.01, value: 2.691768 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.225, alpha: 0.05, value: 2.133999 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.225, alpha: 0.10, value: 1.834228 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.250, alpha: 0.01, value: 2.721730 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.250, alpha: 0.05, value: 2.158763 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.250, alpha: 0.10, value: 1.864499 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.275, alpha: 0.01, value: 2.744963 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.275, alpha: 0.05, value: 2.185745 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.275, alpha: 0.10, value: 1.893721 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.300, alpha: 0.01, value: 2.775057 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.300, alpha: 0.05, value: 2.223767 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.300, alpha: 0.10, value: 1.928818 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.325, alpha: 0.01, value: 2.787141 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.325, alpha: 0.05, value: 2.260349 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.325, alpha: 0.10, value: 1.973909 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.350, alpha: 0.01, value: 2.815740 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.350, alpha: 0.05, value: 2.304671 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.350, alpha: 0.10, value: 2.021161 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.375, alpha: 0.01, value: 2.859897 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.375, alpha: 0.05, value: 2.360468 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.375, alpha: 0.10, value: 2.076834 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.400, alpha: 0.01, value: 2.927844 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.400, alpha: 0.05, value: 2.409977 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.400, alpha: 0.10, value: 2.134806 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.425, alpha: 0.01, value: 3.010024 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.425, alpha: 0.05, value: 2.485742 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.425, alpha: 0.10, value: 2.209410 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.450, alpha: 0.01, value: 3.093499 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.450, alpha: 0.05, value: 2.577356 },
    CritValEntry { detector: Detector::PageCusum1, gamma: 0.450, alpha: 0.10, value: 2.299179 },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::sequential_crit::simulate_critical_value;

    #[test]
    fn grid_membership() {
        assert!(is_on_grid(0.0, 0.05));
        assert!(is_on_grid(0.25, 0.01));
        assert!(is_on_grid(0.45, 0.10));
        assert!(!is_on_grid(0.5, 0.05));
        assert!(!is_on_grid(0.13, 0.05));
    }

    #[test]
    fn lookup_returns_none_off_grid() {
        // Off-grid gamma
        assert!(lookup_critical_value(Detector::PageCusum, 0.333, 0.05).is_none());
        // Off-grid alpha
        assert!(lookup_critical_value(Detector::PageCusum, 0.0, 0.07).is_none());
    }

    #[test]
    fn table_is_fully_populated() {
        // 4 detectors × 19 gammas × 3 alphas = 228 entries
        assert_eq!(CRIT_VAL_TABLE.len(), 228);
        for &d in &[
            Detector::Cusum,
            Detector::Cusum1,
            Detector::PageCusum,
            Detector::PageCusum1,
        ] {
            for &g in &GAMMAS {
                for &a in &ALPHAS {
                    let v = lookup_critical_value(d, g, a)
                        .unwrap_or_else(|| panic!("missing entry: {:?} g={} α={}", d, g, a));
                    assert!(v.is_finite() && v > 0.0);
                }
            }
        }
    }

    #[test]
    fn cusum_is_monotone_in_alpha() {
        // Stricter alpha => higher critical value (greater conservatism).
        for &d in &[Detector::PageCusum, Detector::Cusum1] {
            for &g in &[0.0, 0.25] {
                let v01 = lookup_critical_value(d, g, 0.01).unwrap();
                let v05 = lookup_critical_value(d, g, 0.05).unwrap();
                let v10 = lookup_critical_value(d, g, 0.10).unwrap();
                assert!(
                    v01 > v05 && v05 > v10,
                    "{:?} g={}: expected 0.01>0.05>0.10, got {} {} {}",
                    d,
                    g,
                    v01,
                    v05,
                    v10
                );
            }
        }
    }

    #[test]
    fn two_sided_exceeds_one_sided() {
        // Two-sided detectors must have higher critical values than their
        // one-sided counterparts at the same (γ, α).
        for &g in &[0.0, 0.25, 0.45] {
            for &a in &ALPHAS {
                let two = lookup_critical_value(Detector::PageCusum, g, a).unwrap();
                let one = lookup_critical_value(Detector::PageCusum1, g, a).unwrap();
                assert!(
                    two > one,
                    "PageCusum({})={} should exceed PageCusum1({})={}",
                    g,
                    two,
                    g,
                    one
                );
            }
        }
    }

    /// Regenerate the full critical-value table and print it in the exact
    /// source format expected by `CRIT_VAL_TABLE`. Run this whenever the
    /// simulator or grid changes, then paste the output into this file.
    ///
    /// ```bash
    /// cargo test --features parallel --release \
    ///   -- monitor::sequential_table::tests::regenerate_crit_val_table \
    ///      --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "regeneration utility: prints table entries to stdout, run manually"]
    fn regenerate_crit_val_table() {
        const SAMPLES: usize = 10_000;
        const NPTS: usize = 500;

        let detectors = [
            (Detector::Cusum, "Detector::Cusum"),
            (Detector::Cusum1, "Detector::Cusum1"),
            (Detector::PageCusum, "Detector::PageCusum"),
            (Detector::PageCusum1, "Detector::PageCusum1"),
        ];

        println!();
        println!("// === paste from here into CRIT_VAL_TABLE ===");
        for (det, det_name) in detectors {
            println!("    // {}", det_name);
            for &gamma in GAMMAS.iter() {
                for &alpha in ALPHAS.iter() {
                    let cv = simulate_critical_value(det, gamma, alpha, SAMPLES, NPTS, Some(42));
                    println!(
                        "    CritValEntry {{ detector: {}, gamma: {:.3}, alpha: {:.2}, value: {:.6} }},",
                        det_name, gamma, alpha, cv
                    );
                }
            }
        }
        println!("// === paste up to here ===");
    }
}
