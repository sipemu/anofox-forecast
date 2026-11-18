//! Issue #246: AutoETS model choice must not depend on the units of the data.
//!
//! AICc on a Gaussian likelihood is only scale-invariant when every candidate
//! is scored on the same observations. Multiplying a series by a constant `c`
//! shifts every candidate's log-likelihood by the same `-n ln c`, so the
//! ranking (and hence the selected model) must be unchanged.
//!
//! The panel below mimics short monthly demand series (40-46 observations,
//! period 12): positive values, mixed seasonal strength, noise and trend.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, ModelPool};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// Small deterministic PRNG (xorshift64*) so the panel is reproducible
/// without pulling in a seeded RNG dependency.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn make_panel() -> Vec<Vec<f64>> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut panel = Vec::new();
    for i in 0..60 {
        let n = 40 + (i % 7);
        let level = 50.0 + 400.0 * rng.next_f64();
        let season_amp = [0.0, 0.05, 0.15, 0.4][i % 4];
        let noise = [0.05, 0.15, 0.3][i % 3];
        let slope = if i % 5 == 0 { 0.01 } else { 0.0 };
        let phase = rng.next_f64() * 12.0;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let s = (2.0 * std::f64::consts::PI * (t as f64 + phase) / 12.0).sin();
                let mean = level * (1.0 + slope * t as f64) * (1.0 + season_amp * s);
                (mean * (1.0 + noise * rng.normal())).max(1.0)
            })
            .collect();
        panel.push(values);
    }
    panel
}

fn make_series(values: &[f64]) -> TimeSeries {
    let start = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let ts = (0..values.len())
        .map(|i| start + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(ts, values.to_vec()).unwrap()
}

fn selected(config: &AutoETSConfig, values: &[f64]) -> String {
    let mut model = AutoETS::with_config(config.clone());
    model.fit(&make_series(values)).unwrap();
    model.selected_spec().unwrap().short_name()
}

fn count_switches(config: &AutoETSConfig, scale: f64) -> (usize, Vec<String>) {
    let mut switches = Vec::new();
    let panel = make_panel();
    for (i, values) in panel.iter().enumerate() {
        let scaled: Vec<f64> = values.iter().map(|v| v * scale).collect();
        let a = selected(config, values);
        let b = selected(config, &scaled);
        if a != b {
            switches.push(format!("series {i}: {a} -> {b}"));
        }
    }
    (panel.len(), switches)
}

fn assert_scale_invariant(config: AutoETSConfig) {
    for scale in [1000.0, 1.0 / 1000.0] {
        let (n, switches) = count_switches(&config, scale);
        assert!(
            switches.is_empty(),
            "AutoETS model choice changed for {}/{} series when scaling by {scale}:\n{}",
            switches.len(),
            n,
            switches.join("\n")
        );
    }
}

#[test]
fn autoets_choice_is_scale_invariant_complete_pool() {
    assert_scale_invariant(AutoETSConfig::with_period(12));
}

#[test]
fn autoets_choice_is_scale_invariant_damped_pool() {
    assert_scale_invariant(
        AutoETSConfig::with_period(12).with_model_pool(ModelPool::DampedTrendOnly),
    );
}
