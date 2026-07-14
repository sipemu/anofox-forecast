//! Per-leaf init pathology sweep.
//!
//! For each leaf type instantiable via public API in .skaters() pool,
//! feed billion-scale training data one observation at a time and log
//! (name, obs_idx, mean, std). Flag transient divergence (|mean| or std
//! more than 1e4 × y_scale) at any early observation.
//!
//! Motivated by the GARCH init fix (v0.15.7): a one-line bug caused
//! 60-million-times mixture-σ inflation. Similar init pathologies may
//! exist in other leaves; this sweep catches them systematically.
//!
//! Run: `cargo run --release --features distributional --example leaf_init_pathology_sweep`

use anofox_forecast::models::laplace::leaves::{
    Ar1Leaf, Ar2Leaf, DriftLeaf, EmaLeaf, FractionalDiffLeaf, GarchWrappedLeaf, HoltLeaf, OuLeaf,
    PowerTransformWrapper, SeasonalDifferenceWrapper, SeasonalEmaLeaf, SlowStandardizeWrapper,
    StandardizeWrapper, ThetaLeaf, YjWrappedLeaf,
};
use anofox_forecast::models::laplace::Leaf;
use std::fs;

fn load_cif_2016_54() -> Vec<f64> {
    let path = "validation/data/cif_2016.tsf";
    let bytes = fs::read(path).unwrap_or_default();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        if !in_data {
            if line.trim_start().starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let toks: Vec<&str> = line.split(':').collect();
        if toks.len() < 2 {
            continue;
        }
        let vals_str = toks[toks.len() - 1];
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if values.len() >= 20 {
            series.push(values);
        }
    }
    series[54].clone()
}

fn sweep_leaf<L: Leaf>(name: &str, mut leaf: L, values: &[f64], y_scale: f64) {
    let mut worst_mean: (usize, f64) = (0, 0.0);
    let mut worst_std: (usize, f64) = (0, 0.0);
    for (i, &y) in values.iter().enumerate() {
        // First check predict_one BEFORE observe (state after previous obs).
        let g = leaf.predict_one();
        if g.mean.abs() > worst_mean.1 {
            worst_mean = (i, g.mean.abs());
        }
        if g.std > worst_std.1 {
            worst_std = (i, g.std);
        }
        leaf.observe(y);
    }
    // Final predict after all obs.
    let g_final = leaf.predict_one();
    let flag_mean = if worst_mean.1 > 1e4 * y_scale {
        format!(" ⚠ mean at obs {} was {:.2e}", worst_mean.0, worst_mean.1)
    } else {
        String::new()
    };
    let flag_std = if worst_std.1 > 1e4 * y_scale {
        format!(" ⚠ std at obs {} was {:.2e}", worst_std.0, worst_std.1)
    } else {
        String::new()
    };
    println!(
        "{name:<40} final: μ={:.3e} σ={:.3e}    peaks: |μ|={:.3e}@{} σ={:.3e}@{}{flag_mean}{flag_std}",
        g_final.mean,
        g_final.std,
        worst_mean.1,
        worst_mean.0,
        worst_std.1,
        worst_std.0,
    );
}

fn main() {
    let values = load_cif_2016_54();
    let y_scale: f64 = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    println!(
        "Sweep on cif_2016[54]: N={}, y_scale={:.3e}\n",
        values.len(),
        y_scale
    );
    println!(
        "Flag threshold: |mean| or std > 1e4 × y_scale = {:.3e}\n",
        1e4 * y_scale
    );

    // Simple leaves.
    sweep_leaf("EmaLeaf(0.05)", EmaLeaf::new(0.05), &values, y_scale);
    sweep_leaf("EmaLeaf(0.1)", EmaLeaf::new(0.1), &values, y_scale);
    sweep_leaf("EmaLeaf(0.3)", EmaLeaf::new(0.3), &values, y_scale);
    sweep_leaf("DriftLeaf(0.1)", DriftLeaf::new(0.1), &values, y_scale);
    sweep_leaf("Ar1Leaf(0.1)", Ar1Leaf::new(0.1), &values, y_scale);
    sweep_leaf("Ar2Leaf(0.1)", Ar2Leaf::new(0.1), &values, y_scale);
    sweep_leaf(
        "HoltLeaf(0.1,0.02,1.0)",
        HoltLeaf::new(0.1, 0.02, 1.0),
        &values,
        y_scale,
    );
    sweep_leaf("ThetaLeaf(0.1)", ThetaLeaf::new(0.1), &values, y_scale);
    sweep_leaf("OuLeaf(0.1)", OuLeaf::new(0.1), &values, y_scale);
    sweep_leaf(
        "FractionalDiffLeaf(0.4,0.1,0.1)",
        FractionalDiffLeaf::new(0.4, 0.1, 0.1),
        &values,
        y_scale,
    );
    sweep_leaf(
        "SeasonalEmaLeaf(12,0.15)",
        SeasonalEmaLeaf::new(12, 0.15),
        &values,
        y_scale,
    );

    // Wrapped leaves — most likely culprits (previous session found GARCH).
    sweep_leaf(
        "PowerTransform(0.5) @ Ema(0.1)",
        PowerTransformWrapper::new(Box::new(EmaLeaf::new(0.1)), 0.5),
        &values,
        y_scale,
    );
    sweep_leaf(
        "Standardize @ Ema(0.1)",
        StandardizeWrapper::new(Box::new(EmaLeaf::new(0.1)), 0.05),
        &values,
        y_scale,
    );
    sweep_leaf(
        "SlowStandardize @ Ema(0.3)",
        SlowStandardizeWrapper::new(Box::new(EmaLeaf::new(0.3)), 0.02),
        &values,
        y_scale,
    );
    sweep_leaf(
        "SeasonalDiff(1) @ Ema(0.1)",
        SeasonalDifferenceWrapper::new(Box::new(EmaLeaf::new(0.1)), 1),
        &values,
        y_scale,
    );
    sweep_leaf(
        "SeasonalDiff(12) @ Ema(0.1)",
        SeasonalDifferenceWrapper::new(Box::new(EmaLeaf::new(0.1)), 12),
        &values,
        y_scale,
    );
    sweep_leaf(
        "YJ(0.0) @ Ema(0.1)",
        YjWrappedLeaf::new(Box::new(EmaLeaf::new(0.1)), 0.0),
        &values,
        y_scale,
    );
    sweep_leaf(
        "YJ(0.5) @ Ema(0.1)",
        YjWrappedLeaf::new(Box::new(EmaLeaf::new(0.1)), 0.5),
        &values,
        y_scale,
    );
    sweep_leaf(
        "Garch(1e-6,0.05,0.9) @ Ema(0.1) [FIXED]",
        GarchWrappedLeaf::new(Box::new(EmaLeaf::new(0.1)), 1e-6, 0.05, 0.9),
        &values,
        y_scale,
    );
}
