//! Issue #216: ERM reconciliation on biased base forecasts.
//!
//! ERM (Ben Taieb & Koo 2019) learns the bottom-level projection `P` by ridge
//! regression of leaf actuals on base forecasts, dropping MinT's
//! unbiasedness constraint (`S P S = S`). That constraint makes MinT /
//! BottomUp pass a systematic bias in the base forecasts straight through to
//! the reconciled forecasts; ERM can learn to undo it. This test measures
//! that on a synthetic 2-level hierarchy whose base forecasts are biased
//! (leaves over-forecast by 25%, aggregates under-forecast by 10%) and
//! checks that every reconciled output is coherent.

use anofox_forecast::hierarchy::{HierarchyTree, ReconciliationMethod};
use std::collections::HashMap;

const T_TRAIN: usize = 60;
const H: usize = 12;
const NODES: [&str; 7] = ["Total", "A", "B", "A1", "A2", "B1", "B2"];
const LEAVES: [&str; 4] = ["A1", "A2", "B1", "B2"];

fn tree() -> HierarchyTree {
    HierarchyTree::new(vec![
        ("Total", &["A", "B"][..]),
        ("A", &["A1", "A2"][..]),
        ("B", &["B1", "B2"][..]),
    ])
    .unwrap()
}

/// Deterministic uniform noise in [-0.5, 0.5].
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
}

/// Coherent actuals for all nodes (`[node][t]`, NODES order) and biased,
/// noisy base forecasts of them.
fn data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_t = T_TRAIN + H;
    let mut rng = Lcg(216);
    let levels = [40.0, 25.0, 30.0, 15.0];
    let leaves: Vec<Vec<f64>> = levels
        .iter()
        .enumerate()
        .map(|(j, &lvl)| {
            (0..n_t)
                .map(|t| {
                    let season =
                        0.2 * lvl * (2.0 * std::f64::consts::PI * (t + 3 * j) as f64 / 12.0).sin();
                    lvl + season + 2.0 * rng.next()
                })
                .collect()
        })
        .collect();
    let sum = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>();
    let a = sum(&leaves[0], &leaves[1]);
    let b = sum(&leaves[2], &leaves[3]);
    let total = sum(&a, &b);
    let actual = vec![
        total,
        a,
        b,
        leaves[0].clone(),
        leaves[1].clone(),
        leaves[2].clone(),
        leaves[3].clone(),
    ];

    let base = actual
        .iter()
        .enumerate()
        .map(|(i, series)| {
            let bias = if i >= 3 { 1.25 } else { 0.90 };
            series.iter().map(|y| bias * y + 3.0 * rng.next()).collect()
        })
        .collect();
    (actual, base)
}

fn rmse(a: &[f64], f: &[f64]) -> f64 {
    (a.iter().zip(f).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Mean over nodes of the holdout RMSE scaled by the node's mean level.
fn score(actual: &[Vec<f64>], fc: &HashMap<String, Vec<f64>>) -> f64 {
    NODES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let holdout = &actual[i][T_TRAIN..];
            let level = holdout.iter().sum::<f64>() / H as f64;
            rmse(holdout, &fc[*name]) / level
        })
        .sum::<f64>()
        / NODES.len() as f64
}

fn assert_coherent(fc: &HashMap<String, Vec<f64>>, label: &str) {
    for (parent, [k0, k1]) in [
        ("Total", ["A", "B"]),
        ("A", ["A1", "A2"]),
        ("B", ["B1", "B2"]),
    ] {
        assert_eq!(fc[parent].len(), H, "{label}: {parent} horizon");
        for (t, (p, (a, b))) in fc[parent]
            .iter()
            .zip(fc[k0].iter().zip(&fc[k1]))
            .enumerate()
        {
            let s = a + b;
            assert!(
                (p - s).abs() <= 1e-8 * s.abs().max(1.0),
                "{label}: {parent}[{t}] = {p} but children sum to {s}"
            );
        }
    }
}

#[test]
fn erm_removes_systematic_base_forecast_bias_that_mint_keeps() {
    let (actual, base) = data();
    let holdout_base: Vec<(String, Vec<f64>)> = NODES
        .iter()
        .enumerate()
        .map(|(i, n)| (n.to_string(), base[i][T_TRAIN..].to_vec()))
        .collect();

    let mut tree = tree();
    // MinTraceShrink uses in-sample residuals (actual - base) of the window.
    tree.set_residuals(
        NODES
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let r = (0..T_TRAIN).map(|t| actual[i][t] - base[i][t]).collect();
                (n.to_string(), r)
            })
            .collect(),
    );
    tree.with_erm_training(
        NODES
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), base[i][..T_TRAIN].to_vec()))
            .collect(),
        LEAVES
            .iter()
            .map(|n| {
                let i = NODES.iter().position(|x| x == n).unwrap();
                (n.to_string(), actual[i][..T_TRAIN].to_vec())
            })
            .collect(),
    );

    let base_map: HashMap<String, Vec<f64>> = holdout_base.iter().cloned().collect();
    let base_score = score(&actual, &base_map);
    let mut scores = Vec::new();
    for (label, method) in [
        ("BottomUp", ReconciliationMethod::BottomUp),
        ("MinTraceOls", ReconciliationMethod::MinTraceOls),
        ("MinTraceStruct", ReconciliationMethod::MinTraceStruct),
        ("MinTraceShrink", ReconciliationMethod::MinTraceShrink),
        ("Erm(auto)", ReconciliationMethod::Erm { lambda: None }),
        ("Erm(1.0)", ReconciliationMethod::Erm { lambda: Some(1.0) }),
    ] {
        let fc: HashMap<String, Vec<f64>> = tree
            .reconcile(&holdout_base, method)
            .unwrap()
            .into_iter()
            .collect();
        assert_coherent(&fc, label);
        let s = score(&actual, &fc);
        eprintln!("{label:>15}: mean relative RMSE {s:.4}");
        scores.push((label, s));
    }
    eprintln!(
        "{:>15}: mean relative RMSE {base_score:.4}",
        "base (incoherent)"
    );

    let best_mint = scores
        .iter()
        .filter(|(l, _)| !l.starts_with("Erm"))
        .map(|&(_, s)| s)
        .fold(f64::INFINITY, f64::min);
    for &(label, s) in scores.iter().filter(|(l, _)| l.starts_with("Erm")) {
        // ERM must at least halve the error of the best unbiased method.
        assert!(
            s < 0.5 * best_mint,
            "{label} ({s:.4}) does not beat the best BottomUp/MinT method ({best_mint:.4}) by 2x"
        );
    }
}
