//! Hierarchical Forecasting with Reconciliation example.
//!
//! Run with: cargo run --example hierarchy

use anofox_forecast::hierarchy::{HierarchyTree, ReconciliationMethod};
use std::collections::HashMap;

fn main() {
    println!("=== Hierarchical Forecasting Example ===\n");

    println!("When forecasting across hierarchical groups (e.g., Total -> Region -> Store),");
    println!("independently generated forecasts rarely add up. Reconciliation methods");
    println!("produce coherent forecasts that respect the hierarchical structure.\n");

    // -----------------------------------------------------------------------
    // Build a 3-level hierarchy:
    //
    //            Total
    //           /     \
    //        North    South
    //        / \       / \
    //      N1  N2    S1  S2
    // -----------------------------------------------------------------------
    println!("--- Building the Hierarchy ---");

    let tree = HierarchyTree::new(vec![
        ("Total", &["North", "South"]),
        ("North", &["N1", "N2"]),
        ("South", &["S1", "S2"]),
    ])
    .unwrap();

    println!("Nodes (BFS order): {:?}", tree.node_names());
    println!("Node count: {}", tree.len());
    println!(
        "Children of Total: {:?}",
        tree.children_of("Total").unwrap()
    );
    println!(
        "Children of North: {:?}",
        tree.children_of("North").unwrap()
    );
    println!(
        "Children of South: {:?}",
        tree.children_of("South").unwrap()
    );

    // -----------------------------------------------------------------------
    // Base forecasts (incoherent -- they don't add up)
    // -----------------------------------------------------------------------
    let horizon = 3;
    let base_forecasts: Vec<(String, Vec<f64>)> = vec![
        ("Total".into(), vec![500.0, 520.0, 540.0]),
        ("North".into(), vec![280.0, 290.0, 300.0]),
        ("South".into(), vec![240.0, 250.0, 260.0]),
        ("N1".into(), vec![150.0, 155.0, 160.0]),
        ("N2".into(), vec![120.0, 125.0, 130.0]),
        ("S1".into(), vec![100.0, 105.0, 110.0]),
        ("S2".into(), vec![130.0, 135.0, 140.0]),
    ];

    println!("\n--- Base Forecasts (Incoherent) ---");
    println!("These forecasts were generated independently and do NOT add up.\n");
    print_forecasts(&base_forecasts, horizon);
    println!();
    print_coherence_check(&base_forecasts, horizon);

    // -----------------------------------------------------------------------
    // 1. Bottom-Up Reconciliation
    // -----------------------------------------------------------------------
    println!("\n--- 1. Bottom-Up Reconciliation ---");
    println!("Trusts the bottom-level (leaf) forecasts and aggregates upward.");
    println!("Upper-level base forecasts are replaced by the sum of their children.\n");

    let bottom_up = tree
        .reconcile(&base_forecasts, ReconciliationMethod::BottomUp)
        .unwrap();

    print_forecasts(&bottom_up, horizon);
    println!();
    print_coherence_check(&bottom_up, horizon);

    // -----------------------------------------------------------------------
    // 2. Top-Down Reconciliation
    // -----------------------------------------------------------------------
    println!("\n--- 2. Top-Down Reconciliation ---");
    println!("Trusts the top-level forecast and disaggregates downward using");
    println!("historical proportions from actual data.\n");

    // Supply historical actuals so the tree can compute proportions.
    let mut tree_td = tree.clone();
    let mut actuals = HashMap::new();
    actuals.insert("Total".into(), vec![480.0, 490.0, 500.0, 510.0]);
    actuals.insert("North".into(), vec![260.0, 270.0, 275.0, 280.0]);
    actuals.insert("South".into(), vec![220.0, 220.0, 225.0, 230.0]);
    actuals.insert("N1".into(), vec![140.0, 145.0, 148.0, 150.0]);
    actuals.insert("N2".into(), vec![120.0, 125.0, 127.0, 130.0]);
    actuals.insert("S1".into(), vec![90.0, 92.0, 95.0, 100.0]);
    actuals.insert("S2".into(), vec![130.0, 128.0, 130.0, 130.0]);
    tree_td.set_actuals(actuals.clone());

    let top_down = tree_td
        .reconcile(&base_forecasts, ReconciliationMethod::TopDown)
        .unwrap();

    print_forecasts(&top_down, horizon);
    println!();
    print_coherence_check(&top_down, horizon);

    // -----------------------------------------------------------------------
    // 3. Middle-Out Reconciliation
    // -----------------------------------------------------------------------
    println!("\n--- 3. Middle-Out Reconciliation ---");
    println!("Trusts forecasts at a chosen middle level (here level 1: North, South).");
    println!("Aggregates upward to the root and disaggregates downward to leaves.\n");

    let middle_out = tree_td
        .reconcile(
            &base_forecasts,
            ReconciliationMethod::MiddleOut { middle_level: 1 },
        )
        .unwrap();

    print_forecasts(&middle_out, horizon);
    println!();
    print_coherence_check(&middle_out, horizon);

    // -----------------------------------------------------------------------
    // 4. MinTrace OLS Reconciliation
    // -----------------------------------------------------------------------
    println!("\n--- 4. MinTrace OLS Reconciliation ---");
    println!("Optimal combination that minimises the trace of the reconciled");
    println!("forecast error covariance using OLS estimation.");
    println!("Adjusts ALL levels to produce the best coherent set.\n");

    let mintrace_ols = tree
        .reconcile(&base_forecasts, ReconciliationMethod::MinTraceOls)
        .unwrap();

    print_forecasts(&mintrace_ols, horizon);
    println!();
    print_coherence_check(&mintrace_ols, horizon);

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------
    println!("\n--- Method Comparison (h=1 forecasts) ---");
    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Method", "Total", "North", "South", "N1", "N2", "S1", "S2"
    );
    println!("{:-<79}", "");

    #[allow(clippy::type_complexity)]
    let methods: Vec<(&str, &Vec<(String, Vec<f64>)>)> = vec![
        ("Base", &base_forecasts),
        ("BottomUp", &bottom_up),
        ("TopDown", &top_down),
        ("MiddleOut(1)", &middle_out),
        ("MinTraceOls", &mintrace_ols),
    ];

    let node_order = ["Total", "North", "South", "N1", "N2", "S1", "S2"];

    for (label, forecasts) in &methods {
        let lookup: HashMap<&str, &Vec<f64>> = forecasts
            .iter()
            .map(|(name, vals)| (name.as_str(), vals))
            .collect();

        print!("{:<15}", label);
        for node in &node_order {
            print!(" {:>8.1}", lookup[node][0]);
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Best practices
    // -----------------------------------------------------------------------
    println!("\n--- When to Use Each Method ---");
    println!(
        "
1. BottomUp:
   - When bottom-level forecasts are most reliable
   - Simple, no extra data needed
   - Top-level forecasts are fully determined by leaves

2. TopDown:
   - When the aggregate (top-level) forecast is most accurate
   - Requires historical actuals to compute proportions
   - Leaf forecasts are fully determined by the root

3. MiddleOut:
   - When a middle level (e.g., regions) has the best forecasts
   - Combines upward aggregation with downward disaggregation
   - Requires historical actuals for downward proportions

4. MinTraceOls:
   - Optimal statistical combination of all levels
   - Adjusts every node; does not fully trust any single level
   - Best accuracy when base forecasts have comparable quality
"
    );

    println!("=== Hierarchical Forecasting Example Complete ===");
}

/// Pretty-print reconciled forecasts as a table.
fn print_forecasts(forecasts: &[(String, Vec<f64>)], horizon: usize) {
    let mut header = format!("{:<10}", "Node");
    for h in 1..=horizon {
        header.push_str(&format!(" {:>10}", format!("h={}", h)));
    }
    println!("{}", header);
    println!("{:-<width$}", "", width = 10 + horizon * 11);

    for (name, vals) in forecasts {
        let mut row = format!("{:<10}", name);
        for v in vals {
            row.push_str(&format!(" {:>10.2}", v));
        }
        println!("{}", row);
    }
}

/// Check and display whether the hierarchy adds up at each horizon step.
fn print_coherence_check(forecasts: &[(String, Vec<f64>)], horizon: usize) {
    let lookup: HashMap<&str, &Vec<f64>> = forecasts
        .iter()
        .map(|(name, vals)| (name.as_str(), vals))
        .collect();

    let mut coherent = true;
    #[allow(clippy::needless_range_loop)]
    for h in 0..horizon {
        let north_sum = lookup["N1"][h] + lookup["N2"][h];
        let south_sum = lookup["S1"][h] + lookup["S2"][h];
        let total_sum = lookup["North"][h] + lookup["South"][h];

        let north_ok = (lookup["North"][h] - north_sum).abs() < 1e-6;
        let south_ok = (lookup["South"][h] - south_sum).abs() < 1e-6;
        let total_ok = (lookup["Total"][h] - total_sum).abs() < 1e-6;

        if !north_ok || !south_ok || !total_ok {
            coherent = false;
        }
    }

    if coherent {
        println!("Coherence check: PASSED (all levels add up)");
    } else {
        println!("Coherence check: FAILED (levels do not add up)");
    }
}
