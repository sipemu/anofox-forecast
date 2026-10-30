---
quick_id: 261030-qwn
issue: 216
branch: feat/216-erm-reconciliation
---

# Plan: #216 ERM reconciliation

## What
Issue #216 asks for an ERM (Ben Taieb & Koo 2019) variant of
`hierarchy::ReconciliationMethod`: learn `P = B'Y(Y'Y + lambda I)^-1` by ridge regression of
leaf actuals on base forecasts over a training window, reconciled = `S P y_hat`, with a training
history setter and a default / auto lambda.

## Finding before coding
`origin/main` already ships it (phases 07/08, commits d40e294, d687521, 8ac663c, c4b9aa8,
bc79526, 0785c2c; released before v0.18.0): `ReconciliationMethod::Erm { lambda: Option<f64> }`
(None = Ledoit-Wolf-style auto lambda), `HierarchyTree::with_erm_training(base, leaf)`, the ridge
solve in `erm_reconcile`, and unit tests for the reference formula, a hand-checkable asymmetric
hierarchy, multi-horizon coherence, error guards, auto-lambda, and a grouped/crossed end-to-end
RMSSE comparison against MinTraceStruct. The GitHub issue is still open.

## How
Do not re-implement. Close the one gap against the issue's motivation: ERM's stated advantage
is dropping MinT's unbiasedness condition, which helps when base forecasts are **biased**; the
existing comparison test uses unbiased noise. Add `tests/hierarchy_erm_bias.rs`: 7-node tree,
leaves over-forecast 25%, aggregates under-forecast 10%; compare BottomUp / MinTraceOls /
MinTraceStruct / MinTraceShrink / Erm(auto) / Erm(1.0) on a holdout, assert coherence for every
method and that ERM at least halves the best non-ERM error.

## Verification
- New test passes and prints the comparison; existing hierarchy tests pass.
- fmt, clippy -D warnings, full `cargo test --all-features`.
