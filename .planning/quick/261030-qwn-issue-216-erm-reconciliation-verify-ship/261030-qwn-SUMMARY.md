---
quick_id: 261030-qwn
issue: 216
branch: feat/216-erm-reconciliation
status: complete
commits: [80fff4c]
---

# Summary: #216 ERM reconciliation

## Outcome
ERM was **already implemented on `origin/main`** (phases 07/08: `d40e294`, `d687521`, `8ac663c`,
`c4b9aa8`, `bc79526`, `0785c2c`), though the GitHub issue is still open:
- `ReconciliationMethod::Erm { lambda: Option<f64> }` -- `Some(l)`: fixed ridge penalty
  (finite, >= 0); `None`: Ledoit-Wolf-style auto lambda (the MinTraceShrink-style ergonomics the
  issue asks for).
- `HierarchyTree::with_erm_training(base_history, leaf_history)` -- option 1 of the issue's API
  sketch (all-node base-forecast history + leaf actuals, T periods).
- Solve `P = B Y' (Y Y' + lambda I)^-1` (node x T layout; identical to the paper's
  `B'Y(Y'Y + lambda I)^-1` in T x n layout), reconciled = `S P y_hat`; works for trees and for
  grouped / crossed hierarchies built with `from_summing_matrix`.
- Existing unit tests in `src/hierarchy/mod.rs`: reference-formula correctness, hand-checkable
  asymmetric hierarchy, multi-horizon coherence, missing-history / shape / negative / NaN lambda /
  empty-T guards, auto-lambda (incl. T=1 and constant-forecast cases), grouped/crossed
  end-to-end RMSSE vs MinTraceStruct (ERM 0.92 vs MinTraceStruct 2.11 vs unreconciled 2.29).

No re-implementation was done.

## Added: biased-base-forecast benchmark (`tests/hierarchy_erm_bias.rs`)
The existing comparison uses unbiased base-forecast noise, so it does not exercise the property
the issue cites as ERM's reason to exist (dropping MinT's unbiasedness condition helps when base
forecasts are biased). New test: 7-node tree (Total -> A, B -> 4 leaves), seasonal leaves,
T=60 training periods, H=12 holdout; base forecasts biased (leaves x1.25, aggregates x0.90) plus
noise. Mean over nodes of holdout RMSE / node level:

| method | mean relative RMSE |
|---|--:|
| base (incoherent) | 0.1868 |
| BottomUp | 0.2505 |
| MinTraceOls | 0.0798 |
| MinTraceShrink | 0.0792 |
| MinTraceStruct | 0.0553 |
| **Erm (auto lambda)** | **0.0195** |
| **Erm (lambda = 1.0)** | **0.0195** |

ERM is 2.8x better than the best MinT variant here; every method's output is verified coherent
(parent = sum of children to 1e-8 relative at every horizon). The test asserts ERM < 0.5 x best
non-ERM. It is a capability measurement, not a bug regression test (it passes on `origin/main`).

## Compatibility note
`ReconciliationMethod` is **not** `#[non_exhaustive]`. Adding `Erm` (already released) was a
breaking change for downstream code that matches exhaustively on the enum. Marking the enum
`#[non_exhaustive]` now would itself be breaking (forces a wildcard arm), so it is left as is;
worth doing at the next semver-major bump so future methods can be added compatibly.

## Deviation
The PLAN was written after the gap analysis (finding that ERM already shipped), together with
the new test, not before any code was drafted.

## Gates
- `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `cargo test --all-features --lib hierarchy`: 45 passed.
- Full `cargo test --all-features --no-fail-fast`: 3767 passed, 0 failed (3766 on origin/main + 1 new).
