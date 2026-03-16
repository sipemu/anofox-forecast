# Hierarchical Forecasting with Reconciliation

**Run:** `cargo run --example hierarchy`

## What this example demonstrates

Independently generated forecasts across hierarchical groups (e.g., Total -> Region -> Store) rarely add up. This example builds a 3-level hierarchy and applies four reconciliation methods -- Bottom-Up, Top-Down, Middle-Out, and MinTrace OLS -- to produce coherent forecasts that respect the tree structure. Each method is verified with a coherence check and compared in a summary table.

## Sections

1. **Build the hierarchy** -- Constructs a 3-level `HierarchyTree` (Total -> North/South -> N1/N2/S1/S2) and inspects node names and children.
2. **Base forecasts** -- Defines incoherent base forecasts for all 7 nodes over a 3-step horizon and prints a coherence check showing they do not add up.
3. **Bottom-Up reconciliation** -- Trusts leaf-level forecasts and aggregates upward to parent nodes.
4. **Top-Down reconciliation** -- Trusts the top-level forecast, supplies historical actuals, and disaggregates downward using historical proportions.
5. **Middle-Out reconciliation** -- Trusts the middle level (North, South) and combines upward aggregation with downward disaggregation.
6. **MinTrace OLS reconciliation** -- Optimally adjusts all levels to minimise the trace of the reconciled forecast error covariance.
7. **Method comparison** -- Prints a side-by-side table of h=1 forecasts from all methods.
8. **When to use each method** -- Summarises guidance on choosing among the four approaches.

## Key types

- `HierarchyTree` -- tree structure defining parent-child relationships
- `ReconciliationMethod` -- enum with `BottomUp`, `TopDown`, `MiddleOut`, and `MinTraceOls` variants
- `HierarchyTree::reconcile()` -- produces coherent forecasts from incoherent base forecasts
- `HierarchyTree::set_actuals()` -- supplies historical data needed by Top-Down and Middle-Out
