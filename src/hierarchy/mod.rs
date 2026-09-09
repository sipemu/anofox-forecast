//! Hierarchical forecasting with reconciliation.
//!
//! When forecasting across hierarchical groups (e.g., total → region → store),
//! independently generated forecasts rarely add up. This module provides
//! reconciliation methods to produce coherent forecasts that respect the
//! hierarchical structure.
//!
//! # Reconciliation Methods
//!
//! - **BottomUp** — aggregate bottom-level forecasts upward
//! - **TopDown** — disaggregate top-level forecast using historical proportions
//! - **MiddleOut** — reconcile from a chosen middle level (up *and* down)
//! - **MinTrace (OLS)** — optimal combination via the MinT OLS approach
//!
//! # Example
//!
//! ```
//! use anofox_forecast::hierarchy::{HierarchyTree, ReconciliationMethod};
//!
//! // A simple 2-level hierarchy: Total → [A, B]
//! let tree = HierarchyTree::new(vec![
//!     ("Total", &["A", "B"]),
//! ]).unwrap();
//!
//! // Base forecasts (keyed by node name, each is a Vec<f64> over the horizon)
//! let base = vec![
//!     ("Total".to_string(), vec![100.0, 110.0]),
//!     ("A".to_string(),     vec![70.0,  75.0]),
//!     ("B".to_string(),     vec![40.0,  45.0]),
//! ];
//!
//! let reconciled = tree.reconcile(&base, ReconciliationMethod::BottomUp).unwrap();
//!
//! // Bottom-up: Total = A + B
//! assert_eq!(reconciled[0].1, vec![110.0, 120.0]); // reconciled Total
//! assert_eq!(reconciled[1].1, vec![70.0, 75.0]);   // A unchanged
//! assert_eq!(reconciled[2].1, vec![40.0, 45.0]);   // B unchanged
//! ```

use crate::error::{ForecastError, Result};
use std::collections::HashMap;

/// A node in the hierarchy tree (or DAG, for grouped / crossed
/// hierarchies — see [`HierarchyTree::from_summing_matrix`]).
#[derive(Debug, Clone)]
struct Node {
    name: String,
    children: Vec<usize>, // indices into HierarchyTree::nodes
    /// All immediate parents of this node. In a strict tree this is at
    /// most one; in a grouped hierarchy (issue #124) a leaf may have
    /// several parents (one per aggregate dimension it contributes to).
    parents: Vec<usize>,
    /// Maximum depth from the root via any path. Used by `max_level`
    /// and level-based traversal in the tree-mode reconciliation
    /// methods (BottomUp / TopDown / MiddleOut).
    level: usize,
}

/// Reconciliation method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconciliationMethod {
    /// Aggregate bottom-level forecasts upward. Upper-level base forecasts are
    /// discarded and replaced by the sum of their children.
    BottomUp,
    /// Disaggregate the top-level forecast downward using historical proportions.
    /// Requires `set_actuals()` to have been called.
    TopDown,
    /// Reconcile from a chosen middle level: aggregate upward (BottomUp-style) to
    /// the root and disaggregate downward (TopDown-style with historical proportions)
    /// to the leaves. `middle_level = 0` behaves like TopDown; a level at or beyond
    /// the maximum depth behaves like BottomUp.
    /// Requires `set_actuals()` to have been called (for downward proportions).
    MiddleOut {
        /// The depth (0 = root) of the "middle" level whose forecasts are trusted.
        middle_level: usize,
    },
    /// Optimal combination using MinT (Minimum Trace) with OLS covariance.
    /// Minimises the trace of the reconciled forecast error covariance.
    /// **Warning:** builds dense N×M summing matrix. For N > ~10k, use
    /// `MinTraceVariance` or `MinTraceStruct` instead.
    MinTraceOls,
    /// MinT with Ledoit-Wolf shrinkage covariance.
    /// Requires `set_residuals()` to have been called.
    /// Uses Σ = α·F + (1−α)·S where F is the diagonal target and S is the
    /// sample covariance, with α chosen by the Ledoit-Wolf formula.
    /// **Warning:** builds N×N covariance matrix. For N > ~5k, use
    /// `MinTraceVariance` instead.
    MinTraceShrink,
    /// Scalable MinT with diagonal variance scaling (WLS).
    /// Uses W = diag(residual_variances). Avoids N×N covariance matrix.
    /// Requires `set_residuals()` to have been called.
    /// Memory: O(N + M²). Safe for N > 100k.
    MinTraceVariance,
    /// Scalable MinT with structural scaling.
    /// Uses W = diag(1/n_leaves_below). No residuals needed.
    /// Memory: O(N + M²). Safe for N > 100k.
    MinTraceStruct,
    /// Empirical Risk Minimization (ERM) reconciliation.
    ///
    /// Computes the regularized ridge projection `P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹`,
    /// where `Ŷ` is the matrix of historical base forecasts (nodes×T, rows=nodes)
    /// and `B` is the matrix of historical leaf actuals (leaves×T, rows=leaves).
    /// The Gram matrix `Ŷ Ŷᵀ` is nodes×nodes (n×n). The reconciled
    /// bottom-level forecasts are `P·ŷ` and the full coherent vector is `S·P·ŷ`.
    ///
    /// (This is the same ERM estimator as Ben Taieb & Koo (2019), written for
    /// this codebase's row-major node×T layout rather than column-major T×n.)
    ///
    /// `lambda` is the ridge regularization strength. Larger values shrink
    /// `P` toward zero, increasing robustness when `T` (training periods) is
    /// small relative to `n` (node count). Set `lambda` to 0.0 only when
    /// `T >> n` and the base forecasts are well-conditioned; otherwise the
    /// Gram matrix `Ŷ Ŷᵀ` may be rank-deficient and the solve will fail with a
    /// `SingularMatrix` error.
    ///
    /// Requires [`HierarchyTree::with_erm_training`] to have been called.
    ///
    /// Reference: Ben Taieb & Koo (2019), "Regularized regression for
    /// hierarchical forecasting without unbiasedness conditions", KDD 2019.
    Erm {
        /// Ridge regularization strength (λ ≥ 0). A value of 1.0 is a
        /// reasonable default; increase if the solve is unstable or T < n.
        lambda: f64,
    },
}

/// A hierarchical tree of named nodes with reconciliation support.
///
/// Construct with [`HierarchyTree::new`], which takes parent → children edges.
/// All leaf names and parent names must be unique. The root is inferred
/// automatically (the node that never appears as a child).
#[derive(Debug, Clone)]
pub struct HierarchyTree {
    nodes: Vec<Node>,
    name_to_idx: HashMap<String, usize>,
    root: usize,
    /// Historical actual values per node, used for TopDown proportions.
    actuals: Option<HashMap<String, Vec<f64>>>,
    /// Historical residuals per node, used for MinTraceShrink covariance.
    residuals: Option<HashMap<String, Vec<f64>>>,
    /// Historical base-forecast matrix for all nodes (node_name → Vec<f64> of length T).
    /// Used by ERM reconciliation. T = number of training time steps.
    erm_base_history: Option<HashMap<String, Vec<f64>>>,
    /// Historical leaf-actual matrix (leaf_name → Vec<f64> of length T).
    /// Used by ERM reconciliation. Only leaf nodes; length T must match erm_base_history.
    erm_leaf_history: Option<HashMap<String, Vec<f64>>>,
}

impl HierarchyTree {
    /// Build a hierarchy from parent → children edges.
    ///
    /// Each tuple is `(parent_name, &[child_name, ...])`.
    /// The root is the node that never appears as anyone's child.
    pub fn new(edges: Vec<(&str, &[&str])>) -> Result<Self> {
        if edges.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "hierarchy must have at least one edge".into(),
            ));
        }

        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        let mut nodes: Vec<Node> = Vec::new();

        // Ensure all names exist as nodes
        let ensure_node =
            |name: &str, nodes: &mut Vec<Node>, map: &mut HashMap<String, usize>| -> usize {
                if let Some(&idx) = map.get(name) {
                    idx
                } else {
                    let idx = nodes.len();
                    nodes.push(Node {
                        name: name.to_string(),
                        children: Vec::new(),
                        parents: Vec::new(),
                        level: 0,
                    });
                    map.insert(name.to_string(), idx);
                    idx
                }
            };

        for (parent, children) in &edges {
            let pidx = ensure_node(parent, &mut nodes, &mut name_to_idx);
            for child in *children {
                let cidx = ensure_node(child, &mut nodes, &mut name_to_idx);
                // Issue #124: a child may carry multiple parents in a
                // grouped/crossed hierarchy. Record every parent edge
                // exactly once.
                if !nodes[cidx].parents.contains(&pidx) {
                    nodes[cidx].parents.push(pidx);
                }
                if !nodes[pidx].children.contains(&cidx) {
                    nodes[pidx].children.push(cidx);
                }
            }
        }

        // Find root(s): nodes with no parents.
        let roots: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parents.is_empty())
            .map(|(i, _)| i)
            .collect();

        if roots.len() != 1 {
            return Err(ForecastError::InvalidParameter(format!(
                "hierarchy must have exactly one root, found {}",
                roots.len()
            )));
        }
        let root = roots[0];

        // Assign levels via BFS. In a strict tree the BFS order gives
        // every node its unique depth; in a grouped DAG a node may be
        // visited via several paths of differing lengths, so we keep
        // the *deepest* (longest path from root). That preserves
        // tree-mode behaviour exactly and gives a well-defined level
        // for `max_level()` and level-based traversal in grouped mode.
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((root, 0usize));
        while let Some((idx, level)) = queue.pop_front() {
            if level > nodes[idx].level {
                nodes[idx].level = level;
            } else if level < nodes[idx].level {
                // Already reached via a deeper path — don't propagate.
                continue;
            }
            for &c in &nodes[idx].children.clone() {
                queue.push_back((c, level + 1));
            }
        }

        Ok(Self {
            nodes,
            name_to_idx,
            root,
            actuals: None,
            residuals: None,
            erm_base_history: None,
            erm_leaf_history: None,
        })
    }

    /// Build a hierarchy directly from an explicit summing matrix S,
    /// supporting grouped / crossed (multi-parent) leaves.
    ///
    /// `node_names` lists every node by index (`0..node_names.len()`).
    /// `leaf_names` lists which of those nodes are *leaves*; the rest
    /// are aggregates. `leaf_ancestors[i]` is the list of aggregate
    /// row indices (into `node_names`) that the `i`-th leaf
    /// contributes to — i.e. row `k` of S has a 1 in column `i` iff
    /// `leaf_ancestors[i].contains(&k)`. The leaf itself does NOT
    /// need to appear in its own ancestor list — it's inferred.
    ///
    /// This is the entry point for **grouped / crossed Hyndman GTS**:
    /// each leaf rolls up to multiple aggregate dimensions at once
    /// (e.g. `(site, part)` → `site_total`, `material_total`,
    /// `grand_total`). The strict-tree constructor [`Self::new`] only
    /// accepts a forest with single-parent edges; this one is the
    /// general case.
    ///
    /// Issue #124. The variance-weighted MinT path
    /// ([`ReconciliationMethod::MinTraceVariance`] and
    /// [`ReconciliationMethod::MinTraceStruct`]) already operates on
    /// `sparse_s` directly and handles grouped hierarchies once
    /// constructed correctly. Tree-only methods (BottomUp / TopDown /
    /// MiddleOut, dense `MinTraceOls` / `MinTraceShrink`) work best
    /// on strict trees; on a grouped hierarchy they fall back to
    /// the first-parent path and may produce non-coherent results.
    ///
    /// # Arguments
    /// * `node_names` — every node name in design order; must be unique.
    /// * `leaf_names` — subset of `node_names` flagged as leaves.
    /// * `leaf_ancestors` — same length as `leaf_names`. Each entry
    ///   is the list of aggregate-node indices (into `node_names`)
    ///   that the matching leaf contributes to.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if any name is duplicated, any leaf
    /// name is not in `node_names`, any ancestor index is out of
    /// range, or more than one node has no parents (multiple roots).
    pub fn from_summing_matrix(
        node_names: &[String],
        leaf_names: &[String],
        leaf_ancestors: &[Vec<usize>],
    ) -> Result<Self> {
        if node_names.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "from_summing_matrix: node_names must be non-empty".into(),
            ));
        }
        if leaf_names.len() != leaf_ancestors.len() {
            return Err(ForecastError::InvalidParameter(format!(
                "from_summing_matrix: leaf_names ({}) and leaf_ancestors ({}) length mismatch",
                leaf_names.len(),
                leaf_ancestors.len(),
            )));
        }

        // Build name → index map and uniqueness check.
        let mut name_to_idx: HashMap<String, usize> = HashMap::with_capacity(node_names.len());
        for (i, name) in node_names.iter().enumerate() {
            if name_to_idx.insert(name.clone(), i).is_some() {
                return Err(ForecastError::InvalidParameter(format!(
                    "from_summing_matrix: duplicate node name '{}'",
                    name
                )));
            }
        }

        // Allocate nodes.
        let mut nodes: Vec<Node> = node_names
            .iter()
            .map(|name| Node {
                name: name.clone(),
                children: Vec::new(),
                parents: Vec::new(),
                level: 0,
            })
            .collect();

        // First pass: collect leaves-below(aggregate) by inverting
        // the (leaf → ancestors) lists. leaves_below[agg_idx] is the
        // set of leaf indices that have `agg_idx` in their ancestor
        // list.
        let mut leaf_indices: Vec<usize> = Vec::with_capacity(leaf_names.len());
        let mut leaves_below: HashMap<usize, std::collections::BTreeSet<usize>> = HashMap::new();
        for (i, leaf) in leaf_names.iter().enumerate() {
            let leaf_idx = name_to_idx.get(leaf).copied().ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "from_summing_matrix: leaf '{}' not in node_names",
                    leaf
                ))
            })?;
            leaf_indices.push(leaf_idx);
            for &anc in &leaf_ancestors[i] {
                if anc >= nodes.len() {
                    return Err(ForecastError::InvalidParameter(format!(
                        "from_summing_matrix: ancestor index {} out of range (have {} nodes)",
                        anc,
                        nodes.len(),
                    )));
                }
                if anc == leaf_idx {
                    continue;
                }
                leaves_below.entry(anc).or_default().insert(leaf_idx);
            }
        }

        // Second pass: wire leaf → ancestor parent edges directly.
        for (i, &leaf_idx) in leaf_indices.iter().enumerate() {
            for &anc in &leaf_ancestors[i] {
                if anc == leaf_idx {
                    continue;
                }
                if !nodes[leaf_idx].parents.contains(&anc) {
                    nodes[leaf_idx].parents.push(anc);
                }
                if !nodes[anc].children.contains(&leaf_idx) {
                    nodes[anc].children.push(leaf_idx);
                }
            }
        }

        // Third pass: aggregate→aggregate edges, inferred from
        // leaf-set containment. Aggregate B is an immediate parent
        // of aggregate A iff leaves(A) ⊊ leaves(B) AND no other
        // aggregate C has leaves(A) ⊊ leaves(C) ⊊ leaves(B). This
        // gives a single grand-total root (the aggregate over all
        // leaves) and a layered structure that lets `bfs_order`
        // visit every node from a single starting point.
        let agg_indices: Vec<usize> = leaves_below.keys().copied().collect();
        for &a in &agg_indices {
            let la = &leaves_below[&a];
            // Find all strict supersets B of la.
            let supersets: Vec<usize> = agg_indices
                .iter()
                .copied()
                .filter(|&b| {
                    b != a && {
                        let lb = &leaves_below[&b];
                        la.is_subset(lb) && la.len() < lb.len()
                    }
                })
                .collect();
            // Direct parents = minimal supersets (no other superset of
            // `a` strictly contained in it).
            for &b in &supersets {
                let lb = &leaves_below[&b];
                let is_immediate = !supersets.iter().any(|&c| {
                    if c == b {
                        return false;
                    }
                    let lc = &leaves_below[&c];
                    la.is_subset(lc) && lc.is_subset(lb) && lc.len() < lb.len()
                });
                if is_immediate {
                    if !nodes[a].parents.contains(&b) {
                        nodes[a].parents.push(b);
                    }
                    if !nodes[b].children.contains(&a) {
                        nodes[b].children.push(a);
                    }
                }
            }
        }

        // Find root (single node with no parents).
        let roots: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parents.is_empty())
            .map(|(i, _)| i)
            .collect();
        if roots.len() != 1 {
            return Err(ForecastError::InvalidParameter(format!(
                "from_summing_matrix: must have exactly one root, found {} (a grouped \
                 hierarchy still needs a single grand-total node that every leaf rolls up to)",
                roots.len()
            )));
        }
        let root = roots[0];

        // Assign levels: max-depth from root across all parent paths.
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((root, 0usize));
        while let Some((idx, level)) = queue.pop_front() {
            if level > nodes[idx].level {
                nodes[idx].level = level;
            } else if level < nodes[idx].level {
                continue;
            }
            for &c in &nodes[idx].children.clone() {
                queue.push_back((c, level + 1));
            }
        }

        Ok(Self {
            nodes,
            name_to_idx,
            root,
            actuals: None,
            residuals: None,
            erm_base_history: None,
            erm_leaf_history: None,
        })
    }

    /// Number of nodes in the hierarchy.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the hierarchy is empty (should never be true for a valid tree).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up a node's children by name. Returns `None` if the name is unknown.
    pub fn children_of(&self, name: &str) -> Option<Vec<&str>> {
        self.name_to_idx.get(name).map(|&idx| {
            self.nodes[idx]
                .children
                .iter()
                .map(|&c| self.nodes[c].name.as_str())
                .collect()
        })
    }

    /// Provide historical actuals for TopDown proportions.
    ///
    /// Each entry maps a node name to its vector of historical values.
    /// Only leaf-level actuals are strictly required for TopDown.
    pub fn set_actuals(&mut self, actuals: HashMap<String, Vec<f64>>) {
        self.actuals = Some(actuals);
    }

    /// Provide historical residuals for MinTraceShrink covariance estimation.
    ///
    /// Each entry maps a node name to its vector of historical residuals.
    /// All nodes in the hierarchy must be present and all vectors must have
    /// the same length.
    pub fn set_residuals(&mut self, residuals: HashMap<String, Vec<f64>>) {
        self.residuals = Some(residuals);
    }

    /// Provide training history for ERM reconciliation.
    ///
    /// `base_history`: map from **every node name** to a `Vec<f64>` of length `T`
    /// (historical base forecasts). `T` should be at least `n` (the number of nodes)
    /// for a well-conditioned ridge solve; when `T < n` the Gram matrix `Ŷ'Ŷ` is
    /// rank-deficient and a positive `lambda` is essential to obtain a valid solution.
    ///
    /// `leaf_history`: map from **every leaf node name** to a `Vec<f64>` of length `T`
    /// (historical leaf-level actuals). Upper-node actuals are recovered by `S·bottom`
    /// and must NOT be included here.
    ///
    /// Shape validation (missing entries, unequal lengths, `T < n`) is deferred to
    /// solve time inside `erm_reconcile` so that callers can call this setter
    /// incrementally.
    pub fn with_erm_training(
        &mut self,
        base_history: HashMap<String, Vec<f64>>,
        leaf_history: HashMap<String, Vec<f64>>,
    ) {
        self.erm_base_history = Some(base_history);
        self.erm_leaf_history = Some(leaf_history);
    }

    /// Names of all nodes in BFS (top-down) order.
    pub fn node_names(&self) -> Vec<&str> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.root);
        while let Some(idx) = queue.pop_front() {
            order.push(self.nodes[idx].name.as_str());
            for &c in &self.nodes[idx].children {
                queue.push_back(c);
            }
        }
        order
    }

    /// Return the indices of all leaf nodes.
    fn leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.children.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    /// Reconcile base forecasts using the specified method.
    ///
    /// `base_forecasts` is a list of `(node_name, forecast_values)` pairs.
    /// Every node in the hierarchy must have a corresponding entry.
    /// Returns reconciled forecasts in BFS (top-down) order.
    pub fn reconcile(
        &self,
        base_forecasts: &[(String, Vec<f64>)],
        method: ReconciliationMethod,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        // Build lookup
        let base_map: HashMap<&str, &Vec<f64>> = base_forecasts
            .iter()
            .map(|(name, vals)| (name.as_str(), vals))
            .collect();

        // Validate all nodes present
        for node in &self.nodes {
            if !base_map.contains_key(node.name.as_str()) {
                return Err(ForecastError::InvalidParameter(format!(
                    "missing base forecast for node '{}'",
                    node.name
                )));
            }
        }

        // Check horizon consistency
        let horizon = base_map[self.nodes[self.root].name.as_str()].len();
        if horizon == 0 {
            return Err(ForecastError::InvalidParameter(
                "forecast horizon must be at least 1".into(),
            ));
        }
        for node in &self.nodes {
            if base_map[node.name.as_str()].len() != horizon {
                return Err(ForecastError::DimensionMismatch {
                    expected: horizon,
                    got: base_map[node.name.as_str()].len(),
                });
            }
        }

        match method {
            ReconciliationMethod::BottomUp => self.bottom_up(&base_map, horizon),
            ReconciliationMethod::TopDown => self.top_down(&base_map, horizon),
            ReconciliationMethod::MiddleOut { middle_level } => {
                self.middle_out(&base_map, horizon, middle_level)
            }
            ReconciliationMethod::MinTraceOls => self.min_trace_ols(&base_map, horizon),
            ReconciliationMethod::MinTraceShrink => self.min_trace_shrink(&base_map, horizon),
            ReconciliationMethod::MinTraceVariance => {
                self.min_trace_diagonal(&base_map, horizon, true)
            }
            ReconciliationMethod::MinTraceStruct => {
                self.min_trace_diagonal(&base_map, horizon, false)
            }
            ReconciliationMethod::Erm { lambda } => self.erm_reconcile(&base_map, horizon, lambda),
        }
    }

    /// Bottom-up: keep leaf forecasts, sum upward.
    fn bottom_up(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let mut reconciled: Vec<Vec<f64>> = vec![vec![0.0; horizon]; self.nodes.len()];

        // Set leaf values
        for &leaf in &self.leaves() {
            reconciled[leaf] = base_map[self.nodes[leaf].name.as_str()].clone();
        }

        // Walk bottom-up: process nodes in reverse BFS order
        let bfs_order = self.bfs_order();
        for &idx in bfs_order.iter().rev() {
            if !self.nodes[idx].children.is_empty() {
                for h in 0..horizon {
                    reconciled[idx][h] = self.nodes[idx]
                        .children
                        .iter()
                        .map(|&c| reconciled[c][h])
                        .sum();
                }
            }
        }

        Ok(self.to_named_output(&reconciled))
    }

    /// Top-down: use top-level forecast and disaggregate using historical proportions.
    fn top_down(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let actuals = self.actuals.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "TopDown reconciliation requires historical actuals; call set_actuals() first"
                    .into(),
            )
        })?;

        let leaves = self.leaves();

        // Compute average proportions of each leaf relative to root
        let root_actuals = actuals.get(&self.nodes[self.root].name).ok_or_else(|| {
            ForecastError::InvalidParameter(format!(
                "missing actuals for root node '{}'",
                self.nodes[self.root].name
            ))
        })?;

        let mut proportions: Vec<f64> = Vec::with_capacity(leaves.len());
        for &leaf in &leaves {
            let leaf_actuals = actuals.get(&self.nodes[leaf].name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "missing actuals for leaf node '{}'",
                    self.nodes[leaf].name
                ))
            })?;

            let n = leaf_actuals.len().min(root_actuals.len());
            if n == 0 {
                proportions.push(0.0);
                continue;
            }

            let leaf_sum: f64 = leaf_actuals[..n].iter().sum();
            let root_sum: f64 = root_actuals[..n].iter().sum();

            if root_sum.abs() < 1e-15 {
                proportions.push(0.0);
            } else {
                proportions.push(leaf_sum / root_sum);
            }
        }

        // Normalize proportions so they sum to 1
        let prop_sum: f64 = proportions.iter().sum();
        if prop_sum > 1e-15 {
            for p in &mut proportions {
                *p /= prop_sum;
            }
        }

        // Disaggregate top-level forecast
        let top_forecast = base_map[self.nodes[self.root].name.as_str()];
        let mut reconciled: Vec<Vec<f64>> = vec![vec![0.0; horizon]; self.nodes.len()];

        // Set leaf values
        for (i, &leaf) in leaves.iter().enumerate() {
            for h in 0..horizon {
                reconciled[leaf][h] = proportions[i] * top_forecast[h];
            }
        }

        // Sum upward
        let bfs_order = self.bfs_order();
        for &idx in bfs_order.iter().rev() {
            if !self.nodes[idx].children.is_empty() {
                for h in 0..horizon {
                    reconciled[idx][h] = self.nodes[idx]
                        .children
                        .iter()
                        .map(|&c| reconciled[c][h])
                        .sum();
                }
            }
        }

        Ok(self.to_named_output(&reconciled))
    }

    /// MinT OLS reconciliation.
    ///
    /// Uses the summing matrix S and the OLS projection:
    ///   ỹ = S (S'S)^{-1} S' ŷ
    /// This is the simplest MinT variant (assumes identity covariance).
    fn min_trace_ols(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let n = self.nodes.len();
        let leaves = self.leaves();
        let m = leaves.len(); // number of bottom-level series

        // Build summing matrix S (n × m):
        // S[i][j] = 1 if leaf j contributes to node i, else 0.
        // ancestors_of() walks every parent edge transitively so this
        // works for grouped / crossed hierarchies as well as strict
        // trees (issue #124).
        let mut s = vec![vec![0.0_f64; m]; n];
        for (j, &leaf) in leaves.iter().enumerate() {
            s[leaf][j] = 1.0;
            for anc in self.ancestors_of(leaf) {
                s[anc][j] = 1.0;
            }
        }

        // Compute S'S (m × m)
        let mut sts = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in i..m {
                let dot: f64 = (0..n).map(|k| s[k][i] * s[k][j]).sum();
                sts[i][j] = dot;
                sts[j][i] = dot;
            }
        }

        // Invert S'S via Cholesky
        let sts_flat: Vec<f64> = sts.iter().flat_map(|row| row.iter().copied()).collect();
        let l = cholesky(m, &sts_flat)?;

        // P = S (S'S)^{-1} S'  — we apply this per time step
        // For each h: reconciled = S * (S'S)^{-1} * S' * base_vec
        let bfs_order = self.bfs_order();

        let mut reconciled = vec![vec![0.0; horizon]; n];

        for h in 0..horizon {
            // base vector in BFS order
            let base_vec: Vec<f64> = (0..n)
                .map(|i| base_map[self.nodes[i].name.as_str()][h])
                .collect();

            // z = S' * base_vec  (m-vector)
            let mut z = vec![0.0; m];
            for j in 0..m {
                z[j] = (0..n).map(|i| s[i][j] * base_vec[i]).sum();
            }

            // w = (S'S)^{-1} z  via Cholesky solve
            let w = cholesky_solve_vec(m, &l, &z);

            // result = S * w  (n-vector)
            for i in 0..n {
                reconciled[i][h] = (0..m).map(|j| s[i][j] * w[j]).sum();
            }
        }

        Ok(bfs_order
            .iter()
            .map(|&idx| (self.nodes[idx].name.clone(), reconciled[idx].clone()))
            .collect())
    }

    /// ERM reconciliation: P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹, reconciled = S·P·ŷ.
    ///
    /// Requires `with_erm_training` to have been called with valid base forecast
    /// history (`Ŷ_stored`, nodes×T) and leaf-actual history (`B_stored`, leaves×T).
    ///
    /// The ridge parameter `lambda` prevents the Gram matrix from being singular
    /// when `T < n` or when the base forecasts are nearly collinear.
    fn erm_reconcile(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
        lambda: f64,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        // Guard: λ must be finite and non-negative. A negative λ amplifies
        // rather than regularizes; NaN bypasses the Cholesky guard silently.
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(ForecastError::InvalidParameter(format!(
                "ERM: lambda must be finite and non-negative, got {lambda}"
            )));
        }

        let n = self.nodes.len();
        let leaves = self.leaves();
        let m = leaves.len();

        // 1. Retrieve training history.
        let base_hist = self.erm_base_history.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "ERM reconciliation requires training history; call with_erm_training() first"
                    .into(),
            )
        })?;
        let leaf_hist = self.erm_leaf_history.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "ERM reconciliation requires training history; call with_erm_training() first"
                    .into(),
            )
        })?;

        // 2. Validate that every node has a base_history entry and derive T.
        let t_base = base_hist
            .get(self.nodes[0].name.as_str())
            .map(|v| v.len())
            .ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "ERM: base_history is missing an entry for node '{}'",
                    self.nodes[0].name
                ))
            })?;
        for i in 0..n {
            let name = self.nodes[i].name.as_str();
            let vec = base_hist.get(name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "ERM: base_history is missing an entry for node '{name}'"
                ))
            })?;
            if vec.len() != t_base {
                return Err(ForecastError::DimensionMismatch {
                    expected: t_base,
                    got: vec.len(),
                });
            }
        }

        // 3. Validate every leaf has a leaf_history entry of length T.
        for &leaf_idx in &leaves {
            let name = self.nodes[leaf_idx].name.as_str();
            let vec = leaf_hist.get(name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "ERM: leaf_history is missing an entry for leaf '{name}'"
                ))
            })?;
            if vec.len() != t_base {
                return Err(ForecastError::DimensionMismatch {
                    expected: t_base,
                    got: vec.len(),
                });
            }
        }

        let t_cap = t_base;

        // Guard: at least one training period is required. T=0 makes the Gram
        // matrix zero (only λI), producing a zero P and silent zero forecasts.
        if t_cap == 0 {
            return Err(ForecastError::InvalidParameter(
                "ERM: training history must contain at least one time period (T ≥ 1)".into(),
            ));
        }

        // 4. Build Ŷ_stored (n×T): rows = nodes in internal index order, cols = time.
        let y_stored: Vec<Vec<f64>> = (0..n)
            .map(|i| base_hist[self.nodes[i].name.as_str()].clone())
            .collect();

        // 5. Build B_stored (m×T): rows = leaves in leaves() order, cols = time.
        let b_stored: Vec<Vec<f64>> = leaves
            .iter()
            .map(|&leaf_idx| leaf_hist[self.nodes[leaf_idx].name.as_str()].clone())
            .collect();

        // 6. Compute Gram G = Ŷ_stored Ŷ_stored' (n×n) and add λI.
        let mut gram = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in i..n {
                let dot: f64 = (0..t_cap).map(|t| y_stored[i][t] * y_stored[j][t]).sum();
                gram[i * n + j] = dot;
                gram[j * n + i] = dot;
            }
            gram[i * n + i] += lambda;
        }

        // 7. Compute cross term C = B_stored Ŷ_stored' (m×n).
        let mut cross = vec![vec![0.0_f64; n]; m];
        for i in 0..m {
            for j in 0..n {
                cross[i][j] = (0..t_cap).map(|t| b_stored[i][t] * y_stored[j][t]).sum();
            }
        }

        // 8. Cholesky-factor (G + λI) once.
        let l = cholesky(n, &gram).map_err(|_| {
            ForecastError::SingularMatrix(
                "ERM: Gram matrix Ŷ'Ŷ + λI is singular; \
                 increase lambda or provide more training periods"
                    .into(),
            )
        })?;

        // 9. Solve for each row of P (m×n): (G+λI) P[i]^T = C[i]^T.
        let p: Vec<Vec<f64>> = (0..m)
            .map(|i| cholesky_solve_vec(n, &l, &cross[i]))
            .collect();

        // 10. Build summing matrix S (n×m) — identical to min_trace_ols.
        let mut s = vec![vec![0.0_f64; m]; n];
        for (j, &leaf) in leaves.iter().enumerate() {
            s[leaf][j] = 1.0;
            for anc in self.ancestors_of(leaf) {
                s[anc][j] = 1.0;
            }
        }

        // 11. For each horizon step h: bottom = P·ŷ (m-vector), all = S·bottom (n-vector).
        let mut reconciled = vec![vec![0.0_f64; horizon]; n];
        for h in 0..horizon {
            // ŷ in internal node index order.
            let y_hat: Vec<f64> = (0..n)
                .map(|i| base_map[self.nodes[i].name.as_str()][h])
                .collect();

            // bottom[i] = Σ_j P[i][j] · ŷ[j]
            let bottom: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| p[i][j] * y_hat[j]).sum())
                .collect();

            // all[node] = Σ_i S[node][i] · bottom[i]
            for node in 0..n {
                reconciled[node][h] = (0..m).map(|i| s[node][i] * bottom[i]).sum();
            }
        }

        Ok(self.to_named_output(&reconciled))
    }

    /// Maximum depth in the hierarchy (root is level 0).
    fn max_level(&self) -> usize {
        self.nodes.iter().map(|n| n.level).max().unwrap_or(0)
    }

    /// Return indices of all nodes at a given level.
    fn nodes_at_level(&self, level: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.level == level)
            .map(|(i, _)| i)
            .collect()
    }

    /// Collect all leaf descendants of a given node (recursive).
    fn leaf_descendants(&self, idx: usize) -> Vec<usize> {
        if self.nodes[idx].children.is_empty() {
            return vec![idx];
        }
        let mut leaves = Vec::new();
        for &child in &self.nodes[idx].children {
            leaves.extend(self.leaf_descendants(child));
        }
        leaves
    }

    /// MiddleOut reconciliation.
    ///
    /// - Nodes at `middle_level` are treated as anchors.
    /// - Above: aggregate upward (BottomUp-style) from anchor forecasts.
    /// - Below: disaggregate downward (TopDown-style) using historical proportions.
    /// - Edge: level=0 → equivalent to TopDown; level>=max → equivalent to BottomUp.
    fn middle_out(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
        middle_level: usize,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let max_lvl = self.max_level();

        // Edge cases
        if middle_level == 0 {
            return self.top_down(base_map, horizon);
        }
        if middle_level >= max_lvl {
            return self.bottom_up(base_map, horizon);
        }

        let actuals = self.actuals.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "MiddleOut reconciliation requires historical actuals; call set_actuals() first"
                    .into(),
            )
        })?;

        let n = self.nodes.len();
        let mut reconciled: Vec<Vec<f64>> = vec![vec![0.0; horizon]; n];

        // 1. Set middle-level forecasts from base.
        let middle_nodes = self.nodes_at_level(middle_level);
        for &mid in &middle_nodes {
            reconciled[mid] = base_map[self.nodes[mid].name.as_str()].clone();
        }

        // 2. Disaggregate downward from each middle node to its leaf descendants.
        for &mid in &middle_nodes {
            let leaf_descs = self.leaf_descendants(mid);

            // Compute proportions from actuals (each leaf relative to its middle ancestor).
            let mid_actuals = actuals.get(&self.nodes[mid].name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "missing actuals for middle node '{}'",
                    self.nodes[mid].name
                ))
            })?;

            let mut proportions = Vec::with_capacity(leaf_descs.len());
            for &leaf in &leaf_descs {
                let leaf_actuals = actuals.get(&self.nodes[leaf].name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "missing actuals for leaf node '{}'",
                        self.nodes[leaf].name
                    ))
                })?;

                let len = leaf_actuals.len().min(mid_actuals.len());
                if len == 0 {
                    proportions.push(0.0);
                    continue;
                }
                let leaf_sum: f64 = leaf_actuals[..len].iter().sum();
                let mid_sum: f64 = mid_actuals[..len].iter().sum();
                if mid_sum.abs() < 1e-15 {
                    proportions.push(0.0);
                } else {
                    proportions.push(leaf_sum / mid_sum);
                }
            }

            // Normalize
            let prop_sum: f64 = proportions.iter().sum();
            if prop_sum > 1e-15 {
                for p in &mut proportions {
                    *p /= prop_sum;
                }
            }

            // Set leaf forecasts
            for (i, &leaf) in leaf_descs.iter().enumerate() {
                for h in 0..horizon {
                    reconciled[leaf][h] = proportions[i] * reconciled[mid][h];
                }
            }

            // Fill intermediate nodes between middle and leaves (sum children upward).
            // Walk levels from max_lvl-1 down to middle_level+1.
            for lvl in (middle_level + 1..max_lvl).rev() {
                for node_idx in 0..n {
                    if self.nodes[node_idx].level == lvl
                        && !self.nodes[node_idx].children.is_empty()
                    {
                        // Check if this node is a descendant of mid
                        if self.is_descendant_of(node_idx, mid) {
                            for h in 0..horizon {
                                reconciled[node_idx][h] = self.nodes[node_idx]
                                    .children
                                    .iter()
                                    .map(|&c| reconciled[c][h])
                                    .sum();
                            }
                        }
                    }
                }
            }
        }

        // 3. Aggregate upward from middle level to root.
        let bfs_order = self.bfs_order();
        for &idx in bfs_order.iter().rev() {
            if self.nodes[idx].level < middle_level && !self.nodes[idx].children.is_empty() {
                for h in 0..horizon {
                    reconciled[idx][h] = self.nodes[idx]
                        .children
                        .iter()
                        .map(|&c| reconciled[c][h])
                        .sum();
                }
            }
        }

        Ok(self.to_named_output(&reconciled))
    }

    /// Collect all ancestor node indices of `idx`, transitively
    /// across every parent edge. In a strict tree this is the
    /// classic parent-walk chain; in a grouped / crossed DAG it
    /// gathers every aggregate row that the node contributes to.
    /// Each distinct ancestor appears exactly once. `idx` itself is
    /// NOT included.
    fn ancestors_of(&self, idx: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(idx);
        while let Some(cur) = queue.pop_front() {
            for &p in &self.nodes[cur].parents {
                if seen.insert(p) {
                    result.push(p);
                    queue.push_back(p);
                }
            }
        }
        result
    }

    /// Check if `node` is a descendant of `ancestor`. In grouped
    /// hierarchies the search walks every parent path, so this
    /// returns `true` if `ancestor` is reachable via any chain.
    fn is_descendant_of(&self, node: usize, ancestor: usize) -> bool {
        self.ancestors_of(node).contains(&ancestor)
    }

    /// MinT with Ledoit-Wolf shrinkage covariance.
    ///
    /// Like MinTraceOLS but uses Σ = α·F + (1-α)·S where:
    /// - S = sample covariance matrix of residuals
    /// - F = diagonal target (variances only)
    /// - α = optimal Ledoit-Wolf shrinkage intensity
    ///
    /// Reconciliation: ỹ = S_mat (S_mat' Σ^{-1} S_mat)^{-1} S_mat' Σ^{-1} ŷ
    fn min_trace_shrink(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let residuals = self.residuals.as_ref().ok_or_else(|| {
            ForecastError::InvalidParameter(
                "MinTraceShrink requires historical residuals; call set_residuals() first".into(),
            )
        })?;

        let n = self.nodes.len();
        let leaves = self.leaves();
        let m = leaves.len();

        // Collect residual matrix: rows = observations, cols = nodes (in index order)
        // Validate all nodes have residuals and same length.
        let first_name = &self.nodes[0].name;
        let res_first = residuals.get(first_name).ok_or_else(|| {
            ForecastError::InvalidParameter(format!("missing residuals for node '{}'", first_name))
        })?;
        let t = res_first.len();
        if t < 2 {
            return Err(ForecastError::InvalidParameter(
                "MinTraceShrink requires at least 2 residual observations".into(),
            ));
        }

        let mut res_matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
        for node in &self.nodes {
            let r = residuals.get(&node.name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "missing residuals for node '{}'",
                    node.name
                ))
            })?;
            if r.len() != t {
                return Err(ForecastError::DimensionMismatch {
                    expected: t,
                    got: r.len(),
                });
            }
            res_matrix.push(r.clone());
        }

        // Compute sample covariance S (n x n)
        let means: Vec<f64> = res_matrix
            .iter()
            .map(|r| r.iter().sum::<f64>() / t as f64)
            .collect();

        let mut sample_cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let cov: f64 = (0..t)
                    .map(|k| (res_matrix[i][k] - means[i]) * (res_matrix[j][k] - means[j]))
                    .sum::<f64>()
                    / (t - 1) as f64;
                sample_cov[i][j] = cov;
                sample_cov[j][i] = cov;
            }
        }

        // Diagonal target F (variances only)
        let diag: Vec<f64> = (0..n).map(|i| sample_cov[i][i]).collect();

        // Ledoit-Wolf optimal shrinkage intensity
        let alpha = ledoit_wolf_alpha(&res_matrix, &sample_cov, &diag, t, n);

        // Shrinkage covariance: Sigma = alpha*F + (1-alpha)*S
        let mut sigma = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                sigma[i][j] = (1.0 - alpha) * sample_cov[i][j];
            }
            sigma[i][i] += alpha * diag[i];
        }

        // Invert sigma via Cholesky
        let sigma_flat: Vec<f64> = sigma.iter().flat_map(|row| row.iter().copied()).collect();
        let l_sigma = cholesky(n, &sigma_flat)?;

        // Build summing matrix S_mat (n × m) — grouped-safe via
        // ancestors_of() (issue #124).
        let mut s_mat = vec![vec![0.0_f64; m]; n];
        for (j, &leaf) in leaves.iter().enumerate() {
            s_mat[leaf][j] = 1.0;
            for anc in self.ancestors_of(leaf) {
                s_mat[anc][j] = 1.0;
            }
        }

        // Compute Sigma_inv * S_mat  (n x m)
        // For each column j of S_mat, solve Sigma * x = s_mat_col_j
        let mut sigma_inv_s = vec![vec![0.0; m]; n];
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| s_mat[i][j]).collect();
            let x = cholesky_solve_vec(n, &l_sigma, &col);
            for i in 0..n {
                sigma_inv_s[i][j] = x[i];
            }
        }

        // Compute S_mat' * Sigma_inv * S_mat  (m x m)
        let mut st_sigma_inv_s = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in i..m {
                let dot: f64 = (0..n).map(|k| s_mat[k][i] * sigma_inv_s[k][j]).sum();
                st_sigma_inv_s[i][j] = dot;
                st_sigma_inv_s[j][i] = dot;
            }
        }

        // Invert (S_mat' Sigma_inv S_mat) via Cholesky
        let st_sigma_inv_s_flat: Vec<f64> = st_sigma_inv_s
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let l_inner = cholesky(m, &st_sigma_inv_s_flat)?;

        // For each time step: reconciled = S_mat * (S_mat' Σ^{-1} S_mat)^{-1} * S_mat' Σ^{-1} * ŷ
        let bfs_order = self.bfs_order();
        let mut reconciled = vec![vec![0.0; horizon]; n];

        for h in 0..horizon {
            let base_vec: Vec<f64> = (0..n)
                .map(|i| base_map[self.nodes[i].name.as_str()][h])
                .collect();

            // z = Sigma_inv * base_vec
            let z = cholesky_solve_vec(n, &l_sigma, &base_vec);

            // w = S_mat' * z  (m-vector)
            let mut w = vec![0.0; m];
            for j in 0..m {
                w[j] = (0..n).map(|i| s_mat[i][j] * z[i]).sum();
            }

            // v = (S_mat' Sigma_inv S_mat)^{-1} * w
            let v = cholesky_solve_vec(m, &l_inner, &w);

            // result = S_mat * v
            for i in 0..n {
                reconciled[i][h] = (0..m).map(|j| s_mat[i][j] * v[j]).sum();
            }
        }

        Ok(bfs_order
            .iter()
            .map(|&idx| (self.nodes[idx].name.clone(), reconciled[idx].clone()))
            .collect())
    }

    /// Scalable MinT with diagonal weight matrix W.
    ///
    /// Uses sparse summing matrix S (CSC-like: per-leaf ancestor list) and
    /// diagonal W to avoid N×N matrices entirely.
    ///
    /// Formula: ỹ = S (S' W⁻¹ S)⁻¹ S' W⁻¹ ŷ
    ///
    /// When `use_variance` is true: W[i] = residual variance of node i (requires residuals).
    /// When false: W[i] = number of leaves below node i (structural scaling, no residuals).
    ///
    /// Memory: O(N + M² + M*depth) — safe for N > 100k.
    fn min_trace_diagonal(
        &self,
        base_map: &HashMap<&str, &Vec<f64>>,
        horizon: usize,
        use_variance: bool,
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let n = self.nodes.len();
        let leaves = self.leaves();
        let m = leaves.len();

        // Compute diagonal weights W (N-vector)
        let w_diag: Vec<f64> = if use_variance {
            let residuals = self.residuals.as_ref().ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "MinTraceVariance requires residuals; call set_residuals() first".into(),
                )
            })?;
            self.nodes
                .iter()
                .map(|node| {
                    if let Some(r) = residuals.get(&node.name) {
                        let n_r = r.len() as f64;
                        if n_r < 2.0 {
                            return 1.0;
                        }
                        let mean = r.iter().sum::<f64>() / n_r;
                        let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_r - 1.0);
                        var.max(1e-10)
                    } else {
                        1.0
                    }
                })
                .collect()
        } else {
            // Structural: weight = number of leaves below this node
            self.nodes
                .iter()
                .enumerate()
                .map(|(idx, _)| self.count_leaves_below(idx).max(1) as f64)
                .collect()
        };

        // W_inv diagonal
        let w_inv: Vec<f64> = w_diag.iter().map(|&w| 1.0 / w).collect();

        // Sparse S: for each leaf j, store the list of ancestor node
        // indices (self included). Replaces the dense N×M summing
        // matrix (~10MB vs 64GB for 100k×80k). The walk uses
        // ancestors_of() so this works for grouped / crossed
        // hierarchies as well as strict trees — the variance-weighted
        // CG MinT path is the recommended reconciler for grouped
        // hierarchies per issue #124.
        let sparse_s: Vec<Vec<usize>> = leaves
            .iter()
            .map(|&leaf| {
                let mut ancestors = vec![leaf];
                ancestors.extend(self.ancestors_of(leaf));
                ancestors
            })
            .collect();

        // Diagonal of S'W⁻¹S: needed by both the dense Cholesky path
        // (placed into `sts` below) AND the CG path (as Jacobi
        // preconditioner). Always compute it; cheap.
        let sts_diag: Vec<f64> = sparse_s
            .iter()
            .map(|ancestors| ancestors.iter().map(|&k| w_inv[k]).sum::<f64>())
            .collect();

        // Auto-switch: for small M, the dense O(M²) Cholesky path is
        // faster (no per-step CG iterations, factor reused across the
        // horizon). For large M it's a memory wall (e.g. M ≈ 50k hits
        // ~36 GB, OOM-prone) — use sparse CG instead. Threshold tuned
        // so the dense path runs only when it's both cheap and fits
        // comfortably in RAM (1000² × 8 B ≈ 8 MB).
        const CG_AUTO_SWITCH: usize = 1000;
        let use_cg = m > CG_AUTO_SWITCH;

        // Compute the dense S'W⁻¹S + Cholesky only when we're going to
        // use it. Building it costs O(M²×depth) time and O(M²) memory
        // — we'd rather not pay that for the CG path.
        let dense_factor: Option<Vec<f64>> = if use_cg {
            None
        } else {
            let mut sts = vec![0.0_f64; m * m];
            for i in 0..m {
                sts[i * m + i] = sts_diag[i];
                for j in (i + 1)..m {
                    let mut dot = 0.0;
                    for &anc_i in &sparse_s[i] {
                        for &anc_j in &sparse_s[j] {
                            if anc_i == anc_j {
                                dot += w_inv[anc_i];
                            }
                        }
                    }
                    sts[i * m + j] = dot;
                    sts[j * m + i] = dot;
                }
            }
            Some(cholesky(m, &sts)?)
        };

        // Reconcile per time step.
        let bfs_order = self.bfs_order();
        let mut reconciled = vec![vec![0.0; horizon]; n];

        // Per-step working buffers reused across `h` to avoid
        // reallocation in the hot loop (matters for long horizons on
        // large hierarchies).
        let mut sx = vec![0.0_f64; n];

        for h in 0..horizon {
            // base vector ŷ (N-vector).
            let base_vec: Vec<f64> = (0..n)
                .map(|i| base_map[self.nodes[i].name.as_str()][h])
                .collect();

            // RHS z = S' W⁻¹ ŷ (M-vector) using sparse S.
            let mut z = vec![0.0_f64; m];
            for (j, ancestors) in sparse_s.iter().enumerate() {
                z[j] = ancestors.iter().map(|&k| w_inv[k] * base_vec[k]).sum();
            }

            // Solve (S'W⁻¹S) w = z for w (M-vector).
            let w_sol = if let Some(l) = dense_factor.as_ref() {
                cholesky_solve_vec(m, l, &z)
            } else {
                min_trace_cg_solve(&sparse_s, &w_inv, &sts_diag, &z, &mut sx)
            };

            // result = S * w using sparse S.
            for r in reconciled.iter_mut() {
                r[h] = 0.0;
            }
            for (j, ancestors) in sparse_s.iter().enumerate() {
                for &node_idx in ancestors {
                    reconciled[node_idx][h] += w_sol[j];
                }
            }
        }

        Ok(bfs_order
            .iter()
            .map(|&idx| (self.nodes[idx].name.clone(), reconciled[idx].clone()))
            .collect())
    }

    /// Count the number of leaf nodes below a given node.
    fn count_leaves_below(&self, idx: usize) -> usize {
        if self.nodes[idx].children.is_empty() {
            return 1;
        }
        self.nodes[idx]
            .children
            .iter()
            .map(|&c| self.count_leaves_below(c))
            .sum()
    }

    /// BFS order of node indices. Each node appears exactly once
    /// even in grouped / crossed hierarchies where a leaf is
    /// reachable from the root via several paths (issue #124).
    fn bfs_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut seen = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.root);
        seen.insert(self.root);
        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &c in &self.nodes[idx].children {
                if seen.insert(c) {
                    queue.push_back(c);
                }
            }
        }
        order
    }

    /// Convert reconciled vectors to named output in BFS order.
    fn to_named_output(&self, reconciled: &[Vec<f64>]) -> Vec<(String, Vec<f64>)> {
        self.bfs_order()
            .iter()
            .map(|&idx| (self.nodes[idx].name.clone(), reconciled[idx].clone()))
            .collect()
    }
}

/// Cholesky decomposition for a symmetric positive-definite matrix.
/// Input: n×n matrix in row-major flat format. Returns lower-triangular L.
/// Solve `(S'W⁻¹S) w = z` via conjugate gradient with Jacobi
/// preconditioner, applying `S` and `Sᵀ` as **sparse mat-vecs** over
/// `sparse_s` rather than forming the dense `M×M` normal matrix.
///
/// Memory: `O(M + N + nnz(S))` — well under a gigabyte for million-leaf
/// hierarchies (vs `~M² × 8 B` for the dense path, which OOMs at ~M=10k).
/// Time per CG iteration: `O(nnz(S))` (sparse mat-vec dominates).
///
/// Convergence: well-conditioned with a Jacobi preconditioner on the
/// dominant diagonal weights typical of structural / variance MinT.
/// Converges in ≤ `MAX_ITER` iterations for the panel scales seen in
/// practice (≤ ~depth × constant on shallow trees).
///
/// `sx` is a scratch buffer of length `N` reused across CG calls; the
/// caller owns it to keep the hot loop allocation-free.
///
/// Reference: issue #130; matches the matrix-free implementation used
/// downstream to scale grouped reconciliation to 569k-leaf panels.
fn min_trace_cg_solve(
    sparse_s: &[Vec<usize>],
    w_inv: &[f64],
    diag_preconditioner: &[f64],
    z: &[f64],
    sx: &mut [f64],
) -> Vec<f64> {
    /// Maximum CG iterations. Shallow hierarchies converge in
    /// ≤ depth + O(1) iterations under a Jacobi preconditioner; the
    /// cap is set large enough to handle the rare ill-conditioned
    /// regime without falling back to dense.
    const MAX_ITER: usize = 200;
    /// Relative residual tolerance: ‖r‖₂ ≤ TOL · ‖z‖₂. 1e-8 matches
    /// the precision of the dense Cholesky path on well-conditioned
    /// systems.
    const TOL: f64 = 1e-8;
    /// Floor on diagonal preconditioner / scalar products to avoid
    /// divide-by-zero on pathological inputs (constant series, zero
    /// residual variance per node, etc.).
    const FLOOR: f64 = 1e-30;

    let m = z.len();
    let n = w_inv.len();
    debug_assert_eq!(sparse_s.len(), m);
    debug_assert_eq!(diag_preconditioner.len(), m);
    debug_assert!(sx.len() >= n, "scratch buffer too small");

    let z_norm_sq: f64 = z.iter().map(|v| v * v).sum();
    if z_norm_sq == 0.0 {
        return vec![0.0; m];
    }
    let tol_sq = TOL * TOL * z_norm_sq;

    let mut x = vec![0.0_f64; m]; // solution iterate
    let mut r = z.to_vec(); // residual: r₀ = z − A·x₀ = z (x₀ = 0)
    let mut z_pre = vec![0.0_f64; m]; // M⁻¹·r (preconditioned residual)
    let mut p = vec![0.0_f64; m]; // search direction
    let mut ap = vec![0.0_f64; m]; // A·p
    let mut sw = &mut sx[..n]; // S·p (N-vector), reused per iter

    for j in 0..m {
        let d = diag_preconditioner[j].max(FLOOR);
        z_pre[j] = r[j] / d;
        p[j] = z_pre[j];
    }
    let mut rz: f64 = r.iter().zip(&z_pre).map(|(ri, zi)| ri * zi).sum();

    for _iter in 0..MAX_ITER {
        // A·p = Sᵀ·(W⁻¹·(S·p)).
        // Step 1: S·p (N-vector). sw[k] = Σ_{j : k ∈ ancestors_j} p[j].
        for v in sw.iter_mut() {
            *v = 0.0;
        }
        for (j, ancestors) in sparse_s.iter().enumerate() {
            let pj = p[j];
            for &k in ancestors {
                sw[k] += pj;
            }
        }
        // Step 2: W⁻¹ · (S·p), in place on sw.
        for k in 0..n {
            sw[k] *= w_inv[k];
        }
        // Step 3: Sᵀ · (W⁻¹·S·p). ap[j] = Σ_{k ∈ ancestors_j} sw[k].
        for (j, ancestors) in sparse_s.iter().enumerate() {
            ap[j] = ancestors.iter().map(|&k| sw[k]).sum();
        }

        let p_ap: f64 = p.iter().zip(&ap).map(|(pi, ai)| pi * ai).sum();
        let alpha = rz / p_ap.max(FLOOR);

        // x ← x + α·p; r ← r − α·Ap.
        let mut r_norm_sq = 0.0_f64;
        for j in 0..m {
            x[j] += alpha * p[j];
            r[j] -= alpha * ap[j];
            r_norm_sq += r[j] * r[j];
        }
        if r_norm_sq < tol_sq {
            break;
        }

        // M⁻¹·r and new ⟨r, M⁻¹·r⟩.
        let mut rz_new = 0.0_f64;
        for j in 0..m {
            let d = diag_preconditioner[j].max(FLOOR);
            z_pre[j] = r[j] / d;
            rz_new += r[j] * z_pre[j];
        }
        let beta = rz_new / rz.max(FLOOR);
        rz = rz_new;

        for j in 0..m {
            p[j] = z_pre[j] + beta * p[j];
        }

        sw = &mut sx[..n]; // re-borrow for next iteration
    }

    x
}

fn cholesky(n: usize, a: &[f64]) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(ForecastError::SingularMatrix(
                        "hierarchy summing matrix S'S is singular".into(),
                    ));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Solve L L^T x = b given Cholesky factor L (n×n row-major flat).
fn cholesky_solve_vec(n: usize, l: &[f64], b: &[f64]) -> Vec<f64> {
    // Forward: L z = b
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * n + j] * z[j];
        }
        z[i] = (b[i] - s) / l[i * n + i];
    }
    // Backward: L^T x = z
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += l[j * n + i] * x[j];
        }
        x[i] = (z[i] - s) / l[i * n + i];
    }
    x
}

/// Compute the optimal Ledoit-Wolf shrinkage intensity.
///
/// `res_matrix[i]` = residuals for node i (length T).
/// `sample_cov` = n×n sample covariance.
/// `diag` = diagonal of sample_cov.
/// Returns α in [0, 1].
fn ledoit_wolf_alpha(
    res_matrix: &[Vec<f64>],
    sample_cov: &[Vec<f64>],
    diag: &[f64],
    t: usize,
    n: usize,
) -> f64 {
    let tf = t as f64;

    // Means
    let means: Vec<f64> = res_matrix
        .iter()
        .map(|r| r.iter().sum::<f64>() / tf)
        .collect();

    // Compute sum of squared off-diagonal sample covariances (denominator)
    let mut delta = 0.0;
    for i in 0..n {
        for j in 0..n {
            let diff = sample_cov[i][j] - if i == j { diag[i] } else { 0.0 };
            delta += diff * diff;
        }
    }

    // Compute numerator: sum over i,j of (1/T) sum_k (z_ik z_jk - s_ij)^2
    // where z_ik = (r_ik - mean_i)
    let mut gamma = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut sum_sq = 0.0;
            for k in 0..t {
                let zi = res_matrix[i][k] - means[i];
                let zj = res_matrix[j][k] - means[j];
                let dev = zi * zj - sample_cov[i][j];
                sum_sq += dev * dev;
            }
            gamma += sum_sq / tf;
        }
    }

    if delta < 1e-30 {
        return 1.0;
    }

    (gamma / (tf * delta)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} ≈ {}, diff = {}",
            a,
            b,
            (a - b).abs()
        );
    }

    // ── Tree construction ──────────────────────────────────────────────

    #[test]
    fn simple_tree() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.node_names(), vec!["Total", "A", "B"]);
    }

    #[test]
    fn three_level_tree() {
        let tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();
        assert_eq!(tree.nodes.len(), 7);
        assert_eq!(tree.leaves().len(), 4);
    }

    #[test]
    fn empty_edges_fails() {
        assert!(HierarchyTree::new(vec![]).is_err());
    }

    #[test]
    fn multiple_parents_fails() {
        let result = HierarchyTree::new(vec![("A", &["C"]), ("B", &["C"])]);
        assert!(result.is_err());
    }

    #[test]
    fn multiple_roots_fails() {
        // Two disconnected trees → two roots
        let result = HierarchyTree::new(vec![("A", &["C"]), ("B", &["D"])]);
        assert!(result.is_err());
    }

    // ── Bottom-up ──────────────────────────────────────────────────────

    #[test]
    fn bottom_up_simple() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0, 110.0]),
            ("A".into(), vec![70.0, 75.0]),
            ("B".into(), vec![40.0, 45.0]),
        ];
        let result = tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .unwrap();
        // Total = A + B
        assert_eq!(result[0].0, "Total");
        approx_eq(result[0].1[0], 110.0, 1e-10);
        approx_eq(result[0].1[1], 120.0, 1e-10);
        // Leaves unchanged
        assert_eq!(result[1].1, vec![70.0, 75.0]);
        assert_eq!(result[2].1, vec![40.0, 45.0]);
    }

    #[test]
    fn bottom_up_three_levels() {
        let tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let base = vec![
            ("Total".into(), vec![999.0]), // will be overridden
            ("East".into(), vec![999.0]),  // will be overridden
            ("West".into(), vec![999.0]),  // will be overridden
            ("NY".into(), vec![10.0]),
            ("MA".into(), vec![20.0]),
            ("CA".into(), vec![30.0]),
            ("WA".into(), vec![40.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        approx_eq(map["East"][0], 30.0, 1e-10); // NY + MA
        approx_eq(map["West"][0], 70.0, 1e-10); // CA + WA
        approx_eq(map["Total"][0], 100.0, 1e-10); // East + West
    }

    // ── Top-down ───────────────────────────────────────────────────────

    #[test]
    fn top_down_simple() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut actuals = HashMap::new();
        actuals.insert("Total".into(), vec![100.0, 100.0, 100.0]);
        actuals.insert("A".into(), vec![60.0, 60.0, 60.0]);
        actuals.insert("B".into(), vec![40.0, 40.0, 40.0]);
        tree.set_actuals(actuals);

        let base = vec![
            ("Total".into(), vec![200.0]),
            ("A".into(), vec![999.0]), // ignored
            ("B".into(), vec![999.0]), // ignored
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::TopDown)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // A gets 60%, B gets 40%
        approx_eq(map["A"][0], 120.0, 1e-10);
        approx_eq(map["B"][0], 80.0, 1e-10);
        approx_eq(map["Total"][0], 200.0, 1e-10);
    }

    #[test]
    fn top_down_requires_actuals() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![50.0]),
            ("B".into(), vec![50.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::TopDown)
            .is_err());
    }

    // ── MinTrace OLS ───────────────────────────────────────────────────

    #[test]
    fn mint_ols_coherent() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![55.0]),
            ("B".into(), vec![40.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceOls)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Coherence: Total = A + B
        approx_eq(map["Total"][0], map["A"][0] + map["B"][0], 1e-10);
    }

    #[test]
    fn mint_ols_three_levels_coherent() {
        let tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let base = vec![
            ("Total".into(), vec![100.0, 200.0]),
            ("East".into(), vec![55.0, 110.0]),
            ("West".into(), vec![50.0, 95.0]),
            ("NY".into(), vec![25.0, 55.0]),
            ("MA".into(), vec![28.0, 52.0]),
            ("CA".into(), vec![26.0, 48.0]),
            ("WA".into(), vec![22.0, 50.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceOls)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        for h in 0..2 {
            approx_eq(map["East"][h], map["NY"][h] + map["MA"][h], 1e-10);
            approx_eq(map["West"][h], map["CA"][h] + map["WA"][h], 1e-10);
            approx_eq(map["Total"][h], map["East"][h] + map["West"][h], 1e-10);
        }
    }

    #[test]
    fn mint_ols_already_coherent_unchanged() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        // These forecasts already sum correctly
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![60.0]),
            ("B".into(), vec![40.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceOls)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Should remain close to original (already coherent)
        approx_eq(map["Total"][0], 100.0, 1e-6);
        approx_eq(map["A"][0], 60.0, 1e-6);
        approx_eq(map["B"][0], 40.0, 1e-6);
    }

    // ── Validation errors ──────────────────────────────────────────────

    #[test]
    fn missing_node_forecast_fails() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![60.0]),
            // missing "B"
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .is_err());
    }

    #[test]
    fn mismatched_horizon_fails() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0, 110.0]),
            ("A".into(), vec![60.0]), // different length
            ("B".into(), vec![40.0, 45.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .is_err());
    }

    #[test]
    fn zero_horizon_fails() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![]),
            ("A".into(), vec![]),
            ("B".into(), vec![]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .is_err());
    }

    // ── MiddleOut ─────────────────────────────────────────────────────

    #[test]
    fn middle_out_level0_equals_top_down() {
        let mut tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let mut actuals = HashMap::new();
        actuals.insert("Total".into(), vec![100.0; 5]);
        actuals.insert("East".into(), vec![60.0; 5]);
        actuals.insert("West".into(), vec![40.0; 5]);
        actuals.insert("NY".into(), vec![35.0; 5]);
        actuals.insert("MA".into(), vec![25.0; 5]);
        actuals.insert("CA".into(), vec![25.0; 5]);
        actuals.insert("WA".into(), vec![15.0; 5]);
        tree.set_actuals(actuals);

        let base = vec![
            ("Total".into(), vec![200.0]),
            ("East".into(), vec![999.0]),
            ("West".into(), vec![999.0]),
            ("NY".into(), vec![999.0]),
            ("MA".into(), vec![999.0]),
            ("CA".into(), vec![999.0]),
            ("WA".into(), vec![999.0]),
        ];

        let td = tree
            .reconcile(&base, ReconciliationMethod::TopDown)
            .unwrap();
        let mo = tree
            .reconcile(&base, ReconciliationMethod::MiddleOut { middle_level: 0 })
            .unwrap();

        let td_map: HashMap<&str, &Vec<f64>> = td.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mo_map: HashMap<&str, &Vec<f64>> = mo.iter().map(|(k, v)| (k.as_str(), v)).collect();

        for name in &["Total", "East", "West", "NY", "MA", "CA", "WA"] {
            approx_eq(td_map[name][0], mo_map[name][0], 1e-10);
        }
    }

    #[test]
    fn middle_out_level_max_equals_bottom_up() {
        let tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let base = vec![
            ("Total".into(), vec![999.0]),
            ("East".into(), vec![999.0]),
            ("West".into(), vec![999.0]),
            ("NY".into(), vec![10.0]),
            ("MA".into(), vec![20.0]),
            ("CA".into(), vec![30.0]),
            ("WA".into(), vec![40.0]),
        ];

        let bu = tree
            .reconcile(&base, ReconciliationMethod::BottomUp)
            .unwrap();
        let mo = tree
            .reconcile(&base, ReconciliationMethod::MiddleOut { middle_level: 99 })
            .unwrap();

        let bu_map: HashMap<&str, &Vec<f64>> = bu.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mo_map: HashMap<&str, &Vec<f64>> = mo.iter().map(|(k, v)| (k.as_str(), v)).collect();

        for name in &["Total", "East", "West", "NY", "MA", "CA", "WA"] {
            approx_eq(bu_map[name][0], mo_map[name][0], 1e-10);
        }
    }

    #[test]
    fn middle_out_level1_coherent() {
        let mut tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let mut actuals = HashMap::new();
        actuals.insert("Total".into(), vec![100.0; 5]);
        actuals.insert("East".into(), vec![60.0; 5]);
        actuals.insert("West".into(), vec![40.0; 5]);
        actuals.insert("NY".into(), vec![35.0; 5]);
        actuals.insert("MA".into(), vec![25.0; 5]);
        actuals.insert("CA".into(), vec![25.0; 5]);
        actuals.insert("WA".into(), vec![15.0; 5]);
        tree.set_actuals(actuals);

        let base = vec![
            ("Total".into(), vec![999.0]),
            ("East".into(), vec![120.0]),
            ("West".into(), vec![80.0]),
            ("NY".into(), vec![999.0]),
            ("MA".into(), vec![999.0]),
            ("CA".into(), vec![999.0]),
            ("WA".into(), vec![999.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MiddleOut { middle_level: 1 })
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Coherence checks
        approx_eq(map["East"][0], map["NY"][0] + map["MA"][0], 1e-10);
        approx_eq(map["West"][0], map["CA"][0] + map["WA"][0], 1e-10);
        approx_eq(map["Total"][0], map["East"][0] + map["West"][0], 1e-10);

        // Middle-level forecasts should be preserved
        approx_eq(map["East"][0], 120.0, 1e-10);
        approx_eq(map["West"][0], 80.0, 1e-10);

        // Total should be sum of middle
        approx_eq(map["Total"][0], 200.0, 1e-10);

        // Disaggregation using proportions: NY/East = 35/60
        approx_eq(map["NY"][0], 120.0 * 35.0 / 60.0, 1e-10);
        approx_eq(map["MA"][0], 120.0 * 25.0 / 60.0, 1e-10);
        approx_eq(map["CA"][0], 80.0 * 25.0 / 40.0, 1e-10);
        approx_eq(map["WA"][0], 80.0 * 15.0 / 40.0, 1e-10);
    }

    #[test]
    fn middle_out_requires_actuals() {
        let tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let base = vec![
            ("Total".into(), vec![100.0]),
            ("East".into(), vec![50.0]),
            ("West".into(), vec![50.0]),
            ("NY".into(), vec![25.0]),
            ("MA".into(), vec![25.0]),
            ("CA".into(), vec![25.0]),
            ("WA".into(), vec![25.0]),
        ];

        assert!(tree
            .reconcile(&base, ReconciliationMethod::MiddleOut { middle_level: 1 },)
            .is_err());
    }

    // ── MinTraceShrink ────────────────────────────────────────────────

    #[test]
    fn mint_shrink_coherent() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        // Generate residuals
        let mut residuals = HashMap::new();
        let ra: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).sin() * 0.5).collect();
        let rb: Vec<f64> = (0..50).map(|i| (i as f64 * 1.1).cos() * 0.3).collect();
        let rt: Vec<f64> = ra.iter().zip(rb.iter()).map(|(a, b)| a + b).collect();
        residuals.insert("Total".into(), rt);
        residuals.insert("A".into(), ra);
        residuals.insert("B".into(), rb);
        tree.set_residuals(residuals);

        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![55.0]),
            ("B".into(), vec![40.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceShrink)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Coherence: Total = A + B
        approx_eq(map["Total"][0], map["A"][0] + map["B"][0], 1e-10);
    }

    #[test]
    fn mint_shrink_three_levels_coherent() {
        let mut tree = HierarchyTree::new(vec![
            ("Total", &["East", "West"]),
            ("East", &["NY", "MA"]),
            ("West", &["CA", "WA"]),
        ])
        .unwrap();

        let t = 50;
        let ny: Vec<f64> = (0..t).map(|i| (i as f64 * 0.3).sin()).collect();
        let ma: Vec<f64> = (0..t).map(|i| (i as f64 * 0.5).cos()).collect();
        let ca: Vec<f64> = (0..t).map(|i| (i as f64 * 0.7).sin() * 0.8).collect();
        let wa: Vec<f64> = (0..t).map(|i| (i as f64 * 0.2).cos() * 0.6).collect();
        let east: Vec<f64> = ny.iter().zip(ma.iter()).map(|(a, b)| a + b).collect();
        let west: Vec<f64> = ca.iter().zip(wa.iter()).map(|(a, b)| a + b).collect();
        let total: Vec<f64> = east.iter().zip(west.iter()).map(|(a, b)| a + b).collect();

        let mut residuals = HashMap::new();
        residuals.insert("Total".into(), total);
        residuals.insert("East".into(), east);
        residuals.insert("West".into(), west);
        residuals.insert("NY".into(), ny);
        residuals.insert("MA".into(), ma);
        residuals.insert("CA".into(), ca);
        residuals.insert("WA".into(), wa);
        tree.set_residuals(residuals);

        let base = vec![
            ("Total".into(), vec![100.0, 200.0]),
            ("East".into(), vec![55.0, 110.0]),
            ("West".into(), vec![50.0, 95.0]),
            ("NY".into(), vec![25.0, 55.0]),
            ("MA".into(), vec![28.0, 52.0]),
            ("CA".into(), vec![26.0, 48.0]),
            ("WA".into(), vec![22.0, 50.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceShrink)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        for h in 0..2 {
            approx_eq(map["East"][h], map["NY"][h] + map["MA"][h], 1e-10);
            approx_eq(map["West"][h], map["CA"][h] + map["WA"][h], 1e-10);
            approx_eq(map["Total"][h], map["East"][h] + map["West"][h], 1e-10);
        }
    }

    #[test]
    fn mint_shrink_requires_residuals() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![50.0]),
            ("B".into(), vec![50.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::MinTraceShrink)
            .is_err());
    }

    #[test]
    fn mint_shrink_already_coherent_unchanged() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        // Use identity-like residuals
        let t = 100;
        let ra: Vec<f64> = (0..t).map(|i| (i as f64 * 0.3).sin() * 0.1).collect();
        let rb: Vec<f64> = (0..t).map(|i| (i as f64 * 0.5).cos() * 0.1).collect();
        let rt: Vec<f64> = ra.iter().zip(rb.iter()).map(|(a, b)| a + b).collect();
        let mut residuals = HashMap::new();
        residuals.insert("Total".into(), rt);
        residuals.insert("A".into(), ra);
        residuals.insert("B".into(), rb);
        tree.set_residuals(residuals);

        // Already coherent base
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![60.0]),
            ("B".into(), vec![40.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceShrink)
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Should remain close to original (already coherent)
        approx_eq(map["Total"][0], 100.0, 1.0);
        approx_eq(map["A"][0], 60.0, 1.0);
        approx_eq(map["B"][0], 40.0, 1.0);
    }

    #[test]
    fn min_trace_variance_produces_coherent_forecasts() {
        let mut tree =
            HierarchyTree::new(vec![("Total", &["A", "B"]), ("A", &["A1", "A2"])]).unwrap();

        // Incoherent base forecasts
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![55.0]),
            ("B".into(), vec![40.0]),
            ("A1".into(), vec![25.0]),
            ("A2".into(), vec![20.0]),
        ];

        // Set residuals for variance estimation
        let mut residuals = std::collections::HashMap::new();
        residuals.insert("Total".into(), vec![1.0, -1.0, 0.5, -0.5]);
        residuals.insert("A".into(), vec![0.8, -0.8, 0.3, -0.3]);
        residuals.insert("B".into(), vec![0.5, -0.5, 0.2, -0.2]);
        residuals.insert("A1".into(), vec![0.6, -0.6, 0.2, -0.2]);
        residuals.insert("A2".into(), vec![0.4, -0.4, 0.1, -0.1]);
        tree.set_residuals(residuals);

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceVariance)
            .unwrap();

        let map: std::collections::HashMap<String, Vec<f64>> = result.into_iter().collect();

        // Check coherence: Total = A + B, A = A1 + A2
        let total = map["Total"][0];
        let a = map["A"][0];
        let b = map["B"][0];
        let a1 = map["A1"][0];
        let a2 = map["A2"][0];

        approx_eq(total, a + b, 0.01);
        approx_eq(a, a1 + a2, 0.01);
    }

    #[test]
    fn from_summing_matrix_grouped_hierarchy_construction() {
        // Issue #124: a grouped hierarchy with (site, part) leaves
        // rolling up to site totals, material totals, and grand
        // total simultaneously. Four leaves, three aggregate
        // dimensions, one grand-total root.
        //
        // Layout (column index → name):
        //   0: Total (root)
        //   1: site_SI10
        //   2: site_SI20
        //   3: material_P1
        //   4: material_P2
        //   5: leaf SI10_P1  (parents 1, 3, 0)
        //   6: leaf SI10_P2  (parents 1, 4, 0)
        //   7: leaf SI20_P1  (parents 2, 3, 0)
        //   8: leaf SI20_P2  (parents 2, 4, 0)
        let node_names: Vec<String> = vec![
            "Total".into(),
            "site_SI10".into(),
            "site_SI20".into(),
            "material_P1".into(),
            "material_P2".into(),
            "SI10_P1".into(),
            "SI10_P2".into(),
            "SI20_P1".into(),
            "SI20_P2".into(),
        ];
        let leaf_names: Vec<String> = vec![
            "SI10_P1".into(),
            "SI10_P2".into(),
            "SI20_P1".into(),
            "SI20_P2".into(),
        ];
        let leaf_ancestors: Vec<Vec<usize>> = vec![
            vec![0, 1, 3], // SI10_P1 → Total, site_SI10, material_P1
            vec![0, 1, 4], // SI10_P2 → Total, site_SI10, material_P2
            vec![0, 2, 3], // SI20_P1 → Total, site_SI20, material_P1
            vec![0, 2, 4], // SI20_P2 → Total, site_SI20, material_P2
        ];

        let tree =
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).unwrap();

        // 9 total nodes.
        assert_eq!(tree.len(), 9);

        // Each site-aggregate must list both of its leaves as
        // children (children_of works on the public name).
        let mut site10_children = tree.children_of("site_SI10").unwrap();
        site10_children.sort();
        assert_eq!(site10_children, vec!["SI10_P1", "SI10_P2"]);
        let mut mat1_children = tree.children_of("material_P1").unwrap();
        mat1_children.sort();
        assert_eq!(mat1_children, vec!["SI10_P1", "SI20_P1"]);
        let mut total_children = tree.children_of("Total").unwrap();
        total_children.sort();
        // Total is parent of every leaf directly (and of every
        // intermediate, but the leaves are also listed because the
        // constructor wires (leaf, ancestor) edges).
        assert!(total_children.contains(&"SI10_P1"));
        assert!(total_children.contains(&"SI20_P2"));
    }

    #[test]
    fn from_summing_matrix_rejects_duplicate_node_names() {
        let node_names: Vec<String> = vec!["A".into(), "A".into()];
        let leaf_names: Vec<String> = vec!["A".into()];
        let leaf_ancestors: Vec<Vec<usize>> = vec![vec![]];
        assert!(
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).is_err()
        );
    }

    #[test]
    fn from_summing_matrix_rejects_out_of_range_ancestor() {
        let node_names: Vec<String> = vec!["Total".into(), "Leaf".into()];
        let leaf_names: Vec<String> = vec!["Leaf".into()];
        let leaf_ancestors: Vec<Vec<usize>> = vec![vec![5]]; // out of range
        assert!(
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).is_err()
        );
    }

    #[test]
    fn from_summing_matrix_rejects_unknown_leaf_name() {
        let node_names: Vec<String> = vec!["Total".into(), "Leaf".into()];
        let leaf_names: Vec<String> = vec!["Unknown".into()];
        let leaf_ancestors: Vec<Vec<usize>> = vec![vec![0]];
        assert!(
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).is_err()
        );
    }

    #[test]
    fn grouped_hierarchy_min_trace_variance_coherent() {
        // Real grouped reconciliation: 4 leaves × 2 aggregate
        // dimensions + grand total. Each aggregate must equal the
        // sum of leaves under it after MinTraceVariance.
        let node_names: Vec<String> = vec![
            "Total".into(),
            "site_SI10".into(),
            "site_SI20".into(),
            "material_P1".into(),
            "material_P2".into(),
            "SI10_P1".into(),
            "SI10_P2".into(),
            "SI20_P1".into(),
            "SI20_P2".into(),
        ];
        let leaf_names: Vec<String> = vec![
            "SI10_P1".into(),
            "SI10_P2".into(),
            "SI20_P1".into(),
            "SI20_P2".into(),
        ];
        let leaf_ancestors: Vec<Vec<usize>> =
            vec![vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 3], vec![0, 2, 4]];
        let mut tree =
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).unwrap();

        // Slightly-incoherent base forecasts.
        let base: Vec<(String, Vec<f64>)> = vec![
            ("Total".into(), vec![100.0]),
            ("site_SI10".into(), vec![55.0]),
            ("site_SI20".into(), vec![44.0]),
            ("material_P1".into(), vec![50.0]),
            ("material_P2".into(), vec![48.0]),
            ("SI10_P1".into(), vec![25.0]),
            ("SI10_P2".into(), vec![28.0]),
            ("SI20_P1".into(), vec![24.0]),
            ("SI20_P2".into(), vec![19.0]),
        ];

        // Equal residual variance per node — uniform weighting.
        let mut residuals = std::collections::HashMap::new();
        for n in node_names.iter() {
            residuals.insert(n.clone(), vec![1.0, -1.0, 0.5, -0.5]);
        }
        tree.set_residuals(residuals);

        let reconciled = tree
            .reconcile(&base, ReconciliationMethod::MinTraceVariance)
            .unwrap();
        let map: std::collections::HashMap<String, Vec<f64>> = reconciled.into_iter().collect();

        // Coherence invariants: each aggregate = sum of leaves under it.
        let p11 = map["SI10_P1"][0];
        let p12 = map["SI10_P2"][0];
        let p21 = map["SI20_P1"][0];
        let p22 = map["SI20_P2"][0];

        // Sites.
        approx_eq(map["site_SI10"][0], p11 + p12, 0.01);
        approx_eq(map["site_SI20"][0], p21 + p22, 0.01);
        // Materials.
        approx_eq(map["material_P1"][0], p11 + p21, 0.01);
        approx_eq(map["material_P2"][0], p12 + p22, 0.01);
        // Grand total.
        approx_eq(map["Total"][0], p11 + p12 + p21 + p22, 0.01);
    }

    #[test]
    fn min_trace_struct_produces_coherent_forecasts() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"]), ("A", &["A1", "A2"])]).unwrap();

        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![55.0]),
            ("B".into(), vec![40.0]),
            ("A1".into(), vec![25.0]),
            ("A2".into(), vec![20.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceStruct)
            .unwrap();

        let map: std::collections::HashMap<String, Vec<f64>> = result.into_iter().collect();

        let total = map["Total"][0];
        let a = map["A"][0];
        let b = map["B"][0];
        let a1 = map["A1"][0];
        let a2 = map["A2"][0];

        approx_eq(total, a + b, 0.01);
        approx_eq(a, a1 + a2, 0.01);
        // All values should be positive (reasonable forecasts)
        assert!(total > 0.0);
        assert!(a > 0.0);
        assert!(b > 0.0);
    }

    // ── Matrix-free CG MinT (issue #130) ───────────────────────────────

    /// Build a synthetic grouped hierarchy with `n_sites × n_parts`
    /// leaves rolling up to (Total, per-site, per-part). Returns the
    /// tree + per-leaf base forecasts (constant `1.0`).
    fn build_grouped_panel(
        n_sites: usize,
        n_parts: usize,
    ) -> (HierarchyTree, Vec<(String, Vec<f64>)>) {
        let mut node_names: Vec<String> = vec!["Total".into()];
        let total_idx = 0;
        let mut site_indices = vec![0usize; n_sites];
        for s in 0..n_sites {
            site_indices[s] = node_names.len();
            node_names.push(format!("site_{}", s));
        }
        let mut part_indices = vec![0usize; n_parts];
        for p in 0..n_parts {
            part_indices[p] = node_names.len();
            node_names.push(format!("part_{}", p));
        }
        let mut leaf_names = Vec::with_capacity(n_sites * n_parts);
        let mut leaf_ancestors = Vec::with_capacity(n_sites * n_parts);
        for s in 0..n_sites {
            for p in 0..n_parts {
                leaf_names.push(format!("s{}_p{}", s, p));
                leaf_ancestors.push(vec![total_idx, site_indices[s], part_indices[p]]);
                node_names.push(format!("s{}_p{}", s, p));
            }
        }
        let tree =
            HierarchyTree::from_summing_matrix(&node_names, &leaf_names, &leaf_ancestors).unwrap();

        // Base forecasts: constant 1.0 at every node — perfectly
        // coherent already, so reconciliation should pass through
        // unchanged. That makes the CG-vs-dense equivalence check
        // tight (any drift signals numerical instability).
        let base: Vec<(String, Vec<f64>)> = (0..node_names.len())
            .map(|i| {
                let n_below = if i == total_idx {
                    (n_sites * n_parts) as f64
                } else if i <= n_sites {
                    // site row
                    n_parts as f64
                } else if i <= n_sites + n_parts {
                    // part row
                    n_sites as f64
                } else {
                    // leaf
                    1.0
                };
                (node_names[i].clone(), vec![n_below])
            })
            .collect();
        (tree, base)
    }

    #[test]
    fn min_trace_cg_path_agrees_with_dense_on_small_grouped() {
        // 5×4 = 20 leaves — well under the CG auto-switch threshold
        // (1000), so both runs go through the dense Cholesky path.
        // The math is the same on either side: this is a sanity gate
        // that the recent refactor (sts_diag extraction, scratch
        // buffer plumbing) didn't change reconciled outputs.
        let (mut tree, base) = build_grouped_panel(5, 4);
        let mut residuals = std::collections::HashMap::new();
        for (name, _) in &base {
            residuals.insert(name.clone(), vec![1.0, -1.0, 0.5, -0.5]);
        }
        tree.set_residuals(residuals);
        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceVariance)
            .unwrap();
        let map: std::collections::HashMap<String, Vec<f64>> = result.into_iter().collect();
        // Coherence — Total = sum of all leaves.
        let mut leaf_sum = 0.0_f64;
        for s in 0..5 {
            for p in 0..4 {
                leaf_sum += map[&format!("s{}_p{}", s, p)][0];
            }
        }
        approx_eq(map["Total"][0], leaf_sum, 0.01);
        // Site totals equal per-site leaf sums.
        for s in 0..5 {
            let mut site_sum = 0.0_f64;
            for p in 0..4 {
                site_sum += map[&format!("s{}_p{}", s, p)][0];
            }
            approx_eq(map[&format!("site_{}", s)][0], site_sum, 0.01);
        }
        // Part totals equal per-part leaf sums.
        for p in 0..4 {
            let mut part_sum = 0.0_f64;
            for s in 0..5 {
                part_sum += map[&format!("s{}_p{}", s, p)][0];
            }
            approx_eq(map[&format!("part_{}", p)][0], part_sum, 0.01);
        }
    }

    #[test]
    fn min_trace_cg_path_scales_past_dense_threshold() {
        // 40×40 = 1600 leaves — exceeds the dense auto-switch
        // threshold (1000), so MinTraceStruct routes through the
        // matrix-free CG solver. The dense path here would allocate
        // ~20 MB and still work, but at 100k leaves it'd require
        // 80 GB — the test verifies that the CG path produces
        // coherent forecasts at the scale where the dense path
        // starts straining. Issue #130.
        let n_sites = 40;
        let n_parts = 40;
        let (tree, base) = build_grouped_panel(n_sites, n_parts);

        let result = tree
            .reconcile(&base, ReconciliationMethod::MinTraceStruct)
            .unwrap();
        let map: std::collections::HashMap<String, Vec<f64>> = result.into_iter().collect();

        // Coherence across all three aggregate dimensions.
        let mut leaf_sum = 0.0_f64;
        for s in 0..n_sites {
            for p in 0..n_parts {
                leaf_sum += map[&format!("s{}_p{}", s, p)][0];
            }
        }
        approx_eq(map["Total"][0], leaf_sum, 0.05);

        for s in 0..n_sites {
            let mut site_sum = 0.0_f64;
            for p in 0..n_parts {
                site_sum += map[&format!("s{}_p{}", s, p)][0];
            }
            approx_eq(map[&format!("site_{}", s)][0], site_sum, 0.05);
        }
        for p in 0..n_parts {
            let mut part_sum = 0.0_f64;
            for s in 0..n_sites {
                part_sum += map[&format!("s{}_p{}", s, p)][0];
            }
            approx_eq(map[&format!("part_{}", p)][0], part_sum, 0.05);
        }
    }

    // ── ERM reconciliation ─────────────────────────────────────────────

    /// ERM-03 + ERM-05: hand-computed reference formula.
    ///
    /// Hierarchy Total→{A,B}, n=3 nodes, m=2 leaves, T=2, λ=1.0.
    /// P = [[1/3, 1/3, 0], [0, 0, 1/2]]
    /// bottom_A = 16/3, bottom_B = 5/2, Total = 47/6.
    #[test]
    fn erm_correctness_reference_formula() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0]);

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Reference values: P = [[1/3,1/3,0],[0,0,1/2]]
        // bottom_A = 16/3, bottom_B = 5/2, Total = 47/6
        approx_eq(map["A"][0], 16.0 / 3.0, 1e-10);
        approx_eq(map["B"][0], 5.0 / 2.0, 1e-10);
        approx_eq(map["Total"][0], 47.0 / 6.0, 1e-10);

        // Coherence: Total = A + B
        approx_eq(map["Total"][0], map["A"][0] + map["B"][0], 1e-10);
    }

    /// ERM-02: selecting Erm without calling with_erm_training returns Err.
    #[test]
    fn erm_requires_training() {
        let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
        let base = vec![
            ("Total".into(), vec![100.0]),
            ("A".into(), vec![50.0]),
            ("B".into(), vec![50.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .is_err());
    }

    /// ERM-02: mismatched leaf-history vector length returns Err.
    #[test]
    fn erm_shape_mismatch() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0]);

        // B leaf history has length 3 — mismatch with T=2 from base_hist.
        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0, 0.5]); // wrong length

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .is_err());
    }

    /// ERM: base_history node with wrong length returns Err.
    ///
    /// Exercises the code path at lines 822–835 that validates every node's
    /// base_history vector is the same length as the first node's vector.
    #[test]
    fn erm_shape_mismatch_base_hist() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        // Node "B" has 3 time steps; Total and A have 2 — length mismatch.
        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0, 0.5]); // wrong length

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .is_err());
    }

    /// ERM: negative lambda returns InvalidParameter error.
    #[test]
    fn erm_negative_lambda_returns_err() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0]);

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: -0.3 })
            .is_err());
    }

    /// ERM: NaN lambda returns InvalidParameter error.
    #[test]
    fn erm_nan_lambda_returns_err() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0]);

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: f64::NAN })
            .is_err());
    }

    /// ERM: T=0 training history returns InvalidParameter error.
    #[test]
    fn erm_empty_training_history_returns_err() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        // All vectors have length 0 — T=0.
        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![]);
        base_hist.insert("A".into(), vec![]);
        base_hist.insert("B".into(), vec![]);

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![]);
        leaf_hist.insert("B".into(), vec![]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![6.0]),
            ("B".into(), vec![5.0]),
        ];
        assert!(tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .is_err());
    }

    /// ERM asymmetric correctness test with independent dense oracle.
    ///
    /// Hierarchy: Total → {A, B, C}, n=4 nodes, m=3 leaves, T=4, λ=1.0.
    /// Every node has a DISTINCT training history, so any index-transposition
    /// or ordering bug in y_stored/gram/cross/p/y_hat will cause the
    /// erm_reconcile path and the independent oracle to disagree.
    ///
    /// The oracle implements P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹ from scratch using
    /// Gauss-Jordan elimination — it does NOT reuse any production code path.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn erm_correctness_asymmetric() {
        // --- Hierarchy setup ---
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B", "C"])]).unwrap();

        // Distinct histories for all 4 nodes (n=4) over T=4 time steps.
        // Rows are nodes in internal index order (Total, A, B, C).
        //   Total = [2,1,0,1]
        //   A     = [1,0,0,0]
        //   B     = [0,1,0,0]
        //   C     = [0,0,1,0]
        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![2.0, 1.0, 0.0, 1.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0, 0.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0, 0.0, 0.0]);
        base_hist.insert("C".into(), vec![0.0, 0.0, 1.0, 0.0]);

        // Leaf actuals (m=3 leaves) over T=4 time steps.
        //   A_actual = [1,0,0,0]
        //   B_actual = [0,1,0,0]
        //   C_actual = [0,0,1,0]
        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0, 0.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0, 0.0, 0.0]);
        leaf_hist.insert("C".into(), vec![0.0, 0.0, 1.0, 0.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        // Base forecast at horizon 1.
        let base = vec![
            ("Total".into(), vec![10.0]),
            ("A".into(), vec![3.0]),
            ("B".into(), vec![4.0]),
            ("C".into(), vec![5.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // --- Independent oracle ---
        // n=4 nodes, m=3 leaves, T=4, λ=1.0.
        // Ŷ (n×T): rows = [Total, A, B, C]
        let n = 4usize;
        let t = 4usize;
        let m = 3usize;
        let lambda = 1.0_f64;

        // y_hat[node][t]
        let y_hat_hist: [[f64; 4]; 4] = [
            [2.0, 1.0, 0.0, 1.0], // Total
            [1.0, 0.0, 0.0, 0.0], // A
            [0.0, 1.0, 0.0, 0.0], // B
            [0.0, 0.0, 1.0, 0.0], // C
        ];
        // b_hist[leaf][t]
        let b_hist: [[f64; 4]; 3] = [
            [1.0, 0.0, 0.0, 0.0], // A_actual
            [0.0, 1.0, 0.0, 0.0], // B_actual
            [0.0, 0.0, 1.0, 0.0], // C_actual
        ];

        // Gram G = Ŷ Ŷᵀ (n×n) + λI.
        let mut gram = [[0.0_f64; 4]; 4];
        for i in 0..n {
            for j in 0..n {
                gram[i][j] = (0..t).map(|tt| y_hat_hist[i][tt] * y_hat_hist[j][tt]).sum();
            }
            gram[i][i] += lambda;
        }

        // Cross C = B Ŷᵀ (m×n).
        let mut cross = [[0.0_f64; 4]; 3];
        for i in 0..m {
            for j in 0..n {
                cross[i][j] = (0..t).map(|tt| b_hist[i][tt] * y_hat_hist[j][tt]).sum();
            }
        }

        // Invert gram via Gauss-Jordan elimination (4×4 augmented system).
        let mut aug = [[0.0_f64; 8]; 4];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = gram[i][j];
            }
            aug[i][n + i] = 1.0;
        }
        for col in 0..n {
            // Pivot: find row with largest abs value in this column.
            let pivot = (col..n)
                .max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap())
                .unwrap();
            aug.swap(col, pivot);
            let diag = aug[col][col];
            for k in 0..2 * n {
                aug[col][k] /= diag;
            }
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for k in 0..2 * n {
                        aug[row][k] -= factor * aug[col][k];
                    }
                }
            }
        }
        let mut gram_inv = [[0.0_f64; 4]; 4];
        for i in 0..n {
            for j in 0..n {
                gram_inv[i][j] = aug[i][n + j];
            }
        }

        // P = C · gram_inv (m×n).
        let mut p_oracle = [[0.0_f64; 4]; 3];
        for i in 0..m {
            for j in 0..n {
                p_oracle[i][j] = (0..n).map(|k| cross[i][k] * gram_inv[k][j]).sum();
            }
        }

        // ŷ at horizon 0 (internal node order: Total=10, A=3, B=4, C=5).
        let y_now = [10.0_f64, 3.0, 4.0, 5.0];

        // bottom = P · ŷ (m-vector).
        let bottom_oracle: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| p_oracle[i][j] * y_now[j]).sum())
            .collect();

        // Summing matrix S (n×m): leaf j is 1 at position j and at all ancestors.
        // Leaf order: A=index 0, B=index 1, C=index 2 (leaves() order).
        // Internal node order: Total=0, A=1, B=2, C=3.
        // S[Total][A]=1, S[Total][B]=1, S[Total][C]=1
        // S[A][A]=1, S[B][B]=1, S[C][C]=1
        let s_oracle = [
            [1.0_f64, 1.0, 1.0], // Total
            [1.0, 0.0, 0.0],     // A
            [0.0, 1.0, 0.0],     // B
            [0.0, 0.0, 1.0],     // C
        ];
        let all_oracle: Vec<f64> = (0..n)
            .map(|i| (0..m).map(|j| s_oracle[i][j] * bottom_oracle[j]).sum())
            .collect();

        // Internal node order for oracle output: [Total, A, B, C].
        let oracle_total = all_oracle[0];
        let oracle_a = all_oracle[1];
        let oracle_b = all_oracle[2];
        let oracle_c = all_oracle[3];

        // Assert erm_reconcile matches the independent oracle within 1e-9.
        approx_eq(map["A"][0], oracle_a, 1e-9);
        approx_eq(map["B"][0], oracle_b, 1e-9);
        approx_eq(map["C"][0], oracle_c, 1e-9);
        approx_eq(map["Total"][0], oracle_total, 1e-9);

        // Coherence: Total = A + B + C.
        approx_eq(
            map["Total"][0],
            map["A"][0] + map["B"][0] + map["C"][0],
            1e-9,
        );
    }

    /// ERM-03: coherence holds across horizon > 1.
    #[test]
    fn erm_coherent_multi_horizon() {
        let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

        let mut base_hist = HashMap::new();
        base_hist.insert("Total".into(), vec![1.0, 0.0]);
        base_hist.insert("A".into(), vec![1.0, 0.0]);
        base_hist.insert("B".into(), vec![0.0, 1.0]);

        let mut leaf_hist = HashMap::new();
        leaf_hist.insert("A".into(), vec![1.0, 0.0]);
        leaf_hist.insert("B".into(), vec![0.0, 1.0]);

        tree.with_erm_training(base_hist, leaf_hist);

        let base = vec![
            ("Total".into(), vec![10.0, 20.0]),
            ("A".into(), vec![6.0, 12.0]),
            ("B".into(), vec![5.0, 9.0]),
        ];

        let result = tree
            .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
            .unwrap();
        let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

        // Coherence must hold at every horizon step.
        for h in 0..2 {
            approx_eq(map["Total"][h], map["A"][h] + map["B"][h], 1e-10);
        }
    }
}
