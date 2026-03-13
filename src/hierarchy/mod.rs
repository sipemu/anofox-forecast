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

/// A node in the hierarchy tree.
#[derive(Debug, Clone)]
struct Node {
    name: String,
    children: Vec<usize>, // indices into HierarchyTree::nodes
    parent: Option<usize>,
    level: usize,
}

/// Reconciliation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    MinTraceOls,
    /// MinT with Ledoit-Wolf shrinkage covariance.
    /// Requires `set_residuals()` to have been called.
    /// Uses Σ = α·F + (1−α)·S where F is the diagonal target and S is the
    /// sample covariance, with α chosen by the Ledoit-Wolf formula.
    MinTraceShrink,
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
                        parent: None,
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
                if nodes[cidx].parent.is_some() {
                    return Err(ForecastError::InvalidParameter(format!(
                        "node '{}' has multiple parents",
                        child
                    )));
                }
                nodes[cidx].parent = Some(pidx);
                nodes[pidx].children.push(cidx);
            }
        }

        // Find root (no parent)
        let roots: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| i)
            .collect();

        if roots.len() != 1 {
            return Err(ForecastError::InvalidParameter(format!(
                "hierarchy must have exactly one root, found {}",
                roots.len()
            )));
        }
        let root = roots[0];

        // Assign levels via BFS
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((root, 0usize));
        while let Some((idx, level)) = queue.pop_front() {
            nodes[idx].level = level;
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
        // S[i][j] = 1 if leaf j contributes to node i, else 0
        let mut s = vec![vec![0.0_f64; m]; n];
        for (j, &leaf) in leaves.iter().enumerate() {
            // Walk up from leaf to root, marking all ancestors
            let mut cur = leaf;
            s[cur][j] = 1.0;
            while let Some(parent) = self.nodes[cur].parent {
                s[parent][j] = 1.0;
                cur = parent;
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

    /// Check if `node` is a descendant of `ancestor`.
    fn is_descendant_of(&self, node: usize, ancestor: usize) -> bool {
        let mut cur = node;
        while let Some(parent) = self.nodes[cur].parent {
            if parent == ancestor {
                return true;
            }
            cur = parent;
        }
        false
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

        // Build summing matrix S_mat (n x m)
        let mut s_mat = vec![vec![0.0_f64; m]; n];
        for (j, &leaf) in leaves.iter().enumerate() {
            let mut cur = leaf;
            s_mat[cur][j] = 1.0;
            while let Some(parent) = self.nodes[cur].parent {
                s_mat[parent][j] = 1.0;
                cur = parent;
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

    /// BFS order of node indices.
    fn bfs_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.root);
        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &c in &self.nodes[idx].children {
                queue.push_back(c);
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
}
