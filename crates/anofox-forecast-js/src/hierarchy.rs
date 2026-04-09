//! Hierarchical forecasting / reconciliation bindings for JavaScript.
//!
//! Exposes `HierarchyTree` and seven reconciliation methods (BottomUp,
//! TopDown, MiddleOut, MinTraceOls, MinTraceShrink, MinTraceVariance,
//! MinTraceStruct) for coherent multi-level forecasts.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use anofox_forecast::hierarchy::{
    HierarchyTree as InnerHierarchyTree, ReconciliationMethod as InnerMethod,
};

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

fn parse_method(name: &str, middle_level: Option<usize>) -> Result<InnerMethod, JsError> {
    match name {
        "bottomUp" | "BottomUp" | "bottom_up" => Ok(InnerMethod::BottomUp),
        "topDown" | "TopDown" | "top_down" => Ok(InnerMethod::TopDown),
        "middleOut" | "MiddleOut" | "middle_out" => Ok(InnerMethod::MiddleOut {
            middle_level: middle_level.unwrap_or(1),
        }),
        "minTraceOls" | "MinTraceOls" | "min_trace_ols" => Ok(InnerMethod::MinTraceOls),
        "minTraceShrink" | "MinTraceShrink" | "min_trace_shrink" => Ok(InnerMethod::MinTraceShrink),
        "minTraceVariance" | "MinTraceVariance" | "min_trace_variance" => {
            Ok(InnerMethod::MinTraceVariance)
        }
        "minTraceStruct" | "MinTraceStruct" | "min_trace_struct" => Ok(InnerMethod::MinTraceStruct),
        other => Err(JsError::new(&format!(
            "Unknown reconciliation method '{}'. Use: bottomUp, topDown, middleOut, \
             minTraceOls, minTraceShrink, minTraceVariance, minTraceStruct",
            other
        ))),
    }
}

// JS input / output shapes
#[derive(Deserialize)]
struct EdgeJs {
    parent: String,
    children: Vec<String>,
}

#[derive(Deserialize)]
struct ForecastEntryJs {
    name: String,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct ReconciledEntryJs {
    name: String,
    values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// HierarchyTree wrapper
// ---------------------------------------------------------------------------

/// A tree of named nodes with support for coherent forecast reconciliation.
///
/// ```javascript
/// import { HierarchyTree } from '@sipemu/anofox-forecast';
///
/// // total → [A, B]; A → [A1, A2]; B → [B1, B2]
/// const tree = new HierarchyTree([
///     { parent: "total", children: ["A", "B"] },
///     { parent: "A", children: ["A1", "A2"] },
///     { parent: "B", children: ["B1", "B2"] },
/// ]);
///
/// const baseForecasts = [
///     { name: "total", values: [100, 110] },
///     { name: "A",     values: [55, 60] },
///     { name: "B",     values: [50, 55] },
///     { name: "A1",    values: [25, 28] },
///     { name: "A2",    values: [28, 30] },
///     { name: "B1",    values: [22, 25] },
///     { name: "B2",    values: [25, 28] },
/// ];
///
/// const reconciled = tree.reconcile(baseForecasts, "bottomUp");
/// ```
#[wasm_bindgen]
pub struct HierarchyTree {
    inner: InnerHierarchyTree,
}

#[wasm_bindgen]
impl HierarchyTree {
    /// Build a hierarchy from parent → children edges.
    ///
    /// Expects a JS array like `[{parent: "total", children: ["A", "B"]}, ...]`.
    /// The root is inferred as the node that never appears as a child.
    #[wasm_bindgen(constructor)]
    pub fn new(edges: JsValue) -> Result<HierarchyTree, JsError> {
        let parsed: Vec<EdgeJs> =
            serde_wasm_bindgen::from_value(edges).map_err(|e| JsError::new(&e.to_string()))?;

        // Build owned storage for the lifetimes expected by InnerHierarchyTree::new.
        let child_refs: Vec<Vec<&str>> = parsed
            .iter()
            .map(|e| e.children.iter().map(|s| s.as_str()).collect())
            .collect();
        let edge_refs: Vec<(&str, &[&str])> = parsed
            .iter()
            .enumerate()
            .map(|(i, e)| (e.parent.as_str(), child_refs[i].as_slice()))
            .collect();

        let tree = InnerHierarchyTree::new(edge_refs).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner: tree })
    }

    /// Number of nodes in the tree.
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Names of all nodes in BFS (top-down) order.
    #[wasm_bindgen(js_name = nodeNames)]
    pub fn node_names(&self) -> Vec<String> {
        self.inner
            .node_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Provide historical actuals for TopDown / MiddleOut proportions.
    ///
    /// Input: `[{name: "A", values: [...]}, ...]`.
    #[wasm_bindgen(js_name = setActuals)]
    pub fn set_actuals(&mut self, actuals: JsValue) -> Result<(), JsError> {
        let parsed: Vec<ForecastEntryJs> =
            serde_wasm_bindgen::from_value(actuals).map_err(|e| JsError::new(&e.to_string()))?;
        let map: HashMap<String, Vec<f64>> =
            parsed.into_iter().map(|e| (e.name, e.values)).collect();
        self.inner.set_actuals(map);
        Ok(())
    }

    /// Provide historical residuals for MinTraceShrink / MinTraceVariance.
    #[wasm_bindgen(js_name = setResiduals)]
    pub fn set_residuals(&mut self, residuals: JsValue) -> Result<(), JsError> {
        let parsed: Vec<ForecastEntryJs> =
            serde_wasm_bindgen::from_value(residuals).map_err(|e| JsError::new(&e.to_string()))?;
        let map: HashMap<String, Vec<f64>> =
            parsed.into_iter().map(|e| (e.name, e.values)).collect();
        self.inner.set_residuals(map);
        Ok(())
    }

    /// Reconcile base forecasts using the specified method.
    ///
    /// @param baseForecasts - `[{name: string, values: number[]}, ...]` covering every node
    /// @param method - "bottomUp" | "topDown" | "middleOut" | "minTraceOls" | "minTraceShrink" | "minTraceVariance" | "minTraceStruct"
    /// @param middleLevel - Required only for "middleOut" (default: 1)
    /// @returns Array of `{name, values}` entries in BFS order
    pub fn reconcile(
        &self,
        base_forecasts: JsValue,
        method: &str,
        middle_level: Option<usize>,
    ) -> Result<JsValue, JsError> {
        let parsed: Vec<ForecastEntryJs> = serde_wasm_bindgen::from_value(base_forecasts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let base: Vec<(String, Vec<f64>)> =
            parsed.into_iter().map(|e| (e.name, e.values)).collect();
        let m = parse_method(method, middle_level)?;
        let result = self
            .inner
            .reconcile(&base, m)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let out: Vec<ReconciledEntryJs> = result
            .into_iter()
            .map(|(name, values)| ReconciledEntryJs { name, values })
            .collect();
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }
}
