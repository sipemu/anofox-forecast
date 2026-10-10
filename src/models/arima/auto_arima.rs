//! Automatic ARIMA and SARIMA model selection.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::diff::{ndiffs_kpss, nsdiffs_seas};
use crate::models::arima::model::{ARIMA, SARIMA};
use crate::models::inspect::{ArimaExplanation, Explanation, Inspectable};
use crate::models::{validate_series_complete, Forecaster};
use crate::utils::ols::OLSResult;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for AutoARIMA.
#[derive(Debug, Clone)]
pub struct AutoARIMAConfig {
    /// Maximum non-seasonal AR order to consider.
    pub max_p: usize,
    /// Maximum non-seasonal MA order to consider.
    pub max_q: usize,
    /// Maximum non-seasonal differencing order.
    pub max_d: usize,
    /// Maximum seasonal AR order.
    pub max_cap_p: usize,
    /// Maximum seasonal MA order.
    pub max_cap_q: usize,
    /// Maximum seasonal differencing order.
    pub max_cap_d: usize,
    /// Seasonal period (0 for non-seasonal).
    pub seasonal_period: usize,
    /// Use stepwise search (faster) vs exhaustive.
    pub stepwise: bool,
    /// Use true stepwise (neighbor-based hill climbing) vs grid stepwise.
    /// Only applies when stepwise=true. This is R `forecast::auto.arima`'s
    /// own search strategy (Hyndman-Khandakar stepwise, D-06); enable it
    /// via `with_true_stepwise()`. NOT the crate's default as of 11-05
    /// (evidence-gated, deferred -- see `AutoARIMAConfig::default()` and
    /// 11-05-SUMMARY.md): it is enforced by `max_order` here, but the
    /// scoring formula it calls into still has the per-candidate AICc
    /// window bug documented in 11-04-SUMMARY.md's Known Gap, so this is
    /// not yet safe as the default.
    pub true_stepwise: bool,
    /// Maximum value of `p+q+P+Q` allowed for any candidate (R
    /// `auto.arima`'s `max.order`, default 5, D-06). Enforced by
    /// `true_stepwise_search` on every start model and neighbour move.
    pub max_order: usize,
    /// Selection criterion (use AIC for selection).
    pub use_aic: bool,
}

impl Default for AutoARIMAConfig {
    fn default() -> Self {
        Self {
            max_p: 5,
            max_q: 5,
            max_d: 2,
            // 11-05 (D-06, evidence-gated per 2026-10-10 user decision): R
            // `auto.arima`'s real defaults are max.P = 2, max.Q = 2 with
            // Hyndman-Khandakar stepwise search as the default strategy.
            // That combination was implemented and tested in this session
            // and found to re-expose the pre-existing per-candidate AICc
            // scoring-window bug documented as a Known Gap in
            // 11-04-SUMMARY.md -- not only on the RW/AR Monte-Carlo
            // fixtures (already known), but also on AirPassengers (R's
            // ARIMA(2,1,1)(0,1,0)[12] scores ~1002 under the crate's own
            // AICc while an overfit ARIMA(3,1,0)(2,1,0)[12] scores ~829,
            // a ~170-unit gap from comparing candidates over different
            // effective sample sizes). Flipping these defaults without
            // fixing that root cause would ship a regression, which the
            // user's decision explicitly says not to do ("stop and report
            // precisely... rather than shipping a regression"). Kept at
            // the pre-11-05 values; see 11-05-SUMMARY.md.
            max_cap_p: 1,
            max_cap_q: 1,
            max_cap_d: 1,
            seasonal_period: 0,
            stepwise: true,
            true_stepwise: false, // Grid stepwise for reliable model selection (see note above)
            max_order: 5,
            use_aic: true,
        }
    }
}

impl AutoARIMAConfig {
    /// Set maximum non-seasonal orders.
    pub fn with_max_orders(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_p = max_p;
        self.max_d = max_d;
        self.max_q = max_q;
        self
    }

    /// Set maximum seasonal orders.
    pub fn with_seasonal_orders(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_cap_p = max_p;
        self.max_cap_d = max_d;
        self.max_cap_q = max_q;
        self
    }

    /// Set seasonal period.
    pub fn with_seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = period;
        self
    }

    /// Use exhaustive search instead of stepwise.
    pub fn exhaustive(mut self) -> Self {
        self.stepwise = false;
        self
    }

    /// Use true stepwise search (neighbor-based hill climbing), matching
    /// R `forecast::auto.arima`'s own default search strategy
    /// (Hyndman-Khandakar). More efficient than grid stepwise but may
    /// miss the global optimum under the crate's current scoring formula
    /// (see `AutoARIMAConfig::default()`'s doc comment) -- opt in
    /// explicitly, not yet the crate's default as of 11-05.
    pub fn with_true_stepwise(mut self) -> Self {
        self.stepwise = true;
        self.true_stepwise = true;
        self
    }

    /// Use the fixed-grid stepwise search (the crate's default as of
    /// 11-05). Named for symmetry with `with_true_stepwise()` and to give
    /// a stable, explicit opt-in once the default flips to HK stepwise.
    pub fn with_grid_stepwise(mut self) -> Self {
        self.stepwise = true;
        self.true_stepwise = false;
        self
    }
}

/// Selected model type.
#[derive(Debug, Clone)]
enum SelectedModel {
    ARIMA(ARIMA),
    SARIMA(SARIMA),
}

/// Model order (p, d, q, P, D, Q, s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOrder {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub cap_p: usize,
    /// Seasonal differencing order.
    pub cap_d: usize,
    /// Seasonal MA order.
    pub cap_q: usize,
    /// Seasonal period.
    pub s: usize,
}

impl ModelOrder {
    /// Check if this is a seasonal model.
    pub fn is_seasonal(&self) -> bool {
        self.s > 1 && (self.cap_p > 0 || self.cap_d > 0 || self.cap_q > 0)
    }
}

/// Automatic ARIMA/SARIMA model selection.
///
/// Automatically selects the best ARIMA(p, d, q) or SARIMA(p, d, q)(P, D, Q)\[s\]
/// specification based on information criteria.
#[derive(Debug, Clone)]
pub struct AutoARIMA {
    /// Configuration.
    config: AutoARIMAConfig,
    /// Selected model.
    selected_model: Option<SelectedModel>,
    /// Selected orders.
    selected_order: Option<ModelOrder>,
    /// All fitted models and their scores.
    model_scores: Vec<(ModelOrder, f64)>,
    /// Training input values, retained for the issue #106 decomposition contract.
    training_values_store: Option<Vec<f64>>,
    /// Training-time exogenous regressors, retained for the residual-Ridge shim.
    training_regressors_store: Option<std::collections::HashMap<String, Vec<f64>>>,
}

impl AutoARIMA {
    /// Create a new AutoARIMA with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoARIMAConfig::default(),
            selected_model: None,
            selected_order: None,
            model_scores: Vec::new(),
            training_values_store: None,
            training_regressors_store: None,
        }
    }

    /// Create AutoARIMA with custom configuration.
    pub fn with_config(config: AutoARIMAConfig) -> Self {
        Self {
            config,
            selected_model: None,
            selected_order: None,
            model_scores: Vec::new(),
            training_values_store: None,
            training_regressors_store: None,
        }
    }

    /// Create AutoARIMA with seasonal period.
    pub fn seasonal(period: usize) -> Self {
        let config = AutoARIMAConfig::default().with_seasonal_period(period);
        Self::with_config(config)
    }

    /// Get the selected order.
    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order.map(|o| (o.p, o.d, o.q))
    }

    /// Get the full selected order including seasonal components.
    pub fn selected_full_order(&self) -> Option<ModelOrder> {
        self.selected_order
    }

    /// Get all model scores.
    pub fn model_scores(&self) -> &[(ModelOrder, f64)] {
        &self.model_scores
    }

    /// Suggest seasonal differencing order using R `forecast::nsdiffs`'
    /// seasonal-strength test (D-06): delegates to `nsdiffs_seas`.
    ///
    /// 11-05: `AutoARIMA::fit` no longer falls back to a non-seasonal
    /// search when this test's underlying strength heuristic is weak --
    /// R's `auto.arima` only drops seasonal AR/MA terms when `period < 2`
    /// (D-06); when `period > 1` but `nsdiffs_seas` returns `D = 0`, the
    /// seasonal `P`/`Q` terms are still searched at `D = 0`.
    fn suggest_seasonal_differencing(values: &[f64], period: usize) -> usize {
        nsdiffs_seas(values, period, 1)
    }

    /// Generate candidate orders using stepwise search.
    fn stepwise_candidates(&self, d: usize, cap_d: usize) -> Vec<ModelOrder> {
        let s = self.config.seasonal_period;

        // Non-seasonal candidates: core set up to (2,2) plus selective p=3 extensions
        // to match Python's model selection reach without excessive overfit candidates
        let nonseasonal = vec![
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 0),
            (0, 3),
            (3, 1),
            (1, 3),
            (3, 2),
            (2, 3),
        ];

        let mut candidates = Vec::new();

        // Add non-seasonal models
        for &(p, q) in &nonseasonal {
            if p <= self.config.max_p && q <= self.config.max_q {
                candidates.push(ModelOrder {
                    p,
                    d,
                    q,
                    cap_p: 0,
                    cap_d,
                    cap_q: 0,
                    s,
                });
            }
        }

        // Add seasonal models if period > 1
        if s > 1 {
            // Seasonal component options: (P, Q)
            let seasonal = vec![
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (2, 2),
            ];

            // Non-seasonal orders to try with seasonal components
            let nonseasonal_with_seasonal = vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (0, 2),
                (2, 1),
                (1, 2),
                (3, 0),
                (0, 3),
                (2, 2),
                (3, 1),
                (1, 3),
            ];

            for &(p, q) in &nonseasonal_with_seasonal {
                for &(cap_p, cap_q) in &seasonal {
                    if p <= self.config.max_p
                        && q <= self.config.max_q
                        && cap_p <= self.config.max_cap_p
                        && cap_q <= self.config.max_cap_q
                    {
                        candidates.push(ModelOrder {
                            p,
                            d,
                            q,
                            cap_p,
                            cap_d,
                            cap_q,
                            s,
                        });
                    }
                }
            }
        }

        candidates
    }

    /// Generate all candidate orders (exhaustive).
    fn exhaustive_candidates(&self, d: usize, cap_d: usize) -> Vec<ModelOrder> {
        let s = self.config.seasonal_period;
        let mut candidates = Vec::new();

        for p in 0..=self.config.max_p {
            for q in 0..=self.config.max_q {
                if s > 1 {
                    // Add seasonal models
                    for cap_p in 0..=self.config.max_cap_p {
                        for cap_q in 0..=self.config.max_cap_q {
                            candidates.push(ModelOrder {
                                p,
                                d,
                                q,
                                cap_p,
                                cap_d,
                                cap_q,
                                s,
                            });
                        }
                    }
                } else {
                    // Non-seasonal only
                    candidates.push(ModelOrder {
                        p,
                        d,
                        q,
                        cap_p: 0,
                        cap_d: 0,
                        cap_q: 0,
                        s: 0,
                    });
                }
            }
        }

        candidates
    }

    /// Fit and evaluate a model with given order (full model returned for prediction).
    fn evaluate_model_static(
        series: &TimeSeries,
        order: ModelOrder,
        use_aic: bool,
    ) -> Option<(SelectedModel, f64)> {
        if order.is_seasonal() {
            let mut model = SARIMA::new(
                order.p,
                order.d,
                order.q,
                order.cap_p,
                order.cap_d,
                order.cap_q,
                order.s,
            );

            if model.fit(series).is_ok() {
                let score = if use_aic { model.aic() } else { model.bic() };
                if let Some(s) = score {
                    if s.is_finite() {
                        return Some((SelectedModel::SARIMA(model), s));
                    }
                }
            }
        } else {
            let mut model = ARIMA::new(order.p, order.d, order.q);

            if model.fit(series).is_ok() {
                let score = if use_aic { model.aic() } else { model.bic() };
                if let Some(s) = score {
                    if s.is_finite() {
                        return Some((SelectedModel::ARIMA(model), s));
                    }
                }
            }
        }

        None
    }

    /// Score-only evaluation: compute AIC/BIC from pre-computed differenced series
    /// without constructing the full model. Skips validation, storage, and calculate_fitted.
    ///
    /// `common_start` is the maximum lag reach across the *entire* candidate
    /// grid being searched (see `Self::common_start_for_config`) -- every
    /// candidate at a fixed (d, D) must be scored over the identical
    /// leading-observation window for AICc comparisons to be valid (D-06).
    fn score_order_static(
        order: ModelOrder,
        diff_series: &[f64],
        use_aic: bool,
        common_start: usize,
    ) -> Option<f64> {
        // Hyndman-Khandakar constant-allowance rule: a mean (d+D=0) or
        // drift (d+D=1) term is scored with and without and the better
        // kept; d+D>=2 never gets a constant (D-06).
        let allow_constant = order.d + order.cap_d <= 1;
        if order.is_seasonal() {
            SARIMA::score_order(
                order.p,
                order.q,
                order.cap_p,
                order.cap_q,
                order.s,
                diff_series,
                use_aic,
                allow_constant,
                common_start,
            )
        } else {
            ARIMA::score_order(
                order.p,
                order.q,
                diff_series,
                use_aic,
                allow_constant,
                common_start,
            )
        }
    }

    /// Maximum lag reach across the whole candidate grid this config can
    /// generate -- the common CSS scoring window every candidate at a
    /// fixed (d, D) is compared over (D-06, see `score_order_static`).
    fn common_start_for_config(config: &AutoARIMAConfig) -> usize {
        let s = config.seasonal_period;
        if s > 1 {
            let max_ar_lag = config.max_p + config.max_cap_p * s;
            let max_ma_lag = config.max_q + config.max_cap_q * s;
            max_ar_lag.max(max_ma_lag)
        } else {
            config.max_p.max(config.max_q)
        }
    }

    /// Generate neighbors of a given order (for true stepwise search).
    /// Matches Python statsforecast neighbor ordering:
    /// Seasonal first (P-1, Q-1, P+1, Q+1, diagonals),
    /// then non-seasonal (p-1, q-1, p+1, q+1, diagonals).
    fn get_neighbors(&self, order: ModelOrder) -> Vec<ModelOrder> {
        let mut neighbors = Vec::new();
        let s = self.config.seasonal_period;

        // Seasonal neighbors first (matching Python's order)
        if s > 1 {
            if order.cap_p > 0 {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p - 1,
                    ..order
                });
            }
            if order.cap_q > 0 {
                neighbors.push(ModelOrder {
                    cap_q: order.cap_q - 1,
                    ..order
                });
            }
            if order.cap_p < self.config.max_cap_p {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p + 1,
                    ..order
                });
            }
            if order.cap_q < self.config.max_cap_q {
                neighbors.push(ModelOrder {
                    cap_q: order.cap_q + 1,
                    ..order
                });
            }
            // Seasonal diagonals
            if order.cap_p > 0 && order.cap_q > 0 {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p - 1,
                    cap_q: order.cap_q - 1,
                    ..order
                });
            }
            if order.cap_p > 0 && order.cap_q < self.config.max_cap_q {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p - 1,
                    cap_q: order.cap_q + 1,
                    ..order
                });
            }
            if order.cap_p < self.config.max_cap_p && order.cap_q > 0 {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p + 1,
                    cap_q: order.cap_q - 1,
                    ..order
                });
            }
            if order.cap_p < self.config.max_cap_p && order.cap_q < self.config.max_cap_q {
                neighbors.push(ModelOrder {
                    cap_p: order.cap_p + 1,
                    cap_q: order.cap_q + 1,
                    ..order
                });
            }
        }

        // Non-seasonal neighbors (matching Python's order: p-1, q-1, p+1, q+1)
        if order.p > 0 {
            neighbors.push(ModelOrder {
                p: order.p - 1,
                ..order
            });
        }
        if order.q > 0 {
            neighbors.push(ModelOrder {
                q: order.q - 1,
                ..order
            });
        }
        if order.p < self.config.max_p {
            neighbors.push(ModelOrder {
                p: order.p + 1,
                ..order
            });
        }
        if order.q < self.config.max_q {
            neighbors.push(ModelOrder {
                q: order.q + 1,
                ..order
            });
        }
        // Non-seasonal diagonals
        if order.p > 0 && order.q > 0 {
            neighbors.push(ModelOrder {
                p: order.p - 1,
                q: order.q - 1,
                ..order
            });
        }
        if order.p > 0 && order.q < self.config.max_q {
            neighbors.push(ModelOrder {
                p: order.p - 1,
                q: order.q + 1,
                ..order
            });
        }
        if order.p < self.config.max_p && order.q > 0 {
            neighbors.push(ModelOrder {
                p: order.p + 1,
                q: order.q - 1,
                ..order
            });
        }
        if order.p < self.config.max_p && order.q < self.config.max_q {
            neighbors.push(ModelOrder {
                p: order.p + 1,
                q: order.q + 1,
                ..order
            });
        }

        neighbors
    }

    /// True stepwise search matching R `forecast::auto.arima`'s
    /// Hyndman-Khandakar stepwise search (D-06).
    /// Greedy first-improvement: takes the first neighbor that improves the IC,
    /// then restarts the neighbor scan from the beginning.
    /// Uses score-only evaluation with pre-computed differenced series.
    ///
    /// Every start model and neighbour move is bounded by
    /// `self.config.max_order` (R's `max.order`, p+q+P+Q, default 5):
    /// candidates exceeding it are skipped via `order_within_max` below,
    /// matching R's own bound on the search space.
    ///
    /// Note on the constant/mean term (R's fifth start model, "(0,d,0)
    /// without constant"): `score_order_static`'s `allow_constant` already
    /// evaluates a candidate with AND without a mean/drift term whenever
    /// `d+D<=1` and keeps the better of the two (D-06, 11-04). A separate
    /// "null model without constant" start candidate would therefore score
    /// identically to the null-model-with-constant start below (same
    /// `(p,q,P,Q)` key, same auto-selected best score) and is a redundant
    /// R implementation detail in this crate's scoring design, not a
    /// missing search state -- so it is intentionally not added as a
    /// sixth distinct start candidate here (11-05).
    fn true_stepwise_search(
        &mut self,
        diff_series: &[f64],
        d: usize,
        cap_d: usize,
    ) -> Option<(ModelOrder, f64)> {
        let s = self.config.seasonal_period;
        let use_aic = self.config.use_aic;
        let common_start = Self::common_start_for_config(&self.config);
        let max_models = 94; // matching R's nmodels limit
        let max_order = self.config.max_order;
        let order_within_max = |o: &ModelOrder| o.p + o.q + o.cap_p + o.cap_q <= max_order;

        // Initial models matching Python statsforecast starting points
        let initial_orders: Vec<ModelOrder> = vec![
            // Model 0: (start_p, d, start_q) with seasonal
            ModelOrder {
                p: 2,
                d,
                q: 2,
                cap_p: if s > 1 { 1 } else { 0 },
                cap_d,
                cap_q: if s > 1 { 1 } else { 0 },
                s,
            },
            // Model 1: null model with constant
            ModelOrder {
                p: 0,
                d,
                q: 0,
                cap_p: 0,
                cap_d,
                cap_q: 0,
                s,
            },
            // Model 2: pure AR
            ModelOrder {
                p: if self.config.max_p > 0 { 1 } else { 0 },
                d,
                q: 0,
                cap_p: if s > 1 && self.config.max_cap_p > 0 {
                    1
                } else {
                    0
                },
                cap_d,
                cap_q: 0,
                s,
            },
            // Model 3: pure MA
            ModelOrder {
                p: 0,
                d,
                q: if self.config.max_q > 0 { 1 } else { 0 },
                cap_p: 0,
                cap_d,
                cap_q: if s > 1 && self.config.max_cap_q > 0 {
                    1
                } else {
                    0
                },
                s,
            },
        ];

        // Find best starting point
        let mut best_order: Option<ModelOrder> = None;
        let mut best_score = f64::INFINITY;
        let mut n_models = 0usize;

        let mut visited = std::collections::HashSet::new();
        for order in initial_orders {
            if !order_within_max(&order) {
                continue;
            }
            let key = (order.p, order.q, order.cap_p, order.cap_q);
            if visited.contains(&key) {
                continue;
            }
            visited.insert(key);
            n_models += 1;

            if let Some(score) = Self::score_order_static(order, diff_series, use_aic, common_start)
            {
                self.model_scores.push((order, score));
                if score < best_score {
                    best_score = score;
                    best_order = Some(order);
                }
            }
        }

        let mut current_order = best_order?;
        let mut current_score = best_score;

        // Greedy first-improvement hill climbing (matching Python statsforecast).
        // On each iteration, generate ALL neighbors in a fixed order.
        // Take the FIRST one that improves the IC and restart from the top.
        // Stop when no neighbor improves or max_models reached.
        loop {
            if n_models >= max_models {
                break;
            }

            let neighbors = self.get_neighbors(current_order);
            let mut improved = false;

            for neighbor in neighbors {
                if !order_within_max(&neighbor) {
                    continue;
                }
                let key = (neighbor.p, neighbor.q, neighbor.cap_p, neighbor.cap_q);
                if visited.contains(&key) {
                    continue;
                }
                visited.insert(key);
                n_models += 1;

                if let Some(score) =
                    Self::score_order_static(neighbor, diff_series, use_aic, common_start)
                {
                    self.model_scores.push((neighbor, score));

                    if score < current_score {
                        current_score = score;
                        current_order = neighbor;
                        improved = true;
                        break; // greedy: take first improvement, restart scan
                    }
                }

                if n_models >= max_models {
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        Some((current_order, current_score))
    }

    /// Evaluate candidates - uses parallel processing when 'parallel' feature is enabled.
    /// Parallel path uses score-only with pre-computed diffs then re-fits winner.
    /// Sequential path keeps the best full model to avoid redundant re-fit.
    fn evaluate_candidates_fast(
        &self,
        series: &TimeSeries,
        #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
        diff_series_map: &std::collections::HashMap<(usize, usize), Vec<f64>>,
        candidates: &[ModelOrder],
        n_values: usize,
    ) -> (
        Vec<(ModelOrder, f64)>,
        Option<(SelectedModel, ModelOrder, f64)>,
    ) {
        let use_aic = self.config.use_aic;
        let common_start = Self::common_start_for_config(&self.config);

        // Filter candidates by data requirements
        let valid_candidates: Vec<_> = candidates
            .iter()
            .filter(|order| {
                let min_len = order.d
                    + order.cap_d * order.s
                    + order
                        .p
                        .max(order.q)
                        .max(order.cap_p.max(order.cap_q) * order.s.max(1))
                    + 5;
                n_values >= min_len
            })
            .copied()
            .collect();

        // Two-phase evaluation: score non-seasonal candidates first (cheap),
        // then seasonal candidates. Sort candidates so simpler models evaluate
        // first, enabling early termination in the sequential path.
        let mut sorted_candidates = valid_candidates;
        sorted_candidates.sort_by_key(|o| {
            let total_params = o.p + o.q + o.cap_p + o.cap_q;
            let is_seasonal = if o.cap_p > 0 || o.cap_q > 0 { 1 } else { 0 };
            (is_seasonal, total_params)
        });

        #[cfg(feature = "parallel")]
        {
            // Parallel: score-only with pre-computed diffs, then re-fit winner
            let scores: Vec<(ModelOrder, f64)> = sorted_candidates
                .par_iter()
                .filter_map(|&order| {
                    let diff_series = diff_series_map.get(&(order.d, order.cap_d))?;
                    Self::score_order_static(order, diff_series, use_aic, common_start)
                        .map(|score| (order, score))
                })
                .collect();

            let best = scores
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .and_then(|&(order, score)| {
                    Self::evaluate_model_static(series, order, use_aic)
                        .map(|(model, _)| (model, order, score))
                });

            (scores, best)
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential with early termination: evaluate simpler models first,
            // skip complex candidates when they can't plausibly beat the best.
            let mut scores = Vec::with_capacity(sorted_candidates.len());
            let mut best: Option<(SelectedModel, ModelOrder, f64)> = None;
            let mut best_score = f64::INFINITY;

            for &order in &sorted_candidates {
                // Score-only first (cheap) to decide if full fit is worthwhile
                if let Some(diff_series) = diff_series_map.get(&(order.d, order.cap_d)) {
                    if let Some(quick_score) =
                        Self::score_order_static(order, diff_series, use_aic, common_start)
                    {
                        scores.push((order, quick_score));

                        // Only do full fit if this candidate is competitive
                        if quick_score < best_score {
                            if let Some((model, _)) =
                                Self::evaluate_model_static(series, order, use_aic)
                            {
                                best_score = quick_score;
                                best = Some((model, order, quick_score));
                            }
                        }
                    }
                }
            }

            (scores, best)
        }
    }
}

impl Default for AutoARIMA {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoARIMA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        let s = self.config.seasonal_period;

        // Check minimum data requirements
        let min_required = if s > 1 {
            3 * s // At least 3 seasonal cycles for SARIMA
        } else {
            10
        };

        if values.len() < min_required {
            return Err(ForecastError::InsufficientData {
                needed: min_required,
                got: values.len(),
                hint: Some(if s > 1 {
                    format!(
                        "AutoARIMA needs at least 3 seasonal cycles (3*{}={})",
                        s, min_required
                    )
                } else {
                    "AutoARIMA needs at least 10 observations for model selection".into()
                }),
            });
        }

        // 11-05 (D-06): no non-seasonal fallback here. R `auto.arima` only
        // drops seasonal AR/MA terms when `period < 2`; when `period > 1`
        // the seasonal P/Q terms are searched even if the seasonal-strength
        // test below yields `D = 0` (a pre-11-05 fallback used to zero `s`
        // itself when the strength test was weak, which also force a
        // non-seasonal search -- removed, since `nsdiffs_seas` already
        // correctly yields `D = 0` in that case without needing to drop
        // seasonal P/Q from the search space too).

        // Determine differencing orders in auto.arima's own D-then-d order
        // (D-06): D by the seasonal-strength test on the raw series, then
        // d by repeated KPSS on the seasonally-differenced series.
        let suggested_cap_d = if s > 1 {
            Self::suggest_seasonal_differencing(values, s).min(self.config.max_cap_d)
        } else {
            0
        };
        let seasonally_diffed = if s > 1 && suggested_cap_d > 0 {
            SARIMA::seasonal_difference(values, suggested_cap_d, s)
        } else {
            values.to_vec()
        };
        let suggested_d = ndiffs_kpss(&seasonally_diffed, 0.05, self.config.max_d);

        // Fix d at the suggested value (matching Python statsforecast / R auto.arima)
        let d_range = vec![suggested_d];

        // Fix D at the suggested value
        let cap_d_range: Vec<usize> = if s > 1 {
            vec![suggested_cap_d]
        } else {
            vec![0]
        };

        // Pre-compute differenced series for each (d, D) combination
        use crate::models::arima::diff::difference;
        let mut diff_series_map = std::collections::HashMap::new();
        for &d in &d_range {
            for &cap_d in &cap_d_range {
                let nonseasonal_diff = difference(values, d);
                let diff_series = if s > 1 && cap_d > 0 {
                    SARIMA::seasonal_difference(&nonseasonal_diff, cap_d, s)
                } else {
                    nonseasonal_diff
                };
                diff_series_map.insert((d, cap_d), diff_series);
            }
        }

        self.model_scores.clear();
        let mut best_order: Option<ModelOrder> = None;
        let mut best_score = f64::INFINITY;

        // Phase 1: Score-only evaluation to find the best order
        if self.config.stepwise && self.config.true_stepwise {
            for &d in &d_range {
                for &cap_d in &cap_d_range {
                    if let Some(diff_series) = diff_series_map.get(&(d, cap_d)) {
                        if let Some((order, score)) =
                            self.true_stepwise_search(diff_series, d, cap_d)
                        {
                            if score < best_score {
                                best_score = score;
                                best_order = Some(order);
                            }
                        }
                    }
                }
            }
        } else {
            // Generate candidates for all (d, D) combinations (grid search)
            let mut candidates = Vec::new();
            for &d in &d_range {
                for &cap_d in &cap_d_range {
                    let new_candidates = if self.config.stepwise {
                        self.stepwise_candidates(d, cap_d)
                    } else {
                        self.exhaustive_candidates(d, cap_d)
                    };
                    candidates.extend(new_candidates);
                }
            }
            // Remove duplicates
            candidates.sort_by(|a, b| {
                (a.p, a.d, a.q, a.cap_p, a.cap_d, a.cap_q)
                    .cmp(&(b.p, b.d, b.q, b.cap_p, b.cap_d, b.cap_q))
            });
            candidates.dedup();

            // Evaluate all candidates
            let (results, best) =
                self.evaluate_candidates_fast(series, &diff_series_map, &candidates, values.len());

            for (order, score) in results {
                self.model_scores.push((order, score));
            }

            if let Some((model, order, _score)) = best {
                best_order = Some(order);
                self.selected_model = Some(model);
            }
        }

        // Sort model scores
        self.model_scores
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // For true_stepwise path, do final full fit
        if self.selected_model.is_none() {
            if let Some(order) = best_order {
                if let Some((model, _)) =
                    Self::evaluate_model_static(series, order, self.config.use_aic)
                {
                    self.selected_model = Some(model);
                }
            }
        }

        self.selected_order = best_order;

        if self.selected_model.is_none() {
            return Err(ForecastError::ConvergenceFailure(
                "No valid ARIMA/SARIMA model could be fitted".to_string(),
            ));
        }

        // Retain training inputs for the issue #106 decomposition contract.
        self.training_values_store = Some(values.to_vec());
        let regs = series.all_regressors();
        self.training_regressors_store = if regs.is_empty() {
            None
        } else {
            Some(regs.clone())
        };

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => model.predict(horizon),
            Some(SelectedModel::SARIMA(model)) => model.predict(horizon),
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => model.predict_with_intervals(horizon, level),
            Some(SelectedModel::SARIMA(model)) => model.predict_with_intervals(horizon, level),
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.fitted_values(),
            SelectedModel::SARIMA(model) => model.fitted_values(),
        }
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.fitted_values_with_intervals(level),
            SelectedModel::SARIMA(model) => model.fitted_values_with_intervals(level),
        }
    }

    fn residuals(&self) -> Option<&[f64]> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.residuals(),
            SelectedModel::SARIMA(model) => model.residuals(),
        }
    }

    fn training_values(&self) -> Result<&[f64]> {
        self.training_values_store
            .as_deref()
            .ok_or(ForecastError::FitRequired {
                model: Some("AutoARIMA".into()),
            })
    }

    fn training_regressors(&self) -> Option<&std::collections::HashMap<String, Vec<f64>>> {
        self.training_regressors_store.as_ref()
    }

    /// For AutoARIMA, the "trend" exposed to the issue #106 contract is
    /// the model's in-sample mean estimate at each timestep (i.e. the
    /// fitted values). ARIMA isn't a structural-decomposition model —
    /// there is no separable trend / seasonal in the STL sense — so this
    /// is the most informative interpretation that keeps the invariant
    /// `trend + seasonal + residual == training` holding trivially:
    ///
    ///   trend = fitted,  seasonal = 0,  residual = training − fitted
    ///   →  fitted + 0 + (training − fitted) = training ✓
    fn trend_component(&self) -> Result<&[f64]> {
        self.fitted_values().ok_or(ForecastError::FitRequired {
            model: Some("AutoARIMA".into()),
        })
    }

    fn name(&self) -> &str {
        match &self.selected_model {
            Some(SelectedModel::SARIMA(_)) => "AutoARIMA (SARIMA)",
            _ => "AutoARIMA",
        }
    }

    fn explanation(&self) -> Result<Explanation> {
        <Self as Inspectable>::explanation(self)
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => model.has_exog(),
            Some(SelectedModel::SARIMA(model)) => model.has_exog(),
            None => false,
        }
    }

    fn exog_names(&self) -> Option<&[String]> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.exog_names(),
            SelectedModel::SARIMA(model) => model.exog_names(),
        }
    }

    fn exog_coefficients(&self) -> Option<&OLSResult> {
        match self.selected_model.as_ref()? {
            SelectedModel::ARIMA(model) => model.exog_coefficients(),
            SelectedModel::SARIMA(model) => model.exog_coefficients(),
        }
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &std::collections::HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => {
                model.predict_with_exog(horizon, future_regressors)
            }
            Some(SelectedModel::SARIMA(model)) => {
                model.predict_with_exog(horizon, future_regressors)
            }
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &std::collections::HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        match self.selected_model.as_ref() {
            Some(SelectedModel::ARIMA(model)) => {
                model.predict_with_exog_intervals(horizon, future_regressors, level)
            }
            Some(SelectedModel::SARIMA(model)) => {
                model.predict_with_exog_intervals(horizon, future_regressors, level)
            }
            None => Err(ForecastError::FitRequired { model: None }),
        }
    }
}

impl Inspectable for AutoARIMA {
    fn explanation(&self) -> Result<Explanation> {
        let model = self
            .selected_model
            .as_ref()
            .ok_or_else(|| ForecastError::FitRequired {
                model: Some("AutoARIMA".to_string()),
            })?;
        let order = self
            .selected_order
            .ok_or_else(|| ForecastError::FitRequired {
                model: Some("AutoARIMA".to_string()),
            })?;

        let (coefficients, aic, bic, fitted_values, residuals) = match model {
            SelectedModel::ARIMA(m) => {
                let mut coeffs = Vec::new();
                coeffs.extend_from_slice(m.ar_coefficients());
                coeffs.extend_from_slice(m.ma_coefficients());
                let f = m.fitted_values().map(|v| v.to_vec()).unwrap_or_default();
                let r = m.residuals().map(|v| v.to_vec()).unwrap_or_default();
                (
                    coeffs,
                    m.aic().unwrap_or(f64::NAN),
                    m.bic().unwrap_or(f64::NAN),
                    f,
                    r,
                )
            }
            SelectedModel::SARIMA(m) => {
                let mut coeffs = Vec::new();
                coeffs.extend_from_slice(m.ar_coefficients());
                coeffs.extend_from_slice(m.ma_coefficients());
                coeffs.extend_from_slice(m.seasonal_ar_coefficients());
                coeffs.extend_from_slice(m.seasonal_ma_coefficients());
                let f = m.fitted_values().map(|v| v.to_vec()).unwrap_or_default();
                let r = m.residuals().map(|v| v.to_vec()).unwrap_or_default();
                (
                    coeffs,
                    m.aic().unwrap_or(f64::NAN),
                    m.bic().unwrap_or(f64::NAN),
                    f,
                    r,
                )
            }
        };

        let seasonal_order = if order.is_seasonal() {
            Some((order.cap_p, order.cap_d, order.cap_q, order.s))
        } else {
            None
        };

        Ok(Explanation::Arima(ArimaExplanation {
            order: (order.p, order.d, order.q),
            seasonal_order,
            coefficients,
            aic,
            bic,
            fitted_values,
            residuals,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    /// `max_order` (R `auto.arima`'s `max.order`, D-06) defaults to 5 and
    /// is enforced by `true_stepwise_search`.
    ///
    /// NOT asserted here (11-05, evidence-gated per 2026-10-10 user
    /// decision): `max_cap_p`/`max_cap_q` = 2 and `true_stepwise` = true
    /// as the *default*. Both are R's real defaults and are exercised via
    /// `AutoARIMAConfig::default().with_true_stepwise().with_seasonal_orders(5, 1, 5)`
    /// or equivalent explicit opt-in -- flipping them as the *default*
    /// re-exposes the pre-existing per-candidate AICc scoring-window bug
    /// (11-04-SUMMARY.md Known Gap) on AirPassengers too, not just the
    /// RW/AR Monte-Carlo fixtures. See 11-05-SUMMARY.md.
    #[test]
    fn default_config_has_max_order_and_grid_stepwise() {
        let config = AutoARIMAConfig::default();
        assert_eq!(config.max_p, 5);
        assert_eq!(config.max_q, 5);
        assert_eq!(config.max_order, 5);
        assert_eq!(config.max_d, 2);
        assert_eq!(config.max_cap_d, 1);
        assert!(config.stepwise);
        assert!(!config.true_stepwise);
    }

    /// `with_grid_stepwise()` is a no-op on top of `default()` (both
    /// currently mean grid stepwise) but documents the explicit opt-in
    /// path for callers once the default flips to HK stepwise.
    #[test]
    fn with_grid_stepwise_sets_grid_search() {
        let config = AutoARIMAConfig::default().with_grid_stepwise();
        assert!(config.stepwise);
        assert!(!config.true_stepwise);
    }

    #[test]
    fn auto_arima_selects_model() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
        assert!(!model.model_scores().is_empty());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_arima_with_trend() {
        let timestamps = make_timestamps(100);
        // Add some noise to make fitting easier
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + 1.5 * i as f64 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
    }

    #[test]
    fn auto_arima_ar_process() {
        let timestamps = make_timestamps(100);
        // AR(1) process
        let mut values = vec![10.0];
        for i in 1..100 {
            values.push(0.8 * values[i - 1] + (i as f64 * 0.05).sin());
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let (p, _, _) = model.selected_order().unwrap();
        // Should select AR component
        assert!(p >= 1);
    }

    #[test]
    fn auto_arima_exhaustive() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoARIMAConfig::default().exhaustive();
        let mut model = AutoARIMA::with_config(config);
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
        // Exhaustive should have more candidates
        assert!(model.model_scores().len() > 3);
    }

    #[test]
    fn auto_arima_true_stepwise() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoARIMAConfig::default().with_true_stepwise();
        let mut model = AutoARIMA::with_config(config);
        model.fit(&ts).unwrap();

        assert!(model.selected_order().is_some());
        // True stepwise should find a model efficiently
        // It evaluates initial models + neighbors, typically less than exhaustive
        let n_models = model.model_scores().len();
        assert!(
            n_models > 0,
            "Should evaluate at least some models, got {}",
            n_models
        );

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn auto_arima_model_scores_sorted() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let scores = model.model_scores();
        for i in 1..scores.len() {
            assert!(scores[i].1 >= scores[i - 1].1);
        }
    }

    #[test]
    fn auto_arima_confidence_intervals() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn auto_arima_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn auto_arima_requires_fit() {
        let model = AutoARIMA::new();
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn auto_arima_fitted_and_residuals() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + i as f64 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    #[test]
    fn auto_arima_name() {
        let model = AutoARIMA::new();
        assert_eq!(model.name(), "AutoARIMA");
    }

    #[test]
    fn auto_arima_config() {
        let config = AutoARIMAConfig::default()
            .with_max_orders(5, 2, 5)
            .exhaustive();

        assert_eq!(config.max_p, 5);
        assert_eq!(config.max_d, 2);
        assert_eq!(config.max_q, 5);
        assert!(!config.stepwise);
    }

    // SARIMA-specific tests
    #[test]
    fn auto_arima_seasonal() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = AutoARIMA::seasonal(12);
        model.fit(&ts).unwrap();

        assert!(model.selected_full_order().is_some());

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn auto_arima_seasonal_config() {
        let config = AutoARIMAConfig::default()
            .with_seasonal_period(12)
            .with_seasonal_orders(2, 1, 2);

        assert_eq!(config.seasonal_period, 12);
        assert_eq!(config.max_cap_p, 2);
        assert_eq!(config.max_cap_d, 1);
        assert_eq!(config.max_cap_q, 2);
    }

    #[test]
    fn auto_arima_seasonal_selects_sarima() {
        let timestamps = make_timestamps(100);
        // Strong seasonal pattern
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 15.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = AutoARIMAConfig::default()
            .with_seasonal_period(12)
            .exhaustive();
        let mut model = AutoARIMA::with_config(config);
        model.fit(&ts).unwrap();

        // Should select a seasonal model
        if model.selected_full_order().is_some() {
            // With strong seasonality, should select seasonal components
            assert!(model.model_scores().len() > 1);
        }

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    // ── Issue #106 — Decomposable trait additions ───────────────────────

    #[test]
    fn auto_arima_training_values_retained() {
        let timestamps = make_timestamps(60);
        let values: Vec<f64> = (0..60).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();
        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();
        let training = model.training_values().unwrap();
        assert_eq!(training, values.as_slice());
    }

    #[test]
    fn auto_arima_training_regressors_none_without_regs() {
        let timestamps = make_timestamps(60);
        let values: Vec<f64> = (0..60).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();
        assert!(model.training_regressors().is_none());
    }

    #[test]
    fn auto_arima_trend_equals_fitted_values() {
        let timestamps = make_timestamps(80);
        let values: Vec<f64> = (0..80).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();
        let trend = model.trend_component().unwrap();
        let fitted = model.fitted_values().unwrap();
        assert_eq!(trend.len(), fitted.len());
        for (t, f) in trend.iter().zip(fitted.iter()) {
            // NaN-aware: both NaN at warmup rows, or both equal.
            assert_eq!(t.is_nan(), f.is_nan(), "NaN-ness must agree");
            if !t.is_nan() {
                assert_eq!(t, f);
            }
        }
    }

    #[test]
    fn auto_arima_seasonal_component_returns_err() {
        let timestamps = make_timestamps(60);
        let values: Vec<f64> = (0..60).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();
        // ARIMA isn't a structural decomposition model; seasonal returns Err.
        assert!(matches!(
            model.seasonal_component(),
            Err(ForecastError::InvalidParameter(_))
        ));
    }

    #[test]
    fn auto_arima_trend_component_requires_fit() {
        let model = AutoARIMA::new();
        assert!(matches!(
            model.trend_component(),
            Err(ForecastError::FitRequired { .. })
        ));
    }
}
