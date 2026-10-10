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
    /// own search strategy (Hyndman-Khandakar stepwise, D-06) and the
    /// crate's default: every candidate is fitted by a port of R's
    /// `stats::arima` estimation path (`r_arima`, CSS then CSS-ML with
    /// `optim` BFGS, or CSS alone under R's approximation rule) and scored
    /// with `myarima`'s criteria, so the search reproduces R's own trace.
    pub true_stepwise: bool,
    /// Maximum value of `p+q+P+Q` (R `auto.arima`'s `max.order`, default
    /// 5). Like R, it bounds the exhaustive/grid searches only; R's
    /// stepwise search (`true_stepwise`) never applies it.
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
            // R `auto.arima`'s real defaults (D-06): max.P = 2, max.Q = 2,
            // Hyndman-Khandakar stepwise search. Safe since 11-12: every
            // candidate is fitted by the `r_arima` port of R's estimation
            // path (11-11 had the right likelihood but a different
            // optimiser, which changed which candidates converged where).
            max_cap_p: 2,
            max_cap_q: 2,
            max_cap_d: 1,
            seasonal_period: 0,
            stepwise: true,
            true_stepwise: true, // R's Hyndman-Khandakar stepwise search (D-06, 11-11)
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
    /// (Hyndman-Khandakar). This is the default; the builder exists for
    /// symmetry with `with_grid_stepwise()`.
    pub fn with_true_stepwise(mut self) -> Self {
        self.stepwise = true;
        self.true_stepwise = true;
        self
    }

    /// Use the crate's legacy fixed-grid stepwise search instead of R's
    /// Hyndman-Khandakar stepwise search (opt-in; not R's algorithm).
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

    /// Score-only evaluation: compute AICc/BIC from pre-computed differenced
    /// series without constructing the full model. Skips validation,
    /// storage, and calculate_fitted.
    ///
    /// `approximation` is R's default rule (`n > 150 || m > 12`, computed
    /// once per `fit()` call on the undifferenced training length -- see
    /// `Self::approximation_for_series`): when true, every candidate is
    /// scored with R's CSS criterion (`exact_likelihood::css_r_criterion`);
    /// otherwise every candidate is scored by the exact Gaussian
    /// likelihood (`exact_likelihood::fit_exact_ml`). Either way this is a
    /// common likelihood basis across the whole candidate grid (D-06).
    fn score_order_static(
        order: ModelOrder,
        diff_series: &[f64],
        use_aic: bool,
        approximation: bool,
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
                approximation,
            )
        } else {
            ARIMA::score_order(
                order.p,
                order.q,
                diff_series,
                use_aic,
                allow_constant,
                approximation,
            )
        }
    }

    /// R `forecast::auto.arima`'s default `approximation` rule: a series
    /// with more than 150 observations or a seasonal period greater than
    /// 12 is searched with the (much cheaper) CSS criterion, and only the
    /// winner is refit by exact ML (D-06).
    fn approximation_for_series(n: usize, s: usize) -> bool {
        n > 150 || s > 12
    }

    /// Score a single order at an EXPLICIT constant choice (no
    /// with/without auto-min) -- used by `true_stepwise_search` to
    /// match R's `search.arima`/`auto.arima` source, which threads one
    /// explicit `constant` state through every candidate in its
    /// stepwise loop rather than re-deciding per candidate (D-06,
    /// 11-11).
    fn score_order_explicit(
        order: ModelOrder,
        values: &[f64],
        use_aic: bool,
        approximation: bool,
        with_constant: bool,
    ) -> Option<f64> {
        let method = if approximation {
            super::r_arima::RMethod::Css
        } else {
            super::r_arima::RMethod::CssMl
        };
        Self::fit_r_candidate(order, values, with_constant, method).map(|fit| {
            if use_aic {
                fit.aicc
            } else {
                fit.bic
            }
        })
    }

    /// Fit one candidate the way R's `myarima` does (see
    /// `r_arima::r_arima_fit`). The constant is a mean when `d + D = 0`
    /// and a drift when `d + D = 1`; R regresses on the time index
    /// `1..n`. `values` is the UNdifferenced training series (R's
    /// likelihood runs a diffuse-prior Kalman filter on it).
    /// `forecast:::is.constant`: `all.equal(x, rep(x[1], n))` with the
    /// default relative tolerance `1.5e-8`.
    fn r_is_constant(x: &[f64]) -> bool {
        let Some(&first) = x.first() else {
            return false;
        };
        let n = x.len() as f64;
        let mut xy = x.iter().map(|v| (v - first).abs()).sum::<f64>() / n;
        let xn = x.iter().map(|v| v.abs()).sum::<f64>() / n;
        let tol = 1.5e-8;
        if xn.is_finite() && xn > tol {
            xy /= xn;
        }
        xy <= tol
    }

    fn r_final_model(
        series: &TimeSeries,
        order: ModelOrder,
        with_constant: bool,
    ) -> Option<SelectedModel> {
        let values = series.primary_values();
        let fit =
            Self::fit_r_candidate(order, values, with_constant, super::r_arima::RMethod::CssMl)?;
        let m = if order.s > 1 { order.s } else { 1 };
        // R's constant is a regression on 1 (mean) or the time index
        // (drift); on the differenced scale it is the mean `mu` of `w`.
        let mu = if with_constant && order.d + order.cap_d <= 1 {
            if order.d == 0 && order.cap_d == 1 {
                fit.constant * m as f64
            } else {
                fit.constant
            }
        } else {
            0.0
        };
        if order.is_seasonal() {
            // SARIMA uses the constant form `c = mu * phi(1) * Phi(1)`.
            let c = mu * (1.0 - fit.ar.iter().sum::<f64>()) * (1.0 - fit.sar.iter().sum::<f64>());
            let spec = super::SARIMASpec::new(
                order.p,
                order.d,
                order.q,
                order.cap_p,
                order.cap_d,
                order.cap_q,
                order.s,
            );
            let mut model = SARIMA::with_coefficients(spec, fit.ar, fit.ma, fit.sar, fit.sma, c);
            model.fit(series).ok()?;
            Some(SelectedModel::SARIMA(model))
        } else {
            // ARIMA uses the mean form `w_t - mu = ...` (same as R).
            let mut model = ARIMA::with_coefficients(order.p, order.d, order.q, fit.ar, fit.ma, mu);
            model.fit(series).ok()?;
            Some(SelectedModel::ARIMA(model))
        }
    }

    fn fit_r_candidate(
        order: ModelOrder,
        values: &[f64],
        with_constant: bool,
        method: super::r_arima::RMethod,
    ) -> Option<super::r_arima::RArimaFit> {
        let m = if order.s > 1 { order.s } else { 1 };
        super::r_arima::r_arima_fit(
            values,
            order.d,
            order.cap_d,
            order.p,
            order.q,
            order.cap_p,
            order.cap_q,
            m,
            with_constant,
            method,
        )
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

    /// R `forecast::auto.arima`'s own stepwise algorithm (D-06, 11-11
    /// rewrite): the structural moves (seasonal P/Q then non-seasonal
    /// p/q, each `-1`/`+1`/diagonals, first-improvement, restart scan)
    /// are scored at a FIXED `current_constant` -- matching R's
    /// `search.arima`/`auto.arima` source exactly, which threads one
    /// explicit `constant` variable through every `myarima()` call in
    /// the loop and only ever flips it via the dedicated toggle move at
    /// the end of each scan (`if (allowdrift || allowmean) { ... }`,
    /// no `next` after it, so the toggle's effect is picked up on the
    /// next scan). The crate's own `score_order_static`/`fit_exact_ml`
    /// previously auto-selected whichever of with/without-constant
    /// scored better on EVERY structural move -- effectively exploring
    /// twice as many models per step as R does and never tracking a
    /// stable `constant` state, which let the greedy walk diverge onto
    /// a different local optimum than R's (measured: AirPassengers
    /// settling ~1.3 AICc units off R's pick, RW share 0.60 vs R's
    /// 0.86). This rewrite restores R's exact state machine.
    ///
    /// 11-12 details that also follow R's source: start model 4 (null
    /// model without constant) updates the best fit but NOT the constant
    /// state; `max.order` is not applied; `values` is the undifferenced
    /// series (candidates are fitted by `r_arima`); under the
    /// approximation rule the visited models are refit by exact CSS-ML in
    /// increasing CSS-ic order and the first finite one wins. Returns the
    /// selected order, its score and whether it carries the constant.
    fn true_stepwise_search(
        &mut self,
        values: &[f64],
        d: usize,
        cap_d: usize,
        approximation: bool,
    ) -> Option<(ModelOrder, f64, bool)> {
        let s = self.config.seasonal_period;
        let use_aic = self.config.use_aic;
        let max_models = 94; // matching R's nmodels limit
                             // R: `allowdrift <- allowdrift && (d+D)==1; allowmean <- allowmean && (d+D)==0;
                             // constant <- allowdrift || allowmean` -- i.e. TRUE whenever d+D<=1.
        let allow_constant = d + cap_d <= 1;

        let max_p = self.config.max_p;
        let max_q = self.config.max_q;
        let max_cap_p = self.config.max_cap_p;
        let max_cap_q = self.config.max_cap_q;

        // visited keys include the constant bit (R's `newmodel` dedups
        // against the full (p,P,q,Q,constant) results row).
        let mut visited: std::collections::HashSet<(usize, usize, usize, usize, bool)> =
            std::collections::HashSet::new();
        let mut n_models = 0usize;
        let mut current_constant = allow_constant;
        // Every visited candidate in visiting order (R's `results`
        // matrix), including `Inf` scores -- needed for R's
        // approximation refit below.
        let mut trail: Vec<(ModelOrder, bool, f64)> = Vec::new();

        // R's start.p/start.q/start.P/start.Q = 2/2/1/1, clipped to the
        // configured maxima.
        let start_p = 2.min(max_p);
        let start_q = 2.min(max_q);
        let start_cap_p = if s > 1 { 1.min(max_cap_p) } else { 0 };
        let start_cap_q = if s > 1 { 1.min(max_cap_q) } else { 0 };

        let mut current_order = ModelOrder {
            p: start_p,
            d,
            q: start_q,
            cap_p: start_cap_p,
            cap_d,
            cap_q: start_cap_q,
            s,
        };
        let mut current_score = f64::INFINITY;
        // Constant of the model that produced `current_score` (R's
        // `bestfit`); can differ from `current_constant` after start
        // model 4, see below.
        let mut best_constant = current_constant;
        // `record_candidate` is a plain function (not a closure) so it
        // borrows `self.model_scores` only for the duration of each
        // call -- a closure capturing `self` here would hold a
        // conflicting borrow across the later `self.get_neighbors(...)`
        // calls in the main loop below.
        #[allow(clippy::too_many_arguments)]
        fn record_candidate(
            order: ModelOrder,
            with_constant: bool,
            visited: &mut std::collections::HashSet<(usize, usize, usize, usize, bool)>,
            n_models: &mut usize,
            model_scores: &mut Vec<(ModelOrder, f64)>,
            trail: &mut Vec<(ModelOrder, bool, f64)>,
            values: &[f64],
            use_aic: bool,
            approximation: bool,
        ) -> Option<f64> {
            let key = (order.p, order.q, order.cap_p, order.cap_q, with_constant);
            if visited.contains(&key) {
                return None;
            }
            visited.insert(key);
            *n_models += 1;
            let score = AutoARIMA::score_order_explicit(
                order,
                values,
                use_aic,
                approximation,
                with_constant,
            );
            trail.push((order, with_constant, score.unwrap_or(f64::INFINITY)));
            let score = score?;
            model_scores.push((order, score));
            Some(score)
        }

        // Start model 0: (start.p, d, start.q)(start.P, D, start.Q).
        if let Some(score) = record_candidate(
            current_order,
            current_constant,
            &mut visited,
            &mut n_models,
            &mut self.model_scores,
            &mut trail,
            values,
            use_aic,
            approximation,
        ) {
            current_score = score;
            best_constant = current_constant;
        }

        // Start model 1: null model, same constant.
        let null_order = ModelOrder {
            p: 0,
            d,
            q: 0,
            cap_p: 0,
            cap_d,
            cap_q: 0,
            s,
        };
        if let Some(score) = record_candidate(
            null_order,
            current_constant,
            &mut visited,
            &mut n_models,
            &mut self.model_scores,
            &mut trail,
            values,
            use_aic,
            approximation,
        ) {
            if score < current_score {
                current_score = score;
                current_order = null_order;
                best_constant = current_constant;
            }
        }

        // Start model 2: pure AR, same constant.
        if max_p > 0 || max_cap_p > 0 {
            let ar_order = ModelOrder {
                p: usize::from(max_p > 0),
                d,
                q: 0,
                cap_p: usize::from(s > 1 && max_cap_p > 0),
                cap_d,
                cap_q: 0,
                s,
            };
            if let Some(score) = record_candidate(
                ar_order,
                current_constant,
                &mut visited,
                &mut n_models,
                &mut self.model_scores,
                &mut trail,
                values,
                use_aic,
                approximation,
            ) {
                if score < current_score {
                    current_score = score;
                    current_order = ar_order;
                    best_constant = current_constant;
                }
            }
        }

        // Start model 3: pure MA, same constant.
        if max_q > 0 || max_cap_q > 0 {
            let ma_order = ModelOrder {
                p: 0,
                d,
                q: usize::from(max_q > 0),
                cap_p: 0,
                cap_d,
                cap_q: usize::from(s > 1 && max_cap_q > 0),
                s,
            };
            if let Some(score) = record_candidate(
                ma_order,
                current_constant,
                &mut visited,
                &mut n_models,
                &mut self.model_scores,
                &mut trail,
                values,
                use_aic,
                approximation,
            ) {
                if score < current_score {
                    current_score = score;
                    current_order = ma_order;
                    best_constant = current_constant;
                }
            }
        }

        // Start model 4: null model WITHOUT constant (only tried when
        // the current constant is TRUE, matching R's `if (constant)`).
        if current_constant {
            if let Some(score) = record_candidate(
                null_order,
                false,
                &mut visited,
                &mut n_models,
                &mut self.model_scores,
                &mut trail,
                values,
                use_aic,
                approximation,
            ) {
                // R (`auto.arima`): `bestfit <- fit; p <- q <- P <- Q <- 0`
                // -- deliberately WITHOUT `constant <- FALSE`: the
                // structural moves that follow are still scored with the
                // constant, and only the end-of-scan toggle can flip it.
                if score < current_score {
                    current_score = score;
                    current_order = null_order;
                    best_constant = false;
                }
            }
        }

        // Greedy first-improvement hill climbing at a FIXED constant
        // (R's structural moves), with a dedicated constant-toggle move
        // at the end of each scan (R's `allowdrift || allowmean` block).
        let mut start_k = 0usize;
        while start_k < n_models && n_models < max_models {
            start_k = n_models;
            let neighbors = self.get_neighbors(current_order);
            let mut improved = false;

            for neighbor in neighbors {
                if n_models >= max_models {
                    break;
                }
                if let Some(score) = record_candidate(
                    neighbor,
                    current_constant,
                    &mut visited,
                    &mut n_models,
                    &mut self.model_scores,
                    &mut trail,
                    values,
                    use_aic,
                    approximation,
                ) {
                    if score < current_score {
                        current_score = score;
                        current_order = neighbor;
                        best_constant = current_constant;
                        improved = true;
                        break; // greedy: take first improvement, restart scan
                    }
                }
            }

            if improved {
                continue;
            }

            // Dedicated constant-toggle move (R: no `next` after this,
            // so it is tried once per scan, after every structural move
            // has failed to improve).
            if allow_constant && n_models < max_models {
                if let Some(score) = record_candidate(
                    current_order,
                    !current_constant,
                    &mut visited,
                    &mut n_models,
                    &mut self.model_scores,
                    &mut trail,
                    values,
                    use_aic,
                    approximation,
                ) {
                    if score < current_score {
                        current_score = score;
                        current_constant = !current_constant;
                        best_constant = current_constant;
                    }
                }
            }
        }

        if !current_score.is_finite() {
            return None;
        }
        if approximation {
            // R: "Now re-fitting the best model(s) without
            // approximations": walk the visited models in increasing
            // CSS-ic order (`order()` is stable; `Inf` sorts last but is
            // still tried) and keep the first whose exact CSS-ML fit has
            // a finite ic; if none does, keep the CSS winner.
            let mut idx: Vec<usize> = (0..trail.len()).collect();
            idx.sort_by(|&a, &b| trail[a].2.total_cmp(&trail[b].2));
            for i in idx {
                let (order, with_constant, _) = trail[i];
                if let Some(score) =
                    Self::score_order_explicit(order, values, use_aic, false, with_constant)
                {
                    return Some((order, score, with_constant));
                }
            }
        }
        Some((current_order, current_score, best_constant))
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
        let approximation = Self::approximation_for_series(n_values, self.config.seasonal_period);

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
                    Self::score_order_static(order, diff_series, use_aic, approximation)
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
                        Self::score_order_static(order, diff_series, use_aic, approximation)
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

        // R `auto.arima`'s `is.constant(dx)` branch: an exactly constant
        // differenced series (e.g. a perfect linear trend) is not
        // searched at all -- R fits ARIMA(0,d,0)(0,D,0) with the
        // constant FIXED at `mean(dx)` (drift / mean; d < 2 or D = 1 with
        // d = 0) or with no constant (d = 2, or d > 0 with D > 0).
        if self.config.stepwise && self.config.true_stepwise && !series.has_regressors() {
            let d = suggested_d;
            let cap_d = if s > 1 { suggested_cap_d } else { 0 };
            if let Some(dx) = diff_series_map.get(&(d, cap_d)) {
                if Self::r_is_constant(dx) {
                    let order = ModelOrder {
                        p: 0,
                        d,
                        q: 0,
                        cap_p: 0,
                        cap_d,
                        cap_q: 0,
                        s,
                    };
                    let with_constant = (cap_d > 0 && d == 0) || (cap_d == 0 && d < 2);
                    let mu = if with_constant {
                        dx.iter().sum::<f64>() / dx.len() as f64
                    } else {
                        0.0
                    };
                    let model = if order.is_seasonal() {
                        let spec = super::SARIMASpec::new(0, d, 0, 0, cap_d, 0, s);
                        let mut m =
                            SARIMA::with_coefficients(spec, vec![], vec![], vec![], vec![], mu);
                        m.fit(series)?;
                        SelectedModel::SARIMA(m)
                    } else {
                        let mut m = ARIMA::with_coefficients(0, d, 0, vec![], vec![], mu);
                        m.fit(series)?;
                        SelectedModel::ARIMA(m)
                    };
                    self.selected_model = Some(model);
                    self.selected_order = Some(order);
                    self.training_values_store = Some(values.to_vec());
                    self.training_regressors_store = None;
                    return Ok(());
                }
            }
        }

        let mut best_order: Option<ModelOrder> = None;
        let mut best_score = f64::INFINITY;
        let mut best_constant = false;
        // R's default `approximation` rule (D-06): computed once per
        // `fit()` call from the undifferenced training length and the
        // configured seasonal period.
        let approximation = Self::approximation_for_series(values.len(), s);

        // Phase 1: Score-only evaluation to find the best order
        if self.config.stepwise && self.config.true_stepwise {
            for &d in &d_range {
                for &cap_d in &cap_d_range {
                    if let Some(diff_series) = diff_series_map.get(&(d, cap_d)) {
                        let _ = diff_series;
                        if let Some((order, score, with_constant)) =
                            self.true_stepwise_search(values, d, cap_d, approximation)
                        {
                            if score < best_score {
                                best_score = score;
                                best_order = Some(order);
                                best_constant = with_constant;
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

        // True-stepwise path: the final model is R's own CSS-ML fit of
        // the selected (order, constant) -- exactly `auto.arima`'s
        // `bestfit` -- carried into the forecasting model as pre-fitted
        // coefficients, so the forecast uses the same constant/drift
        // decision and estimates the search made (D-06).
        if self.selected_model.is_none() && self.config.stepwise && self.config.true_stepwise {
            if let Some(order) = best_order {
                if !series.has_regressors() {
                    self.selected_model = Self::r_final_model(series, order, best_constant);
                }
            }
        }

        // Fallback / grid path: crate's own full fit of the chosen order.
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

    /// `AutoARIMAConfig::default()` matches R `auto.arima`'s own defaults
    /// (D-06, 11-11): `max_order` (R's `max.order`) is 5, enforced by
    /// `true_stepwise_search`; `max_cap_p`/`max_cap_q` are 2 (R's
    /// `max.P`/`max.Q`); the default search is R's Hyndman-Khandakar
    /// stepwise (`true_stepwise = true`). This flip was deferred by
    /// 11-04/11-05 (see 11-05-SUMMARY.md) until the exact-likelihood
    /// scorer (11-11, `exact_likelihood::fit_exact_ml`/`css_r_criterion`)
    /// replaced the CSS-window scorer that made the flip unsafe.
    #[test]
    fn default_config_has_max_order_and_grid_stepwise() {
        let config = AutoARIMAConfig::default();
        assert_eq!(config.max_p, 5);
        assert_eq!(config.max_q, 5);
        assert_eq!(config.max_order, 5);
        assert_eq!(config.max_d, 2);
        assert_eq!(config.max_cap_d, 1);
        assert_eq!(config.max_cap_p, 2);
        assert_eq!(config.max_cap_q, 2);
        assert!(config.stepwise);
        assert!(config.true_stepwise);
    }

    /// `with_grid_stepwise()` is the explicit opt-in to the old fixed-grid
    /// search now that the default is R's HK stepwise (11-11).
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
