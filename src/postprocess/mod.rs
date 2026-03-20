//! Probabilistic forecasting via postprocessing.
//!
//! This module provides methods to convert point forecasts into calibrated
//! predictive distributions. It follows the approach of
//! [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl).
//!
//! # Overview
//!
//! Postprocessing methods enable model-agnostic uncertainty quantification:
//! - **Conformal Prediction**: Distribution-free intervals with coverage guarantees
//! - **Historical Simulation**: Non-parametric empirical error distribution
//! - **Normal Predictor**: Gaussian error assumption baseline
//! - **IDR**: Isotonic Distributional Regression (state-of-the-art calibration)
//! - **QRA**: Quantile Regression Averaging for ensemble combining
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{PointForecasts, ConformalPredictor};
//!
//! // Create point forecasts
//! let forecasts = PointForecasts::from_values(vec![10.0, 12.0, 14.0]);
//!
//! // Apply conformal prediction for 90% coverage intervals
//! let mut cp = ConformalPredictor::new(0.90);
//! cp.fit(&historical_forecasts, &historical_actuals);
//! let intervals = cp.predict(&forecasts);
//! ```

mod backtest;
mod binned_intervals;
mod conformal;
mod conformalize;
mod historical_sim;
mod idr;
mod normal;
mod processor;
mod qra;
mod types;

pub use backtest::{BacktestConfig, BacktestFold, BacktestResult, CalibratedModelByHorizon};
pub use binned_intervals::{BinnedConformalPredictor, BinnedConformalResult};
pub use conformal::{ConformalMethod, ConformalPredictor, ConformalResult, PerStepConformalResult};
pub use conformalize::{
    conformalize, conformalize_with_config, ConformalizeConfig, ConformalizeResult,
};
pub use historical_sim::{HistoricalSimResult, HistoricalSimulator};
pub use idr::{IDRPredictor, IDRResult};
pub use normal::{NormalPredictor, NormalResult};
pub use processor::{PostModel, PostProcessor, TrainedModel};
pub use qra::{QRAPredictor, QRARegularization, QRAResult};
pub use types::{PointForecasts, PredictionIntervals, QuantileForecasts};
