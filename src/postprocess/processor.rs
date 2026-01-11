//! Unified PostProcessor interface.
//!
//! This module provides a trait-based interface for postprocessing methods,
//! allowing uniform access to conformal prediction, historical simulation,
//! normal prediction, IDR, and QRA methods.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{PostProcessor, PostModel, PointForecasts};
//!
//! // Create a processor with conformal prediction
//! let processor = PostProcessor::conformal(0.90);
//!
//! // Train on historical data
//! let trained = processor.train(&forecasts, &actuals)?;
//!
//! // Generate prediction intervals for new forecasts
//! let intervals = trained.predict(&new_forecasts)?;
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::{
    ConformalMethod, ConformalPredictor, ConformalResult,
    HistoricalSimResult, HistoricalSimulator,
    NormalPredictor, NormalResult,
    IDRPredictor, IDRResult,
    PointForecasts, PredictionIntervals, QuantileForecasts,
};

/// Postprocessing model specification.
#[derive(Debug, Clone)]
pub enum PostModel {
    /// Conformal prediction with specified coverage.
    Conformal {
        /// Target coverage level (e.g., 0.90 for 90% intervals).
        coverage: f64,
        /// Conformal method to use.
        method: ConformalMethod,
    },
    /// Historical simulation with optional rolling window.
    HistoricalSim {
        /// Target quantile levels.
        quantiles: Vec<f64>,
        /// Optional rolling window size for adaptivity.
        window_size: Option<usize>,
    },
    /// Normal predictor assuming Gaussian errors.
    Normal {
        /// Target quantile levels.
        quantiles: Vec<f64>,
    },
    /// Isotonic Distributional Regression.
    IDR {
        /// Quantile levels for output.
        quantiles: Vec<f64>,
    },
}

impl PostModel {
    /// Create a conformal model with default split method.
    pub fn conformal(coverage: f64) -> Self {
        Self::Conformal {
            coverage,
            method: ConformalMethod::default(),
        }
    }

    /// Create a conformal model with specified method.
    pub fn conformal_with_method(coverage: f64, method: ConformalMethod) -> Self {
        Self::Conformal { coverage, method }
    }

    /// Create a historical simulation model.
    pub fn historical_sim(quantiles: Vec<f64>) -> Self {
        Self::HistoricalSim { quantiles, window_size: None }
    }

    /// Create a historical simulation model with rolling window.
    pub fn historical_sim_rolling(quantiles: Vec<f64>, window_size: usize) -> Self {
        Self::HistoricalSim { quantiles, window_size: Some(window_size) }
    }

    /// Create a normal predictor model.
    pub fn normal(quantiles: Vec<f64>) -> Self {
        Self::Normal { quantiles }
    }

    /// Create an IDR model with specified quantiles.
    pub fn idr(quantiles: Vec<f64>) -> Self {
        Self::IDR { quantiles }
    }
}

/// Result of training a postprocessor.
#[derive(Debug, Clone)]
pub enum TrainedModel {
    /// Trained conformal predictor.
    Conformal(ConformalResult),
    /// Trained historical simulator.
    HistoricalSim(HistoricalSimResult),
    /// Trained normal predictor.
    Normal(NormalResult),
    /// Trained IDR predictor.
    IDR(IDRResult),
}

/// Unified interface for postprocessing methods.
///
/// Provides a common API for training and prediction across different
/// postprocessing methods.
#[derive(Debug, Clone)]
pub struct PostProcessor {
    model: PostModel,
}

impl PostProcessor {
    /// Create a new postprocessor with the specified model.
    pub fn new(model: PostModel) -> Self {
        Self { model }
    }

    /// Create a conformal prediction postprocessor.
    pub fn conformal(coverage: f64) -> Self {
        Self::new(PostModel::conformal(coverage))
    }

    /// Create a historical simulation postprocessor.
    pub fn historical_sim(quantiles: Vec<f64>) -> Self {
        Self::new(PostModel::historical_sim(quantiles))
    }

    /// Create a normal prediction postprocessor.
    pub fn normal(quantiles: Vec<f64>) -> Self {
        Self::new(PostModel::normal(quantiles))
    }

    /// Create an IDR postprocessor.
    pub fn idr(quantiles: Vec<f64>) -> Self {
        Self::new(PostModel::idr(quantiles))
    }

    /// Get the model specification.
    pub fn model(&self) -> &PostModel {
        &self.model
    }

    /// Train the postprocessor on historical data.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// A trained model that can be used for prediction.
    pub fn train(
        &self,
        forecasts: &PointForecasts,
        actuals: &[f64],
    ) -> Result<TrainedModel> {
        let forecast_values = forecasts.values();

        match &self.model {
            PostModel::Conformal { coverage, method } => {
                let predictor = ConformalPredictor::new(*coverage, method.clone());
                let result = predictor.fit(forecast_values, actuals)?;
                Ok(TrainedModel::Conformal(result))
            }
            PostModel::HistoricalSim { quantiles, window_size } => {
                let simulator = if let Some(w) = window_size {
                    HistoricalSimulator::with_window(quantiles.clone(), *w)
                } else {
                    HistoricalSimulator::new(quantiles.clone())
                };
                let result = simulator.fit(forecast_values, actuals)?;
                Ok(TrainedModel::HistoricalSim(result))
            }
            PostModel::Normal { quantiles } => {
                let predictor = NormalPredictor::new(quantiles.clone());
                let result = predictor.fit(forecast_values, actuals)?;
                Ok(TrainedModel::Normal(result))
            }
            PostModel::IDR { quantiles } => {
                let predictor = IDRPredictor::new(quantiles.clone());
                let result = predictor.fit(forecast_values, actuals)?;
                Ok(TrainedModel::IDR(result))
            }
        }
    }

    /// Generate prediction intervals from a trained model.
    ///
    /// # Arguments
    ///
    /// * `trained` - The trained model from `train()`
    /// * `forecasts` - New point forecasts
    ///
    /// # Returns
    ///
    /// Prediction intervals for the forecasts.
    pub fn predict_intervals(
        &self,
        trained: &TrainedModel,
        forecasts: &PointForecasts,
    ) -> Result<PredictionIntervals> {
        let values = forecasts.values();

        match (trained, &self.model) {
            (TrainedModel::Conformal(result), PostModel::Conformal { coverage, method }) => {
                let predictor = ConformalPredictor::new(*coverage, method.clone());
                Ok(predictor.predict_values(result, values))
            }
            (TrainedModel::HistoricalSim(result), PostModel::HistoricalSim { quantiles, window_size }) => {
                let simulator = if let Some(w) = window_size {
                    HistoricalSimulator::with_window(quantiles.clone(), *w)
                } else {
                    HistoricalSimulator::new(quantiles.clone())
                };
                // Get outer quantiles for intervals
                let q_forecasts = simulator.predict_values(result, values)?;
                // Compute coverage from quantiles (upper - lower)
                let coverage = quantiles.last().unwrap_or(&0.9) - quantiles.first().unwrap_or(&0.1);
                quantiles_to_intervals(&q_forecasts, coverage)
            }
            (TrainedModel::Normal(result), PostModel::Normal { quantiles }) => {
                let predictor = NormalPredictor::new(quantiles.clone());
                let q_forecasts = predictor.predict_values(result, values)?;
                let coverage = quantiles.last().unwrap_or(&0.9) - quantiles.first().unwrap_or(&0.1);
                quantiles_to_intervals(&q_forecasts, coverage)
            }
            (TrainedModel::IDR(result), PostModel::IDR { quantiles }) => {
                let predictor = IDRPredictor::new(quantiles.clone());
                let q_forecasts = predictor.predict_values(result, values)?;
                let coverage = quantiles.last().unwrap_or(&0.9) - quantiles.first().unwrap_or(&0.1);
                quantiles_to_intervals(&q_forecasts, coverage)
            }
            _ => Err(ForecastError::InvalidParameter(
                "trained model does not match processor model".to_string()
            ))
        }
    }

    /// Generate quantile forecasts from a trained model.
    ///
    /// # Arguments
    ///
    /// * `trained` - The trained model from `train()`
    /// * `forecasts` - New point forecasts
    ///
    /// # Returns
    ///
    /// Quantile forecasts at the model's specified quantile levels.
    pub fn predict_quantiles(
        &self,
        trained: &TrainedModel,
        forecasts: &PointForecasts,
    ) -> Result<QuantileForecasts> {
        let values = forecasts.values();

        match (trained, &self.model) {
            (TrainedModel::HistoricalSim(result), PostModel::HistoricalSim { quantiles, window_size }) => {
                let simulator = if let Some(w) = window_size {
                    HistoricalSimulator::with_window(quantiles.clone(), *w)
                } else {
                    HistoricalSimulator::new(quantiles.clone())
                };
                simulator.predict_values(result, values)
            }
            (TrainedModel::Normal(result), PostModel::Normal { quantiles }) => {
                let predictor = NormalPredictor::new(quantiles.clone());
                predictor.predict_values(result, values)
            }
            (TrainedModel::IDR(result), PostModel::IDR { quantiles }) => {
                let predictor = IDRPredictor::new(quantiles.clone());
                predictor.predict_values(result, values)
            }
            (TrainedModel::Conformal(result), PostModel::Conformal { coverage, method }) => {
                // Conformal only produces two quantiles (lower/upper)
                let predictor = ConformalPredictor::new(*coverage, method.clone());
                let intervals = predictor.predict_values(result, values);

                let alpha = 1.0 - *coverage;
                let lower_q = alpha / 2.0;
                let upper_q = 1.0 - alpha / 2.0;

                let q_values: Vec<Vec<f64>> = intervals.lower().iter()
                    .zip(intervals.upper().iter())
                    .map(|(&l, &u)| vec![l, u])
                    .collect();

                QuantileForecasts::from_values(vec![lower_q, upper_q], q_values)
            }
            _ => Err(ForecastError::InvalidParameter(
                "trained model does not match processor model".to_string()
            ))
        }
    }

    /// Convert point forecasts to quantile forecasts in one step.
    ///
    /// This is a convenience method that trains on data and predicts in one call.
    ///
    /// # Arguments
    ///
    /// * `train_forecasts` - Historical point forecasts for training
    /// * `train_actuals` - Corresponding actual values
    /// * `predict_forecasts` - New forecasts to convert
    pub fn point_to_quantiles(
        &self,
        train_forecasts: &PointForecasts,
        train_actuals: &[f64],
        predict_forecasts: &PointForecasts,
    ) -> Result<QuantileForecasts> {
        let trained = self.train(train_forecasts, train_actuals)?;
        self.predict_quantiles(&trained, predict_forecasts)
    }
}

/// Extract prediction intervals from quantile forecasts.
/// Uses the first and last quantiles as lower and upper bounds.
fn quantiles_to_intervals(
    quantiles: &QuantileForecasts,
    coverage: f64,
) -> Result<PredictionIntervals> {
    let n_q = quantiles.n_quantiles();
    if n_q < 2 {
        return Err(ForecastError::InvalidParameter(
            "need at least 2 quantiles for intervals".to_string()
        ));
    }

    let lower_idx = 0;
    let upper_idx = n_q - 1;

    let lowers: Vec<f64> = (0..quantiles.n_times())
        .map(|t| quantiles.at_time(t).unwrap()[lower_idx])
        .collect();
    let uppers: Vec<f64> = (0..quantiles.n_times())
        .map(|t| quantiles.at_time(t).unwrap()[upper_idx])
        .collect();

    PredictionIntervals::from_bounds(lowers, uppers, coverage)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PostModel tests
    // =========================================================================

    mod post_model {
        use super::*;

        #[test]
        fn conformal_constructor() {
            let model = PostModel::conformal(0.90);
            if let PostModel::Conformal { coverage, .. } = model {
                assert!((coverage - 0.90).abs() < 1e-10);
            } else {
                panic!("Expected Conformal");
            }
        }

        #[test]
        fn conformal_with_method_constructor() {
            let model = PostModel::conformal_with_method(0.95, ConformalMethod::JackknifePlus);
            if let PostModel::Conformal { coverage, method } = model {
                assert!((coverage - 0.95).abs() < 1e-10);
                assert!(matches!(method, ConformalMethod::JackknifePlus));
            } else {
                panic!("Expected Conformal");
            }
        }

        #[test]
        fn historical_sim_constructor() {
            let model = PostModel::historical_sim(vec![0.1, 0.5, 0.9]);
            if let PostModel::HistoricalSim { quantiles, window_size } = model {
                assert_eq!(quantiles, vec![0.1, 0.5, 0.9]);
                assert!(window_size.is_none());
            } else {
                panic!("Expected HistoricalSim");
            }
        }

        #[test]
        fn historical_sim_rolling_constructor() {
            let model = PostModel::historical_sim_rolling(vec![0.5], 20);
            if let PostModel::HistoricalSim { window_size, .. } = model {
                assert_eq!(window_size, Some(20));
            } else {
                panic!("Expected HistoricalSim");
            }
        }

        #[test]
        fn normal_constructor() {
            let model = PostModel::normal(vec![0.1, 0.5, 0.9]);
            if let PostModel::Normal { quantiles } = model {
                assert_eq!(quantiles, vec![0.1, 0.5, 0.9]);
            } else {
                panic!("Expected Normal");
            }
        }

        #[test]
        fn idr_constructor() {
            let model = PostModel::idr(vec![0.1, 0.5, 0.9]);
            if let PostModel::IDR { quantiles } = model {
                assert_eq!(quantiles, vec![0.1, 0.5, 0.9]);
            } else {
                panic!("Expected IDR");
            }
        }

        #[test]
        fn model_is_clonable() {
            let model = PostModel::conformal(0.90);
            let cloned = model.clone();
            if let (PostModel::Conformal { coverage: c1, .. }, PostModel::Conformal { coverage: c2, .. }) = (model, cloned) {
                assert!((c1 - c2).abs() < 1e-10);
            }
        }
    }

    // =========================================================================
    // PostProcessor construction tests
    // =========================================================================

    mod construction {
        use super::*;

        #[test]
        fn new_creates_processor() {
            let processor = PostProcessor::new(PostModel::conformal(0.90));
            matches!(processor.model(), PostModel::Conformal { .. });
        }

        #[test]
        fn conformal_shortcut() {
            let processor = PostProcessor::conformal(0.95);
            if let PostModel::Conformal { coverage, .. } = processor.model() {
                assert!((*coverage - 0.95).abs() < 1e-10);
            } else {
                panic!("Expected Conformal");
            }
        }

        #[test]
        fn historical_sim_shortcut() {
            let processor = PostProcessor::historical_sim(vec![0.1, 0.5, 0.9]);
            matches!(processor.model(), PostModel::HistoricalSim { .. });
        }

        #[test]
        fn normal_shortcut() {
            let processor = PostProcessor::normal(vec![0.1, 0.5, 0.9]);
            if let PostModel::Normal { quantiles } = processor.model() {
                assert_eq!(quantiles, &vec![0.1, 0.5, 0.9]);
            } else {
                panic!("Expected Normal");
            }
        }

        #[test]
        fn idr_shortcut() {
            let processor = PostProcessor::idr(vec![0.1, 0.5, 0.9]);
            if let PostModel::IDR { quantiles } = processor.model() {
                assert_eq!(quantiles, &vec![0.1, 0.5, 0.9]);
            } else {
                panic!("Expected IDR");
            }
        }

        #[test]
        fn processor_is_clonable() {
            let processor = PostProcessor::conformal(0.90);
            let cloned = processor.clone();
            matches!(cloned.model(), PostModel::Conformal { .. });
        }
    }

    // =========================================================================
    // Training tests
    // =========================================================================

    mod training {
        use super::*;

        fn make_data() -> (PointForecasts, Vec<f64>) {
            let forecasts = PointForecasts::from_values((0..30).map(|i| i as f64).collect());
            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();
            (forecasts, actuals)
        }

        #[test]
        fn train_conformal() {
            let processor = PostProcessor::conformal(0.90);
            let (forecasts, actuals) = make_data();

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_ok());
            matches!(result.unwrap(), TrainedModel::Conformal(_));
        }

        #[test]
        fn train_historical_sim() {
            let processor = PostProcessor::historical_sim(vec![0.1, 0.5, 0.9]);
            let (forecasts, actuals) = make_data();

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_ok());
            matches!(result.unwrap(), TrainedModel::HistoricalSim(_));
        }

        #[test]
        fn train_normal() {
            let processor = PostProcessor::normal(vec![0.1, 0.5, 0.9]);
            let (forecasts, actuals) = make_data();

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_ok());
            matches!(result.unwrap(), TrainedModel::Normal(_));
        }

        #[test]
        fn train_idr() {
            let processor = PostProcessor::idr(vec![0.1, 0.5, 0.9]);
            let (forecasts, actuals) = make_data();

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_ok());
            matches!(result.unwrap(), TrainedModel::IDR(_));
        }

        #[test]
        fn train_fails_on_empty_data() {
            let processor = PostProcessor::conformal(0.90);
            let forecasts = PointForecasts::empty();
            let actuals: Vec<f64> = vec![];

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn train_fails_on_mismatched_lengths() {
            let processor = PostProcessor::conformal(0.90);
            let forecasts = PointForecasts::from_values(vec![1.0, 2.0, 3.0]);
            let actuals = vec![1.0, 2.0]; // Mismatched length

            let result = processor.train(&forecasts, &actuals);
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Prediction tests
    // =========================================================================

    mod prediction {
        use super::*;

        fn make_train_data() -> (PointForecasts, Vec<f64>) {
            let forecasts = PointForecasts::from_values((0..50).map(|i| i as f64).collect());
            let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();
            (forecasts, actuals)
        }

        fn make_predict_data() -> PointForecasts {
            PointForecasts::from_values((50..55).map(|i| i as f64).collect())
        }

        #[test]
        fn predict_intervals_conformal() {
            let processor = PostProcessor::conformal(0.90);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let intervals = processor.predict_intervals(&trained, &predict_f);

            assert!(intervals.is_ok());
            let intervals = intervals.unwrap();
            assert_eq!(intervals.len(), 5);
        }

        #[test]
        fn predict_intervals_historical_sim() {
            let processor = PostProcessor::historical_sim(vec![0.1, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let intervals = processor.predict_intervals(&trained, &predict_f);

            assert!(intervals.is_ok());
        }

        #[test]
        fn predict_intervals_normal() {
            let processor = PostProcessor::normal(vec![0.1, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let intervals = processor.predict_intervals(&trained, &predict_f);

            assert!(intervals.is_ok());
        }

        #[test]
        fn predict_intervals_idr() {
            let processor = PostProcessor::idr(vec![0.1, 0.5, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let intervals = processor.predict_intervals(&trained, &predict_f);

            assert!(intervals.is_ok());
        }

        #[test]
        fn intervals_are_valid() {
            let processor = PostProcessor::conformal(0.90);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let intervals = processor.predict_intervals(&trained, &predict_f).unwrap();

            // Check lower <= upper
            for i in 0..intervals.len() {
                let lower = intervals.lower()[i];
                let upper = intervals.upper()[i];
                assert!(lower <= upper, "lower should be <= upper at index {}", i);
            }
        }
    }

    // =========================================================================
    // Quantile prediction tests
    // =========================================================================

    mod quantile_prediction {
        use super::*;

        fn make_train_data() -> (PointForecasts, Vec<f64>) {
            let forecasts = PointForecasts::from_values((0..50).map(|i| i as f64).collect());
            let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();
            (forecasts, actuals)
        }

        fn make_predict_data() -> PointForecasts {
            PointForecasts::from_values((50..55).map(|i| i as f64).collect())
        }

        #[test]
        fn predict_quantiles_historical_sim() {
            let processor = PostProcessor::historical_sim(vec![0.1, 0.5, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let result = processor.predict_quantiles(&trained, &predict_f);

            assert!(result.is_ok());
            let qf = result.unwrap();
            assert_eq!(qf.n_times(), 5);
            assert_eq!(qf.n_quantiles(), 3);
        }

        #[test]
        fn predict_quantiles_normal() {
            let processor = PostProcessor::normal(vec![0.1, 0.5, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let result = processor.predict_quantiles(&trained, &predict_f);

            assert!(result.is_ok());
        }

        #[test]
        fn predict_quantiles_idr() {
            let processor = PostProcessor::idr(vec![0.1, 0.5, 0.9]);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let result = processor.predict_quantiles(&trained, &predict_f);

            assert!(result.is_ok());
        }

        #[test]
        fn predict_quantiles_conformal() {
            let processor = PostProcessor::conformal(0.90);
            let (train_f, train_a) = make_train_data();
            let predict_f = make_predict_data();

            let trained = processor.train(&train_f, &train_a).unwrap();
            let result = processor.predict_quantiles(&trained, &predict_f);

            assert!(result.is_ok());
            let qf = result.unwrap();
            // Conformal produces 2 quantiles (lower/upper)
            assert_eq!(qf.n_quantiles(), 2);
        }
    }

    // =========================================================================
    // point_to_quantiles convenience method tests
    // =========================================================================

    mod point_to_quantiles {
        use super::*;

        #[test]
        fn works_with_historical_sim() {
            let processor = PostProcessor::historical_sim(vec![0.1, 0.5, 0.9]);
            let train_f = PointForecasts::from_values((0..50).map(|i| i as f64).collect());
            let train_a: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();
            let predict_f = PointForecasts::from_values((50..55).map(|i| i as f64).collect());

            let result = processor.point_to_quantiles(&train_f, &train_a, &predict_f);

            assert!(result.is_ok());
            let qf = result.unwrap();
            assert_eq!(qf.n_times(), 5);
        }

        #[test]
        fn works_with_idr() {
            let processor = PostProcessor::idr(vec![0.1, 0.5, 0.9]);
            let train_f = PointForecasts::from_values((0..50).map(|i| i as f64).collect());
            let train_a: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();
            let predict_f = PointForecasts::from_values((50..55).map(|i| i as f64).collect());

            let result = processor.point_to_quantiles(&train_f, &train_a, &predict_f);

            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // Model mismatch tests
    // =========================================================================

    mod model_mismatch {
        use super::*;

        #[test]
        fn predict_fails_on_model_mismatch() {
            let conformal = PostProcessor::conformal(0.90);
            let normal = PostProcessor::normal(vec![0.1, 0.9]);

            let forecasts = PointForecasts::from_values((0..30).map(|i| i as f64).collect());
            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();

            // Train with conformal
            let trained = conformal.train(&forecasts, &actuals).unwrap();

            // Try to predict with normal - should fail
            let predict_f = PointForecasts::from_values(vec![30.0, 31.0]);
            let result = normal.predict_intervals(&trained, &predict_f);

            assert!(result.is_err());
        }
    }

    // =========================================================================
    // TrainedModel tests
    // =========================================================================

    mod trained_model {
        use super::*;

        #[test]
        fn trained_model_is_clonable() {
            let processor = PostProcessor::conformal(0.90);
            let forecasts = PointForecasts::from_values((0..30).map(|i| i as f64).collect());
            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();

            let trained = processor.train(&forecasts, &actuals).unwrap();
            let _cloned = trained.clone();
        }
    }
}
