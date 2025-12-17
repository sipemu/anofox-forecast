//! Forecaster trait defining the common interface for all models.

use crate::core::{Forecast, TimeSeries};
use crate::error::Result;

/// Common interface for all forecasting models.
///
/// This trait is object-safe and can be used with `Box<dyn Forecaster>`.
pub trait Forecaster {
    /// Fit the model to the time series data.
    fn fit(&mut self, series: &TimeSeries) -> Result<()>;

    /// Generate predictions for the specified horizon.
    fn predict(&self, horizon: usize) -> Result<Forecast>;

    /// Generate predictions with confidence intervals.
    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        // Default implementation just returns point predictions
        let _ = level;
        self.predict(horizon)
    }

    /// Get the fitted values (in-sample predictions).
    fn fitted_values(&self) -> Option<&[f64]>;

    /// Get the residuals (actual - fitted).
    fn residuals(&self) -> Option<&[f64]>;

    /// Get the model name.
    fn name(&self) -> &str;

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted_values().is_some()
    }
}

/// Type alias for boxed forecaster trait objects.
///
/// # Example
///
/// ```
/// use anofox_forecast::models::{BoxedForecaster, Forecaster};
/// use anofox_forecast::models::baseline::Naive;
///
/// let model: BoxedForecaster = Box::new(Naive::new());
/// assert_eq!(model.name(), "Naive");
/// ```
pub type BoxedForecaster = Box<dyn Forecaster>;

/// Model specification for batch forecasting.
///
/// Contains a model factory function, name, and whether it supports native intervals.
///
/// # Example
///
/// ```
/// use anofox_forecast::models::{ModelSpec, BoxedForecaster};
/// use anofox_forecast::models::baseline::{Naive, SeasonalNaive};
///
/// let specs = vec![
///     ModelSpec::new("Naive", || Box::new(Naive::new()), true),
///     ModelSpec::with_period("SeasonalNaive", |p| Box::new(SeasonalNaive::new(p)), 12, true),
/// ];
///
/// for spec in &specs {
///     let model = spec.create();
///     assert!(!model.is_fitted());
/// }
/// ```
pub struct ModelSpec {
    /// Display name of the model
    pub name: &'static str,
    /// Factory function to create a new instance
    factory: Box<dyn Fn() -> BoxedForecaster + Send + Sync>,
    /// Whether the model supports native confidence intervals
    pub has_intervals: bool,
}

impl ModelSpec {
    /// Create a model spec with a simple factory.
    pub fn new<F>(name: &'static str, factory: F, has_intervals: bool) -> Self
    where
        F: Fn() -> BoxedForecaster + Send + Sync + 'static,
    {
        Self {
            name,
            factory: Box::new(factory),
            has_intervals,
        }
    }

    /// Create a model spec with a period parameter.
    pub fn with_period<F>(
        name: &'static str,
        factory: F,
        period: usize,
        has_intervals: bool,
    ) -> Self
    where
        F: Fn(usize) -> BoxedForecaster + Send + Sync + 'static,
    {
        Self {
            name,
            factory: Box::new(move || factory(period)),
            has_intervals,
        }
    }

    /// Create a new model instance.
    pub fn create(&self) -> BoxedForecaster {
        (self.factory)()
    }
}

/// Collection of model specifications for batch forecasting.
///
/// # Example
///
/// ```
/// use anofox_forecast::models::{ModelRegistry, ModelSpec};
/// use anofox_forecast::models::baseline::Naive;
///
/// let mut registry = ModelRegistry::new();
/// registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
///
/// // Create models from specs
/// for spec in registry.iter() {
///     let model = spec.create();
///     assert_eq!(model.name(), spec.name);
/// }
/// ```
pub struct ModelRegistry {
    models: Vec<ModelSpec>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Register a model specification.
    pub fn register(&mut self, spec: ModelSpec) {
        self.models.push(spec);
    }

    /// Get the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Iterate over model specifications.
    pub fn iter(&self) -> impl Iterator<Item = &ModelSpec> {
        self.models.iter()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::baseline::{Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage};
    use crate::models::exponential::SimpleExponentialSmoothing;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }

    fn make_test_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn test_boxed_forecaster() {
        let model: BoxedForecaster = Box::new(Naive::new());
        assert_eq!(model.name(), "Naive");
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_boxed_forecaster_fit_predict() {
        let mut model: BoxedForecaster = Box::new(Naive::new());
        let ts = make_test_series(20);

        assert!(model.fit(&ts).is_ok());
        assert!(model.is_fitted());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn test_boxed_forecaster_with_intervals() {
        let mut model: BoxedForecaster = Box::new(Naive::new());
        let ts = make_test_series(20);

        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        assert_eq!(forecast.horizon(), 5);
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn test_model_spec_simple() {
        let spec = ModelSpec::new("Naive", || Box::new(Naive::new()), true);
        assert_eq!(spec.name, "Naive");
        assert!(spec.has_intervals);

        let model = spec.create();
        assert_eq!(model.name(), "Naive");
    }

    #[test]
    fn test_model_spec_with_period() {
        let spec = ModelSpec::with_period(
            "SeasonalNaive",
            |p| Box::new(SeasonalNaive::new(p)),
            12,
            true,
        );
        let model = spec.create();
        assert_eq!(model.name(), "SeasonalNaive");
    }

    #[test]
    fn test_model_spec_no_intervals() {
        let spec = ModelSpec::new(
            "SES",
            || Box::new(SimpleExponentialSmoothing::new(0.3)),
            false,
        );
        assert!(!spec.has_intervals);
    }

    #[test]
    fn test_model_spec_creates_independent_instances() {
        let spec = ModelSpec::new("Naive", || Box::new(Naive::new()), true);
        let ts = make_test_series(20);

        let mut model1 = spec.create();
        let model2 = spec.create();

        // Fit model1 but not model2
        model1.fit(&ts).unwrap();

        assert!(model1.is_fitted());
        assert!(!model2.is_fitted());
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();
        assert!(registry.is_empty());

        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        assert_eq!(registry.len(), 1);

        let names: Vec<_> = registry.iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["Naive"]);
    }

    #[test]
    fn test_model_registry_default() {
        let registry = ModelRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_batch_create() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::with_period(
            "SeasonalNaive",
            |p| Box::new(SeasonalNaive::new(p)),
            12,
            true,
        ));

        let models: Vec<_> = registry.iter().map(|s| s.create()).collect();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name(), "Naive");
        assert_eq!(models[1].name(), "SeasonalNaive");
    }

    #[test]
    fn test_registry_multiple_models() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RandomWalk",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));
        registry.register(ModelSpec::new(
            "SES",
            || Box::new(SimpleExponentialSmoothing::new(0.3)),
            false,
        ));
        registry.register(ModelSpec::with_period(
            "WindowAvg",
            |p| Box::new(WindowAverage::new(p)),
            5,
            false,
        ));

        assert_eq!(registry.len(), 4);

        let intervals_count = registry.iter().filter(|s| s.has_intervals).count();
        assert_eq!(intervals_count, 2);
    }

    #[test]
    fn test_registry_batch_fit_predict() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RandomWalk",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));

        let ts = make_test_series(30);
        let mut results = Vec::new();

        for spec in registry.iter() {
            let mut model = spec.create();
            if model.fit(&ts).is_ok() {
                if let Ok(forecast) = model.predict(5) {
                    results.push((spec.name.to_string(), forecast.primary().to_vec()));
                }
            }
        }

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.len(), 5);
        assert_eq!(results[1].1.len(), 5);
    }

    #[test]
    fn test_forecaster_trait_methods() {
        let mut model = Naive::new();
        let ts = make_test_series(20);

        // Before fit
        assert!(!model.is_fitted());
        assert!(model.fitted_values().is_none());
        assert!(model.residuals().is_none());

        // After fit
        model.fit(&ts).unwrap();
        assert!(model.is_fitted());
        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.name(), "Naive");
    }

    #[test]
    fn test_boxed_forecaster_residuals() {
        let mut model: BoxedForecaster = Box::new(Naive::new());
        let ts = make_test_series(20);

        model.fit(&ts).unwrap();

        let residuals = model.residuals().unwrap();
        assert_eq!(residuals.len(), 20);
    }
}
