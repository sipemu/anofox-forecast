use std::fmt;
use std::time::{Duration, Instant};

/// Execution metadata for a single model fit+predict cycle.
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Name of the model, e.g. "ARIMA(1,1,1)".
    pub model_name: String,
    /// Duration of the fitting phase.
    pub fit_duration: Option<Duration>,
    /// Duration of the prediction phase.
    pub predict_duration: Option<Duration>,
    /// Number of observations used for fitting.
    pub n_observations: usize,
    /// Forecast horizon.
    pub horizon: usize,
    /// Whether the model converged.
    pub converged: bool,
    /// Error message if the model failed.
    pub error_message: Option<String>,
}

/// Timer utility for measuring execution.
#[derive(Debug)]
pub struct ExecutionTimer {
    start: Instant,
}

impl ExecutionMetadata {
    /// Create new metadata with default/empty fields.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            fit_duration: None,
            predict_duration: None,
            n_observations: 0,
            horizon: 0,
            converged: false,
            error_message: None,
        }
    }

    /// Set the fit duration (builder pattern).
    pub fn with_fit(mut self, duration: Duration) -> Self {
        self.fit_duration = Some(duration);
        self
    }

    /// Set the predict duration (builder pattern).
    pub fn with_predict(mut self, duration: Duration) -> Self {
        self.predict_duration = Some(duration);
        self
    }

    /// Set the number of observations (builder pattern).
    pub fn with_observations(mut self, n: usize) -> Self {
        self.n_observations = n;
        self
    }

    /// Set the forecast horizon (builder pattern).
    pub fn with_horizon(mut self, h: usize) -> Self {
        self.horizon = h;
        self
    }

    /// Set the convergence status (builder pattern).
    pub fn with_convergence(mut self, converged: bool) -> Self {
        self.converged = converged;
        self
    }

    /// Set an error message (builder pattern).
    pub fn with_error(mut self, msg: impl Into<String>) -> Self {
        self.error_message = Some(msg.into());
        self
    }

    /// Total duration (fit + predict). Returns `Duration::ZERO` if neither is set.
    pub fn total_duration(&self) -> Duration {
        let fit = self.fit_duration.unwrap_or(Duration::ZERO);
        let predict = self.predict_duration.unwrap_or(Duration::ZERO);
        fit + predict
    }

    /// Observations processed per second during fitting.
    /// Returns `None` if `fit_duration` is not set or is zero.
    pub fn observations_per_second(&self) -> Option<f64> {
        let fit = self.fit_duration?;
        let secs = fit.as_secs_f64();
        if secs == 0.0 {
            return None;
        }
        Some(self.n_observations as f64 / secs)
    }
}

impl ExecutionTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Return the elapsed duration since the timer was started.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return the elapsed duration, consuming self.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }
}

impl fmt::Display for ExecutionMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model: {}", self.model_name)?;

        if let Some(fit) = self.fit_duration {
            write!(f, " | fit: {:.3}s", fit.as_secs_f64())?;
        }
        if let Some(predict) = self.predict_duration {
            write!(f, " | predict: {:.3}s", predict.as_secs_f64())?;
        }

        write!(
            f,
            " | converged: {}",
            if self.converged { "yes" } else { "no" }
        )?;

        if let Some(ref err) = self.error_message {
            write!(f, " | error: {}", err)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_new_defaults() {
        let meta = ExecutionMetadata::new("TestModel");
        assert_eq!(meta.model_name, "TestModel");
        assert!(meta.fit_duration.is_none());
        assert!(meta.predict_duration.is_none());
        assert_eq!(meta.n_observations, 0);
        assert_eq!(meta.horizon, 0);
        assert!(!meta.converged);
        assert!(meta.error_message.is_none());
    }

    #[test]
    fn metadata_builder_chain() {
        let meta = ExecutionMetadata::new("ARIMA(1,1,1)")
            .with_fit(Duration::from_millis(100))
            .with_predict(Duration::from_millis(20))
            .with_observations(500)
            .with_horizon(12)
            .with_convergence(true);

        assert_eq!(meta.model_name, "ARIMA(1,1,1)");
        assert_eq!(meta.fit_duration, Some(Duration::from_millis(100)));
        assert_eq!(meta.predict_duration, Some(Duration::from_millis(20)));
        assert_eq!(meta.n_observations, 500);
        assert_eq!(meta.horizon, 12);
        assert!(meta.converged);
        assert!(meta.error_message.is_none());
    }

    #[test]
    fn total_duration_sum() {
        let meta = ExecutionMetadata::new("ETS")
            .with_fit(Duration::from_millis(100))
            .with_predict(Duration::from_millis(50));
        assert_eq!(meta.total_duration(), Duration::from_millis(150));

        // Only fit set
        let meta2 = ExecutionMetadata::new("ETS").with_fit(Duration::from_millis(80));
        assert_eq!(meta2.total_duration(), Duration::from_millis(80));

        // Neither set
        let meta3 = ExecutionMetadata::new("ETS");
        assert_eq!(meta3.total_duration(), Duration::ZERO);
    }

    #[test]
    fn observations_per_second() {
        let meta = ExecutionMetadata::new("ARIMA")
            .with_observations(1000)
            .with_fit(Duration::from_secs(2));
        let ops = meta.observations_per_second().unwrap();
        assert!((ops - 500.0).abs() < 1e-9);

        // No fit duration -> None
        let meta2 = ExecutionMetadata::new("ARIMA").with_observations(1000);
        assert!(meta2.observations_per_second().is_none());

        // Zero duration -> None
        let meta3 = ExecutionMetadata::new("ARIMA")
            .with_observations(1000)
            .with_fit(Duration::ZERO);
        assert!(meta3.observations_per_second().is_none());
    }

    #[test]
    fn timer_elapsed() {
        let timer = ExecutionTimer::start();
        // Do a trivial amount of work to ensure some time passes
        let mut _sum = 0u64;
        for i in 0..1000 {
            _sum += i;
        }
        let elapsed = timer.elapsed();
        // Just check it doesn't panic and returns a duration
        let _ = elapsed.as_nanos();

        // Stop consumes the timer
        let timer2 = ExecutionTimer::start();
        let stopped = timer2.stop();
        let _ = stopped.as_nanos();
    }

    #[test]
    fn display_format() {
        let meta = ExecutionMetadata::new("ARIMA(1,1,1)")
            .with_fit(Duration::from_millis(123))
            .with_predict(Duration::from_millis(45))
            .with_convergence(true);

        let display = format!("{}", meta);
        assert!(display.contains("ARIMA(1,1,1)"));
        assert!(display.contains("fit:"));
        assert!(display.contains("predict:"));
        assert!(display.contains("converged: yes"));
        assert!(!display.contains("error:"));

        // With error
        let meta_err = ExecutionMetadata::new("BadModel").with_error("matrix singular");
        let display_err = format!("{}", meta_err);
        assert!(display_err.contains("BadModel"));
        assert!(display_err.contains("converged: no"));
        assert!(display_err.contains("error: matrix singular"));
    }

    #[test]
    fn with_error_sets_message() {
        let meta = ExecutionMetadata::new("Model").with_error("failed to converge");
        assert_eq!(meta.error_message.as_deref(), Some("failed to converge"));
    }
}
