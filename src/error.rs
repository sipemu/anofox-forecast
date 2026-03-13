//! Error types for the anofox-forecast library.

use thiserror::Error;

/// Result type alias for forecast operations.
pub type Result<T> = std::result::Result<T, ForecastError>;

/// Errors that can occur during forecasting operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ForecastError {
    /// Input data is empty.
    #[error("empty input data")]
    EmptyData,

    /// Insufficient data points for the operation.
    #[error("insufficient data: need at least {needed}, got {got}{}", .hint.as_deref().map(|h| format!(" ({})", h)).unwrap_or_default())]
    InsufficientData {
        needed: usize,
        got: usize,
        hint: Option<String>,
    },

    /// Invalid parameter value.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Dimension mismatch between data structures.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Timestamp-related error.
    #[error("timestamp error: {0}")]
    TimestampError(String),

    /// Model has not been fitted yet.
    #[error("{}", match model { Some(m) => format!("Model '{}' must be fitted before prediction", m), None => "model must be fitted before prediction".to_string() })]
    FitRequired { model: Option<String> },

    /// A sub-model within an ensemble or composite model failed.
    #[error("sub-model '{model_name}' failed: {source}")]
    SubModelError {
        model_name: String,
        source: Box<ForecastError>,
    },

    /// Missing values detected when not allowed.
    #[error("missing values detected in data")]
    MissingValues,

    /// Frequency inference failed.
    #[error("could not infer frequency: {0}")]
    FrequencyInference(String),

    /// Index out of bounds.
    #[error("index out of bounds: {index} (size: {size})")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Computation error (e.g., numerical issues).
    #[error("computation error: {0}")]
    ComputationError(String),

    /// Convergence failure during optimization or model fitting.
    #[error("convergence failure: {0}")]
    ConvergenceFailure(String),

    /// Linear algebra failure (e.g., singular matrix, failed decomposition).
    #[error("singular matrix: {0}")]
    SingularMatrix(String),

    /// Serialization or persistence error.
    #[error("serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_descriptive() {
        let err = ForecastError::EmptyData;
        assert_eq!(err.to_string(), "empty input data");

        let err = ForecastError::InsufficientData {
            needed: 10,
            got: 5,
            hint: None,
        };
        assert_eq!(
            err.to_string(),
            "insufficient data: need at least 10, got 5"
        );

        let err = ForecastError::InsufficientData {
            needed: 10,
            got: 5,
            hint: Some("test hint".into()),
        };
        assert_eq!(
            err.to_string(),
            "insufficient data: need at least 10, got 5 (test hint)"
        );

        let err = ForecastError::InvalidParameter("window must be positive".to_string());
        assert_eq!(
            err.to_string(),
            "invalid parameter: window must be positive"
        );

        let err = ForecastError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 3, got 2");

        let err = ForecastError::FitRequired { model: None };
        assert_eq!(err.to_string(), "model must be fitted before prediction");

        let err = ForecastError::FitRequired {
            model: Some("TestModel".to_string()),
        };
        assert_eq!(
            err.to_string(),
            "Model 'TestModel' must be fitted before prediction"
        );

        let inner = ForecastError::EmptyData;
        let err = ForecastError::SubModelError {
            model_name: "SES".to_string(),
            source: Box::new(inner),
        };
        assert_eq!(err.to_string(), "sub-model 'SES' failed: empty input data");
    }

    #[test]
    fn errors_are_clonable_and_comparable() {
        let err1 = ForecastError::EmptyData;
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }
}
