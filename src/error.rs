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
    #[error("insufficient data: need at least {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

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
    #[error("model must be fitted before prediction")]
    FitRequired,

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_descriptive() {
        let err = ForecastError::EmptyData;
        assert_eq!(err.to_string(), "empty input data");

        let err = ForecastError::InsufficientData { needed: 10, got: 5 };
        assert_eq!(
            err.to_string(),
            "insufficient data: need at least 10, got 5"
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

        let err = ForecastError::FitRequired;
        assert_eq!(err.to_string(), "model must be fitted before prediction");
    }

    #[test]
    fn errors_are_clonable_and_comparable() {
        let err1 = ForecastError::EmptyData;
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }
}
