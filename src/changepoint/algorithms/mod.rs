//! Trait-based changepoint-detection algorithms.
//!
//! Each detector implements [`Detector`](crate::changepoint::Detector)
//! and is generic over a [`Cost`](crate::changepoint::Cost).

pub mod pelt;

pub use pelt::PeltDetector;
