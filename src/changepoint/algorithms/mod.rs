//! Trait-based changepoint-detection algorithms.
//!
//! Each detector implements [`Detector`](crate::changepoint::Detector)
//! and is generic over a [`Cost`](crate::changepoint::Cost) (except
//! [`KernelCpdDetector`] which has its own kernel choice).

pub mod binseg;
pub mod bottom_up;
pub mod dynp;
pub mod kernel_cpd;
pub mod pelt;
pub mod window;

pub use binseg::BinsegDetector;
pub use bottom_up::BottomUpDetector;
pub use dynp::DynpDetector;
pub use kernel_cpd::{KernelCpdDetector, KernelKind};
pub use pelt::PeltDetector;
pub use window::WindowDetector;
