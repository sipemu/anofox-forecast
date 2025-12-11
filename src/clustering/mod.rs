//! Time series clustering algorithms.
//!
//! Provides Dynamic Time Warping (DTW) distance measures and k-means clustering.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::clustering::{dtw_distance, kmeans, KMeansConfig, DistanceMetric};
//!
//! // Compute DTW distance between two series
//! let a = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let dist = dtw_distance(&a, &b);
//! assert_eq!(dist, 0.0);
//!
//! // Cluster time series
//! let series = vec![
//!     vec![1.0, 2.0, 1.0],
//!     vec![1.1, 2.1, 1.1],
//!     vec![10.0, 11.0, 10.0],
//!     vec![10.1, 11.1, 10.1],
//! ];
//! let config = KMeansConfig::default().k(2).seed(42);
//! let result = kmeans(&series, &config);
//! assert_eq!(result.centroids.len(), 2);
//! ```

pub mod dtw;
pub mod kmeans;

// Re-export from dtw
pub use dtw::{
    dtw_distance, dtw_distance_normalized, dtw_distance_windowed, dtw_pairwise, dtw_path,
    euclidean_distance, manhattan_distance,
};

// Re-export from kmeans
pub use kmeans::{elbow_inertias, kmeans, DistanceMetric, KMeansConfig, KMeansResult};
