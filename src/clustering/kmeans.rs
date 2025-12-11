//! K-means clustering for time series.
//!
//! Provides k-means with DTW or Euclidean distance.

use super::dtw::{dtw_distance, euclidean_distance};

/// Distance metric for clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (requires same-length series)
    Euclidean,
    /// Dynamic Time Warping distance
    DTW,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Euclidean
    }
}

/// K-means configuration.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub k: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Random seed for initialization
    pub seed: Option<u64>,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 3,
            max_iter: 100,
            metric: DistanceMetric::Euclidean,
            seed: None,
            tolerance: 1e-4,
        }
    }
}

impl KMeansConfig {
    /// Set number of clusters.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set distance metric.
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// K-means clustering result.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster assignments for each series (0-indexed)
    pub labels: Vec<usize>,
    /// Cluster centroids
    pub centroids: Vec<Vec<f64>>,
    /// Inertia (sum of distances to nearest centroid)
    pub inertia: f64,
    /// Number of iterations performed
    pub n_iter: usize,
}

impl KMeansResult {
    /// Get indices of series in a specific cluster.
    pub fn cluster_members(&self, cluster: usize) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the size of each cluster.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let k = self.centroids.len();
        let mut sizes = vec![0; k];
        for &label in &self.labels {
            if label < k {
                sizes[label] += 1;
            }
        }
        sizes
    }
}

/// Perform k-means clustering on time series.
///
/// # Arguments
/// * `series` - Vector of time series
/// * `config` - K-means configuration
pub fn kmeans(series: &[Vec<f64>], config: &KMeansConfig) -> KMeansResult {
    let n = series.len();
    let k = config.k.min(n);

    if n == 0 || k == 0 {
        return KMeansResult {
            labels: Vec::new(),
            centroids: Vec::new(),
            inertia: 0.0,
            n_iter: 0,
        };
    }

    // Initialize centroids (k-means++ style)
    let mut centroids = initialize_centroids(series, k, config);

    let mut labels = vec![0; n];
    let mut prev_inertia = f64::INFINITY;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Assignment step
        let mut inertia = 0.0;
        for (i, s) in series.iter().enumerate() {
            let (nearest, dist) = find_nearest_centroid(s, &centroids, config.metric);
            labels[i] = nearest;
            inertia += dist;
        }

        // Check convergence
        if (prev_inertia - inertia).abs() < config.tolerance {
            break;
        }
        prev_inertia = inertia;

        // Update step
        centroids = update_centroids(series, &labels, k, config.metric);
    }

    // Final inertia
    let inertia = compute_inertia(series, &labels, &centroids, config.metric);

    KMeansResult {
        labels,
        centroids,
        inertia,
        n_iter,
    }
}

/// Initialize centroids using k-means++ algorithm.
fn initialize_centroids(series: &[Vec<f64>], k: usize, config: &KMeansConfig) -> Vec<Vec<f64>> {
    let n = series.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::with_capacity(k);

    // Use seed for reproducibility if provided
    let first_idx = if let Some(seed) = config.seed {
        (seed as usize) % n
    } else {
        0
    };

    centroids.push(series[first_idx].clone());

    // Add remaining centroids
    for _ in 1..k {
        // Compute distances to nearest existing centroid
        let mut distances: Vec<f64> = series
            .iter()
            .map(|s| {
                centroids
                    .iter()
                    .map(|c| compute_distance(s, c, config.metric))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Convert to probabilities
        let sum: f64 = distances.iter().sum();
        if sum > 0.0 {
            for d in &mut distances {
                *d /= sum;
            }
        }

        // Select next centroid (proportional to squared distance)
        let mut cumsum = 0.0;
        let threshold = if let Some(seed) = config.seed {
            ((seed + centroids.len() as u64) % 1000) as f64 / 1000.0
        } else {
            0.5
        };

        let mut selected = n - 1;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }

        centroids.push(series[selected].clone());
    }

    centroids
}

/// Find the nearest centroid for a series.
fn find_nearest_centroid(
    series: &[f64],
    centroids: &[Vec<f64>],
    metric: DistanceMetric,
) -> (usize, f64) {
    let mut min_dist = f64::INFINITY;
    let mut nearest = 0;

    for (i, centroid) in centroids.iter().enumerate() {
        let dist = compute_distance(series, centroid, metric);
        if dist < min_dist {
            min_dist = dist;
            nearest = i;
        }
    }

    (nearest, min_dist)
}

/// Compute distance between two series.
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DTW => dtw_distance(a, b),
    }
}

/// Update centroids based on cluster assignments.
fn update_centroids(
    series: &[Vec<f64>],
    labels: &[usize],
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<f64>> {
    let mut centroids = Vec::with_capacity(k);

    for cluster in 0..k {
        let members: Vec<&Vec<f64>> = series
            .iter()
            .zip(labels.iter())
            .filter(|(_, &l)| l == cluster)
            .map(|(s, _)| s)
            .collect();

        if members.is_empty() {
            // Keep a placeholder for empty clusters
            centroids.push(vec![0.0]);
        } else {
            match metric {
                DistanceMetric::Euclidean => {
                    // Simple mean
                    centroids.push(compute_mean_series(&members));
                }
                DistanceMetric::DTW => {
                    // For DTW, use medoid (series with minimum total distance to others)
                    centroids.push(compute_medoid(&members, metric));
                }
            }
        }
    }

    centroids
}

/// Compute element-wise mean of multiple series.
fn compute_mean_series(series: &[&Vec<f64>]) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let n = series.len();
    let len = series[0].len();

    (0..len)
        .map(|i| {
            series
                .iter()
                .filter_map(|s| s.get(i).copied())
                .sum::<f64>()
                / n as f64
        })
        .collect()
}

/// Compute medoid (series minimizing total distance to others).
fn compute_medoid(series: &[&Vec<f64>], metric: DistanceMetric) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }
    if series.len() == 1 {
        return series[0].clone();
    }

    let mut min_total_dist = f64::INFINITY;
    let mut medoid_idx = 0;

    for (i, s1) in series.iter().enumerate() {
        let total_dist: f64 = series
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s2)| compute_distance(s1, s2, metric))
            .sum();

        if total_dist < min_total_dist {
            min_total_dist = total_dist;
            medoid_idx = i;
        }
    }

    series[medoid_idx].clone()
}

/// Compute inertia (total within-cluster sum of distances).
fn compute_inertia(
    series: &[Vec<f64>],
    labels: &[usize],
    centroids: &[Vec<f64>],
    metric: DistanceMetric,
) -> f64 {
    series
        .iter()
        .zip(labels.iter())
        .map(|(s, &l)| {
            if l < centroids.len() {
                compute_distance(s, &centroids[l], metric)
            } else {
                0.0
            }
        })
        .sum()
}

/// Elbow method helper: compute inertia for different k values.
pub fn elbow_inertias(series: &[Vec<f64>], max_k: usize) -> Vec<f64> {
    (1..=max_k.min(series.len()))
        .map(|k| {
            let config = KMeansConfig::default().k(k);
            let result = kmeans(series, &config);
            result.inertia
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn generate_cluster_data() -> Vec<Vec<f64>> {
        vec![
            // Cluster 1: low values
            vec![1.0, 2.0, 1.0, 2.0, 1.0],
            vec![1.5, 2.5, 1.5, 2.5, 1.5],
            vec![1.2, 2.2, 1.2, 2.2, 1.2],
            // Cluster 2: high values
            vec![10.0, 11.0, 10.0, 11.0, 10.0],
            vec![10.5, 11.5, 10.5, 11.5, 10.5],
            vec![10.2, 11.2, 10.2, 11.2, 10.2],
        ]
    }

    // ==================== kmeans ====================

    #[test]
    fn kmeans_finds_clusters() {
        let data = generate_cluster_data();
        let config = KMeansConfig::default().k(2).seed(42);
        let result = kmeans(&data, &config);

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.centroids.len(), 2);

        // First 3 should be in same cluster, last 3 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], vec![0.9, 1.9, 2.9]];
        let config = KMeansConfig::default().k(1);
        let result = kmeans(&data, &config);

        // All should be in cluster 0
        assert!(result.labels.iter().all(|&l| l == 0));
        assert_eq!(result.centroids.len(), 1);
    }

    #[test]
    fn kmeans_k_equals_n() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let config = KMeansConfig::default().k(3);
        let result = kmeans(&data, &config);

        // Each point should be its own cluster
        assert_eq!(result.centroids.len(), 3);
        // Inertia should be 0 (each point is at its centroid)
        assert_relative_eq!(result.inertia, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn kmeans_empty() {
        let data: Vec<Vec<f64>> = vec![];
        let config = KMeansConfig::default();
        let result = kmeans(&data, &config);

        assert!(result.labels.is_empty());
        assert!(result.centroids.is_empty());
    }

    #[test]
    fn kmeans_with_dtw() {
        let data = generate_cluster_data();
        let config = KMeansConfig::default().k(2).metric(DistanceMetric::DTW).seed(42);
        let result = kmeans(&data, &config);

        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.centroids.len(), 2);
    }

    // ==================== cluster_members ====================

    #[test]
    fn cluster_members_basic() {
        let data = generate_cluster_data();
        let config = KMeansConfig::default().k(2).seed(42);
        let result = kmeans(&data, &config);

        let c0_members = result.cluster_members(0);
        let c1_members = result.cluster_members(1);

        // All indices should be accounted for
        assert_eq!(c0_members.len() + c1_members.len(), 6);
    }

    // ==================== cluster_sizes ====================

    #[test]
    fn cluster_sizes_basic() {
        let data = generate_cluster_data();
        let config = KMeansConfig::default().k(2).seed(42);
        let result = kmeans(&data, &config);

        let sizes = result.cluster_sizes();

        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0] + sizes[1], 6);
        // With clear clusters, should be 3 and 3
        assert_eq!(sizes[0], 3);
        assert_eq!(sizes[1], 3);
    }

    // ==================== elbow_inertias ====================

    #[test]
    fn elbow_inertias_decreasing() {
        let data = generate_cluster_data();
        let inertias = elbow_inertias(&data, 4);

        // Inertia should generally decrease as k increases
        assert_eq!(inertias.len(), 4);
        for i in 1..inertias.len() {
            assert!(inertias[i] <= inertias[i - 1] + 1e-6);
        }
    }

    // ==================== config builder ====================

    #[test]
    fn config_builder() {
        let config = KMeansConfig::default()
            .k(5)
            .max_iter(50)
            .metric(DistanceMetric::DTW)
            .seed(123);

        assert_eq!(config.k, 5);
        assert_eq!(config.max_iter, 50);
        assert_eq!(config.metric, DistanceMetric::DTW);
        assert_eq!(config.seed, Some(123));
    }

    #[test]
    fn distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Euclidean);
    }
}
