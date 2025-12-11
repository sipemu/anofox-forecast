//! Time Series Clustering Example
//!
//! This example demonstrates how to cluster time series using
//! Dynamic Time Warping (DTW) and k-means algorithms.
//!
//! Run with: cargo run --example time_series_clustering

use anofox_forecast::clustering::{
    dtw_distance, dtw_distance_normalized, dtw_distance_windowed, dtw_pairwise, dtw_path,
    elbow_inertias, euclidean_distance, kmeans, manhattan_distance, DistanceMetric, KMeansConfig,
};

fn main() {
    println!("=== Time Series Clustering Example ===\n");

    // =========================================================================
    // Dynamic Time Warping Distance
    // =========================================================================
    println!("--- Dynamic Time Warping (DTW) Distance ---\n");

    // Two similar series
    let series_a = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let series_b = vec![1.0, 2.0, 3.0, 2.0, 1.0];

    // Shifted version of series_a
    let series_c = vec![0.5, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5];

    println!("Series A: {:?}", series_a);
    println!("Series B: {:?}", series_b);
    println!("Series C: {:?} (shifted version)", series_c);
    println!();

    // DTW handles different length series and time shifts
    println!("DTW distances:");
    println!(
        "  dtw(A, B) = {:.4} (identical)",
        dtw_distance(&series_a, &series_b)
    );
    println!(
        "  dtw(A, C) = {:.4} (DTW aligns despite shift)",
        dtw_distance(&series_a, &series_c)
    );

    // Compare with Euclidean (only works for same length)
    println!("\nEuclidean distance:");
    println!(
        "  euclidean(A, B) = {:.4}",
        euclidean_distance(&series_a, &series_b)
    );
    println!(
        "  euclidean(A, C) = {} (different lengths)",
        if euclidean_distance(&series_a, &series_c).is_infinite() {
            "undefined"
        } else {
            "calculated"
        }
    );

    // =========================================================================
    // DTW Variants
    // =========================================================================
    println!("\n--- DTW Variants ---\n");

    let s1 = vec![1.0, 3.0, 4.0, 3.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 1.0];
    let s2 = vec![0.0, 1.0, 3.0, 4.0, 3.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 1.0];

    println!("Series 1: {:?}", s1);
    println!("Series 2: {:?}", s2);
    println!();

    // Standard DTW
    let dtw_std = dtw_distance(&s1, &s2);
    println!("Standard DTW:      {:.4}", dtw_std);

    // Windowed DTW (Sakoe-Chiba band)
    let dtw_win_2 = dtw_distance_windowed(&s1, &s2, 2);
    let dtw_win_5 = dtw_distance_windowed(&s1, &s2, 5);
    println!("Windowed DTW (w=2): {:.4}", dtw_win_2);
    println!("Windowed DTW (w=5): {:.4}", dtw_win_5);
    println!("  Note: Larger window allows more warping flexibility");

    // Normalized DTW
    let dtw_norm = dtw_distance_normalized(&s1, &s2);
    println!(
        "Normalized DTW:    {:.4} (divided by path length)",
        dtw_norm
    );

    // =========================================================================
    // DTW Alignment Path
    // =========================================================================
    println!("\n--- DTW Alignment Path ---\n");

    let short_a = vec![1.0, 2.0, 3.0];
    let short_b = vec![1.0, 1.5, 2.0, 2.5, 3.0];

    println!("Series A: {:?}", short_a);
    println!("Series B: {:?}", short_b);

    let path = dtw_path(&short_a, &short_b);
    println!("\nAlignment path (i, j) pairs:");
    for (i, j) in &path {
        println!(
            "  A[{}]={:.1} <-> B[{}]={:.1}",
            i, short_a[*i], j, short_b[*j]
        );
    }
    println!("\nThe path shows how elements are aligned.");
    println!("Multiple B elements can align to single A element.");

    // =========================================================================
    // Distance Metrics Comparison
    // =========================================================================
    println!("\n--- Distance Metrics Comparison ---\n");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];

    println!("Series X: {:?}", x);
    println!("Series Y: {:?}", y);
    println!();

    println!("Distance metrics:");
    println!("  Euclidean (L2):  {:.4}", euclidean_distance(&x, &y));
    println!("  Manhattan (L1):  {:.4}", manhattan_distance(&x, &y));
    println!("  DTW:             {:.4}", dtw_distance(&x, &y));

    println!("\nWhen to use each:");
    println!("  Euclidean: Same-length series, point-to-point comparison");
    println!("  Manhattan: More robust to outliers than Euclidean");
    println!("  DTW: Different lengths, time-shifted patterns");

    // =========================================================================
    // Pairwise Distance Matrix
    // =========================================================================
    println!("\n--- Pairwise Distance Matrix ---\n");

    let series_collection = vec![
        vec![1.0, 2.0, 1.0, 2.0, 1.0], // Pattern A
        vec![1.1, 2.1, 1.1, 2.1, 1.1], // Similar to A
        vec![5.0, 6.0, 5.0, 6.0, 5.0], // Pattern B (high values)
        vec![5.1, 6.1, 5.1, 6.1, 5.1], // Similar to B
    ];

    println!("Computing pairwise DTW distances for 4 series...\n");

    let dist_matrix = dtw_pairwise(&series_collection);

    println!("Distance matrix:");
    print!("      ");
    for i in 0..dist_matrix.len() {
        print!("  S{}   ", i);
    }
    println!();

    for (i, row) in dist_matrix.iter().enumerate() {
        print!("S{}  ", i);
        for &d in row {
            print!("{:6.2} ", d);
        }
        println!();
    }

    println!("\nObservation: S0-S1 and S2-S3 have small distances (similar patterns)");

    // =========================================================================
    // K-Means Clustering
    // =========================================================================
    println!("\n--- K-Means Clustering ---\n");

    // Generate clustered time series data
    let cluster_data = vec![
        // Cluster 1: Low oscillating
        vec![1.0, 2.0, 1.0, 2.0, 1.0],
        vec![1.2, 2.2, 1.2, 2.2, 1.2],
        vec![0.8, 1.8, 0.8, 1.8, 0.8],
        vec![1.1, 2.1, 1.1, 2.1, 1.1],
        // Cluster 2: High oscillating
        vec![10.0, 11.0, 10.0, 11.0, 10.0],
        vec![10.2, 11.2, 10.2, 11.2, 10.2],
        vec![9.8, 10.8, 9.8, 10.8, 9.8],
        vec![10.1, 11.1, 10.1, 11.1, 10.1],
        // Cluster 3: Trending up
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.1, 2.1, 3.1, 4.1, 5.1],
        vec![0.9, 1.9, 2.9, 3.9, 4.9],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ];

    println!("Generated 12 time series in 3 natural clusters:");
    println!("  Series 0-3:  Low oscillating pattern");
    println!("  Series 4-7:  High oscillating pattern");
    println!("  Series 8-11: Trending up pattern");
    println!();

    // K-means with Euclidean distance
    let config = KMeansConfig::default().k(3).seed(42).max_iter(100);

    let result = kmeans(&cluster_data, &config);

    println!("K-Means Results (Euclidean distance):");
    println!("  Number of clusters: {}", result.centroids.len());
    println!("  Iterations: {}", result.n_iter);
    println!("  Inertia: {:.4}", result.inertia);
    println!();

    println!("Cluster assignments:");
    for (i, &label) in result.labels.iter().enumerate() {
        let expected = i / 4; // Expected cluster based on data generation
        let match_str = if label == expected || matches_expected(i, label, &result.labels) {
            "✓"
        } else {
            "?"
        };
        println!("  Series {:2} -> Cluster {} {}", i, label, match_str);
    }

    println!("\nCluster sizes: {:?}", result.cluster_sizes());

    // =========================================================================
    // K-Means with DTW Distance
    // =========================================================================
    println!("\n--- K-Means with DTW Distance ---\n");

    let dtw_config = KMeansConfig::default()
        .k(3)
        .metric(DistanceMetric::DTW)
        .seed(42);

    let dtw_result = kmeans(&cluster_data, &dtw_config);

    println!("K-Means Results (DTW distance):");
    println!("  Iterations: {}", dtw_result.n_iter);
    println!("  Inertia: {:.4}", dtw_result.inertia);
    println!("  Cluster sizes: {:?}", dtw_result.cluster_sizes());

    println!("\nNote: DTW is slower but better for time-shifted patterns");

    // =========================================================================
    // Elbow Method for Optimal K
    // =========================================================================
    println!("\n--- Elbow Method (Optimal K Selection) ---\n");

    let inertias = elbow_inertias(&cluster_data, 6);

    println!("Inertia for different K values:");
    println!("{:>3} {:>12}", "K", "Inertia");
    println!("{}", "-".repeat(17));
    for (k, &inertia) in inertias.iter().enumerate() {
        let k = k + 1;
        let elbow = if k == 3 { " <- elbow" } else { "" };
        println!("{:>3} {:>12.2}{}", k, inertia, elbow);
    }

    println!("\nLook for the 'elbow' where inertia decrease slows down.");
    println!("In this case, K=3 is optimal (matches our 3 natural clusters).");

    // =========================================================================
    // Working with Cluster Results
    // =========================================================================
    println!("\n--- Working with Cluster Results ---\n");

    println!("Cluster members:");
    for cluster_id in 0..result.centroids.len() {
        let members = result.cluster_members(cluster_id);
        println!("  Cluster {}: {:?}", cluster_id, members);
    }

    println!("\nCluster centroids (first 3 values):");
    for (i, centroid) in result.centroids.iter().enumerate() {
        let preview: Vec<String> = centroid
            .iter()
            .take(3)
            .map(|v| format!("{:.2}", v))
            .collect();
        println!("  Cluster {}: [{}...]", i, preview.join(", "));
    }

    // =========================================================================
    // Configuration Options
    // =========================================================================
    println!("\n--- KMeansConfig Options ---\n");

    println!("KMeansConfig::default()");
    println!("  .k(n)              - Number of clusters");
    println!("  .max_iter(n)       - Maximum iterations");
    println!("  .metric(m)         - DistanceMetric::Euclidean or DTW");
    println!("  .seed(n)           - Random seed for reproducibility");
    println!();

    println!("Example:");
    println!("  let config = KMeansConfig::default()");
    println!("      .k(5)");
    println!("      .metric(DistanceMetric::DTW)");
    println!("      .seed(42);");

    // =========================================================================
    // Practical Tips
    // =========================================================================
    println!("\n--- Practical Tips ---\n");

    println!("1. Preprocessing:");
    println!("   - Normalize/standardize series before clustering");
    println!("   - Consider differencing for trend removal");
    println!();

    println!("2. Distance metric choice:");
    println!("   - Euclidean: Fast, requires same-length series");
    println!("   - DTW: Handles shifts/stretches, slower");
    println!();

    println!("3. Choosing K:");
    println!("   - Use elbow method");
    println!("   - Consider domain knowledge");
    println!("   - Try multiple K values and evaluate");
    println!();

    println!("4. DTW windowing:");
    println!("   - Use dtw_distance_windowed for large series");
    println!("   - Reduces computation and prevents extreme warping");
    println!();

    println!("5. Reproducibility:");
    println!("   - Always set seed for reproducible results");
    println!("   - K-means++ initialization depends on random selection");
}

// Helper function to check if clustering is reasonable
fn matches_expected(series_idx: usize, label: usize, labels: &[usize]) -> bool {
    let group_start = (series_idx / 4) * 4;
    let group_labels: Vec<usize> = (group_start..group_start + 4)
        .filter(|&i| i < labels.len())
        .map(|i| labels[i])
        .collect();

    // Check if most of the group has the same label
    group_labels.iter().filter(|&&l| l == label).count() >= 3
}
