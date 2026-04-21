//! Largest Lyapunov Exponent (LLE) via the Rosenstein et al. (1993) algorithm.
//!
//! Estimates the rate of divergence of nearby trajectories in a
//! delay-embedding reconstruction of the time series. A positive LLE
//! indicates chaos; zero or negative indicates regular / periodic dynamics.
//!
//! Steps:
//! 1. Embed the scalar series into `m`-dimensional vectors via Takens delay
//!    embedding: `v_i = (x_i, x_{i+τ}, …, x_{i+(m-1)τ})`.
//! 2. For each embedded point, find the nearest neighbor at least `theiler`
//!    time steps away (Theiler window to avoid temporal correlations).
//! 3. Track the mean log-divergence of these neighbor pairs over time.
//! 4. The LLE is the slope of the mean log-divergence curve in its linear
//!    growth regime.

/// Estimate the Largest Lyapunov Exponent.
///
/// # Arguments
/// * `series` — the scalar time series
/// * `m` — embedding dimension (typical: 2–7)
/// * `tau` — delay (typical: 1 or the first minimum of AMI)
/// * `theiler` — Theiler window (minimum temporal separation for neighbors;
///   typical: tau or 2*tau)
/// * `max_iter` — how many steps to track divergence (typical: n/10)
///
/// # Returns
/// The estimated LLE (slope of mean log-divergence). Positive = chaos.
/// Returns `None` if the series is too short for the given parameters.
pub fn largest_lyapunov_exponent(
    series: &[f64],
    m: usize,
    tau: usize,
    theiler: usize,
    max_iter: usize,
) -> Option<f64> {
    let n = series.len();
    let n_embed = n.checked_sub((m - 1) * tau)?;
    if n_embed < 2 * theiler + 2 || max_iter == 0 {
        return None;
    }

    // Step 1: Build delay embedding vectors.
    // v_i = (series[i], series[i+tau], …, series[i+(m-1)*tau])
    let embed = |i: usize| -> Vec<f64> { (0..m).map(|k| series[i + k * tau]).collect() };

    // Step 2: Find nearest neighbor for each point (Theiler window).
    let mut nn_idx = vec![0usize; n_embed];
    for i in 0..n_embed {
        let vi = embed(i);
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for j in 0..n_embed {
            if (i as isize - j as isize).unsigned_abs() < theiler {
                continue;
            }
            let vj = embed(j);
            let dist: f64 = vi
                .iter()
                .zip(vj.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist && dist > 0.0 {
                best_dist = dist;
                best_j = j;
            }
        }
        nn_idx[i] = best_j;
    }

    // Step 3: Track mean log-divergence.
    let mut divergence = vec![0.0_f64; max_iter];
    let mut counts = vec![0usize; max_iter];

    for i in 0..n_embed {
        let j = nn_idx[i];
        for k in 0..max_iter {
            let i_k = i + k;
            let j_k = j + k;
            if i_k + (m - 1) * tau >= n || j_k + (m - 1) * tau >= n {
                break;
            }
            let vi = embed(i_k);
            let vj = embed(j_k);
            let dist: f64 = vi
                .iter()
                .zip(vj.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist > 0.0 {
                divergence[k] += dist.ln();
                counts[k] += 1;
            }
        }
    }

    // Mean log-divergence curve.
    let curve: Vec<f64> = divergence
        .iter()
        .zip(counts.iter())
        .map(|(&d, &c)| if c > 0 { d / c as f64 } else { f64::NAN })
        .collect();

    // Step 4: Fit slope in the "linear growth" region.
    // Use the first half of non-NaN values.
    let valid: Vec<(f64, f64)> = curve
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i as f64, v))
        .collect();

    if valid.len() < 3 {
        return None;
    }

    let use_n = (valid.len() / 2).max(3);
    let pts = &valid[..use_n];

    // Simple linear regression: slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / use_n as f64;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / use_n as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for &(x, y) in pts {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    if sxx < 1e-30 {
        return None;
    }

    Some(sxy / sxx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lle_sine_is_near_zero_or_negative() {
        // Pure sine wave: periodic → LLE ≤ 0.
        let series: Vec<f64> = (0..500)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 50.0).sin())
            .collect();
        let lle = largest_lyapunov_exponent(&series, 3, 10, 20, 50);
        assert!(lle.is_some());
        let val = lle.unwrap();
        assert!(
            val < 0.5,
            "periodic LLE should be near 0 or negative, got {}",
            val
        );
    }

    #[test]
    fn lle_returns_none_for_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        assert!(largest_lyapunov_exponent(&series, 3, 1, 2, 5).is_none());
    }

    #[test]
    fn lle_logistic_map_is_positive() {
        // Logistic map at r = 3.9: chaotic → LLE > 0.
        let n = 1000;
        let mut series = vec![0.0; n];
        series[0] = 0.1;
        for i in 1..n {
            series[i] = 3.9 * series[i - 1] * (1.0 - series[i - 1]);
        }
        let lle = largest_lyapunov_exponent(&series, 3, 1, 5, 50);
        assert!(lle.is_some());
        let val = lle.unwrap();
        assert!(
            val > 0.0,
            "chaotic logistic map LLE should be positive, got {}",
            val
        );
    }
}
