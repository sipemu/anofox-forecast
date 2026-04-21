//! Transfer Entropy (TE) — directional information flow from source to target.
//!
//! `TE(X → Y, h) = I(Y_t; X_{t-h} | Y_{t-1}, …, Y_{t-h+1})`
//!
//! Measures how much additional information the source X provides about the
//! future of Y beyond Y's own history. Implemented as conditional MI via
//! linear residualization + kNN MI.

use super::ami::linear_residualize;
use super::knn_mi::knn_mutual_information;

/// Compute the Transfer Entropy curve from `source` to `target` for
/// lags h = 1..max_lag.
///
/// `TE(X → Y, h) = I(Y_t; X_{t-h} | Y_{t-1}, …, Y_{t-h+1})`
///
/// At lag 1, there is no conditioning (TE = MI between X_{t-1} and Y_t).
/// At lag h > 1, Y's own recent history is conditioned out via linear
/// residualization.
///
/// # Arguments
/// * `source` — the driving series X (length N)
/// * `target` — the response series Y (length N)
/// * `max_lag` — number of lags to compute
///
/// # Returns
/// `Vec<f64>` of length `max_lag`, where `result[h-1]` is TE at lag h.
pub fn transfer_entropy_curve(source: &[f64], target: &[f64], max_lag: usize) -> Vec<f64> {
    let n = source.len();
    assert_eq!(
        n,
        target.len(),
        "source and target must have the same length"
    );
    let k = 8;
    let mut result = Vec::with_capacity(max_lag);

    for h in 1..=max_lag {
        if n <= h + k {
            result.push(0.0);
            continue;
        }

        let usable = n - h;

        // Y_future = target[h..h+usable] — what we're trying to predict.
        let y_future = &target[h..h + usable];
        // X_past = source[0..usable] at lag h — the candidate predictor.
        let x_past = &source[..usable];

        if h == 1 {
            // No conditioning: TE(1) = I(Y_{t}; X_{t-1}).
            result.push(knn_mutual_information(y_future, x_past, k));
            continue;
        }

        // Conditioning: Z = [Y_{t-1}, Y_{t-2}, …, Y_{t-h+1}] = target
        // history at lags 1..h-1 relative to the prediction target.
        let z_cols: Vec<Vec<f64>> = (1..h)
            .map(|j| target[h - j..h - j + usable].to_vec())
            .collect();

        let y_resid = linear_residualize(y_future, &z_cols);
        let x_resid = linear_residualize(x_past, &z_cols);

        if y_resid.len() < k + 1 {
            result.push(0.0);
            continue;
        }

        result.push(knn_mutual_information(&y_resid, &x_resid, k));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn te_from_driver_to_follower_is_positive() {
        // X drives Y with a 1-step lag: Y_t = 0.7 * X_{t-1} + noise.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 500;
        let x: Vec<f64> = (0..n).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = 0.7 * x[t - 1] + (rng.gen::<f64>() - 0.5) * 0.5;
        }

        let te = transfer_entropy_curve(&x, &y, 3);
        assert!(te[0] > 0.2, "TE(1) should be large: {:.4}", te[0]);
    }

    #[test]
    fn te_from_independent_is_near_zero() {
        // Use properly random-looking sequences to avoid deterministic
        // correlations between sin/modular-arithmetic patterns.
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let x: Vec<f64> = (0..400).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();
        let y: Vec<f64> = (0..400).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();
        let te = transfer_entropy_curve(&x, &y, 3);
        for &v in &te {
            assert!(v < 0.4, "TE from independent should be near 0: {:.4}", v);
        }
    }
}
