//! Theoretical AMI for validation.

/// Theoretical AMI of an AR(1) process with coefficient φ.
///
/// `I(X_t; X_{t+h}) = -0.5 * ln(1 - φ^{2h})`
///
/// Returns MI in nats for each lag h = 1..max_lag.
pub fn ar1_theoretical_ami(phi: f64, max_lag: usize) -> Vec<f64> {
    (1..=max_lag)
        .map(|h| {
            let rho2 = phi.powi(2 * h as i32);
            if rho2 >= 1.0 {
                f64::INFINITY
            } else {
                -0.5 * (1.0 - rho2).ln()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ar1_phi0_is_zero() {
        let ami = ar1_theoretical_ami(0.0, 5);
        for &v in &ami {
            assert_relative_eq!(v, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn ar1_decays_with_lag() {
        let ami = ar1_theoretical_ami(0.7, 10);
        for i in 1..ami.len() {
            assert!(ami[i] < ami[i - 1], "AMI should decay for AR(1)");
        }
    }

    #[test]
    fn ar1_phi09_lag1_matches_known_value() {
        // I(1) = -0.5 * ln(1 - 0.81) ≈ 0.8340
        let ami = ar1_theoretical_ami(0.9, 1);
        assert_relative_eq!(ami[0], -0.5 * (1.0 - 0.81_f64).ln(), epsilon = 1e-10);
    }
}
