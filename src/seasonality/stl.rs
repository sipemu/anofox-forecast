//! STL (Seasonal-Trend decomposition using LOESS) implementation.
//!
//! STL decomposes a time series into three components:
//! - Trend: The underlying long-term pattern
//! - Seasonal: The repeating seasonal pattern
//! - Remainder: The residual after removing trend and seasonal

/// Result of STL decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct STLResult {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder component.
    pub remainder: Vec<f64>,
}

impl STLResult {
    /// Get the seasonal strength (0 to 1).
    /// Values close to 1 indicate strong seasonality.
    pub fn seasonal_strength(&self) -> f64 {
        let var_remainder = variance(&self.remainder);
        let seasonal_plus_remainder: Vec<f64> = self
            .seasonal
            .iter()
            .zip(self.remainder.iter())
            .map(|(s, r)| s + r)
            .collect();
        let var_sr = variance(&seasonal_plus_remainder);

        if var_sr < 1e-10 {
            return 0.0;
        }

        (1.0 - var_remainder / var_sr).max(0.0)
    }

    /// Get the trend strength (0 to 1).
    /// Values close to 1 indicate strong trend.
    pub fn trend_strength(&self) -> f64 {
        let var_remainder = variance(&self.remainder);
        let trend_plus_remainder: Vec<f64> = self
            .trend
            .iter()
            .zip(self.remainder.iter())
            .map(|(t, r)| t + r)
            .collect();
        let var_tr = variance(&trend_plus_remainder);

        if var_tr < 1e-10 {
            return 0.0;
        }

        (1.0 - var_remainder / var_tr).max(0.0)
    }
}

/// Pre-allocated scratch buffers for STL decomposition.
///
/// Reusing this struct across multiple `decompose_with_scratch` calls avoids
/// repeated heap allocation. Buffers are automatically resized as needed.
#[derive(Debug, Clone, Default)]
pub struct StlScratch {
    seasonal: Vec<f64>,
    trend: Vec<f64>,
    weights: Vec<f64>,
    detrended: Vec<f64>,
    deseasonalized: Vec<f64>,
    remainder: Vec<f64>,
    lp_buf_a: Vec<f64>,
    lp_buf_b: Vec<f64>,
    low_pass: Vec<f64>,
    unit_weights: Vec<f64>,
    cycle_subseries: Vec<f64>,
    // Subseries buffers used inside smooth_cycle_subseries_into
    subseries_values: Vec<f64>,
    subseries_weights: Vec<f64>,
    subseries_indices: Vec<usize>,
    smoothed: Vec<f64>,
}

impl StlScratch {
    /// Create a new empty scratch buffer set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize all length-n buffers and zero them out.
    fn prepare(&mut self, n: usize, period: usize) {
        self.seasonal.resize(n, 0.0);
        self.seasonal.fill(0.0);
        self.trend.resize(n, 0.0);
        self.trend.fill(0.0);
        self.weights.resize(n, 0.0);
        self.weights.fill(1.0);
        self.detrended.resize(n, 0.0);
        self.deseasonalized.resize(n, 0.0);
        self.remainder.resize(n, 0.0);
        self.lp_buf_a.resize(n, 0.0);
        self.lp_buf_b.resize(n, 0.0);
        self.low_pass.resize(n, 0.0);
        self.unit_weights.resize(n, 0.0);
        self.unit_weights.fill(1.0);
        self.cycle_subseries.resize(n, 0.0);

        // Subseries buffers: max possible length per subseries
        let max_subseries_len = n.div_ceil(period);
        self.subseries_values
            .reserve(max_subseries_len.saturating_sub(self.subseries_values.capacity()));
        self.subseries_weights
            .reserve(max_subseries_len.saturating_sub(self.subseries_weights.capacity()));
        self.subseries_indices
            .reserve(max_subseries_len.saturating_sub(self.subseries_indices.capacity()));
        self.smoothed
            .reserve(max_subseries_len.saturating_sub(self.smoothed.capacity()));
    }
}

/// STL decomposition configuration and algorithm.
#[derive(Debug, Clone)]
pub struct STL {
    /// Seasonal period.
    seasonal_period: usize,
    /// Seasonal LOESS smoothing parameter (ns).
    seasonal_smoothness: usize,
    /// Trend LOESS smoothing parameter (nt).
    trend_smoothness: usize,
    /// Low-pass filter parameter (nl).
    low_pass_smoothness: usize,
    /// Number of inner iterations.
    inner_iterations: usize,
    /// Number of outer (robustness) iterations.
    outer_iterations: usize,
    /// Use robust fitting.
    robust: bool,
}

impl STL {
    /// Create a new STL decomposer with the given seasonal period.
    pub fn new(seasonal_period: usize) -> Self {
        // Default parameters following Cleveland et al. (1990)
        let ns = seasonal_period;
        let nt = (1.5 * seasonal_period as f64 / (1.0 - 1.5 / ns as f64)).ceil() as usize;
        let nt = if nt % 2 == 0 { nt + 1 } else { nt }; // Must be odd
        let nl = seasonal_period;
        let nl = if nl % 2 == 0 { nl + 1 } else { nl }; // Must be odd

        Self {
            seasonal_period,
            seasonal_smoothness: ns | 1, // Ensure odd
            trend_smoothness: nt,
            low_pass_smoothness: nl,
            inner_iterations: 2,
            outer_iterations: 0,
            robust: false,
        }
    }

    /// Set custom seasonal smoothness (ns parameter).
    pub fn with_seasonal_smoothness(mut self, ns: usize) -> Self {
        self.seasonal_smoothness = if ns % 2 == 0 { ns + 1 } else { ns };
        self
    }

    /// Set custom trend smoothness (nt parameter).
    pub fn with_trend_smoothness(mut self, nt: usize) -> Self {
        self.trend_smoothness = if nt % 2 == 0 { nt + 1 } else { nt };
        self
    }

    /// Enable robust fitting with default iterations.
    pub fn robust(mut self) -> Self {
        self.robust = true;
        self.outer_iterations = 6;
        self
    }

    /// Set number of outer (robustness) iterations.
    pub fn with_outer_iterations(mut self, n: usize) -> Self {
        self.outer_iterations = n;
        if n > 0 {
            self.robust = true;
        }
        self
    }

    /// Set number of inner iterations.
    pub fn with_inner_iterations(mut self, n: usize) -> Self {
        self.inner_iterations = n;
        self
    }

    /// Decompose the time series.
    pub fn decompose(&self, series: &[f64]) -> Option<STLResult> {
        let mut scratch = StlScratch::new();
        self.decompose_with_scratch(series, &mut scratch)
    }

    /// Decompose the time series, reusing pre-allocated scratch buffers.
    ///
    /// This avoids heap allocation when called repeatedly (e.g., across
    /// multiple series of similar length). Pass the same `StlScratch`
    /// instance to successive calls.
    pub fn decompose_with_scratch(
        &self,
        series: &[f64],
        scratch: &mut StlScratch,
    ) -> Option<STLResult> {
        let n = series.len();
        if n < 2 * self.seasonal_period {
            return None;
        }

        let period = self.seasonal_period;
        scratch.prepare(n, period);

        // Outer loop (robustness)
        let outer_iters = if self.robust {
            self.outer_iterations.max(1)
        } else {
            1
        };

        for _ in 0..outer_iters {
            // Inner loop
            for _ in 0..self.inner_iterations {
                // Step 1: Detrending
                for (d, (y, t)) in scratch
                    .detrended
                    .iter_mut()
                    .zip(series.iter().zip(scratch.trend.iter()))
                {
                    *d = y - t;
                }

                // Step 2: Cycle-subseries smoothing
                self.smooth_cycle_subseries_into_scratch(scratch);

                // Step 3: Low-pass filter of smoothed cycle-subseries
                self.low_pass_filter_into(
                    &scratch.cycle_subseries,
                    &mut scratch.lp_buf_a,
                    &mut scratch.lp_buf_b,
                    &mut scratch.low_pass,
                    &scratch.unit_weights,
                );

                // Step 4: Detrending of smoothed cycle-subseries
                for i in 0..n {
                    scratch.seasonal[i] = scratch.cycle_subseries[i] - scratch.low_pass[i];
                }

                // Step 5: Deseasonalizing
                for (d, (y, s)) in scratch
                    .deseasonalized
                    .iter_mut()
                    .zip(series.iter().zip(scratch.seasonal.iter()))
                {
                    *d = y - s;
                }

                // Step 6: Trend smoothing
                Self::loess_smooth_into(
                    &scratch.deseasonalized,
                    self.trend_smoothness,
                    &scratch.weights,
                    &mut scratch.trend,
                );
            }

            // Update robustness weights
            if self.robust {
                for ((r, (y, s)), t) in scratch
                    .remainder
                    .iter_mut()
                    .zip(series.iter().zip(scratch.seasonal.iter()))
                    .zip(scratch.trend.iter())
                {
                    *r = y - s - t;
                }
                scratch.weights = self.compute_robustness_weights(&scratch.remainder);
            }
        }

        // Compute final remainder
        for ((r, (y, s)), t) in scratch
            .remainder
            .iter_mut()
            .zip(series.iter().zip(scratch.seasonal.iter()))
            .zip(scratch.trend.iter())
        {
            *r = y - s - t;
        }

        // Move results out of scratch (replace with empty vecs to allow reuse)
        Some(STLResult {
            trend: std::mem::take(&mut scratch.trend),
            seasonal: std::mem::take(&mut scratch.seasonal),
            remainder: std::mem::take(&mut scratch.remainder),
        })
    }

    /// Smooth cycle-subseries using scratch buffers from StlScratch.
    fn smooth_cycle_subseries_into_scratch(&self, scratch: &mut StlScratch) {
        let n = scratch.detrended.len();
        let period = self.seasonal_period;

        // Process each cycle-subseries (one for each position in the seasonal cycle)
        for cycle_pos in 0..period {
            // Clear and reuse scratch subseries buffers
            scratch.subseries_values.clear();
            scratch.subseries_weights.clear();
            scratch.subseries_indices.clear();

            for i in (cycle_pos..n).step_by(period) {
                scratch.subseries_values.push(scratch.detrended[i]);
                scratch.subseries_weights.push(scratch.weights[i]);
                scratch.subseries_indices.push(i);
            }

            // Smooth the subseries into reusable buffer
            Self::loess_smooth_into(
                &scratch.subseries_values,
                self.seasonal_smoothness,
                &scratch.subseries_weights,
                &mut scratch.smoothed,
            );

            // Put smoothed values back
            for (&idx, &smooth_val) in scratch
                .subseries_indices
                .iter()
                .zip(scratch.smoothed.iter())
            {
                scratch.cycle_subseries[idx] = smooth_val;
            }
        }
    }

    /// Low-pass filter using moving averages, reusing scratch buffers.
    fn low_pass_filter_into(
        &self,
        series: &[f64],
        buf_a: &mut Vec<f64>,
        buf_b: &mut Vec<f64>,
        result: &mut Vec<f64>,
        unit_weights: &[f64],
    ) {
        let period = self.seasonal_period;

        // Apply three moving averages: MA(period), MA(period), MA(3)
        Self::moving_average_into(series, period, buf_a);
        Self::moving_average_into(buf_a, period, buf_b);
        Self::moving_average_into(buf_b, 3, buf_a);

        // Apply LOESS to the result
        Self::loess_smooth_into(buf_a, self.low_pass_smoothness, unit_weights, result);
    }

    /// Simple centered moving average, writing into pre-allocated buffer.
    fn moving_average_into(series: &[f64], window: usize, result: &mut Vec<f64>) {
        let n = series.len();
        let half = window / 2;
        result.resize(n, 0.0);

        for (i, res) in result.iter_mut().enumerate() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let sum: f64 = series[start..end].iter().sum();
            *res = sum / (end - start) as f64;
        }
    }

    /// LOESS smoothing, writing into pre-allocated buffer.
    fn loess_smooth_into(values: &[f64], span: usize, weights: &[f64], result: &mut Vec<f64>) {
        let n = values.len();
        result.resize(n, 0.0);
        if n == 0 {
            return;
        }

        let half_span = span / 2;

        for i in 0..n {
            // Determine window
            let start = i.saturating_sub(half_span);
            let end = (i + half_span + 1).min(n);

            // Compute weighted local regression (simplified to weighted mean)
            let mut sum_weights = 0.0;
            let mut sum_values = 0.0;

            for j in start..end {
                let dist = (i as f64 - j as f64).abs();
                let max_dist = half_span as f64 + 1.0;
                let u = dist / max_dist;
                let tricube = if u < 1.0 {
                    (1.0 - u.powi(3)).powi(3)
                } else {
                    0.0
                };
                let w = tricube * weights[j];
                sum_weights += w;
                sum_values += w * values[j];
            }

            result[i] = if sum_weights > 0.0 {
                sum_values / sum_weights
            } else {
                values[i]
            };
        }
    }

    /// Compute robustness weights based on remainder.
    fn compute_robustness_weights(&self, remainder: &[f64]) -> Vec<f64> {
        let n = remainder.len();

        // Compute median absolute deviation — sort in-place, no extra clone
        let mut sorted: Vec<f64> = remainder.iter().map(|r| r.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let h = 6.0 * median; // Tuning constant

        // Compute bisquare weights
        remainder
            .iter()
            .map(|r| {
                if h < 1e-10 {
                    return 1.0;
                }
                let u = r.abs() / h;
                if u < 1.0 {
                    (1.0 - u * u).powi(2)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl Default for STL {
    fn default() -> Self {
        Self::new(12) // Monthly seasonality default
    }
}

/// Convenience wrapper for STL decomposition with a builder API.
///
/// # Example
///
/// ```
/// use anofox_forecast::seasonality::StlBuilder;
///
/// let series: Vec<f64> = (0..120)
///     .map(|i| {
///         let trend = 0.1 * i as f64;
///         let seasonal = 10.0 * ((2.0 * std::f64::consts::PI * i as f64 / 12.0).sin());
///         trend + seasonal
///     })
///     .collect();
///
/// let result = StlBuilder::new(12)
///     .seasonal_window(7)
///     .trend_window(15)
///     .robust(true)
///     .decompose(&series)
///     .unwrap();
///
/// assert_eq!(result.trend.len(), series.len());
/// ```
#[derive(Debug, Clone)]
pub struct StlBuilder {
    stl: STL,
    scratch: StlScratch,
}

impl StlBuilder {
    /// Create a new STL builder with the given seasonal period.
    pub fn new(period: usize) -> Self {
        Self {
            stl: STL::new(period),
            scratch: StlScratch::new(),
        }
    }

    /// Set the seasonal smoothing window (ns parameter).
    /// The value will be rounded up to odd if even.
    pub fn seasonal_window(mut self, window: usize) -> Self {
        self.stl = self.stl.with_seasonal_smoothness(window);
        self
    }

    /// Set the trend smoothing window (nt parameter).
    /// The value will be rounded up to odd if even.
    pub fn trend_window(mut self, window: usize) -> Self {
        self.stl = self.stl.with_trend_smoothness(window);
        self
    }

    /// Enable or disable robust fitting.
    /// When enabled, uses 6 outer iterations by default.
    pub fn robust(mut self, enable: bool) -> Self {
        if enable {
            self.stl = self.stl.robust();
        }
        self
    }

    /// Set the number of inner iterations.
    pub fn inner_iterations(mut self, n: usize) -> Self {
        self.stl = self.stl.with_inner_iterations(n);
        self
    }

    /// Set the number of outer (robustness) iterations.
    pub fn outer_iterations(mut self, n: usize) -> Self {
        self.stl = self.stl.with_outer_iterations(n);
        self
    }

    /// Run STL decomposition on the given series.
    ///
    /// This allocates fresh scratch buffers. For repeated calls on
    /// series of similar length, use [`decompose_reuse`](Self::decompose_reuse)
    /// which caches buffers across calls.
    pub fn decompose(&self, series: &[f64]) -> Option<STLResult> {
        self.stl.decompose(series)
    }

    /// Run STL decomposition, reusing cached scratch buffers.
    ///
    /// More efficient than [`decompose`](Self::decompose) when called
    /// repeatedly (e.g., decomposing many series of similar length),
    /// because heap allocations are amortized across calls.
    pub fn decompose_reuse(&mut self, series: &[f64]) -> Option<STLResult> {
        self.stl.decompose_with_scratch(series, &mut self.scratch)
    }

    /// Get a reference to the underlying STL configuration.
    pub fn config(&self) -> &STL {
        &self.stl
    }
}

/// Compute variance.
fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_seasonal_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal =
                    10.0 * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin());
                trend + seasonal
            })
            .collect()
    }

    #[test]
    fn stl_basic_decomposition() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.seasonal.len(), series.len());
        assert_eq!(result.remainder.len(), series.len());

        // Verify additive decomposition: y = trend + seasonal + remainder
        for i in 0..series.len() {
            let reconstructed = result.trend[i] + result.seasonal[i] + result.remainder[i];
            assert!(
                (series[i] - reconstructed).abs() < 1e-10,
                "Reconstruction failed at index {}: {} vs {}",
                i,
                series[i],
                reconstructed
            );
        }
    }

    #[test]
    fn stl_detects_seasonality() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        // Should detect strong seasonality
        let strength = result.seasonal_strength();
        assert!(
            strength > 0.5,
            "Expected strong seasonality, got {}",
            strength
        );
    }

    #[test]
    fn stl_detects_trend() {
        let n = 120;
        let period = 12;
        // Strong trend with weak seasonality
        let series: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 2.0 * i as f64;
                let seasonal =
                    0.1 * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin());
                trend + seasonal
            })
            .collect();

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        let strength = result.trend_strength();
        assert!(strength > 0.9, "Expected strong trend, got {}", strength);
    }

    #[test]
    fn stl_trend_only() {
        let n = 100;
        let period = 10;
        // Only trend, no seasonality
        let series: Vec<f64> = (0..n).map(|i| 5.0 + 0.5 * i as f64).collect();

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        // Seasonal component should be small
        let seasonal_var = variance(&result.seasonal);
        let series_var = variance(&series);
        assert!(
            seasonal_var < series_var * 0.1,
            "Seasonal variance {} should be small compared to series variance {}",
            seasonal_var,
            series_var
        );
    }

    #[test]
    fn stl_constant_series() {
        let n = 100;
        let period = 10;
        let series = vec![5.0; n];

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        // All components should be flat/zero
        for &s in &result.seasonal {
            assert!(s.abs() < 1e-6, "Seasonal should be near zero");
        }
        for &r in &result.remainder {
            assert!(r.abs() < 1e-6, "Remainder should be near zero");
        }
    }

    #[test]
    fn stl_insufficient_data() {
        let period = 12;
        let series = vec![1.0; 10]; // Less than 2 * period

        let stl = STL::new(period);
        assert!(stl.decompose(&series).is_none());
    }

    #[test]
    fn stl_robust_decomposition() {
        let period = 12;
        let mut series = generate_seasonal_series(120, period);
        // Add outliers
        series[30] = 100.0;
        series[60] = -100.0;

        let stl = STL::new(period).robust();
        let result = stl.decompose(&series).unwrap();

        // Robust fitting should still capture some pattern
        let strength = result.seasonal_strength();
        assert!(
            strength > 0.1,
            "Robust STL should still detect seasonality: {}",
            strength
        );
    }

    #[test]
    fn stl_custom_smoothness() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period)
            .with_seasonal_smoothness(7)
            .with_trend_smoothness(21)
            .with_inner_iterations(3);

        let result = stl.decompose(&series).unwrap();
        assert_eq!(result.trend.len(), series.len());
    }

    #[test]
    fn stl_different_periods() {
        // Weekly (period 7)
        let series_weekly = generate_seasonal_series(70, 7);
        let stl_weekly = STL::new(7);
        assert!(stl_weekly.decompose(&series_weekly).is_some());

        // Quarterly (period 4)
        let series_quarterly = generate_seasonal_series(40, 4);
        let stl_quarterly = STL::new(4);
        assert!(stl_quarterly.decompose(&series_quarterly).is_some());
    }

    #[test]
    fn stl_result_seasonal_strength_range() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        let strength = result.seasonal_strength();
        assert!(
            (0.0..=1.0).contains(&strength),
            "Seasonal strength should be in [0, 1]: {}",
            strength
        );
    }

    #[test]
    fn stl_result_trend_strength_range() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period);
        let result = stl.decompose(&series).unwrap();

        let strength = result.trend_strength();
        assert!(
            (0.0..=1.0).contains(&strength),
            "Trend strength should be in [0, 1]: {}",
            strength
        );
    }

    // ==================== StlBuilder tests ====================

    #[test]
    fn stl_builder_basic() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let result = StlBuilder::new(period).decompose(&series).unwrap();

        assert_eq!(result.trend.len(), series.len());
        assert_eq!(result.seasonal.len(), series.len());
        assert_eq!(result.remainder.len(), series.len());

        // Verify additive decomposition
        for i in 0..series.len() {
            let reconstructed = result.trend[i] + result.seasonal[i] + result.remainder[i];
            assert!(
                (series[i] - reconstructed).abs() < 1e-10,
                "Reconstruction failed at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_builder_with_all_options() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let result = StlBuilder::new(period)
            .seasonal_window(7)
            .trend_window(15)
            .robust(true)
            .inner_iterations(3)
            .outer_iterations(4)
            .decompose(&series)
            .unwrap();

        assert_eq!(result.trend.len(), series.len());
        assert!(result.seasonal_strength() > 0.0);
    }

    #[test]
    fn stl_builder_robust_with_outliers() {
        let period = 12;
        let mut series = generate_seasonal_series(120, period);
        series[30] = 100.0;
        series[60] = -100.0;

        let result = StlBuilder::new(period)
            .robust(true)
            .decompose(&series)
            .unwrap();

        let strength = result.seasonal_strength();
        assert!(
            strength > 0.1,
            "Robust builder should detect seasonality: {}",
            strength,
        );
    }

    #[test]
    fn stl_builder_robust_false_is_noop() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        // robust(false) should behave the same as no robust call
        let result_default = StlBuilder::new(period).decompose(&series).unwrap();
        let result_no_robust = StlBuilder::new(period)
            .robust(false)
            .decompose(&series)
            .unwrap();

        for i in 0..series.len() {
            assert!(
                (result_default.trend[i] - result_no_robust.trend[i]).abs() < 1e-10,
                "robust(false) should match default at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_builder_insufficient_data() {
        let period = 12;
        let series = vec![1.0; 10];

        assert!(StlBuilder::new(period).decompose(&series).is_none());
    }

    #[test]
    fn stl_builder_matches_stl_direct() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let direct = STL::new(period)
            .with_seasonal_smoothness(7)
            .with_trend_smoothness(21)
            .decompose(&series)
            .unwrap();

        let builder = StlBuilder::new(period)
            .seasonal_window(7)
            .trend_window(21)
            .decompose(&series)
            .unwrap();

        for i in 0..series.len() {
            assert!(
                (direct.trend[i] - builder.trend[i]).abs() < 1e-10,
                "Builder and direct STL should match at index {}",
                i,
            );
            assert!(
                (direct.seasonal[i] - builder.seasonal[i]).abs() < 1e-10,
                "Builder and direct STL seasonal should match at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_builder_config_access() {
        let builder = StlBuilder::new(12).seasonal_window(7).trend_window(15);

        let config = builder.config();
        // Just verify we can access the underlying config without panic
        let _ = format!("{:?}", config);
    }

    // ==================== StlScratch / buffer reuse tests ====================

    #[test]
    fn stl_decompose_with_scratch_matches_decompose() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let stl = STL::new(period);
        let result_alloc = stl.decompose(&series).unwrap();

        let mut scratch = StlScratch::new();
        let result_scratch = stl.decompose_with_scratch(&series, &mut scratch).unwrap();

        for i in 0..series.len() {
            assert!(
                (result_alloc.trend[i] - result_scratch.trend[i]).abs() < 1e-10,
                "Scratch decompose trend should match at index {}",
                i,
            );
            assert!(
                (result_alloc.seasonal[i] - result_scratch.seasonal[i]).abs() < 1e-10,
                "Scratch decompose seasonal should match at index {}",
                i,
            );
            assert!(
                (result_alloc.remainder[i] - result_scratch.remainder[i]).abs() < 1e-10,
                "Scratch decompose remainder should match at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_scratch_reuse_across_calls() {
        let period = 12;
        let series_a = generate_seasonal_series(120, period);
        let series_b = generate_seasonal_series(96, period);

        let stl = STL::new(period);
        let mut scratch = StlScratch::new();

        // First call
        let result_a = stl.decompose_with_scratch(&series_a, &mut scratch).unwrap();
        assert_eq!(result_a.trend.len(), 120);

        // Second call reuses scratch (different length series)
        let result_b = stl.decompose_with_scratch(&series_b, &mut scratch).unwrap();
        assert_eq!(result_b.trend.len(), 96);

        // Verify correctness of second call against fresh decompose
        let reference = stl.decompose(&series_b).unwrap();
        for i in 0..96 {
            assert!(
                (result_b.trend[i] - reference.trend[i]).abs() < 1e-10,
                "Reused scratch should produce correct results at index {}",
                i,
            );
        }
    }

    #[test]
    fn stl_builder_decompose_reuse() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let mut builder = StlBuilder::new(period).seasonal_window(7).trend_window(15);

        // decompose_reuse should produce the same results as decompose
        let result_alloc = builder.decompose(&series).unwrap();
        let result_reuse = builder.decompose_reuse(&series).unwrap();

        for i in 0..series.len() {
            assert!(
                (result_alloc.trend[i] - result_reuse.trend[i]).abs() < 1e-10,
                "Builder decompose_reuse trend should match at index {}",
                i,
            );
        }
    }
}
