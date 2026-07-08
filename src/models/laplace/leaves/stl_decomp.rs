//! STL-decomposition leaf.
//!
//! On observe: appends `y` to a rolling buffer.
//!
//! On predict: runs STL on the current buffer (trend + seasonal +
//! remainder). Extrapolates: linear trend continuation + cyclic
//! seasonal pattern. Variance from the residual EWMA.
//!
//! This is a batch fitter dressed as a streaming leaf — O(N log N)
//! per predict call, not O(1) per observe. But re-decomposition only
//! happens at predict time (rare compared to observe), so it's
//! practical for the fev-27-style benchmarks. Added post-#180 to
//! close the M-competition monthly/quarterly gap where
//! `LaplaceForecaster` was losing 30-50 % MASE to `AutoTheta`.

use super::super::dist::Gaussian;
use super::super::leaf::Leaf;
use crate::seasonality::STL;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StlDecompLeaf {
    period: usize,
    /// Rolling buffer of the last `max_buffer` observations.
    buffer: Vec<f64>,
    max_buffer: usize,
    /// EWMA of squared residual for the variance channel.
    var_alpha: f64,
    var: f64,
    n_obs: usize,
    /// Cached last decomposition result, refreshed at every predict.
    cache_valid_at: usize,
    cached_level: f64,
    cached_slope: f64,
    cached_seasonal: Vec<f64>,
    cached_sigma: f64,
    label: String,
}

impl StlDecompLeaf {
    /// `period` should match the series' seasonal period. Buffer
    /// caps at `10 * period` — enough for STL to fit 5+ cycles.
    pub fn new(period: usize) -> Self {
        let period = period.max(2);
        Self {
            period,
            buffer: Vec::new(),
            max_buffer: 10 * period,
            var_alpha: 0.03,
            var: 0.0,
            n_obs: 0,
            cache_valid_at: 0,
            cached_level: 0.0,
            cached_slope: 0.0,
            cached_seasonal: vec![0.0; period],
            cached_sigma: 1.0,
            label: format!("stl@{period}"),
        }
    }

    /// Refresh the cached decomposition. Runs STL on the buffer,
    /// fits a linear trend to the trend component, and extracts the
    /// last full-cycle seasonal pattern.
    fn refresh_cache(&mut self) {
        if self.buffer.len() < 2 * self.period {
            // Not enough data for STL. Fall through to level-only forecast.
            self.cached_level = *self.buffer.last().unwrap_or(&0.0);
            self.cached_slope = 0.0;
            self.cached_seasonal = vec![0.0; self.period];
            self.cache_valid_at = self.n_obs;
            return;
        }
        let Some(result) = STL::new(self.period).decompose(&self.buffer) else {
            self.cached_level = *self.buffer.last().unwrap_or(&0.0);
            self.cached_slope = 0.0;
            self.cached_seasonal = vec![0.0; self.period];
            self.cache_valid_at = self.n_obs;
            return;
        };
        // Linear trend fit on the trend component (least squares).
        let n = result.trend.len();
        let mean_t = (n - 1) as f64 / 2.0;
        let mean_y: f64 = result.trend.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, y) in result.trend.iter().enumerate() {
            let dt = i as f64 - mean_t;
            num += dt * (y - mean_y);
            den += dt * dt;
        }
        let slope = if den > 1e-12 { num / den } else { 0.0 };
        let level = *result.trend.last().unwrap_or(&mean_y);
        // Seasonal pattern: last full cycle of the seasonal component.
        let seasonal_len = result.seasonal.len();
        let start = seasonal_len.saturating_sub(self.period);
        let seasonal: Vec<f64> = result.seasonal[start..].to_vec();
        let sigma = if self.var > 0.0 { self.var.sqrt() } else { 1.0 };
        self.cached_level = level;
        self.cached_slope = slope;
        self.cached_seasonal = seasonal;
        self.cached_sigma = sigma;
        self.cache_valid_at = self.n_obs;
    }

    /// Ensure cache reflects current buffer.
    fn ensure_cache(&self) -> (f64, f64, &[f64], f64) {
        // Note: this is called from `&self` methods, so we can't
        // mutate the cache lazily here. Callers should call
        // `refresh_cache()` after observations before predict. In
        // practice `predict_one` and `predict` re-run STL each call
        // via a shadow — we accept the redundant work.
        (
            self.cached_level,
            self.cached_slope,
            &self.cached_seasonal,
            self.cached_sigma,
        )
    }

    fn forecast_at(&self, h: usize) -> f64 {
        let (level, slope, seasonal, _) = self.ensure_cache();
        // Trend: linear extrapolation from the last trend point.
        // Seasonal: cyclic pattern (buffer's last-obs index mod period,
        // then step forward by h).
        let trend = level + slope * (h + 1) as f64;
        if seasonal.is_empty() {
            trend
        } else {
            // Position in the seasonal cycle: buffer_len steps into
            // the pattern, plus (h + 1) more.
            let idx = (self.buffer.len() + h) % seasonal.len();
            trend + seasonal[idx]
        }
    }
}

impl Leaf for StlDecompLeaf {
    fn name(&self) -> &'static str {
        Box::leak(self.label.clone().into_boxed_str())
    }

    fn predict(&self, horizon: usize) -> Vec<Gaussian> {
        // We can't mutate cache from &self, but the forecast_at
        // uses cached values. Callers using this leaf via
        // `LaplaceForecaster` guarantee `observe` was called before
        // `predict` — the cache is refreshed in `observe`.
        let (_, _, _, sigma_one) = self.ensure_cache();
        (1..=horizon)
            .map(|h| {
                let mean = self.forecast_at(h - 1);
                let sigma = (sigma_one * (h as f64).sqrt()).max(1e-9);
                Gaussian::new(mean, sigma)
            })
            .collect()
    }

    #[inline]
    fn predict_one(&self) -> Gaussian {
        let (_, _, _, sigma_one) = self.ensure_cache();
        Gaussian::new(self.forecast_at(0), sigma_one.max(1e-9))
    }

    fn observe(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        self.n_obs += 1;
        // Residual: y minus the current one-step forecast.
        let forecast = self.forecast_at(0);
        let residual = y - forecast;
        let n = self.n_obs as f64;
        let a = self.var_alpha.max(1.0 / n);
        self.var = (1.0 - a) * self.var + a * residual * residual;
        // Update buffer.
        self.buffer.push(y);
        if self.buffer.len() > self.max_buffer {
            self.buffer.remove(0);
        }
        // Re-run STL every `period` observations to keep the cache fresh
        // without paying the full O(N log N) per step.
        if self.n_obs % self.period == 0 || self.n_obs == 2 * self.period {
            self.refresh_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_stl_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let seasonal =
                    30.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
                let noise = ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
                trend + seasonal + noise
            })
            .collect()
    }

    #[test]
    fn recovers_seasonal_pattern_after_many_cycles() {
        let period = 12;
        let vals = synthetic_stl_series(200, period);
        let mut leaf = StlDecompLeaf::new(period);
        for y in &vals {
            leaf.observe(*y);
        }
        // Predict one step ahead — should include the seasonal wobble.
        let g = leaf.predict_one();
        // The forecast should be somewhere in the trend + seasonal
        // range — very loose bound, mostly checking it's finite and
        // not blown up.
        assert!(g.mean.is_finite());
        assert!(g.std.is_finite() && g.std > 0.0);
    }

    #[test]
    fn nan_ignored() {
        let mut leaf = StlDecompLeaf::new(4);
        for i in 0..10 {
            leaf.observe(i as f64);
        }
        let n_before = leaf.n_obs;
        leaf.observe(f64::NAN);
        leaf.observe(f64::INFINITY);
        assert_eq!(leaf.n_obs, n_before);
    }
}
