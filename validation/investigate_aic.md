# AutoETS AIC/AICc Calculation Discrepancy

## Problem

statsforecast and Rust forecast library produce drastically different AICc values for the same ETS models on noisy_seasonal data.

## Evidence

###stats forecast (Python):
- Selected: ETS(A,N,N)
- Alpha: 0.0001
- AIC: 884.75
- AICc: 885.00
- BIC: 892.56
- Log-likelihood: -439.37
- n_params: 3
- Sigma2: 68.25

### Rust:
- A,N,N: AICc = 718.65
- A,A,N: AICc = 709.62
- A,N,A: AICc = 672.70
- A,N,M: AICc = 672.70 (SELECTED)
- A,A,A: AICc = 680.92

## Key Differences

1. **statsforecast AICc ~885** vs **Rust A,N,N AICc ~719**: Difference of ~166
2. Rust selects seasonal model (A,N,M) while statsforecast selects non-seasonal (A,N,N)

## Potential Causes

### 1. Number of Parameters (k)
statsforecast reports `n_params=3` for ETS(A,N,N):
- alpha (1)
- initial level (1)
- sigma^2 (1)

Rust `num_params()` for ETS(A,N,N) counts:
```rust
fn num_params(&self) -> usize {
    let mut count = 1; // alpha
    // ... no trend
    // ... no seasonal
    count += 1; // initial level
    // ... no trend initial
    // ... no seasonals
    count
}
```
Returns 2 for A,N,N (alpha + initial level).

**BUT**: Looking at the code, statsforecast includes sigma^2 as a parameter!

### 2. Log-Likelihood Formula
statsforecast formula (from output):
```
ll = -0.5 * n * (1 + ln(sigma2) + ln(2*pi))
```

Rust formula (from ets.rs line 940):
```rust
let ll = -0.5 * n * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());
```

**These appear identical!**

### 3. Residuals Calculation
statsforecast:
- Uses `n = len(residuals)` for LL calculation
- Excludes initial observations from residuals

Rust (from ets.rs line 931-938):
```rust
let valid_residuals: Vec<f64> = residuals[start_idx..].to_vec();
let n = valid_residuals.len() as f64;
let variance = valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
```

For seasonal models, `start_idx = period = 12`, so we lose 12 observations!

###  4. AIC Formula

stats forecast:
```
AIC = -2 * LL + 2 * k
```

Rust:
```rust
self.aic = Some(-2.0 * ll + 2.0 * k);
```

**Identical!**

### 5. AICc Formula

statsforecast:
```
AICc = -2 * LL + 2 * k * n / (n - k - 1)
```

Rust:
```rust
self.aicc = Some(-2.0 * ll + 2.0 * k * n / (n - k - 1.0).max(1.0));
```

**Identical!**

## ROOT CAUSE HYPOTHESIS

The issue is in **how residuals are counted**:

1. **statsforecast** likely calculates LL over ALL observations (n=100)
2. **Rust** calculates LL over observations AFTER initial period (n=88 for seasonal)

This dramatically affects both LL and AIC:
- Smaller n → Higher (better) log-likelihood
- Higher LL → Lower (better) AIC/AICc

## CRITICAL ISSUE IN ETS.RS

Line 931-938:
```rust
let valid_residuals: Vec<f64> = residuals[start_idx..].to_vec();
if !valid_residuals.is_empty() {
    let variance = valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
    self.residual_variance = Some(variance);

    // Calculate information criteria
    let n = valid_residuals.len() as f64;  // <-- PROBLEM!
```

For non-seasonal (start_idx=0): n = 100
For seasonal (start_idx=12): n = 88

This gives seasonal models an unfair advantage!

## FIX

The `n` used in AIC calculations should be the **total sample size**, not the number of fitted residuals. However, we should still calculate variance only over valid residuals.

Change:
```rust
let valid_residuals: Vec<f64> = residuals[start_idx..].to_vec();
if !valid_residuals.is_empty() {
    let variance = valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
    self.residual_variance = Some(variance);

    // Use TOTAL sample size for AIC, not just fitted residuals
    let n = self.n as f64;  // <-- FIX: Use total sample size
    let k = self.num_params() as f64;
    let ll = -0.5 * valid_residuals.len() as f64 * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());
```

Actually wait... let me check statsforecast more carefully.
