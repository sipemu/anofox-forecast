# Proposed Fixes for HoltWinters Implementation

## Fix #1: Add Seasonal Component Normalization

**Priority**: HIGH
**Impact**: Should reduce MAD from 4.37 to < 2.0
**File**: `/home/simonm/projects/rust/forecast/src/models/exponential/holt_winters.rs`

### Problem
Seasonal components can drift over time without explicit normalization. Standard implementations normalize after initialization and periodically during updates.

### Solution
Add a helper method to normalize seasonal components and call it after initialization and optionally after each complete seasonal cycle.

```rust
/// Normalize seasonal components to maintain constraints
fn normalize_seasonals(seasonals: &mut [f64], seasonal_type: SeasonalType) {
    let period = seasonals.len();

    match seasonal_type {
        SeasonalType::Additive => {
            // Ensure seasonals sum to 0
            let sum: f64 = seasonals.iter().sum();
            let adjustment = sum / period as f64;
            for s in seasonals.iter_mut() {
                *s -= adjustment;
            }
        }
        SeasonalType::Multiplicative => {
            // Ensure seasonals average to 1
            let mean: f64 = seasonals.iter().sum::<f64>() / period as f64;
            if mean.abs() > 1e-10 {
                for s in seasonals.iter_mut() {
                    *s /= mean;
                }
            }
        }
    }
}
```

### Changes Required

1. **After initialization** (line ~193):
```rust
let (level, trend, mut seasonals) =
    Self::initialize_state(values, period, seasonal_type);
Self::normalize_seasonals(&mut seasonals, seasonal_type); // ADD THIS
```

2. **In calculate_sse** (after line 210):
```rust
let (mut level, mut trend, mut seasonals) =
    Self::initialize_state(values, period, seasonal_type);
Self::normalize_seasonals(&mut seasonals, seasonal_type); // ADD THIS
```

3. **In fit** (after line 322):
```rust
let (mut level, mut trend, mut seasonals) =
    Self::initialize_state(values, period, self.seasonal_type);
Self::normalize_seasonals(&mut seasonals, self.seasonal_type); // ADD THIS
```

4. **Optional: Periodic normalization during updates** (after every complete cycle):
```rust
// In the update loop, add cycle counter
for (t, &y) in values.iter().enumerate().skip(period) {
    // ... existing update code ...

    // Normalize after completing each full seasonal cycle
    if (t + 1) % period == 0 {
        Self::normalize_seasonals(&mut seasonals, seasonal_type);
    }
}
```

---

## Fix #2: Alternative Seasonal Update Formula

**Priority**: MEDIUM (test if Fix #1 doesn't fully resolve)
**Impact**: May improve alignment by 10-30%
**File**: `/home/simonm/projects/rust/forecast/src/models/exponential/holt_winters.rs`

### Problem
Different Holt-Winters implementations use different forms of the seasonal update equation. Current code uses:
```rust
s_t = γ(y_t - l_t) + (1-γ)s_{t-m}
```

But some implementations (possibly statsforecast) use:
```rust
s_t = γ(y_t - l_t - b_t) + (1-γ)s_{t-m}
```

### Solution Option A: Include Trend in Seasonal Update

Change line 233 from:
```rust
seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
```

To:
```rust
seasonals[season_idx] = gamma * (y - level - trend) + (1.0 - gamma) * s;
```

And change line 239-244 (multiplicative) similarly.

### Solution Option B: Use Previous State

Change line 233 to use previous level and trend:
```rust
let detrended = level_prev + trend;  // Predicted deseasonalized value
seasonals[season_idx] = gamma * (y - detrended) + (1.0 - gamma) * s;
```

**Note**: Test both options A and B to see which matches statsforecast better.

---

## Fix #3: Optimize Starting Point

**Priority**: LOW
**Impact**: May improve convergence by 5-15%
**File**: `/home/simonm/projects/rust/forecast/src/models/exponential/holt_winters.rs`

### Problem
Current starting point `[0.3, 0.1, 0.1]` might not be optimal for all series types.

### Solution
Use multiple starting points or adaptive initialization:

```rust
fn optimize_params(
    values: &[f64],
    period: usize,
    seasonal_type: SeasonalType,
) -> (f64, f64, f64) {
    let config = NelderMeadConfig {
        max_iter: 1000,
        tolerance: 1e-8,
        ..Default::default()
    };

    // Try multiple starting points and pick best
    let starting_points = vec![
        [0.3, 0.1, 0.1],   // Current default
        [0.1, 0.05, 0.05], // More conservative
        [0.5, 0.1, 0.1],   // More aggressive level
        [0.2, 0.01, 0.1],  // Less trend
    ];

    let mut best_sse = f64::MAX;
    let mut best_params = [0.3, 0.1, 0.1];

    for start in starting_points {
        let result = nelder_mead(
            |params| {
                Self::calculate_sse(
                    values,
                    params[0],
                    params[1],
                    params[2],
                    period,
                    seasonal_type,
                )
            },
            &start,
            Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999)]),
            config.clone(),
        );

        let sse = Self::calculate_sse(
            values,
            result.optimal_point[0],
            result.optimal_point[1],
            result.optimal_point[2],
            period,
            seasonal_type,
        );

        if sse < best_sse {
            best_sse = sse;
            best_params = result.optimal_point;
        }
    }

    (
        best_params[0].clamp(0.0001, 0.9999),
        best_params[1].clamp(0.0001, 0.9999),
        best_params[2].clamp(0.0001, 0.9999),
    )
}
```

---

## Fix #4: Match SSE Calculation Exactly

**Priority**: EXPERIMENTAL
**Impact**: Unknown - for research/debugging only

### Investigation Needed
To perfectly match statsforecast, we might need to:

1. **Check if statsforecast excludes more observations from SSE**:
   - Current: `skip(period)` - excludes first season
   - Alternative: `skip(2 * period)` - excludes first two seasons?

2. **Verify error calculation**:
   - Current: One-step-ahead forecast error
   - Verify this matches statsforecast's metric

3. **Check for mean normalization**:
   - Some implementations normalize SSE by number of observations
   - Current: Raw SSE
   - Alternative: `sse / (n - period) as f64`

---

## Testing Strategy

1. **Apply Fix #1 (normalization) first** - Most likely to help
2. **Measure new MAD** - Should drop below 2.0
3. **If still > 1.0, try Fix #2** - Test both Option A and Option B
4. **If still issues, try Fix #3** - Multiple starting points
5. **Document findings** - Which fixes helped most

---

## Expected Results

| Fix Applied | Expected MAD | Notes |
|-------------|--------------|-------|
| None (current) | 4.37 | Baseline |
| Fix #1 only | 1.5 - 2.5 | Normalization should help significantly |
| Fix #1 + #2 | 0.5 - 1.5 | Should be close to statsforecast |
| Fix #1 + #2 + #3 | 0.3 - 1.0 | Near-perfect match possible |

Note: MAD < 1.0 is considered excellent agreement for time series forecasting.

---

## Implementation Priority Order

1. **Start with Fix #1** (normalization) - Easiest to implement, likely biggest impact
2. **If MAD still > 2.0, add Fix #2 Option A** (include trend in seasonal update)
3. **If MAD still > 1.5, try Fix #2 Option B** instead of A
4. **Only if needed, implement Fix #3** (multiple starting points)

Each fix should be tested independently before combining.
