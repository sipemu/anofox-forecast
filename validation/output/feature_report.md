# Feature Validation Report

Generated: 2025-12-12 15:57:19

## Summary

- **Rust implementation**: anofox-forecast features module
- **Python implementation**: tsfresh

- **Total Rust features**: 1210
- **Total tsfresh features**: 1210
- **Matched comparisons**: 1166

### Overall Match Rate: 93.31%
- Matching: 1088/1166

## Perfectly Matching Features

**97 features** have 100% match rate across all series types:

- `value__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"`
- `value__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"`
- `value__approximate_entropy__m_2__r_0.2`
- `value__ar_coefficient__coeff_0__k_10`
- `value__absolute_sum_of_changes`
- `value__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"`
- `value__agg_autocorrelation__f_agg_"mean"__maxlag_10`
- `value__ar_coefficient__coeff_3__k_10`
- `value__autocorrelation__lag_1`
- `value__autocorrelation__lag_2`
- `value__autocorrelation__lag_10`
- `value__autocorrelation__lag_5`
- `value__binned_entropy__max_bins_10`
- `value__c3__lag_1`
- `value__autocorrelation__lag_3`
- `value__c3__lag_2`
- `value__c3__lag_3`
- `value__change_quantiles__f_agg_"mean"__isabs_false__qh_1.0__ql_0.0`
- `value__change_quantiles__f_agg_"mean"__isabs_true__qh_1.0__ql_0.0`
- `value__cid_ce__normalize_false`
- ... and 77 more

## Features with Discrepancies

| Feature | Match Rate | Mean Abs Diff | Max Abs Diff |
|---------|------------|---------------|--------------|
| `value__agg_autocorrelation__f_agg_"var"__maxlag_10` | 0.00% | 0.011662 | 0.042257 |
| `value__agg_autocorrelation__f_agg_"std"__maxlag_10` | 0.00% | 0.013531 | 0.033359 |
| `value__change_quantiles__f_agg_"var"__isabs_true__` | 0.00% | 0.141409 | 0.466078 |
| `value__kurtosis` | 0.00% | 0.049359 | 0.072286 |
| `value__skewness` | 0.00% | 0.003636 | 0.021255 |
| `value__sample_entropy` | 27.27% | 0.022595 | 0.136512 |
| `value__linear_trend__attr_"pvalue"` | 36.36% | 0.000991 | 0.003221 |
| `value__linear_trend__attr_"rvalue"` | 63.64% | 0.067778 | 0.329968 |
| `value__agg_linear_trend__attr_"rvalue"__chunk_len_` | 63.64% | 0.198295 | 0.962754 |

## Largest Discrepancies (Top 20)

| Series | Feature | Rust | tsfresh | Difference |
|--------|---------|------|---------|------------|
| stationary | `value__agg_linear_trend__attr_"rvalue"__` | 0.481377 | -0.481377 | 0.962754 |
| high_frequency | `value__agg_linear_trend__attr_"rvalue"__` | 0.256066 | -0.256066 | 0.512133 |
| multiplicative_seasonal | `value__change_quantiles__f_agg_"var"__is` | 18.643101 | 18.177024 | 0.466078 |
| seasonal_negative | `value__agg_linear_trend__attr_"rvalue"__` | 0.189358 | -0.189358 | 0.378716 |
| high_frequency | `value__linear_trend__attr_"rvalue"` | 0.164984 | -0.164984 | 0.329968 |
| seasonal | `value__agg_linear_trend__attr_"rvalue"__` | 0.163821 | -0.163821 | 0.327642 |
| stationary | `value__linear_trend__attr_"rvalue"` | 0.158291 | -0.158291 | 0.316582 |
| noisy_seasonal | `value__change_quantiles__f_agg_"var"__is` | 5.489863 | 5.270268 | 0.219595 |
| structural_break | `value__change_quantiles__f_agg_"var"__is` | 4.380053 | 4.197551 | 0.182502 |
| seasonal | `value__change_quantiles__f_agg_"var"__is` | 5.425953 | 5.250922 | 0.175031 |
| stationary | `value__sample_entropy` | 2.549445 | 2.412933 | 0.136512 |
| trend_seasonal | `value__change_quantiles__f_agg_"var"__is` | 5.336251 | 5.202845 | 0.133406 |
| trend | `value__change_quantiles__f_agg_"var"__is` | 4.701074 | 4.596605 | 0.104468 |
| long_memory | `value__change_quantiles__f_agg_"var"__is` | 2.081601 | 2.001539 | 0.080062 |
| intermittent | `value__kurtosis` | 0.539341 | 0.467056 | 0.072286 |
| long_memory | `value__kurtosis` | 0.350825 | 0.282291 | 0.068534 |
| stationary | `value__change_quantiles__f_agg_"var"__is` | 1.759738 | 1.692056 | 0.067682 |
| noisy_seasonal | `value__kurtosis` | -0.147411 | -0.206031 | 0.058619 |
| stationary | `value__kurtosis` | -0.162264 | -0.220587 | 0.058324 |
| high_frequency | `value__change_quantiles__f_agg_"var"__is` | 1.922794 | 1.864528 | 0.058266 |

## Match Rate by Series Type

| Series Type | Match Rate | Matching/Total |
|-------------|------------|----------------|
| high_frequency | 91.51% | 97/106 |
| intermittent | 93.40% | 99/106 |
| long_memory | 93.40% | 99/106 |
| multiplicative_seasonal | 95.28% | 101/106 |
| noisy_seasonal | 94.34% | 100/106 |
| seasonal | 91.51% | 97/106 |
| seasonal_negative | 91.51% | 97/106 |
| stationary | 91.51% | 97/106 |
| structural_break | 94.34% | 100/106 |
| trend | 94.34% | 100/106 |
| trend_seasonal | 95.28% | 101/106 |

## Notes

Expected causes of discrepancies:
- Different numerical precision in implementations
- Different algorithm implementations (e.g., entropy calculations)
- Different handling of edge cases (empty series, constant values)
- Different default parameter values
- Boolean features may have different representations

Tolerance thresholds used:
- Relative tolerance: 1e-05
- Absolute tolerance: 1e-09
