# Forecast Validation Report

Generated: 2026-03-04 16:50:00

## Summary

- **Rust implementation**: anofox-forecast
- **Python implementation**: statsforecast (NIXTLA)
- **Forecast horizon**: 12 steps
- **Confidence levels**: 80%, 90%, 95%

- **Models compared**: 29
- **Series types**: 25
- **Total comparisons**: 725

- **High agreement (corr >= 0.99)**: 333 combinations
- **Lower agreement (corr < 0.95)**: 101 combinations

---

## Results by Model

### ADIDA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 3.1949 | 3.1949 | 3.1949 | N/A | N/A |
| damped_trend | 0.0022 | 0.0022 | 0.0022 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0030 | 0.0030 | 0.0030 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0008 | 0.0008 | 0.0008 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 3.0990 | 3.0990 | 3.0990 | N/A | N/A |
| seasonal_negative | 2.6849 | 2.6849 | 2.6849 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 12.7968 | 12.7968 | 12.7968 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### ARIMA_1_1_1

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.5507 | 0.5532 | 0.9670 | 0.6701 | 5.7185 |
| asymmetric_seasonal | 3.6361 | 3.9829 | 4.5617 | 0.9992 | 19.2354 |
| bimodal_seasonal | 0.1889 | 0.1889 | 0.3514 | 0.6285 | -8.2734 |
| damped_trend | 0.5267 | 0.5228 | 0.7932 | 0.7065 | 10.9775 |
| exponential_trend | 14.0471 | 15.1872 | 20.4325 | -0.9303 | -7.3167 |
| heteroscedastic | 0.8009 | 0.8114 | 0.9481 | 0.9822 | 21.4527 |
| high_frequency | 0.0302 | 0.0301 | 0.0530 | 0.8263 | 4.8605 |
| intermittent | 0.5158 | 0.5113 | 0.5962 | 0.8384 | 16.7521 |
| long_memory | 1.9778 | 2.0124 | 2.3446 | -0.6938 | 27.5581 |
| low_count | 0.0642 | 0.0644 | 0.0893 | 0.9061 | 15.3640 |
| multiplicative_seasonal | 0.5980 | 0.2722 | 1.8862 | 0.5668 | -72.9068 |
| multiplicative_trend_seasonal | 0.6692 | 0.5664 | 1.7998 | 0.8135 | -38.0016 |
| noisy_seasonal | 0.1798 | 0.1798 | 0.2262 | 0.7992 | 48.1947 |
| outlier_series | 2.5767 | 2.6738 | 4.0815 | 0.7241 | 29.7959 |
| quarterly_seasonal | 17.0104 | 16.9333 | 18.9572 | 0.2335 | 67.7511 |
| random_walk | 1.2726 | 1.2727 | 2.3504 | 0.3968 | -0.8590 |
| seasonal | 0.6782 | 0.7615 | 0.8181 | 0.9980 | -20.0008 |
| seasonal_negative | 0.2886 | 0.2946 | 0.5394 | 0.9949 | -37.9256 |
| seasonal_trend_break | 0.5254 | 0.5583 | 0.8189 | 0.9536 | -19.5721 |
| stationary | 0.0428 | 0.0429 | 0.0598 | 0.5039 | 22.6524 |
| step_seasonal | 0.2147 | 0.2147 | 0.3992 | 0.3693 | 1.4291 |
| strong_seasonal | 2.4550 | 2.3856 | 5.0535 | 0.8540 | -132.2136 |
| structural_break | 1.5564 | 1.5554 | 2.6015 | -0.4815 | 18.8294 |
| trend | 4.0220 | 4.0286 | 6.6559 | -0.5139 | 10.1677 |
| trend_seasonal | 2.1006 | 2.0889 | 4.0191 | 0.6912 | -9.4782 |


### AutoARIMA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.2606 | 0.1970 | 0.7865 | -0.0713 | -3.2661 |
| asymmetric_seasonal | 0.3416 | 0.3363 | 0.7180 | 0.9992 | 2.2370 |
| bimodal_seasonal | 1.1255 | 0.8066 | 2.9762 | 0.9829 | 3.7886 |
| damped_trend | 0.6555 | 0.6491 | 1.0338 | 0.7844 | 0.2868 |
| exponential_trend | 0.5082 | 0.3381 | 1.3608 | 0.9978 | -0.3749 |
| heteroscedastic | 25.0142 | 27.3301 | 35.2558 | -0.2993 | -52.4051 |
| high_frequency | 2.0222 | 2.1034 | 3.1872 | 0.0096 | -5.7310 |
| intermittent | 0.7783 | 0.7597 | 1.3850 | 0.0270 | 0.7565 |
| long_memory | 0.6590 | 0.6758 | 1.1062 | N/A | -0.8499 |
| low_count | 1.1964 | 1.1782 | 2.7088 | -0.3649 | -2.6445 |
| multiplicative_seasonal | 4.4409 | 4.8193 | 7.7560 | 0.9920 | -7.0886 |
| multiplicative_trend_seasonal | 1.4404 | 1.0948 | 3.3142 | 0.9955 | 0.1243 |
| noisy_seasonal | 6.3639 | 5.9302 | 11.2442 | 0.5436 | -16.5376 |
| outlier_series | 5.0819 | 4.9416 | 6.6603 | 0.9887 | -6.3254 |
| quarterly_seasonal | 1.3383 | 1.3619 | 1.5452 | 0.9999 | 1.8964 |
| random_walk | 0.7126 | 0.7678 | 1.1492 | N/A | -4.8306 |
| seasonal | 0.4609 | 0.3932 | 1.0145 | 0.9976 | 2.3282 |
| seasonal_negative | 0.5893 | 0.7181 | 0.9555 | 0.9990 | 0.7007 |
| seasonal_trend_break | 1.6678 | 1.8585 | 2.7927 | 0.9949 | -3.5558 |
| stationary | 1.5613 | 1.7288 | 2.5277 | 0.4534 | 2.2856 |
| step_seasonal | 1.1308 | 1.1370 | 1.4845 | 0.9996 | 0.3922 |
| strong_seasonal | 1.2396 | 1.2531 | 1.4936 | 1.0000 | 1.6361 |
| structural_break | 6.6233 | 5.3615 | 14.8929 | N/A | 1.3931 |
| trend | 4.4623 | 4.4124 | 8.4350 | N/A | -6.0240 |
| trend_seasonal | 2.7755 | 3.1102 | 3.9134 | 0.9778 | -4.1365 |


### AutoETS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.5668 | 0.4200 | 1.4253 | N/A | -7.4105 |
| asymmetric_seasonal | 0.1004 | 0.0662 | 0.2731 | 0.9999 | -0.4168 |
| bimodal_seasonal | 0.1521 | 0.1839 | 0.3248 | 0.9997 | N/A |
| damped_trend | 0.7226 | 0.5787 | 1.6605 | 0.3311 | -0.9294 |
| exponential_trend | 0.4330 | 0.3520 | 1.0747 | 0.9907 | -0.9937 |
| heteroscedastic | 0.9114 | 0.4621 | 2.8709 | N/A | -0.3350 |
| high_frequency | 0.5906 | 0.2993 | 1.4138 | N/A | -9.2807 |
| intermittent | 0.8499 | 0.7857 | 2.0435 | N/A | -0.5180 |
| long_memory | 1.3952 | 1.4381 | 2.8009 | N/A | -3.4566 |
| low_count | 0.8881 | 0.5893 | 2.7074 | N/A | -1.2206 |
| multiplicative_seasonal | 0.3677 | 0.2328 | 1.2121 | 0.9998 | N/A |
| multiplicative_trend_seasonal | 1.2565 | 1.1684 | 2.4429 | 0.9992 | N/A |
| noisy_seasonal | 2.4415 | 1.4483 | 9.6944 | N/A | -3.1106 |
| outlier_series | 1.0114 | 0.6263 | 3.0362 | 0.9830 | -6.6973 |
| quarterly_seasonal | 0.3281 | 0.2703 | 0.7982 | 0.9989 | -1.4873 |
| random_walk | 0.3425 | 0.3655 | 0.7061 | N/A | -5.9335 |
| seasonal | 0.2700 | 0.2996 | 0.4953 | 0.9992 | -0.7424 |
| seasonal_negative | 0.1108 | 0.0968 | 0.2870 | 0.9999 | -0.4393 |
| seasonal_trend_break | 4.8309 | 4.4259 | 8.3092 | 0.9490 | -6.4478 |
| stationary | 1.1260 | 1.0898 | 3.8255 | N/A | -1.5979 |
| step_seasonal | 0.2191 | 0.2092 | 0.5494 | 0.9997 | -0.9098 |
| strong_seasonal | 0.1577 | 0.0872 | 0.4158 | 1.0000 | -0.6035 |
| structural_break | 1.4556 | 1.2160 | 4.9517 | N/A | -8.8664 |
| trend | 0.6833 | 0.5538 | 1.9174 | 0.9002 | -0.6806 |
| trend_seasonal | 0.1963 | 0.1731 | 0.6283 | 0.9995 | -1.0185 |


### AutoTBATS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.9124 | 0.6234 | 2.6962 | 0.5800 | -10.0727 |
| asymmetric_seasonal | 1.1914 | 0.7918 | 3.7858 | 0.9825 | -7.3594 |
| bimodal_seasonal | 1.0789 | 0.8970 | 2.2491 | 0.9848 | -8.9399 |
| damped_trend | 0.5172 | 0.4703 | 0.8907 | 0.6942 | -7.8610 |
| exponential_trend | 2.3134 | 2.1043 | 5.2712 | 0.9986 | -2.6131 |
| heteroscedastic | 0.7286 | 0.6742 | 1.4789 | 0.9249 | -14.1590 |
| high_frequency | 1.9153 | 1.5830 | 3.8819 | -0.3012 | -12.1289 |
| intermittent | 1.1030 | 1.1030 | 1.5466 | 0.9748 | 14.9742 |
| long_memory | 1.4635 | 1.5717 | 2.2934 | 0.9522 | -20.5932 |
| low_count | 0.8452 | 0.8452 | 1.3907 | 0.9315 | 14.2510 |
| multiplicative_seasonal | 5.4454 | 4.4640 | 14.6319 | 0.9924 | -12.4908 |
| multiplicative_trend_seasonal | 5.7196 | 5.1601 | 8.7126 | 0.9969 | -10.1265 |
| noisy_seasonal | 1.2288 | 1.2914 | 2.8369 | 0.7302 | -28.4005 |
| outlier_series | 1.7888 | 1.6635 | 3.5683 | 0.9786 | -20.7943 |
| quarterly_seasonal | 0.9267 | 0.4824 | 2.9569 | 0.9908 | -9.2134 |
| random_walk | 1.8840 | 1.8618 | 3.7602 | -0.7655 | -9.4099 |
| seasonal | 0.6524 | 0.5830 | 1.5689 | 0.9979 | -7.3228 |
| seasonal_negative | 1.3502 | 0.5154 | 5.1619 | 0.9851 | 14.1637 |
| seasonal_trend_break | 5.1750 | 5.9646 | 7.3428 | 0.9614 | -12.4959 |
| stationary | 1.3298 | 1.2986 | 2.4990 | 0.8876 | -14.2371 |
| step_seasonal | 0.6679 | 0.5843 | 1.3431 | 0.9979 | -6.1856 |
| strong_seasonal | 5.6908 | 3.1380 | 13.7188 | 0.9924 | -6.7104 |
| structural_break | 2.1956 | 1.7731 | 5.1737 | -0.0900 | -23.3337 |
| trend | 3.2063 | 3.7004 | 5.4938 | 0.9711 | -12.3612 |
| trend_seasonal | 1.5389 | 1.7116 | 2.3433 | 0.9909 | -8.8201 |


### AutoTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0133 | 0.0133 | 0.0144 | 0.0000 | -5.6799 |
| asymmetric_seasonal | 0.0679 | 0.0681 | 0.0801 | 1.0000 | 10.5293 |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 8.1899 |
| damped_trend | 0.7013 | 0.8720 | 1.3371 | 0.2200 | 5.9827 |
| exponential_trend | 0.7249 | 0.5810 | 2.2379 | 0.9423 | -3.1396 |
| heteroscedastic | 0.7692 | 0.7688 | 0.8065 | 0.9999 | 20.0699 |
| high_frequency | 0.0917 | 0.0850 | 0.2002 | 0.9959 | -7.5400 |
| intermittent | 0.0762 | 0.0762 | 0.0794 | 1.0000 | 13.4816 |
| long_memory | 0.3061 | 0.3061 | 0.4798 | -0.9676 | 12.3896 |
| low_count | 0.6998 | 0.6998 | 0.7335 | 0.0000 | 13.8986 |
| multiplicative_seasonal | 1.7972 | 1.2619 | 3.6903 | 0.9994 | 0.0430 |
| multiplicative_trend_seasonal | 2.0114 | 1.5257 | 3.8613 | 0.9988 | -0.8629 |
| noisy_seasonal | 2.6012 | 1.3688 | 9.2910 | -0.1550 | 20.1435 |
| outlier_series | 1.3792 | 1.2198 | 2.2650 | 0.9988 | 20.9280 |
| quarterly_seasonal | 0.8930 | 0.8886 | 1.0733 | 1.0000 | 14.0623 |
| random_walk | 0.1024 | 0.1024 | 0.1888 | 1.0000 | -5.4070 |
| seasonal | 0.2996 | 0.2976 | 0.3596 | 1.0000 | 9.4535 |
| seasonal_negative | 0.2227 | 0.2227 | 0.2287 | 1.0000 | -19.3229 |
| seasonal_trend_break | 0.0692 | 0.0560 | 0.1459 | 1.0000 | -3.8839 |
| stationary | 1.3777 | 1.3777 | 1.3860 | 1.0000 | 20.5388 |
| step_seasonal | 0.3072 | 0.3093 | 0.3743 | 1.0000 | 9.7192 |
| strong_seasonal | 0.5520 | 0.5570 | 0.7962 | 1.0000 | 9.1284 |
| structural_break | 0.0996 | 0.0925 | 0.2295 | 0.9983 | -3.3277 |
| trend | 1.6509 | 1.6178 | 3.4496 | 0.7914 | 9.6985 |
| trend_seasonal | 2.6215 | 2.2761 | 4.0574 | 0.9990 | 2.8279 |


### Croston

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### CrostonSBA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### DynamicOptimizedTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0071 | 0.0071 | 0.0071 | N/A | -5.8288 |
| asymmetric_seasonal | 0.1038 | 0.1072 | 0.1215 | 1.0000 | 1.8004 |
| bimodal_seasonal | 0.1593 | 0.1656 | 0.1802 | 1.0000 | -0.7752 |
| damped_trend | 0.6889 | 0.7920 | 1.2313 | 0.2212 | 4.6420 |
| exponential_trend | 3.6894 | 3.4622 | 6.6535 | 0.2834 | -3.3415 |
| heteroscedastic | 0.7689 | 0.7685 | 0.8065 | 0.9392 | 13.5876 |
| high_frequency | 0.1340 | 0.1336 | 0.1380 | 1.0000 | -7.9372 |
| intermittent | 0.1255 | 0.1255 | 0.1263 | N/A | 13.5597 |
| long_memory | 0.0993 | 0.0993 | 0.0993 | N/A | 10.5644 |
| low_count | 0.6048 | 0.6048 | 0.6048 | N/A | 13.9209 |
| multiplicative_seasonal | 4.2558 | 2.9885 | 8.7781 | 0.9963 | -7.4581 |
| multiplicative_trend_seasonal | 5.5527 | 4.2103 | 10.3903 | 0.9914 | -1.1064 |
| noisy_seasonal | 2.7087 | 1.5719 | 9.0879 | N/A | 19.4359 |
| outlier_series | 0.2249 | 0.1986 | 0.4559 | 0.9999 | 9.4949 |
| quarterly_seasonal | 0.8975 | 0.8917 | 1.0760 | 1.0000 | 5.4213 |
| random_walk | 0.5510 | 0.5518 | 1.0140 | -0.0000 | -5.3795 |
| seasonal | 0.3144 | 0.3119 | 0.3822 | 1.0000 | 0.7747 |
| seasonal_negative | 0.2395 | 0.2395 | 0.2396 | 1.0000 | -19.3489 |
| seasonal_trend_break | 0.0010 | 0.0010 | 0.0012 | 1.0000 | -4.0535 |
| stationary | 0.9213 | 0.9213 | 0.9213 | -0.0000 | 12.1595 |
| step_seasonal | 0.3727 | 0.3738 | 0.4580 | 1.0000 | -0.1575 |
| strong_seasonal | 0.5105 | 0.5174 | 0.7612 | 1.0000 | -19.1875 |
| structural_break | 0.0184 | 0.0185 | 0.0190 | 1.0000 | -3.2206 |
| trend | 4.4068 | 3.7750 | 8.3025 | 0.3268 | 5.5216 |
| trend_seasonal | 2.4406 | 1.8672 | 4.5143 | 0.9952 | -1.1996 |


### DynamicTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 1.1087 | 1.1061 | 1.1308 | -0.9346 | 25.9470 |
| asymmetric_seasonal | 0.0052 | 0.0058 | 0.0102 | 1.0000 | 32.3391 |
| bimodal_seasonal | 0.0015 | 0.0010 | 0.0038 | 1.0000 | 43.0275 |
| damped_trend | 0.6809 | 0.8605 | 1.2483 | 0.2207 | 12.5274 |
| exponential_trend | 8.5353 | 8.4999 | 10.3640 | 0.2831 | 23.0580 |
| heteroscedastic | 0.0153 | 0.0148 | 0.0316 | 0.9795 | 30.5068 |
| high_frequency | 2.9626 | 2.9571 | 3.0346 | 0.9994 | 26.4615 |
| intermittent | 0.0114 | 0.0115 | 0.0226 | -0.9886 | 10.1395 |
| long_memory | 0.1379 | 0.1341 | 0.2891 | -0.9901 | 25.0109 |
| low_count | 0.0354 | 0.0354 | 0.0707 | -0.9877 | 9.8555 |
| multiplicative_seasonal | 2.7416 | 1.9757 | 5.2192 | 0.9990 | 36.5447 |
| multiplicative_trend_seasonal | 2.7349 | 2.0771 | 5.1016 | 0.9977 | 22.2772 |
| noisy_seasonal | 2.2539 | 0.7441 | 10.0582 | -0.1506 | 42.7989 |
| outlier_series | 0.9433 | 0.8321 | 1.8999 | 0.9980 | 33.6519 |
| quarterly_seasonal | 0.0142 | 0.0147 | 0.0281 | 1.0000 | 40.5551 |
| random_walk | 1.4045 | 1.4053 | 1.8547 | 0.9878 | 34.3917 |
| seasonal | 0.0041 | 0.0030 | 0.0101 | 1.0000 | 36.8917 |
| seasonal_negative | 0.0003 | 0.0001 | 0.0010 | 1.0000 | -19.8700 |
| seasonal_trend_break | 3.7513 | 3.7235 | 4.8024 | 0.9999 | 29.7007 |
| stationary | 0.0763 | 0.0759 | 0.1533 | 0.9862 | 33.8249 |
| step_seasonal | 0.0234 | 0.0186 | 0.0551 | 1.0000 | 36.5909 |
| strong_seasonal | 0.0225 | 0.0134 | 0.0595 | 1.0000 | 79.2535 |
| structural_break | 1.2734 | 1.2637 | 2.0754 | 0.9668 | 32.2597 |
| trend | 2.7636 | 2.4658 | 5.2128 | 0.3269 | 13.5523 |
| trend_seasonal | 0.4998 | 0.3958 | 1.3141 | 0.9979 | 13.5581 |


### GARCH

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 3.0032 | 2.3382 | 6.6152 | 0.4505 | 5.9067 |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0001 | 1.0000 | -31.9337 |
| bimodal_seasonal | 0.5932 | 0.3373 | 1.9286 | 0.9991 | -143.1353 |
| damped_trend | 0.4756 | 0.3980 | 0.9594 | 1.0000 | -14.2256 |
| exponential_trend | 0.0001 | 0.0000 | 0.0001 | 1.0000 | -2029.1834 |
| heteroscedastic | 4.3971 | 3.5277 | 9.9890 | 0.4711 | 6.6411 |
| high_frequency | 0.0058 | 0.0046 | 0.0122 | 1.0000 | -49.2929 |
| intermittent | 0.1674 | 0.1419 | 0.3391 | 1.0000 | -19.9935 |
| long_memory | 0.1513 | 0.1236 | 0.3037 | 1.0000 | -93.8310 |
| low_count | 0.1223 | 0.1022 | 0.2466 | 1.0000 | -15.9589 |
| multiplicative_seasonal | 0.0001 | 0.0001 | 0.0003 | 1.0000 | -2122.3072 |
| multiplicative_trend_seasonal | 0.0006 | 0.0005 | 0.0012 | 1.0000 | -1518.7661 |
| noisy_seasonal | 0.0133 | 0.0107 | 0.0278 | 1.0000 | -253.2999 |
| outlier_series | 0.0001 | 0.0001 | 0.0004 | 1.0000 | -416.1053 |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -13.6908 |
| random_walk | 1.0140 | 0.8018 | 2.3313 | 0.9954 | -37.7159 |
| seasonal | 0.0748 | 0.0627 | 0.1510 | 1.0000 | -194.6602 |
| seasonal_negative | 0.3577 | 0.3000 | 0.7220 | 1.0000 | -174.6458 |
| seasonal_trend_break | 0.0008 | 0.0006 | 0.0022 | 1.0000 | -344.9443 |
| stationary | 0.0000 | 0.0000 | 0.0001 | 1.0000 | -40.6906 |
| step_seasonal | 0.0012 | 0.0006 | 0.0040 | 1.0000 | -266.4994 |
| strong_seasonal | 0.1126 | 0.0653 | 0.3652 | 1.0000 | -3151.5921 |
| structural_break | 0.0001 | 0.0000 | 0.0006 | 1.0000 | -337.0759 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -925.2734 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0001 | 1.0000 | -430.6339 |


### HistoricAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### Holt

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0128 | 0.0128 | 0.0232 | 1.0000 | 5.0746 |
| asymmetric_seasonal | 0.0256 | 0.0256 | 0.0473 | 1.0000 | 0.0042 |
| bimodal_seasonal | 0.0295 | 0.0295 | 0.0544 | 1.0000 | -1.7088 |
| damped_trend | 0.5529 | 0.5529 | 0.8265 | 1.0000 | 11.8997 |
| exponential_trend | 0.0828 | 0.0772 | 0.1840 | 1.0000 | 3.3895 |
| heteroscedastic | 0.0198 | 0.0198 | 0.0202 | 1.0000 | 20.9145 |
| high_frequency | 0.0112 | 0.0112 | 0.0176 | 1.0000 | 4.7931 |
| intermittent | 0.0055 | 0.0055 | 0.0068 | 1.0000 | 15.6151 |
| long_memory | 0.0047 | 0.0047 | 0.0070 | 1.0000 | 27.2467 |
| low_count | 0.0050 | 0.0050 | 0.0053 | 1.0000 | 14.1439 |
| multiplicative_seasonal | 0.0070 | 0.0070 | 0.0129 | 1.0000 | -2.6172 |
| multiplicative_trend_seasonal | 0.0143 | 0.0143 | 0.0264 | 1.0000 | -1.6403 |
| noisy_seasonal | 0.0451 | 0.0451 | 0.0462 | 1.0000 | 44.4353 |
| outlier_series | 0.0461 | 0.0461 | 0.0814 | 1.0000 | 13.0773 |
| quarterly_seasonal | 0.4527 | 0.4527 | 0.8343 | 1.0000 | 47.4105 |
| random_walk | 0.0063 | 0.0063 | 0.0116 | 1.0000 | -0.2973 |
| seasonal | 0.2030 | 0.2030 | 0.3755 | 1.0000 | -1.4031 |
| seasonal_negative | 0.0378 | 0.0378 | 0.0697 | 1.0000 | -1.1607 |
| seasonal_trend_break | 0.0363 | 0.0363 | 0.0671 | 1.0000 | -1.4112 |
| stationary | 1.3704 | 1.3704 | 1.8018 | 1.0000 | 21.0428 |
| step_seasonal | 0.0583 | 0.0583 | 0.1071 | 1.0000 | 1.7599 |
| strong_seasonal | 15.2025 | 17.2563 | 19.5764 | 0.9491 | -41.2760 |
| structural_break | 0.0152 | 0.0152 | 0.0256 | 1.0000 | 18.5182 |
| trend | 0.0033 | 0.0033 | 0.0034 | 1.0000 | 16.0695 |
| trend_seasonal | 0.0035 | 0.0035 | 0.0065 | 1.0000 | -1.1993 |


### HoltWinters

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 2.1847 | 2.1268 | 4.4044 | -0.0755 | -3.3385 |
| asymmetric_seasonal | 0.3889 | 0.3397 | 0.9240 | 0.9982 | 0.7911 |
| bimodal_seasonal | 0.3121 | 0.2810 | 0.7430 | 0.9992 | 1.0143 |
| damped_trend | 1.4480 | 1.4307 | 2.6507 | 0.5282 | 1.4301 |
| exponential_trend | 0.4717 | 0.4749 | 0.6608 | 0.9996 | -0.7292 |
| heteroscedastic | 0.8545 | 0.8819 | 1.5780 | 0.8670 | 1.0893 |
| high_frequency | 1.4446 | 0.5444 | 4.7717 | 0.1291 | -7.1087 |
| intermittent | 1.0762 | 1.2098 | 1.8219 | 0.8906 | 0.9580 |
| long_memory | 1.5722 | 1.4759 | 2.9862 | 0.7126 | 2.0876 |
| low_count | 0.6297 | 0.5631 | 1.1321 | 0.9486 | 1.2409 |
| multiplicative_seasonal | 4.3721 | 4.9084 | 7.5350 | 0.9966 | -3.6011 |
| multiplicative_trend_seasonal | 4.5459 | 4.2695 | 10.3430 | 0.9778 | -1.4409 |
| noisy_seasonal | 1.5284 | 1.1797 | 3.7337 | 0.8986 | 3.8954 |
| outlier_series | 1.4450 | 1.4824 | 2.6180 | 0.9662 | 3.7594 |
| quarterly_seasonal | 0.8657 | 0.7994 | 1.8641 | 0.9936 | 1.6132 |
| random_walk | 1.3773 | 1.4570 | 3.0681 | 0.8590 | -4.2143 |
| seasonal | 0.5694 | 0.5031 | 1.5096 | 0.9965 | 1.1633 |
| seasonal_negative | 0.5933 | 0.5395 | 1.2169 | 0.9986 | 0.4349 |
| seasonal_trend_break | 4.7035 | 4.8163 | 8.6073 | 0.9361 | -3.7703 |
| stationary | 1.7900 | 1.3539 | 4.0998 | 0.5531 | 1.7175 |
| step_seasonal | 0.5062 | 0.4904 | 1.1852 | 0.9986 | 1.2206 |
| strong_seasonal | 0.3815 | 0.3697 | 1.3628 | 0.9999 | 1.0389 |
| structural_break | 1.0310 | 0.9732 | 2.6900 | 0.9129 | -5.0560 |
| trend | 0.7958 | 0.5524 | 1.9553 | 0.9151 | 2.8235 |
| trend_seasonal | 0.5714 | 0.2921 | 1.6695 | 0.9922 | 1.7740 |


### IMAPA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 3.1949 | 3.1949 | 3.1949 | N/A | N/A |
| damped_trend | 0.0022 | 0.0022 | 0.0022 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0030 | 0.0030 | 0.0030 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0008 | 0.0008 | 0.0008 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 3.0990 | 3.0990 | 3.0990 | N/A | N/A |
| seasonal_negative | 2.6849 | 2.6849 | 2.6849 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 12.7968 | 12.7968 | 12.7968 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### MFLES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0001 | 1.0000 | N/A |
| asymmetric_seasonal | 0.0002 | 0.0001 | 0.0006 | 1.0000 | N/A |
| bimodal_seasonal | 0.0003 | 0.0003 | 0.0006 | 1.0000 | N/A |
| damped_trend | 0.0004 | 0.0004 | 0.0006 | 1.0000 | N/A |
| exponential_trend | 0.0133 | 0.0122 | 0.0261 | 1.0000 | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| high_frequency | 0.0002 | 0.0002 | 0.0004 | 1.0000 | N/A |
| intermittent | 0.0218 | 0.0327 | 0.0327 | 0.9999 | N/A |
| long_memory | 0.0005 | 0.0006 | 0.0010 | 1.0000 | N/A |
| low_count | 0.0179 | 0.0226 | 0.0291 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0205 | 0.0139 | 0.0418 | 1.0000 | N/A |
| multiplicative_trend_seasonal | 0.0231 | 0.0152 | 0.0461 | 1.0000 | N/A |
| noisy_seasonal | 0.0002 | 0.0002 | 0.0004 | 1.0000 | N/A |
| outlier_series | 0.0069 | 0.0048 | 0.0122 | 1.0000 | N/A |
| quarterly_seasonal | 0.0002 | 0.0002 | 0.0003 | 1.0000 | N/A |
| random_walk | 0.0024 | 0.0018 | 0.0036 | 1.0000 | N/A |
| seasonal | 0.0002 | 0.0002 | 0.0005 | 1.0000 | N/A |
| seasonal_negative | 0.0101 | 0.0089 | 0.0253 | 1.0000 | N/A |
| seasonal_trend_break | 0.0002 | 0.0002 | 0.0003 | 1.0000 | N/A |
| stationary | 0.0001 | 0.0001 | 0.0003 | 1.0000 | N/A |
| step_seasonal | 0.0007 | 0.0007 | 0.0012 | 1.0000 | N/A |
| strong_seasonal | 0.0007 | 0.0008 | 0.0018 | 1.0000 | N/A |
| structural_break | 0.0068 | 0.0052 | 0.0106 | 1.0000 | N/A |
| trend | 0.2536 | 0.2908 | 0.3484 | 0.9999 | N/A |
| trend_seasonal | 0.0111 | 0.0073 | 0.0219 | 1.0000 | N/A |


### MSTLForecaster

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.1840 | 0.1858 | 0.4119 | 0.9855 | -13.5555 |
| asymmetric_seasonal | 0.6106 | 0.7238 | 0.9734 | 0.9995 | -7.7832 |
| bimodal_seasonal | 0.2246 | 0.2151 | 0.5063 | 0.9997 | -8.8708 |
| damped_trend | 0.3130 | 0.2766 | 0.7198 | 0.9649 | -7.9843 |
| exponential_trend | 0.2668 | 0.2940 | 0.4401 | 0.9996 | -3.3262 |
| heteroscedastic | 0.3302 | 0.3899 | 0.5563 | 0.9893 | N/A |
| high_frequency | 2.2443 | 2.5045 | 3.5280 | 0.8470 | -20.1783 |
| intermittent | 0.2935 | 0.2612 | 0.9110 | 0.9878 | -10.2119 |
| long_memory | 0.3408 | 0.3398 | 0.6745 | 0.9900 | -20.6850 |
| low_count | 0.2114 | 0.1942 | 0.5713 | 0.9878 | -10.4153 |
| multiplicative_seasonal | 1.6031 | 1.7312 | 2.4174 | 0.9999 | -17.6582 |
| multiplicative_trend_seasonal | 0.7589 | 0.7023 | 1.1988 | 0.9997 | -25.7483 |
| noisy_seasonal | 1.7042 | 1.7804 | 2.5946 | 0.9909 | -28.3772 |
| outlier_series | 0.2377 | 0.1551 | 0.9483 | 0.9988 | -18.8110 |
| quarterly_seasonal | 0.3568 | 0.2542 | 0.7988 | 0.9996 | -9.4894 |
| random_walk | 0.3868 | 0.3699 | 0.6374 | 0.9835 | -8.9751 |
| seasonal | 0.1119 | 0.0908 | 0.3005 | 0.9998 | -7.3917 |
| seasonal_negative | 0.7645 | 0.7502 | 1.0136 | 0.9999 | -3.2723 |
| seasonal_trend_break | 0.1105 | 0.1121 | 0.2427 | 0.9998 | -12.3506 |
| stationary | 1.4137 | 1.3221 | 2.0655 | 0.9757 | -12.5530 |
| step_seasonal | 0.1719 | 0.1704 | 0.3701 | 0.9999 | -6.7935 |
| strong_seasonal | 1.5171 | 1.5155 | 2.3144 | 0.9999 | -6.5379 |
| structural_break | 0.1990 | 0.1653 | 0.4248 | 0.9949 | -23.0211 |
| trend | 0.1308 | 0.1311 | 0.3864 | 0.9977 | -11.1338 |
| trend_seasonal | 0.1842 | 0.1855 | 0.3828 | 0.9998 | -7.4345 |


### Naive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0047 |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0159 |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0120 |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0067 |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0024 |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0101 |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0054 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0088 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0143 |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0074 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0184 |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0116 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0249 |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0165 |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0256 |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0021 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0099 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0082 |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0099 |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0111 |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0181 |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0316 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0110 |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0082 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0085 |


### OptimizedTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.1883 | 0.1883 | 0.1894 | -0.0000 | -4.1882 |
| asymmetric_seasonal | 0.0671 | 0.0675 | 0.0791 | 1.0000 | 10.4918 |
| bimodal_seasonal | 0.1711 | 0.1777 | 0.1940 | 1.0000 | 9.4973 |
| damped_trend | 0.7062 | 0.7486 | 1.3391 | 0.3487 | 1.5472 |
| exponential_trend | 0.7249 | 0.5810 | 2.2379 | 0.9423 | -3.1396 |
| heteroscedastic | 0.6374 | 0.6374 | 0.6377 | -0.0000 | 17.2554 |
| high_frequency | 0.3499 | 0.3502 | 0.3793 | 0.9997 | -7.5835 |
| intermittent | 0.0762 | 0.0762 | 0.0794 | 1.0000 | 13.4816 |
| long_memory | 0.0403 | 0.0403 | 0.0444 | 1.0000 | 11.2093 |
| low_count | 0.7938 | 0.7938 | 0.8609 | -0.0000 | 13.7334 |
| multiplicative_seasonal | 0.6794 | 0.5364 | 1.1980 | 1.0000 | -9.5045 |
| multiplicative_trend_seasonal | 0.7976 | 0.6506 | 1.3242 | 0.9999 | -0.0535 |
| noisy_seasonal | 2.4915 | 1.1329 | 9.5149 | -0.1550 | 15.5357 |
| outlier_series | 0.6002 | 0.5328 | 0.8491 | 0.9999 | 16.1341 |
| quarterly_seasonal | 0.8920 | 0.8878 | 1.0724 | 1.0000 | 12.5654 |
| random_walk | 0.1024 | 0.1024 | 0.1888 | 1.0000 | -5.4070 |
| seasonal | 0.2989 | 0.2970 | 0.3586 | 1.0000 | 9.3013 |
| seasonal_negative | 0.2747 | 0.2747 | 0.2750 | 1.0000 | -19.3479 |
| seasonal_trend_break | 0.1316 | 0.1215 | 0.1894 | 1.0000 | -4.1382 |
| stationary | 1.4333 | 1.4333 | 1.4449 | 1.0000 | 19.4074 |
| step_seasonal | 0.2875 | 0.2868 | 0.3584 | 1.0000 | 8.4209 |
| strong_seasonal | 0.0394 | 0.0369 | 0.0711 | 1.0000 | 4.6849 |
| structural_break | 0.3110 | 0.3123 | 0.4066 | 0.9996 | -3.3544 |
| trend | 1.6509 | 1.6178 | 3.4496 | 0.7914 | 9.6985 |
| trend_seasonal | 0.5921 | 0.5287 | 0.8997 | 1.0000 | 4.3267 |


### RandomWalkWithDrift

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.7886 |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -2.6731 |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -2.0245 |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.1296 |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.3459 |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.6950 |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.9086 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.4802 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -2.4181 |
| low_count | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.2409 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -3.0998 |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.9428 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -4.1908 |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -2.7714 |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -4.3097 |
| random_walk | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.3534 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.6636 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3743 |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.6712 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.8642 |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -3.0520 |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -5.3314 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.8602 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3644 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.4205 |


### SARIMA_1_1_1_1_1_1_12

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.9620 | 0.8021 | 1.8865 | 0.9881 | -0.8961 |
| asymmetric_seasonal | 1.0001 | 1.0772 | 1.2011 | 0.9998 | 3.3633 |
| bimodal_seasonal | 0.2484 | 0.2451 | 0.3663 | 0.9999 | 3.0580 |
| damped_trend | 0.5634 | 0.5654 | 0.8783 | 0.9866 | 2.1665 |
| exponential_trend | 1.8857 | 1.7147 | 3.0765 | 0.9987 | -2.0689 |
| heteroscedastic | 1.7043 | 1.7251 | 3.1183 | 0.9695 | 2.6330 |
| high_frequency | 3.2418 | 3.5341 | 6.8253 | 0.4191 | -8.6031 |
| intermittent | 1.1715 | 1.1583 | 2.0800 | 0.9504 | 3.3295 |
| long_memory | 1.9183 | 1.7317 | 3.0078 | 0.9744 | 3.5963 |
| low_count | 0.3074 | 0.2231 | 0.6453 | 0.9592 | 2.9162 |
| multiplicative_seasonal | 0.1719 | 0.1766 | 0.2526 | 1.0000 | -5.7517 |
| multiplicative_trend_seasonal | 1.0317 | 0.9981 | 1.5324 | 0.9998 | 1.0582 |
| noisy_seasonal | 1.9814 | 2.1161 | 2.9675 | 0.9935 | 12.0239 |
| outlier_series | 2.2067 | 2.2758 | 3.6251 | 0.9895 | -5.1983 |
| quarterly_seasonal | 0.4265 | 0.4210 | 0.6648 | 0.9998 | 2.3819 |
| random_walk | 0.2488 | 0.2602 | 0.4276 | 0.9957 | -4.0316 |
| seasonal | 0.0969 | 0.0423 | 0.3357 | 0.9998 | 0.9938 |
| seasonal_negative | 0.4088 | 0.3867 | 0.6212 | 0.9998 | 0.4476 |
| seasonal_trend_break | 2.3510 | 2.6429 | 4.3393 | 0.9857 | -3.6055 |
| stationary | 1.0626 | 1.0411 | 2.2309 | 0.9659 | 2.1218 |
| step_seasonal | 0.8126 | 0.8223 | 1.3594 | 0.9997 | 1.6958 |
| strong_seasonal | 0.1977 | 0.1832 | 0.4310 | 1.0000 | 1.3891 |
| structural_break | 0.5745 | 0.6072 | 0.9764 | 0.9822 | -2.6793 |
| trend | 0.7166 | 0.6529 | 1.5362 | 0.9249 | 2.8724 |
| trend_seasonal | 0.5946 | 0.5161 | 0.9668 | 0.9993 | 2.8257 |


### SES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### SeasonalES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### SeasonalNaive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0032 |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0027 |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0029 |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0024 |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0071 |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0046 |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0061 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0035 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0064 |
| low_count | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0030 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0057 |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0059 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0089 |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0066 |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0032 |
| random_walk | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0039 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0026 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0011 |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0054 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0043 |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0023 |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0022 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0068 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0065 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0039 |


### SeasonalWindowAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### TBATS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.9854 | 1.0389 | 2.0436 | 0.9557 | -13.7195 |
| asymmetric_seasonal | 1.0928 | 1.0035 | 2.3856 | 0.9893 | -8.9305 |
| bimodal_seasonal | 1.6839 | 1.4218 | 3.8236 | 0.9836 | -11.8864 |
| damped_trend | 0.3994 | 0.4290 | 0.6876 | 0.3360 | -7.7938 |
| exponential_trend | 2.1179 | 1.8848 | 4.9506 | 0.9988 | -3.7222 |
| heteroscedastic | 3.3804 | 2.9987 | 5.1229 | -0.5110 | -18.5920 |
| high_frequency | 3.1522 | 3.8259 | 4.1908 | -0.6716 | -21.8530 |
| intermittent | 1.0864 | 1.0899 | 1.5538 | 0.9641 | 14.5175 |
| long_memory | 2.3745 | 2.3226 | 3.7556 | 0.8769 | -27.1253 |
| low_count | 0.9536 | 0.9529 | 1.4903 | 0.9256 | 13.7489 |
| multiplicative_seasonal | 1.9061 | 1.5599 | 5.4301 | 0.9948 | -13.3848 |
| multiplicative_trend_seasonal | 5.7196 | 5.1601 | 8.7126 | 0.9969 | -10.1265 |
| noisy_seasonal | 1.5071 | 1.8191 | 2.4733 | 0.8449 | -33.8607 |
| outlier_series | 1.0654 | 0.9223 | 2.3380 | 0.9886 | -22.5574 |
| quarterly_seasonal | 1.7614 | 1.1940 | 3.7392 | 0.9857 | -12.0560 |
| random_walk | 1.4522 | 1.8277 | 2.0045 | 0.8947 | -9.4656 |
| seasonal | 0.9908 | 0.9949 | 1.5273 | 0.9995 | -9.2698 |
| seasonal_negative | 2.0774 | 1.4332 | 6.2652 | 0.9649 | 13.1795 |
| seasonal_trend_break | 7.2456 | 8.6201 | 11.2666 | 0.9025 | -12.5001 |
| stationary | 1.8272 | 2.0886 | 3.7162 | 0.8870 | -14.6860 |
| step_seasonal | 1.6202 | 1.2637 | 3.8397 | 0.9865 | -7.9490 |
| strong_seasonal | 5.1848 | 3.7080 | 17.0335 | 0.9859 | -15.4290 |
| structural_break | 2.0947 | 1.7445 | 4.8742 | 0.0760 | -26.2932 |
| trend | 1.7856 | 1.8263 | 2.7896 | 0.9881 | -12.2741 |
| trend_seasonal | 1.5389 | 1.7116 | 2.3433 | 0.9909 | -8.8201 |


### TSB

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### Theta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 1.1294 | 1.1294 | 1.1294 | 1.0000 | 3.7612 |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 7.0355 |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 8.1899 |
| damped_trend | 0.7018 | 0.8003 | 1.2834 | 0.2943 | 7.4269 |
| exponential_trend | 6.8185 | 6.6260 | 8.2378 | 0.8814 | 35.9406 |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 13.4155 |
| high_frequency | 2.9398 | 2.9512 | 2.9909 | 1.0000 | 10.1739 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10.0557 |
| long_memory | 0.2464 | 0.2464 | 0.2464 | 1.0000 | 18.0775 |
| low_count | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 9.0797 |
| multiplicative_seasonal | 1.3547 | 1.3618 | 1.7455 | 1.0000 | 26.7611 |
| multiplicative_trend_seasonal | 1.5709 | 1.5672 | 1.9670 | 1.0000 | 25.9260 |
| noisy_seasonal | 2.2540 | 0.8042 | 9.9606 | -0.1416 | 25.7384 |
| outlier_series | 0.1572 | 0.1573 | 0.1799 | 1.0000 | 22.0629 |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 9.3201 |
| random_walk | 0.9514 | 0.9514 | 0.9514 | 1.0000 | 11.8372 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 6.9671 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -21.0416 |
| seasonal_trend_break | 3.6007 | 3.5951 | 4.4210 | 1.0000 | 17.2309 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 13.6698 |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 6.4874 |
| strong_seasonal | 0.0000 | 0.0000 | 0.0001 | 1.0000 | 7.2480 |
| structural_break | 1.9598 | 1.9695 | 2.0243 | 1.0000 | 16.7732 |
| trend | 1.8274 | 2.0826 | 3.0086 | 0.6713 | 26.0710 |
| trend_seasonal | 1.0022 | 1.0117 | 1.2521 | 1.0000 | 11.9908 |


### WindowAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| ar1 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| asymmetric_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| bimodal_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| damped_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| exponential_trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| heteroscedastic | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| low_count | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| outlier_series | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| quarterly_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| random_walk | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_trend_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| step_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| strong_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


---

## Confidence Interval Comparison

Mean CI width differences (Rust - statsforecast) by level:

| Model | Series | 80% | 90% | 95% |
|-------|--------|-----|-----|-----|
| ADIDA | ar1 | N/A | N/A | N/A |
| ARIMA_1_1_1 | ar1 | 3.7380 | 4.7991 | 5.7185 |
| AutoARIMA | ar1 | -2.1361 | -2.7410 | -3.2661 |
| AutoETS | ar1 | -4.8459 | -6.2191 | -7.4105 |
| AutoTBATS | ar1 | -6.5862 | -8.4533 | -10.0727 |
| AutoTheta | ar1 | -4.4347 | -5.1429 | -5.6799 |
| Croston | ar1 | N/A | N/A | N/A |
| CrostonSBA | ar1 | N/A | N/A | N/A |
| DynamicOptimizedTheta | ar1 | -4.5268 | -5.2705 | -5.8288 |
| DynamicTheta | ar1 | 16.2118 | 21.3686 | 25.9470 |
| GARCH | ar1 | 3.8615 | 4.9570 | 5.9067 |
| HistoricAverage | ar1 | N/A | N/A | N/A |
| Holt | ar1 | 3.3170 | 4.2587 | 5.0746 |
| HoltWinters | ar1 | -2.1835 | -2.8018 | -3.3385 |
| IMAPA | ar1 | N/A | N/A | N/A |
| MFLES | ar1 | N/A | N/A | N/A |
| MSTLForecaster | ar1 | -8.8635 | -11.3761 | -13.5555 |
| Naive | ar1 | 0.0019 | 0.0039 | 0.0047 |
| OptimizedTheta | ar1 | -3.4594 | -3.8910 | -4.1882 |
| RandomWalkWithDrift | ar1 | -0.5168 | -0.6619 | -0.7886 |
| SARIMA_1_1_1_1_1_1_12 | ar1 | -0.5866 | -0.7521 | -0.8961 |
| SES | ar1 | N/A | N/A | N/A |
| SeasonalES | ar1 | N/A | N/A | N/A |
| SeasonalNaive | ar1 | 0.0013 | 0.0027 | 0.0032 |
| SeasonalWindowAverage | ar1 | N/A | N/A | N/A |
| TBATS | ar1 | -8.9709 | -11.5138 | -13.7195 |
| TSB | ar1 | N/A | N/A | N/A |
| Theta | ar1 | 1.7401 | 2.7820 | 3.7612 |
| WindowAverage | ar1 | N/A | N/A | N/A |
| ADIDA | asymmetric_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | asymmetric_seasonal | 12.5738 | 16.1427 | 19.2354 |
| AutoARIMA | asymmetric_seasonal | 1.4621 | 1.8773 | 2.2370 |
| AutoETS | asymmetric_seasonal | -0.2730 | -0.3498 | -0.4168 |
| AutoTBATS | asymmetric_seasonal | -4.8121 | -6.1762 | -7.3594 |
| AutoTheta | asymmetric_seasonal | 6.7537 | 8.5709 | 10.5293 |
| Croston | asymmetric_seasonal | N/A | N/A | N/A |
| CrostonSBA | asymmetric_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | asymmetric_seasonal | 0.9089 | 0.9700 | 1.8004 |
| DynamicTheta | asymmetric_seasonal | 21.0113 | 26.8702 | 32.3391 |
| GARCH | asymmetric_seasonal | -20.8808 | -26.7996 | -31.9337 |
| HistoricAverage | asymmetric_seasonal | N/A | N/A | N/A |
| Holt | asymmetric_seasonal | -0.0011 | 0.0034 | 0.0042 |
| HoltWinters | asymmetric_seasonal | 0.5168 | 0.6639 | 0.7911 |
| IMAPA | asymmetric_seasonal | N/A | N/A | N/A |
| MFLES | asymmetric_seasonal | N/A | N/A | N/A |
| MSTLForecaster | asymmetric_seasonal | -5.0891 | -6.5318 | -7.7832 |
| Naive | asymmetric_seasonal | 0.0065 | 0.0132 | 0.0159 |
| OptimizedTheta | asymmetric_seasonal | 6.7286 | 8.5382 | 10.4918 |
| RandomWalkWithDrift | asymmetric_seasonal | -1.7517 | -2.2434 | -2.6731 |
| SARIMA_1_1_1_1_1_1_12 | asymmetric_seasonal | 2.1984 | 2.8225 | 3.3633 |
| SES | asymmetric_seasonal | N/A | N/A | N/A |
| SeasonalES | asymmetric_seasonal | N/A | N/A | N/A |
| SeasonalNaive | asymmetric_seasonal | 0.0011 | 0.0022 | 0.0027 |
| SeasonalWindowAverage | asymmetric_seasonal | N/A | N/A | N/A |
| TBATS | asymmetric_seasonal | -5.8394 | -7.4947 | -8.9305 |
| TSB | asymmetric_seasonal | N/A | N/A | N/A |
| Theta | asymmetric_seasonal | 4.4695 | 5.6388 | 7.0355 |
| WindowAverage | asymmetric_seasonal | N/A | N/A | N/A |
| ADIDA | bimodal_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | bimodal_seasonal | -5.4126 | -6.9434 | -8.2734 |
| AutoARIMA | bimodal_seasonal | 2.4765 | 3.1794 | 3.7886 |
| AutoETS | bimodal_seasonal | N/A | N/A | N/A |
| AutoTBATS | bimodal_seasonal | -5.8455 | -7.5026 | -8.9399 |
| AutoTheta | bimodal_seasonal | 5.1921 | 6.5824 | 8.1899 |
| Croston | bimodal_seasonal | N/A | N/A | N/A |
| CrostonSBA | bimodal_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | bimodal_seasonal | -0.8613 | -1.2996 | -0.7752 |
| DynamicTheta | bimodal_seasonal | 27.9660 | 35.8130 | 43.0275 |
| GARCH | bimodal_seasonal | -93.5927 | -120.1230 | -143.1353 |
| HistoricAverage | bimodal_seasonal | N/A | N/A | N/A |
| Holt | bimodal_seasonal | -1.1202 | -1.4342 | -1.7088 |
| HoltWinters | bimodal_seasonal | 0.6626 | 0.8512 | 1.0143 |
| IMAPA | bimodal_seasonal | N/A | N/A | N/A |
| MFLES | bimodal_seasonal | N/A | N/A | N/A |
| MSTLForecaster | bimodal_seasonal | -5.8003 | -7.4446 | -8.8708 |
| Naive | bimodal_seasonal | 0.0049 | 0.0100 | 0.0120 |
| OptimizedTheta | bimodal_seasonal | 6.0344 | 7.6558 | 9.4973 |
| RandomWalkWithDrift | bimodal_seasonal | -1.3267 | -1.6991 | -2.0245 |
| SARIMA_1_1_1_1_1_1_12 | bimodal_seasonal | 1.9987 | 2.5663 | 3.0580 |
| SES | bimodal_seasonal | N/A | N/A | N/A |
| SeasonalES | bimodal_seasonal | N/A | N/A | N/A |
| SeasonalNaive | bimodal_seasonal | 0.0012 | 0.0024 | 0.0029 |
| SeasonalWindowAverage | bimodal_seasonal | N/A | N/A | N/A |
| TBATS | bimodal_seasonal | -7.7640 | -9.9700 | -11.8864 |
| TSB | bimodal_seasonal | N/A | N/A | N/A |
| Theta | bimodal_seasonal | 5.1921 | 6.5824 | 8.1899 |
| WindowAverage | bimodal_seasonal | N/A | N/A | N/A |
| ADIDA | damped_trend | N/A | N/A | N/A |
| ARIMA_1_1_1 | damped_trend | 7.1767 | 9.2126 | 10.9775 |
| AutoARIMA | damped_trend | 0.1870 | 0.2407 | 0.2868 |
| AutoETS | damped_trend | -0.6081 | -0.7800 | -0.9294 |
| AutoTBATS | damped_trend | -5.1401 | -6.5971 | -7.8610 |
| AutoTheta | damped_trend | 3.7675 | 4.7552 | 5.9827 |
| Croston | damped_trend | N/A | N/A | N/A |
| CrostonSBA | damped_trend | N/A | N/A | N/A |
| DynamicOptimizedTheta | damped_trend | 2.8678 | 3.5814 | 4.6420 |
| DynamicTheta | damped_trend | 8.0425 | 10.2402 | 12.5274 |
| GARCH | damped_trend | -9.3020 | -11.9385 | -14.2256 |
| HistoricAverage | damped_trend | N/A | N/A | N/A |
| Holt | damped_trend | 7.7797 | 9.9865 | 11.8997 |
| HoltWinters | damped_trend | 0.9346 | 1.2001 | 1.4301 |
| IMAPA | damped_trend | N/A | N/A | N/A |
| MFLES | damped_trend | N/A | N/A | N/A |
| MSTLForecaster | damped_trend | -5.2207 | -6.7007 | -7.9843 |
| Naive | damped_trend | 0.0028 | 0.0056 | 0.0067 |
| OptimizedTheta | damped_trend | 0.8269 | 0.9429 | 1.5472 |
| RandomWalkWithDrift | damped_trend | -0.7402 | -0.9481 | -1.1296 |
| SARIMA_1_1_1_1_1_1_12 | damped_trend | 1.4160 | 1.8182 | 2.1665 |
| SES | damped_trend | N/A | N/A | N/A |
| SeasonalES | damped_trend | N/A | N/A | N/A |
| SeasonalNaive | damped_trend | 0.0010 | 0.0020 | 0.0024 |
| SeasonalWindowAverage | damped_trend | N/A | N/A | N/A |
| TBATS | damped_trend | -5.0967 | -6.5411 | -7.7938 |
| TSB | damped_trend | N/A | N/A | N/A |
| Theta | damped_trend | 4.7118 | 5.9672 | 7.4269 |
| WindowAverage | damped_trend | N/A | N/A | N/A |
| ADIDA | exponential_trend | N/A | N/A | N/A |
| ARIMA_1_1_1 | exponential_trend | -4.7846 | -6.1404 | -7.3167 |
| AutoARIMA | exponential_trend | -0.2453 | -0.3147 | -0.3749 |
| AutoETS | exponential_trend | -0.6499 | -0.8339 | -0.9937 |
| AutoTBATS | exponential_trend | -1.7087 | -2.1930 | -2.6131 |
| AutoTheta | exponential_trend | -2.4135 | -2.8476 | -3.1396 |
| Croston | exponential_trend | N/A | N/A | N/A |
| CrostonSBA | exponential_trend | N/A | N/A | N/A |
| DynamicOptimizedTheta | exponential_trend | -2.5367 | -2.9973 | -3.3415 |
| DynamicTheta | exponential_trend | 14.6937 | 19.1630 | 23.0580 |
| GARCH | exponential_trend | -1326.8165 | -1702.9445 | -2029.1834 |
| HistoricAverage | exponential_trend | N/A | N/A | N/A |
| Holt | exponential_trend | 2.2159 | 2.8446 | 3.3895 |
| HoltWinters | exponential_trend | -0.4770 | -0.6120 | -0.7292 |
| IMAPA | exponential_trend | N/A | N/A | N/A |
| MFLES | exponential_trend | N/A | N/A | N/A |
| MSTLForecaster | exponential_trend | -2.1749 | -2.7914 | -3.3262 |
| Naive | exponential_trend | 0.0010 | 0.0020 | 0.0024 |
| OptimizedTheta | exponential_trend | -2.4135 | -2.8476 | -3.1396 |
| RandomWalkWithDrift | exponential_trend | -0.2267 | -0.2903 | -0.3459 |
| SARIMA_1_1_1_1_1_1_12 | exponential_trend | -1.3530 | -1.7363 | -2.0689 |
| SES | exponential_trend | N/A | N/A | N/A |
| SeasonalES | exponential_trend | N/A | N/A | N/A |
| SeasonalNaive | exponential_trend | 0.0029 | 0.0059 | 0.0071 |
| SeasonalWindowAverage | exponential_trend | N/A | N/A | N/A |
| TBATS | exponential_trend | -2.4339 | -3.1238 | -3.7222 |
| TSB | exponential_trend | N/A | N/A | N/A |
| Theta | exponential_trend | 23.1154 | 29.9726 | 35.9406 |
| WindowAverage | exponential_trend | N/A | N/A | N/A |
| ADIDA | heteroscedastic | N/A | N/A | N/A |
| ARIMA_1_1_1 | heteroscedastic | 14.0252 | 18.0036 | 21.4527 |
| AutoARIMA | heteroscedastic | -34.2667 | -43.9797 | -52.4051 |
| AutoETS | heteroscedastic | -0.2199 | -0.2812 | -0.3350 |
| AutoTBATS | heteroscedastic | -9.2581 | -11.8826 | -14.1590 |
| AutoTheta | heteroscedastic | 12.8551 | 16.3510 | 20.0699 |
| Croston | heteroscedastic | N/A | N/A | N/A |
| CrostonSBA | heteroscedastic | N/A | N/A | N/A |
| DynamicOptimizedTheta | heteroscedastic | 8.5032 | 10.6808 | 13.5876 |
| DynamicTheta | heteroscedastic | 19.6768 | 25.1059 | 30.5068 |
| GARCH | heteroscedastic | 4.3415 | 5.5734 | 6.6411 |
| HistoricAverage | heteroscedastic | N/A | N/A | N/A |
| Holt | heteroscedastic | 13.6733 | 17.5519 | 20.9145 |
| HoltWinters | heteroscedastic | 0.7114 | 0.9142 | 1.0893 |
| IMAPA | heteroscedastic | N/A | N/A | N/A |
| MFLES | heteroscedastic | N/A | N/A | N/A |
| MSTLForecaster | heteroscedastic | N/A | N/A | N/A |
| Naive | heteroscedastic | 0.0041 | 0.0083 | 0.0101 |
| OptimizedTheta | heteroscedastic | 11.0007 | 13.9578 | 17.2554 |
| RandomWalkWithDrift | heteroscedastic | -1.1108 | -1.4226 | -1.6950 |
| SARIMA_1_1_1_1_1_1_12 | heteroscedastic | 1.7204 | 2.2097 | 2.6330 |
| SES | heteroscedastic | N/A | N/A | N/A |
| SeasonalES | heteroscedastic | N/A | N/A | N/A |
| SeasonalNaive | heteroscedastic | 0.0019 | 0.0038 | 0.0046 |
| SeasonalWindowAverage | heteroscedastic | N/A | N/A | N/A |
| TBATS | heteroscedastic | -12.1567 | -15.6029 | -18.5920 |
| TSB | heteroscedastic | N/A | N/A | N/A |
| Theta | heteroscedastic | 8.5044 | 10.7665 | 13.4155 |
| WindowAverage | heteroscedastic | N/A | N/A | N/A |
| ADIDA | high_frequency | N/A | N/A | N/A |
| ARIMA_1_1_1 | high_frequency | 3.1769 | 4.0790 | 4.8605 |
| AutoARIMA | high_frequency | -3.7479 | -4.8096 | -5.7310 |
| AutoETS | high_frequency | -6.0688 | -7.7886 | -9.2807 |
| AutoTBATS | high_frequency | -7.9307 | -10.1789 | -12.1289 |
| AutoTheta | high_frequency | -5.8138 | -6.8495 | -7.5400 |
| Croston | high_frequency | N/A | N/A | N/A |
| CrostonSBA | high_frequency | N/A | N/A | N/A |
| DynamicOptimizedTheta | high_frequency | -6.0720 | -7.1887 | -7.9372 |
| DynamicTheta | high_frequency | 16.4028 | 21.6741 | 26.4615 |
| GARCH | high_frequency | -32.2317 | -41.3679 | -49.2929 |
| HistoricAverage | high_frequency | N/A | N/A | N/A |
| Holt | high_frequency | 3.1328 | 4.0225 | 4.7931 |
| HoltWinters | high_frequency | -4.6487 | -5.9658 | -7.1087 |
| IMAPA | high_frequency | N/A | N/A | N/A |
| MFLES | high_frequency | N/A | N/A | N/A |
| MSTLForecaster | high_frequency | -13.1939 | -16.9342 | -20.1783 |
| Naive | high_frequency | 0.0022 | 0.0045 | 0.0054 |
| OptimizedTheta | high_frequency | -5.8423 | -6.8860 | -7.5835 |
| RandomWalkWithDrift | high_frequency | -0.5954 | -0.7626 | -0.9086 |
| SARIMA_1_1_1_1_1_1_12 | high_frequency | -5.6260 | -7.2200 | -8.6031 |
| SES | high_frequency | N/A | N/A | N/A |
| SeasonalES | high_frequency | N/A | N/A | N/A |
| SeasonalNaive | high_frequency | 0.0025 | 0.0051 | 0.0061 |
| SeasonalWindowAverage | high_frequency | N/A | N/A | N/A |
| TBATS | high_frequency | -14.2890 | -18.3396 | -21.8530 |
| TSB | high_frequency | N/A | N/A | N/A |
| Theta | high_frequency | 5.7726 | 8.0164 | 10.1739 |
| WindowAverage | high_frequency | N/A | N/A | N/A |
| ADIDA | intermittent | N/A | N/A | N/A |
| ARIMA_1_1_1 | intermittent | 10.9521 | 14.0588 | 16.7521 |
| AutoARIMA | intermittent | 0.4940 | 0.6349 | 0.7565 |
| AutoETS | intermittent | -0.3393 | -0.4348 | -0.5180 |
| AutoTBATS | intermittent | 9.7897 | 12.5667 | 14.9742 |
| AutoTheta | intermittent | 8.6182 | 10.9449 | 13.4816 |
| Croston | intermittent | N/A | N/A | N/A |
| CrostonSBA | intermittent | N/A | N/A | N/A |
| DynamicOptimizedTheta | intermittent | 8.6657 | 11.0004 | 13.5597 |
| DynamicTheta | intermittent | 6.4271 | 8.1363 | 10.1395 |
| GARCH | intermittent | -13.0737 | -16.7791 | -19.9935 |
| HistoricAverage | intermittent | N/A | N/A | N/A |
| Holt | intermittent | 10.2087 | 13.1046 | 15.6151 |
| HoltWinters | intermittent | 0.6257 | 0.8040 | 0.9580 |
| IMAPA | intermittent | N/A | N/A | N/A |
| MFLES | intermittent | N/A | N/A | N/A |
| MSTLForecaster | intermittent | -6.6772 | -8.5701 | -10.2119 |
| Naive | intermittent | 0.0036 | 0.0073 | 0.0088 |
| OptimizedTheta | intermittent | 8.6182 | 10.9449 | 13.4816 |
| RandomWalkWithDrift | intermittent | -0.9700 | -1.2423 | -1.4802 |
| SARIMA_1_1_1_1_1_1_12 | intermittent | 2.1761 | 2.7941 | 3.3295 |
| SES | intermittent | N/A | N/A | N/A |
| SeasonalES | intermittent | N/A | N/A | N/A |
| SeasonalNaive | intermittent | 0.0014 | 0.0029 | 0.0035 |
| SeasonalWindowAverage | intermittent | N/A | N/A | N/A |
| TBATS | intermittent | 9.4911 | 12.1834 | 14.5175 |
| TSB | intermittent | N/A | N/A | N/A |
| Theta | intermittent | 6.3751 | 8.0710 | 10.0557 |
| WindowAverage | intermittent | N/A | N/A | N/A |
| ADIDA | long_memory | N/A | N/A | N/A |
| ARIMA_1_1_1 | long_memory | 18.0166 | 23.1274 | 27.5581 |
| AutoARIMA | long_memory | -0.5569 | -0.7133 | -0.8499 |
| AutoETS | long_memory | -2.2611 | -2.9009 | -3.4566 |
| AutoTBATS | long_memory | -13.4652 | -17.2824 | -20.5932 |
| AutoTheta | long_memory | 7.6409 | 9.9771 | 12.3896 |
| Croston | long_memory | N/A | N/A | N/A |
| CrostonSBA | long_memory | N/A | N/A | N/A |
| DynamicOptimizedTheta | long_memory | 6.4400 | 8.4493 | 10.5644 |
| DynamicTheta | long_memory | 15.9020 | 20.5892 | 25.0109 |
| GARCH | long_memory | -61.3539 | -78.7455 | -93.8310 |
| HistoricAverage | long_memory | N/A | N/A | N/A |
| Holt | long_memory | 17.8130 | 22.8660 | 27.2467 |
| HoltWinters | long_memory | 1.3637 | 1.7519 | 2.0876 |
| IMAPA | long_memory | N/A | N/A | N/A |
| MFLES | long_memory | N/A | N/A | N/A |
| MSTLForecaster | long_memory | -13.5252 | -17.3594 | -20.6850 |
| Naive | long_memory | 0.0059 | 0.0119 | 0.0143 |
| OptimizedTheta | long_memory | 6.8297 | 8.9253 | 11.2093 |
| RandomWalkWithDrift | long_memory | -1.5846 | -2.0295 | -2.4181 |
| SARIMA_1_1_1_1_1_1_12 | long_memory | 2.3499 | 3.0181 | 3.5963 |
| SES | long_memory | N/A | N/A | N/A |
| SeasonalES | long_memory | N/A | N/A | N/A |
| SeasonalNaive | long_memory | 0.0026 | 0.0053 | 0.0064 |
| SeasonalWindowAverage | long_memory | N/A | N/A | N/A |
| TBATS | long_memory | -17.7363 | -22.7643 | -27.1253 |
| TSB | long_memory | N/A | N/A | N/A |
| Theta | long_memory | 11.3597 | 14.7505 | 18.0775 |
| WindowAverage | long_memory | N/A | N/A | N/A |
| ADIDA | low_count | N/A | N/A | N/A |
| ARIMA_1_1_1 | low_count | 10.0446 | 12.8938 | 15.3640 |
| AutoARIMA | low_count | -1.7297 | -2.2193 | -2.6445 |
| AutoETS | low_count | -0.7986 | -1.0244 | -1.2206 |
| AutoTBATS | low_count | 9.3169 | 11.9598 | 14.2510 |
| AutoTheta | low_count | 8.9086 | 11.3348 | 13.8986 |
| Croston | low_count | N/A | N/A | N/A |
| CrostonSBA | low_count | N/A | N/A | N/A |
| DynamicOptimizedTheta | low_count | 8.9230 | 11.3436 | 13.9209 |
| DynamicTheta | low_count | 6.2630 | 7.9377 | 9.8555 |
| GARCH | low_count | -10.4355 | -13.3931 | -15.9589 |
| HistoricAverage | low_count | N/A | N/A | N/A |
| Holt | low_count | 9.2469 | 11.8699 | 14.1439 |
| HoltWinters | low_count | 0.8108 | 1.0414 | 1.2409 |
| IMAPA | low_count | N/A | N/A | N/A |
| MFLES | low_count | N/A | N/A | N/A |
| MSTLForecaster | low_count | -6.8102 | -8.7408 | -10.4153 |
| Naive | low_count | 0.0030 | 0.0061 | 0.0074 |
| OptimizedTheta | low_count | 8.8029 | 11.1862 | 13.7334 |
| RandomWalkWithDrift | low_count | -0.8132 | -1.0415 | -1.2409 |
| SARIMA_1_1_1_1_1_1_12 | low_count | 1.9060 | 2.4473 | 2.9162 |
| SES | low_count | N/A | N/A | N/A |
| SeasonalES | low_count | N/A | N/A | N/A |
| SeasonalNaive | low_count | 0.0012 | 0.0025 | 0.0030 |
| SeasonalWindowAverage | low_count | N/A | N/A | N/A |
| TBATS | low_count | 8.9886 | 11.5384 | 13.7489 |
| TSB | low_count | N/A | N/A | N/A |
| Theta | low_count | 5.7579 | 7.2907 | 9.0797 |
| WindowAverage | low_count | N/A | N/A | N/A |
| ADIDA | multiplicative_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | multiplicative_seasonal | -47.6739 | -61.1854 | -72.9068 |
| AutoARIMA | multiplicative_seasonal | -4.6355 | -5.9489 | -7.0886 |
| AutoETS | multiplicative_seasonal | N/A | N/A | N/A |
| AutoTBATS | multiplicative_seasonal | -8.1640 | -10.4804 | -12.4908 |
| AutoTheta | multiplicative_seasonal | -0.2788 | -0.1641 | 0.0430 |
| Croston | multiplicative_seasonal | N/A | N/A | N/A |
| CrostonSBA | multiplicative_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | multiplicative_seasonal | -5.1204 | -6.7720 | -7.4581 |
| DynamicTheta | multiplicative_seasonal | 23.5908 | 30.4560 | 36.5447 |
| GARCH | multiplicative_seasonal | -1387.7070 | -1781.0965 | -2122.3072 |
| HistoricAverage | multiplicative_seasonal | N/A | N/A | N/A |
| Holt | multiplicative_seasonal | -1.7157 | -2.1966 | -2.6172 |
| HoltWinters | multiplicative_seasonal | -2.3553 | -3.0222 | -3.6011 |
| IMAPA | multiplicative_seasonal | N/A | N/A | N/A |
| MFLES | multiplicative_seasonal | N/A | N/A | N/A |
| MSTLForecaster | multiplicative_seasonal | -11.5461 | -14.8192 | -17.6582 |
| Naive | multiplicative_seasonal | 0.0076 | 0.0153 | 0.0184 |
| OptimizedTheta | multiplicative_seasonal | -6.5265 | -8.5733 | -9.5045 |
| RandomWalkWithDrift | multiplicative_seasonal | -2.0313 | -2.6016 | -3.0998 |
| SARIMA_1_1_1_1_1_1_12 | multiplicative_seasonal | -3.7615 | -4.8270 | -5.7517 |
| SES | multiplicative_seasonal | N/A | N/A | N/A |
| SeasonalES | multiplicative_seasonal | N/A | N/A | N/A |
| SeasonalNaive | multiplicative_seasonal | 0.0023 | 0.0047 | 0.0057 |
| SeasonalWindowAverage | multiplicative_seasonal | N/A | N/A | N/A |
| TBATS | multiplicative_seasonal | -8.7479 | -11.2302 | -13.3848 |
| TSB | multiplicative_seasonal | N/A | N/A | N/A |
| Theta | multiplicative_seasonal | 17.1898 | 22.2584 | 26.7611 |
| WindowAverage | multiplicative_seasonal | N/A | N/A | N/A |
| ADIDA | multiplicative_trend_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | multiplicative_trend_seasonal | -24.8500 | -31.8920 | -38.0016 |
| AutoARIMA | multiplicative_trend_seasonal | 0.0807 | 0.1043 | 0.1243 |
| AutoETS | multiplicative_trend_seasonal | N/A | N/A | N/A |
| AutoTBATS | multiplicative_trend_seasonal | -6.6192 | -8.4970 | -10.1265 |
| AutoTheta | multiplicative_trend_seasonal | -0.9415 | -0.9561 | -0.8629 |
| Croston | multiplicative_trend_seasonal | N/A | N/A | N/A |
| CrostonSBA | multiplicative_trend_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | multiplicative_trend_seasonal | -1.0128 | -1.1587 | -1.1064 |
| DynamicTheta | multiplicative_trend_seasonal | 14.1164 | 18.4355 | 22.2772 |
| GARCH | multiplicative_trend_seasonal | -993.0719 | -1274.5888 | -1518.7661 |
| HistoricAverage | multiplicative_trend_seasonal | N/A | N/A | N/A |
| Holt | multiplicative_trend_seasonal | -1.0753 | -1.3767 | -1.6403 |
| HoltWinters | multiplicative_trend_seasonal | -0.9428 | -1.2093 | -1.4409 |
| IMAPA | multiplicative_trend_seasonal | N/A | N/A | N/A |
| MFLES | multiplicative_trend_seasonal | N/A | N/A | N/A |
| MSTLForecaster | multiplicative_trend_seasonal | -16.8359 | -21.6087 | -25.7483 |
| Naive | multiplicative_trend_seasonal | 0.0048 | 0.0096 | 0.0116 |
| OptimizedTheta | multiplicative_trend_seasonal | -0.3017 | -0.2543 | -0.0535 |
| RandomWalkWithDrift | multiplicative_trend_seasonal | -1.2731 | -1.6305 | -1.9428 |
| SARIMA_1_1_1_1_1_1_12 | multiplicative_trend_seasonal | 0.6912 | 0.8880 | 1.0582 |
| SES | multiplicative_trend_seasonal | N/A | N/A | N/A |
| SeasonalES | multiplicative_trend_seasonal | N/A | N/A | N/A |
| SeasonalNaive | multiplicative_trend_seasonal | 0.0024 | 0.0049 | 0.0059 |
| SeasonalWindowAverage | multiplicative_trend_seasonal | N/A | N/A | N/A |
| TBATS | multiplicative_trend_seasonal | -6.6192 | -8.4970 | -10.1265 |
| TSB | multiplicative_trend_seasonal | N/A | N/A | N/A |
| Theta | multiplicative_trend_seasonal | 16.5734 | 21.5258 | 25.9260 |
| WindowAverage | multiplicative_trend_seasonal | N/A | N/A | N/A |
| ADIDA | noisy_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | noisy_seasonal | 31.5084 | 40.4461 | 48.1947 |
| AutoARIMA | noisy_seasonal | -10.8152 | -13.8789 | -16.5376 |
| AutoETS | noisy_seasonal | -2.0354 | -2.6105 | -3.1106 |
| AutoTBATS | noisy_seasonal | -18.5702 | -23.8344 | -28.4005 |
| AutoTheta | noisy_seasonal | 12.6003 | 15.8539 | 20.1435 |
| Croston | noisy_seasonal | N/A | N/A | N/A |
| CrostonSBA | noisy_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | noisy_seasonal | 12.1233 | 15.2024 | 19.4359 |
| DynamicTheta | noisy_seasonal | 27.4100 | 34.8619 | 42.7989 |
| GARCH | noisy_seasonal | -165.6257 | -212.5761 | -253.2999 |
| HistoricAverage | noisy_seasonal | N/A | N/A | N/A |
| Holt | noisy_seasonal | 29.0506 | 37.2911 | 44.4353 |
| HoltWinters | noisy_seasonal | 2.5452 | 3.2691 | 3.8954 |
| IMAPA | noisy_seasonal | N/A | N/A | N/A |
| MFLES | noisy_seasonal | N/A | N/A | N/A |
| MSTLForecaster | noisy_seasonal | -18.5549 | -23.8149 | -28.3772 |
| Naive | noisy_seasonal | 0.0102 | 0.0206 | 0.0249 |
| OptimizedTheta | noisy_seasonal | 9.5239 | 11.8187 | 15.5357 |
| RandomWalkWithDrift | noisy_seasonal | -2.7462 | -3.5172 | -4.1908 |
| SARIMA_1_1_1_1_1_1_12 | noisy_seasonal | 7.8592 | 10.0907 | 12.0239 |
| SES | noisy_seasonal | N/A | N/A | N/A |
| SeasonalES | noisy_seasonal | N/A | N/A | N/A |
| SeasonalNaive | noisy_seasonal | 0.0037 | 0.0074 | 0.0089 |
| SeasonalWindowAverage | noisy_seasonal | N/A | N/A | N/A |
| TBATS | noisy_seasonal | -21.9813 | -28.3112 | -33.8607 |
| TSB | noisy_seasonal | N/A | N/A | N/A |
| Theta | noisy_seasonal | 16.2583 | 20.5493 | 25.7384 |
| WindowAverage | noisy_seasonal | N/A | N/A | N/A |
| ADIDA | outlier_series | N/A | N/A | N/A |
| ARIMA_1_1_1 | outlier_series | 19.4790 | 25.0054 | 29.7959 |
| AutoARIMA | outlier_series | -4.1372 | -5.3085 | -6.3254 |
| AutoETS | outlier_series | -4.3800 | -5.6206 | -6.6973 |
| AutoTBATS | outlier_series | -13.5967 | -17.4511 | -20.7943 |
| AutoTheta | outlier_series | 13.3902 | 16.9102 | 20.9280 |
| Croston | outlier_series | N/A | N/A | N/A |
| CrostonSBA | outlier_series | N/A | N/A | N/A |
| DynamicOptimizedTheta | outlier_series | 5.6908 | 7.5290 | 9.4949 |
| DynamicTheta | outlier_series | 21.6900 | 27.5521 | 33.6519 |
| GARCH | outlier_series | -272.0789 | -349.2066 | -416.1053 |
| HistoricAverage | outlier_series | N/A | N/A | N/A |
| Holt | outlier_series | 8.5469 | 10.9747 | 13.0773 |
| HoltWinters | outlier_series | 2.4568 | 3.1549 | 3.7594 |
| IMAPA | outlier_series | N/A | N/A | N/A |
| MFLES | outlier_series | N/A | N/A | N/A |
| MSTLForecaster | outlier_series | -12.2998 | -15.7867 | -18.8110 |
| Naive | outlier_series | 0.0068 | 0.0137 | 0.0165 |
| OptimizedTheta | outlier_series | 10.1132 | 12.7159 | 16.1341 |
| RandomWalkWithDrift | outlier_series | -1.8161 | -2.3260 | -2.7714 |
| SARIMA_1_1_1_1_1_1_12 | outlier_series | -3.4007 | -4.3626 | -5.1983 |
| SES | outlier_series | N/A | N/A | N/A |
| SeasonalES | outlier_series | N/A | N/A | N/A |
| SeasonalNaive | outlier_series | 0.0027 | 0.0055 | 0.0066 |
| SeasonalWindowAverage | outlier_series | N/A | N/A | N/A |
| TBATS | outlier_series | -14.7496 | -18.9308 | -22.5574 |
| TSB | outlier_series | N/A | N/A | N/A |
| Theta | outlier_series | 14.1322 | 17.8626 | 22.0629 |
| WindowAverage | outlier_series | N/A | N/A | N/A |
| ADIDA | quarterly_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | quarterly_seasonal | 44.2945 | 56.8583 | 67.7511 |
| AutoARIMA | quarterly_seasonal | 1.2393 | 1.5915 | 1.8964 |
| AutoETS | quarterly_seasonal | -0.9730 | -1.2482 | -1.4873 |
| AutoTBATS | quarterly_seasonal | -6.0244 | -7.7322 | -9.2134 |
| AutoTheta | quarterly_seasonal | 9.0122 | 11.4916 | 14.0623 |
| Croston | quarterly_seasonal | N/A | N/A | N/A |
| CrostonSBA | quarterly_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | quarterly_seasonal | 3.1992 | 3.9458 | 5.4213 |
| DynamicTheta | quarterly_seasonal | 26.3272 | 33.7142 | 40.5551 |
| GARCH | quarterly_seasonal | -8.9520 | -11.4896 | -13.6908 |
| HistoricAverage | quarterly_seasonal | N/A | N/A | N/A |
| Holt | quarterly_seasonal | 30.9957 | 39.7880 | 47.4105 |
| HoltWinters | quarterly_seasonal | 1.0541 | 1.3538 | 1.6132 |
| IMAPA | quarterly_seasonal | N/A | N/A | N/A |
| MFLES | quarterly_seasonal | N/A | N/A | N/A |
| MSTLForecaster | quarterly_seasonal | -6.2048 | -7.9638 | -9.4894 |
| Naive | quarterly_seasonal | 0.0105 | 0.0212 | 0.0256 |
| OptimizedTheta | quarterly_seasonal | 8.0072 | 10.1879 | 12.5654 |
| RandomWalkWithDrift | quarterly_seasonal | -2.8241 | -3.6170 | -4.3097 |
| SARIMA_1_1_1_1_1_1_12 | quarterly_seasonal | 1.5566 | 1.9989 | 2.3819 |
| SES | quarterly_seasonal | N/A | N/A | N/A |
| SeasonalES | quarterly_seasonal | N/A | N/A | N/A |
| SeasonalNaive | quarterly_seasonal | 0.0013 | 0.0026 | 0.0032 |
| SeasonalWindowAverage | quarterly_seasonal | N/A | N/A | N/A |
| TBATS | quarterly_seasonal | -7.8830 | -10.1177 | -12.0560 |
| TSB | quarterly_seasonal | N/A | N/A | N/A |
| Theta | quarterly_seasonal | 5.9117 | 7.5118 | 9.3201 |
| WindowAverage | quarterly_seasonal | N/A | N/A | N/A |
| ADIDA | random_walk | N/A | N/A | N/A |
| ARIMA_1_1_1 | random_walk | -0.5622 | -0.7209 | -0.8590 |
| AutoARIMA | random_walk | -3.1588 | -4.0540 | -4.8306 |
| AutoETS | random_walk | -3.8799 | -4.9796 | -5.9335 |
| AutoTBATS | random_walk | -6.1528 | -7.8970 | -9.4099 |
| AutoTheta | random_walk | -3.9466 | -4.7294 | -5.4070 |
| Croston | random_walk | N/A | N/A | N/A |
| CrostonSBA | random_walk | N/A | N/A | N/A |
| DynamicOptimizedTheta | random_walk | -3.9261 | -4.7018 | -5.3795 |
| DynamicTheta | random_walk | 22.0770 | 28.6754 | 34.3917 |
| GARCH | random_walk | -24.6621 | -31.6522 | -37.7159 |
| HistoricAverage | random_walk | N/A | N/A | N/A |
| Holt | random_walk | -0.1949 | -0.2495 | -0.2973 |
| HoltWinters | random_walk | -2.7559 | -3.5368 | -4.2143 |
| IMAPA | random_walk | N/A | N/A | N/A |
| MFLES | random_walk | N/A | N/A | N/A |
| MSTLForecaster | random_walk | -5.8685 | -7.5322 | -8.9751 |
| Naive | random_walk | 0.0009 | 0.0018 | 0.0021 |
| OptimizedTheta | random_walk | -3.9466 | -4.7294 | -5.4070 |
| RandomWalkWithDrift | random_walk | -0.2316 | -0.2966 | -0.3534 |
| SARIMA_1_1_1_1_1_1_12 | random_walk | -2.6364 | -3.3834 | -4.0316 |
| SES | random_walk | N/A | N/A | N/A |
| SeasonalES | random_walk | N/A | N/A | N/A |
| SeasonalNaive | random_walk | 0.0016 | 0.0032 | 0.0039 |
| SeasonalWindowAverage | random_walk | N/A | N/A | N/A |
| TBATS | random_walk | -6.1892 | -7.9438 | -9.4656 |
| TSB | random_walk | N/A | N/A | N/A |
| Theta | random_walk | 7.3334 | 9.7484 | 11.8372 |
| WindowAverage | random_walk | N/A | N/A | N/A |
| ADIDA | seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | seasonal | -13.0800 | -16.7853 | -20.0008 |
| AutoARIMA | seasonal | 1.5218 | 1.9539 | 2.3282 |
| AutoETS | seasonal | -0.4858 | -0.6231 | -0.7424 |
| AutoTBATS | seasonal | -4.7882 | -6.1455 | -7.3228 |
| AutoTheta | seasonal | 6.0445 | 7.6846 | 9.4535 |
| Croston | seasonal | N/A | N/A | N/A |
| CrostonSBA | seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | seasonal | 0.2195 | 0.1093 | 0.7747 |
| DynamicTheta | seasonal | 23.9826 | 30.7090 | 36.8917 |
| GARCH | seasonal | -127.2830 | -163.3641 | -194.6602 |
| HistoricAverage | seasonal | N/A | N/A | N/A |
| Holt | seasonal | -0.9198 | -1.1776 | -1.4031 |
| HoltWinters | seasonal | 0.7601 | 0.9763 | 1.1633 |
| IMAPA | seasonal | N/A | N/A | N/A |
| MFLES | seasonal | N/A | N/A | N/A |
| MSTLForecaster | seasonal | -4.8332 | -6.2033 | -7.3917 |
| Naive | seasonal | 0.0041 | 0.0082 | 0.0099 |
| OptimizedTheta | seasonal | 5.9422 | 7.5519 | 9.3013 |
| RandomWalkWithDrift | seasonal | -1.0902 | -1.3963 | -1.6636 |
| SARIMA_1_1_1_1_1_1_12 | seasonal | 0.6492 | 0.8340 | 0.9938 |
| SES | seasonal | N/A | N/A | N/A |
| SeasonalES | seasonal | N/A | N/A | N/A |
| SeasonalNaive | seasonal | 0.0011 | 0.0021 | 0.0026 |
| SeasonalWindowAverage | seasonal | N/A | N/A | N/A |
| TBATS | seasonal | -6.0548 | -7.7752 | -9.2698 |
| TSB | seasonal | N/A | N/A | N/A |
| Theta | seasonal | 4.4188 | 5.5980 | 6.9671 |
| WindowAverage | seasonal | N/A | N/A | N/A |
| ADIDA | seasonal_negative | N/A | N/A | N/A |
| ARIMA_1_1_1 | seasonal_negative | -24.7996 | -31.8282 | -37.9256 |
| AutoARIMA | seasonal_negative | 0.4579 | 0.5881 | 0.7007 |
| AutoETS | seasonal_negative | -0.2874 | -0.3687 | -0.4393 |
| AutoTBATS | seasonal_negative | 9.2602 | 11.8865 | 14.1637 |
| AutoTheta | seasonal_negative | -13.1134 | -17.1015 | -19.3229 |
| Croston | seasonal_negative | N/A | N/A | N/A |
| CrostonSBA | seasonal_negative | N/A | N/A | N/A |
| DynamicOptimizedTheta | seasonal_negative | -13.1325 | -17.1529 | -19.3489 |
| DynamicTheta | seasonal_negative | -13.4699 | -17.5585 | -19.8700 |
| GARCH | seasonal_negative | -114.1962 | -146.5674 | -174.6458 |
| HistoricAverage | seasonal_negative | N/A | N/A | N/A |
| Holt | seasonal_negative | -0.7609 | -0.9742 | -1.1607 |
| HoltWinters | seasonal_negative | 0.2841 | 0.3650 | 0.4349 |
| IMAPA | seasonal_negative | N/A | N/A | N/A |
| MFLES | seasonal_negative | N/A | N/A | N/A |
| MSTLForecaster | seasonal_negative | -2.1397 | -2.7462 | -3.2723 |
| Naive | seasonal_negative | 0.0034 | 0.0068 | 0.0082 |
| OptimizedTheta | seasonal_negative | -13.1281 | -17.1312 | -19.3479 |
| RandomWalkWithDrift | seasonal_negative | -0.9006 | -1.1534 | -1.3743 |
| SARIMA_1_1_1_1_1_1_12 | seasonal_negative | 0.2924 | 0.3756 | 0.4476 |
| SES | seasonal_negative | N/A | N/A | N/A |
| SeasonalES | seasonal_negative | N/A | N/A | N/A |
| SeasonalNaive | seasonal_negative | 0.0004 | 0.0009 | 0.0011 |
| SeasonalWindowAverage | seasonal_negative | N/A | N/A | N/A |
| TBATS | seasonal_negative | 8.6166 | 11.0605 | 13.1795 |
| TSB | seasonal_negative | N/A | N/A | N/A |
| Theta | seasonal_negative | -14.2371 | -18.5439 | -21.0416 |
| WindowAverage | seasonal_negative | N/A | N/A | N/A |
| ADIDA | seasonal_trend_break | N/A | N/A | N/A |
| ARIMA_1_1_1 | seasonal_trend_break | -12.7997 | -16.4255 | -19.5721 |
| AutoARIMA | seasonal_trend_break | -2.3256 | -2.9841 | -3.5558 |
| AutoETS | seasonal_trend_break | -4.2164 | -5.4112 | -6.4478 |
| AutoTBATS | seasonal_trend_break | -8.1707 | -10.4869 | -12.4959 |
| AutoTheta | seasonal_trend_break | -3.1982 | -3.5415 | -3.8839 |
| Croston | seasonal_trend_break | N/A | N/A | N/A |
| CrostonSBA | seasonal_trend_break | N/A | N/A | N/A |
| DynamicOptimizedTheta | seasonal_trend_break | -3.3136 | -3.7059 | -4.0535 |
| DynamicTheta | seasonal_trend_break | 18.7332 | 24.6468 | 29.7007 |
| GARCH | seasonal_trend_break | -225.5490 | -289.4864 | -344.9443 |
| HistoricAverage | seasonal_trend_break | N/A | N/A | N/A |
| Holt | seasonal_trend_break | -0.9251 | -1.1844 | -1.4112 |
| HoltWinters | seasonal_trend_break | -2.4658 | -3.1641 | -3.7703 |
| IMAPA | seasonal_trend_break | N/A | N/A | N/A |
| MFLES | seasonal_trend_break | N/A | N/A | N/A |
| MSTLForecaster | seasonal_trend_break | -8.0756 | -10.3649 | -12.3506 |
| Naive | seasonal_trend_break | 0.0041 | 0.0082 | 0.0099 |
| OptimizedTheta | seasonal_trend_break | -3.3645 | -3.7550 | -4.1382 |
| RandomWalkWithDrift | seasonal_trend_break | -1.0952 | -1.4026 | -1.6712 |
| SARIMA_1_1_1_1_1_1_12 | seasonal_trend_break | -2.3582 | -3.0259 | -3.6055 |
| SES | seasonal_trend_break | N/A | N/A | N/A |
| SeasonalES | seasonal_trend_break | N/A | N/A | N/A |
| SeasonalNaive | seasonal_trend_break | 0.0022 | 0.0045 | 0.0054 |
| SeasonalWindowAverage | seasonal_trend_break | N/A | N/A | N/A |
| TBATS | seasonal_trend_break | -8.1734 | -10.4904 | -12.5001 |
| TSB | seasonal_trend_break | N/A | N/A | N/A |
| Theta | seasonal_trend_break | 10.6086 | 14.1738 | 17.2309 |
| WindowAverage | seasonal_trend_break | N/A | N/A | N/A |
| ADIDA | stationary | N/A | N/A | N/A |
| ARIMA_1_1_1 | stationary | 14.8095 | 19.0104 | 22.6524 |
| AutoARIMA | stationary | 1.4936 | 1.9181 | 2.2856 |
| AutoETS | stationary | -1.0456 | -1.3410 | -1.5979 |
| AutoTBATS | stationary | -9.3092 | -11.9481 | -14.2371 |
| AutoTheta | stationary | 13.1655 | 16.7515 | 20.5388 |
| Croston | stationary | N/A | N/A | N/A |
| CrostonSBA | stationary | N/A | N/A | N/A |
| DynamicOptimizedTheta | stationary | 7.5914 | 9.5239 | 12.1595 |
| DynamicTheta | stationary | 21.8454 | 27.8891 | 33.8249 |
| GARCH | stationary | -26.6069 | -34.1486 | -40.6906 |
| HistoricAverage | stationary | N/A | N/A | N/A |
| Holt | stationary | 13.7572 | 17.6596 | 21.0428 |
| HoltWinters | stationary | 1.1221 | 1.4414 | 1.7175 |
| IMAPA | stationary | N/A | N/A | N/A |
| MFLES | stationary | N/A | N/A | N/A |
| MSTLForecaster | stationary | -8.2080 | -10.5348 | -12.5530 |
| Naive | stationary | 0.0045 | 0.0092 | 0.0111 |
| OptimizedTheta | stationary | 12.4088 | 15.7655 | 19.4074 |
| RandomWalkWithDrift | stationary | -1.2216 | -1.5646 | -1.8642 |
| SARIMA_1_1_1_1_1_1_12 | stationary | 1.3862 | 1.7806 | 2.1218 |
| SES | stationary | N/A | N/A | N/A |
| SeasonalES | stationary | N/A | N/A | N/A |
| SeasonalNaive | stationary | 0.0018 | 0.0036 | 0.0043 |
| SeasonalWindowAverage | stationary | N/A | N/A | N/A |
| TBATS | stationary | -9.6027 | -12.3249 | -14.6860 |
| TSB | stationary | N/A | N/A | N/A |
| Theta | stationary | 8.6745 | 10.9869 | 13.6698 |
| WindowAverage | stationary | N/A | N/A | N/A |
| ADIDA | step_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | step_seasonal | 0.9300 | 1.1992 | 1.4291 |
| AutoARIMA | step_seasonal | 0.2560 | 0.3292 | 0.3922 |
| AutoETS | step_seasonal | -0.5952 | -0.7635 | -0.9098 |
| AutoTBATS | step_seasonal | -4.0446 | -5.1911 | -6.1856 |
| AutoTheta | step_seasonal | 6.2179 | 7.9289 | 9.7192 |
| Croston | step_seasonal | N/A | N/A | N/A |
| CrostonSBA | step_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | step_seasonal | -0.4223 | -0.6815 | -0.1575 |
| DynamicTheta | step_seasonal | 23.7852 | 30.4773 | 36.5909 |
| GARCH | step_seasonal | -174.2564 | -223.6534 | -266.4994 |
| HistoricAverage | step_seasonal | N/A | N/A | N/A |
| Holt | step_seasonal | 1.1464 | 1.4768 | 1.7599 |
| HoltWinters | step_seasonal | 0.7976 | 1.0243 | 1.2206 |
| IMAPA | step_seasonal | N/A | N/A | N/A |
| MFLES | step_seasonal | N/A | N/A | N/A |
| MSTLForecaster | step_seasonal | -4.4421 | -5.7013 | -6.7935 |
| Naive | step_seasonal | 0.0074 | 0.0150 | 0.0181 |
| OptimizedTheta | step_seasonal | 5.3444 | 6.7957 | 8.4209 |
| RandomWalkWithDrift | step_seasonal | -2.0000 | -2.5615 | -3.0520 |
| SARIMA_1_1_1_1_1_1_12 | step_seasonal | 1.1083 | 1.4232 | 1.6958 |
| SES | step_seasonal | N/A | N/A | N/A |
| SeasonalES | step_seasonal | N/A | N/A | N/A |
| SeasonalNaive | step_seasonal | 0.0009 | 0.0019 | 0.0023 |
| SeasonalWindowAverage | step_seasonal | N/A | N/A | N/A |
| TBATS | step_seasonal | -5.1932 | -6.6681 | -7.9490 |
| TSB | step_seasonal | N/A | N/A | N/A |
| Theta | step_seasonal | 4.1049 | 5.2166 | 6.4874 |
| WindowAverage | step_seasonal | N/A | N/A | N/A |
| ADIDA | strong_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | strong_seasonal | -86.4538 | -110.9573 | -132.2136 |
| AutoARIMA | strong_seasonal | 1.0693 | 1.3731 | 1.6361 |
| AutoETS | strong_seasonal | -0.3950 | -0.5065 | -0.6035 |
| AutoTBATS | strong_seasonal | -4.3877 | -5.6315 | -6.7104 |
| AutoTheta | strong_seasonal | 5.8473 | 7.4428 | 9.1284 |
| Croston | strong_seasonal | N/A | N/A | N/A |
| CrostonSBA | strong_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | strong_seasonal | -13.1598 | -17.2455 | -19.1875 |
| DynamicTheta | strong_seasonal | 51.6946 | 66.2912 | 79.2535 |
| GARCH | strong_seasonal | -2060.7213 | -2644.8997 | -3151.5921 |
| HistoricAverage | strong_seasonal | N/A | N/A | N/A |
| Holt | strong_seasonal | -26.9966 | -34.6402 | -41.2760 |
| HoltWinters | strong_seasonal | 0.6788 | 0.8718 | 1.0389 |
| IMAPA | strong_seasonal | N/A | N/A | N/A |
| MFLES | strong_seasonal | N/A | N/A | N/A |
| MSTLForecaster | strong_seasonal | -4.2749 | -5.4867 | -6.5379 |
| Naive | strong_seasonal | 0.0130 | 0.0263 | 0.0316 |
| OptimizedTheta | strong_seasonal | 2.9101 | 3.6554 | 4.6849 |
| RandomWalkWithDrift | strong_seasonal | -3.4937 | -4.4745 | -5.3314 |
| SARIMA_1_1_1_1_1_1_12 | strong_seasonal | 0.9077 | 1.1658 | 1.3891 |
| SES | strong_seasonal | N/A | N/A | N/A |
| SeasonalES | strong_seasonal | N/A | N/A | N/A |
| SeasonalNaive | strong_seasonal | 0.0009 | 0.0019 | 0.0022 |
| SeasonalWindowAverage | strong_seasonal | N/A | N/A | N/A |
| TBATS | strong_seasonal | -10.0819 | -12.9440 | -15.4290 |
| TSB | strong_seasonal | N/A | N/A | N/A |
| Theta | strong_seasonal | 4.6178 | 5.8647 | 7.2480 |
| WindowAverage | strong_seasonal | N/A | N/A | N/A |
| ADIDA | structural_break | N/A | N/A | N/A |
| ARIMA_1_1_1 | structural_break | 12.3096 | 15.8021 | 18.8294 |
| AutoARIMA | structural_break | 0.9095 | 1.1691 | 1.3931 |
| AutoETS | structural_break | -5.7983 | -7.4410 | -8.8664 |
| AutoTBATS | structural_break | -15.2571 | -19.5823 | -23.3337 |
| AutoTheta | structural_break | -2.8874 | -3.2487 | -3.3277 |
| Croston | structural_break | N/A | N/A | N/A |
| CrostonSBA | structural_break | N/A | N/A | N/A |
| DynamicOptimizedTheta | structural_break | -2.8427 | -3.1641 | -3.2206 |
| DynamicTheta | structural_break | 20.2885 | 26.5743 | 32.2597 |
| GARCH | structural_break | -220.4041 | -282.8831 | -337.0759 |
| HistoricAverage | structural_break | N/A | N/A | N/A |
| Holt | structural_break | 12.1061 | 15.5409 | 18.5182 |
| HoltWinters | structural_break | -3.3071 | -4.2432 | -5.0560 |
| IMAPA | structural_break | N/A | N/A | N/A |
| MFLES | structural_break | N/A | N/A | N/A |
| MSTLForecaster | structural_break | -15.0527 | -19.3199 | -23.0211 |
| Naive | structural_break | 0.0045 | 0.0092 | 0.0110 |
| OptimizedTheta | structural_break | -2.9073 | -3.2797 | -3.3544 |
| RandomWalkWithDrift | structural_break | -1.2190 | -1.5612 | -1.8602 |
| SARIMA_1_1_1_1_1_1_12 | structural_break | -1.7533 | -2.2486 | -2.6793 |
| SES | structural_break | N/A | N/A | N/A |
| SeasonalES | structural_break | N/A | N/A | N/A |
| SeasonalNaive | structural_break | 0.0028 | 0.0056 | 0.0068 |
| SeasonalWindowAverage | structural_break | N/A | N/A | N/A |
| TBATS | structural_break | -17.1922 | -22.0659 | -26.2932 |
| TSB | structural_break | N/A | N/A | N/A |
| Theta | structural_break | 10.2548 | 13.6204 | 16.7732 |
| WindowAverage | structural_break | N/A | N/A | N/A |
| ADIDA | trend | N/A | N/A | N/A |
| ARIMA_1_1_1 | trend | 6.6467 | 8.5329 | 10.1677 |
| AutoARIMA | trend | -3.9396 | -5.0555 | -6.0240 |
| AutoETS | trend | -0.4456 | -0.5712 | -0.6806 |
| AutoTBATS | trend | -8.0840 | -10.3748 | -12.3612 |
| AutoTheta | trend | 6.1365 | 7.7453 | 9.6985 |
| Croston | trend | N/A | N/A | N/A |
| CrostonSBA | trend | N/A | N/A | N/A |
| DynamicOptimizedTheta | trend | 3.4003 | 4.2306 | 5.5216 |
| DynamicTheta | trend | 8.4117 | 11.0910 | 13.5523 |
| GARCH | trend | -605.0071 | -776.5140 | -925.2734 |
| HistoricAverage | trend | N/A | N/A | N/A |
| Holt | trend | 10.5058 | 13.4859 | 16.0695 |
| HoltWinters | trend | 1.8453 | 2.3695 | 2.8235 |
| IMAPA | trend | N/A | N/A | N/A |
| MFLES | trend | N/A | N/A | N/A |
| MSTLForecaster | trend | -7.2800 | -9.3437 | -11.1338 |
| Naive | trend | 0.0034 | 0.0068 | 0.0082 |
| OptimizedTheta | trend | 6.1365 | 7.7453 | 9.6985 |
| RandomWalkWithDrift | trend | -0.8941 | -1.1451 | -1.3644 |
| SARIMA_1_1_1_1_1_1_12 | trend | 1.8772 | 2.4106 | 2.8724 |
| SES | trend | N/A | N/A | N/A |
| SeasonalES | trend | N/A | N/A | N/A |
| SeasonalNaive | trend | 0.0027 | 0.0054 | 0.0065 |
| SeasonalWindowAverage | trend | N/A | N/A | N/A |
| TBATS | trend | -8.0269 | -10.3016 | -12.2741 |
| TSB | trend | N/A | N/A | N/A |
| Theta | trend | 16.6242 | 21.6192 | 26.0710 |
| WindowAverage | trend | N/A | N/A | N/A |
| ADIDA | trend_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | trend_seasonal | -6.1995 | -7.9545 | -9.4782 |
| AutoARIMA | trend_seasonal | -2.7053 | -3.4715 | -4.1365 |
| AutoETS | trend_seasonal | -0.6664 | -0.8548 | -1.0185 |
| AutoTBATS | trend_seasonal | -5.7679 | -7.4025 | -8.8201 |
| AutoTheta | trend_seasonal | 1.4114 | 2.0868 | 2.8279 |
| Croston | trend_seasonal | N/A | N/A | N/A |
| CrostonSBA | trend_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | trend_seasonal | -1.2218 | -1.3771 | -1.1996 |
| DynamicTheta | trend_seasonal | 8.3797 | 11.0788 | 13.5581 |
| GARCH | trend_seasonal | -281.5787 | -361.3994 | -430.6339 |
| HistoricAverage | trend_seasonal | N/A | N/A | N/A |
| Holt | trend_seasonal | -0.7862 | -1.0065 | -1.1993 |
| HoltWinters | trend_seasonal | 1.1594 | 1.4888 | 1.7740 |
| IMAPA | trend_seasonal | N/A | N/A | N/A |
| MFLES | trend_seasonal | N/A | N/A | N/A |
| MSTLForecaster | trend_seasonal | -4.8611 | -6.2392 | -7.4345 |
| Naive | trend_seasonal | 0.0035 | 0.0070 | 0.0085 |
| OptimizedTheta | trend_seasonal | 2.6405 | 3.2729 | 4.3267 |
| RandomWalkWithDrift | trend_seasonal | -0.9309 | -1.1922 | -1.4205 |
| SARIMA_1_1_1_1_1_1_12 | trend_seasonal | 1.8470 | 2.3714 | 2.8257 |
| SES | trend_seasonal | N/A | N/A | N/A |
| SeasonalES | trend_seasonal | N/A | N/A | N/A |
| SeasonalNaive | trend_seasonal | 0.0016 | 0.0032 | 0.0039 |
| SeasonalWindowAverage | trend_seasonal | N/A | N/A | N/A |
| TBATS | trend_seasonal | -5.7679 | -7.4025 | -8.8201 |
| TSB | trend_seasonal | N/A | N/A | N/A |
| Theta | trend_seasonal | 7.4022 | 9.7765 | 11.9908 |
| WindowAverage | trend_seasonal | N/A | N/A | N/A |


---

## Detailed Point Forecast Differences

Largest absolute differences:

| Model | Series | Step | Rust | statsforecast | Difference |
|-------|--------|------|------|---------------|------------|
| AutoARIMA | heteroscedastic | 12 | 52.7939 | 17.5380 | 35.2558 |
| AutoARIMA | heteroscedastic | 4 | 50.4505 | 19.7809 | 30.6696 |
| AutoARIMA | heteroscedastic | 7 | 52.0160 | 21.8782 | 30.1378 |
| AutoARIMA | heteroscedastic | 5 | 51.4586 | 21.3487 | 30.1099 |
| AutoARIMA | heteroscedastic | 11 | 51.0421 | 22.5456 | 28.4965 |
| AutoARIMA | heteroscedastic | 9 | 51.4262 | 23.4617 | 27.9645 |
| AutoARIMA | heteroscedastic | 8 | 51.6023 | 24.9065 | 26.6958 |
| AutoARIMA | heteroscedastic | 3 | 55.6357 | 31.5864 | 24.0493 |
| AutoARIMA | heteroscedastic | 6 | 50.4690 | 27.5501 | 22.9189 |
| AutoARIMA | heteroscedastic | 10 | 49.2970 | 27.6081 | 21.6889 |

---

## Metrics by Forecast Horizon Step

Aggregated metrics across all models and series types by forecast step:

| Step | MAD | Median | Max Diff | Mean Diff | Std |
|------|-----|--------|----------|-----------|-----|
| 1 | 0.5137 | 0.0010 | 16.1002 | 0.0481 | 1.4742 |
| 2 | 0.6069 | 0.0022 | 16.7127 | 0.0947 | 1.7084 |
| 3 | 0.5720 | 0.0030 | 24.0493 | 0.0392 | 1.7184 |
| 4 | 0.5982 | 0.0033 | 30.6696 | 0.0360 | 1.9136 |
| 5 | 0.6493 | 0.0039 | 30.1099 | 0.0839 | 2.0409 |
| 6 | 0.7196 | 0.0043 | 22.9189 | 0.0226 | 2.0374 |
| 7 | 0.8200 | 0.0038 | 30.1378 | -0.0068 | 2.3767 |
| 8 | 0.7487 | 0.0040 | 26.6958 | 0.0889 | 2.1775 |
| 9 | 0.7611 | 0.0036 | 27.9645 | 0.0595 | 2.2524 |
| 10 | 0.8520 | 0.0034 | 21.6889 | 0.0643 | 2.3416 |
| 11 | 0.8970 | 0.0036 | 28.4965 | 0.1084 | 2.5495 |
| 12 | 0.9851 | 0.0034 | 35.2558 | -0.0102 | 2.8478 |

---

## Notes

- **MAD**: Mean Absolute Difference between forecasts
- **Median**: Median Absolute Difference (robust to outliers)
- **Max Diff**: Maximum absolute difference
- **Correlation**: Pearson correlation between forecast values
- **CI Width Diff**: Mean difference in confidence interval width (Rust - statsforecast)

Differences are expected due to:
- Different optimization algorithms for parameter estimation
- Different numerical precision
- Different default parameter values
- Implementation variations in confidence interval calculation
