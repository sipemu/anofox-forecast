# Forecast Validation Report

Generated: 2025-12-12 12:42:02

## Summary

- **Rust implementation**: anofox-forecast
- **Python implementation**: statsforecast (NIXTLA)
- **Forecast horizon**: 12 steps
- **Confidence levels**: 80%, 90%, 95%

- **Models compared**: 29
- **Series types**: 11
- **Total comparisons**: 319

- **High agreement (corr >= 0.99)**: 193 combinations
- **Lower agreement (corr < 0.95)**: 47 combinations

---

## Results by Model

### ADIDA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0030 | 0.0030 | 0.0030 | N/A | N/A |
| multiplicative_seasonal | 0.0003 | 0.0003 | 0.0003 | N/A | N/A |
| noisy_seasonal | 0.0008 | 0.0008 | 0.0008 | N/A | N/A |
| seasonal | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| seasonal_negative | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| stationary | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0001 | 0.0001 | 0.0001 | 1.0000 | N/A |


### ARIMA_1_1_1

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0288 | 0.0288 | 0.0516 | 0.8265 | 4.8730 |
| intermittent | 0.5158 | 0.5113 | 0.5962 | 0.8384 | 16.7521 |
| long_memory | 1.9778 | 2.0124 | 2.3446 | -0.6939 | 27.5581 |
| multiplicative_seasonal | 1.8412 | 1.7507 | 3.9119 | 0.4620 | -93.4102 |
| noisy_seasonal | 0.1798 | 0.1798 | 0.2262 | 0.7992 | 48.1947 |
| seasonal | 0.0286 | 0.0282 | 0.0560 | 0.9999 | -23.3023 |
| seasonal_negative | 0.2886 | 0.2946 | 0.5394 | 0.9949 | -37.9256 |
| stationary | 0.0428 | 0.0429 | 0.0598 | 0.5039 | 22.6524 |
| structural_break | 1.5564 | 1.5554 | 2.6015 | -0.4816 | 18.8293 |
| trend | 4.0219 | 4.0286 | 6.6558 | -0.5139 | 10.1665 |
| trend_seasonal | 2.1003 | 2.0886 | 4.0188 | 0.7108 | -11.1858 |


### AutoARIMA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 2.4590 | 2.0446 | 5.6759 | -0.0615 | 0.5739 |
| intermittent | 0.5283 | 0.2884 | 1.8932 | 0.0000 | 1.7859 |
| long_memory | 0.3920 | 0.3896 | 0.7830 | N/A | -0.7432 |
| multiplicative_seasonal | 1.2752 | 0.9936 | 4.2101 | 0.9991 | 1.5566 |
| noisy_seasonal | 2.0681 | 1.4590 | 5.1766 | 0.8984 | 2.8163 |
| seasonal | 0.4609 | 0.3932 | 1.0145 | 0.9976 | 2.3282 |
| seasonal_negative | 0.5756 | 0.6505 | 0.9400 | 0.9988 | 0.7076 |
| stationary | 2.6987 | 2.7642 | 3.3738 | -0.0000 | 2.5098 |
| structural_break | 6.5697 | 5.3116 | 14.8115 | 0.0000 | 1.3994 |
| trend | 1.1791 | 0.8532 | 3.0542 | 0.8000 | 2.1285 |
| trend_seasonal | 0.2541 | 0.1608 | 0.7666 | 0.9984 | 0.9710 |


### AutoETS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0261 | 0.0261 | 0.0261 | N/A | 5.0129 |
| intermittent | 0.2462 | 0.2462 | 0.2705 | -0.0000 | 15.7281 |
| long_memory | 0.1445 | 0.1445 | 0.1445 | N/A | 27.9636 |
| multiplicative_seasonal | 0.9760 | 0.8461 | 2.0604 | 0.9989 | -1.3428 |
| noisy_seasonal | 1.4130 | 1.4130 | 1.5522 | -0.0000 | 44.6545 |
| seasonal | 0.4864 | 0.4179 | 1.2734 | 0.9967 | 1.1332 |
| seasonal_negative | 0.8652 | 0.7814 | 1.6761 | 0.9988 | 0.3094 |
| stationary | 1.1889 | 1.1889 | 1.3054 | N/A | 21.0028 |
| structural_break | 0.0019 | 0.0019 | 0.0019 | N/A | 18.4030 |
| trend | 0.0033 | 0.0033 | 0.0034 | 1.0000 | 16.0695 |
| trend_seasonal | 0.5714 | 0.2921 | 1.6695 | 0.9922 | 1.7740 |


### AutoTBATS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 1.9153 | 1.5830 | 3.8819 | -0.3012 | -12.1289 |
| intermittent | 1.1030 | 1.1030 | 1.5466 | 0.9748 | 14.9742 |
| long_memory | 1.4635 | 1.5717 | 2.2934 | 0.9522 | -20.5932 |
| multiplicative_seasonal | 5.2185 | 4.2063 | 14.3142 | 0.9927 | -11.6584 |
| noisy_seasonal | 1.2288 | 1.2914 | 2.8369 | 0.7302 | -28.4005 |
| seasonal | 0.6524 | 0.5830 | 1.5689 | 0.9979 | -7.3228 |
| seasonal_negative | 1.3502 | 0.5154 | 5.1619 | 0.9851 | 14.1637 |
| stationary | 1.3298 | 1.2986 | 2.4990 | 0.8876 | -14.2371 |
| structural_break | 2.1956 | 1.7731 | 5.1737 | -0.0900 | -23.3337 |
| trend | 2.4567 | 2.0682 | 4.6036 | 0.9925 | -11.1526 |
| trend_seasonal | 1.7996 | 1.9483 | 2.6845 | 0.9921 | -8.4888 |


### AutoTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.3499 | 0.3502 | 0.3793 | 0.9997 | -7.5647 |
| intermittent | 0.0762 | 0.0762 | 0.0794 | 1.0000 | 13.4816 |
| long_memory | 0.6257 | 0.6257 | 0.9343 | -0.9676 | 12.4631 |
| multiplicative_seasonal | 0.7042 | 0.5566 | 1.2410 | 1.0000 | 2.8594 |
| noisy_seasonal | 0.2877 | 0.2877 | 0.4256 | 0.0000 | 29.6325 |
| seasonal | 0.2996 | 0.2976 | 0.3597 | 1.0000 | 9.5103 |
| seasonal_negative | 0.2615 | 0.2615 | 0.2653 | 1.0000 | -19.3393 |
| stationary | 1.4509 | 1.4509 | 1.4686 | 1.0000 | 20.5508 |
| structural_break | 0.3110 | 0.3123 | 0.4066 | 0.9996 | -3.2545 |
| trend | 0.7631 | 0.7631 | 1.0388 | 1.0000 | 9.5374 |
| trend_seasonal | 0.5918 | 0.5284 | 0.8994 | 1.0000 | 6.4876 |


### Croston

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### CrostonSBA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### DynamicOptimizedTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.1340 | 0.1336 | 0.1380 | 1.0000 | -7.9327 |
| intermittent | 0.1255 | 0.1255 | 0.1263 | -0.0000 | 13.5709 |
| long_memory | 0.0993 | 0.0993 | 0.0993 | 0.7154 | 12.2416 |
| multiplicative_seasonal | 4.2555 | 2.9882 | 8.7776 | 0.9963 | 1.9153 |
| noisy_seasonal | 0.1010 | 0.1010 | 0.1010 | 1.0000 | 29.7767 |
| seasonal | 0.3138 | 0.3113 | 0.3814 | 1.0000 | 9.5584 |
| seasonal_negative | 0.2395 | 0.2395 | 0.2396 | 1.0000 | -19.2661 |
| stationary | 0.9207 | 0.9207 | 0.9207 | -1.0000 | 17.6630 |
| structural_break | 0.0184 | 0.0185 | 0.0190 | 1.0000 | -3.1711 |
| trend | 3.9947 | 3.9952 | 6.7358 | N/A | 5.1582 |
| trend_seasonal | 2.4406 | 1.8672 | 4.5143 | 0.9952 | -1.1186 |


### DynamicTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 2.9626 | 2.9571 | 3.0346 | 0.9994 | 26.4613 |
| intermittent | 0.0114 | 0.0115 | 0.0226 | -0.9886 | 10.1395 |
| long_memory | 0.1379 | 0.1341 | 0.2891 | -0.9901 | 25.0765 |
| multiplicative_seasonal | 2.7416 | 1.9757 | 5.2192 | 0.9990 | 36.5104 |
| noisy_seasonal | 0.0961 | 0.0953 | 0.1948 | -0.9858 | 44.8168 |
| seasonal | 0.0042 | 0.0030 | 0.0101 | 1.0000 | 36.8928 |
| seasonal_negative | 0.0003 | 0.0001 | 0.0010 | 1.0000 | -19.8838 |
| stationary | 0.0764 | 0.0759 | 0.1534 | 0.9862 | 33.8177 |
| structural_break | 1.2734 | 1.2637 | 2.0754 | 0.9668 | 32.2641 |
| trend | 2.5826 | 2.5868 | 3.9469 | -0.9881 | 13.0884 |
| trend_seasonal | 0.4998 | 0.3958 | 1.3141 | 0.9979 | 13.5591 |


### GARCH

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0057 | 0.0045 | 0.0120 | 1.0000 | -49.2904 |
| intermittent | 0.1674 | 0.1419 | 0.3390 | 1.0000 | -19.9937 |
| long_memory | 0.2673 | 0.2227 | 0.5384 | 1.0000 | -98.2051 |
| multiplicative_seasonal | 0.0001 | 0.0001 | 0.0003 | 1.0000 | -2122.3072 |
| noisy_seasonal | 0.0138 | 0.0111 | 0.0287 | 1.0000 | -253.3305 |
| seasonal | 0.0467 | 0.0387 | 0.0939 | 1.0000 | -193.1890 |
| seasonal_negative | 0.3577 | 0.3000 | 0.7220 | 1.0000 | -174.6458 |
| stationary | 3.8825 | 3.5039 | 8.3288 | 0.4634 | 0.2108 |
| structural_break | 0.0002 | 0.0000 | 0.0015 | 1.0000 | -337.0812 |
| trend | 0.0003 | 0.0002 | 0.0006 | 1.0000 | -925.3041 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0001 | 1.0000 | -430.6339 |


### HistoricAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### Holt

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0112 | 0.0112 | 0.0176 | 1.0000 | 4.7931 |
| intermittent | 0.1050 | 0.1050 | 0.1271 | 1.0000 | 15.5999 |
| long_memory | 0.0047 | 0.0047 | 0.0070 | 1.0000 | 27.2467 |
| multiplicative_seasonal | 0.0070 | 0.0070 | 0.0129 | 1.0000 | -2.6172 |
| noisy_seasonal | 0.0627 | 0.0627 | 0.0634 | 1.0000 | 44.4353 |
| seasonal | 0.2030 | 0.2030 | 0.3755 | 1.0000 | -1.4031 |
| seasonal_negative | 0.0378 | 0.0378 | 0.0697 | 1.0000 | -1.1607 |
| stationary | 1.3704 | 1.3704 | 1.8018 | 1.0000 | 21.0428 |
| structural_break | 0.0152 | 0.0152 | 0.0256 | 1.0000 | 18.5182 |
| trend | 0.0033 | 0.0033 | 0.0034 | 1.0000 | 16.0695 |
| trend_seasonal | 0.0035 | 0.0035 | 0.0065 | 1.0000 | -1.1993 |


### HoltWinters

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 1.4446 | 0.5444 | 4.7717 | 0.1291 | -7.1087 |
| intermittent | 1.0762 | 1.2098 | 1.8219 | 0.8906 | 0.9580 |
| long_memory | 1.5722 | 1.4759 | 2.9862 | 0.7126 | 2.0876 |
| multiplicative_seasonal | 4.3721 | 4.9084 | 7.5350 | 0.9966 | -3.6011 |
| noisy_seasonal | 1.5284 | 1.1797 | 3.7337 | 0.8986 | 3.8954 |
| seasonal | 0.5694 | 0.5031 | 1.5096 | 0.9965 | 1.1633 |
| seasonal_negative | 0.5933 | 0.5395 | 1.2169 | 0.9986 | 0.4349 |
| stationary | 1.7900 | 1.3539 | 4.0998 | 0.5531 | 1.7175 |
| structural_break | 1.0310 | 0.9732 | 2.6900 | 0.9129 | -5.0560 |
| trend | 0.7958 | 0.5524 | 1.9553 | 0.9151 | 2.8235 |
| trend_seasonal | 0.5714 | 0.2921 | 1.6695 | 0.9922 | 1.7740 |


### IMAPA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0030 | 0.0030 | 0.0030 | N/A | N/A |
| multiplicative_seasonal | 0.0003 | 0.0003 | 0.0003 | N/A | N/A |
| noisy_seasonal | 0.0008 | 0.0008 | 0.0008 | N/A | N/A |
| seasonal | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| seasonal_negative | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| stationary | 0.0001 | 0.0001 | 0.0001 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0001 | 0.0001 | 0.0001 | 1.0000 | N/A |


### MFLES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.9372 | 0.9426 | 0.9529 | 1.0000 | N/A |
| intermittent | 0.7978 | 0.8087 | 0.8087 | 0.9999 | N/A |
| long_memory | 0.1457 | 0.1459 | 0.1543 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0492 | 0.0469 | 0.0705 | 1.0000 | N/A |
| noisy_seasonal | 3.4233 | 3.4045 | 3.8421 | 1.0000 | N/A |
| seasonal | 0.9804 | 0.9751 | 1.1960 | 1.0000 | N/A |
| seasonal_negative | 0.5228 | 0.5217 | 0.5381 | 1.0000 | N/A |
| stationary | 1.4021 | 1.4001 | 1.4596 | 1.0000 | N/A |
| structural_break | 0.7670 | 0.7686 | 0.8104 | 1.0000 | N/A |
| trend | 1.0421 | 0.9979 | 1.2903 | 0.9999 | N/A |
| trend_seasonal | 1.2539 | 1.2652 | 1.6238 | 1.0000 | N/A |


### MSTLForecaster

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 2.2443 | 2.5045 | 3.5280 | 0.8470 | -20.1783 |
| intermittent | 0.2935 | 0.2612 | 0.9110 | 0.9878 | -10.2119 |
| long_memory | 0.3408 | 0.3398 | 0.6745 | 0.9900 | -20.6850 |
| multiplicative_seasonal | 1.6031 | 1.7312 | 2.4174 | 0.9999 | -17.6582 |
| noisy_seasonal | 1.7042 | 1.7804 | 2.5946 | 0.9909 | -28.3772 |
| seasonal | 0.1119 | 0.0908 | 0.3005 | 0.9998 | -7.3917 |
| seasonal_negative | 0.7645 | 0.7502 | 1.0136 | 0.9999 | -3.2723 |
| stationary | 1.4137 | 1.3221 | 2.0655 | 0.9758 | -12.5530 |
| structural_break | 0.1990 | 0.1653 | 0.4248 | 0.9949 | -23.0211 |
| trend | 0.1308 | 0.1311 | 0.3863 | 0.9977 | -11.1338 |
| trend_seasonal | 0.1842 | 0.1855 | 0.3828 | 0.9998 | -7.4345 |


### Naive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0054 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0088 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0143 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0184 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0249 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0099 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0082 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0111 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0110 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0082 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0085 |


### OptimizedTheta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.3499 | 0.3502 | 0.3793 | 0.9997 | -7.5647 |
| intermittent | 0.0762 | 0.0762 | 0.0794 | 1.0000 | 13.4816 |
| long_memory | 0.0403 | 0.0403 | 0.0444 | 1.0000 | 12.4464 |
| multiplicative_seasonal | 0.7042 | 0.5566 | 1.2410 | 1.0000 | 2.8594 |
| noisy_seasonal | 0.3577 | 0.3577 | 0.4956 | 0.0000 | 30.0325 |
| seasonal | 0.2990 | 0.2970 | 0.3587 | 1.0000 | 9.5094 |
| seasonal_negative | 0.2747 | 0.2747 | 0.2750 | 1.0000 | -19.3158 |
| stationary | 1.4509 | 1.4509 | 1.4686 | 1.0000 | 20.5508 |
| structural_break | 0.3110 | 0.3123 | 0.4066 | 0.9996 | -3.2545 |
| trend | 0.7631 | 0.7631 | 1.0388 | 1.0000 | 9.5374 |
| trend_seasonal | 0.5918 | 0.5284 | 0.8994 | 1.0000 | 6.4876 |


### RandomWalkWithDrift

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.9086 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.4802 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -2.4181 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -3.0998 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -4.1908 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.6636 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3743 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.8642 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.8602 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3644 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.4205 |


### SARIMA_1_1_1_1_1_1_12

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 3.2418 | 3.5341 | 6.8253 | 0.4191 | -8.6031 |
| intermittent | 1.1703 | 1.1577 | 2.0731 | 0.9511 | 3.3305 |
| long_memory | 1.9117 | 1.7267 | 3.0052 | 0.9734 | 3.6024 |
| multiplicative_seasonal | 0.0939 | 0.0940 | 0.1692 | 1.0000 | -5.4952 |
| noisy_seasonal | 1.9945 | 2.0675 | 3.0365 | 0.9930 | 12.0229 |
| seasonal | 0.1006 | 0.0494 | 0.3390 | 0.9998 | 0.9497 |
| seasonal_negative | 0.4088 | 0.3868 | 0.6212 | 0.9998 | 0.4476 |
| stationary | 1.0254 | 1.0930 | 1.6296 | 0.9798 | 2.0897 |
| structural_break | 0.5791 | 0.6121 | 0.9808 | 0.9822 | -2.6118 |
| trend | 0.6968 | 0.6882 | 1.4552 | 0.9284 | 2.8368 |
| trend_seasonal | 0.5946 | 0.5161 | 0.9668 | 0.9993 | 2.8257 |


### SES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### SeasonalES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### SeasonalNaive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0061 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0035 |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0064 |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0057 |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0089 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0026 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0011 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0043 |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0068 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0065 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0039 |


### SeasonalWindowAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### TBATS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 2.3410 | 2.5100 | 2.9179 | 0.7990 | -18.2349 |
| intermittent | 1.0864 | 1.0899 | 1.5538 | 0.9641 | 14.5175 |
| long_memory | 2.3184 | 2.3991 | 4.2652 | 0.3309 | -21.1075 |
| multiplicative_seasonal | 1.9459 | 1.5625 | 5.5616 | 0.9946 | -13.5194 |
| noisy_seasonal | 1.5061 | 1.8177 | 2.4737 | 0.8450 | -33.8609 |
| seasonal | 0.9909 | 0.9949 | 1.5273 | 0.9995 | -9.2698 |
| seasonal_negative | 2.0774 | 1.4332 | 6.2652 | 0.9649 | 13.1795 |
| stationary | 1.8845 | 2.1660 | 3.8305 | 0.8789 | -13.9044 |
| structural_break | 1.9530 | 1.7452 | 5.3605 | 0.4434 | -23.6466 |
| trend | 3.0169 | 2.6163 | 5.7963 | 0.9716 | -12.5514 |
| trend_seasonal | 2.2622 | 1.9320 | 5.1648 | 0.9928 | -14.5659 |


### TSB

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### Theta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 2.9398 | 2.9512 | 2.9908 | 1.0000 | 10.2114 |
| intermittent | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10.0557 |
| long_memory | 0.2464 | 0.2464 | 0.2464 | 1.0000 | 18.0757 |
| multiplicative_seasonal | 1.3547 | 1.3618 | 1.7455 | 1.0000 | 26.8412 |
| noisy_seasonal | 0.0003 | 0.0003 | 0.0003 | 1.0000 | 30.2732 |
| seasonal | 0.0001 | 0.0001 | 0.0001 | 1.0000 | 7.0238 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -21.0594 |
| stationary | 0.0001 | 0.0001 | 0.0001 | 1.0000 | 13.6964 |
| structural_break | 1.9598 | 1.9695 | 2.0243 | 1.0000 | 16.8045 |
| trend | 1.1805 | 1.1805 | 1.1805 | 1.0000 | 25.9529 |
| trend_seasonal | 1.0021 | 1.0117 | 1.2521 | 1.0000 | 11.9903 |


### WindowAverage

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| high_frequency | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| intermittent | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| long_memory | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| noisy_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| structural_break | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


---

## Confidence Interval Comparison

Mean CI width differences (Rust - statsforecast) by level:

| Model | Series | 80% | 90% | 95% |
|-------|--------|-----|-----|-----|
| ADIDA | high_frequency | N/A | N/A | N/A |
| ARIMA_1_1_1 | high_frequency | 3.1851 | 4.0895 | 4.8730 |
| AutoARIMA | high_frequency | 0.3747 | 0.4816 | 0.5739 |
| AutoETS | high_frequency | 3.2765 | 4.2069 | 5.0129 |
| AutoTBATS | high_frequency | -7.9307 | -10.1789 | -12.1289 |
| AutoTheta | high_frequency | -5.8290 | -6.8696 | -7.5647 |
| Croston | high_frequency | N/A | N/A | N/A |
| CrostonSBA | high_frequency | N/A | N/A | N/A |
| DynamicOptimizedTheta | high_frequency | -6.0688 | -7.1848 | -7.9327 |
| DynamicTheta | high_frequency | 16.4027 | 21.6740 | 26.4613 |
| GARCH | high_frequency | -32.2301 | -41.3659 | -49.2904 |
| HistoricAverage | high_frequency | N/A | N/A | N/A |
| Holt | high_frequency | 3.1328 | 4.0225 | 4.7931 |
| HoltWinters | high_frequency | -4.6487 | -5.9658 | -7.1087 |
| IMAPA | high_frequency | N/A | N/A | N/A |
| MFLES | high_frequency | N/A | N/A | N/A |
| MSTLForecaster | high_frequency | -13.1939 | -16.9342 | -20.1783 |
| Naive | high_frequency | 0.0022 | 0.0045 | 0.0054 |
| OptimizedTheta | high_frequency | -5.8290 | -6.8696 | -7.5647 |
| RandomWalkWithDrift | high_frequency | -0.5954 | -0.7626 | -0.9086 |
| SARIMA_1_1_1_1_1_1_12 | high_frequency | -5.6260 | -7.2200 | -8.6031 |
| SES | high_frequency | N/A | N/A | N/A |
| SeasonalES | high_frequency | N/A | N/A | N/A |
| SeasonalNaive | high_frequency | 0.0025 | 0.0051 | 0.0061 |
| SeasonalWindowAverage | high_frequency | N/A | N/A | N/A |
| TBATS | high_frequency | -11.9225 | -15.3028 | -18.2349 |
| TSB | high_frequency | N/A | N/A | N/A |
| Theta | high_frequency | 5.7991 | 8.0491 | 10.2114 |
| WindowAverage | high_frequency | N/A | N/A | N/A |
| ADIDA | intermittent | N/A | N/A | N/A |
| ARIMA_1_1_1 | intermittent | 10.9521 | 14.0588 | 16.7521 |
| AutoARIMA | intermittent | 1.1670 | 1.4987 | 1.7859 |
| AutoETS | intermittent | 10.2826 | 13.1994 | 15.7281 |
| AutoTBATS | intermittent | 9.7897 | 12.5667 | 14.9742 |
| AutoTheta | intermittent | 8.6182 | 10.9449 | 13.4816 |
| Croston | intermittent | N/A | N/A | N/A |
| CrostonSBA | intermittent | N/A | N/A | N/A |
| DynamicOptimizedTheta | intermittent | 8.6732 | 11.0102 | 13.5709 |
| DynamicTheta | intermittent | 6.4271 | 8.1363 | 10.1395 |
| GARCH | intermittent | -13.0738 | -16.7793 | -19.9937 |
| HistoricAverage | intermittent | N/A | N/A | N/A |
| Holt | intermittent | 10.1988 | 13.0918 | 15.5999 |
| HoltWinters | intermittent | 0.6257 | 0.8040 | 0.9580 |
| IMAPA | intermittent | N/A | N/A | N/A |
| MFLES | intermittent | N/A | N/A | N/A |
| MSTLForecaster | intermittent | -6.6772 | -8.5701 | -10.2119 |
| Naive | intermittent | 0.0036 | 0.0073 | 0.0088 |
| OptimizedTheta | intermittent | 8.6182 | 10.9449 | 13.4816 |
| RandomWalkWithDrift | intermittent | -0.9700 | -1.2423 | -1.4802 |
| SARIMA_1_1_1_1_1_1_12 | intermittent | 2.1768 | 2.7950 | 3.3305 |
| SES | intermittent | N/A | N/A | N/A |
| SeasonalES | intermittent | N/A | N/A | N/A |
| SeasonalNaive | intermittent | 0.0014 | 0.0029 | 0.0035 |
| SeasonalWindowAverage | intermittent | N/A | N/A | N/A |
| TBATS | intermittent | 9.4911 | 12.1834 | 14.5175 |
| TSB | intermittent | N/A | N/A | N/A |
| Theta | intermittent | 6.3751 | 8.0711 | 10.0557 |
| WindowAverage | intermittent | N/A | N/A | N/A |
| ADIDA | long_memory | N/A | N/A | N/A |
| ARIMA_1_1_1 | long_memory | 18.0166 | 23.1274 | 27.5581 |
| AutoARIMA | long_memory | -0.4872 | -0.6238 | -0.7432 |
| AutoETS | long_memory | 18.2817 | 23.4677 | 27.9636 |
| AutoTBATS | long_memory | -13.4652 | -17.2824 | -20.5932 |
| AutoTheta | long_memory | 7.6772 | 10.0044 | 12.4631 |
| Croston | long_memory | N/A | N/A | N/A |
| CrostonSBA | long_memory | N/A | N/A | N/A |
| DynamicOptimizedTheta | long_memory | 7.5709 | 9.8875 | 12.2416 |
| DynamicTheta | long_memory | 15.9462 | 20.6454 | 25.0765 |
| GARCH | long_memory | -64.2139 | -82.4163 | -98.2051 |
| HistoricAverage | long_memory | N/A | N/A | N/A |
| Holt | long_memory | 17.8130 | 22.8660 | 27.2467 |
| HoltWinters | long_memory | 1.3637 | 1.7519 | 2.0876 |
| IMAPA | long_memory | N/A | N/A | N/A |
| MFLES | long_memory | N/A | N/A | N/A |
| MSTLForecaster | long_memory | -13.5252 | -17.3594 | -20.6850 |
| Naive | long_memory | 0.0059 | 0.0119 | 0.0143 |
| OptimizedTheta | long_memory | 7.6663 | 9.9904 | 12.4464 |
| RandomWalkWithDrift | long_memory | -1.5846 | -2.0295 | -2.4181 |
| SARIMA_1_1_1_1_1_1_12 | long_memory | 2.3539 | 3.0232 | 3.6024 |
| SES | long_memory | N/A | N/A | N/A |
| SeasonalES | long_memory | N/A | N/A | N/A |
| SeasonalNaive | long_memory | 0.0026 | 0.0053 | 0.0064 |
| SeasonalWindowAverage | long_memory | N/A | N/A | N/A |
| TBATS | long_memory | -13.8015 | -17.7140 | -21.1075 |
| TSB | long_memory | N/A | N/A | N/A |
| Theta | long_memory | 11.3585 | 14.7490 | 18.0757 |
| WindowAverage | long_memory | N/A | N/A | N/A |
| ADIDA | multiplicative_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | multiplicative_seasonal | -61.0804 | -78.3924 | -93.4102 |
| AutoARIMA | multiplicative_seasonal | 1.0172 | 1.3063 | 1.5566 |
| AutoETS | multiplicative_seasonal | -0.8785 | -1.1269 | -1.3428 |
| AutoTBATS | multiplicative_seasonal | -7.6202 | -9.7822 | -11.6584 |
| AutoTheta | multiplicative_seasonal | 1.7679 | 2.2086 | 2.8594 |
| Croston | multiplicative_seasonal | N/A | N/A | N/A |
| CrostonSBA | multiplicative_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | multiplicative_seasonal | 1.1557 | 1.4046 | 1.9153 |
| DynamicTheta | multiplicative_seasonal | 23.5673 | 30.4264 | 36.5104 |
| GARCH | multiplicative_seasonal | -1387.7070 | -1781.0965 | -2122.3072 |
| HistoricAverage | multiplicative_seasonal | N/A | N/A | N/A |
| Holt | multiplicative_seasonal | -1.7158 | -2.1966 | -2.6172 |
| HoltWinters | multiplicative_seasonal | -2.3553 | -3.0222 | -3.6011 |
| IMAPA | multiplicative_seasonal | N/A | N/A | N/A |
| MFLES | multiplicative_seasonal | N/A | N/A | N/A |
| MSTLForecaster | multiplicative_seasonal | -11.5461 | -14.8192 | -17.6582 |
| Naive | multiplicative_seasonal | 0.0076 | 0.0153 | 0.0184 |
| OptimizedTheta | multiplicative_seasonal | 1.7679 | 2.2086 | 2.8594 |
| RandomWalkWithDrift | multiplicative_seasonal | -2.0313 | -2.6016 | -3.0998 |
| SARIMA_1_1_1_1_1_1_12 | multiplicative_seasonal | -3.5938 | -4.6117 | -5.4952 |
| SES | multiplicative_seasonal | N/A | N/A | N/A |
| SeasonalES | multiplicative_seasonal | N/A | N/A | N/A |
| SeasonalNaive | multiplicative_seasonal | 0.0023 | 0.0047 | 0.0057 |
| SeasonalWindowAverage | multiplicative_seasonal | N/A | N/A | N/A |
| TBATS | multiplicative_seasonal | -8.8356 | -11.3430 | -13.5194 |
| TSB | multiplicative_seasonal | N/A | N/A | N/A |
| Theta | multiplicative_seasonal | 17.2450 | 22.3275 | 26.8412 |
| WindowAverage | multiplicative_seasonal | N/A | N/A | N/A |
| ADIDA | noisy_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | noisy_seasonal | 31.5085 | 40.4461 | 48.1947 |
| AutoARIMA | noisy_seasonal | 1.8396 | 2.3634 | 2.8163 |
| AutoETS | noisy_seasonal | 29.1939 | 37.4751 | 44.6545 |
| AutoTBATS | noisy_seasonal | -18.5702 | -23.8344 | -28.4005 |
| AutoTheta | noisy_seasonal | 18.8259 | 23.8136 | 29.6325 |
| Croston | noisy_seasonal | N/A | N/A | N/A |
| CrostonSBA | noisy_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | noisy_seasonal | 18.9048 | 23.9196 | 29.7767 |
| DynamicTheta | noisy_seasonal | 28.7304 | 36.5574 | 44.8168 |
| GARCH | noisy_seasonal | -165.6457 | -212.6017 | -253.3305 |
| HistoricAverage | noisy_seasonal | N/A | N/A | N/A |
| Holt | noisy_seasonal | 29.0506 | 37.2911 | 44.4353 |
| HoltWinters | noisy_seasonal | 2.5452 | 3.2691 | 3.8954 |
| IMAPA | noisy_seasonal | N/A | N/A | N/A |
| MFLES | noisy_seasonal | N/A | N/A | N/A |
| MSTLForecaster | noisy_seasonal | -18.5549 | -23.8149 | -28.3772 |
| Naive | noisy_seasonal | 0.0102 | 0.0206 | 0.0249 |
| OptimizedTheta | noisy_seasonal | 19.0874 | 24.1493 | 30.0325 |
| RandomWalkWithDrift | noisy_seasonal | -2.7462 | -3.5172 | -4.1908 |
| SARIMA_1_1_1_1_1_1_12 | noisy_seasonal | 7.8585 | 10.0899 | 12.0229 |
| SES | noisy_seasonal | N/A | N/A | N/A |
| SeasonalES | noisy_seasonal | N/A | N/A | N/A |
| SeasonalNaive | noisy_seasonal | 0.0037 | 0.0074 | 0.0089 |
| SeasonalWindowAverage | noisy_seasonal | N/A | N/A | N/A |
| TBATS | noisy_seasonal | -21.9814 | -28.3114 | -33.8609 |
| TSB | noisy_seasonal | N/A | N/A | N/A |
| Theta | noisy_seasonal | 19.2299 | 24.3673 | 30.2732 |
| WindowAverage | noisy_seasonal | N/A | N/A | N/A |
| ADIDA | seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | seasonal | -15.2387 | -19.5560 | -23.3023 |
| AutoARIMA | seasonal | 1.5218 | 1.9539 | 2.3282 |
| AutoETS | seasonal | 0.7404 | 0.9510 | 1.1332 |
| AutoTBATS | seasonal | -4.7882 | -6.1455 | -7.3228 |
| AutoTheta | seasonal | 6.0826 | 7.7341 | 9.5103 |
| Croston | seasonal | N/A | N/A | N/A |
| CrostonSBA | seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | seasonal | 6.1147 | 7.7678 | 9.5584 |
| DynamicTheta | seasonal | 23.9833 | 30.7099 | 36.8928 |
| GARCH | seasonal | -126.3211 | -162.1294 | -193.1890 |
| HistoricAverage | seasonal | N/A | N/A | N/A |
| Holt | seasonal | -0.9198 | -1.1776 | -1.4031 |
| HoltWinters | seasonal | 0.7601 | 0.9763 | 1.1633 |
| IMAPA | seasonal | N/A | N/A | N/A |
| MFLES | seasonal | N/A | N/A | N/A |
| MSTLForecaster | seasonal | -4.8332 | -6.2033 | -7.3917 |
| Naive | seasonal | 0.0041 | 0.0082 | 0.0099 |
| OptimizedTheta | seasonal | 6.0819 | 7.7332 | 9.5094 |
| RandomWalkWithDrift | seasonal | -1.0902 | -1.3963 | -1.6636 |
| SARIMA_1_1_1_1_1_1_12 | seasonal | 0.6203 | 0.7970 | 0.9497 |
| SES | seasonal | N/A | N/A | N/A |
| SeasonalES | seasonal | N/A | N/A | N/A |
| SeasonalNaive | seasonal | 0.0011 | 0.0021 | 0.0026 |
| SeasonalWindowAverage | seasonal | N/A | N/A | N/A |
| TBATS | seasonal | -6.0548 | -7.7752 | -9.2698 |
| TSB | seasonal | N/A | N/A | N/A |
| Theta | seasonal | 4.4569 | 5.6474 | 7.0238 |
| WindowAverage | seasonal | N/A | N/A | N/A |
| ADIDA | seasonal_negative | N/A | N/A | N/A |
| ARIMA_1_1_1 | seasonal_negative | -24.7996 | -31.8282 | -37.9256 |
| AutoARIMA | seasonal_negative | 0.4624 | 0.5938 | 0.7076 |
| AutoETS | seasonal_negative | 0.2021 | 0.2596 | 0.3094 |
| AutoTBATS | seasonal_negative | 9.2602 | 11.8865 | 14.1637 |
| AutoTheta | seasonal_negative | -13.1244 | -17.1158 | -19.3393 |
| Croston | seasonal_negative | N/A | N/A | N/A |
| CrostonSBA | seasonal_negative | N/A | N/A | N/A |
| DynamicOptimizedTheta | seasonal_negative | -13.0769 | -17.0807 | -19.2661 |
| DynamicTheta | seasonal_negative | -13.4792 | -17.5705 | -19.8838 |
| GARCH | seasonal_negative | -114.1962 | -146.5674 | -174.6458 |
| HistoricAverage | seasonal_negative | N/A | N/A | N/A |
| Holt | seasonal_negative | -0.7609 | -0.9742 | -1.1607 |
| HoltWinters | seasonal_negative | 0.2841 | 0.3650 | 0.4349 |
| IMAPA | seasonal_negative | N/A | N/A | N/A |
| MFLES | seasonal_negative | N/A | N/A | N/A |
| MSTLForecaster | seasonal_negative | -2.1397 | -2.7462 | -3.2723 |
| Naive | seasonal_negative | 0.0034 | 0.0068 | 0.0082 |
| OptimizedTheta | seasonal_negative | -13.1065 | -17.1033 | -19.3158 |
| RandomWalkWithDrift | seasonal_negative | -0.9006 | -1.1534 | -1.3743 |
| SARIMA_1_1_1_1_1_1_12 | seasonal_negative | 0.2924 | 0.3756 | 0.4476 |
| SES | seasonal_negative | N/A | N/A | N/A |
| SeasonalES | seasonal_negative | N/A | N/A | N/A |
| SeasonalNaive | seasonal_negative | 0.0004 | 0.0009 | 0.0011 |
| SeasonalWindowAverage | seasonal_negative | N/A | N/A | N/A |
| TBATS | seasonal_negative | 8.6166 | 11.0605 | 13.1795 |
| TSB | seasonal_negative | N/A | N/A | N/A |
| Theta | seasonal_negative | -14.2490 | -18.5594 | -21.0594 |
| WindowAverage | seasonal_negative | N/A | N/A | N/A |
| ADIDA | stationary | N/A | N/A | N/A |
| ARIMA_1_1_1 | stationary | 14.8095 | 19.0104 | 22.6524 |
| AutoARIMA | stationary | 1.6401 | 2.1062 | 2.5098 |
| AutoETS | stationary | 13.7311 | 17.6261 | 21.0028 |
| AutoTBATS | stationary | -9.3092 | -11.9481 | -14.2371 |
| AutoTheta | stationary | 13.1772 | 16.7614 | 20.5508 |
| Croston | stationary | N/A | N/A | N/A |
| CrostonSBA | stationary | N/A | N/A | N/A |
| DynamicOptimizedTheta | stationary | 11.2855 | 14.3244 | 17.6630 |
| DynamicTheta | stationary | 21.8406 | 27.8829 | 33.8177 |
| GARCH | stationary | 0.1370 | 0.1769 | 0.2108 |
| HistoricAverage | stationary | N/A | N/A | N/A |
| Holt | stationary | 13.7572 | 17.6596 | 21.0428 |
| HoltWinters | stationary | 1.1221 | 1.4414 | 1.7175 |
| IMAPA | stationary | N/A | N/A | N/A |
| MFLES | stationary | N/A | N/A | N/A |
| MSTLForecaster | stationary | -8.2080 | -10.5348 | -12.5530 |
| Naive | stationary | 0.0045 | 0.0092 | 0.0111 |
| OptimizedTheta | stationary | 13.1772 | 16.7614 | 20.5508 |
| RandomWalkWithDrift | stationary | -1.2216 | -1.5646 | -1.8642 |
| SARIMA_1_1_1_1_1_1_12 | stationary | 1.3652 | 1.7537 | 2.0897 |
| SES | stationary | N/A | N/A | N/A |
| SeasonalES | stationary | N/A | N/A | N/A |
| SeasonalNaive | stationary | 0.0018 | 0.0036 | 0.0043 |
| SeasonalWindowAverage | stationary | N/A | N/A | N/A |
| TBATS | stationary | -9.0689 | -11.6538 | -13.9044 |
| TSB | stationary | N/A | N/A | N/A |
| Theta | stationary | 8.6923 | 11.0101 | 13.6964 |
| WindowAverage | stationary | N/A | N/A | N/A |
| ADIDA | structural_break | N/A | N/A | N/A |
| ARIMA_1_1_1 | structural_break | 12.3095 | 15.8020 | 18.8293 |
| AutoARIMA | structural_break | 0.9136 | 1.1744 | 1.3994 |
| AutoETS | structural_break | 12.0308 | 15.4442 | 18.4030 |
| AutoTBATS | structural_break | -15.2571 | -19.5823 | -23.3337 |
| AutoTheta | structural_break | -2.8387 | -3.1937 | -3.2545 |
| Croston | structural_break | N/A | N/A | N/A |
| CrostonSBA | structural_break | N/A | N/A | N/A |
| DynamicOptimizedTheta | structural_break | -2.8087 | -3.1216 | -3.1711 |
| DynamicTheta | structural_break | 20.2916 | 26.5781 | 32.2641 |
| GARCH | structural_break | -220.4076 | -282.8875 | -337.0812 |
| HistoricAverage | structural_break | N/A | N/A | N/A |
| Holt | structural_break | 12.1061 | 15.5409 | 18.5182 |
| HoltWinters | structural_break | -3.3071 | -4.2432 | -5.0560 |
| IMAPA | structural_break | N/A | N/A | N/A |
| MFLES | structural_break | N/A | N/A | N/A |
| MSTLForecaster | structural_break | -15.0527 | -19.3199 | -23.0211 |
| Naive | structural_break | 0.0045 | 0.0092 | 0.0110 |
| OptimizedTheta | structural_break | -2.8387 | -3.1937 | -3.2545 |
| RandomWalkWithDrift | structural_break | -1.2190 | -1.5612 | -1.8602 |
| SARIMA_1_1_1_1_1_1_12 | structural_break | -1.7092 | -2.1920 | -2.6118 |
| SES | structural_break | N/A | N/A | N/A |
| SeasonalES | structural_break | N/A | N/A | N/A |
| SeasonalNaive | structural_break | 0.0028 | 0.0056 | 0.0068 |
| SeasonalWindowAverage | structural_break | N/A | N/A | N/A |
| TBATS | structural_break | -15.4617 | -19.8448 | -23.6466 |
| TSB | structural_break | N/A | N/A | N/A |
| Theta | structural_break | 10.2763 | 13.6473 | 16.8045 |
| WindowAverage | structural_break | N/A | N/A | N/A |
| ADIDA | trend | N/A | N/A | N/A |
| ARIMA_1_1_1 | trend | 6.6459 | 8.5320 | 10.1665 |
| AutoARIMA | trend | 1.3910 | 1.7863 | 2.1285 |
| AutoETS | trend | 10.5058 | 13.4859 | 16.0695 |
| AutoTBATS | trend | -7.2924 | -9.3596 | -11.1526 |
| AutoTheta | trend | 6.0375 | 7.6223 | 9.5374 |
| Croston | trend | N/A | N/A | N/A |
| CrostonSBA | trend | N/A | N/A | N/A |
| DynamicOptimizedTheta | trend | 3.1664 | 3.9327 | 5.1582 |
| DynamicTheta | trend | 8.1078 | 10.7013 | 13.0884 |
| GARCH | trend | -605.0271 | -776.5398 | -925.3041 |
| HistoricAverage | trend | N/A | N/A | N/A |
| Holt | trend | 10.5058 | 13.4859 | 16.0695 |
| HoltWinters | trend | 1.8453 | 2.3695 | 2.8235 |
| IMAPA | trend | N/A | N/A | N/A |
| MFLES | trend | N/A | N/A | N/A |
| MSTLForecaster | trend | -7.2800 | -9.3437 | -11.1338 |
| Naive | trend | 0.0034 | 0.0068 | 0.0082 |
| OptimizedTheta | trend | 6.0375 | 7.6223 | 9.5374 |
| RandomWalkWithDrift | trend | -0.8941 | -1.1451 | -1.3644 |
| SARIMA_1_1_1_1_1_1_12 | trend | 1.8539 | 2.3807 | 2.8368 |
| SES | trend | N/A | N/A | N/A |
| SeasonalES | trend | N/A | N/A | N/A |
| SeasonalNaive | trend | 0.0027 | 0.0054 | 0.0065 |
| SeasonalWindowAverage | trend | N/A | N/A | N/A |
| TBATS | trend | -8.2072 | -10.5336 | -12.5514 |
| TSB | trend | N/A | N/A | N/A |
| Theta | trend | 16.5465 | 21.5197 | 25.9529 |
| WindowAverage | trend | N/A | N/A | N/A |
| ADIDA | trend_seasonal | N/A | N/A | N/A |
| ARIMA_1_1_1 | trend_seasonal | -7.3160 | -9.3875 | -11.1858 |
| AutoARIMA | trend_seasonal | 0.6344 | 0.8149 | 0.9710 |
| AutoETS | trend_seasonal | 1.1594 | 1.4888 | 1.7740 |
| AutoTBATS | trend_seasonal | -5.5506 | -7.1240 | -8.4888 |
| AutoTheta | trend_seasonal | 4.0896 | 5.1555 | 6.4876 |
| Croston | trend_seasonal | N/A | N/A | N/A |
| CrostonSBA | trend_seasonal | N/A | N/A | N/A |
| DynamicOptimizedTheta | trend_seasonal | -1.1660 | -1.3068 | -1.1186 |
| DynamicTheta | trend_seasonal | 8.3803 | 11.0797 | 13.5591 |
| GARCH | trend_seasonal | -281.5787 | -361.3994 | -430.6339 |
| HistoricAverage | trend_seasonal | N/A | N/A | N/A |
| Holt | trend_seasonal | -0.7862 | -1.0065 | -1.1993 |
| HoltWinters | trend_seasonal | 1.1594 | 1.4888 | 1.7740 |
| IMAPA | trend_seasonal | N/A | N/A | N/A |
| MFLES | trend_seasonal | N/A | N/A | N/A |
| MSTLForecaster | trend_seasonal | -4.8611 | -6.2392 | -7.4345 |
| Naive | trend_seasonal | 0.0035 | 0.0070 | 0.0085 |
| OptimizedTheta | trend_seasonal | 4.0896 | 5.1555 | 6.4876 |
| RandomWalkWithDrift | trend_seasonal | -0.9309 | -1.1922 | -1.4205 |
| SARIMA_1_1_1_1_1_1_12 | trend_seasonal | 1.8470 | 2.3714 | 2.8257 |
| SES | trend_seasonal | N/A | N/A | N/A |
| SeasonalES | trend_seasonal | N/A | N/A | N/A |
| SeasonalNaive | trend_seasonal | 0.0016 | 0.0032 | 0.0039 |
| SeasonalWindowAverage | trend_seasonal | N/A | N/A | N/A |
| TBATS | trend_seasonal | -9.5034 | -12.2103 | -14.5659 |
| TSB | trend_seasonal | N/A | N/A | N/A |
| Theta | trend_seasonal | 7.4019 | 9.7761 | 11.9903 |
| WindowAverage | trend_seasonal | N/A | N/A | N/A |


---

## Detailed Point Forecast Differences

Largest absolute differences:

| Model | Series | Step | Rust | statsforecast | Difference |
|-------|--------|------|------|---------------|------------|
| AutoARIMA | structural_break | 10 | 53.8888 | 68.7003 | -14.8115 |
| AutoTBATS | multiplicative_seasonal | 12 | 124.7621 | 139.0763 | -14.3142 |
| AutoARIMA | structural_break | 12 | 55.4951 | 68.7003 | -13.2052 |
| AutoARIMA | structural_break | 11 | 56.3395 | 68.7003 | -12.3608 |
| AutoTBATS | multiplicative_seasonal | 11 | 122.1259 | 132.7486 | -10.6227 |
| AutoARIMA | structural_break | 9 | 59.8376 | 68.7003 | -8.8627 |
| DynamicOptimizedTheta | multiplicative_seasonal | 12 | 126.1968 | 134.9744 | -8.7776 |
| GARCH | stationary | 6 | -8.6553 | -0.3264 | -8.3288 |
| DynamicOptimizedTheta | multiplicative_seasonal | 11 | 122.7940 | 130.7389 | -7.9450 |
| HoltWinters | multiplicative_seasonal | 11 | 131.7695 | 124.2345 | 7.5350 |

---

## Metrics by Forecast Horizon Step

Aggregated metrics across all models and series types by forecast step:

| Step | MAD | Median | Max Diff | Mean Diff | Std |
|------|-----|--------|----------|-----------|-----|
| 1 | 0.4933 | 0.0030 | 6.4724 | 0.0214 | 1.0864 |
| 2 | 0.4583 | 0.0033 | 5.8415 | -0.0006 | 0.9914 |
| 3 | 0.4610 | 0.0033 | 6.8253 | -0.0447 | 0.9689 |
| 4 | 0.4305 | 0.0049 | 4.6066 | -0.0031 | 0.8802 |
| 5 | 0.4643 | 0.0053 | 4.6845 | 0.0210 | 0.9660 |
| 6 | 0.5491 | 0.0064 | 8.3288 | -0.0354 | 1.2507 |
| 7 | 0.5284 | 0.0075 | 6.3033 | -0.0338 | 1.1648 |
| 8 | 0.5189 | 0.0043 | 6.4490 | -0.0473 | 1.1386 |
| 9 | 0.5537 | 0.0046 | 8.8627 | -0.1048 | 1.2340 |
| 10 | 0.6213 | 0.0054 | 14.8115 | -0.0872 | 1.5210 |
| 11 | 0.7326 | 0.0088 | 12.3608 | -0.0694 | 1.7160 |
| 12 | 0.8004 | 0.0113 | 14.3142 | -0.1281 | 1.9167 |

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
