# Forecast Validation Report

Generated: 2025-12-11 21:59:39

## Summary

- **Rust implementation**: anofox-forecast
- **Python implementation**: statsforecast (NIXTLA)
- **Forecast horizon**: 12 steps
- **Confidence levels**: 80%, 90%, 95%

- **Models compared**: 13
- **Series types**: 6
- **Total comparisons**: 78

- **High agreement (corr >= 0.99)**: 39 combinations
- **Lower agreement (corr < 0.95)**: 21 combinations

---

## Results by Model

### ARIMA_1_1_1

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 1.8412 | 1.7507 | 3.9119 | 0.4620 | -93.4102 |
| seasonal | 0.0286 | 0.0282 | 0.0560 | 0.9999 | -23.3023 |
| seasonal_negative | 0.2886 | 0.2946 | 0.5394 | 0.9949 | -37.9256 |
| stationary | 0.0428 | 0.0429 | 0.0598 | 0.5039 | 22.6524 |
| trend | 4.0219 | 4.0286 | 6.6558 | -0.5139 | 10.1665 |
| trend_seasonal | 2.1003 | 2.0886 | 4.0188 | 0.7108 | -11.1858 |


### AutoARIMA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 11.2087 | 11.4473 | 20.7239 | 0.8323 | 31.3154 |
| seasonal | 2.5937 | 2.2150 | 6.3064 | 0.9102 | 22.7468 |
| seasonal_negative | 12.5027 | 12.5141 | 22.4959 | -0.0987 | 19.4343 |
| stationary | 0.3131 | 0.2672 | 0.7297 | 0.0000 | 18.2932 |
| trend | 0.7026 | 0.7525 | 1.3890 | 0.9642 | 13.2213 |
| trend_seasonal | 2.9901 | 2.2626 | 7.1445 | 0.8766 | 19.5622 |


### AutoETS

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.9760 | 0.8461 | 2.0603 | 0.9989 | -1.3428 |
| seasonal | 0.4864 | 0.4179 | 1.2734 | 0.9967 | 1.1332 |
| seasonal_negative | 0.8652 | 0.7814 | 1.6762 | 0.9988 | 0.3094 |
| stationary | 1.9889 | 1.1330 | 5.4030 | N/A | 1.9897 |
| trend | 1.0121 | 0.8969 | 2.0424 | 0.8277 | 3.3976 |
| trend_seasonal | 0.5714 | 0.2921 | 1.6695 | 0.9922 | 1.7740 |


### Croston

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| seasonal_negative | 2.1709 | 2.1709 | 2.1709 | 1.0000 | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |


### CrostonSBA

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| seasonal_negative | 2.0623 | 2.0623 | 2.0623 | N/A | N/A |
| stationary | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |


### Holt

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 8.4923 | 8.4923 | 15.6789 | -1.0000 | 180.4397 |
| seasonal | 26.7723 | 26.7723 | 49.4250 | 1.0000 | 106.9294 |
| seasonal_negative | 11.3203 | 11.3203 | 20.8989 | 1.0000 | 93.7986 |
| stationary | 0.9095 | 0.8433 | 2.0259 | -1.0000 | 36.4958 |
| trend | 1.3967 | 1.3967 | 1.8974 | 1.0000 | 29.3627 |
| trend_seasonal | 1.7824 | 1.7824 | 3.2906 | 1.0000 | 7.1100 |


### HoltWinters

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 4.3721 | 4.9084 | 7.5350 | 0.9966 | -3.6011 |
| seasonal | 0.5694 | 0.5031 | 1.5096 | 0.9965 | 1.1633 |
| seasonal_negative | 0.5933 | 0.5395 | 1.2169 | 0.9986 | 0.4349 |
| stationary | 1.7900 | 1.3539 | 4.0998 | 0.5531 | 1.7175 |
| trend | 0.7958 | 0.5524 | 1.9553 | 0.9151 | 2.8235 |
| trend_seasonal | 0.5714 | 0.2921 | 1.6695 | 0.9922 | 1.7740 |


### Naive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0184 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0099 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0082 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0111 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0082 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0085 |


### RandomWalkWithDrift

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -3.0998 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.6636 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3743 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.8642 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.3644 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -1.4205 |


### SES

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 27.5914 | 27.5914 | 27.5914 | N/A | N/A |
| seasonal | 10.2026 | 10.2026 | 10.2026 | -1.0000 | N/A |
| seasonal_negative | 7.9560 | 7.9560 | 7.9560 | N/A | N/A |
| stationary | 0.9208 | 0.9208 | 0.9208 | N/A | N/A |
| trend | 3.4043 | 3.4043 | 3.4043 | N/A | N/A |
| trend_seasonal | 8.6749 | 8.6749 | 8.6749 | 1.0000 | N/A |


### SeasonalNaive

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0057 |
| seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0026 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0011 |
| stationary | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0043 |
| trend | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0065 |
| trend_seasonal | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0039 |


### TSB

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 39.0642 | 42.6603 | 66.6753 | N/A | N/A |
| seasonal | 20.2837 | 22.1509 | 34.6204 | 0.0000 | N/A |
| seasonal_negative | 4.5248 | 4.7636 | 6.3584 | -0.0000 | N/A |
| stationary | 19.2091 | 20.9774 | 32.7863 | N/A | N/A |
| trend | 22.1783 | 24.2199 | 37.8541 | 0.0000 | N/A |
| trend_seasonal | 19.1431 | 20.9054 | 32.6737 | 0.0000 | N/A |


### Theta

| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |
|-------------|-----|--------|----------|-------------|---------------------|
| multiplicative_seasonal | 1.3547 | 1.3618 | 1.7455 | 1.0000 | 26.8412 |
| seasonal | 0.0001 | 0.0001 | 0.0001 | 1.0000 | 7.0238 |
| seasonal_negative | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -21.0594 |
| stationary | 0.8360 | 0.6574 | 2.7502 | 0.1872 | 13.8244 |
| trend | 1.8274 | 2.0826 | 3.0086 | 0.6713 | 26.0523 |
| trend_seasonal | 1.0021 | 1.0117 | 1.2521 | 1.0000 | 11.9903 |


---

## Confidence Interval Comparison

Mean CI width differences (Rust - statsforecast) by level:

| Model | Series | 80% | 90% | 95% |
|-------|--------|-----|-----|-----|
| ARIMA_1_1_1 | multiplicative_seasonal | -61.0804 | -78.3924 | -93.4102 |
| AutoARIMA | multiplicative_seasonal | 20.4738 | 26.2806 | 31.3154 |
| AutoETS | multiplicative_seasonal | -0.8785 | -1.1269 | -1.3428 |
| Croston | multiplicative_seasonal | N/A | N/A | N/A |
| CrostonSBA | multiplicative_seasonal | N/A | N/A | N/A |
| Holt | multiplicative_seasonal | 117.9690 | 151.4292 | 180.4397 |
| HoltWinters | multiplicative_seasonal | -2.3553 | -3.0222 | -3.6011 |
| Naive | multiplicative_seasonal | 0.0076 | 0.0153 | 0.0184 |
| RandomWalkWithDrift | multiplicative_seasonal | -2.0313 | -2.6016 | -3.0998 |
| SES | multiplicative_seasonal | N/A | N/A | N/A |
| SeasonalNaive | multiplicative_seasonal | 0.0023 | 0.0047 | 0.0057 |
| TSB | multiplicative_seasonal | N/A | N/A | N/A |
| Theta | multiplicative_seasonal | 17.2450 | 22.3275 | 26.8412 |
| ARIMA_1_1_1 | seasonal | -15.2387 | -19.5560 | -23.3023 |
| AutoARIMA | seasonal | 14.8716 | 19.0897 | 22.7468 |
| AutoETS | seasonal | 0.7404 | 0.9510 | 1.1332 |
| Croston | seasonal | N/A | N/A | N/A |
| CrostonSBA | seasonal | N/A | N/A | N/A |
| Holt | seasonal | 69.9092 | 89.7377 | 106.9294 |
| HoltWinters | seasonal | 0.7601 | 0.9763 | 1.1633 |
| Naive | seasonal | 0.0041 | 0.0082 | 0.0099 |
| RandomWalkWithDrift | seasonal | -1.0902 | -1.3963 | -1.6636 |
| SES | seasonal | N/A | N/A | N/A |
| SeasonalNaive | seasonal | 0.0011 | 0.0021 | 0.0026 |
| TSB | seasonal | N/A | N/A | N/A |
| Theta | seasonal | 4.4569 | 5.6474 | 7.0238 |
| ARIMA_1_1_1 | seasonal_negative | -24.7996 | -31.8282 | -37.9256 |
| AutoARIMA | seasonal_negative | 12.7061 | 16.3097 | 19.4343 |
| AutoETS | seasonal_negative | 0.2021 | 0.2596 | 0.3094 |
| Croston | seasonal_negative | N/A | N/A | N/A |
| CrostonSBA | seasonal_negative | N/A | N/A | N/A |
| Holt | seasonal_negative | 61.3245 | 78.7180 | 93.7986 |
| HoltWinters | seasonal_negative | 0.2841 | 0.3650 | 0.4349 |
| Naive | seasonal_negative | 0.0034 | 0.0068 | 0.0082 |
| RandomWalkWithDrift | seasonal_negative | -0.9006 | -1.1534 | -1.3743 |
| SES | seasonal_negative | N/A | N/A | N/A |
| SeasonalNaive | seasonal_negative | 0.0004 | 0.0009 | 0.0011 |
| TSB | seasonal_negative | N/A | N/A | N/A |
| Theta | seasonal_negative | -14.2490 | -18.5594 | -21.0594 |
| ARIMA_1_1_1 | stationary | 14.8095 | 19.0104 | 22.6524 |
| AutoARIMA | stationary | 11.9595 | 15.3521 | 18.2932 |
| AutoETS | stationary | 1.3001 | 1.6698 | 1.9897 |
| Croston | stationary | N/A | N/A | N/A |
| CrostonSBA | stationary | N/A | N/A | N/A |
| Holt | stationary | 23.8606 | 30.6282 | 36.4958 |
| HoltWinters | stationary | 1.1221 | 1.4414 | 1.7175 |
| Naive | stationary | 0.0045 | 0.0092 | 0.0111 |
| RandomWalkWithDrift | stationary | -1.2216 | -1.5646 | -1.8642 |
| SES | stationary | N/A | N/A | N/A |
| SeasonalNaive | stationary | 0.0018 | 0.0036 | 0.0043 |
| TSB | stationary | N/A | N/A | N/A |
| Theta | stationary | 8.7760 | 11.1175 | 13.8244 |
| ARIMA_1_1_1 | trend | 6.6459 | 8.5320 | 10.1665 |
| AutoARIMA | trend | 8.6436 | 11.0957 | 13.2213 |
| AutoETS | trend | 2.2208 | 2.8514 | 3.3976 |
| Croston | trend | N/A | N/A | N/A |
| CrostonSBA | trend | N/A | N/A | N/A |
| Holt | trend | 19.1970 | 24.6418 | 29.3627 |
| HoltWinters | trend | 1.8453 | 2.3695 | 2.8235 |
| Naive | trend | 0.0034 | 0.0068 | 0.0082 |
| RandomWalkWithDrift | trend | -0.8941 | -1.1451 | -1.3644 |
| SES | trend | N/A | N/A | N/A |
| SeasonalNaive | trend | 0.0027 | 0.0054 | 0.0065 |
| TSB | trend | N/A | N/A | N/A |
| Theta | trend | 16.6115 | 21.6031 | 26.0523 |
| ARIMA_1_1_1 | trend_seasonal | -7.3160 | -9.3875 | -11.1858 |
| AutoARIMA | trend_seasonal | 12.7895 | 16.4171 | 19.5622 |
| AutoETS | trend_seasonal | 1.1594 | 1.4888 | 1.7740 |
| Croston | trend_seasonal | N/A | N/A | N/A |
| CrostonSBA | trend_seasonal | N/A | N/A | N/A |
| Holt | trend_seasonal | 4.6465 | 5.9668 | 7.1100 |
| HoltWinters | trend_seasonal | 1.1594 | 1.4888 | 1.7740 |
| Naive | trend_seasonal | 0.0035 | 0.0070 | 0.0085 |
| RandomWalkWithDrift | trend_seasonal | -0.9309 | -1.1922 | -1.4205 |
| SES | trend_seasonal | N/A | N/A | N/A |
| SeasonalNaive | trend_seasonal | 0.0016 | 0.0032 | 0.0039 |
| TSB | trend_seasonal | N/A | N/A | N/A |
| Theta | trend_seasonal | 7.4019 | 9.7761 | 11.9903 |


---

## Detailed Point Forecast Differences

Largest absolute differences:

| Model | Series | Step | Rust | statsforecast | Difference |
|-------|--------|------|------|---------------|------------|
| TSB | multiplicative_seasonal | 12 | 30.4916 | 97.1668 | -66.6753 |
| TSB | multiplicative_seasonal | 11 | 33.8795 | 97.1668 | -63.2873 |
| TSB | multiplicative_seasonal | 10 | 37.6439 | 97.1668 | -59.5229 |
| TSB | multiplicative_seasonal | 9 | 41.8266 | 97.1668 | -55.3403 |
| TSB | multiplicative_seasonal | 8 | 46.4740 | 97.1668 | -50.6929 |
| Holt | seasonal | 12 | 111.6656 | 62.2406 | 49.4250 |
| TSB | multiplicative_seasonal | 7 | 51.6378 | 97.1668 | -45.5291 |
| Holt | seasonal | 11 | 107.4147 | 62.1084 | 45.3063 |
| Holt | seasonal | 10 | 103.1639 | 61.9763 | 41.1876 |
| TSB | multiplicative_seasonal | 6 | 57.3753 | 97.1668 | -39.7916 |

---

## Metrics by Forecast Horizon Step

Aggregated metrics across all models and series types by forecast step:

| Step | MAD | Median | Max Diff | Mean Diff | Std |
|------|-----|--------|----------|-----------|-----|
| 1 | 1.4683 | 0.0466 | 27.5914 | 0.6871 | 4.0153 |
| 2 | 2.1394 | 0.6129 | 27.5914 | 0.4290 | 4.6131 |
| 3 | 2.5088 | 0.7958 | 27.5914 | 0.2161 | 5.3147 |
| 4 | 2.8209 | 0.3382 | 27.5914 | 0.0613 | 6.3674 |
| 5 | 3.2510 | 0.2226 | 33.4165 | 0.1017 | 7.5546 |
| 6 | 4.0669 | 0.8314 | 39.7916 | 0.0526 | 8.8293 |
| 7 | 4.3812 | 0.2573 | 45.5291 | -0.2432 | 9.8292 |
| 8 | 4.5553 | 0.3178 | 50.6929 | -0.3020 | 10.5732 |
| 9 | 4.7598 | 0.5543 | 55.3403 | -0.6760 | 11.1864 |
| 10 | 4.8444 | 0.4335 | 59.5229 | -0.8536 | 11.8102 |
| 11 | 5.2740 | 0.7585 | 63.2873 | -1.0015 | 12.5925 |
| 12 | 5.5156 | 0.7200 | 66.6753 | -1.3674 | 13.2995 |

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
