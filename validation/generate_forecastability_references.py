#!/usr/bin/env python3
"""Generate reference values for validating the Rust forecastability module.

Computes ground-truth statistics using numpy/scipy on deterministic test
data, then prints them as Rust const arrays for use in validation tests.

Run: python3 validation/generate_forecastability_references.py
"""

import json
import numpy as np
from scipy import stats
from scipy.special import digamma as scipy_digamma

# ── Deterministic test data generators ──────────────────────────────────

def make_ar1(n, phi, seed=42):
    """Generate AR(1): x[t] = phi * x[t-1] + noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x

def make_white_noise(n, seed=99):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def make_logistic_map(n, r=3.9, x0=0.1):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = r * x[t - 1] * (1 - x[t - 1])
    return x

def make_sine(n, period=50):
    return np.sin(2 * np.pi * np.arange(n) / period)

# ── Reference computations ──────────────────────────────────────────────

def ar1_theoretical_ami(phi, max_lag):
    """Exact AMI of AR(1): I(h) = -0.5 * ln(1 - phi^{2h})."""
    return [-0.5 * np.log(1 - phi ** (2 * h)) for h in range(1, max_lag + 1)]

def pearson_at_lag(series, h):
    return abs(np.corrcoef(series[:-h], series[h:])[0, 1])

def spearman_at_lag(series, h):
    rho, _ = stats.spearmanr(series[:-h], series[h:])
    return abs(rho)

def kendall_at_lag(series, h):
    tau, _ = stats.kendalltau(series[:-h], series[h:])
    return abs(tau)

def distance_correlation_ref(x, y):
    """Naive O(n^2) distance correlation matching Szekely/Rizzo."""
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return np.sqrt(max(dcov2 / np.sqrt(dvar_x * dvar_y), 0))

def gcmi_ref(x, y):
    """GCMI: rank -> probit -> Pearson -> -0.5 * log2(1 - rho^2)."""
    from scipy.stats import rankdata, norm
    n = len(x)
    rx = norm.ppf(rankdata(x) / (n + 1))
    ry = norm.ppf(rankdata(y) / (n + 1))
    rho = np.corrcoef(rx, ry)[0, 1]
    return -0.5 * np.log2(1 - rho ** 2)

# ── Generate all references ─────────────────────────────────────────────

def main():
    results = {}

    # 1. AR(1) theoretical AMI
    phi = 0.8
    max_lag = 10
    results["ar1_theoretical_ami_phi08"] = ar1_theoretical_ami(phi, max_lag)

    # 2. AR(1) empirical reference (large n for stable estimates)
    ar1 = make_ar1(2000, 0.8, seed=42)

    # Pearson / Spearman / Kendall at lags 1..5
    results["ar1_pearson"] = [pearson_at_lag(ar1, h) for h in range(1, 6)]
    results["ar1_spearman"] = [spearman_at_lag(ar1, h) for h in range(1, 6)]
    results["ar1_kendall"] = [kendall_at_lag(ar1, h) for h in range(1, 6)]

    # GCMI at lags 1..5
    results["ar1_gcmi"] = [gcmi_ref(ar1[:-h], ar1[h:]) for h in range(1, 6)]

    # Distance correlation at lag 1
    results["ar1_dcor_lag1"] = distance_correlation_ref(ar1[:-1], ar1[1:])

    # 3. Independent variables — all measures should be near 0
    wn1 = make_white_noise(1000, seed=99)
    wn2 = make_white_noise(1000, seed=77)
    results["independent_pearson"] = abs(np.corrcoef(wn1, wn2)[0, 1])
    results["independent_spearman"] = abs(stats.spearmanr(wn1, wn2)[0])
    results["independent_kendall"] = abs(stats.kendalltau(wn1, wn2)[0])
    results["independent_dcor"] = distance_correlation_ref(wn1, wn2)
    results["independent_gcmi"] = gcmi_ref(wn1, wn2)

    # 4. Perfect linear dependence: y = 3x + 2
    x_lin = np.arange(100, dtype=float)
    y_lin = 3 * x_lin + 2
    results["linear_dcor"] = distance_correlation_ref(x_lin, y_lin)
    results["linear_gcmi"] = gcmi_ref(x_lin, y_lin)

    # 5. Nonlinear dependence: y = x^2 (symmetric x)
    x_quad = np.linspace(-5, 5, 200)
    y_quad = x_quad ** 2
    results["quadratic_pearson"] = abs(np.corrcoef(x_quad, y_quad)[0, 1])
    results["quadratic_dcor"] = distance_correlation_ref(x_quad, y_quad)

    # 6. Kendall on small exact data
    x_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_small = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    results["reversed_kendall"] = abs(stats.kendalltau(x_small, y_small)[0])

    x_conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_conc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    results["concordant_kendall"] = abs(stats.kendalltau(x_conc, y_conc)[0])

    # 7. Digamma at exact integer values
    results["digamma_values"] = [float(scipy_digamma(i)) for i in range(1, 11)]

    # Print as JSON for easy parsing
    # Convert numpy types to Python native
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = {k: convert(v) for k, v in results.items()}
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
