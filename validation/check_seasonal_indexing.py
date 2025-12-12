#!/usr/bin/env python3
"""Check seasonal indexing logic."""

# Data has 100 observations
n = 100
period = 12

print("Seasonal Indexing Analysis")
print("=" * 60)
print()

print(f"Data length: {n}")
print(f"Seasonal period: {period}")
print()

# Last few observations and their seasonal indices
print("Last 15 observations and their seasonal indices:")
for i in range(n-15, n):
    season_idx = i % period
    print(f"  obs[{i:3d}]: season_idx = {i} % {period} = {season_idx}")
print()

# Forecast horizon and expected seasonal indices
print("Forecast horizon and expected seasonal indices:")
for h in range(1, 13):
    # Rust formula: (n + h - 1) % period
    rust_idx = (n + h - 1) % period

    # Alternative: The NEXT seasonal index after the last observation
    last_idx = (n - 1) % period  # Last observation's season
    expected_idx = (last_idx + h) % period

    print(f"  h={h:2d}: rust_idx={(n + h - 1):3d} % {period} = {rust_idx:2d}  |  expected={(last_idx + h):3d} % {period} = {expected_idx:2d}  |  match={rust_idx == expected_idx}")

print()
print("Analysis:")
print(f"  Last observation index: {n-1}")
print(f"  Last observation season: {(n-1) % period}")
print(f"  For h=1, we want the season AFTER the last, which is: {((n-1) % period + 1) % period}")
print(f"  Rust formula gives: {(n + 1 - 1) % period}")
print(f"  These match: {((n-1) % period + 1) % period == (n + 1 - 1) % period}")
