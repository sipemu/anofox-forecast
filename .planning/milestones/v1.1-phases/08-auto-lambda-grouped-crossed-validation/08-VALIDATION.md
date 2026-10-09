---
phase: "8"
slug: "auto-lambda-grouped-crossed-validation"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-09"
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for ERM auto-λ + grouped/crossed end-to-end validation.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (built-in) |
| **Config file** | none — inline `#[cfg(test)]` in `src/hierarchy/mod.rs` and/or a `tests/` integration file |
| **Quick run command** | `cargo test --lib "hierarchy::tests::erm"` |
| **Full suite command** | `cargo test && cargo clippy --all-targets --all-features -- -D warnings` |
| **Estimated runtime** | ~5s quick / ~2–4 min full (NOTE: `cargo test --all-features` OOMs the linker here — use default `cargo test` + `cargo test --lib`) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib "hierarchy::tests::erm"`
- **After every plan wave:** Run `cargo test --lib hierarchy::`
- **Before `/gsd-verify-work`:** Full default-feature suite + clippy `--all-features` green
- **Max feedback latency:** ~5 seconds (quick)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 1 | ERM-04 | `Erm{lambda: Option<f64>}`; None → LW auto-λ; Some(λ) → fixed (Phase 7 guards); 9 Phase-7 tests updated to Some | unit | `cargo test --lib "hierarchy::tests::erm"` | ⚠️ existing | ⬜ pending |
| 8-01-02 | 01 | 1 | ERM-04 | auto-λ produces a finite, well-conditioned, sensible shrink on crossed data | unit | `cargo test --lib erm_auto_lambda` | ❌ W0 | ⬜ pending |
| 8-01-03 | 01 | 1 | ERM-06 | grouped/crossed hierarchy: ERM auto-λ coherent across all 9 nodes × H; RMSSE ≤ unreconciled; table vs MinTrace | integration | `cargo test --lib erm_grouped_crossed` | ❌ W0 | ⬜ pending |
| 8-01-04 | 01 | 1 | ERM-06 | committed before/after results note (unreconciled / MinTrace / ERM auto-λ RMSSE table) | manual_procedural | doc review of docs/audits/erm-grouped-validation-results.md | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] ERM auto-λ + validation tests (inline `#[cfg(test)]` in `src/hierarchy/mod.rs`, mirroring the Phase 7 ERM tests):
  - `Option<f64>` signature: None → auto-λ, Some(λ) → fixed; Phase 7 correctness tests updated to `Some(λ)` and still passing (ERM-04)
  - `erm_auto_lambda` helper produces a finite, non-negative, well-conditioned λ; degeneracy guards (small T) (ERM-04)
  - grouped/crossed end-to-end: coherence across all nodes/horizons (the hard proof) + RMSSE comparison vs unreconciled + MinTrace, ERM run with auto-λ (None) (ERM-06, SC4)
- [ ] `docs/audits/erm-grouped-validation-results.md` — committed RMSSE results table (before/after).

*Existing `cargo test` + `crate::utils::rmsse` cover execution.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| npm/WASM package still builds | ERM (SC5) | Separate wasm-pack toolchain; the `Option<f64>` arm + serde round-trip | Run the repo WASM build target |
| Auto-λ "behaves sensibly" narrative | ERM-04/06 SC4 | Judgment of the committed RMSSE table | Review docs/audits/erm-grouped-validation-results.md |

*ERM-04/06 core behaviours have automated tests; the results-table judgment is review-backed.*

---

## Validation Sign-Off

- [ ] `Erm{lambda: Option<f64>}` — None auto-λ default + Some(λ) fixed, both selectable (ERM-04)
- [ ] auto-λ helper produces finite/well-conditioned shrink; degeneracy guarded
- [ ] grouped/crossed coherence proof + RMSSE before/after vs unreconciled + MinTrace committed (ERM-06)
- [ ] auto-λ (None) exercised in the end-to-end validation (SC4)
- [ ] Phase 7 ERM tests updated to Some(λ) and passing; pre-existing hierarchy tests pass
- [ ] No watch-mode flags
- [ ] clippy `--all-features` green
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
