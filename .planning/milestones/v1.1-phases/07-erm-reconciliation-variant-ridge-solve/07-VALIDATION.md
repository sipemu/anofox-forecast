---
phase: "7"
slug: "erm-reconciliation-variant-ridge-solve"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-09"
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for the ERM reconciliation variant + ridge solve.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (built-in) |
| **Config file** | none — inline `#[cfg(test)]` in `src/hierarchy/mod.rs` (matching existing mint tests) and/or a `tests/` integration file |
| **Quick run command** | `cargo test --lib hierarchy::` (or `cargo test --lib erm`) |
| **Full suite command** | `cargo test && cargo clippy --all-targets --all-features -- -D warnings` |
| **Estimated runtime** | ~5s quick / ~2–4 min full (NOTE: `cargo test --all-features` OOMs the linker in this sandbox — use default `cargo test` + `cargo test --lib`) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib hierarchy::`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd-verify-work`:** Full default-feature suite + clippy `--all-features` green
- **Max feedback latency:** ~5 seconds (quick)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 7-01-01 | 01 | 1 | ERM-01 | `Erm { lambda }` variant compiles; existing hierarchy code unchanged; serde round-trips | unit | `cargo test --lib hierarchy::` | ❌ W0 | ⬜ pending |
| 7-01-02 | 01 | 1 | ERM-02 | `with_erm_training` stores base+leaf history; missing history → ForecastError | unit | `cargo test --lib erm` | ❌ W0 | ⬜ pending |
| 7-01-03 | 01 | 1 | ERM-03 | ridge solve P = B'Ŷ(Ŷ'Ŷ+λI)⁻¹; bottom = P·ŷ, all = S·P·ŷ | unit | `cargo test --lib erm` | ❌ W0 | ⬜ pending |
| 7-01-04 | 01 | 1 | ERM-05 | hand-computed hierarchy (Total→{A,B}, T=2, λ=1): A=16/3, B=5/2, Total=47/6; coherence S·bottom==all | unit | `cargo test --lib erm_correctness` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] ERM unit tests (inline `#[cfg(test)]` in `src/hierarchy/mod.rs` mirroring `mint_ols_coherent` etc., or a `tests/` integration file):
  - variant-exists + backward-compat compile (ERM-01)
  - training-history setter + missing-history error + shape validation (ERM-02)
  - ridge-solve output shape + coherence (ERM-03)
  - **the hand-computed reference-formula correctness test** (ERM-05) — the core deliverable.

*Existing `cargo test` infrastructure covers execution.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| npm/WASM package still builds | ERM (SC5) | Separate wasm-pack toolchain; JS binding `parse_method()` catch-all compiles unchanged | Run the repo WASM build target |

*All ERM correctness/behaviour claims (ERM-01/02/03/05) have automated unit tests.*

---

## Validation Sign-Off

- [ ] `Erm { lambda }` variant exists; existing hierarchy code compiles unchanged (ERM-01)
- [ ] Training-history API + missing-history error covered (ERM-02)
- [ ] Ridge-solve output + coherence covered (ERM-03)
- [ ] Hand-computed reference-formula correctness test passes within tolerance (ERM-05)
- [ ] No watch-mode flags
- [ ] clippy `--all-features` green
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
