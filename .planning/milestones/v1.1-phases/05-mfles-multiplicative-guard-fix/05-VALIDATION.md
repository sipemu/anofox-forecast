---
phase: "5"
slug: "mfles-multiplicative-guard-fix"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-09"
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (built-in) |
| **Config file** | none — `Cargo.toml` [[test]] auto-discovery of `tests/*.rs` |
| **Quick run command** | `cargo test --test issue_219_mfles_multiplicative_runaway` |
| **Full suite command** | `cargo test && cargo clippy --all-targets --all-features -- -D warnings` |
| **Estimated runtime** | ~5s quick / ~2–4 min full |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --test issue_219_mfles_multiplicative_runaway` (plus any touched MFLES unit tests)
- **After every plan wave:** Run `cargo test`
- **Before `/gsd-verify-work`:** Full suite + clippy must be green
- **Max feedback latency:** ~5 seconds (quick), ~240 seconds (full)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | MULT-01 | — | Auto mode picks additive when min/median < 0.10 | unit | `cargo test mfles_ mode_guard` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 1 | MULT-02 | — | ln() input floored at 0.01×median (no log crater) | unit | `cargo test mfles_ log_floor` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | MULT-03 | — | predict clamps forecast ≤ 10×in-sample max, ≥ 0 | unit | `cargo test mfles_ back_clamp` | ❌ W0 | ⬜ pending |
| 5-01-04 | 01 | 1 | MULT-04 | — | #219 repro forecasts at ~level (no ~13× blow-up) | regression | `cargo test --test issue_219_mfles_multiplicative_runaway` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/issue_219_mfles_multiplicative_runaway.rs` — regression test reproducing the #219 near-zero series (level ≈ 2k, one near-zero month), asserting forecast within a bounded multiple of level.
- [ ] MFLES unit tests (inline `#[cfg(test)]` in `src/models/mfles.rs` or the regression file) for each guard: mode-selection, log floor, back-transform clamp.

*Existing `cargo test` infrastructure covers execution — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| npm/WASM package still builds | MULT (SC5) | WASM build is a separate toolchain (`wasm-pack`), not part of `cargo test` | Run `make wasm` (or the repo's WASM build target) and confirm success |

*All robustness behaviors (MULT-01..04) have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 240s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
