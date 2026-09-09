---
phase: "6"
slug: "multiplicative-guard-bug-class-audit"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-09"
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for the multiplicative-guard bug-class audit.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (built-in) |
| **Config file** | none — `Cargo.toml` [[test]] auto-discovery; `[[test]] required-features` for gated audits |
| **Quick run command** | `cargo test --test mult05_guard_audit_assertions` |
| **Full suite command** | `cargo test && cargo clippy --all-targets --all-features -- -D warnings` |
| **Estimated runtime** | ~5s quick / ~2–4 min full (note: `cargo test --all-features` OOMs the linker in this sandbox — use default `cargo test` + targeted `--features distributional` runs) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --test mult05_guard_audit_assertions`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd-verify-work`:** Full default-feature suite + clippy `--all-features` must be green
- **Max feedback latency:** ~5 seconds (quick)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 1 | MULT-05 | Inventory doc lists every in-scope model/path with a verdict | manual_procedural | doc review of docs/audits/multiplicative-guard-audit.md | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | MULT-05 | Already-safe models proven: near-zero series forecasts ~level (Theta, AutoETS) | integration | `cargo test --test mult05_guard_audit_assertions` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | MULT-06 | No new offender found → no fix needed; documented as such | manual_procedural | inventory verdict rows (all pass/N-A) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/mult05_guard_audit_assertions.rs` — guard-assertion tests for the already-safe at-risk-looking models (at minimum Theta seasonal-factor guard + AutoETS non-positive guard), each feeding a near-zero-relative-to-level series and asserting forecast stays ~level.
- [ ] `docs/audits/multiplicative-guard-audit.md` — the committed inventory (create `docs/audits/`).

*Existing `cargo test` infrastructure covers execution.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Inventory completeness (every in-scope path has a verdict) | MULT-05 | Completeness of a doc table is a review judgment, not an assertion | Review docs/audits/multiplicative-guard-audit.md against the scout list in 06-CONTEXT / 06-RESEARCH |
| npm/WASM package still builds | SC5 | Separate wasm-pack toolchain | Run the repo WASM build target |

*Behavioral audit claims for the exercised models have automated tests; the inventory-completeness and no-new-offender claims are review-backed.*

---

## Validation Sign-Off

- [ ] Every in-scope model/path has a verdict row (pass/fail/N-A) in the inventory
- [ ] Each "already-safe" claim for an at-risk-looking model is backed by a passing guard-assertion test where practical
- [ ] Any offender (none expected per research) has a regression test proving the pre-fix blow-up is gone
- [ ] No watch-mode flags
- [ ] clippy `--all-features` green
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
