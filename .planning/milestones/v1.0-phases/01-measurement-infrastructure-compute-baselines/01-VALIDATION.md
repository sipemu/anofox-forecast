---
phase: 1
slug: measurement-infrastructure-compute-baselines
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo (criterion 0.5, iai-callgrind, dhat, cargo test) + GitHub Actions |
| **Config file** | none — Wave 0 adds the harness crate + dev-deps |
| **Quick run command** | `cargo test -p anofox-bench-harness` |
| **Full suite command** | `bash scripts/update_criterion.sh && bash scripts/update_iai.sh && bash scripts/update_dhat.sh && bash scripts/update_wasm_size.sh` |
| **Estimated runtime** | ~varies (criterion local capture minutes; iai valgrind seconds/min) |

---

## Sampling Rate

- **After every task commit:** Run `cargo build` / relevant `cargo test`
- **After every plan wave:** Run the dimension's `scripts/update_*.sh`
- **Before `/gsd-verify-work`:** All four baseline JSONs committed; `bench.yml` + `wasm-size.yml` green
- **Max feedback latency:** local capture

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 1 | REQ-{XX} | — | N/A | integration | `{command}` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Planner: populate this map from PLAN.md tasks — one row per task, mapping each of MEAS-01..04 / PERF-01..06 to an automated command.*

---

## Wave 0 Requirements

- [ ] `crates/anofox-bench-harness/` — new workspace member owning baseline serde structs + read/compare logic
- [ ] iai-callgrind, dhat dev-dependencies added and building
- [ ] `.planning/baselines/` directory created

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Local criterion baseline capture (not on CI) | PERF-03 / MEAS-02 | Wall-clock noise forbids CI capture (locked constraint) | Run `scripts/update_criterion.sh` on a quiet local machine |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
