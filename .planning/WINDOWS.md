---
schema_version: 1
open_count: 4
waived_count: 0
fixed_count: 0
total_count: 4
last_updated: 2026-10-30T12:18:45.870Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | stub | .planning/baselines/iai.json |  | instruction_count=0 placeholder for all 3 hot paths — regenerate via bash scripts/update_iai.sh with valgrind installed | open |  | 2026-08-09T20:42:36.447Z |  |
| 2 | 01 | stub | .planning/baselines/criterion.json |  | criterion.json: all median_ns/mad_ns values are 0.0 placeholders; regenerate with bash scripts/update_criterion.sh on a quiet local machine (D-03 local-only capture) | open |  | 2026-08-09T21:10:06.047Z |  |
| 3 | quick-261030-go2 | deviation | src/utils/optimization/basin_backend.rs |  | lbfgs_optimize_legacy panics on memory_size=0 (pre-existing lbfgs crate assertion, unrelated to basin); not fixed, out of scope for GH-234 step 1 - see docs/audits/basin-optimizer-parity.md | open |  | 2026-10-30T12:18:45.770Z |  |
| 4 | quick-261030-go2 | lint-warning | src/models/smart.rs |  | cargo clippy --all-targets -D warnings fails at HEAD (unused imports/dead code), pre-existing and unrelated to this task - verified identical via git stash before/after | open |  | 2026-10-30T12:18:45.870Z |  |

````json
[
  {
    "id": 1,
    "kind": "stub",
    "phase": "01",
    "file": ".planning/baselines/iai.json",
    "line": null,
    "description": "instruction_count=0 placeholder for all 3 hot paths — regenerate via bash scripts/update_iai.sh with valgrind installed",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-09T20:42:36.447Z",
    "resolved_at": null
  },
  {
    "id": 2,
    "kind": "stub",
    "phase": "01",
    "file": ".planning/baselines/criterion.json",
    "line": null,
    "description": "criterion.json: all median_ns/mad_ns values are 0.0 placeholders; regenerate with bash scripts/update_criterion.sh on a quiet local machine (D-03 local-only capture)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-09T21:10:06.047Z",
    "resolved_at": null
  },
  {
    "id": 3,
    "kind": "deviation",
    "phase": "quick-261030-go2",
    "file": "src/utils/optimization/basin_backend.rs",
    "line": null,
    "description": "lbfgs_optimize_legacy panics on memory_size=0 (pre-existing lbfgs crate assertion, unrelated to basin); not fixed, out of scope for GH-234 step 1 - see docs/audits/basin-optimizer-parity.md",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-30T12:18:45.770Z",
    "resolved_at": null,
    "milestone": "v1.1"
  },
  {
    "id": 4,
    "kind": "lint-warning",
    "phase": "quick-261030-go2",
    "file": "src/models/smart.rs",
    "line": null,
    "description": "cargo clippy --all-targets -D warnings fails at HEAD (unused imports/dead code), pre-existing and unrelated to this task - verified identical via git stash before/after",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-30T12:18:45.870Z",
    "resolved_at": null,
    "milestone": "v1.1"
  }
]
````
