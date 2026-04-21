---
phase: quick-1
plan: 1
subsystem: documentation

tags: [surrogate, data-collection, verification, worker-stall, investigation]

requires:
  - phase: 02.2-collect-rl-step-only-minimal-change-from-2-1
    provides: 02.2-VERIFICATION.md (6/6 truths verified)

provides:
  - "Worker respawn investigation issue consolidating 3 existing stall/data-loss/deadlock issues"

affects:
  - "Future collection runs: investigation steps documented for 10+ GB scaling"

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - issues/worker-respawn-investigation-needed.md
  modified: []

key-decisions:
  - "No new Phase 02.2 verification document needed -- existing 02.2-VERIFICATION.md is thorough and passed 6/6"
  - "Worker respawn issue framed as investigation tracker, not a fix -- NUMBA_NUM_THREADS=1 fix proposed but deferred to application before next large collection"

duration: 1min
completed: 2026-03-11
---

# Quick Task 1: Phase 02.2 Verification Confirmation and Worker Respawn Investigation Issue

**Confirmed Phase 02.2 verification (6/6 truths passed) and created worker respawn investigation issue consolidating 3 existing issues with 5 concrete investigation steps for 10+ GB collection scaling**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-03-11T00:41:23Z
- **Completed:** 2026-03-11T00:42:44Z
- **Tasks:** 1/1
- **Files created:** 1

## Accomplishments

### Phase 02.2 Verification Confirmation

The existing `02.2-VERIFICATION.md` was confirmed as present, thorough, and passed:
- **Score:** 6/6 observable truths verified
- **Requirements:** RLDC-01, RLDC-02, RLDC-03 all satisfied
- **Anti-patterns:** 1 minor warning (W&B total_transitions overcount, does not affect data correctness)
- **No new verification document needed** -- the machine-generated verification is complete

### Worker Respawn Investigation Issue

Created `issues/worker-respawn-investigation-needed.md` consolidating findings from:
1. `surrogate-worker-7-stall-respawn.md` -- operational timeline of 6 stalls
2. `surrogate-data-loss-on-worker-respawn.md` -- 19% data loss from in-memory transition loss
3. `numba-thread-pool-deadlock-worker-stalls.md` -- root cause: Numba thread pool deadlock (NUMBA_NUM_THREADS unset, defaults to 48)

The issue documents:
- Diagnosed root cause and stall rate (~1 per 9.6 worker-hours)
- Proposed but unapplied NUMBA_NUM_THREADS=1 fix
- 5 concrete investigation steps for when longer collection runs are attempted

## Task Commits

1. **Task 1: Confirm verification + create investigation issue** -- `65effd1` (docs)

## Deviations from Plan

None -- plan executed exactly as written.

## Self-Check

- [x] `issues/worker-respawn-investigation-needed.md` exists
- [x] Commit `65effd1` exists in git log
- [x] Phase 02.2 verification confirmed (02.2-VERIFICATION.md, 6/6 passed)
