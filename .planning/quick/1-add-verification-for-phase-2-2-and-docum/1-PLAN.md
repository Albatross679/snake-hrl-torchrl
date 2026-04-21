---
phase: quick-1
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - issues/worker-respawn-investigation-needed.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "Phase 02.2 verification document exists and confirms all 6 must-have truths are verified"
    - "A new issue document captures the worker respawn investigation need, referencing existing related issues"
  artifacts:
    - path: "issues/worker-respawn-investigation-needed.md"
      provides: "Investigation tracker for worker respawn root cause analysis after training completes"
      contains: "status: open"
  key_links: []
---

<objective>
Create documentation for Phase 02.2 verification status and a worker respawn investigation issue.

Purpose: Phase 02.2 already has a machine-generated 02.2-VERIFICATION.md (passed, 6/6 truths verified).
The existing verification is thorough and complete. The second deliverable is a new issue document
that consolidates the worker respawn investigation needs -- referencing the three existing related
issues (surrogate-worker-7-stall-respawn.md, surrogate-data-loss-on-worker-respawn.md,
numba-thread-pool-deadlock-worker-stalls.md) -- and framing what investigation is still needed
once training completes and longer collection runs are attempted.

Output: One issue document in issues/.
</objective>

<context>
@.planning/STATE.md
@.planning/phases/02.2-collect-rl-step-only-minimal-change-from-2-1/02.2-VERIFICATION.md
@.planning/phases/02.2-collect-rl-step-only-minimal-change-from-2-1/02.2-01-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Confirm Phase 02.2 verification is complete and create worker respawn investigation issue</name>
  <files>issues/worker-respawn-investigation-needed.md</files>
  <action>
**Step A: Confirm Phase 02.2 verification status**

Read the existing `.planning/phases/02.2-collect-rl-step-only-minimal-change-from-2-1/02.2-VERIFICATION.md`
and confirm it is present and passed (6/6 truths). Report the existing verification status to the user --
no new verification document is needed since 02.2-VERIFICATION.md already exists and is thorough.

**Step B: Create worker respawn investigation issue**

Create `issues/worker-respawn-investigation-needed.md` with proper frontmatter per CLAUDE.md spec.

The issue should:

1. Use status `open`, severity `medium`, subtype `performance`
2. Frame this as an investigation that should happen when training completes and longer collection
   runs are attempted (Phase 02.2 collection to 10 GB)
3. Reference the three existing related issues by filename:
   - `surrogate-worker-7-stall-respawn.md` -- operational timeline of 6 stalls in Phase 02.1 run
   - `surrogate-data-loss-on-worker-respawn.md` -- 19% data loss from in-memory transitions lost on respawn
   - `numba-thread-pool-deadlock-worker-stalls.md` -- root cause analysis pointing to Numba thread pool
4. Note what has already been diagnosed:
   - Root cause: Numba thread pool deadlock in forked workers (NUMBA_NUM_THREADS unset, defaults to 48)
   - Rate: ~1 stall per 9.6 worker-hours during Phase 02.1 (6 stalls across 3.5 hours, 16 workers)
   - Quick fix proposed but NOT yet applied: set NUMBA_NUM_THREADS=1 and NUMBA_THREADING_LAYER=workqueue
   - Data loss mechanism: respawned workers lose in-memory transitions (up to 49,999 with current 50k flush threshold)
5. List what investigation is still needed:
   - Verify whether Phase 02.2 collection (which is currently running) experiences the same stall rate
   - Apply the NUMBA_NUM_THREADS=1 fix and measure whether stall rate drops to zero
   - Implement the data loss mitigation: reduce flush interval from 50k to 10k, add SIGTERM handler
   - Monitor whether longer runs (10+ GB) show increasing stall rates or new failure modes
   - Check if steps_per_run=1 (Phase 02.2) changes stall characteristics vs steps_per_run=4 (Phase 02.1)
6. Tags: [surrogate, data-collection, worker-stall, investigation, numba]
  </action>
  <verify>
    <automated>test -f /home/coder/snake-hrl-torchrl/issues/worker-respawn-investigation-needed.md && grep -q "status: open" /home/coder/snake-hrl-torchrl/issues/worker-respawn-investigation-needed.md && grep -q "numba-thread-pool-deadlock" /home/coder/snake-hrl-torchrl/issues/worker-respawn-investigation-needed.md && echo "OK" || echo "MISSING"</automated>
  </verify>
  <done>
  - Phase 02.2 verification confirmed as existing and passed (02.2-VERIFICATION.md, 6/6 truths)
  - Worker respawn investigation issue created with open status, referencing all 3 related issues, listing investigation steps needed
  </done>
</task>

</tasks>

<verification>
- `test -f issues/worker-respawn-investigation-needed.md` -- issue file exists
- `grep "status: open" issues/worker-respawn-investigation-needed.md` -- status is open
- `grep "surrogate-worker-7-stall-respawn" issues/worker-respawn-investigation-needed.md` -- references related issue
- `grep "surrogate-data-loss-on-worker-respawn" issues/worker-respawn-investigation-needed.md` -- references related issue
- `grep "numba-thread-pool-deadlock" issues/worker-respawn-investigation-needed.md` -- references related issue
- `test -f .planning/phases/02.2-collect-rl-step-only-minimal-change-from-2-1/02.2-VERIFICATION.md` -- verification exists
</verification>

<success_criteria>
- Phase 02.2 verification confirmed as existing (02.2-VERIFICATION.md passed 6/6)
- Worker respawn investigation issue created with proper CLAUDE.md frontmatter
- Issue references all 3 existing related issues and lists concrete investigation steps
- Issue framed as work to do when longer collection runs are attempted
</success_criteria>

<output>
No SUMMARY file needed for quick plans.
</output>
