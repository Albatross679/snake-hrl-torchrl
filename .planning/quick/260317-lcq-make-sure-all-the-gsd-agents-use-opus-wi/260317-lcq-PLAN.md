---
phase: quick-260317-lcq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .claude/get-shit-done/bin/lib/model-profiles.cjs
  - .claude/get-shit-done/references/model-profiles.md
autonomous: true
requirements: [ALL-OPUS]
must_haves:
  truths:
    - "Every agent in the quality profile resolves to opus"
    - "Reference documentation matches the code"
  artifacts:
    - path: ".claude/get-shit-done/bin/lib/model-profiles.cjs"
      provides: "Model profile mapping (source of truth)"
      contains: "quality: 'opus'"
    - path: ".claude/get-shit-done/references/model-profiles.md"
      provides: "Human-readable profile documentation"
      contains: "opus for all 15 agents in quality column"
  key_links:
    - from: ".claude/get-shit-done/bin/lib/model-profiles.cjs"
      to: ".claude/get-shit-done/references/model-profiles.md"
      via: "manual sync"
      pattern: "quality.*opus"
---

<objective>
Change the GSD quality model profile so ALL 15 agents use Opus — no exceptions.

Purpose: The user wants maximum reasoning power across all agents when using the quality profile.
Output: Updated model-profiles.cjs and model-profiles.md with all quality entries set to opus.
</objective>

<execution_context>
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.claude/get-shit-done/bin/lib/model-profiles.cjs
@.claude/get-shit-done/references/model-profiles.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Set all quality profile entries to opus in model-profiles.cjs</name>
  <files>.claude/get-shit-done/bin/lib/model-profiles.cjs</files>
  <action>
In MODEL_PROFILES (lines 9-25), change the quality value from 'sonnet' to 'opus' for these 8 agents:
- gsd-research-synthesizer (line 15)
- gsd-codebase-mapper (line 17)
- gsd-verifier (line 18)
- gsd-plan-checker (line 19)
- gsd-integration-checker (line 20)
- gsd-nyquist-auditor (line 21)
- gsd-ui-checker (line 23)
- gsd-ui-auditor (line 24)

After editing, every line in MODEL_PROFILES must have `quality: 'opus'`. Do NOT change the balanced or budget columns.
  </action>
  <verify>
    <automated>node -e "const {MODEL_PROFILES} = require('./.claude/get-shit-done/bin/lib/model-profiles.cjs'); const all = Object.entries(MODEL_PROFILES).every(([k,v]) => v.quality === 'opus'); console.log('all_opus:', all); if(!all) { Object.entries(MODEL_PROFILES).filter(([k,v]) => v.quality !== 'opus').forEach(([k,v]) => console.log('FAIL:', k, v.quality)); process.exit(1); }"</automated>
  </verify>
  <done>All 15 agents in MODEL_PROFILES have quality: 'opus'</done>
</task>

<task type="auto">
  <name>Task 2: Update model-profiles.md reference documentation</name>
  <files>.claude/get-shit-done/references/model-profiles.md</files>
  <action>
Update the profile table (lines 7-21) so every agent in the quality column shows "opus". The 8 agents currently showing "sonnet" must be changed to "opus":
- gsd-research-synthesizer
- gsd-codebase-mapper
- gsd-verifier
- gsd-plan-checker
- gsd-integration-checker
- gsd-nyquist-auditor

Also add the 3 missing UI agents to the table (they exist in code but not in docs):
- gsd-ui-researcher | opus | sonnet | haiku | inherit
- gsd-ui-checker | opus | sonnet | haiku | inherit
- gsd-ui-auditor | opus | sonnet | haiku | inherit

Update the "Profile Philosophy" section for quality to say: "Opus for ALL agents — maximum reasoning power with no exceptions" (remove the "Sonnet for read-only verification" line).

Remove the "Why Haiku for gsd-codebase-mapper?" rationale paragraph since it no longer applies to the quality profile (it still uses haiku in budget, but the rationale specifically justified the quality column choice).
  </action>
  <verify>
    <automated>grep -c "sonnet" .claude/get-shit-done/references/model-profiles.md | xargs -I{} test {} -lt 15 && grep -c "| opus |" .claude/get-shit-done/references/model-profiles.md | xargs -I{} test {} -ge 15 && echo "PASS" || echo "FAIL: check quality column entries"</automated>
  </verify>
  <done>Documentation table shows opus for all 15 agents in quality column, all 15 agents listed, philosophy section updated</done>
</task>

</tasks>

<verification>
- node -e validates all MODEL_PROFILES entries have quality: 'opus'
- grep confirms documentation table updated
- No changes to balanced or budget columns in either file
</verification>

<success_criteria>
- All 15 GSD agents resolve to opus when quality profile is active
- Reference docs match the code exactly
- balanced and budget profiles unchanged
</success_criteria>

<output>
After completion, create `.planning/quick/260317-lcq-make-sure-all-the-gsd-agents-use-opus-wi/260317-lcq-SUMMARY.md`
</output>
