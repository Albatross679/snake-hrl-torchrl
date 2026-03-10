---
phase: 6
slug: write-research-report-in-latex
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual + shell checks (LaTeX document — no pytest) |
| **Config file** | `report/Makefile` (Wave 0 creates this) |
| **Quick run command** | `make check` (latexmk dry-run or section existence checks) |
| **Full suite command** | `make pdf` (full compilation via Docker/latexmk) |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `make check` (verify .tex file compiles without fatal errors)
- **After every plan wave:** Run `make pdf` (verify full PDF builds successfully)
- **Before `/gsd:verify-work`:** Full suite must be green (PDF builds, all sections present)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | Setup | compilation | `ls report/report.tex` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 0 | Setup | compilation | `make check` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | Background | structural | `grep -c "\\\\section" report/report.tex` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 1 | Methods | structural | `grep "Methods" report/report.tex` | ❌ W0 | ⬜ pending |
| 6-04-01 | 04 | 1 | Results | structural | `grep "placeholder" report/report.tex` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `report/report.tex` — main LaTeX document created
- [ ] `report/references.bib` — bibliography file created
- [ ] `report/Makefile` — build system (Docker + latexmk) created
- [ ] Compilation environment verified (Docker image available or texlive installed)

*Wave 0 establishes the compilation infrastructure before any content tasks run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Mathematical notation correct | Background section | Human review of PDEs | Read Cosserat rod equations; verify match with `knowledge/surrogate-mathematical-formulation.md` |
| Citations complete and accurate | Related Work section | Bibliography verification | Spot-check 5 random citations in PDF against knowledge/ source files |
| Placeholder macros render visually | Results sections | Visual PDF review | Open PDF, verify placeholders show in yellow/highlighted boxes |
| Section flow and readability | Full document | Writing quality | Read full document aloud; check logical transitions |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
