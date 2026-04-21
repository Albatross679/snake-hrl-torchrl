# Phase 17: Design PINN Debugging Workflow and Diagnostic Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 17-design-pinn-debugging-workflow-and-diagnostic-pipeline
**Areas discussed:** Probe PDE scope, Diagnostic integration, NTK diagnostics, Skill structure

---

## Probe PDE Scope

### How many probe PDEs?

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 probes | Heat → Advection → Burgers → Reaction-Diffusion → Simplified Cosserat | |
| First 4 only | Skip ProbePDE5 (simplified Cosserat) | |
| 3 core probes | Heat + Burgers + Reaction-Diffusion only | |

**User's choice:** Generic probe PDEs (not project-specific) + PDE system analysis for the actual system
**Notes:** User wants the approach to be generic and reusable, not dedicated to this project's Cosserat problem. But also wants diagnosis of the specific PDE system being approximated. Result: two-part approach — generic probes for pipeline validation + system analysis for the actual PDE.

### Probe trigger

| Option | Description | Selected |
|--------|-------------|----------|
| Manual pre-flight | User runs probe suite explicitly | |
| Auto before training | Runs at start of train_pinn.py unless --skip-probes | ✓ |
| Both modes | Auto by default + standalone script | |

**User's choice:** Auto before training

### PDE system diagnosis

| Option | Description | Selected |
|--------|-------------|----------|
| Both (Recommended) | Generic probes + system analysis | ✓ |
| Pipeline probes only | Just generic probe PDEs | |
| System analysis only | Skip generic probes | |

**User's choice:** Both

---

## Diagnostic Integration

### Integration approach

| Option | Description | Selected |
|--------|-------------|----------|
| Middleware class | PINNDiagnostics injected into train_pinn.py | ✓ |
| Callback/hook system | Register diagnostic hooks at training events | |
| Post-hoc analysis script | Separate script reads W&B logs after training | |

**User's choice:** Middleware class

### Failure detection aggressiveness

| Option | Description | Selected |
|--------|-------------|----------|
| W&B alerts only | Fire wandb.alert() on failure signatures | |
| Alert + auto-stop | Alerts plus automatic early stopping | |
| Log only, no alerts | Just log diagnostic metrics to W&B | ✓ |

**User's choice:** Log only, no alerts

---

## NTK Diagnostics

| Option | Description | Selected |
|--------|-------------|----------|
| Include as optional | Disabled by default, enabled with --ntk flag | |
| Defer entirely | Skip NTK in this phase | |
| Include always | Run every 50 epochs as default | ✓ |

**User's choice:** Include always
**Notes:** User wants comprehensive monitoring despite computational cost.

---

## Skill Structure

### Relationship to rl-debug

| Option | Description | Selected |
|--------|-------------|----------|
| Independent skill | Fully separate .claude/skills/pinn-debug/ | ✓ |
| Shared base + specializations | Extract common patterns into shared base | |
| Extend rl-debug skill | Add PINN sections to existing rl-debug | |

**User's choice:** Independent skill

### Decision tree depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full inline | All decision trees written directly in SKILL.md | ✓ |
| Reference diagnostics module | Skill points to source code for details | |
| Layered: summary + deep refs | Summary in SKILL.md, detailed docs in references/ | |

**User's choice:** Full inline

---

## Claude's Discretion

- Exact probe PDE implementations and pass criteria thresholds
- Specific diagnostic metric thresholds for decision tree branching
- W&B metric naming conventions
- PDE system analysis structure (class vs function)
- Number of generic probe PDEs (3-4)
- NTK computation implementation details

## Deferred Ideas

- W&B alerts and auto-stopping — may add later if manual monitoring insufficient
- Shared infrastructure between rl-debug and pinn-debug skills
- Loss landscape visualization
- Causal weighting integration
- PINNacle benchmark integration
