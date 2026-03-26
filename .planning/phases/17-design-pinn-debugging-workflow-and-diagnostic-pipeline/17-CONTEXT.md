# Phase 17: Design PINN Debugging Workflow and Diagnostic Pipeline - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build diagnostic instrumentation and a Claude Code skill for systematic PINN training failure detection. Covers: probe PDE pre-flight validation, runtime diagnostic metrics (loss ratios, gradient norms, residual statistics, NTK eigenvalues), decision tree for fault isolation, and a pinn-debug Claude Code skill. Does NOT cover: new PINN training capabilities, model architecture changes, or training campaign execution.

</domain>

<decisions>
## Implementation Decisions

### Probe PDE scope
- **D-01:** Generic probe PDEs — not project-specific. Covers fundamental PINN capabilities: data fitting, BC/IC enforcement, nonlinear PDE + loss balancing, multi-scale/Fourier features
- **D-02:** Include PDE system analysis step that checks the user's actual PDE system: per-term residual magnitudes, nondimensionalization quality, stiffness indicators, condition number
- **D-03:** Both generic probes AND system analysis — probes validate pipeline mechanics, system analysis validates the specific problem setup
- **D-04:** Auto-run before training by default (integrated into train_pinn.py), with `--skip-probes` flag to opt out

### Diagnostic integration
- **D-05:** Middleware class pattern — `PINNDiagnostics` class injected into train_pinn.py, mirroring `src/trainers/diagnostics.py` pattern
- **D-06:** Non-invasive: trainer calls `diagnostics.log_step()` each epoch to compute and log metrics
- **D-07:** Log-only to W&B — no automated alerts (no `wandb.alert()`), no auto-stopping. User monitors dashboard manually
- **D-08:** Diagnostic metrics: loss component ratios, per-loss-term gradient norms, residual spatial distribution, per-component physics violation magnitudes, ReLoBRaLo weight health

### NTK diagnostics
- **D-09:** Include NTK eigenvalue computation as default diagnostic — runs every 50 epochs
- **D-10:** Parameter subsampling (n_params_sample=500) for tractability on 512x4 models
- **D-11:** Logs: eigenvalue_max, eigenvalue_min, condition_number, spectral_decay_rate

### Skill structure
- **D-12:** Independent skill at `.claude/skills/pinn-debug/` — fully separate from rl-debug, no shared infrastructure
- **D-13:** Same 4-phase structure as rl-debug: (1) probe validation, (2) dashboard diagnostics, (3) decision tree for "loss not decreasing", (4) physics-specific sub-tree
- **D-14:** Full decision tree inline in SKILL.md — all failure signatures, diagnostic checks, and remediation steps written directly so Claude can reason from the skill without reading source code
- **D-15:** references/ folder for detailed failure mode documentation (Wang et al. NTK analysis, Krishnapriyan failure taxonomy, etc.)

### Claude's Discretion
- Exact probe PDE implementations (analytical solutions, pass criteria thresholds)
- Specific diagnostic metric thresholds for decision tree branching
- W&B metric naming conventions (diagnostics/ prefix established in RL diagnostics)
- How to structure the PDE system analysis (class vs function vs standalone script)
- Number of generic probe PDEs (3-4 covering the key capabilities)
- NTK computation implementation details (Jacobian subsampling strategy)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing diagnostic pattern (mirror this)
- `src/trainers/diagnostics.py` — RL TrainingDiagnostics middleware class, check_alerts(), explained variance, action stats. Mirror this pattern for PINN diagnostics.
- `.claude/skills/rl-debug/SKILL.md` — RL debug skill structure: 4-phase diagnostic process, probe envs, decision trees. Mirror for pinn-debug skill.
- `src/trainers/probe_envs.py` — RL probe environment implementations. Analogous pattern for probe PDEs.

### Existing PINN implementation (integrate with this)
- `src/pinn/train_pinn.py` — Current PINN training loop. Diagnostics middleware hooks into this.
- `src/pinn/loss_balancing.py` — ReLoBRaLo implementation. Diagnostics monitors weight health.
- `src/pinn/collocation.py` — Collocation sampling with RAR. Diagnostics monitors residual distribution.
- `src/pinn/physics_residual.py` — CosseratRHS computation. PDE system analysis checks this.
- `src/pinn/models.py` — DD-PINN model architecture. NTK diagnostics computes on this.
- `src/pinn/ansatz.py` — Damped sinusoidal ansatz. Probe PDE5-equivalent tests ansatz pipeline.
- `src/pinn/nondim.py` — Nondimensionalization. PDE system analysis validates scaling.

### Phase 17 research
- `.planning/phases/17-design-pinn-debugging-workflow-and-diagnostic-pipeline/17-RESEARCH.md` — Comprehensive research: failure mode taxonomy, probe PDE designs, diagnostic metrics, decision trees, code examples, literature references

### PINN failure mode literature
- Wang et al. "When and Why PINNs Fail to Train" (NeurIPS 2021) — NTK eigenvalue analysis, convergence rate mismatch
- Krishnapriyan et al. "Characterizing Possible Failure Modes in PINNs" (NeurIPS 2021) — Failure taxonomy, probe PDE methodology
- Bischof & Kraus (arXiv:2110.09813) — ReLoBRaLo algorithm (already implemented)

### Phase 13 context (prior decisions)
- `.planning/phases/13-implement-pinn-and-dd-pinn-surrogate-models/13-CONTEXT.md` — DD-PINN implementation decisions, code organization, physics residual depth

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/trainers/diagnostics.py`: RL TrainingDiagnostics class — mirror structure for PINNDiagnostics (deque-based history, compute methods, W&B logging)
- `src/trainers/probe_envs.py`: RL probe environments — mirror pattern for probe PDEs (progressive complexity, pass/fail criteria)
- `.claude/skills/rl-debug/SKILL.md`: Skill template with 4-phase structure, decision trees, reference docs
- `src/pinn/collocation.py`: Sobol sampling already available — reuse for probe PDE collocation points
- `scipy.stats.qmc.Sobol`: Already used in codebase for quasi-random sampling

### Established Patterns
- Diagnostic middleware: non-invasive class with `log_step()` called per epoch, uses `deque(maxlen=N)` for rolling history
- W&B metric namespacing: `diagnostics/` prefix for diagnostic metrics (established in RL diagnostics)
- Probe validation: progressive complexity probes that each test one additional capability
- Skill structure: SKILL.md with frontmatter, phases, decision trees, references/ folder

### Integration Points
- `src/pinn/train_pinn.py` — inject PINNDiagnostics middleware, add probe pre-flight call
- `src/pinn/diagnostics.py` — NEW module for PINNDiagnostics class
- `src/pinn/probe_pdes.py` — NEW module for probe PDE suite
- `.claude/skills/pinn-debug/SKILL.md` — NEW skill file
- `.claude/skills/pinn-debug/references/` — NEW reference docs folder

</code_context>

<specifics>
## Specific Ideas

- Generic probe PDEs should test fundamental PINN capabilities without being tied to Cosserat rod physics — makes the debugging tool reusable for any PINN project
- PDE system analysis is the project-specific part: checks the actual CosseratRHS for nondimensionalization quality, term magnitude balance, and stiffness
- The decision tree should be readable as-is in SKILL.md — Claude should be able to diagnose failures from the skill alone without reading source code
- NTK diagnostics included as default (not optional) because the user wants comprehensive monitoring, even at computational cost

</specifics>

<deferred>
## Deferred Ideas

- W&B alerts and auto-stopping — decided against for this phase, could add in future if manual monitoring proves insufficient
- Shared infrastructure between rl-debug and pinn-debug skills — decided against, keeping independent. Could refactor later if patterns converge
- Loss landscape visualization — mentioned in research but not discussed. ~20 lines of code, could add as follow-up
- Causal weighting integration — research flags as diagnostic recommendation in decision tree, not a training default. Implementation deferred to training phase
- PINNacle benchmark integration — 20+ PDE benchmark suite, overkill for diagnostic probes

</deferred>

---

*Phase: 17-design-pinn-debugging-workflow-and-diagnostic-pipeline*
*Context gathered: 2026-03-26*
