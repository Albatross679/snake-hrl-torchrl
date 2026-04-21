---
name: storage-audit
description: >
  Audit machine storage and clean up disk space. Scans for large files, caches (pip, HuggingFace, W&B, torch, conda),
  model checkpoints, training output directories, duplicate git repos, and zombie/temp files.
  Presents categorized findings with sizes, then asks user for confirmation before deleting anything.
  Use when: (1) User asks to "check disk space", "clean up storage", "audit storage", "free up space",
  "list big files", "what's taking up space", (2) User mentions disk is full or running low,
  (3) User wants to remove old training runs, caches, or stale artifacts from the machine.
---

# Storage Audit & Cleanup

## Workflow

### Phase 1: Scan

Run the scan script to collect storage data:

```bash
bash <skill-path>/scripts/scan.sh
```

Parse each `SECTION:` block from the output. Build a summary table covering:

| Category | What to report |
|---|---|
| Disk overview | Total/used/available, usage percentage |
| Large directories | Top dirs under `$HOME` by size |
| Caches | pip, HuggingFace, W&B, torch, conda, npm, yarn, go-build, bazel |
| Training outputs | `output/`, `checkpoints/`, `runs/`, `wandb/`, `mlruns/`, `lightning_logs/` dirs |
| Large files (>100M) | Individual files, grouped by category (model weights, CUDA libs, git LFS, etc.) |
| Git repos | All repos with sizes, flag duplicates sharing same remote origin |
| Zombie files | `__pycache__/`, `*.tmp`, `*.swp`, `core.*`, `nohup.out`, `.DS_Store` |

Present findings as a ranked table sorted by size (largest first).

### Phase 2: Recommend

Categorize cleanup candidates into:

1. **Safe to delete** — caches (`pip cache purge`, `~/.cache/huggingface/`, `~/.cache/wandb/`), zombie files, empty/failed training runs (4K dirs with no checkpoints)
2. **Likely safe** — old training runs where a better run exists, duplicate repos, stale W&B local logs
3. **Ask first** — model checkpoints that may be the only copy, large git LFS objects, anything under the project's working directory

For training output directories: identify which runs have actual model checkpoints vs. empty/crashed runs. For runs with checkpoints, check W&B summaries or metrics logs to find the best run by primary metric and flag the rest as cleanup candidates.

### Phase 3: Confirm & Execute

Use the `AskUserQuestion` tool to present cleanup options and get explicit confirmation. Structure as:

- One question per cleanup category (safe caches, training runs, zombie files, duplicate repos, etc.)
- Show what will be deleted and how much space will be freed
- Include a "Skip" option for each category

**NEVER delete without user confirmation.** After each approved deletion, report bytes freed.

After all deletions, run `df -h /` and show the before/after comparison.

## Safe-list (NEVER delete)

These paths are always protected regardless of what the scan finds:

- `~/.vscode-server/` — VS Code server, extensions, settings
- `~/.ssh/` — SSH keys
- `~/.gitconfig` — Git identity
- `~/.local/lib/` — Installed Python packages (pip site-packages, CUDA libs)
- All paths listed in CLAUDE.md's "File Protection" section (data/, evaluate.py, etc.)

## Cleanup Commands Reference

| Target | Command |
|---|---|
| pip cache | `pip cache purge` |
| HuggingFace cache | `rm -rf ~/.cache/huggingface/` |
| W&B cache | `rm -rf ~/.cache/wandb/` |
| W&B local logs | `rm -rf <project>/wandb/` |
| Torch cache | `rm -rf ~/.cache/torch/` |
| Single output dir | `rm -rf <path>` |
| All __pycache__ | `find <root> -type d -name __pycache__ -exec rm -rf {} +` |
| Temp/swap files | `find <root> -type f \( -name "*.tmp" -o -name "*.swp" -o -name "*.swo" \) -delete` |
