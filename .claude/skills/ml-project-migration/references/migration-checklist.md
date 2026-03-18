# ML Project Migration — Reference Checklist

## Pre-Migration Audit

```bash
# Find files > 100MB (candidates for cloud storage)
find . -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'

# Check directory sizes
du -sh model/ output/ data/ .cache/ ~/.cache/huggingface/ 2>/dev/null
```

## File Categorization Table

| Category | Typical Contents | Transfer Via |
|----------|-----------------|-------------|
| Code & configs | `*.py`, `*.sh`, `*.toml`, `*.yaml` | Git |
| Documentation | `*.md`, `*.tex`, `report/` | Git |
| Project deps | `pyproject.toml` | Git |
| Small data | `data/*.sql`, `data/*.csv` (<50 MB) | Git (or Git LFS) |
| CI/CD | `.github/workflows/` | Git |
| Dockerfile | `Dockerfile`, `.dockerignore` | Git |
| Claude config | `CLAUDE.md`, `.claude/skills/` | Git |
| Scripts | `script/setup.sh`, `script/b2-pull.sh` | Git |
| Large data | datasets >100 MB, `.db` files | Cloud storage |
| Model weights | `*.pt`, `*.bin`, `*.safetensors` | Cloud storage |
| Checkpoints | `output/*/checkpoints/` | Cloud storage |
| HF cache | `~/.cache/huggingface/` | Cloud storage or re-download |
| Experiment logs | W&B runs | Already in cloud |
| Secrets | `.env` | SCP (never Git) |
| Personal config | `.claude/memory/`, `.claude/settings.local.json` | rsync (optional) |
| Build artifacts | `__pycache__/`, `.mypy_cache/`, `.wandb/` | Never (regenerated) |

## .gitignore Template for ML Projects

```gitignore
# Model artifacts
model/
output/
checkpoint-*/
*.onnx

# Caches
__pycache__/
.mypy_cache/
.pytest_cache/
.ipynb_checkpoints/
*.pyc

# Environment & secrets
.env
.env.local
.env.*.local

# IDE/personal
.vscode/settings.json
.idea/
.claude/settings.local.json
.claude/memory/

# OS
.DS_Store
Thumbs.db

# W&B local
wandb/

# LaTeX build artifacts
*.aux
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz

# Misc
*.tar.gz
*.zip
nohup.out
*.skill
```

## B2 Command Patterns

```bash
# Upload a file
b2 file upload <bucket> <local-path> <remote-path>

# Upload all checkpoints
find output/ -name "*.pt" -type f | while read -r f; do
    b2 file upload <bucket> "$f" "$f"
done

# Download a file
b2 file download "b2://<bucket>/path/to/file" local/path/to/file

# List bucket contents
b2 ls <bucket>
```

## Post-Migration Verification

```bash
python3 -c "import torch; print(torch.cuda.is_available())"  # GPU
python3 -c "import wandb; wandb.login()"                      # W&B
claude --version                                                # Claude Code
gh auth status                                                  # GitHub CLI
ls -lh output/                                                  # model weights
```
