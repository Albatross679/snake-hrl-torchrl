# ML Project Migration — Reference Checklist

## Pre-Migration Audit

```bash
# Find files > 100MB (candidates for cloud storage)
find . -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'

# Check directory sizes
du -sh model/ output/ data/ .cache/ ~/.cache/huggingface/ 2>/dev/null
```

## File Categorization — Detailed Examples

See SKILL.md for the 4-bucket summary. Below are common file types and where they fall:

| File pattern | Bucket | Notes |
|-------------|--------|-------|
| `*.py`, `*.sh`, `*.toml`, `*.yaml` | Git | Code & configs |
| `*.md`, `*.tex`, `report/` | Git | Documentation |
| `data/*.sql`, `data/*.csv` (<50 MB) | Git | Or Git LFS |
| `.github/workflows/` | Git | CI/CD |
| `Dockerfile`, `.dockerignore` | Git | |
| `CLAUDE.md`, `.claude/skills/` | Git | |
| `script/setup.sh`, `script/b2-pull.sh` | Git | |
| datasets >100 MB, `.db` files | Cloud storage | |
| `*.pt`, `*.bin`, `*.safetensors` | Cloud storage | Model weights |
| `output/*/checkpoints/` | Cloud storage | |
| `~/.cache/huggingface/` | Cloud storage | Or re-download |
| W&B runs | Already in cloud | |
| `.env` | Manual (SCP) | Never Git |
| `.claude/memory/`, `.claude/settings.local.json` | Manual (rsync) | Optional |
| `__pycache__/`, `.mypy_cache/`, `.wandb/` | Skip | Regenerated |

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
