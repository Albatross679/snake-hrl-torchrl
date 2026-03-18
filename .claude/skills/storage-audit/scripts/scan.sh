#!/usr/bin/env bash
# Storage audit scanner — outputs structured report for Claude to parse.
# Usage: bash scan.sh [root_dir]  (default: $HOME)
set -euo pipefail

ROOT="${1:-$HOME}"
SEP="================================================================"

echo "$SEP"
echo "SECTION: DISK_OVERVIEW"
echo "$SEP"
df -h / /home 2>/dev/null | head -20

echo ""
echo "$SEP"
echo "SECTION: LARGE_DIRS_HOME"
echo "$SEP"
du -sh "$ROOT"/*/ 2>/dev/null | sort -rh | head -30

echo ""
echo "$SEP"
echo "SECTION: LARGE_FILES_100M"
echo "$SEP"
find "$ROOT" -type f -size +100M 2>/dev/null \
  | grep -v '\.vscode-server/' \
  | grep -v '\.ssh/' \
  | head -60

echo ""
echo "$SEP"
echo "SECTION: CACHE_SIZES"
echo "$SEP"
for d in "$ROOT/.cache/pip" "$ROOT/.cache/huggingface" "$ROOT/.cache/wandb" \
         "$ROOT/.cache/torch" "$ROOT/.cache/conda" "$ROOT/.cache/yarn" \
         "$ROOT/.cache/npm" "$ROOT/.cache/go-build" "$ROOT/.cache/bazel"; do
  if [ -d "$d" ]; then
    du -sh "$d" 2>/dev/null
  fi
done

echo ""
echo "$SEP"
echo "SECTION: GIT_REPOS"
echo "$SEP"
# Find all git repos under home (max depth 3) and report size
find "$ROOT" -maxdepth 3 -name ".git" -type d 2>/dev/null | while read gitdir; do
  repo=$(dirname "$gitdir")
  size=$(du -sh "$repo" 2>/dev/null | cut -f1)
  origin=$(git -C "$repo" remote get-url origin 2>/dev/null || echo "no-remote")
  echo "$size  $repo  ($origin)"
done | sort -rh | head -20

echo ""
echo "$SEP"
echo "SECTION: OUTPUT_TRAINING_DIRS"
echo "$SEP"
# Find common ML output directories
for pattern in "output" "checkpoints" "runs" "wandb" "mlruns" "lightning_logs"; do
  find "$ROOT" -maxdepth 4 -type d -name "$pattern" 2>/dev/null | while read d; do
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    count=$(find "$d" -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "$size  $d  (${count} subdirs)"
  done
done | sort -rh | head -30

echo ""
echo "$SEP"
echo "SECTION: ZOMBIE_FILES"
echo "$SEP"
# Find common zombie/temp files: core dumps, .tmp, nohup.out, *.pyc caches
find "$ROOT" -maxdepth 4 \( \
  -name "core" -o -name "core.*" -o -name "*.tmp" -o -name "nohup.out" \
  -o -name "*.swp" -o -name "*.swo" -o -name ".DS_Store" \
  -o -name "__pycache__" -type d \
  \) 2>/dev/null | head -30

echo ""
echo "$SEP"
echo "SECTION: DUPLICATE_REPOS"
echo "$SEP"
# Detect repos sharing the same remote origin
declare -A seen_origins
find "$ROOT" -maxdepth 3 -name ".git" -type d 2>/dev/null | while read gitdir; do
  repo=$(dirname "$gitdir")
  origin=$(git -C "$repo" remote get-url origin 2>/dev/null || echo "")
  if [ -n "$origin" ]; then
    echo "$origin  $repo"
  fi
done | sort | uniq -D -f0 2>/dev/null || true

echo ""
echo "$SEP"
echo "SECTION: SCAN_COMPLETE"
echo "$SEP"
