#!/usr/bin/env bash
#
# Download model weights and data from Backblaze B2.
#
# Prerequisites:
#   - b2 CLI installed (pip install b2)
#   - B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY env vars set
#
# Usage:
#   ./script/b2-pull.sh
#
# Add new files by appending:
#   b2 file download "b2://bucket/path" local/path

set -euo pipefail

echo "==> Authenticating with Backblaze B2..."
b2 account authorize "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY"

echo "==> Downloading files from B2..."

# -- Add b2 file download lines below --
# Example:
# b2 file download "b2://mlworkflow/snake-hrl/model/checkpoint.pt" model/checkpoint.pt

echo "==> B2 pull complete."
