#!/usr/bin/env bash
# setup.sh - Zotero CLI Skill - One-Click Setup
#
# Installs zotcli via pipx, applies the Python 3.14 compatibility patch,
# and configures the Zotero API connection.
#
# Prerequisites:
#   - ZOTERO_API_KEY and ZOTERO_LIBRARY_ID set in environment (e.g. ~/.bashrc)
#   - pipx installed (pip install pipx)
#
# Usage:
#   bash setup.sh           # Install and configure
#   bash setup.sh --check   # Only verify installation (no changes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

# Use XDG config dir on Linux, macOS Application Support as fallback
if [[ "$(uname)" == "Darwin" ]]; then
  CONFIG_DIR="${HOME}/Library/Application Support/zotcli"
else
  CONFIG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/zotcli"
fi
CONFIG_FILE="${CONFIG_DIR}/config.ini"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
fi

# --- Helpers ---
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; }
skip() { echo "  [SKIP] $1"; }

check_command() {
  if command -v "$1" &>/dev/null; then
    ok "$1 found at $(command -v "$1")"
    return 0
  else
    fail "$1 not found"
    return 1
  fi
}

echo "========================================"
echo "  Zotero CLI Skill - Setup"
echo "========================================"
echo ""

# --- Phase 1: Check prerequisites ---
echo ":: Phase 1 - Prerequisites"

MISSING=0

if ! check_command pipx; then
  if [[ "$CHECK_ONLY" == false ]]; then
    echo "  Installing pipx via pip..."
    pip install --user pipx 2>&1
    python3 -m pipx ensurepath 2>&1
    export PATH="${HOME}/.local/bin:${PATH}"
    if check_command pipx; then
      ok "pipx installed successfully"
    else
      fail "pipx installation failed"
      MISSING=1
    fi
  else
    MISSING=1
  fi
fi

if [[ -z "${ZOTERO_API_KEY:-}" ]]; then
  fail "ZOTERO_API_KEY not set in environment"
  MISSING=1
else
  ok "ZOTERO_API_KEY is set"
fi

if [[ -z "${ZOTERO_LIBRARY_ID:-}" ]]; then
  fail "ZOTERO_LIBRARY_ID not set in environment"
  MISSING=1
else
  ok "ZOTERO_LIBRARY_ID is set (${ZOTERO_LIBRARY_ID})"
fi

if [[ "$CHECK_ONLY" == true && $MISSING -eq 1 ]]; then
  echo ""
  echo "Some prerequisites are missing. Run without --check to install."
  exit 1
fi

echo ""

# --- Phase 2: Install zotero-cli ---
echo ":: Phase 2 - Install zotero-cli"

if command -v zotcli &>/dev/null; then
  ok "zotcli already installed ($(zotcli --help 2>&1 | head -1 || echo 'installed'))"
else
  if [[ "$CHECK_ONLY" == true ]]; then
    fail "zotcli not installed"
  else
    echo "  Installing zotero-cli via pipx..."
    pipx install zotero-cli
    if command -v zotcli &>/dev/null; then
      ok "zotcli installed successfully"
    else
      fail "zotcli installation failed"
      exit 1
    fi
  fi
fi

echo ""

# --- Phase 3: Apply Python 3.14 compatibility patch ---
echo ":: Phase 3 - Python 3.14 compatibility patch"

VENV_SITE_PACKAGES="$(pipx runpip zotero-cli show -f zotero-cli 2>/dev/null | grep "Location:" | awk '{print $2}')"
TARGET_INDEX="${VENV_SITE_PACKAGES}/zotero_cli/index.py"
PATCH_SOURCE="${PATCHES_DIR}/index.py"

if [[ ! -f "$TARGET_INDEX" ]]; then
  fail "Could not find zotero_cli/index.py at ${TARGET_INDEX}"
  echo "  You may need to adjust the path for your Python version."
elif [[ "$CHECK_ONLY" == true ]]; then
  if grep -q "named parameter binding" "$TARGET_INDEX" 2>/dev/null || diff -q "$PATCH_SOURCE" "$TARGET_INDEX" &>/dev/null; then
    ok "Python 3.14 patch already applied"
  else
    fail "Python 3.14 patch not applied"
  fi
else
  cp "$PATCH_SOURCE" "$TARGET_INDEX"
  ok "Patched index.py for Python 3.14 named parameter binding"
fi

echo ""

# --- Phase 4: Configure zotcli ---
echo ":: Phase 4 - Configuration"

if [[ -f "$CONFIG_FILE" ]]; then
  ok "Config file exists at ${CONFIG_FILE}"
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "  Current config:"
    sed 's/api_key = .*/api_key = ****/' "$CONFIG_FILE" | while read -r line; do echo "    $line"; done
  fi
else
  if [[ "$CHECK_ONLY" == true ]]; then
    fail "Config file not found at ${CONFIG_FILE}"
  else
    echo "  Creating config directory..."
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_FILE" <<CONF
[zotcli]
api_key = ${ZOTERO_API_KEY}
library_id = ${ZOTERO_LIBRARY_ID}
sync_method = zotcloud
sync_interval = 300
note_format = markdown
CONF
    ok "Config written to ${CONFIG_FILE}"
  fi
fi

echo ""

# --- Phase 5: Initial sync ---
echo ":: Phase 5 - Library sync"

if [[ "$CHECK_ONLY" == true ]]; then
  skip "Sync check (run without --check to sync)"
else
  echo "  Syncing Zotero library index..."
  if zotcli sync 2>&1; then
    ok "Library synced successfully"
  else
    fail "Sync failed (check API key and network)"
  fi
fi

echo ""
echo "========================================"
if [[ "$CHECK_ONLY" == true ]]; then
  echo "  Verification complete."
else
  echo "  Setup complete! Try: zotcli query \"reinforcement learning\""
fi
echo "========================================"
