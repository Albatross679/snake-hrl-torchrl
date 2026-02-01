# LaTeX Auto-Cleanup Setup Guide

This guide explains how to automatically delete LaTeX auxiliary files (`.aux`, `.log`, `.out`, `.synctex.gz`, etc.) after each compilation in VS Code/Cursor with the LaTeX Workshop extension.

## Prerequisites

- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension installed
- A working LaTeX distribution (e.g., TeX Live)

## Setup Steps

### Step 1: Install latexmk

LaTeX Workshop uses `latexmk` for cleanup. Install it if not already present:

```bash
# Ubuntu/Debian
sudo apt-get install -y latexmk

# macOS (with Homebrew)
brew install latexmk

# Or via TeX Live
tlmgr install latexmk
```

### Step 2: Create `.latexmkrc` Configuration

Create a file named `.latexmkrc` in your project root directory with the following content:

```perl
# Additional files to clean with latexmk -c
$clean_ext = "synctex.gz synctex.gz(busy) run.xml";
```

This tells `latexmk` to also remove `.synctex.gz` files during cleanup (not included by default).

### Step 3: Configure LaTeX Workshop Settings

Add the following to your `.vscode/settings.json` (workspace settings) or user settings:

```json
{
  "latex-workshop.latex.autoClean.run": "onBuilt"
}
```

**Options for `autoClean.run`:**
- `"never"` - Never auto-clean (default)
- `"onFailed"` - Clean only when build fails
- `"onBuilt"` - Clean after every successful build

### Step 4: Reload the Window

After making settings changes, reload VS Code/Cursor:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Reload Window" and press Enter

## What Gets Cleaned

By default, `latexmk -c` removes:
- `.aux` - Auxiliary file for cross-references
- `.log` - Compilation log
- `.out` - Hyperref bookmarks
- `.fls` - File list
- `.fdb_latexmk` - latexmk database
- `.bbl`, `.blg` - Bibliography files
- `.toc`, `.lof`, `.lot` - Table of contents, figures, tables

With the `.latexmkrc` configuration above, it also removes:
- `.synctex.gz` - SyncTeX file for PDF-source synchronization

## Files That Are Kept

- `.tex` - Your source files
- `.pdf` - The compiled output
- `.bib` - Bibliography source files

## Optional: Hide Files in Explorer

If you prefer to keep files on disk but hide them from the file explorer, add to your `settings.json`:

```json
{
  "files.exclude": {
    "**/*.aux": true,
    "**/*.log": true,
    "**/*.out": true,
    "**/*.synctex.gz": true,
    "**/*.fls": true,
    "**/*.fdb_latexmk": true
  }
}
```

## Troubleshooting

### "spawn latexmk ENOENT" error
`latexmk` is not installed. Run `sudo apt-get install -y latexmk`.

### `.synctex.gz` not being deleted
Make sure you have a `.latexmkrc` file in your project directory with `$clean_ext = "synctex.gz";`.

### Settings not taking effect
Reload the window after changing settings (`Ctrl+Shift+P` → "Reload Window").

## Note on SyncTeX

Deleting `.synctex.gz` disables the "click in PDF to jump to source" feature. If you need this feature, remove `synctex.gz` from the `.latexmkrc` file.
