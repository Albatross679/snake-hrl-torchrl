---
name: zotero-cli
description: Search, read, and manage the user's Zotero reference library via the zotcli CLI. Use when the user mentions Zotero, references, papers, citations, bibliography, literature, or wants to search/add/export academic papers.
---

# Zotero CLI

Manage the user's Zotero library through the `zotcli` command-line tool.

## Setup

- **Binary**: `zotcli` (installed via `pipx install zotero-cli`)
- **Config**: `~/.config/zotcli/config.ini` (Linux) or `~/Library/Application Support/zotcli/config.ini` (macOS)
- **Env vars**: `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID` (in `~/.bashrc` or `~/.zshrc`)
- **Library type**: `user` (personal library, ID `15060502`)
- **Python 3.14 compat patch** applied to `index.py` (named parameter binding fix)

## Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `zotcli query "<terms>"` | Full-text search across library | `zotcli query "reinforcement learning"` |
| `zotcli read <KEY>` | Read/display an item's attachment | `zotcli read ENCCGYQN` |
| `zotcli add-note <KEY>` | Add a new note to an item | `zotcli add-note ENCCGYQN` |
| `zotcli edit-note <KEY>` | Edit an existing note | `zotcli edit-note ENCCGYQN` |
| `zotcli export-note <KEY>` | Export a note | `zotcli export-note ENCCGYQN` |
| `zotcli sync` | Re-sync the local search index | `zotcli sync` |

## Search Tips

- The search index uses SQLite FTS4
- Wildcard: `*` (but empty result for bare `*`)
- Boolean: `term1 OR term2`
- To list all items, use a broad alphabetic OR query:
  ```
  zotcli query "a OR b OR c OR d OR e OR f OR g OR h OR i OR j OR k OR l OR m OR n OR o OR p OR q OR r OR s OR t OR u OR v OR w OR x OR y OR z"
  ```
- Output format: `[KEY] Author(s): Title (Date)`

## Workflow

### Searching for papers
1. Run `zotcli query "<search terms>"`
2. Parse output lines: `[KEY] Author: Title (Date)`
3. Present results in a readable table or list

### Reading a paper's details
1. Get the item key from a search result (e.g., `ENCCGYQN`)
2. Run `zotcli read <KEY>` to access the attachment

### Adding notes to papers
1. Search for the paper to get its key
2. Run `zotcli add-note <KEY>` with note content

### Exporting citations
1. Search for relevant papers
2. Collect item keys from results
3. Use the Zotero API directly for BibTeX/RIS export if needed:
   ```bash
   curl -s -H "Zotero-API-Key: $ZOTERO_API_KEY" \
     "https://api.zotero.org/users/$ZOTERO_LIBRARY_ID/items/<KEY>?format=bibtex"
   ```

## Library Overview

The user's library (~37 items) covers:
- **Snake robotics / soft robots**: CPG locomotion, RL-guided navigation, soft robot modeling
- **Reinforcement learning**: SAC, Q-learning, hierarchical RL, neuroevolution
- **Recommendation systems**: collaborative filtering, deep learning for e-commerce
- **Path planning**: warehouse robots, multi-robot systems
- **Foundational ML**: Adam optimizer, memorization in deep networks

## Common Tasks

| User says... | Action |
|--------------|--------|
| "search my Zotero for X" | `zotcli query "X"` |
| "what papers do I have on X" | `zotcli query "X"` and summarize |
| "list all my papers" | Broad OR query (see Search Tips) |
| "get citation for X" | Search, then curl Zotero API with `format=bibtex` |
| "add a note to paper X" | Search for key, then `zotcli add-note <KEY>` |
| "sync my library" | `zotcli sync` |
