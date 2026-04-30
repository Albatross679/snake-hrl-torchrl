# VS Code Settings Preferences

Preferred `.vscode/settings.json` for new projects. Adapt paths and include only relevant sections.

## Base Settings (Always Apply)

```json
{
    "claude-code.reasoningEffort": "max"
}
```

## LaTeX (When Project Has a Report)

Uses **Tectonic** as the LaTeX build engine via LaTeX Workshop.

```json
{
    "latex-workshop.latex.tools": [
        {
            "name": "tectonic",
            "command": "tectonic",
            "args": [
                "%DOC%.tex"
            ],
            "env": {}
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "tectonic",
            "tools": [
                "tectonic"
            ]
        }
    ],
    "latex-workshop.latex.recipe.default": "tectonic"
}
```

**Note:** The `command` path for tectonic may vary by machine. Use `which tectonic` to find the correct path, or just use `"tectonic"` if it's on PATH.

## How to Apply

1. Create `.vscode/settings.json` in the new project root
2. Start with the base settings
3. Merge in conditional sections (LaTeX, etc.) as needed
4. Adjust machine-specific paths (e.g., tectonic binary location)
