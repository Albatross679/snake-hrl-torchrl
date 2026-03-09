# Temporary File Management

Temporary files live in the `Temporary/` folder and use `fileClass: Temporary`.

## Dropping In

Users may drop in temporary entries in various forms:

- **Text message** — a raw block of text, a quick note, a pasted snippet, or a list of items
- **Image** — a screenshot, photo of handwriting, whiteboard capture, etc. Read the image content and transcribe/extract the relevant information into the note body.
- **Multiple items at once** — a single message may contain several distinct entries. Create a separate file for each.

For each entry, infer the appropriate `type` from the content. When ambiguous, use `Other`.

Place files in `Temporary/`. Always set:
1. `fileClass: Temporary` (first property)
2. `type` (required — one of: Message, Status Update, 3P Update, FAQ, Newsletter, Other)
3. Standard properties: `date_created`, `date_modified`, `tags`

### Drop-in Report

After processing a drop-in, always end your response with a summary:

```
Temporary: created N entries
- filename.md (Type)
- filename.md (Type)
```

## Disposal

### Disposal Mapping by Type

| Type | Default disposal | Notes |
|------|-----------------|-------|
| Message | Delete | Delete after sending or copying out |
| Status Update | Delete | Delete after distribution |
| 3P Update | Delete | Delete after distribution |
| FAQ | Reclassify | Reclassify to `Concept` in `Knowledge/` if reusable; otherwise delete |
| Newsletter | Delete | Delete after distribution |
| Other | Ask user | Could be any action — ask which disposal to apply |

### Disposal Report

After disposing, always end your response with a summary:

```
Temporary: disposed N entries
- filename.md → deleted
- filename.md → reclassified as FileClass (moved to Folder/)
- filename.md → created N Todoist tasks, deleted
- filename.md → merged into existing-note.md, deleted
```
