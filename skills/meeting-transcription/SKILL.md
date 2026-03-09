---
name: meeting-transcription
description: Transcribe meeting audio files with speaker diarization using WhisperX, creating organized meeting folders in Obsidian. Use when the user wants to transcribe a meeting recording, process audio files, create meeting notes from recordings, or mentions speaker diarization, WhisperX, or meeting transcription.
---

# Meeting Transcription

Transcribe audio recordings with speaker diarization and create structured meeting folders in the Obsidian vault.

## Prerequisites

Verify WhisperX is installed:

```bash
conda activate whisperx
python -c "import whisperx; print('WhisperX ready')"
```

If not installed, see [references/setup.md](references/setup.md).

## Workflow

### 1. Gather Information

Required:
- Audio file path (e.g., `Recording.m4a`)
- Meeting topic
- HuggingFace token (for speaker diarization)

Optional:
- Date (defaults to today)
- Participant names (for speaker labeling)
- Number of speakers (auto-detected if omitted)
- Whether to keep audio in meeting folder

### 2. Run Transcription

Execute from vault root:

```bash
conda activate whisperx
python scripts/transcribe_meeting.py "audio.m4a" \
  --topic "Meeting Topic" \
  --date "YYYY-MM-DD" \
  --participants "Person A" "Person B" \
  --speakers 2 \
  --hf-token "HF_TOKEN" \
  --keep-audio
```

### 3. Output Structure

Creates `Meetings/YYYY-MM-DD Topic Name/` containing:

| File | Description |
|------|-------------|
| `YYYY-MM-DD Topic Name.md` | Main note with MeetingNote fileClass |
| `transcript.md` | Speaker-labeled, timestamped transcript |
| `audio.m4a` | Original audio (if `--keep-audio`) |

### 4. Post-Processing

After transcription:
1. Open main meeting note in Obsidian
2. Fill in `summary` field
3. Link actual Contact notes in `participants`
4. Add action items and notes

## Script Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `audio_file` | Yes | Path to audio file |
| `--topic, -t` | Yes | Meeting topic |
| `--hf-token` | Yes | HuggingFace token |
| `--date, -d` | No | Date (YYYY-MM-DD), default: today |
| `--participants, -p` | No | Speaker names in order |
| `--speakers, -s` | No | Number of speakers (auto-detect if omitted) |
| `--keep-audio` | No | Copy audio to meeting folder |
| `--device` | No | `cpu` or `cuda`, default: cpu |
| `--duration` | No | Manual duration (e.g., "45 min") |

## Transcript Format

```markdown
## Speaker Legend
- SPEAKER_00: Person A
- SPEAKER_01: Person B

---

## Transcript

**[00:00] Person A:** Hello, thank you for meeting...

**[00:12] Person B:** Of course, let's discuss...
```

## MeetingNote FileClass Fields

The generated meeting note uses these fields:

| Field | Type | Description |
|-------|------|-------------|
| `fileClass` | - | Always `MeetingNote` |
| `date` | Date | Meeting date |
| `type` | Select | One-on-one or Group meeting |
| `status` | Select | Set to Completed |
| `participants` | MultiLink | Links to Contact notes |
| `summary` | Input | Fill after review |
| `duration` | Input | Auto-detected or manual |
| `has_transcript` | Boolean | true |
| `has_audio` | Boolean | true if `--keep-audio` |
