#!/usr/bin/env python3
"""
Meeting Transcription Script using WhisperX

This script transcribes audio files with speaker diarization and creates
a structured meeting folder in the Obsidian vault.

Prerequisites:
1. Create conda environment:
   conda create -n whisperx python=3.10 -y
   conda activate whisperx

2. Install WhisperX:
   pip install git+https://github.com/m-bain/whisperX.git

3. Install ffmpeg:
   brew install ffmpeg

4. Accept Pyannote terms at:
   https://huggingface.co/pyannote/speaker-diarization-3.1

5. Get a HuggingFace token from https://huggingface.co/settings/tokens

Usage:
    python transcribe_meeting.py "audio.m4a" \
        --topic "Weekly Standup" \
        --date "2026-02-05" \
        --participants "Person A" "Person B" \
        --speakers 2 \
        --hf-token "YOUR_TOKEN" \
        --keep-audio
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import whisperx

# Get vault root (parent of scripts folder)
VAULT_ROOT = Path(__file__).parent.parent
MEETINGS_DIR = VAULT_ROOT / "Meetings"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_audio(
    audio_path: str,
    hf_token: str,
    num_speakers: int = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> tuple[dict, dict]:
    """
    Transcribe audio with WhisperX and perform speaker diarization.

    Args:
        audio_path: Path to the audio file
        hf_token: HuggingFace token for pyannote
        num_speakers: Number of speakers (optional, auto-detected if None)
        device: Device to use ("cpu" or "cuda")
        compute_type: Compute type ("int8" for CPU, "float16" for GPU)

    Returns:
        Tuple of (transcription result, diarization segments)
    """
    print(f"Loading audio from {audio_path}...")
    audio = whisperx.load_audio(audio_path)

    print("Loading Whisper model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=16)

    print("Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    print("Performing speaker diarization...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )
    diarize_segments = diarize_model(
        audio, min_speakers=num_speakers, max_speakers=num_speakers
    )

    print("Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result, diarize_segments


def generate_transcript_md(
    result: dict,
    participant_names: list[str] = None,
) -> str:
    """
    Generate markdown transcript from WhisperX result.

    Args:
        result: WhisperX transcription result with speaker assignments
        participant_names: Optional list of participant names to map to speakers

    Returns:
        Markdown formatted transcript
    """
    segments = result.get("segments", [])

    # Collect unique speakers
    speakers = sorted(set(
        seg.get("speaker", "UNKNOWN")
        for seg in segments
        if seg.get("speaker")
    ))

    # Create speaker mapping
    speaker_map = {}
    if participant_names:
        for i, speaker in enumerate(speakers):
            if i < len(participant_names):
                speaker_map[speaker] = participant_names[i]
            else:
                speaker_map[speaker] = speaker
    else:
        speaker_map = {s: s for s in speakers}

    # Build markdown
    lines = ["## Speaker Legend\n"]
    for speaker in speakers:
        name = speaker_map.get(speaker, speaker)
        if name != speaker:
            lines.append(f"- {speaker}: {name}")
        else:
            lines.append(f"- {speaker}")

    lines.append("\n---\n")
    lines.append("## Transcript\n")

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        name = speaker_map.get(speaker, speaker)
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()

        if text:
            timestamp = format_timestamp(start)
            lines.append(f"**[{timestamp}] {name}:** {text}\n")

    return "\n".join(lines)


def create_meeting_note(
    folder_path: Path,
    topic: str,
    date_str: str,
    participant_names: list[str],
    has_transcript: bool,
    has_audio: bool,
    duration: str = None,
) -> str:
    """
    Generate the main meeting note markdown content.

    Args:
        folder_path: Path to the meeting folder
        topic: Meeting topic
        date_str: Date string (YYYY-MM-DD)
        participant_names: List of participant names
        has_transcript: Whether transcript exists
        has_audio: Whether audio file exists
        duration: Optional duration string

    Returns:
        Markdown content for meeting note
    """
    # Format participants as wiki links
    participant_links = ", ".join(f"[[{name}]]" for name in participant_names) if participant_names else ""

    lines = [
        "---",
        "fileClass: MeetingNote",
        f"date: {date_str}",
        "type: Group meeting" if len(participant_names) > 1 else "type: One-on-one",
        "status: Completed",
        "tags:",
        "  - meeting",
        f"participants: [{participant_links}]" if participant_links else "participants: []",
        f"summary: ",
        f"duration: {duration or ''}",
        f"has_transcript: {str(has_transcript).lower()}",
        f"has_audio: {str(has_audio).lower()}",
        "---",
        "",
        f"# {topic}",
        "",
        "## Summary",
        "",
        "",
        "## Participants",
        "",
    ]

    for name in participant_names:
        lines.append(f"- [[{name}]]")

    lines.extend([
        "",
        "## Agenda",
        "",
        "",
        "## Notes",
        "",
        "",
        "## Action Items",
        "",
        "- [ ] ",
        "",
    ])

    if has_transcript:
        lines.extend([
            "## Transcript",
            "",
            "![[transcript]]",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting audio with speaker diarization"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--topic", "-t",
        required=True,
        help="Meeting topic (used in folder/file naming)"
    )
    parser.add_argument(
        "--date", "-d",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Meeting date (YYYY-MM-DD format, default: today)"
    )
    parser.add_argument(
        "--participants", "-p",
        nargs="+",
        default=[],
        help="Participant names (space-separated, in order of speaking)"
    )
    parser.add_argument(
        "--speakers", "-s",
        type=int,
        default=None,
        help="Number of speakers (auto-detected if not specified)"
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="HuggingFace token for pyannote speaker diarization"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Copy audio file to meeting folder"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory for meetings (default: {MEETINGS_DIR})"
    )
    parser.add_argument(
        "--duration",
        default=None,
        help="Meeting duration (e.g., '45 min')"
    )

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        # Try relative to vault root
        audio_path = VAULT_ROOT / args.audio_file
        if not audio_path.exists():
            print(f"Error: Audio file not found: {args.audio_file}")
            return 1

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else MEETINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create meeting folder
    folder_name = f"{args.date} {args.topic}"
    meeting_folder = output_dir / folder_name
    meeting_folder.mkdir(parents=True, exist_ok=True)

    print(f"Created meeting folder: {meeting_folder}")

    # Transcribe audio
    compute_type = "int8" if args.device == "cpu" else "float16"
    result, _ = transcribe_audio(
        str(audio_path),
        args.hf_token,
        num_speakers=args.speakers,
        device=args.device,
        compute_type=compute_type,
    )

    # Generate transcript
    transcript_content = generate_transcript_md(result, args.participants)
    transcript_path = meeting_folder / "transcript.md"
    transcript_path.write_text(transcript_content)
    print(f"Created transcript: {transcript_path}")

    # Copy audio if requested
    has_audio = False
    if args.keep_audio:
        audio_dest = meeting_folder / f"audio{audio_path.suffix}"
        shutil.copy2(audio_path, audio_dest)
        has_audio = True
        print(f"Copied audio to: {audio_dest}")

    # Calculate duration if not provided
    duration = args.duration
    if not duration:
        try:
            import subprocess
            result_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            if result_probe.returncode == 0:
                seconds = float(result_probe.stdout.strip())
                minutes = int(seconds / 60)
                duration = f"{minutes} min"
        except Exception:
            pass

    # Create meeting note
    note_content = create_meeting_note(
        meeting_folder,
        args.topic,
        args.date,
        args.participants,
        has_transcript=True,
        has_audio=has_audio,
        duration=duration,
    )
    note_path = meeting_folder / f"{folder_name}.md"
    note_path.write_text(note_content)
    print(f"Created meeting note: {note_path}")

    print("\nDone! Meeting folder created successfully.")
    print(f"Location: {meeting_folder}")

    return 0


if __name__ == "__main__":
    exit(main())
