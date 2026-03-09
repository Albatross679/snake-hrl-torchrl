# WhisperX Setup Guide

## Installation

### 1. Create Conda Environment

```bash
conda create -n whisperx python=3.10 -y
conda activate whisperx
```

### 2. Install WhisperX

```bash
pip install git+https://github.com/m-bain/whisperX.git
```

### 3. Install ffmpeg

```bash
brew install ffmpeg
```

### 4. Accept Pyannote Terms

Speaker diarization requires accepting the Pyannote model terms:

1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the terms of use
3. Go to https://huggingface.co/settings/tokens
4. Create or copy your access token

## Verify Installation

```bash
conda activate whisperx
python -c "import whisperx; print('WhisperX ready')"
```

## Troubleshooting

### "No module named 'whisperx'"

Ensure conda environment is activated:
```bash
conda activate whisperx
```

### Diarization fails with 401 error

Your HuggingFace token is invalid or you haven't accepted the Pyannote terms.

### Out of memory on CPU

Try a smaller model by editing the script to use `base` or `small` instead of `large-v3`.

### CUDA not available

The script defaults to CPU. For GPU acceleration, ensure CUDA toolkit is installed and use `--device cuda`.
