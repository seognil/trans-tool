# Local Qwen3-ASR with VAD and MLX Audio

## What It Does

This project is a minimal local audio/video-to-text pipeline for Apple Silicon Macs.

Its fixed workflow is:

1. `ffmpeg` converts the input media to `mono 16k wav`
2. `silero-vad` finds speech regions and guides long-audio chunking so the pipeline spends less time and memory than letting `mlx-audio` handle long chunks directly
3. `mlx-audio` transcribes each chunk locally with `mlx-community/Qwen3-ASR-0.6B-8bit` and auto language detection on
4. the pipeline writes a plain-text transcript

Dependencies are intentionally kept small:

- system tool: `ffmpeg` (includes `ffprobe`)
- Python packages managed by this project: `mlx-audio`, `silero-vad`, `soundfile`

## Installation

System dependency:

```bash
brew install ffmpeg
```

Project environment and Python dependencies:

```bash
uv venv --python 3.10
uv sync --prerelease allow
```

Optional: download the official Qwen3-ASR sample audio for a quick local test:

```bash
uv run python scripts/download_samples.py
```

## Usage

TL;DR:

```bash
uv run qwen3-asr samples/asr_zh.wav
```

Multiple inputs are processed one by one using the same single-file pipeline:

```bash
uv run qwen3-asr samples/asr_zh.wav samples/sample-20min.m4a
```

Module entrypoint is also available if you prefer:

```bash
uv run python -m asr_core samples/asr_zh.wav
```

Install as a user-level tool:

```bash
uv tool install . --prerelease allow
qwen3-asr samples/asr_zh.wav
```

With the default settings, that command writes `samples/asr_zh.wav.txt` next to the input file.

More examples:

```bash
# write the transcript into a custom output directory
uv run qwen3-asr samples/asr_zh.wav --output-dir dist/my-run

# process multiple inputs in order
uv run qwen3-asr samples/asr_zh.wav samples/sample-20min.m4a

# overwrite an existing output file
uv run qwen3-asr samples/asr_zh.wav -y

# skip files that already have final outputs
uv run qwen3-asr samples/asr_zh.wav samples/sample-20min.m4a -s
```

Parameters:

- `inputs` - one or more input audio or video file paths; multiple inputs run sequentially in the given order
- `--output-dir` - optional output directory; default is the input file directory
- `-y`, `--force` - overwrite the output text file if it already exists
- `-s`, `--skip-existing` - skip inputs whose final output text file already exists

Output behavior:

- default output path: the input file name with an extra `.txt` suffix in the same directory
- example: `samples/asr_zh.wav` -> `samples/asr_zh.wav.txt`
- with `--output-dir dist/my-run`: `samples/asr_zh.wav` -> `dist/my-run/asr_zh.wav.txt`
- writes in-progress output to `<output>.tmp` and renames it to the final `.txt` file only after transcription completes
- with `-s`, only an existing final `.txt` file counts as completed; leftover `.tmp` files are ignored and will be regenerated
- for multiple inputs, the tool checks all target output paths before starting any transcription and lists all preflight problems in a single error
- the output text keeps chunk boundaries as blank-line-separated paragraphs

## References

The current design is informed mainly by these upstream projects:

- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio)
  - local MLX audio runtime reference used by the current ASR path
- [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR)
  - original open-source model, API shape, and evaluation defaults
- [`QwenLM/Qwen3-ASR-Toolkit`](https://github.com/QwenLM/Qwen3-ASR-Toolkit)
  - practical long-audio pipeline reference
