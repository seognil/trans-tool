# Local Qwen3-ASR with VAD and MLX

## What It Does

This project is a minimal local audio/video-to-text pipeline for Apple Silicon Macs.

Its default workflow is:

1. `ffmpeg` converts the input media to `mono 16k wav`
2. `silero-vad` finds speech regions and guides long-audio chunking
3. `mlx-qwen3-asr` transcribes each chunk locally with Qwen3-ASR
4. the pipeline writes a plain-text transcript

Dependencies are intentionally kept small:

- system tool: `ffmpeg` (includes `ffprobe`)
- Python packages managed by this project: `mlx-qwen3-asr`, `silero-vad`, `soundfile`

## Installation

System dependency:

```bash
brew install ffmpeg
```

Project environment and Python dependencies:

```bash
uv venv --python 3.10
uv sync
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
uv tool install .
qwen3-asr samples/asr_zh.wav
```

With the default settings, that command writes `samples/asr_zh.wav.txt` next to the input file.

More examples:

```bash
# write the transcript into a custom output directory
uv run qwen3-asr samples/asr_zh.wav --output-dir dist/my-run

# process multiple inputs in order
uv run qwen3-asr samples/asr_zh.wav samples/sample-20min.m4a

# force the language
uv run qwen3-asr samples/asr_zh.wav --language zh

# override the model
uv run qwen3-asr samples/asr_zh.wav --model Qwen/Qwen3-ASR-0.6B

# override the upstream token limit
uv run qwen3-asr samples/asr_zh.wav --max-new-tokens 1024

# overwrite an existing output file
uv run qwen3-asr samples/asr_zh.wav -y
```

Parameters:

- `inputs` - one or more input audio or video file paths; multiple inputs run sequentially in the given order
- `--output-dir` - optional output directory; default is the input file directory
- `--model` - model name or local path, default `Qwen/Qwen3-ASR-0.6B`
- `--language` - optional forced language, otherwise auto-detect
- `--max-new-tokens` - optional override, otherwise `mlx_qwen3_asr` uses its default `1024`
- `-y`, `--force` - overwrite the output text file if it already exists

Output behavior:

- default output path: the input file name with an extra `.txt` suffix in the same directory
- example: `samples/asr_zh.wav` -> `samples/asr_zh.wav.txt`
- with `--output-dir dist/my-run`: `samples/asr_zh.wav` -> `dist/my-run/asr_zh.wav.txt`
- for multiple inputs, the tool checks all target output paths before starting any transcription and lists all preflight problems in a single error
- the output text keeps chunk boundaries as blank-line-separated paragraphs

## References

The current design is informed mainly by these upstream projects:

- [`moona3k/mlx-qwen3-asr`](https://github.com/moona3k/mlx-qwen3-asr)
  - local MLX-based Qwen3-ASR runtime reference
- [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR)
  - original open-source model, API shape, and evaluation defaults
- [`QwenLM/Qwen3-ASR-Toolkit`](https://github.com/QwenLM/Qwen3-ASR-Toolkit)
  - practical long-audio pipeline reference
