"""Audio normalization and loading helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


def normalize_to_wav(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-vn",
        str(destination),
    ]
    run_command(command)


def load_audio_samples(audio_path: Path):
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError(f"audio loading dependencies unavailable: {exc}") from exc

    wav, sample_rate = sf.read(str(audio_path), dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"expected {SAMPLE_RATE}Hz normalized audio, got {sample_rate}Hz for {audio_path}"
        )
    if getattr(wav, "ndim", 1) > 1:
        wav = wav[:, 0]
    return wav


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{stderr}")
    return completed
