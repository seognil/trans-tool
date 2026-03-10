"""Audio normalization and loading helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float | None
    size_bytes: int | None


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
    wav, sample_rate = sf.read(str(audio_path), dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"expected {SAMPLE_RATE}Hz normalized audio, got {sample_rate}Hz for {audio_path}"
        )
    if getattr(wav, "ndim", 1) > 1:
        wav = wav[:, 0]
    return wav


def probe_media_info(source: Path) -> MediaInfo:
    completed = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = json.loads(completed.stdout or "{}")
    format_info = payload.get("format", {})
    return MediaInfo(
        duration_seconds=(
            float(format_info["duration"])
            if format_info.get("duration") not in (None, "")
            else None
        ),
        size_bytes=(
            int(format_info["size"])
            if format_info.get("size") not in (None, "")
            else None
        ),
    )


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{stderr}")
    return completed
