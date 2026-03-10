"""Structured result types for the local ASR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SilenceGap:
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class AudioChunk:
    index: int
    path: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PipelineResult:
    output_path: str
    text: str
    chunk_count: int
    silence_gap_count: int
    audio_duration_seconds: float | None
    normalize_elapsed_seconds: float
    vad_elapsed_seconds: float
    asr_elapsed_seconds: float
    elapsed_seconds: float
