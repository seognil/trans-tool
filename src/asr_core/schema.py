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
    boundary_mode: str


@dataclass(frozen=True)
class PipelineResult:
    source: str
    output_path: str
    model: str
    language: str
    text: str
    chunk_count: int
    silence_gap_count: int
