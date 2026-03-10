"""Silero VAD and chunk planning helpers."""

from __future__ import annotations

import math
from pathlib import Path

from asr_core.audio import (
    SAMPLE_RATE,
    load_audio_samples,
    run_command,
)
from asr_core.schema import AudioChunk, SilenceGap

TARGET_CHUNK_SECONDS = 60
MAX_CHUNK_SECONDS = 120
_SILERO_MODEL = None


def get_silero_speech_timestamps(
    audio_path: Path,
    *,
    min_speech_duration_ms: int = 1500,
    min_silence_duration_ms: int = 500,
) -> tuple[list[dict[str, int]], list[SilenceGap], int]:
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio_samples = load_audio_samples(audio_path)
    model = _load_silero_model(load_silero_vad)
    speech_timestamps = get_speech_timestamps(
        audio_samples,
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
    )
    gaps = _speech_timestamps_to_gaps(speech_timestamps, len(audio_samples))
    return speech_timestamps, gaps, len(audio_samples)


def plan_chunks(
    audio_path: Path,
    chunk_dir: Path,
    *,
    target_chunk_seconds: int = TARGET_CHUNK_SECONDS,
    max_chunk_seconds: int = MAX_CHUNK_SECONDS,
) -> tuple[list[AudioChunk], list[SilenceGap]]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    speech_timestamps, silence_gaps, total_samples = get_silero_speech_timestamps(
        audio_path
    )
    split_points = _toolbox_split_points(
        speech_timestamps,
        silence_gaps=silence_gaps,
        total_samples=total_samples,
        target_chunk_seconds=target_chunk_seconds,
        max_chunk_seconds=max_chunk_seconds,
    )

    chunks: list[AudioChunk] = []
    for index, (start_sample, end_sample) in enumerate(
        zip(split_points[:-1], split_points[1:])
    ):
        start_seconds = start_sample / SAMPLE_RATE
        end_seconds = end_sample / SAMPLE_RATE
        chunk_path = chunk_dir / f"chunk-{index:04d}.wav"
        _export_chunk(audio_path, chunk_path, start_seconds, end_seconds)
        chunks.append(
            AudioChunk(
                index=index,
                path=chunk_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return chunks, silence_gaps


def _toolbox_split_points(
    speech_timestamps: list[dict[str, int]],
    *,
    silence_gaps: list[SilenceGap],
    total_samples: int,
    target_chunk_seconds: int,
    max_chunk_seconds: int,
) -> list[int]:
    target_chunk_samples = target_chunk_seconds * SAMPLE_RATE
    final_split_points = {0, total_samples}
    target_time = target_chunk_samples
    previous_point = 0
    start_candidates = _build_start_candidates(speech_timestamps, silence_gaps)
    while target_time < total_samples:
        chosen_point = _select_boundary(
            target_time=target_time,
            previous_point=previous_point,
            total_samples=total_samples,
            start_candidates=start_candidates,
        )
        final_split_points.add(int(chosen_point))
        previous_point = max(previous_point, int(chosen_point))
        target_time += target_chunk_samples
    ordered_splits = sorted(final_split_points)

    max_chunk_samples = max_chunk_seconds * SAMPLE_RATE
    split_points = [0]
    for end_point in ordered_splits[1:]:
        start_point = split_points[-1]
        segment_length = end_point - start_point
        if segment_length <= max_chunk_samples:
            split_points.append(end_point)
            continue
        subsegment_count = math.ceil(segment_length / max_chunk_samples)
        subsegment_length = segment_length / subsegment_count
        for index in range(1, subsegment_count):
            split_points.append(int(start_point + index * subsegment_length))
        split_points.append(end_point)
    return split_points


def _speech_timestamps_to_gaps(
    speech_timestamps: list[dict[str, int]],
    total_samples: int,
) -> list[SilenceGap]:
    gaps: list[SilenceGap] = []
    cursor = 0
    for segment in speech_timestamps:
        start_sample = int(segment["start"])
        end_sample = int(segment["end"])
        if start_sample > cursor:
            start_seconds = cursor / SAMPLE_RATE
            end_seconds = start_sample / SAMPLE_RATE
            gaps.append(
                SilenceGap(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    duration_seconds=max(0.0, end_seconds - start_seconds),
                )
            )
        cursor = max(cursor, end_sample)
    if cursor < total_samples:
        start_seconds = cursor / SAMPLE_RATE
        end_seconds = total_samples / SAMPLE_RATE
        gaps.append(
            SilenceGap(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=max(0.0, end_seconds - start_seconds),
            )
        )
    return gaps


def _build_start_candidates(
    speech_timestamps: list[dict[str, int]],
    silence_gaps: list[SilenceGap],
) -> list[dict[str, float]]:
    gap_map = {int(round(gap.end_seconds * SAMPLE_RATE)): gap for gap in silence_gaps}
    candidates: list[dict[str, float]] = []
    for segment in speech_timestamps:
        point = int(segment["start"])
        preceding_gap = gap_map.get(point)
        candidates.append(
            {
                "point": point,
                "gap_duration": preceding_gap.duration_seconds
                if preceding_gap
                else 0.0,
            }
        )
    return candidates


def _select_boundary(
    *,
    target_time: int,
    previous_point: int,
    total_samples: int,
    start_candidates: list[dict[str, float]],
) -> int:
    min_offset_samples = int(1.0 * SAMPLE_RATE)
    valid_starts = [
        candidate
        for candidate in start_candidates
        if previous_point + min_offset_samples
        <= int(candidate["point"])
        < total_samples
    ]
    if not valid_starts:
        return target_time
    best = min(valid_starts, key=lambda item: abs(int(item["point"]) - target_time))
    return int(best["point"])


def _load_silero_model(load_silero_vad):
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL


def _export_chunk(
    audio_path: Path,
    chunk_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{max(0.0, end_seconds - start_seconds):.3f}",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(chunk_path),
    ]
    run_command(command)
