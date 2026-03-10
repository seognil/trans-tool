"""Main entry point and orchestration for the local ASR pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from asr_core.asr_mlx_audio import MlxAudioTranscriber
from asr_core.audio import normalize_to_wav, probe_media_info
from asr_core.schema import PipelineResult
from asr_core.vad import MAX_CHUNK_SECONDS, TARGET_CHUNK_SECONDS, plan_chunks

MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local audio/video to text pipeline: ffmpeg normalize, Silero VAD "
            "segmentation, and mlx-audio Qwen3-ASR transcription."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input audio or video file paths.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to the input file directory.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-y",
        "--force",
        action="store_true",
        help="Overwrite the output text file if it already exists.",
    )
    mode_group.add_argument(
        "-s",
        "--skip-existing",
        action="store_true",
        help="Skip inputs whose output text file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        prepared = _prepare_jobs(
            args.inputs,
            args.output_dir,
            force=args.force,
            skip_existing=args.skip_existing,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    jobs, skipped = prepared
    for source, output_path in skipped:
        print(f"skip: {source} -> {output_path}")

    if not jobs:
        print("nothing to do")
        return 0

    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if total_jobs > 1:
            print(f"[{index}/{total_jobs}] input={job[0]}")

        source, output_dir = job
        try:
            result = run_pipeline(
                source=source,
                output_dir=output_dir,
                force=args.force,
            )
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"failed for {source}: {exc}", file=sys.stderr)
            return 1

        print(f"text_chars={len(result.text)}")
        print(f"chunks={result.chunk_count}")
        print(f"silence_gaps={result.silence_gap_count}")
        detail_parts = []
        if result.audio_duration_seconds is not None:
            detail_parts.append(
                f"audio={_format_duration(result.audio_duration_seconds)}"
            )
            if result.elapsed_seconds > 0:
                detail_parts.append(
                    f"speed={result.audio_duration_seconds / result.elapsed_seconds:.2f}x"
                )
        detail_parts.append(
            f"normalize={_format_duration(result.normalize_elapsed_seconds)}"
        )
        detail_parts.append(f"vad={_format_duration(result.vad_elapsed_seconds)}")
        detail_parts.append(f"asr={_format_duration(result.asr_elapsed_seconds)}")
        print(
            f"elapsed={_format_duration(result.elapsed_seconds)} "
            f"({', '.join(detail_parts)})"
        )
        print(f"output={result.output_path}")
        print()

    return 0


def run_pipeline(
    *,
    source: str,
    output_dir: str | None,
    force: bool = False,
) -> PipelineResult:
    pipeline_started = time.perf_counter()
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"input file not found: {source_path}")

    output_path = _resolve_output_path(source_path, output_dir)
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output file already exists: {output_path}\n"
            "Use -y or --force to overwrite it."
        )
    media_info = probe_media_info(source_path)

    print(
        f"input={source_path}, "
        f"duration={_format_duration(media_info.duration_seconds) if media_info.duration_seconds is not None else 'unknown'}, "
        f"size={_format_size(media_info.size_bytes)}"
    )
    print(f"output={output_path}")

    chunk_texts: list[str] = []
    normalize_elapsed = 0.0
    vad_elapsed = 0.0
    asr_elapsed = 0.0

    try:
        with TemporaryDirectory(prefix="qwen-asr-") as temp_dir:
            work_dir = Path(temp_dir)
            chunk_dir = work_dir / "chunks"
            normalized_audio = work_dir / "normalized.wav"

            print("[1/4] normalize audio")
            started = time.perf_counter()
            normalize_to_wav(source_path, normalized_audio)
            normalize_elapsed = time.perf_counter() - started

            print(
                "[2/4] plan chunks with Silero VAD segmentation "
                f"(target={TARGET_CHUNK_SECONDS}s, max={MAX_CHUNK_SECONDS}s)"
            )
            started = time.perf_counter()
            chunks, silence_gaps = plan_chunks(normalized_audio, chunk_dir)
            vad_elapsed = time.perf_counter() - started
            print(f"created {len(chunks)} chunks")
            if silence_gaps:
                print(f"detected {len(silence_gaps)} silence gaps")

            print("[3/4] transcribe chunks with MLX")
            runtime = MlxAudioTranscriber(model=MODEL)
            started = time.perf_counter()
            with temp_output_path.open("w", encoding="utf-8") as handle:
                for index, chunk in enumerate(chunks, start=1):
                    chunk_started = time.perf_counter()
                    text = runtime.transcribe_chunk(chunk)
                    chunk_elapsed = time.perf_counter() - chunk_started
                    if text:
                        if chunk_texts:
                            handle.write("\n\n")
                        handle.write(text)
                        handle.flush()
                        chunk_texts.append(text)

                    elapsed_total = time.perf_counter() - started
                    eta_text = _format_eta(
                        elapsed_total=elapsed_total,
                        finished_chunks=index,
                        total_chunks=len(chunks),
                    )
                    status = (
                        f"  - chunk {index}/{len(chunks)} "
                        f"[{chunk.start_seconds:.1f}s -> {chunk.end_seconds:.1f}s] "
                        f"done in {chunk_elapsed:.1f}s, wrote {len(text)} chars"
                    )
                    if eta_text:
                        status = f"{status}, eta ~ {eta_text}"
                    print(status)
            asr_elapsed = time.perf_counter() - started
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    final_text = "\n\n".join(
        part.strip() for part in chunk_texts if part.strip()
    ).strip()
    temp_output_path.write_text(
        final_text + "\n" if final_text else "", encoding="utf-8"
    )
    temp_output_path.replace(output_path)
    print(f"[4/4] write transcript -> {output_path}")
    print(f"wrote {output_path}")
    return PipelineResult(
        output_path=str(output_path),
        text=final_text,
        chunk_count=len(chunks),
        silence_gap_count=len(silence_gaps),
        audio_duration_seconds=media_info.duration_seconds,
        normalize_elapsed_seconds=normalize_elapsed,
        vad_elapsed_seconds=vad_elapsed,
        asr_elapsed_seconds=asr_elapsed,
        elapsed_seconds=time.perf_counter() - pipeline_started,
    )


def _prepare_jobs(
    inputs: list[str], output_dir: str | None, *, force: bool, skip_existing: bool
) -> tuple[list[tuple[str, str | None]], list[tuple[str, str]]]:
    jobs: list[tuple[str, str | None]] = []
    skipped: list[tuple[str, str]] = []
    seen_outputs: dict[Path, Path] = {}
    errors: list[str] = []

    for source in inputs:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            errors.append(f"missing: {source_path}")
            continue

        output_path = _resolve_output_path(source_path, output_dir)
        if output_path.exists() and skip_existing:
            skipped.append((str(source_path), str(output_path)))
            continue
        if output_path.exists() and not force:
            errors.append(f"exists: {source_path} -> {output_path}")

        first_source = seen_outputs.get(output_path)
        if first_source is not None:
            errors.append(f"conflict: {first_source}, {source_path} -> {output_path}")
            continue

        seen_outputs[output_path] = source_path
        jobs.append((str(source_path), output_dir))

    if errors:
        raise ValueError("batch preflight failed:\n- " + "\n- ".join(errors))

    return jobs, skipped


def _resolve_output_path(source_path: Path, output_dir: str | None) -> Path:
    filename = f"{source_path.name}.txt"
    if output_dir:
        return Path(output_dir).expanduser().resolve() / filename
    return source_path.parent / filename


def _format_eta(
    *, elapsed_total: float, finished_chunks: int, total_chunks: int
) -> str:
    if finished_chunks < 1 or finished_chunks >= total_chunks:
        return ""
    avg_chunk_seconds = elapsed_total / finished_chunks
    remaining_seconds = int(round(avg_chunk_seconds * (total_chunks - finished_chunks)))
    minutes, seconds = divmod(max(0, remaining_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _format_duration(seconds: float) -> str:
    remaining_seconds = int(round(max(0.0, seconds)))
    minutes, secs = divmod(remaining_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)}{units[unit_index]}"
    if value >= 10 or unit_index <= 2:
        return f"{value:.0f}{units[unit_index]}"
    return f"{value:.1f}{units[unit_index]}"
