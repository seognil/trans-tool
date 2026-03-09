"""Main entry point and orchestration for the local ASR pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from asr_core.audio import normalize_to_wav
from asr_core.mlx_runtime import MlxTranscriber
from asr_core.schema import PipelineResult
from asr_core.vad import MAX_CHUNK_SECONDS, TARGET_CHUNK_SECONDS, plan_chunks

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


class Job(TypedDict):
    source: str
    output_dir: str | None


class PreparedJobs(TypedDict):
    jobs: list[Job]
    skipped: list[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local audio/video to text pipeline: ffmpeg normalize, Silero VAD "
            "segmentation, and MLX Qwen3-ASR transcription."
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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for the upstream MLX runtime token limit.",
    )
    parser.add_argument(
        "--language",
        help="Optional forced language, for example 'zh' or 'Chinese'.",
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

    jobs = prepared["jobs"]
    for source, output_path in prepared["skipped"]:
        print(f"skip: {source} -> {output_path}")

    if not jobs:
        print("nothing to do")
        return 0

    exit_code = 0
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if total_jobs > 1:
            print(f"[{index}/{total_jobs}] input={job['source']}")

        source, output_dir = job["source"], job["output_dir"]
        try:
            result = run_pipeline(
                source=source,
                output_dir=output_dir,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                language=args.language,
                force=args.force,
            )
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"failed for {source}: {exc}", file=sys.stderr)
            exit_code = 1
            break

        print(f"language={result.language}")
        print(f"text_chars={len(result.text)}")
        print(f"chunks={result.chunk_count}")
        print(f"silence_gaps={result.silence_gap_count}")
        print(f"elapsed={_format_duration(result.elapsed_seconds)}")
        print(f"output={result.output_path}")

    return exit_code


def run_pipeline(
    *,
    source: str,
    output_dir: str | None,
    model: str,
    max_new_tokens: int | None,
    language: str | None = None,
    force: bool = False,
) -> PipelineResult:
    pipeline_started = time.perf_counter()
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"input file not found: {source_path}")

    output_path = _resolve_output_path(source_path, output_dir)
    temp_output_path = _resolve_temp_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_writable_output(output_path, force=force)

    chunk_languages: list[str] = []
    chunk_texts: list[str] = []

    try:
        with TemporaryDirectory(prefix="qwen-asr-") as temp_dir:
            work_dir = Path(temp_dir)
            chunk_dir = work_dir / "chunks"
            normalized_audio = work_dir / "normalized.wav"

            print("[1/4] normalize audio")
            normalize_to_wav(source_path, normalized_audio)

            print(
                "[2/4] plan chunks with Silero VAD segmentation "
                f"(target={TARGET_CHUNK_SECONDS}s, max={MAX_CHUNK_SECONDS}s)"
            )
            chunks, silence_gaps = plan_chunks(normalized_audio, chunk_dir)
            print(f"created {len(chunks)} chunks")
            if silence_gaps:
                print(f"detected {len(silence_gaps)} silence gaps")

            print("[3/4] transcribe chunks with MLX")
            runtime = MlxTranscriber(
                model=model,
                max_new_tokens=max_new_tokens,
                language=language,
            )
            started = time.perf_counter()
            with temp_output_path.open("w", encoding="utf-8") as handle:
                for index, chunk in enumerate(chunks, start=1):
                    chunk_started = time.perf_counter()
                    detected_language, text = runtime.transcribe_chunk(chunk)
                    chunk_elapsed = time.perf_counter() - chunk_started
                    if detected_language:
                        chunk_languages.append(detected_language)
                    cleaned_text = text.strip()
                    if cleaned_text:
                        if chunk_texts:
                            handle.write("\n\n")
                        handle.write(cleaned_text)
                        handle.flush()
                        chunk_texts.append(cleaned_text)

                    elapsed_total = time.perf_counter() - started
                    eta_text = _format_eta(
                        elapsed_total=elapsed_total,
                        finished_chunks=index,
                        total_chunks=len(chunks),
                    )
                    status = (
                        f"  - chunk {index}/{len(chunks)} "
                        f"[{chunk.start_seconds:.1f}s -> {chunk.end_seconds:.1f}s] "
                        f"done in {chunk_elapsed:.1f}s, wrote {len(cleaned_text)} chars"
                    )
                    if eta_text:
                        status = f"{status}, eta ~ {eta_text}"
                    print(status)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    final_text = _build_output_text(chunk_texts)
    temp_output_path.write_text(
        final_text + "\n" if final_text else "", encoding="utf-8"
    )
    temp_output_path.replace(output_path)
    print(f"[4/4] write transcript -> {output_path}")
    print(f"wrote {output_path}")
    return PipelineResult(
        source=str(source_path),
        output_path=str(output_path),
        model=model,
        language=_pick_language(chunk_languages, language),
        text=final_text,
        chunk_count=len(chunks),
        silence_gap_count=len(silence_gaps),
        elapsed_seconds=time.perf_counter() - pipeline_started,
    )


def _prepare_jobs(
    inputs: list[str], output_dir: str | None, *, force: bool, skip_existing: bool
) -> PreparedJobs:
    jobs: list[Job] = []
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
        jobs.append({"source": str(source_path), "output_dir": output_dir})

    if errors:
        raise ValueError("batch preflight failed:\n- " + "\n- ".join(errors))

    return {"jobs": jobs, "skipped": skipped}


def _resolve_output_path(source_path: Path, output_dir: str | None) -> Path:
    filename = f"{source_path.name}.txt"
    if output_dir:
        return Path(output_dir).expanduser().resolve() / filename
    return source_path.parent / filename


def _resolve_temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.tmp")


def _ensure_writable_output(output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output file already exists: {output_path}\n"
            "Use -y or --force to overwrite it."
        )


def _build_output_text(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip()).strip()


def _pick_language(values: list[str], fallback: str | None) -> str:
    if fallback:
        return fallback
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


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
