"""Thin MLX runtime wrapper for chunk-by-chunk transcription."""

from __future__ import annotations

from asr_core.schema import AudioChunk


class MlxTranscriber:
    def __init__(
        self, model: str, max_new_tokens: int | None, language: str | None = None
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.language = language

    def transcribe_chunk(self, chunk: AudioChunk) -> tuple[str, str]:
        from mlx_qwen3_asr import transcribe

        kwargs = {
            "model": self.model,
            "language": self.language,
            "return_timestamps": False,
        }
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens

        result = transcribe(str(chunk.path), **kwargs)
        language = getattr(result, "language", "") or ""
        text = _normalize_text(getattr(result, "text", ""))
        return language, text


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())
