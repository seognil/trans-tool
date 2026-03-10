"""Chunk transcription via mlx-audio."""

from __future__ import annotations

import gc

from asr_core.schema import AudioChunk


class MlxAudioTranscriber:
    def __init__(self, *, model: str) -> None:
        from mlx_audio.stt.utils import load_model

        self.model = load_model(model)

    def transcribe_chunk(self, chunk: AudioChunk) -> str:
        result = None
        try:
            result = self.model.generate(
                str(chunk.path), language=None, max_tokens=1024, verbose=False
            )
            return " ".join(str(getattr(result, "text", "") or "").strip().split())
        finally:
            del result
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
