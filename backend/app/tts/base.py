"""TTS engine contract shared by Kokoro and the fallback synthesizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from ..config import settings


@dataclass(frozen=True)
class SpeechChunk:
    """A chunk of raw mono audio as int16 little-endian PCM."""

    pcm: np.ndarray  # int16
    sample_rate: int
    last: bool = False


class TTSEngine(Protocol):
    name: str
    sample_rate: int
    voice: str

    def stream(self, text: str) -> Iterator[SpeechChunk]:
        """Synchronous generator - run it in a worker thread."""
        ...  # pragma: no cover


class EngineUnavailable(Exception):
    """Raised by builders when an engine cannot initialise on this host."""


def to_frames(pcm: np.ndarray, frame_samples: int | None = None) -> list[np.ndarray]:
    """Split a PCM array into even frames suitable for WebSocket transport."""
    frame_samples = frame_samples or settings.tts_frame_samples
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    if pcm.size == 0:
        return []
    frames = [
        pcm[i : i + frame_samples]
        for i in range(0, pcm.size, frame_samples)
    ]
    return frames


def float_to_int16(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    """Convert model float output to int16 with gentle headroom clipping."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return (audio * 32767.0).astype(np.int16)
