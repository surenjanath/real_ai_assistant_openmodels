"""TTS subpackage - engine selection lives here."""

from __future__ import annotations

from ..config import settings
from ..logbus import LogBus
from .base import EngineUnavailable, SpeechChunk, TTSEngine
from .fallback_engine import FallbackEngine
from .kokoro_engine import KokoroEngine

__all__ = ["build_engine", "EngineUnavailable", "SpeechChunk", "TTSEngine"]


def build_engine(bus: LogBus) -> tuple[TTSEngine, str]:
    """Resolve the vocal engine per settings.

    Returns (engine, mode) where mode is one of 'kokoro' | 'fallback'.
    """
    choice = settings.tts_engine.lower()

    if choice in ("auto", "kokoro", "pykokoro"):
        try:
            return KokoroEngine.build(bus), "kokoro"
        except EngineUnavailable:
            if choice != "auto":
                bus.publish("error", "tts", f"forced engine '{choice}' unavailable - falling back")

    bus.publish(
        "warn",
        "tts",
        "using dependency-free fallback-synth - run `make setup` on python 3.12 "
        "and `pip install 'pykokoro[coreml]'` for the real Kokoro voice",
    )
    return FallbackEngine(), "fallback"
