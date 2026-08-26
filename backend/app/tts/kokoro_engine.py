"""Kokoro-82M vocal engine (Phase 3).

Adapter that prefers `pykokoro` (the accelerated re-implementation, which can
target the Apple Neural Engine via its Core ML / LiteRT backends) and falls
back to the reference `kokoro` package. Both expose a KPipeline-style API:

    pipeline = KPipeline(lang_code="b")
    for result in pipeline(text, voice="bm_george", speed=1.0):
        result.audio  # float32 waveform @ 24 kHz

The pipeline is a blocking CPU/ANE workload, so `stream()` runs inside a
worker thread (see manager.py) and yields int16 PCM frames over WebSockets.
"""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np

from ..config import settings
from ..logbus import LogBus
from .base import EngineUnavailable, SpeechChunk, float_to_int16, to_frames

_VOICE_LANG = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh"}


def _lang_code(voice: str) -> str:
    prefix = voice.split("_")[0][:1]
    return prefix if prefix in _VOICE_LANG else "b"


def _to_numpy(audio) -> np.ndarray:
    """Normalise torch tensors / numpy arrays from either library."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):  # torch.Tensor
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


class KokoroEngine:
    name = "kokoro-82M"
    sample_rate = settings.sample_rate

    def __init__(self, pipeline, voice: str, library: str) -> None:
        self._pipeline = pipeline
        self.voice = voice
        self.library = library  # "pykokoro" | "kokoro"

    @property
    def label(self) -> str:
        return f"{self.name} [{self.library}]"

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, bus: LogBus | None = None) -> "KokoroEngine":
        voice = settings.tts_voice
        lang = _lang_code(voice)
        errors: list[str] = []

        for module_name in ("pykokoro", "kokoro"):
            try:
                module = __import__(module_name)
                pipeline = cls._construct_pipeline(module, lang)
                if bus is not None:
                    bus.publish("voice", "tts", f"kokoro pipeline online via {module_name} (voice={voice}, lang={lang})")
                return cls(pipeline, voice, module_name)
            except Exception as exc:  # noqa: BLE001 - want every failure mode
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

        detail = "; ".join(errors)
        if bus is not None:
            bus.publish("warn", "tts", f"kokoro unavailable ({detail}) - using fallback-synth")
        raise EngineUnavailable(detail)

    @staticmethod
    def _construct_pipeline(module, lang: str):
        """KPipeline construction differs slightly across library versions."""
        kwargs: dict = {"lang_code": lang}
        device = settings.tts_device
        if device and device != "auto":
            kwargs["device"] = device
        try:
            return module.KPipeline(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return module.KPipeline(lang_code=lang)

    # -- streaming -------------------------------------------------------------

    def stream(self, text: str) -> Iterator[SpeechChunk]:
        text = self._clean(text)
        generator = self._pipeline(text, voice=self.voice, speed=settings.tts_speed)
        for result in generator:
            audio = _to_numpy(getattr(result, "audio", None))
            if audio.size == 0:
                continue
            frames = to_frames(float_to_int16(audio))
            for i, frame in enumerate(frames):
                yield SpeechChunk(pcm=frame, sample_rate=self.sample_rate, last=False)
        yield SpeechChunk(pcm=np.zeros(0, dtype=np.int16), sample_rate=self.sample_rate, last=True)

    @staticmethod
    def _clean(text: str) -> str:
        text = " ".join(text.split())
        text = re.sub(r"```.*?```", " code block ", text, flags=re.S)
        text = re.sub(r"[*_`#>|]", " ", text)
        return text[: settings.tts_max_chars].strip()
