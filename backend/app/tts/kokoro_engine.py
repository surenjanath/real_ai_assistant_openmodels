"""Kokoro-82M vocal engine (Phase 3).

Two adapters, tried in order:

1. **pykokoro** - the accelerated ONNX re-implementation. No torch, and its
   `provider` can target Core ML (Apple Neural Engine) so Metal stays free for
   Ollama inference. Real API (v0.8.x)::

       from pykokoro import KokoroPipeline, PipelineConfig, GenerationConfig
       pipe = KokoroPipeline(PipelineConfig(voice="bm_george",
                                            generation=GenerationConfig(lang="en-gb")))
       for unit in pipe.iter_units(text, unit="sentence"):
           unit.audio        # float32 @ unit.sample_rate

2. **kokoro** - the reference torch implementation, whose `KPipeline` yields
   `result.audio`. Requires Python < 3.13.

`iter_units(unit="sentence")` is what gives us low perceived latency: the first
sentence is speakable in ~1.3 s while the rest of the utterance is still being
synthesised, and the manager streams each sentence out as it lands.

Voice, speed and language are all mutable. Because a Kokoro voice's language is
implied by its prefix (`bm_` -> en-GB, `af_` -> en-US, ...), switching voice can
change the G2P language, which requires a pipeline rebuild - handled here via a
config-key check on every utterance.
"""

from __future__ import annotations

import re
import threading
from typing import Iterator

import numpy as np

from ..config import settings
from ..logbus import LogBus
from .base import EngineUnavailable, SpeechChunk, float_to_int16, to_frames

# Kokoro voice-prefix -> espeak/G2P language tag.
_VOICE_LANG = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
    "d": "de",
}
# The reference `kokoro` KPipeline takes the single-letter form instead.
_LANG_LETTER = {v: k for k, v in _VOICE_LANG.items()}


def lang_for_voice(voice: str) -> str:
    """'bm_george' -> 'en-gb'. Unknown prefixes fall back to British English."""
    prefix = (voice or "").split("_")[0][:1].lower()
    return _VOICE_LANG.get(prefix, "en-gb")


def _to_numpy(audio) -> np.ndarray:
    """Normalise torch tensors / numpy arrays from either library."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):  # torch.Tensor
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def clean_for_speech(text: str) -> str:
    """Strip markdown / code so the model never tries to pronounce syntax."""
    text = re.sub(r"```.*?```", " . ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " a link ", text)
    text = re.sub(r"[*_#>|~]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[: settings.tts_max_chars].strip()


class KokoroEngine:
    """Streaming Kokoro-82M adapter with live voice/speed switching."""

    name = "kokoro-82M"
    sample_rate = settings.sample_rate

    def __init__(self, library: str, voice: str, bus: LogBus | None = None) -> None:
        self.library = library  # "pykokoro" | "kokoro"
        self.voice = voice  # mutable - the registry hot-swaps this
        self.speed = settings.tts_speed  # mutable - registry-controlled
        self.bus = bus
        self._pipeline = None
        self._config_key: tuple | None = None
        self._lock = threading.Lock()  # pipelines are not thread-safe

    @property
    def label(self) -> str:
        return f"{self.name} [{self.library}]"

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, bus: LogBus | None = None) -> "KokoroEngine":
        voice = settings.tts_voice
        errors: list[str] = []

        for library in ("pykokoro", "kokoro"):
            try:
                engine = cls(library, voice, bus)
                engine._ensure_pipeline()  # constructs + validates immediately
                if bus is not None:
                    bus.publish(
                        "voice",
                        "tts",
                        f"kokoro online via {library} "
                        f"(voice={voice}, lang={lang_for_voice(voice)}, "
                        f"quality={settings.tts_quality}, provider={settings.tts_device})",
                    )
                return engine
            except Exception as exc:  # noqa: BLE001 - want every failure mode
                errors.append(f"{library}: {type(exc).__name__}: {exc}")

        detail = "; ".join(errors)
        if bus is not None:
            bus.publish("warn", "tts", f"kokoro unavailable ({detail})")
        raise EngineUnavailable(detail)

    def _current_key(self) -> tuple:
        """Everything that, when changed, requires a new pipeline."""
        return (self.library, lang_for_voice(self.voice), self.voice)

    def _ensure_pipeline(self):
        """(Re)build the pipeline if voice/language changed. Caller holds no lock."""
        key = self._current_key()
        if self._pipeline is not None and key == self._config_key:
            return self._pipeline
        self._pipeline = (
            self._build_pykokoro() if self.library == "pykokoro" else self._build_kokoro()
        )
        self._config_key = key
        return self._pipeline

    def _build_pykokoro(self):
        from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig  # noqa: PLC0415

        kwargs: dict = {
            "voice": self.voice,
            "generation": GenerationConfig(lang=lang_for_voice(self.voice)),
        }
        if settings.tts_quality:
            kwargs["model_quality"] = settings.tts_quality
        device = (settings.tts_device or "auto").lower()
        # 'auto' lets onnxruntime choose; anything else is an explicit provider.
        # 'mps'/'cuda' are torch-isms with no ONNX equivalent here - ignore them.
        if device in ("cpu", "coreml", "openvino", "directml", "dml", "xnnpack", "cuda"):
            kwargs["provider"] = device
        return KokoroPipeline(PipelineConfig(**kwargs))

    def _build_kokoro(self):
        import kokoro  # noqa: PLC0415

        letter = _LANG_LETTER.get(lang_for_voice(self.voice), "b")
        return kokoro.KPipeline(lang_code=letter)

    def warmup(self) -> None:
        """Pay the ONNX session-init cost at boot, not on the first answer."""
        try:
            with self._lock:
                pipeline = self._ensure_pipeline()
                for _ in self._iter_audio(pipeline, "Systems online."):
                    break
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            if self.bus is not None:
                self.bus.publish("warn", "tts", f"warmup skipped: {type(exc).__name__}: {exc}")

    # -- streaming -------------------------------------------------------------

    def _iter_audio(self, pipeline, text: str) -> Iterator[np.ndarray]:
        """Yield one float32 waveform per sentence, library-agnostic."""
        speed = max(0.5, min(2.0, float(self.speed)))
        if self.library == "pykokoro":
            # Speed lives on the nested GenerationConfig, not the top-level
            # PipelineConfig; a mapping here is merged onto the existing one.
            # Sentence units keep first-audio latency low.
            for unit in pipeline.iter_units(
                text, unit="sentence", generation={"speed": speed}
            ):
                yield _to_numpy(getattr(unit, "audio", None))
        else:
            for result in pipeline(text, voice=self.voice, speed=speed):
                yield _to_numpy(getattr(result, "audio", None))

    def stream(self, text: str) -> Iterator[SpeechChunk]:
        text = clean_for_speech(text)
        if not text:
            yield SpeechChunk(pcm=np.zeros(0, dtype=np.int16), sample_rate=self.sample_rate, last=True)
            return
        # Hold the lock for the whole utterance: a single ONNX session must not
        # be driven by two utterances at once.
        with self._lock:
            pipeline = self._ensure_pipeline()
            for audio in self._iter_audio(pipeline, text):
                if audio.size == 0:
                    continue
                for frame in to_frames(float_to_int16(audio)):
                    yield SpeechChunk(pcm=frame, sample_rate=self.sample_rate, last=False)
        yield SpeechChunk(pcm=np.zeros(0, dtype=np.int16), sample_rate=self.sample_rate, last=True)
