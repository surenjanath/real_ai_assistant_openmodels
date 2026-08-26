"""Runtime settings registry: model / voice / speed, switchable live.

Sources of truth:
  * models  - discovered from Ollama /api/tags when reachable, merged with a
              sensible default catalogue (so switching works even before
              Ollama is up; unverified names are accepted but flagged).
  * voices  - the Kokoro-82M v1.0 voice set.
Every change is broadcast to all telemetry clients as a `settings.update`
frame and mirrored onto the live TTS engine, so voice/speed changes are
audible on the very next utterance without a restart.
"""

from __future__ import annotations

import asyncio
import re
import urllib.request
from dataclasses import dataclass, field

from .config import settings
from .logbus import LogBus

DEFAULT_MODELS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen3:8b",
    "qwen2.5:7b",
    "qwen3:30b-a3b",
    "mistral-nemo:12b",
    "gemma3:12b",
    "phi4-mini:3.8b",
    "deepseek-r1:8b",
    "llama3.3:70b",
]

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def _normalise(name: str) -> str:
    name = name.strip().strip(".,!?\"'").lower()
    # Spoken input arrives as "qwen3 8b" - map to the "qwen3:8b" tag form.
    if ":" not in name and " " in name:
        name = name.rsplit(" ", 1)[0] + ":" + name.rsplit(" ", 1)[1]
    return name


@dataclass
class Registry:
    bus: LogBus
    model: str = field(default_factory=lambda: settings.ollama_model)
    voice: str = field(default_factory=lambda: settings.tts_voice)
    speed: float = field(default_factory=lambda: settings.tts_speed)
    models: list[str] = field(default_factory=lambda: sorted({*DEFAULT_MODELS, settings.ollama_model}))
    voices: list[str] = field(default_factory=lambda: list(KOKORO_VOICES))
    model_verified: bool = False
    _engine: object | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ---- discovery -----------------------------------------------------------

    async def refresh_models(self) -> None:
        """Merge live Ollama tags into the catalogue (best effort)."""
        try:
            def fetch() -> dict:
                req = urllib.request.Request(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    import json
                    return json.loads(resp.read().decode())

            info = await asyncio.to_thread(fetch)
            tags = sorted(str(m.get("name", "")).strip() for m in info.get("models", []) if m.get("name"))
            if tags:
                merged = sorted({*self.models, *tags})
                changed = merged != self.models
                self.models = merged
                if self.model in tags:
                    self.model_verified = True
                elif changed:
                    self.bus.publish("info", "settings", f"model catalogue refreshed - {len(tags)} live tag(s) from ollama")
        except Exception:  # noqa: BLE001 - offline is a normal state
            pass

    # ---- application -----------------------------------------------------------

    def attach_engine(self, engine: object) -> None:
        self._engine = engine

    def _mirror_engine(self) -> None:
        engine = self._engine
        if engine is None:
            return
        if hasattr(engine, "voice"):
            engine.voice = self.voice  # type: ignore[attr-defined]
        if hasattr(engine, "speed"):
            engine.speed = self.speed  # type: ignore[attr-defined]

    def _resolve_model(self, value: str) -> tuple[str | None, bool]:
        value = _normalise(value)
        if not value:
            return None, False
        if value in self.models:
            return value, True
        for candidate in self.models:
            if candidate.split(":")[0] == value or candidate.startswith(value):
                return candidate, True
        # Unknown tag: allow it (user may have pulled it after our last probe).
        return (value, False) if re.match(r"^[\w.\-]+(:[\w.\-]+)?$", value) else (None, False)

    def _resolve_voice(self, value: str) -> str | None:
        value = value.strip().strip(".,!?\"'").lower().replace(" ", "_")
        if value in self.voices:
            return value
        for candidate in self.voices:
            if candidate.split("_")[-1] == value or candidate.endswith(value):
                return candidate
        return None

    def apply(
        self,
        model: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
        announce: bool = True,
    ) -> dict:
        """Validate + apply a settings delta; logs and broadcasts every change."""
        applied: dict[str, object] = {}
        errors: list[str] = []

        if model:
            resolved, verified = self._resolve_model(model)
            if resolved:
                if resolved != self.model:
                    self.model = resolved
                    self.model_verified = verified
                    applied["model"] = resolved
                else:
                    verified = self.model_verified
            else:
                errors.append(f"unknown model '{model.strip()}'")

        if voice:
            resolved = self._resolve_voice(voice)
            if resolved:
                if resolved != self.voice:
                    self.voice = resolved
                    applied["voice"] = resolved
            else:
                errors.append(f"unknown voice '{voice.strip()}'")

        if speed is not None:
            try:
                speed_f = max(0.5, min(2.0, round(float(speed), 2)))
                if abs(speed_f - self.speed) > 0.001:
                    self.speed = speed_f
                    applied["speed"] = speed_f
            except (TypeError, ValueError):
                errors.append(f"invalid speed '{speed}'")

        if applied:
            self._mirror_engine()
            if announce:
                parts = [f"{k} -> {v}" for k, v in applied.items()]
                self.bus.publish("success", "settings", "applied " + ", ".join(parts))
            self.broadcast()
        for err in errors:
            self.bus.publish("error", "settings", err)
        return {"ok": not errors, "applied": applied, "errors": errors,
                "settings": self.as_dict()}

    def broadcast(self) -> None:
        """Push the current settings to every telemetry client."""
        self.bus.push_frame({"type": "settings.update", "settings": self.as_dict()})

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "model_verified": self.model_verified,
            "models": self.models,
            "voice": self.voice,
            "voices": self.voices,
            "speed": self.speed,
        }
