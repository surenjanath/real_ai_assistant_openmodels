"""Runtime settings registry: model / voice / speed, switchable live.

Sources of truth:
  * installed - the tags Ollama actually reports from /api/tags. This is the
    ONLY thing that makes a model "verified"; the aspirational catalogue below
    must never be allowed to vouch for a model that was never pulled.
  * models    - installed tags merged with a suggestion catalogue, so the
    dropdown is useful before Ollama is up (unverified entries are flagged).
  * voices    - the Kokoro-82M v1.0 voice set, grouped by accent/gender.

Every change is broadcast to all telemetry clients as a `settings.update`
frame and mirrored onto the live TTS engine, so voice/speed changes are
audible on the very next utterance without a restart.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from dataclasses import dataclass, field

from .config import settings
from .logbus import LogBus

# Suggestions only - shown in the picker, never treated as installed.
DEFAULT_MODELS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen3:8b",
    "qwen2.5:7b",
    "mistral-nemo:12b",
    "gemma3:12b",
    "phi4-mini:3.8b",
    "deepseek-r1:8b",
]

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

# Human labels for the settings panel.
VOICE_LABELS = {
    "a": "American female", "am": "American male",
    "b": "British female", "bm": "British male",
}

# Preference order when auto-selecting a crew model from what is installed:
# small-and-fast first, so a cold machine still feels responsive.
_AUTO_PREFERENCE = [
    r"qwen3.*:(\d|1[0-4])b", r"llama3\.[12]:(3|8)b", r"gemma\d?:\d+b",
    r"mistral", r"phi", r"qwen", r"llama", r".*",
]


def _normalise(name: str) -> str:
    name = name.strip().strip(".,!?\"'").lower()
    # Spoken input arrives as "qwen3 8b" - map to the "qwen3:8b" tag form.
    if ":" not in name and " " in name:
        head, _, tail = name.rpartition(" ")
        # Only join when the tail looks like a size/variant tag ("8b", "3.8b").
        if re.fullmatch(r"[\d.]+[a-z]?", tail):
            name = f"{head}:{tail}"
    return name


def voice_label(voice: str) -> str:
    prefix = voice.split("_")[0].lower()
    key = prefix if prefix in VOICE_LABELS else prefix[:1]
    accent = VOICE_LABELS.get(key, "voice")
    return f"{voice.split('_', 1)[-1].title()} ({accent})"


@dataclass
class Registry:
    bus: LogBus
    model: str = field(default_factory=lambda: settings.ollama_model)
    voice: str = field(default_factory=lambda: settings.tts_voice)
    speed: float = field(default_factory=lambda: settings.tts_speed)
    models: list[str] = field(default_factory=lambda: sorted(DEFAULT_MODELS))
    #: tags Ollama actually reports - the only source of "verified".
    installed: list[str] = field(default_factory=list)
    voices: list[str] = field(default_factory=lambda: list(KOKORO_VOICES))
    #: extended chain-of-thought, opt-in (see Settings.think)
    think: bool = field(default_factory=lambda: settings.think)
    #: model tag -> capability list, as reported by Ollama /api/tags
    capabilities: dict[str, list[str]] = field(default_factory=dict)
    _engine: object | None = None

    @property
    def model_verified(self) -> bool:
        """True only when the crew model is genuinely present on the server."""
        return self.model in self.installed

    @property
    def think_supported(self) -> bool:
        """Whether the current model advertises the `thinking` capability."""
        return "thinking" in self.capabilities.get(self.model, [])

    @property
    def think_active(self) -> bool:
        return self.think and self.think_supported

    # ---- discovery -----------------------------------------------------------

    async def refresh_models(self) -> list[str]:
        """Pull live Ollama tags. Returns the installed list (empty if offline)."""
        try:
            def fetch() -> dict:
                req = urllib.request.Request(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                with urllib.request.urlopen(req, timeout=4) as resp:
                    return json.loads(resp.read().decode())

            info = await asyncio.to_thread(fetch)
        except Exception:  # noqa: BLE001 - offline is a normal state
            return list(self.installed)

        entries = [m for m in info.get("models", []) if m.get("name")]
        tags = sorted({str(m["name"]).strip() for m in entries})
        if not tags:
            return list(self.installed)
        self.capabilities = {
            str(m["name"]).strip(): list(m.get("capabilities") or []) for m in entries
        }
        changed = tags != self.installed
        self.installed = tags
        self.models = sorted({*DEFAULT_MODELS, *tags})
        if changed:
            self.bus.publish("info", "settings", f"ollama catalogue: {len(tags)} model(s) installed")
        return tags

    def autoselect_model(self) -> str | None:
        """Pick a sensible installed model when none is configured/pulled.

        Returns the newly chosen tag, or None if nothing changed.
        """
        if not self.installed or self.model_verified:
            return None
        previous = self.model
        for pattern in _AUTO_PREFERENCE:
            for tag in self.installed:
                if re.match(pattern, tag, re.I):
                    self.model = tag
                    if previous:
                        self.bus.publish(
                            "warn", "settings",
                            f"'{previous}' is not installed - auto-selected '{tag}'",
                        )
                    else:
                        self.bus.publish("success", "settings", f"crew model auto-selected: {tag}")
                    self.broadcast()
                    return tag
        return None

    # ---- application -----------------------------------------------------------

    def attach_engine(self, engine: object) -> None:
        self._engine = engine
        self._mirror_engine()

    def _mirror_engine(self) -> None:
        engine = self._engine
        if engine is None:
            return
        if hasattr(engine, "voice"):
            engine.voice = self.voice  # type: ignore[attr-defined]
        if hasattr(engine, "speed"):
            engine.speed = self.speed  # type: ignore[attr-defined]

    def _resolve_model(self, value: str) -> str | None:
        value = _normalise(value)
        if not value:
            return None
        pool = self.installed or self.models
        if value in pool:
            return value
        # Prefer an exact family match among installed tags.
        for candidate in pool:
            if candidate.split(":")[0] == value:
                return candidate
        for candidate in pool:
            if candidate.startswith(value) or value in candidate:
                return candidate
        # Unknown tag: allow it (the user may pull it next), if it looks like one.
        return value if re.fullmatch(r"[\w.\-]+(:[\w.\-]+)?", value) else None

    def _resolve_voice(self, value: str) -> str | None:
        value = value.strip().strip(".,!?\"'").lower().replace(" ", "_").replace("-", "_")
        if value in self.voices:
            return value
        for candidate in self.voices:
            if candidate.split("_")[-1] == value:
                return candidate
        for candidate in self.voices:
            if candidate.endswith(value) or value in candidate:
                return candidate
        return None

    def apply(
        self,
        model: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
        think: bool | None = None,
        announce: bool = True,
    ) -> dict:
        """Validate + apply a settings delta; logs and broadcasts every change."""
        applied: dict[str, object] = {}
        errors: list[str] = []

        if model:
            resolved = self._resolve_model(model)
            if resolved:
                if resolved != self.model:
                    self.model = resolved
                    applied["model"] = resolved
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

        if think is not None:
            want = bool(think)
            if want and not self.think_supported:
                errors.append(f"'{self.model}' has no thinking mode")
            elif want != self.think:
                self.think = want
                applied["think"] = want

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
            "installed": self.installed,
            "voice": self.voice,
            "voices": self.voices,
            "voice_labels": {v: voice_label(v) for v in self.voices},
            "speed": self.speed,
            "think": self.think,
            "think_supported": self.think_supported,
            "think_active": self.think_active,
        }
