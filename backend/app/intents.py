"""Natural-language settings intents, checked before the agent crew runs.

Lets the user speak or type things like:

    settings / open settings
    list models / what models do you have
    switch model to qwen3 8b        (spoken form of qwen3:8b)
    change voice to af_heart
    speak slower / faster / set speed to 1.2
    list voices
    help
    status

Anything that does not match falls through to the crew unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    kind: str
    value: str | None = None
    raw: str | None = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_MODELS = r"(models?|llms?)"
_VOICES = r"(voices?|narrators?)"


def parse_intent(text: str) -> Intent | None:
    t = _norm(text)

    if t in ("settings", "open settings", "show settings", "configuration",
             "preferences", "settings panel", "system settings"):
        return Intent("settings_show")

    if re.fullmatch(r"(help|commands|what can you do\??)", t):
        return Intent("help")

    if re.fullmatch(r"(status|system status|report status)", t):
        return Intent("status")

    if re.search(rf"^(list|show|enumerate|which|what).{{0,16}}{_MODELS}", t) or re.fullmatch(_MODELS, t):
        return Intent("models_list")
    if re.search(rf"^(list|show|enumerate|which|what).{{0,16}}{_VOICES}", t):
        return Intent("voices_list")

    match = re.search(
        rf"(?:switch|change|use|set|load|make)\s+(?:the\s+)?{_MODELS}\s+(?:to|as|is|:)?\s*([\w.\-: ]+)$", t
    )
    if match:
        return Intent("model_set", value=match.group(2).strip(), raw=t)

    match = re.search(
        rf"(?:switch|change|use|set|load|make)\s+(?:the\s+)?{_VOICES}\s+(?:to|as|is|:)?\s*([\w.\-: ]+)$", t
    )
    if match:
        return Intent("voice_set", value=match.group(2).strip(), raw=t)

    match = re.search(r"(?:set|change|adjust)\s+(?:the\s+)?speed\s+(?:to|at|is|:)?\s*([0-9]*\.?[0-9]+)", t)
    if match:
        return Intent("speed_set", value=match.group(1))

    if re.search(r"\b(speak |talk |read )?(faster|speed up|quicker)\b", t):
        return Intent("speed_up")
    if re.search(r"\b(speak |talk |read )?(slower|slow down)\b", t):
        return Intent("speed_down")

    return None
