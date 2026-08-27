"""Natural-language control intents, checked before the agent crew runs.

Lets the user speak or type things like::

    settings / open settings
    list models / what models do you have
    switch model to qwen3 8b        (spoken form of qwen3:8b)
    change voice to af_heart
    speak slower / faster / set speed to 1.2
    what time is it / what's the date
    stop / never mind
    clear the log
    help
    status

Anything that does not match falls through to the crew unchanged.

Design note: every pattern here is deliberately **anchored**. The previous
implementation searched for bare words, so an ordinary question like "how do I
make this loop run faster?" was swallowed as a speech-rate command and never
reached the crew. Control phrases must look like commands addressed at the
assistant itself, not merely contain a keyword.
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
    text = re.sub(r"\s+", " ", text.strip().lower())
    # Drop a leading wake word / vocative so "jarvis, status" still matches.
    text = re.sub(r"^(?:hey |ok |okay )?jarvis[\s,:-]+", "", text)
    return text.strip(" .!?")


_MODELS = r"(?:models?|llms?|brains?)"
_VOICES = r"(?:voices?|narrators?|accents?)"
# Optional polite lead-in, e.g. "can you please ..." / "could you ...".
_LEAD = r"(?:(?:can|could|would|will) you )?(?:please )?"

_EXACT: list[tuple[str, str]] = [
    (r"settings|open settings|show settings|configuration|preferences|settings panel|system settings|config",
     "settings_show"),
    (r"close settings|hide settings|dismiss settings", "settings_hide"),
    (r"help|commands|what can you do|what can i say", "help"),
    (r"status|system status|report status|sitrep|report|diagnostics?|run diagnostics?|self test",
     "status"),
    (r"stop|stop talking|be quiet|quiet|silence|shut up|never ?mind|cancel|abort|halt|belay that",
     "stop"),
    (r"clear|clear (?:the )?(?:log|logs|screen|terminal|console)|wipe (?:the )?logs?|cls",
     "clear"),
    (r"(?:new|reset|forget|clear) (?:the )?(?:conversation|context|memory|chat)|start over|forget everything",
     "memory_clear"),
    (r"(?:hey )?(?:what(?:'?s| is)(?: the)? )?(?:current )?time(?: is it)?(?: now| please)?"
     r"|what time is it(?: now| right now| please)?|do you (?:have|know) the time"
     r"|tell me the time|clock", "time"),
    (r"(?:what(?:'?s| is)(?: the)? )?(?:current |today'?s )?date(?: today| now| please)?"
     r"|what(?:'?s| is)(?: the)? date today|what day is it(?: today)?"
     r"|what(?:'?s| is) today(?:'s date)?|tell me the date"
     r"|what(?:'?s| is) the day(?: today)?", "date"),
    (r"repeat(?: that)?|say (?:that )?again|come again|pardon", "repeat"),
    (r"who are you|what are you|introduce yourself|identify yourself", "identity"),
    (r"(?:enable|turn on|start|use|allow) (?:deep |extended |chain of )?think(?:ing)?(?: mode)?"
     r"|think (?:harder|deeply|more|it through)|deep think(?:ing)?(?: mode)?(?: on)?"
     r"|reason (?:harder|more|deeply)", "think_on"),
    (r"(?:disable|turn off|stop|no|skip) (?:deep |extended |chain of )?think(?:ing)?(?: mode)?"
     r"|(?:answer|reply|respond) (?:faster|quickly|immediately|directly)|stop overthinking"
     r"|don'?t think(?: so hard| about it)?|deep think(?:ing)?(?: mode)? off", "think_off"),
]


def parse_intent(text: str) -> Intent | None:
    t = _norm(text)
    if not t:
        return None

    for pattern, kind in _EXACT:
        if re.fullmatch(pattern, t):
            return Intent(kind, raw=t)

    # --- listings -----------------------------------------------------------
    # Anchored: must *start* with an enumeration verb aimed at models/voices.
    if re.fullmatch(rf"{_LEAD}(?:list|show|enumerate|name)(?: me)?(?: (?:all|the|your|available))* {_MODELS}", t):
        return Intent("models_list", raw=t)
    if re.fullmatch(rf"{_LEAD}(?:which|what) {_MODELS}(?: (?:do you have|are available|can you use|are installed))?", t):
        return Intent("models_list", raw=t)
    if re.fullmatch(rf"{_LEAD}(?:list|show|enumerate|name)(?: me)?(?: (?:all|the|your|available))* {_VOICES}", t):
        return Intent("voices_list", raw=t)
    if re.fullmatch(rf"{_LEAD}(?:which|what) {_VOICES}(?: (?:do you have|are available|can you use))?", t):
        return Intent("voices_list", raw=t)

    # --- model / voice switching --------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:switch|change|use|set|load|run|make)(?: the| your)? {_MODELS}"
        rf"(?: (?:to|as|into))? ([\w.\-: ]+)",
        t,
    )
    if match:
        return Intent("model_set", value=match.group(1).strip(), raw=t)
    # "switch to qwen3 8b" - model implied.
    match = re.fullmatch(rf"{_LEAD}(?:switch|change) to (?:the )?([\w.\-:]+(?: [\d.]+b)?)", t)
    if match:
        return Intent("model_set", value=match.group(1).strip(), raw=t)

    match = re.fullmatch(
        rf"{_LEAD}(?:switch|change|use|set|load|make)(?: the| your)? {_VOICES}"
        rf"(?: (?:to|as|into))? ([\w.\-: ]+)",
        t,
    )
    if match:
        return Intent("voice_set", value=match.group(1).strip(), raw=t)
    # "speak as bm_george" / "talk like af heart"
    match = re.fullmatch(rf"{_LEAD}(?:speak|talk|sound)(?: to me)? (?:as|like|with) (?:the )?([\w.\- ]+?)(?: voice)?", t)
    if match:
        return Intent("voice_set", value=match.group(1).strip(), raw=t)

    # --- speech rate ---------------------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:set|change|adjust|make)(?: the| your)? (?:speech |speaking |talking )?"
        rf"(?:speed|rate|pace)(?: to| at)? ([0-9]*\.?[0-9]+)x?",
        t,
    )
    if match:
        return Intent("speed_set", value=match.group(1), raw=t)

    # Anchored so "make the build faster" never reaches here.
    if re.fullmatch(rf"{_LEAD}(?:speak|talk|read|say it|go)(?: a bit| a little| much)? (?:faster|quicker|more quickly)", t) \
       or re.fullmatch(rf"{_LEAD}speed up(?: your speech| your voice)?", t):
        return Intent("speed_up", raw=t)
    if re.fullmatch(rf"{_LEAD}(?:speak|talk|read|say it|go)(?: a bit| a little| much)? (?:slower|more slowly)", t) \
       or re.fullmatch(rf"{_LEAD}slow down(?: your speech| your voice)?", t):
        return Intent("speed_down", raw=t)

    return None
