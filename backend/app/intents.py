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
    be more concise / switch persona to friday / engineering mode
    enable tools / what skills do you have
    remember that my sister is called Ada
    what do you know about the deploy script
    make a note: rewire the proxy
    remind me to stretch in 20 minutes
    performance report / how fast are you
    louder / quieter / mute

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
    text = re.sub(r"\s+", " ", text.strip())
    # Drop a leading wake word / vocative so "jarvis, status" still matches.
    text = re.sub(r"^(?:hey |ok |okay )?jarvis[\s,:-]+", "", text, flags=re.I)
    return text.strip(" .!?")


#: Words that name a personality rather than describing an attribute, so
#: "be more concise" switches persona while "be more careful" reaches the model.
_PERSONA_WORDS = {
    "jarvis", "concise", "brief", "short", "terse", "engineer", "engineering",
    "technical", "socratic", "curious", "friday", "warm", "casual", "formal",
    "butler",
    # "be angry" / "rage mode" / "trini mode" reach the disposition switch the
    # same way "be more concise" does. Only words that *name* a personality
    # belong here - "be more careful" must still go to the model.
    "rage", "angry", "furious", "aggressive", "sweary", "trini", "trinidad",
    "trinidadian", "caribbean",
}

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
    (r"(?:list|show|what)(?: are)?(?: your| the)? (?:tools|skills|capabilities|abilities)"
     r"|what can you actually do|list capabilities", "tools_list"),
    (r"(?:enable|turn on|allow|use) (?:your )?(?:tools|skills|tool (?:calling|use))"
     r"|tools on|skills on", "tools_on"),
    (r"(?:disable|turn off|stop using|forbid) (?:your )?(?:tools|skills|tool (?:calling|use))"
     r"|tools off|skills off|answer without tools", "tools_off"),
    (r"(?:list|show)(?: me)?(?: the| your| available)? (?:personas?|personalities|voices? modes?)"
     r"|what personas?(?: do you have)?", "persona_list"),
    (r"(?:enable|turn on|use) (?:long[- ]?term |durable |deep )?(?:recall|memory)"
     r"|recall on|remember (?:everything|things) again", "recall_on"),
    (r"(?:disable|turn off|stop using) (?:long[- ]?term |durable |deep )?(?:recall|memory)"
     r"|recall off|stop remembering|amnesia", "recall_off"),
    (r"(?:what|how much) do you remember|memory (?:status|report|stats)"
     r"|(?:show|check) (?:your )?memory", "memory_stats"),
    (r"(?:performance|speed|latency|timing) (?:report|stats|status)|how fast are you"
     r"|benchmark|metrics|diagnostics report", "metrics"),
    (r"(?:list|show|read)(?: me)?(?: my| the)? notes", "notes_list"),
    (r"(?:list|show|what are)(?: my| the)? reminders(?: do i have)?"
     r"|what am i meant to remember", "reminders_list"),
    (r"mute|silence yourself|volume off", "volume_mute"),
    (r"unmute|volume on|speak up again", "volume_unmute"),
    (r"louder|turn (?:it |the volume )?up|volume up|speak louder", "volume_up"),
    (r"quieter|softer|turn (?:it |the volume )?down|volume down|speak quieter", "volume_down"),
    (r"(?:use|switch to|enable)(?: the)? (?:full |whole )?crew(?: mode| pipeline)?"
     r"|crew mode|deep(?:er)? reasoning mode", "crew_on"),
    (r"(?:use|switch to|enable)(?: the)? fast(?: mode| path)?|fast mode|single pass", "crew_off"),
    (r"forget everything you know about me|wipe (?:your )?memory|erase (?:all )?memory"
     r"|delete (?:all )?(?:my )?data", "memory_wipe"),
    (r"who are you|what are you|introduce yourself|identify yourself", "identity"),
    (r"(?:enable|turn on|start|use|allow) (?:deep |extended |chain of )?think(?:ing)?(?: mode)?"
     r"|think (?:harder|deeply|more|it through)|deep think(?:ing)?(?: mode)?(?: on)?"
     r"|reason (?:harder|more|deeply)", "think_on"),
    (r"(?:disable|turn off|stop|no|skip) (?:deep |extended |chain of )?think(?:ing)?(?: mode)?"
     r"|(?:answer|reply|respond) (?:faster|quickly|immediately|directly)|stop overthinking"
     r"|don'?t think(?: so hard| about it)?|deep think(?:ing)?(?: mode)? off", "think_off"),
]


def _capture(match: re.Match[str], original: str, group: int = 1) -> str:
    """Pull a captured group with its original capitalisation.

    Matching happens against a lowercased string so the patterns stay simple,
    but a stored fact or note should read back as the user wrote it. Slicing
    the original by the match span recovers that, as long as lowercasing did
    not change the length (it can for a few non-ASCII characters, in which
    case the lowercased text is still perfectly usable).
    """
    start, end = match.span(group)
    if len(original) == len(match.string) and start >= 0:
        return original[start:end].strip()
    return match.group(group).strip()


def parse_intent(text: str) -> Intent | None:
    original = _norm(text)
    t = original.lower()
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

    # --- persona -------------------------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:switch|change|set|use|become|be)(?: the| your)? "
        rf"(?:persona(?:lity)?|character|mode|manner|tone)(?: (?:to|as|into))? ([\w .\-]+)",
        t,
    )
    if match:
        return Intent("persona_set", value=match.group(1).strip(), raw=t)
    match = re.fullmatch(rf"{_LEAD}be (?:more |much more )?([\w\-]+)(?: please)?", t)
    if match and match.group(1) in _PERSONA_WORDS:
        return Intent("persona_set", value=match.group(1), raw=t)
    match = re.fullmatch(rf"{_LEAD}([\w\-]+) mode(?: please)?", t)
    if match and match.group(1) in _PERSONA_WORDS:
        return Intent("persona_set", value=match.group(1), raw=t)

    # --- durable memory --------------------------------------------------------
    # "remember that my sister is called Ada" / "remember: the wifi code is 1234"
    match = re.fullmatch(
        rf"{_LEAD}(?:remember|memorise|memorize|store|keep in mind|note)"
        rf"(?: that| this)?[:,]? (.+)",
        t,
    )
    if match:
        return Intent("remember", value=_capture(match, original), raw=t)

    # "what do you know about X" / "what did i say about X"
    match = re.fullmatch(
        rf"{_LEAD}what (?:do you (?:know|remember)|did (?:i|we) (?:say|tell you|discuss))"
        rf"(?: about| regarding)? (.+)",
        t,
    )
    if match:
        return Intent("recall_query", value=_capture(match, original), raw=t)
    match = re.fullmatch(rf"{_LEAD}(?:recall|look up|search (?:your )?memory (?:for|about)) (.+)", t)
    if match:
        return Intent("recall_query", value=_capture(match, original), raw=t)

    # --- notes ------------------------------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:make|take|write|add|jot down|create)(?: a| me a)? note"
        rf"(?: (?:that|saying|about))?[:,]? (.+)",
        t,
    )
    if match:
        return Intent("note_add", value=_capture(match, original), raw=t)

    # --- reminders ---------------------------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:remind me|set a reminder|remind us)(?: to| that| about)? (.+)", t
    )
    if match:
        return Intent("reminder_add", value=_capture(match, original), raw=t)
    match = re.fullmatch(
        rf"{_LEAD}(?:set|start)(?: a)? timer(?: for)? (.+)", t
    )
    if match:
        return Intent("reminder_add", value=f"timer finished in {_capture(match, original)}", raw=t)

    # --- volume -------------------------------------------------------------------
    match = re.fullmatch(
        rf"{_LEAD}(?:set|change|make)(?: the| your)? volume(?: to| at)? "
        rf"([0-9]*\.?[0-9]+)\s*(%|percent)?",
        t,
    )
    if match:
        raw_value = float(match.group(1))
        # "volume 70" and "volume 0.7" both mean the same thing.
        value = raw_value / 100 if (match.group(2) or raw_value > 1) else raw_value
        return Intent("volume_set", value=f"{min(1.0, max(0.0, value)):.2f}", raw=t)

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
