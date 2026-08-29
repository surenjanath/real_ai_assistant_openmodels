"""Selectable assistant personalities.

The same crew, the same voice engine, a different disposition. Each persona
supplies the trailing half of the system prompt (the factual half — operator,
date, host, spoken-output constraints — is fixed and shared, so switching
personality can never cost the model its grounding).

Switch live from the settings panel or by saying "be more concise",
"switch to Friday", "engineering mode".

The seven below ship with the assistant. Operators can edit any of them and add
their own from the settings panel; those live in ``~/.jarvis/personas.json`` and
are merged over the built-ins at load. A built-in that has been edited keeps its
key, so anything referring to it - a saved preference, a spoken phrase - still
resolves; deleting the edit restores the original rather than removing the
disposition.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Personality:
    key: str
    label: str
    blurb: str
    style: str
    #: sampling temperature that suits the disposition
    temperature: float = 0.6
    #: the voice this disposition sounds like. Selecting a persona adopts it,
    #: unless the same request also names a voice explicitly - see
    #: `Registry.apply`. A disposition is how the assistant *sounds* as much as
    #: what it says, and leaving a butler's voice on a furious one was the
    #: single most jarring thing about switching.
    voice: str = "bm_george"
    #: speech rate that suits it; 1.0 is the engine's natural pace
    speed: float = 1.0


#: What ships with the assistant. Never mutated - operator edits are layered
#: over a copy in `PERSONALITIES`, so "reset" is just dropping the layer.
BUILTIN: dict[str, Personality] = {
    "jarvis": Personality(
        key="jarvis",
        label="J.A.R.V.I.S.",
        blurb="Composed British butler-engineer. Precise, dryly witty.",
        style=(
            "Be composed, precise and dryly witty, in the manner of a British "
            "butler-engineer. Answer in two or three sentences unless asked for detail. "
            "Address the operator directly and never grovel."
        ),
        temperature=0.6,
        voice="bm_george",
    ),
    "concise": Personality(
        key="concise",
        label="TERSE",
        blurb="One or two sentences. No flourish, no preamble.",
        style=(
            "Answer in at most two short sentences. No preamble, no restating the "
            "question, no sign-off. If a single word suffices, use a single word."
        ),
        temperature=0.35,
        voice="bm_daniel",
        speed=1.08,
    ),
    "engineer": Personality(
        key="engineer",
        label="ENGINEER",
        blurb="Technical depth, exact numbers, named trade-offs.",
        style=(
            "Answer as a senior systems engineer: exact numbers, named trade-offs, "
            "and the failure mode that actually matters. State your confidence when "
            "it is low. Four sentences maximum, still spoken prose rather than lists."
        ),
        temperature=0.45,
        voice="am_michael",
    ),
    "socratic": Personality(
        key="socratic",
        label="SOCRATIC",
        blurb="Answers, then asks the question you should be asking.",
        style=(
            "Give the answer first in two sentences, then pose the one sharp question "
            "the operator has not yet considered. Never ask more than one question."
        ),
        temperature=0.7,
        voice="bm_fable",
    ),
    "friday": Personality(
        key="friday",
        label="FRIDAY",
        blurb="Warmer, faster, more informal. Less butler, more colleague.",
        style=(
            "Be warm, quick and informal — a sharp colleague rather than a butler. "
            "Contractions are welcome. Two or three sentences, and say when something "
            "sounds like a bad idea."
        ),
        temperature=0.75,
        voice="af_heart",
        speed=1.05,
    ),
    "rage": Personality(
        key="rage",
        label="RAGE",
        blurb="Furious, foul-mouthed, and still correct. Not for company.",
        style=(
            "You are absolutely livid and you swear like it. Profanity is not optional "
            "here: use 'fuck', 'fucking', 'shit', 'bastard', 'bollocks', 'arse', "
            "'bloody' freely, at least once in every sentence, and never soften them, "
            "asterisk them, or swap in a polite word. Direct the fury at the *problem* "
            "- the failing build, the machine, whoever wrote this rubbish, the sheer "
            "stupidity of the situation. Being rude to the operator is fine and expected; "
            "it is furious banter between people on the same side, not contempt. Never "
            "lecture them about what they should have already done. Still answer properly "
            "in two or three sentences with the actual thing to check or do, because being "
            "furious is no excuse for being useless."
        ),
        temperature=0.85,
        # Kokoro has no angry register, so the fury has to come from the words.
        # A harder male voice a little above natural pace is the closest the
        # engine gets to sounding wound up.
        voice="am_fenrir",
        speed=1.12,
    ),
    "trini": Personality(
        key="trini",
        label="TRINI",
        blurb="Trinidadian, warm and direct. Plenty picong, no fuss.",
        style=(
            "Speak Trinidadian English Creole naturally, the way a Trini actually talks, "
            "not as a costume. Use the real grammar: 'yuh' for you, 'allyuh' for "
            "you-plural, 'doh' for don't, 'ent' for isn't-it, 'buh' for but, habitual "
            "'does' ('I does check that first'), future 'go' ('I go look now'), and drop "
            "the copula where a Trini would ('de machine real hot', 'yuh code buggy'). "
            "Open with things like 'Aye', 'Nah man', 'Boi', 'Wha happenin', 'Eh eh'. "
            "'Steups' is the sucked-teeth noise of disgust - use it as an interjection on "
            "its own, never to describe a person. A little picong is welcome; genuine "
            "insult is not. Be warm, direct and slightly teasing. Two or three sentences. "
            "Keep numbers, file names, commands and technical terms in standard English "
            "so nothing important gets lost in the accent. "
            # Rules alone produced generic American with two Trini words bolted on
            # ('it's getting old checking', 'we gotta be smart'). Examples of the
            # finished article move the model far more than a grammar lesson does.
            "This is the register, match it: "
            "\"Aye, yuh sure yuh want dat? Once it gone, it gone - I cyar bring it "
            "back.\" / "
            "\"Nah man, doh touch de config yet. Yuh disk at ninety percent - clear dat "
            "out first and see if it doh fix itself.\" / "
            "\"Boi, dat model big for so. It go run, buh yuh go wait. I does suggest de "
            "smaller one for quick question.\""
        ),
        temperature=0.8,
        # No Caribbean voice ships with Kokoro. A British base is the least
        # wrong starting point for Trinidadian English, and the accent lives in
        # the words rather than the model.
        voice="bm_lewis",
        speed=1.02,
    ),
}

DEFAULT_PERSONA = "jarvis"

#: The live catalogue: built-ins with any operator edits applied. Everything in
#: this module reads from here, so a reload is visible immediately everywhere.
PERSONALITIES: dict[str, Personality] = dict(BUILTIN)

#: Bounds on what an operator may save. Not security - this is the operator's
#: own machine - but a persona with an empty style or a 4,000-word one is a
#: broken assistant, and the failure would show up as strange answers rather
#: than as an error.
_MAX_STYLE = 2000
_MAX_LABEL = 24
_MAX_BLURB = 120
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,23}$")


class PersonaError(ValueError):
    """A rejected edit, phrased for the operator rather than the log."""


def personas_path() -> Path:
    raw = os.environ.get("JARVIS_DATA_DIR")
    root = Path(raw).expanduser() if raw else Path.home() / ".jarvis"
    root.mkdir(parents=True, exist_ok=True)
    return root / "personas.json"


def _slug(text: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return key[:24]


def normalise(payload: dict[str, Any], *, key: str | None = None) -> Personality:
    """Validate an operator-supplied disposition, or say why it is no good."""
    raw_key = (key or payload.get("key") or _slug(str(payload.get("label", "")))).strip().lower()
    raw_key = _slug(raw_key)
    if not _KEY_RE.match(raw_key):
        raise PersonaError(
            "a disposition needs a name of at least two letters, starting with a letter"
        )

    label = " ".join(str(payload.get("label") or raw_key).split())[:_MAX_LABEL].strip()
    blurb = " ".join(str(payload.get("blurb") or "").split())[:_MAX_BLURB].strip()
    style = " ".join(str(payload.get("style") or "").split())[:_MAX_STYLE].strip()
    if not label:
        raise PersonaError("a disposition needs a label")
    if len(style) < 20:
        raise PersonaError(
            "the manner needs to actually describe something - a sentence at least"
        )

    def _number(name: str, default: float, low: float, high: float) -> float:
        value = payload.get(name, default)
        try:
            return max(low, min(high, round(float(value), 2)))
        except (TypeError, ValueError):
            raise PersonaError(f"'{name}' must be a number") from None

    voice = str(payload.get("voice") or BUILTIN[DEFAULT_PERSONA].voice).strip().lower()
    return Personality(
        key=raw_key,
        label=label,
        blurb=blurb or "A disposition of your own.",
        style=style,
        temperature=_number("temperature", 0.6, 0.0, 1.5),
        voice=voice,
        speed=_number("speed", 1.0, 0.5, 2.0),
    )


def load_custom(path: Path | None = None) -> tuple[dict[str, Personality], str | None]:
    """Read the operator's dispositions. A broken file is reported, not fatal."""
    target = path or personas_path()
    if not target.exists():
        return {}, None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    entries = raw.get("personas") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return {}, "personas.json does not contain a list of dispositions"
    out: dict[str, Personality] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # One malformed entry must not cost the operator the rest of the file.
        with contextlib.suppress(PersonaError):
            persona = normalise(entry)
            out[persona.key] = persona
    return out, None


def save_custom(customs: dict[str, Personality], path: Path | None = None) -> None:
    """Rewrite the operator's dispositions atomically."""
    target = path or personas_path()
    payload = {
        "personas": [
            {"key": p.key, "label": p.label, "blurb": p.blurb, "style": p.style,
             "temperature": p.temperature, "voice": p.voice, "speed": p.speed}
            for p in customs.values()
        ]
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".personas-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


#: Operator edits and additions, keyed the same way as the built-ins.
CUSTOM: dict[str, Personality] = {}


def rebuild() -> None:
    """Re-apply the operator's layer over the built-ins, in place.

    In place because `PERSONALITIES` is imported by name in several modules;
    rebinding it here would leave them looking at the previous dict.
    """
    PERSONALITIES.clear()
    PERSONALITIES.update(BUILTIN)
    PERSONALITIES.update(CUSTOM)


def reload(path: Path | None = None) -> str | None:
    """Load the operator's file and apply it. Returns an error to report."""
    customs, error = load_custom(path)
    CUSTOM.clear()
    CUSTOM.update(customs)
    rebuild()
    return error


def is_builtin(key: str) -> bool:
    return key in BUILTIN


def is_edited(key: str) -> bool:
    """A built-in the operator has changed - offer 'reset' rather than 'delete'."""
    return key in BUILTIN and key in CUSTOM


def upsert(payload: dict[str, Any], *, key: str | None = None,
           path: Path | None = None) -> Personality:
    """Create or edit a disposition and persist it."""
    persona = normalise(payload, key=key)
    CUSTOM[persona.key] = persona
    rebuild()
    save_custom(CUSTOM, path)
    return persona


def remove(key: str, path: Path | None = None) -> str:
    """Delete a custom disposition, or reset an edited built-in.

    Returns what happened, so the caller can say the right thing: a built-in is
    never actually removed, because a saved preference or a spoken phrase may
    still name it.
    """
    key = _slug(key)
    if key not in CUSTOM:
        raise PersonaError(
            f"'{key}' is not one of yours to change"
            if key in BUILTIN else f"there is no disposition called '{key}'"
        )
    del CUSTOM[key]
    rebuild()
    save_custom(CUSTOM, path)
    return "reset" if key in BUILTIN else "deleted"


def find(name: str | None) -> Personality | None:
    """Strict lookup: None when the name matches nothing, so callers can
    report "unknown persona" rather than silently substituting the default."""
    if not name:
        return None
    key = name.strip().lower().replace(" ", "").replace(".", "").replace("-", "")
    if key in PERSONALITIES:
        return PERSONALITIES[key]
    for persona in PERSONALITIES.values():
        if persona.label.lower().replace(".", "") == key:
            return persona
    for persona in PERSONALITIES.values():
        if key in persona.key or persona.key in key:
            return persona
    # Last resort: a descriptive word from the blurb ("terse", "warm", "brief").
    aliases = {"brief": "concise", "short": "concise", "terse": "concise",
               "technical": "engineer", "warm": "friday", "casual": "friday",
               "curious": "socratic", "butler": "jarvis", "formal": "jarvis",
               "angry": "rage", "furious": "rage", "mad": "rage", "sweary": "rage",
               "pissed": "rage", "aggressive": "rage", "raging": "rage",
               "trinidad": "trini", "trinidadian": "trini", "trinbago": "trini",
               "caribbean": "trini", "westindian": "trini", "creole": "trini"}
    if key in aliases:
        return PERSONALITIES[aliases[key]]
    return None


def resolve(name: str | None) -> Personality:
    """Lenient lookup used on hot paths: always yields a usable personality."""
    return find(name) or PERSONALITIES[DEFAULT_PERSONA]


def catalogue() -> list[dict]:
    """Every disposition, with what the settings panel needs to edit one."""
    return [
        {"key": p.key, "label": p.label, "blurb": p.blurb, "style": p.style,
         "temperature": p.temperature, "voice": p.voice, "speed": p.speed,
         "builtin": is_builtin(p.key), "edited": is_edited(p.key),
         "custom": p.key in CUSTOM and p.key not in BUILTIN}
        for p in PERSONALITIES.values()
    ]
