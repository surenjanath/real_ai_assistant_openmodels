#!/usr/bin/env python3
"""Pure-logic checks for the safety boundaries and the durable stores.

No server, no model, no audio device, no network — every case here is decided
before a socket would be opened, which is the whole point: the guard has to
refuse `http://127.0.0.1:11434` without ever contacting it.

Run via `make test`.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory import Memory  # noqa: E402
from app.prefs import Prefs  # noqa: E402
from app.skills import (  # noqa: E402
    SkillError,
    _resolve_scratch,
    check_egress,
    convert,
    scratch_root,
)

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"PASS {label}")
    else:
        _failures += 1
        print(f"FAIL {label}" + (f"  ({detail})" if detail else ""))


def refuses(label: str, fn, *args) -> None:
    """Assert that `fn(*args)` raises SkillError rather than returning."""
    try:
        fn(*args)
    except SkillError:
        print(f"PASS {label}")
        return
    global _failures
    _failures += 1
    print(f"FAIL {label}  (it was allowed)")


# ------------------------------------------------------------------- egress

print("\n-- outbound URL guard --")

for target, why in [
    ("http://127.0.0.1:11434/api/tags", "the local Ollama admin API"),
    ("http://localhost:8000/api/health", "this very backend"),
    ("http://169.254.169.254/latest/meta-data/", "the cloud metadata endpoint"),
    ("https://192.168.1.1/", "the local router"),
    ("https://10.0.0.5/", "a private range"),
    ("http://[::1]:3000/", "loopback over IPv6"),
    ("file:///etc/passwd", "a non-http scheme"),
    ("https://user:secret@example.com/", "credentials embedded in the URL"),
    ("http://example.com:22/", "a non-web port"),
]:
    refuses(f"refuses {why}", check_egress, target)

check("allows an ordinary public URL", bool(check_egress("https://example.com/page")))

# ------------------------------------------------------------------- writes

print("\n-- write containment --")

root = scratch_root()
check("writes land in the scratch folder", _resolve_scratch("draft.md").parent == root)
for bad, why in [
    ("../../.ssh/authorized_keys", "a traversal out of the folder"),
    ("/etc/passwd", "an absolute path"),
    ("payload.exe", "a non-text file"),
    ("", "an empty filename"),
]:
    refuses(f"refuses {why}", _resolve_scratch, bad)

# ------------------------------------------------------------- conversions

print("\n-- unit conversion --")

cases = [
    ((100, "c", "f"), 212.0),
    ((32, "f", "c"), 0.0),
    ((0, "c", "k"), 273.15),
    ((1, "mile", "km"), 1.609344),
    ((1, "gb", "mb"), 1024.0),
    ((60, "mph", "kph"), 96.56064),
]
for args, expected in cases:
    got = convert(*args)[0]
    check(f"{args[0]} {args[1]} -> {args[2]}", abs(got - expected) < 1e-6, f"got {got}")

refuses("refuses a cross-dimension conversion", convert, 1, "kg", "metre")
refuses("refuses an unknown unit", convert, 1, "furlong-per-fortnight", "m")

# Temperature is affine, so a round trip is the case most likely to drift.
there, _, _ = convert(37.0, "celsius", "fahrenheit")
back, _, _ = convert(there, "fahrenheit", "celsius")
check("a temperature round trip is stable", abs(back - 37.0) < 1e-9, f"got {back}")

# ----------------------------------------------------------------- prefs

print("\n-- durable preferences --")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "settings.json"
    prefs = Prefs(path)
    check("a missing file reads as empty", prefs.values == {})

    prefs.save({"voice": "af_bella", "speed": 1.35, "not_a_setting": True})
    reloaded = Prefs(path)
    check("saved values survive a restart",
          reloaded.get("voice") == "af_bella" and reloaded.get("speed") == 1.35)
    check("unknown keys are dropped", "not_a_setting" not in reloaded.values)

    path.write_text("{ this is not json")
    broken = Prefs(path)
    check("a corrupt file degrades to empty", broken.values == {})
    check("a corrupt file is reported", broken.error is not None)

    broken.save({"persona": "friday"})
    check("a corrupt file can be overwritten", Prefs(path).get("persona") == "friday")
    Prefs(path).clear()
    check("clearing removes the file", not path.exists())

# ---------------------------------------------------------------- sessions

print("\n-- session archive --")

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "jarvis.db"

    first = Memory(db)
    first.add_turn("user", "what is the capital of France")
    first.add_turn("assistant", "Paris.")
    first.title_session("what is the capital of France")
    first.title_session("something said later")
    first.close()

    second = Memory(db)
    second.add_turn("user", "remind me to stretch")
    second.title_session("remind me to stretch")
    second.close()

    Memory(db).close()  # a restart that nobody spoke into

    view = Memory(db)

    # Four sessions have been opened by now — two with content, the empty
    # restart above, and `view` itself, which has not been spoken into yet.
    # Only the two that hold something may appear.
    listed = view.sessions()
    check("only sessions with content are listed", len(listed) == 2,
          f"got {[(s['id'], s['turns']) for s in listed]}")
    check("the newest conversation is first", listed[0]["id"] > listed[-1]["id"])
    check("a session is named after its opening directive",
          listed[-1]["title"] == "what is the capital of France",
          f"got {listed[-1]['title']!r}")
    check("a title is not overwritten by a later turn",
          "something said later" not in [s["title"] for s in listed])
    check("a live session with nothing in it is not listed",
          not any(s["current"] for s in listed))

    # …and appears, flagged, the moment it does hold something.
    view.add_turn("user", "opening the current conversation")
    live = [s for s in view.sessions() if s["current"]]
    check("the live conversation is flagged once it has content",
          len(live) == 1 and live[0]["id"] == view.session_id,
          f"got {live}")

    turns = view.session_turns(1)
    check("a conversation reads back in order",
          [t["role"] for t in turns] == ["user", "assistant"])

    removed = view.delete_session(1)
    check("deleting a session reports what it removed", removed == 2, f"got {removed}")
    check("the deleted session is gone", 1 not in [s["id"] for s in view.sessions()])

    # Erasing the conversation you are currently in must not orphan the run:
    # the session row has to survive so later turns still have somewhere to go.
    view.delete_session(view.session_id)
    view.add_turn("user", "and still recording")
    check("erasing the live conversation leaves it usable",
          len(view.session_turns(view.session_id)) == 1)

    # -- facts -----------------------------------------------------------
    view.remember_fact("sister's name", "Ada")
    check("a fact recalls exactly", view.recall_fact("sister's name") == "Ada")
    check("a fact recalls fuzzily", view.recall_fact("sister") == "Ada")
    hits = next(f["hits"] for f in view.all_facts() if f["key"] == "sister's name")
    check("a fuzzy recall counts as a hit", hits == 2, f"got {hits}")

    # -- recall must not feed the assistant its own answers ----------------
    #
    # A regression pin for the nastiest failure this store can produce. Asked
    # for a value it could not compute, the model invents one; the invention
    # is persisted like any other turn; recall then serves it back on every
    # later asking as established context, and the model stops calling the
    # tool that would have got it right. One hallucination becomes permanent.
    #
    # The fix is structural, not a prose caveat: automatic recall reads
    # operator turns only. The `recall` *skill* still searches everything, so
    # "what did you tell me earlier" keeps working - the difference is that it
    # is asked for rather than fed in.
    view.add_turn("user", "what is the sha256 of the word jarvis")
    view.add_turn("assistant", "The sha256 of the word jarvis is 8c0e9a0dfabricated.")

    grounding = view.search("sha256 jarvis", limit=6, roles=("user",))
    check("auto-recall returns operator turns", bool(grounding))
    check("auto-recall never returns the assistant's own answers",
          all(h.role == "user" for h in grounding),
          f"got roles {[h.role for h in grounding]}")
    check("...so a fabricated value cannot come back as context",
          not any("fabricated" in h.text for h in grounding))

    everything = view.search("sha256 jarvis", limit=6)
    check("an explicit recall can still see both sides",
          {h.role for h in everything} == {"user", "assistant"},
          f"got roles {[h.role for h in everything]}")
    view.close()

# --------------------------------------------- durable writes announce themselves

print("\n-- a durable change tells the interface --")

# The MIND tab used to read facts, notes and reminders once when it was opened
# and never again, and nothing announced a *write* anyway - only deletes and
# wipes did. So you could say "remember that…", watch the log confirm it was
# stored, and still not find it under FACTS until the page was reloaded.
_m = Memory(Path(tempfile.mkdtemp()) / "memory.db")
_seen: list[str] = []
_m.on_change = _seen.append

_m.remember_fact("lucky number", "my lucky number is seventeen")
check("storing a fact announces it", _seen == ["fact.set"], f"got {_seen}")

_seen.clear(); _note = _m.add_note("rewire the proxy")
check("adding a note announces it", _seen == ["note.add"], f"got {_seen}")
_seen.clear(); _m.delete_note(_note)
check("deleting a note announces it", _seen == ["note.delete"], f"got {_seen}")

_seen.clear(); _rem = _m.add_reminder("stretch", time.time() + 600)
check("setting a reminder announces it", _seen == ["reminder.add"], f"got {_seen}")
_seen.clear(); _m.cancel_reminder(_rem)
check("cancelling one announces it", _seen == ["reminder.cancel"], f"got {_seen}")

_seen.clear(); _m.forget_fact("lucky number")
check("forgetting a fact announces it", _seen == ["fact.forget"], f"got {_seen}")

# A delete that removed nothing is not a change; firing anyway would have
# every open tab reload for nothing.
_seen.clear(); _m.forget_fact("never stored"); _m.delete_note(9999)
check("a no-op delete announces nothing", _seen == [], f"got {_seen}")

# The transcript already streams live. One notification per turn would be
# constant noise for a list that does not show turns at all.
_seen.clear(); _m.add_turn("user", "hello")
check("an ordinary turn stays quiet", _seen == [], f"got {_seen}")

# A listener that throws must not cost the operator the write.
_m.on_change = lambda reason: (_ for _ in ()).throw(RuntimeError("boom"))
_stored = True
try:
    _m.remember_fact("resilient", "still saved")
except Exception:
    _stored = False
check("a broken listener never breaks the write", _stored)
check("...and the fact is really there", _m.recall_fact("resilient") == "still saved")

_m.close()

# ------------------------------------------------------- dispositions

print("\n-- a disposition carries its own voice --")

from app.intents import parse_intent  # noqa: E402
from app.personas import PERSONALITIES, find as find_persona  # noqa: E402
from app.registry import KOKORO_VOICES, Registry  # noqa: E402
from app.logbus import LogBus  # noqa: E402


class _NoPrefs(Prefs):
    def __init__(self) -> None:
        super().__init__(path=Path("/nonexistent/jarvis-persona-test.json"))

    def save(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None


_quiet = LogBus()
_quiet.publish = lambda *a, **k: None  # type: ignore[method-assign]

# A voice that does not exist would leave the disposition silently on whatever
# was selected before, which reads as the setting not working at all.
for _p in PERSONALITIES.values():
    check(f"{_p.label} names a real voice", _p.voice in KOKORO_VOICES, f"got {_p.voice}")
    check(f"{_p.label} speaks at a sane rate", 0.5 <= _p.speed <= 2.0, f"got {_p.speed}")

_reg = Registry(bus=_quiet, prefs=_NoPrefs())
_reg.apply(persona="rage", announce=False)
check("choosing a disposition adopts its voice",
      _reg.voice == PERSONALITIES["rage"].voice, f"got {_reg.voice}")
check("...and its speech rate", abs(_reg.speed - PERSONALITIES["rage"].speed) < 1e-6)

# An explicit voice in the same breath has to win, or the settings panel could
# never override the pairing and "switch to Friday in George's voice" would lie.
_reg.apply(persona="jarvis", voice="af_nova", announce=False)
check("an explicit voice outranks the disposition's", _reg.voice == "af_nova",
      f"got {_reg.voice}")
# ...and a voice change on its own must not drag the disposition with it.
_before = _reg.persona
_reg.apply(voice="bf_emma", announce=False)
check("changing voice alone leaves the disposition", _reg.persona == _before)

print("\n-- the new dispositions are reachable by voice --")

for phrase, expected in [
    ("be angry", "rage"),
    ("rage mode", "rage"),
    ("trini mode", "trini"),
    ("switch persona to trinidadian", "trini"),
    ("switch persona to caribbean", "trini"),
]:
    intent = parse_intent(phrase)
    got = find_persona(intent.value) if intent and intent.value else None
    check(f"{phrase!r} -> {expected}", got is not None and got.key == expected,
          f"got {intent.kind if intent else None}/{got.key if got else None}")

# "be more careful" is an instruction to the model, not a disposition switch;
# widening the persona word list must not have swallowed it.
check("'be more careful' still reaches the model",
      parse_intent("be more careful") is None,
      f"got {parse_intent('be more careful')}")

print("\n-- operator dispositions --")

from app import personas as _P  # noqa: E402

_pfile = Path(tempfile.mkdtemp()) / "personas.json"
_P.CUSTOM.clear()
_P.rebuild()
_builtins = len(_P.BUILTIN)

_new = _P.upsert(
    {"label": "Pirate", "blurb": "Yarr.",
     "style": "Talk like a pirate at all times, with plenty of yarr and matey.",
     "voice": "am_puck", "speed": 1.05},
    path=_pfile,
)
check("a disposition can be added", _new.key == "pirate", f"got {_new.key}")
check("...and joins the catalogue", len(_P.PERSONALITIES) == _builtins + 1)
check("...and is findable by name", (_P.find("pirate") or _P.find("x")).key == "pirate")

# The file is the whole point: a disposition that vanishes on restart is worse
# than not being able to make one.
_P.CUSTOM.clear()
_P.rebuild()
check("a fresh process has only the built-ins", len(_P.PERSONALITIES) == _builtins)
_P.reload(_pfile)
check("...and reloads the operator's from disk", "pirate" in _P.PERSONALITIES)
check("...with its voice intact", _P.PERSONALITIES["pirate"].voice == "am_puck")

# Editing a built-in must keep its key, or every saved preference and spoken
# phrase naming it breaks.
_P.upsert({"label": "J.A.R.V.I.S.", "blurb": "changed",
           "style": "Be exceptionally formal and never use a contraction."},
          key="jarvis", path=_pfile)
check("a built-in can be edited in place", _P.PERSONALITIES["jarvis"].blurb == "changed")
check("...and is flagged as edited", _P.is_edited("jarvis"))
check("...while still being a built-in", _P.is_builtin("jarvis"))
check("resetting it restores the original", _P.remove("jarvis", path=_pfile) == "reset")
check("...literally the shipped text",
      _P.PERSONALITIES["jarvis"].blurb == _P.BUILTIN["jarvis"].blurb)

check("a custom one is deleted outright", _P.remove("pirate", path=_pfile) == "deleted")
refuses_persona = False
try:
    _P.remove("friday", path=_pfile)
except _P.PersonaError:
    refuses_persona = True
check("an untouched built-in cannot be deleted", refuses_persona)

# Rejections have to be specific enough to fix, and must never write.
for bad, why in [
    ({"label": "X", "style": "short"}, "a style that says nothing"),
    ({"label": "", "style": "A perfectly good manner of speaking, at length."}, "no name"),
    ({"label": "!!", "style": "A perfectly good manner of speaking, at length."}, "an unusable key"),
]:
    rejected = False
    try:
        _P.normalise(bad)
    except _P.PersonaError:
        rejected = True
    check(f"rejects {why}", rejected)

# A file someone hand-edited into nonsense must cost them that file, not the
# assistant.
_pfile.write_text("{ not json at all", encoding="utf-8")
_customs, _err = _P.load_custom(_pfile)
check("a corrupt personas.json is reported, not fatal", _customs == {} and _err is not None)
_pfile.write_text('{"personas": [{"label": "Bad"}, {"label": "Good", '
                  '"style": "Speak plainly and briefly at all times please."}]}',
                  encoding="utf-8")
_customs, _err = _P.load_custom(_pfile)
check("one broken entry does not cost the others",
      set(_customs) == {"good"}, f"got {set(_customs)}")

_P.CUSTOM.clear()
_P.rebuild()

print()
if _failures:
    print(f"== {_failures} guard check(s) FAILED")
    sys.exit(1)
print("== all guard checks passed")
