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
    view.close()

print()
if _failures:
    print(f"== {_failures} guard check(s) FAILED")
    sys.exit(1)
print("== all guard checks passed")
