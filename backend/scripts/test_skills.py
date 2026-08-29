#!/usr/bin/env python3
"""Unit tests for the skills the cortex can call.

These exist because a wrong tool result is worse than no tool at all: the model
states it with total confidence and the operator has no way to tell. So the
cases here are the ones where being subtly wrong would go unnoticed —
leap years, month-end clamping, ambiguous date formats — plus the boundaries
every file skill has to hold.

    python scripts/test_skills.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory import Memory  # noqa: E402
from app.skills import SkillError, SkillKit, parse_date  # noqa: E402

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    if condition:
        print(f"PASS {label}")
    else:
        _failures += 1
        print(f"FAIL {label}" + (f"  ({detail})" if detail else ""))


def refuses(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except SkillError:
        print(f"PASS {label}")
        return
    global _failures
    _failures += 1
    print(f"FAIL {label}  (it was allowed)")


def kit() -> SkillKit:
    return SkillKit(Memory(Path(tempfile.mkdtemp()) / "memory.db"))


# --------------------------------------------------------------- date parsing

print("-- reading a written date --")

# A Saturday, deliberately: weekday arithmetic that happens to work on a
# Monday is not evidence of anything.
NOW = datetime(2026, 8, 29)

for text, expected in [
    ("today", "2026-08-29"),
    ("tomorrow", "2026-08-30"),
    ("yesterday", "2026-08-28"),
    ("2026-12-25", "2026-12-25"),
    ("25 December 2026", "2026-12-25"),
    ("Dec 25, 2026", "2026-12-25"),
    ("25th December 2026", "2026-12-25"),
    ("in 10 days", "2026-09-08"),
    ("3 weeks ago", "2026-08-08"),
    ("in 2 months", "2026-10-29"),
    ("next friday", "2026-09-04"),
    ("last monday", "2026-08-24"),
]:
    got = parse_date(text, NOW)
    check(f"{text!r} -> {expected}", got is not None and got.strftime("%Y-%m-%d") == expected,
          f"got {got}")

# A day and month with no year means the next one to come round, or "days
# until Christmas" answers with a negative number for eleven months of the year.
check("a bare date in the future stays in this year",
      parse_date("25 December", NOW).strftime("%Y-%m-%d") == "2026-12-25")
check("a bare date already past rolls to next year",
      parse_date("1 January", NOW).strftime("%Y-%m-%d") == "2027-01-01")

# dd/mm and mm/dd are indistinguishable from the digits, and quietly choosing
# one is how a reminder lands a month late.
refuses("an ambiguous slashed date is refused, not guessed", parse_date, "03/04/2026", NOW)
refuses("an impossible date is refused", parse_date, "2026-02-30", NOW)
check("unparseable input returns nothing", parse_date("sometime soon", NOW) is None)

print()
print("-- date arithmetic --")

k = kit()

# The cases a language model gets wrong from memory.
out = k.shift_date("2026-01-31", months=1)
check("31 January plus a month clamps to February",
      out["result"] == "2026-02-28", f"got {out['result']}")
out = k.shift_date("2024-01-31", months=1)
check("...and to the 29th in a leap year", out["result"] == "2024-02-29", f"got {out['result']}")
out = k.shift_date("2024-02-29", years=1)
check("29 February plus a year clamps to the 28th",
      out["result"] == "2025-02-28", f"got {out['result']}")

out = k.days_between(start="2024-01-01", end="2025-01-01")
check("a leap year is 366 days", out["days"] == 366, f"got {out['days']}")
out = k.days_between(start="2025-01-01", end="2026-01-01")
check("an ordinary year is 365 days", out["days"] == 365, f"got {out['days']}")

out = k.days_between(start="2026-03-01", end="2026-01-01")
check("a backwards range reports the direction", out["direction"] == "past", f"got {out}")
check("...and a negative count", out["days"] == -59, f"got {out['days']}")
check("the same day is neither",
      k.days_between(start="2026-05-05", end="2026-05-05")["days"] == 0)
check("the weekday is computed, not recalled",
      k.days_between(end="2026-12-25")["end_weekday"] == "Friday")

refuses("an unreadable date is refused", k.days_between, "the fourth of never")

# The model's natural reading of "days until X" filled *both* ends with X and
# got zero back. `end` is required and `start` defaults to today precisely so
# the one-argument call is the obvious one.
check("counting to a date needs only that date",
      k.days_between(end="in 10 days")["days"] == 10)
schema = next(s for s in k.schemas() if s["function"]["name"] == "days_between")
check("...and the schema says so",
      schema["function"]["parameters"]["required"] == ["end"],
      f"required={schema['function']['parameters']['required']}")

print()
print("-- digests --")

# The one value in this file that can be checked against the outside world.
out = k.hash_text("hello world")
check("sha256 of 'hello world' is the known value",
      out["hex"] == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
      f"got {out['hex']}")
check("md5 is available too", k.hash_text("hello world", "md5")["hex"]
      == "5eb63bbbe01eeed093cb22bb8f5acdc3")
check("the short form is the first eight characters", out["short"] == out["hex"][:8])
refuses("an unknown algorithm is refused", k.hash_text, "x", "crc32")

print()
print("-- memory can be corrected, not only added to --")

k.remember("editor", "vim")
k.remember("city", "Port of Spain")
check("stored facts can be listed", k.list_facts()["count"] == 2)
check("a fact can be removed", k.forget_fact("editor")["forgotten"] == "editor")
check("...and is then gone", k.list_facts()["count"] == 1)
# The error has to name the keys, or the model cannot recover on its next turn.
try:
    k.forget_fact("editor")
    check("forgetting an unknown key is refused", False, "it was allowed")
except SkillError as exc:
    check("forgetting an unknown key is refused", True)
    check("...and the refusal names what is stored", "city" in str(exc), f"got {exc}")

note = k.take_note("rewire the proxy")
check("a note can be deleted", k.delete_note(note["note_id"])["deleted_note"] == note["note_id"])
refuses("deleting a missing note is refused", k.delete_note, note["note_id"])
refuses("a non-numeric note id is refused", k.delete_note, "second one")

rem = k.set_reminder("stretch", "in 20 minutes")
check("a reminder can be cancelled",
      k.cancel_reminder(rem["reminder_id"])["cancelled_reminder"] == rem["reminder_id"])
refuses("cancelling a missing reminder is refused", k.cancel_reminder, rem["reminder_id"])

print()
print("-- searching inside files --")

sandbox = Path(tempfile.mkdtemp())
(sandbox / "notes.md").write_text("the proxy config lives in nginx\nunrelated line\n")
(sandbox / "code.py").write_text("PROXY_TIMEOUT = 30\n")
(sandbox / "binary.bin").write_bytes(b"\x00proxy\x00")
hidden = sandbox / ".git"
hidden.mkdir()
(hidden / "config.md").write_text("proxy in a dotted folder\n")

import os  # noqa: E402
os.environ["JARVIS_WORKSPACE"] = str(sandbox)

k2 = kit()
found = k2.search_file_contents("proxy", ".")
paths = {Path(m["path"]).name for m in found["matches"]}
check("it finds text inside files", "notes.md" in paths, f"got {paths}")
check("it is case-insensitive", "code.py" in paths, f"got {paths}")
check("it reports the line number",
      any(m["line"] == 1 for m in found["matches"] if Path(m["path"]).name == "notes.md"))
check("it skips binary files", "binary.bin" not in paths, f"got {paths}")
check("it skips dotted folders", "config.md" not in paths, f"got {paths}")
check("a glob narrows the search",
      {Path(m["path"]).name for m in k2.search_file_contents("proxy", ".", "*.py")["matches"]}
      == {"code.py"})
check("a limit is honoured", k2.search_file_contents("e", ".", "*", 1)["count"] == 1)
refuses("an empty query is refused", k2.search_file_contents, "")
refuses("searching outside the workspace is refused", k2.search_file_contents, "proxy", "/etc")

info = k2.file_info("notes.md")
check("file_info counts lines", info["lines"] == 2, f"got {info.get('lines')}")
check("file_info reports the size", info["bytes"] == (sandbox / "notes.md").stat().st_size)
check("file_info describes a folder", k2.file_info(".")["type"] == "dir")
refuses("file_info refuses a missing path", k2.file_info, "nope.md")
refuses("file_info refuses paths outside the workspace", k2.file_info, "/etc/hosts")

print()
print("-- the assistant can clean up after itself --")

written = k2.write_file("draft.md", "a draft")
check("a scratch file is written", Path(written["path"]).exists())
k2.delete_scratch_file("draft.md")
check("...and can be deleted", not Path(written["path"]).exists())
refuses("deleting a missing scratch file is refused", k2.delete_scratch_file, "draft.md")
# The delete path must be as confined as the write path, or it becomes the
# widest hole in the file skills.
refuses("deletion cannot escape the scratch folder",
        k2.delete_scratch_file, "../../.ssh/authorized_keys")
refuses("deletion refuses an absolute path", k2.delete_scratch_file, "/etc/hosts")

print()
print("-- desktop integration stays behind its gate --")

k2.allow_shell = False
for name in ("clipboard_write", "clipboard_read", "open_path"):
    fn = getattr(k2, name)
    refuses(f"{name} is off unless enabled", *( (fn, "x") if name != "clipboard_read" else (fn,) ))

k2.allow_shell = True
refuses("open_path refuses a URL", k2.open_path, "https://example.com")
refuses("open_path refuses a custom scheme", k2.open_path, "file:///etc/passwd")
refuses("open_path refuses outside the workspace", k2.open_path, "/etc/hosts")

print()
print("-- reflexes: grounded before the model gets a say --")

# Whether the model *chooses* to call a tool is a coin toss on a 9B model -
# measured here at 1 call in 12 on questions that plainly needed one. So the
# cases where a wrong answer is both confident and undetectable are computed up
# front and injected, exactly as the arithmetic reflex already was.
from app.reflexes import detect  # noqa: E402

def reflex(text: str) -> str:
    found = detect(text)
    return found[0].detail if found else ""

import hashlib  # noqa: E402

got = reflex("what is the sha256 hash of the word jarvis")
check("a digest is computed, never composed",
      hashlib.sha256(b"jarvis").hexdigest() in got, f"got {got!r}")
check("md5 too", hashlib.md5(b"hello world").hexdigest()  # noqa: S324 - not a security use
      in reflex("give me the md5 of hello world"))
# A question *about* hashing is not a request for one.
check("a conceptual question does not fire it", reflex("what is a sha256 hash") == "",
      f"got {reflex('what is a sha256 hash')!r}")

check("a named holiday is resolved", "25 December" in reflex("how many days until Christmas"),
      f"got {reflex('how many days until Christmas')!r}")
check("a written date is resolved",
      "118 day" in reflex("how long until the 25th of December"),
      f"got {reflex('how long until the 25th of December')!r}")

# "since" looks backwards. A bare day-and-month resolved forwards would answer
# every such question a full year out, confidently.
since = reflex("how many days since new year")
check("'since' resolves to the date that has been", "ago" in since, f"got {since!r}")
check("...and not to next year's", "2027" not in since, f"got {since!r}")

check("an unparseable subject does not fire it",
      reflex("how long until dinner") == "", f"got {reflex('how long until dinner')!r}")
check("arithmetic still fires", "478,462,152" in reflex("what is 48271 * 9912"))

print()
print("-- only the tools a directive could need are described --")

# Attaching all 29 schemas costs ~3,500 prompt tokens and 1.3s of
# time-to-first-token on every directive, whether a tool is called or not.
full = len(k.schemas())
check("a date question does not need the file tools",
      len(k.schemas("how many days until christmas")) < full,
      f"got {len(k.schemas('how many days until christmas'))} of {full}")
check("...but does get the date tools",
      "days_between" in k.relevant("how many days until christmas"))
check("a file question gets the file tools",
      "search_file_contents" in k.relevant("search my files for the proxy config"))
check("a memory question gets the memory tools",
      {"take_note", "list_notes"} <= k.relevant("read my notes"))
check("a machine question gets the host tools",
      "list_processes" in k.relevant("how much memory is this machine using"))

# The core rides along regardless: these apply to almost any directive, and
# being unable to check the date or search memory is worse than a larger prompt.
for name in ("get_datetime", "calculate", "recall"):
    check(f"{name} is always available", name in k.relevant("flip a coin"))

# General knowledge needs no tool, and it is the commonest kind of question -
# measured at 15s to first token with all 29 schemas attached against 4.8s with
# seven. So an unmatched directive falls back to the core, not to everything.
check("a general-knowledge question carries only the core",
      len(k.schemas("what is the capital of Japan")) < full // 2,
      f"got {len(k.schemas('what is the capital of Japan'))} of {full}")
# Possessive phrasing names no noun the patterns know, but plainly concerns the
# operator's own material.
check("'my' pulls in files and memory",
      {"search_file_contents", "list_notes"} <= k.relevant("tell me about my project"))
check("no text at all means no gating", len(k.schemas()) == full)

print()
print("-- the working context is capped by size, not turn count --")

from app.agents.ollama_runtime import OllamaRuntime  # noqa: E402
from app.config import settings  # noqa: E402
from app.logbus import LogBus  # noqa: E402
from app.registry import Registry  # noqa: E402

quiet = LogBus()
quiet.publish = lambda *a, **k: None  # type: ignore[method-assign]
registry = Registry(bus=quiet)
registry.capabilities = {registry.model: ["tools"]}
runtime = OllamaRuntime(quiet, kit=k, registry=registry)

check("tool schemas are costed, not assumed", runtime._tool_token_cost() > 1000,
      f"got {runtime._tool_token_cost()}")
check("the budget leaves room for the answer",
      runtime._context_budget() < settings.num_ctx - runtime._tool_token_cost(),
      f"budget {runtime._context_budget()} of {settings.num_ctx}")

# Eight turns of "yes" and eight turns of an essay are the same *count* and
# wildly different prompts. Time to first token tracks the tokens, so that is
# the unit the cap has to use.
runtime.memory.clear()
for _ in range(6):
    runtime._remember("user", "x" * 8000)
    runtime._remember("assistant", "y" * 8000)
used = sum(len(m["content"]) // 4 for m in runtime.memory)
check("a long conversation is trimmed to the budget", used <= runtime._context_budget(),
      f"{used} tokens vs budget {runtime._context_budget()}")
check("...but never emptied entirely", len(runtime.memory) >= 2, f"got {len(runtime.memory)}")

runtime.memory.clear()
for _ in range(6):
    runtime._remember("user", "short")
    runtime._remember("assistant", "also short")
check("a short conversation is left alone by the size cap",
      len(runtime.memory) == 12, f"got {len(runtime.memory)} turns")

# A conversation nobody has touched for a quarter of an hour is over; carrying
# it forward taxes every turn of the next one.
runtime.memory.clear()
runtime._remember("user", "something from this morning")
runtime._last_turn_at = time.monotonic() - (settings.context_idle_reset_s + 60)
asyncio.run(runtime.run("hello"))  # no model reachable; the reset happens first
check("a stale context is released on the next directive", runtime.memory == [],
      f"got {runtime.memory}")

print()
if _failures:
    print(f"== {_failures} skill check(s) FAILED")
    raise SystemExit(1)
print("== all skill checks passed")
