#!/usr/bin/env python3
"""Unit tests for the self-hearing guard.

On speakers the microphone hears J.A.R.V.I.S. as well as the operator, and a
recogniser will transcribe both. If the assistant answers that transcript it
answers itself, and keeps answering itself. This is the content half of the
defence: whether the words we just heard are words we just said.

The failure mode it guards against is silent in both directions - too loose and
the assistant talks to itself, too tight and it ignores the operator - so both
directions are tested here.

    python scripts/test_echo.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.echoguard import containment, is_echo  # noqa: E402

FAILURES: list[str] = []

ANSWER = (
    "Certainly, sir. The machine is running comfortably: processor load is "
    "eleven percent, memory is at forty two percent, and the battery has four "
    "hours remaining."
)


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name}{f' - {detail}' if detail else ''}")


def main() -> int:
    print("-- the assistant hearing itself --")

    # The microphone catches a fragment of a long answer, never the whole of
    # it, so containment - not symmetry - has to be the measure.
    for fragment in (
        "processor load is eleven percent",
        "memory is at forty two percent and the battery has four hours remaining",
        "the machine is running comfortably",
    ):
        check(f"echo: {fragment[:38]}…", is_echo(fragment, ANSWER), f"score {containment(fragment, ANSWER):.2f}")

    # Mis-transcription is the norm, not the exception: the guard has to hold
    # up when a word or two comes back wrong.
    check(
        "echo survives a mis-heard word",
        is_echo("processor load is eleven percentage", ANSWER),
        f"score {containment('processor load is eleven percentage', ANSWER):.2f}",
    )

    print()
    print("-- the operator actually speaking --")

    for directive in (
        "what is the weather in london",
        "remind me to call the workshop at six",
        "open the settings panel",
        "play something else",
        "how many kilometres is a marathon",
    ):
        check(
            f"passes: {directive[:38]}…",
            not is_echo(directive, ANSWER),
            f"score {containment(directive, ANSWER):.2f}",
        )

    # A follow-up that reuses the assistant's own subject is the hard case: it
    # shares vocabulary by design, and refusing it would make the assistant
    # impossible to have a conversation with.
    check(
        "a follow-up on the same subject passes",
        not is_echo("what about the disk", ANSWER),
        f"score {containment('what about the disk', ANSWER):.2f}",
    )
    check(
        "a correction passes",
        not is_echo("no I meant the graphics card", ANSWER),
        f"score {containment('no I meant the graphics card', ANSWER):.2f}",
    )

    print()
    print("-- short directives are never judged on content --")

    # These share vocabulary with almost any answer. Losing "stop" to the echo
    # guard would be worse than any echo it could prevent.
    for short in ("stop", "louder", "thank you", "yes", "the battery"):
        check(f"short directive survives: {short!r}", not is_echo(short, ANSWER))

    print()
    print("-- nothing spoken, nothing suppressed --")
    check("no recent speech means no echo", not is_echo("processor load is eleven percent", ""))

    print()
    print("-- the spoken window ages out --")

    class _Clock:
        """Stands in for the TTS manager's playback bookkeeping."""

        def __init__(self) -> None:
            self.entries: list[tuple[float, str]] = []

        def add(self, ends_at: float, text: str) -> None:
            self.entries.append((ends_at, text))

        def recent(self, window_s: float) -> str:
            cutoff = time.monotonic() - window_s
            return " ".join(t for ends_at, t in self.entries if ends_at >= cutoff)

    clock = _Clock()
    clock.add(time.monotonic() - 30.0, ANSWER)     # long finished
    clock.add(time.monotonic() - 0.5, "Right away, sir.")
    check("an old utterance falls out of the window", ANSWER not in clock.recent(4.0))
    check("a just-finished utterance stays in it", "Right away" in clock.recent(4.0))

    print()
    if FAILURES:
        print(f"== {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("== all echo-guard checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
