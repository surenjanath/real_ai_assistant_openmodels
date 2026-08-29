#!/usr/bin/env python3
"""Unit tests for the streamed-speech segmenter.

The segmenter is the one piece of the speech path that is pure logic, and it is
the piece most likely to break quietly: a bad split does not raise, it just
makes the assistant say "Dr" and then, half a second later, "Stark is in the
lab". So it gets tested directly, character by character, the way the model
actually feeds it.

    python scripts/test_segmenter.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.tts.segmenter import SentenceSegmenter  # noqa: E402

FAILURES: list[str] = []


def stream(text: str, chunk: int = 5, **kwargs) -> list[str]:
    """Feed `text` through the segmenter in small chunks, as Ollama would."""
    segmenter = SentenceSegmenter(**kwargs)
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out += segmenter.push(text[i : i + chunk])
    return out + segmenter.drain()


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name}{f' - {detail}' if detail else ''}")


def main() -> int:
    # Nothing may ever be lost or duplicated: the operator must hear exactly
    # the answer that was written, whatever the chunk boundaries land on.
    for text in (
        "A flywheel stores energy. It smooths an engine's output. That is all.",
        "Dr. Stark is in the lab. Temperature is 3.5 degrees. See https://x.io/a.b now.",
        "One long clause with no terminal punctuation at all so it must break somewhere sensible",
        'He said "we are done." Then he left.',
    ):
        for chunk in (1, 3, 7, 40, 1000):
            joined = " ".join(stream(text, chunk))
            check(
                f"lossless @chunk={chunk} · {text[:28]}…",
                "".join(joined.split()) == "".join(text.split()),
                f"got {joined!r}",
            )

    frags = stream("A flywheel stores energy. It smooths an engine's output. That is all.")
    check("splits on full stops", len(frags) == 3, f"got {frags}")

    frags = stream("Dr. Stark is in the lab. All good here.")
    check("does not split on an abbreviation", frags[0].startswith("Dr. Stark"), f"got {frags}")

    frags = stream("The reading is 3.5 degrees exactly and holding steady.")
    check("does not split on a decimal point", len(frags) == 1, f"got {frags}")

    frags = stream("Visit https://example.com/a.b.c for the schematic. Done.")
    check("does not split inside a URL", frags[0].endswith("schematic."), f"got {frags}")

    # The opening fragment may break at a clause so speech can start sooner;
    # everything after it waits for a real sentence end.
    long_first = (
        "A flywheel is a heavy rotating disc that stores kinetic energy, smoothing out "
        "the delivery of power from an engine. It is used wherever torque is uneven."
    )
    frags = stream(long_first)
    check("opening fragment breaks early", frags[0].endswith(","), f"got {frags[0]!r}")
    check(
        "opening fragment is short enough to synthesise fast",
        len(frags[0]) <= 150,
        f"{len(frags[0])} chars",
    )
    check("later fragments are whole sentences", frags[-1].endswith("."), f"got {frags[-1]!r}")

    # A wall of text with no punctuation must still be broken up, or the
    # assistant stays silent until the very end of it.
    frags = stream("word " * 120)
    check("breaks a run-on into speakable pieces", len(frags) >= 3, f"got {len(frags)}")
    check("run-on pieces stay bounded", max(len(f) for f in frags) <= 260,
          f"longest {max(len(f) for f in frags)}")

    # A tool round discards the partial answer; the next one must start clean.
    segmenter = SentenceSegmenter()
    segmenter.push("Let me check that for you")
    segmenter.reset()
    out = segmenter.push("The answer is forty two. And that is final.")
    check("reset drops the abandoned partial", not any("check" in f for f in out), f"got {out}")

    check("empty input yields nothing", stream("") == [], "expected []")
    check("whitespace-only yields nothing", stream("   \n  ") == [], "expected []")

    print()
    if FAILURES:
        print(f"== {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("== all segmenter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
