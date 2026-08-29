#!/usr/bin/env python3
"""Unit tests for streamed speech, using the dependency-free vocal engine.

Two failure modes here are both silent and both embarrassing, which is exactly
why they are pinned down with a test rather than left to a listen-and-see:

1. A **tool round** means the prose the model had already emitted was
   scaffolding ("let me look that up"), not the answer. If the abort path is
   wrong the operator hears the scaffolding, or hears it twice.
2. A **barge-in** during generation cancels the stream before it has said
   anything. If `finish()` then reports "nothing was spoken", the orchestrator
   helpfully speaks the whole answer — the exact thing the operator just cut
   off.

    python scripts/test_speech_stream.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.logbus import LogBus  # noqa: E402
from app.tts.fallback_engine import FallbackEngine  # noqa: E402
from app.tts.manager import TTSManager  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name}{f' - {detail}' if detail else ''}")


async def run() -> None:
    bus = LogBus()
    bus.bind_loop()
    bus.publish = lambda *a, **k: None  # type: ignore[method-assign] # keep output clean

    manager = TTSManager(bus, FallbackEngine())
    client = await manager.subscribe()
    seen: list[dict] = []

    async def drain() -> None:
        while True:
            seen.append(await client.queue.get())

    pump = asyncio.create_task(drain())

    # -- a tool round discards the scaffolding it had begun saying -----------
    stream = manager.open_stream()
    stream.feed("Let me look that up for you, one moment. ")
    await asyncio.sleep(0.4)  # let that fragment reach the clients
    started_before = manager.stream_audio_started
    stream.abort()  # what the orchestrator's on_reset() does
    stream.feed("The answer is forty two. That is confirmed. ")
    spoke = await stream.finish()

    flushes = [f for f in seen if f["type"] == "tts.flush"]
    check("the post-tool answer is spoken", spoke)
    check(
        "scaffolding never reaches the voice",
        "look that up" not in stream.spoken_text,
        f"voiced {stream.spoken_text!r}",
    )
    check(
        "the real answer is what gets voiced",
        stream.spoken_text.startswith("The answer is forty two."),
        f"voiced {stream.spoken_text!r}",
    )
    check(
        "clients are told to flush the discarded audio",
        bool(flushes) or not started_before,
        "audio had gone out but no flush was broadcast",
    )

    # -- a barge-in must not be undone by the caller's fallback --------------
    interrupted = manager.open_stream()
    interrupted.feed("This is being interrupted right now. ")
    await asyncio.sleep(0.1)
    await manager.stop(reason="barge-in")
    check(
        "a barged-in stream tells the caller to stay silent",
        await interrupted.finish() is True,
        "finish() reported nothing spoken, so the answer would be re-spoken",
    )

    # -- the closing sentence must survive ---------------------------------
    # A regression pin. Mid-stream the segmenter cannot release a sentence
    # until it has seen the character *after* the full stop, so the final
    # sentence of every answer is only ever released by `finish`. Any bug in
    # that flush costs the last thing the assistant had to say, silently.
    whole = manager.open_stream()
    for delta in ("First the setup. ", "Then the middle part. ", "And finally the conclusion."):
        whole.feed(delta)
    await whole.finish()
    check(
        "the closing sentence is spoken",
        "And finally the conclusion." in whole.spoken_text,
        f"voiced {whole.spoken_text!r}",
    )
    check(
        "no sentence is dropped or duplicated",
        "".join(whole.spoken_text.split())
        == "".join("First the setup. Then the middle part. And finally the conclusion.".split()),
        f"voiced {whole.spoken_text!r}",
    )

    # -- an answer with no sentence end still gets voiced -------------------
    tail_only = manager.open_stream()
    tail_only.feed("no terminal punctuation here at all")
    check("an unterminated answer is still spoken", await tail_only.finish() is True)

    # -- an empty stream reports honestly, so the caller can fall back -------
    empty = manager.open_stream()
    check("an empty stream reports that it said nothing", await empty.finish() is False)

    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)
    await manager.unsubscribe(client)


def main() -> int:
    asyncio.run(run())
    print()
    if FAILURES:
        print(f"== {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("== all streamed-speech checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
