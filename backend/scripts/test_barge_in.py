#!/usr/bin/env python3
"""Unit tests for interrupting the assistant mid-answer.

Two things have to be true for a spoken conversation to feel like one:

1. Something said *while J.A.R.V.I.S. is talking* must reach the assistant
   without waiting for the answer to finish. Directives are drained one at a
   time and a directive is not done until its answer has finished *playing*, so
   the interrupting phrase has to tear down both the voice and the generation
   behind it on arrival - not on its turn in the queue.
2. The assistant must not interrupt *itself*. An open microphone hears the
   speakers, and answering that transcript is a loop with no exit.

Both failures are invisible to a type checker and obvious to anyone in the
room, which is exactly the kind of thing that belongs in a test.

    python scripts/test_barge_in.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.logbus import LogBus  # noqa: E402
from app.orchestrator import Orchestrator  # noqa: E402
from app.prefs import Prefs  # noqa: E402
from app.registry import Registry  # noqa: E402
from app.tts.fallback_engine import FallbackEngine  # noqa: E402
from app.tts.manager import TTSManager  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name}{f' - {detail}' if detail else ''}")


class SlowRuntime:
    """A model that takes its time, and notices when it is cut off."""

    name = "slow"
    mode = "fast"

    def __init__(self, answer: str, seconds: float = 3.0) -> None:
        self.answer = answer
        self.seconds = seconds
        self._aborted = False
        self.aborted_at: float | None = None

    def abort(self) -> None:
        self._aborted = True

    async def run(self, text: str) -> str:
        self._aborted = False
        for _ in range(int(self.seconds / 0.05)):
            await asyncio.sleep(0.05)
            if self._aborted:
                return ""
        return self.answer


class _NoPrefs(Prefs):
    """Prefs that neither read nor write the operator's real preference file."""

    def __init__(self) -> None:
        super().__init__(path=pathlib.Path("/nonexistent/jarvis-test-prefs.json"))

    def save(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None


def build() -> tuple[Orchestrator, TTSManager, LogBus]:
    bus = LogBus()
    bus.bind_loop()
    bus.publish = lambda *a, **k: None  # type: ignore[method-assign]
    tts = TTSManager(bus, FallbackEngine())
    registry = Registry(bus=bus, prefs=_NoPrefs())
    return Orchestrator(bus, tts, registry), tts, bus


async def run() -> None:
    print("-- the assistant does not answer its own voice --")

    orch, tts, _ = build()
    client = await tts.subscribe()
    pump = asyncio.create_task(_drain(client))

    await tts.speak("The processor is at eleven percent and memory is comfortable.")
    await asyncio.sleep(0.05)

    check(
        "a transcript of what was just said is refused",
        await orch.enqueue_nowait("the processor is at eleven percent", "voice") is False,
    )
    check(
        "a real directive is still accepted",
        await orch.enqueue_nowait("what is the weather in london", "voice") is True,
    )
    check(
        "a typed directive is never echo-checked",
        await orch.enqueue_nowait("the processor is at eleven percent", "text") is True,
    )
    check(
        "a short directive survives the guard",
        await orch.enqueue_nowait("stop", "voice") is True,
    )
    # The operator repeating the assistant's own suggestion back to it is a
    # real directive, and content alone cannot tell it from an overheard one.
    # A client that ran it past an echo-cancelled microphone can, and says so.
    check(
        "an acoustically verified directive is not content-checked",
        await orch.enqueue_nowait(
            "the processor is at eleven percent", "voice", verified=True
        ) is True,
    )

    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)
    await tts.unsubscribe(client)

    print()
    print("-- speaking over the assistant cuts it off --")

    orch, tts, _ = build()
    runtime = SlowRuntime("This is a long answer nobody will get to hear.", seconds=5.0)
    orch.runtime = runtime
    client = await tts.subscribe()
    pump = asyncio.create_task(_drain(client))

    # The directive under way, and the operator speaking over it a moment later.
    working = asyncio.create_task(orch.handle("tell me a long story", "text"))
    await asyncio.sleep(0.3)
    check("a directive in flight holds the lock", orch._lock.locked())

    accepted = await orch.enqueue_nowait("actually, never mind, what time is it", "voice")
    check("the interrupting directive is accepted", accepted)

    finished = False
    for _ in range(40):  # 2s ceiling - it should take a fraction of that
        await asyncio.sleep(0.05)
        if working.done():
            finished = True
            break
    check(
        "the interrupted directive gives up promptly",
        finished,
        "it ran to completion instead of being cut short",
    )
    check("the abandoned answer is never spoken", runtime.answer not in tts.recent_speech(30.0))
    check("the lock is released for the next directive", not orch._lock.locked())

    working.cancel()
    await asyncio.gather(working, return_exceptions=True)
    pump.cancel()
    await asyncio.gather(pump, return_exceptions=True)
    await tts.unsubscribe(client)

    print()
    print("-- nothing to interrupt, nothing interrupted --")

    orch, tts, _ = build()
    orch.runtime = SlowRuntime("A perfectly ordinary answer.", seconds=0.2)
    check(
        "an idle assistant accepts a voice directive untouched",
        await orch.enqueue_nowait("what time is it", "voice") is True,
    )


async def _drain(client) -> None:
    while True:
        await client.queue.get()


def main() -> int:
    asyncio.run(run())
    print()
    if FAILURES:
        print(f"== {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("== all barge-in checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
