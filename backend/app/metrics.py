"""Performance instrumentation for the cognition loop.

The interface claims to be a diagnostic system, so it ought to be able to
diagnose *itself*. Every command is timed at the points that actually matter
to how the assistant feels:

    ttft_ms   time from accepting the directive to the first answer token
    total_ms  time until the answer is fully composed
    tok_s     generated tokens per second (approximated from delta lengths)
    voice_ms  time from answer to the first PCM frame leaving the vocal engine

A rolling window keeps percentiles honest without unbounded growth, and each
completed run is broadcast as a ``metrics`` frame plus persisted to the memory
store, so trends survive restarts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .logbus import LogBus

WINDOW = 60


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Run:
    """One command's timing record, filled in as the pipeline progresses."""

    text: str
    origin: str
    model: str
    mode: str
    started: float = field(default_factory=time.monotonic)
    first_token: float | None = None
    #: when the first PCM frame of the answer left the vocal engine
    first_audio: float | None = None
    composed: float | None = None
    spoken: float | None = None
    chars: int = 0
    #: prompt tokens Ollama actually evaluated - the number that explains a
    #: slow first word, since the whole prompt is re-read before any of it
    prompt_tokens: int = 0
    eval_tokens: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    error: str = ""
    kind: str = "reasoning"  # reasoning | control | tool

    def mark_first_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.monotonic()

    def mark_first_audio(self) -> None:
        if self.first_audio is None:
            self.first_audio = time.monotonic()

    @property
    def ttft_ms(self) -> int:
        return int(((self.first_token or self.composed or time.monotonic()) - self.started) * 1000)

    @property
    def total_ms(self) -> int:
        return int(((self.composed or time.monotonic()) - self.started) * 1000)

    @property
    def ttfa_ms(self) -> int | None:
        """Time to the first *spoken* word - what the operator actually waits.

        With streamed speech this is close to `ttft_ms`, because the first
        sentence is voiced while the rest is still being written.
        """
        if self.first_audio is None:
            return None
        return int((self.first_audio - self.started) * 1000)

    @property
    def voice_ms(self) -> int | None:
        """Gap between the answer being composed and speech beginning.

        Negative under streamed speech (we were already talking), so it is
        clamped at zero rather than reported as a nonsense figure.
        """
        if self.first_audio is not None and self.composed is not None:
            return max(0, int((self.first_audio - self.composed) * 1000))
        if self.spoken is None or self.composed is None:
            return None
        return max(0, int((self.spoken - self.composed) * 1000))

    @property
    def tok_s(self) -> float:
        """Tokens/second, estimated at the usual ~4 characters per token."""
        if self.first_token is None or self.composed is None:
            return 0.0
        window = max(1e-3, self.composed - self.first_token)
        return round((self.chars / 4.0) / window, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:120],
            "origin": self.origin,
            "model": self.model,
            "mode": self.mode,
            "kind": self.kind,
            "ttft_ms": self.ttft_ms,
            "ttfa_ms": self.ttfa_ms,
            "total_ms": self.total_ms,
            "voice_ms": self.voice_ms,
            "tok_s": self.tok_s,
            "chars": self.chars,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "tool_calls": self.tool_calls,
            "tools_used": self.tools_used,
            "error": self.error,
        }


class Metrics:
    """Rolling performance window, broadcast to the interface."""

    def __init__(self, bus: LogBus) -> None:
        self.bus = bus
        self.runs: deque[Run] = deque(maxlen=WINDOW)
        self.commands = 0
        self.errors = 0
        self.tool_calls = 0
        self.spoken_chars = 0
        self.booted = time.time()
        self.current: Run | None = None

    def begin(self, text: str, origin: str, model: str, mode: str, kind: str = "reasoning") -> Run:
        run = Run(text=text, origin=origin, model=model, mode=mode, kind=kind)
        self.current = run
        return run

    def publish(self, run: Run | None = None) -> None:
        """Broadcast the window *without* closing a run.

        A run is only finished once its answer has been spoken, and speech can
        take longer than the reasoning did. Without an interim broadcast the
        instrument bar would report a stale time-to-first-word for the whole
        duration of every utterance — which is precisely the moment someone is
        looking at it.
        """
        self.bus.push_frame(self.snapshot(last=run))

    def finish(self, run: Run) -> dict[str, Any]:
        if run.composed is None:
            run.composed = time.monotonic()
        self.runs.append(run)
        self.commands += 1
        self.tool_calls += run.tool_calls
        self.spoken_chars += run.chars
        if run.error:
            self.errors += 1
        if self.current is run:
            self.current = None
        snapshot = self.snapshot(last=run)
        self.bus.push_frame(snapshot)
        return snapshot

    def snapshot(self, last: Run | None = None) -> dict[str, Any]:
        latest = last or (self.runs[-1] if self.runs else None)
        reasoning = [r for r in self.runs if r.kind == "reasoning"]
        # Headline figures describe a reasoning run: a control command answers
        # instantly and would otherwise make the bar read a misleading "0ms".
        headline = (
            latest if latest is not None and latest.kind == "reasoning"
            else (reasoning[-1] if reasoning else None)
        )
        ttfts = [float(r.ttft_ms) for r in reasoning]
        ttfas = [float(r.ttfa_ms) for r in reasoning if r.ttfa_ms is not None]
        totals = [float(r.total_ms) for r in reasoning]
        rates = [r.tok_s for r in reasoning if r.tok_s > 0]
        return {
            "type": "metrics",
            "ts": time.time(),
            "commands": self.commands,
            "errors": self.errors,
            "tool_calls": self.tool_calls,
            "spoken_chars": self.spoken_chars,
            "uptime_s": round(time.time() - self.booted, 1),
            "ttft_ms": {
                "p50": round(median(ttfts)) if ttfts else 0,
                "p95": round(_percentile(ttfts, 95)),
                "last": headline.ttft_ms if headline else 0,
            },
            "ttfa_ms": {
                "p50": round(median(ttfas)) if ttfas else 0,
                "p95": round(_percentile(ttfas, 95)),
                "last": (headline.ttfa_ms or 0) if headline else 0,
            },
            "total_ms": {
                "p50": round(median(totals)) if totals else 0,
                "p95": round(_percentile(totals, 95)),
                "last": headline.total_ms if headline else 0,
            },
            "tok_s": {
                "avg": round(sum(rates) / len(rates), 1) if rates else 0.0,
                "best": round(max(rates), 1) if rates else 0.0,
                "last": headline.tok_s if headline else 0.0,
            },
            "last": latest.as_dict() if latest is not None else None,
            "history": [
                {"ttft_ms": r.ttft_ms, "ttfa_ms": r.ttfa_ms, "total_ms": r.total_ms,
                 "tok_s": r.tok_s, "kind": r.kind, "error": bool(r.error)}
                for r in list(self.runs)[-30:]
            ],
        }
