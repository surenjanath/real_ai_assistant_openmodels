"""Async broadcast bus for telemetry log events.

Every WebSocket client subscribed to `/ws/logs` receives every event published
here; a bounded backlog is kept so freshly connected clients get context.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# Levels -> the color coding mandated by the PRD telemetry panel.
LEVELS = ("info", "voice", "success", "warn", "error")


@dataclass(frozen=True)
class LogEvent:
    ts: float
    level: str
    source: str
    msg: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "log",
            "ts": self.ts,
            "level": self.level,
            "source": self.source,
            "msg": self.msg,
        }


@dataclass
class LogBus:
    backlog_size: int = 60
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    backlog: deque[LogEvent] = field(default_factory=deque)
    counter: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: event loop that owns the subscriber queues, captured on first use.
    _loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self) -> None:
        """Remember the serving loop so worker threads can hand off safely."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _dispatch(self, item: Any) -> None:
        """Enqueue to every subscriber, from the loop thread.

        Agent token callbacks run in a worker thread (`asyncio.to_thread`), and
        `asyncio.Queue.put_nowait` is not thread-safe: it can resolve a waiting
        getter's future without ever waking the loop, so frames would stall
        until unrelated traffic happened along. Anything published off-thread is
        bounced through `call_soon_threadsafe` instead.
        """
        loop = self._loop
        if loop is not None and threading.current_thread() is not threading.main_thread():
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(self._dispatch_now, item)
                    return
            except RuntimeError:
                pass
        self._dispatch_now(item)

    def _dispatch_now(self, item: Any) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                # A slow client should never stall the bus; drop the oldest.
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except Exception:  # pragma: no cover - defensive
                    pass

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        async with self._lock:
            self.subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, level: str, source: str, msg: str) -> LogEvent:
        if level not in LEVELS:
            level = "info"
        event = LogEvent(ts=time.time(), level=level, source=source, msg=msg)
        self.counter += 1
        self.backlog.append(event)
        while len(self.backlog) > self.backlog_size:
            self.backlog.popleft()
        self._dispatch(event)
        # Mirror to the server console for operators.
        print(f"[{level:>7}] {source:<14} {msg}", flush=True)
        return event

    def publish_status(self, status: str, detail: str = "") -> dict[str, Any]:
        return {
            "type": "status",
            "status": status,
            "detail": detail,
            "ts": time.time(),
        }

    def push_frame(self, frame: dict[str, Any]) -> None:
        """Push an arbitrary JSON frame (status / settings / ui control) to
        every telemetry subscriber, bypassing the log backlog."""
        self._dispatch(frame)

    async def events(self, q: asyncio.Queue) -> AsyncIterator[LogEvent]:
        while True:
            yield await q.get()
