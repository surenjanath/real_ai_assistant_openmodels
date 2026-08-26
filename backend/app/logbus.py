"""Async broadcast bus for telemetry log events.

Every WebSocket client subscribed to `/ws/logs` receives every event published
here; a bounded backlog is kept so freshly connected clients get context.
"""

from __future__ import annotations

import asyncio
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
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A slow client should never stall the bus; drop the oldest.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:  # pragma: no cover - defensive
                    pass
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
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except Exception:  # pragma: no cover - defensive
                    pass

    async def events(self, q: asyncio.Queue) -> AsyncIterator[LogEvent]:
        while True:
            yield await q.get()
