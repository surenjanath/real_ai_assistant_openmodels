"""Phase 2 - simulated operational telemetry.

Emits plausible system / pipeline heartbeat lines on the log bus so the
frontend terminal has live data before the full agent stack is wired up.
On a real deployment these lines are interleaved with genuine agent logs.
"""

from __future__ import annotations

import asyncio
import random
import time

from .config import settings
from .logbus import LogBus

_GREETING = [
    "cold boot sequence complete - all subsystems nominal",
    "neural bus attached - ollama router online",
    "audio pipeline primed - 24kHz mono PCM stream ready",
    "telemetry channel open - streaming operational logs",
]

_HEARTBEATS = [
    ("system", "mem {mem} / 64.0 GB unified - gpu {gpu}% - neural engine {ne}%"),
    ("router", "heartbeat ok - {agents} agents registered, 0 stale"),
    ("system", "ws clients: {clients} - event loop lag {lag}ms"),
    ("router", "queue depth 0 - scheduler operating within tolerances"),
    ("system", "disk io {io}MB/s - thermal profile nominal"),
    ("router", "tool registry verified - mcp bridge reachable"),
]

_OSCILLATION = 0.0


def _metrics() -> dict[str, str]:
    global _OSCILLATION
    _OSCILLATION = max(-1.0, min(1.0, _OSCILLATION + random.uniform(-0.35, 0.35)))
    mem = 11.8 + _OSCILLATION * 2.4 + random.uniform(0, 1.2)
    return {
        "mem": f"{mem:.1f}GB",
        "gpu": f"{max(2, int(6 + _OSCILLATION * 5 + random.uniform(0, 4))):d}",
        "ne": f"{max(1, int(4 + _OSCILLATION * 3 + random.uniform(0, 3))):d}",
        "agents": "4",
        "clients": "{clients}",
        "lag": f"{random.uniform(0.4, 3.2):.1f}",
        "io": f"{random.uniform(12, 180):.0f}",
    }


class TelemetrySimulator:
    """Background task that keeps the terminal alive between real events."""

    def __init__(self, bus: LogBus, client_provider=None) -> None:
        self.bus = bus
        self.client_provider = client_provider or (lambda: 0)
        self._task: asyncio.Task | None = None
        self._started = time.monotonic()

    async def start(self) -> None:
        self.bus.publish("success", "boot", _GREETING[0])
        for line in _GREETING[1:]:
            await asyncio.sleep(random.uniform(0.25, 0.7))
            self.bus.publish("info", "boot", line)
        self._task = asyncio.create_task(self._loop(), name="telemetry-sim")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            delay = random.uniform(settings.heartbeat_min_s, settings.heartbeat_max_s)
            await asyncio.sleep(delay)
            source, template = random.choice(_HEARTBEATS)
            metrics = _metrics()
            metrics["clients"] = str(self.client_provider())
            self.bus.publish("info", source, template.format(**metrics))
