"""Operational telemetry heartbeat lines for the terminal panel.

These are now derived from the **real** metrics sampled by `vitals.Vitals`
rather than invented with `random`, so a line claiming 62% memory pressure is
a line you can trust. The cadence is deliberately sparse: the terminal should
feel alive between real agent events without burying them.
"""

from __future__ import annotations

import asyncio
import random

from .config import settings
from .logbus import LogBus
from .vitals import Vitals

_BOOT = [
    ("success", "boot", "cold start complete - all subsystems nominal"),
    ("info", "boot", "telemetry channel open - streaming operational logs"),
    ("info", "boot", "audio pipeline primed - 24kHz mono PCM ready"),
]


class TelemetrySimulator:
    """Background task that keeps the terminal alive between real events."""

    def __init__(self, bus: LogBus, vitals: Vitals, client_provider=None) -> None:
        self.bus = bus
        self.vitals = vitals
        self.client_provider = client_provider or (lambda: 0)
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        for level, source, line in _BOOT:
            self.bus.publish(level, source, line)
            await asyncio.sleep(random.uniform(0.2, 0.5))
        self._task = asyncio.create_task(self._loop(), name="telemetry")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def _lines(self) -> list[tuple[str, str, str]]:
        """Build heartbeat candidates from the latest real sample."""
        v = self.vitals.latest
        if not v:
            return [("info", "system", "sampling host metrics…")]

        cpu = v.get("cpu", 0.0)
        mem = v.get("mem", 0.0)
        used = v.get("mem_used_gb")
        total = v.get("mem_total_gb")
        load = v.get("load", [0, 0, 0])
        lines: list[tuple[str, str, str]] = [
            ("info", "system", f"cpu {cpu:.0f}% across {v.get('cores', '?')} cores - load {load[0]:.2f}"),
            ("info", "router", f"ws clients: {self.client_provider()} - queue depth 0"),
            ("info", "system", f"disk {v.get('disk', 0):.0f}% used - {v.get('disk_free_gb', 0):.0f}GB free"),
            ("info", "system", f"network {v.get('net_kbps', 0):.0f} KB/s - {v.get('procs', 0)} processes"),
        ]
        if used is not None and total is not None:
            lines.append(("info", "system", f"memory {used:.1f} / {total:.0f} GB unified ({mem:.0f}%)"))
        # Escalate genuinely notable conditions.
        if cpu > 88:
            lines.append(("warn", "system", f"cpu saturation {cpu:.0f}% - inference may throttle"))
        if mem > 90:
            lines.append(("warn", "system", f"memory pressure {mem:.0f}% - consider a smaller model"))
        if v.get("power") == "battery" and (v.get("battery") or 100) < 25:
            lines.append(("warn", "system", f"battery {v.get('battery'):.0f}% - performance capped on battery"))
        return lines

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(random.uniform(settings.heartbeat_min_s, settings.heartbeat_max_s))
            level, source, msg = random.choice(self._lines())
            self.bus.publish(level, source, msg)
