"""Real host vitals, streamed to the interface as `vitals` frames.

The previous build fabricated these numbers with `random.uniform`, which made
the HUD gauges decorative. This reads the actual machine via psutil when it is
installed, and degrades to stdlib `os`/`shutil` readings when it is not - so
the gauges always mean something.

Frame shape (see frontend `lib/protocol.ts`)::

    {"type": "vitals", "cpu": 0-100, "mem": 0-100, "mem_used_gb": float,
     "mem_total_gb": float, "disk": 0-100, "net_kbps": float,
     "load": [1m, 5m, 15m], "cores": int, "uptime_s": float,
     "procs": int, "battery": 0-100|None, "power": "ac"|"battery"|None}
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time

from .config import settings
from .logbus import LogBus

try:  # optional but strongly preferred
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore


class Vitals:
    """Samples host metrics on an interval and pushes them to the log bus."""

    def __init__(self, bus: LogBus, client_provider=None) -> None:
        self.bus = bus
        self.client_provider = client_provider or (lambda: 0)
        self._task: asyncio.Task | None = None
        self._boot = time.time()
        self._last_net: tuple[float, float] | None = None  # (ts, bytes)
        self.latest: dict = {}
        if psutil is not None:
            # First call establishes the baseline; it always returns 0.0.
            try:
                psutil.cpu_percent(interval=None)
            except Exception:  # noqa: BLE001
                pass

    @property
    def available(self) -> str:
        return "psutil" if psutil is not None else "stdlib"

    # -- sampling -------------------------------------------------------------

    def sample(self) -> dict:
        cores = os.cpu_count() or 1
        try:
            load = list(os.getloadavg())
        except (OSError, AttributeError):
            load = [0.0, 0.0, 0.0]

        data: dict = {
            "type": "vitals",
            "ts": time.time(),
            "cores": cores,
            "load": [round(v, 2) for v in load],
            "uptime_s": round(time.time() - self._boot, 1),
            "clients": self.client_provider(),
            "source": self.available,
        }

        if psutil is not None:
            try:
                data["cpu"] = round(float(psutil.cpu_percent(interval=None)), 1)
                vm = psutil.virtual_memory()
                data["mem"] = round(float(vm.percent), 1)
                data["mem_used_gb"] = round((vm.total - vm.available) / 1e9, 2)
                data["mem_total_gb"] = round(vm.total / 1e9, 2)
                data["procs"] = len(psutil.pids())
                net = psutil.net_io_counters()
                total = float(net.bytes_sent + net.bytes_recv)
                now = time.time()
                if self._last_net is not None:
                    dt = max(1e-3, now - self._last_net[0])
                    data["net_kbps"] = round(max(0.0, (total - self._last_net[1]) / dt) / 1024, 1)
                else:
                    data["net_kbps"] = 0.0
                self._last_net = (now, total)
                battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
                if battery is not None:
                    data["battery"] = round(float(battery.percent), 1)
                    data["power"] = "ac" if battery.power_plugged else "battery"
            except Exception:  # noqa: BLE001 - never let a probe break the stream
                pass
        else:
            # Load average is the only portable CPU proxy in the stdlib.
            data["cpu"] = round(min(100.0, (load[0] / cores) * 100), 1)
            data["mem"] = 0.0
            data["net_kbps"] = 0.0

        try:
            usage = shutil.disk_usage(os.path.expanduser("~"))
            data["disk"] = round(usage.used / usage.total * 100, 1)
            data["disk_free_gb"] = round(usage.free / 1e9, 1)
        except Exception:  # noqa: BLE001
            data["disk"] = 0.0

        self.latest = data
        return data

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        self.bus.publish("info", "vitals", f"host telemetry online via {self.available}")
        self._task = asyncio.create_task(self._loop(), name="vitals")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                frame = await asyncio.to_thread(self.sample)
                self.bus.push_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(settings.vitals_interval_s)
