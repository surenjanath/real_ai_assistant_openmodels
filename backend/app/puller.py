"""Model acquisition: pull an Ollama model with progress on the telemetry bus.

Installing a model is the one long-running thing the operator is likely to want
from inside the interface, and it is exactly the thing that looks broken when
it happens silently — a 5 GB download with no feedback is indistinguishable
from a hang. So the pull streams Ollama's progress lines straight onto the log
bus as `model.pull` frames, and the interface draws a bar from them.

One pull at a time, cancellable, and a failure is reported rather than raised
into whatever asked for it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .config import settings
from .logbus import LogBus


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


class ModelPuller:
    """Runs at most one `ollama pull` at a time, reporting progress."""

    def __init__(self, bus: LogBus) -> None:
        self.bus = bus
        self._task: asyncio.Task | None = None
        self._cancel = threading.Event()
        self.current: str = ""
        self.progress: dict[str, Any] = {}

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, model: str, on_done: Any | None = None) -> dict[str, Any]:
        model = model.strip()
        if not model:
            return {"ok": False, "error": "model required"}
        if self.busy:
            return {"ok": False, "error": f"already pulling {self.current}"}
        self._cancel.clear()
        self.current = model
        self.progress = {"model": model, "status": "starting", "completed": 0, "total": 0}
        self._task = asyncio.create_task(self._run(model, on_done), name=f"pull-{model}")
        return {"ok": True, "pulling": model}

    def cancel(self) -> bool:
        if not self.busy:
            return False
        self._cancel.set()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {"busy": self.busy, "model": self.current, **self.progress}

    # -- internals ------------------------------------------------------------

    async def _run(self, model: str, on_done: Any | None) -> None:
        started = time.monotonic()
        self.bus.publish("info", "models", f"pulling {model} — this can take several minutes")
        try:
            ok, detail = await asyncio.to_thread(self._pull_blocking, model)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - started
        if ok:
            self.bus.publish("success", "models", f"{model} installed in {elapsed:.0f}s")
        else:
            self.bus.publish("error", "models", f"pull of {model} failed: {detail}")
        self.progress = {"model": model, "status": "done" if ok else "error",
                         "detail": detail, "completed": 0, "total": 0}
        self.bus.push_frame({"type": "model.pull", "done": True, "ok": ok,
                             "model": model, "detail": detail})
        if on_done is not None:
            try:
                result = on_done(model, ok)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - a hook must not mask the outcome
                pass

    def _pull_blocking(self, model: str) -> tuple[bool, str]:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/pull"
        req = urllib.request.Request(
            url,
            data=json.dumps({"model": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_emit = 0.0
        last_status = ""
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for raw in resp:
                    if self._cancel.is_set():
                        return False, "cancelled"
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        return False, str(event["error"])[:200]

                    status = str(event.get("status") or "")
                    completed = int(event.get("completed") or 0)
                    total = int(event.get("total") or 0)
                    self.progress = {"model": model, "status": status,
                                     "completed": completed, "total": total}

                    # Throttle: Ollama emits progress far faster than anyone can
                    # read it, and every frame costs a broadcast to every client.
                    now = time.monotonic()
                    if status != last_status or now - last_emit > 0.5:
                        last_status, last_emit = status, now
                        pct = round(completed / total * 100) if total else None
                        self.bus.push_frame({
                            "type": "model.pull", "model": model, "status": status,
                            "completed": completed, "total": total, "percent": pct,
                            "done": False,
                        })
                        if pct is not None:
                            self.bus.publish(
                                "info", "models",
                                f"{model}: {status} {pct}% ({_human(completed)}/{_human(total)})",
                            )
                        elif status:
                            self.bus.publish("info", "models", f"{model}: {status}")
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        return True, "complete"
