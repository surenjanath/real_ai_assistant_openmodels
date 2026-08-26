"""Command orchestrator: STT/typed command -> agent crew -> voice engine.

This is the nervous-system glue mandated by PRD §4 - it routes a command into
the agentic runtime (Ollama/CrewAI when available, simulation otherwise),
streams every reasoning step to the telemetry bus, then hands the final answer
to the TTS manager so it is spoken and rendered on the holographic interface.
"""

from __future__ import annotations

import asyncio
import time

from .agents.base import Runtime
from .agents.ollama_runtime import OllamaRuntime
from .agents.simulated_runtime import SimulatedRuntime
from .config import settings
from .logbus import LogBus
from .tts.manager import TTSManager


class Orchestrator:
    def __init__(self, bus: LogBus, tts: TTSManager) -> None:
        self.bus = bus
        self.tts = tts
        self.runtime: Runtime | None = None
        self._lock = asyncio.Lock()  # serialise workflows - one crew at a time
        self._worker: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=16)

    async def start(self) -> None:
        runtime = await OllamaRuntime.probe(self.bus)
        if runtime is None:
            runtime = SimulatedRuntime(self.bus, model_hint=settings.ollama_model)
            self.bus.publish(
                "warn",
                "brain",
                f"running SIMULATED crew - start ollama ({settings.ollama_base_url}) and reconnect for live reasoning",
            )
        self.runtime = runtime
        self._worker = asyncio.create_task(self._worker_loop(), name="orchestrator")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    @property
    def runtime_name(self) -> str:
        return self.runtime.name if self.runtime else "booting"

    async def enqueue(self, text: str, origin: str = "text") -> None:
        await self._queue.put((text, origin))

    async def _worker_loop(self) -> None:
        while True:
            text, origin = await self._queue.get()
            try:
                await self.handle(text, origin)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.bus.publish("error", "workflow", f"orchestrator error: {type(exc).__name__}: {exc}")

    async def handle(self, text: str, origin: str = "text") -> None:
        text = " ".join(text.split()).strip()
        if not text:
            return
        async with self._lock:
            started = time.monotonic()
            source = "stt" if origin == "voice" else "router"
            self.bus.publish("voice" if origin == "voice" else "info", source, f'"{text}"')
            self._broadcast_status("thinking", text)
            try:
                answer = await self.runtime.run(text)
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("error", "workflow", f"crew failed: {type(exc).__name__}: {exc}")
                self._broadcast_status("idle")
                return
            elapsed = time.monotonic() - started
            self.bus.publish("success", "workflow", f"crew finished in {elapsed:.1f}s - routing answer to voice engine")
            self.bus.publish("info", "answer", answer)
            self._broadcast_status("speaking", answer[:80])
            await self.tts.speak(answer)
            self._broadcast_status("idle")

    def _broadcast_status(self, status: str, detail: str = "") -> None:
        # Status events ride the same log socket as a dedicated message type.
        for q in list(self.bus.subscribers):
            try:
                q.put_nowait(_StatusSignal(status, detail))  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass


class _StatusSignal:
    """Marker object pushed onto log queues; main.py serialises it to JSON."""

    __slots__ = ("status", "detail")

    def __init__(self, status: str, detail: str) -> None:
        self.status = status
        self.detail = detail

    def as_dict(self) -> dict:
        return {"type": "status", "status": self.status, "detail": self.detail}
