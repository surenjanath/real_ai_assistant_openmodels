"""Command orchestrator: command -> (settings intent | agent crew) -> voice.

Routing order per command:
  1. settings/help/status intents  - handled inline (registry changes, UI
     control frames, spoken confirmations)
  2. otherwise the agent crew runs (Ollama/CrewAI live, or simulation)

Every reasoning step streams to the telemetry bus; the final answer is handed
to the TTS manager so it is spoken and rendered on the holographic interface.
"""

from __future__ import annotations

import asyncio
import time

from .agents.base import Runtime
from .agents.ollama_runtime import OllamaRuntime
from .agents.simulated_runtime import SimulatedRuntime
from .config import settings
from .intents import Intent, parse_intent
from .logbus import LogBus
from .registry import Registry
from .tts.manager import TTSManager

_HELP_TEXT = (
    "You can give me any directive, or say: settings to open the panel; "
    "list models; switch model to a name; change voice to a name; "
    "or speak faster and slower."
)


class Orchestrator:
    def __init__(self, bus: LogBus, tts: TTSManager, registry: Registry) -> None:
        self.bus = bus
        self.tts = tts
        self.registry = registry
        self.runtime: Runtime | None = None
        self._lock = asyncio.Lock()  # serialise workflows - one crew at a time
        self._worker: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=16)

    async def start(self) -> None:
        await self.registry.refresh_models()
        runtime = await OllamaRuntime.probe(self.bus, registry=self.registry)
        if runtime is None:
            runtime = SimulatedRuntime(self.bus, model_hint=self.registry.model)
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
            source = "stt" if origin == "voice" else "router"
            self.bus.publish("voice" if origin == "voice" else "info", source, f'"{text}"')

            intent = parse_intent(text)
            if intent is not None:
                await self._handle_intent(intent)
                return

            started = time.monotonic()
            self._broadcast_status("thinking", text)
            if isinstance(self.runtime, SimulatedRuntime):
                self.runtime.model_hint = self.registry.model
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

    # ------------------------------------------------------------------ intents

    async def _handle_intent(self, intent: Intent) -> None:
        bus, registry, tts = self.bus, self.registry, self.tts
        kind = intent.kind

        if kind == "settings_show":
            bus.push_frame({"type": "ui", "action": "open_settings"})
            self._broadcast_status("speaking")
            await tts.speak(
                f"Settings panel open. Current model {registry.model}, "
                f"voice {registry.voice}, speed {registry.speed:.1f}. "
                "Choose a model or voice from the panel, or tell me to switch."
            )
            self._broadcast_status("idle")
            return

        if kind == "models_list":
            await registry.refresh_models()
            bus.publish("info", "settings", f"{len(registry.models)} models available:")
            for model in registry.models:
                marker = " ●" if model == registry.model else ""
                bus.publish("info", "settings", f"  {model}{marker}")
            names = ", ".join(registry.models[:6])
            more = f", and {len(registry.models) - 6} more" if len(registry.models) > 6 else ""
            self._broadcast_status("speaking")
            await tts.speak(f"Available models include: {names}{more}. Currently using {registry.model}.")
            self._broadcast_status("idle")
            return

        if kind == "voices_list":
            bus.publish("info", "settings", f"{len(registry.voices)} voices installed:")
            for voice in registry.voices:
                marker = " ●" if voice == registry.voice else ""
                bus.publish("info", "settings", f"  {voice}{marker}")
            self._broadcast_status("speaking")
            await tts.speak(f"I can speak with {len(registry.voices)} voices, currently {registry.voice}. Say change voice to, followed by a name.")
            self._broadcast_status("idle")
            return

        if kind == "model_set" and intent.value:
            result = registry.apply(model=intent.value)
            if result["applied"]:
                await registry.refresh_models()
                await self._confirm(
                    f"Model switched to {self.registry.model}. "
                    + ("" if self.registry.model_verified else "Note: not yet present on the Ollama server; pull it if needed.")
                )
            else:
                await self._deny(result["errors"][0] if result["errors"] else "model unchanged")
            return

        if kind == "voice_set" and intent.value:
            result = registry.apply(voice=intent.value)
            if result["applied"]:
                await self._confirm(
                    f"Voice switched to {self.registry.voice}. Listening test: "
                    "all systems nominal, and sounding rather different, I trust."
                )
            else:
                await self._deny(result["errors"][0] if result["errors"] else "voice unchanged")
            return

        if kind == "speed_up":
            before = registry.speed
            registry.apply(speed=round(min(2.0, before + 0.15), 2))
            await self._confirm(f"Speaking faster: speed {registry.speed:.2f}.")
            return

        if kind == "speed_down":
            before = registry.speed
            registry.apply(speed=round(max(0.5, before - 0.15), 2))
            await self._confirm(f"Speaking slower: speed {registry.speed:.2f}.")
            return

        if kind == "speed_set" and intent.value:
            try:
                result = registry.apply(speed=float(intent.value))
            except ValueError:
                await self._deny(f"could not parse speed '{intent.value}'")
                return
            await self._confirm(f"Speed set to {registry.speed:.2f}.")
            return

        if kind == "help":
            await self._confirm(_HELP_TEXT)
            return

        if kind == "status":
            await self._confirm(
                f"Status: voice engine {self.tts.engine.name}, crew {self.runtime_name} on {registry.model}, "
                f"model {'verified' if registry.model_verified else 'unverified'}. All primary systems nominal."
            )
            return

        # Unhandled intent kind - fall through to a helpful denial.
        await self._deny("that settings command is not recognised. Say help for options.")

    async def _confirm(self, text: str) -> None:
        self.bus.publish("voice", "settings", text)
        self._broadcast_status("speaking")
        await self.tts.speak(text)
        self._broadcast_status("idle")

    async def _deny(self, text: str) -> None:
        self.bus.publish("error", "settings", text)
        self._broadcast_status("speaking")
        await self.tts.speak(text)
        self._broadcast_status("idle")

    def _broadcast_status(self, status: str, detail: str = "") -> None:
        self.bus.push_frame({"type": "status", "status": status, "detail": detail})
