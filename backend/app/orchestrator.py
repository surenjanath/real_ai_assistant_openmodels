"""Command orchestrator: command -> (control intent | agent crew) -> voice.

Routing order per command:
  1. control intents - settings, transport (stop/clear), time/date, help,
     status - handled inline with registry changes, UI control frames and
     spoken confirmations.
  2. otherwise the agent crew runs (Ollama live, or the simulation).

Every reasoning step streams to the telemetry bus; the answer streams to the
interface as `answer.delta` frames while it is still being generated, and the
finished text is handed to the TTS manager to be spoken.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from .agents.base import Runtime
from .agents.ollama_runtime import OllamaRuntime
from .agents.simulated_runtime import SimulatedRuntime
from .config import settings
from .intents import Intent, parse_intent
from .logbus import LogBus
from .registry import Registry, voice_label
from .tts.manager import TTSManager

_HELP_TEXT = (
    "Give me any directive and I will answer. For control, say: settings, to open the panel; "
    "list models, or switch model to a name; change voice to a name; speak faster or slower; "
    "status, for a system report; enable or disable thinking; stop, to cut me off; "
    "or new conversation, to clear my memory."
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
        self._last_answer = ""

    async def start(self) -> None:
        await self.registry.refresh_models()
        runtime = await OllamaRuntime.probe(self.bus, registry=self.registry)
        if runtime is None:
            runtime = SimulatedRuntime(self.bus, model_hint=self.registry.model)
            self.bus.publish(
                "warn",
                "brain",
                f"running SIMULATED crew - start ollama ({settings.ollama_base_url}) for live reasoning",
            )
        else:
            runtime.on_delta = self._emit_delta
        self.runtime = runtime
        self._worker = asyncio.create_task(self._worker_loop(), name="orchestrator")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    @property
    def runtime_name(self) -> str:
        return self.runtime.name if self.runtime else "booting"

    @property
    def mode(self) -> str:
        return getattr(self.runtime, "mode", "simulated")

    async def enqueue(self, text: str, origin: str = "text") -> None:
        await self._queue.put((text, origin))

    async def _worker_loop(self) -> None:
        while True:
            text, origin = await self._queue.get()
            try:
                await self.handle(text, origin)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.bus.publish("error", "workflow", f"orchestrator error: {type(exc).__name__}: {exc}")
                self._broadcast_status("idle")

    # -- streaming answer ------------------------------------------------------

    def _emit_delta(self, delta: str) -> None:
        """Called from the model worker thread as answer tokens arrive."""
        self.bus.push_frame({"type": "answer.delta", "text": delta})

    def _emit_answer(self, text: str) -> None:
        self.bus.push_frame({"type": "answer", "text": text})

    async def handle(self, text: str, origin: str = "text") -> None:
        text = " ".join(text.split()).strip()
        if not text:
            return

        intent = parse_intent(text)

        # Transport intents must not queue behind a running crew.
        if intent is not None and intent.kind == "stop":
            await self._handle_stop()
            return

        async with self._lock:
            source = "stt" if origin == "voice" else "you"
            self.bus.publish("voice" if origin == "voice" else "info", source, f'"{text}"')
            self.bus.push_frame({"type": "transcript", "role": "user", "text": text})

            if intent is not None:
                await self._handle_intent(intent)
                return

            started = time.monotonic()
            self._broadcast_status("thinking", text)
            self.bus.push_frame({"type": "answer.start"})
            if isinstance(self.runtime, SimulatedRuntime):
                self.runtime.model_hint = self.registry.model
            try:
                answer = await self.runtime.run(text)
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("error", "workflow", f"crew failed: {type(exc).__name__}: {exc}")
                await self._say("I ran into an error reaching the model. Check the telemetry log.")
                return
            elapsed = time.monotonic() - started
            self.bus.publish("success", "workflow", f"crew finished in {elapsed:.1f}s - routing to voice")
            self._last_answer = answer
            self._emit_answer(answer)
            self.bus.publish("info", "answer", answer)
            self.bus.push_frame({"type": "transcript", "role": "assistant", "text": answer})
            self._broadcast_status("speaking", answer[:120])
            await self.tts.speak(answer)
            self._broadcast_status("idle")

    # ------------------------------------------------------------------ intents

    async def _handle_stop(self) -> None:
        stopped = await self.tts.stop(reason="barge-in")
        self.bus.publish("warn", "control", "stop - speech halted" if stopped else "stop - nothing in flight")
        self._broadcast_status("idle")

    async def _handle_intent(self, intent: Intent) -> None:
        bus, registry, tts = self.bus, self.registry, self.tts
        kind = intent.kind

        if kind == "settings_show":
            bus.push_frame({"type": "ui", "action": "open_settings"})
            await self._say(
                f"Settings panel open. Model {registry.model or 'none'}, "
                f"voice {voice_label(registry.voice)}, speed {registry.speed:.1f}."
            )
            return

        if kind == "settings_hide":
            bus.push_frame({"type": "ui", "action": "close_settings"})
            await self._say("Closing the panel.")
            return

        if kind == "clear":
            bus.push_frame({"type": "ui", "action": "clear_logs"})
            bus.publish("info", "control", "telemetry cleared")
            await self._say("Log cleared.")
            return

        if kind == "memory_clear":
            if hasattr(self.runtime, "clear_memory"):
                self.runtime.clear_memory()
            bus.push_frame({"type": "ui", "action": "clear_transcript"})
            await self._say("Context cleared. Starting fresh.")
            return

        if kind == "time":
            now = datetime.now()
            await self._say(f"It is {now.strftime('%-I:%M %p').lower()}.")
            return

        if kind == "date":
            now = datetime.now()
            await self._say(f"Today is {now.strftime('%A, the %-d of %B, %Y')}.")
            return

        if kind == "identity":
            await self._say(
                f"I am {settings.name}, running entirely on this machine. "
                f"My reasoning uses {registry.model or 'no model yet'} through {self.runtime_name}, "
                f"and I speak with {tts.engine.name}."
            )
            return

        if kind == "repeat":
            if self._last_answer:
                self._broadcast_status("speaking")
                await tts.speak(self._last_answer)
                self._broadcast_status("idle")
            else:
                await self._say("I have not said anything yet.")
            return

        if kind == "models_list":
            await registry.refresh_models()
            installed = registry.installed
            if not installed:
                await self._deny("Ollama is not reachable, so I cannot list installed models.")
                return
            bus.publish("info", "settings", f"{len(installed)} model(s) installed:")
            for model in installed:
                bus.publish("info", "settings", f"  {model}{' ●' if model == registry.model else ''}")
            names = ", ".join(m.split(":")[0] for m in installed[:6])
            more = f", and {len(installed) - 6} more" if len(installed) > 6 else ""
            await self._say(f"I have {len(installed)} models installed: {names}{more}. Currently using {registry.model}.")
            return

        if kind == "voices_list":
            bus.publish("info", "settings", f"{len(registry.voices)} voices installed:")
            for voice in registry.voices:
                bus.publish("info", "settings", f"  {voice}{' ●' if voice == registry.voice else ''}")
            await self._say(
                f"I have {len(registry.voices)} voices, currently {voice_label(registry.voice)}. "
                "Say change voice to, followed by a name."
            )
            return

        if kind == "model_set" and intent.value:
            result = registry.apply(model=intent.value)
            if result["applied"]:
                await registry.refresh_models()
                note = "" if registry.model_verified else " It is not installed yet, so pull it first."
                await self._say(f"Model switched to {registry.model}.{note}")
            else:
                await self._deny(result["errors"][0] if result["errors"] else "Model unchanged.")
            return

        if kind == "voice_set" and intent.value:
            result = registry.apply(voice=intent.value)
            if result["applied"]:
                await self._say(f"Voice switched to {voice_label(registry.voice)}. How does this sound?")
            else:
                await self._deny(result["errors"][0] if result["errors"] else "Voice unchanged.")
            return

        if kind == "speed_up":
            registry.apply(speed=round(min(2.0, registry.speed + 0.15), 2))
            await self._say(f"Speaking faster. Speed {registry.speed:.2f}.")
            return

        if kind == "speed_down":
            registry.apply(speed=round(max(0.5, registry.speed - 0.15), 2))
            await self._say(f"Speaking slower. Speed {registry.speed:.2f}.")
            return

        if kind == "speed_set" and intent.value:
            result = registry.apply(speed=float(intent.value))
            if result["errors"]:
                await self._deny(result["errors"][0])
            else:
                await self._say(f"Speed set to {registry.speed:.2f}.")
            return

        if kind in ("think_on", "think_off"):
            want = kind == "think_on"
            if want and not registry.think_supported:
                await self._deny(
                    f"{registry.model} has no separate thinking mode, so I already answer directly."
                )
                return
            result = registry.apply(think=want)
            if result["errors"]:
                await self._deny(result["errors"][0])
            elif want:
                await self._say("Extended thinking enabled. I will reason before answering, which takes longer.")
            else:
                await self._say("Extended thinking disabled. I will answer straight away.")
            return

        if kind == "help":
            await self._say(_HELP_TEXT)
            return

        if kind == "status":
            engine = tts.engine
            engine_label = getattr(engine, "label", engine.name)
            await self._say(
                f"Voice engine {engine_label}. Reasoning through {self.runtime_name} "
                f"in {self.mode} mode on {registry.model or 'no model'}, "
                f"{'verified' if registry.model_verified else 'not installed'}. "
                f"Extended thinking is {'on' if registry.think_active else 'off'}. "
                "All primary systems nominal."
            )
            return

        await self._deny("That control command is not recognised. Say help for options.")

    async def _say(self, text: str) -> None:
        """Speak a control response and mirror it into the transcript."""
        self.bus.publish("voice", "control", text)
        self.bus.push_frame({"type": "transcript", "role": "assistant", "text": text})
        self._emit_answer(text)
        self._last_answer = text
        self._broadcast_status("speaking", text[:120])
        await self.tts.speak(text)
        self._broadcast_status("idle")

    async def _deny(self, text: str) -> None:
        self.bus.publish("error", "control", text)
        self.bus.push_frame({"type": "transcript", "role": "assistant", "text": text})
        self._broadcast_status("speaking")
        await self.tts.speak(text)
        self._broadcast_status("idle")

    def _broadcast_status(self, status: str, detail: str = "") -> None:
        self.bus.push_frame({"type": "status", "status": status, "detail": detail})
