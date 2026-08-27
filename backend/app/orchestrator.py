"""Command orchestrator: command -> (control intent | agent crew) -> voice.

Routing order per command:
  1. control intents - settings, persona, memory, notes, reminders, transport
     (stop/clear), time/date, help, status - handled inline with registry
     changes, UI control frames and spoken confirmations.
  2. otherwise the agent crew runs (Ollama live, or the simulation), with
     durable recall injected and skills available as native tool calls.

Every reasoning step streams to the telemetry bus; the answer streams to the
interface as `answer.delta` frames while it is still being generated, and the
finished text is handed to the TTS manager to be spoken.

Three cross-cutting subsystems ride along:

  * **neural**  - every stage fires a node on the cognitive graph, so the
    interface can draw the signal actually travelling through the assistant.
  * **memory**  - each turn is persisted, and relevant fragments of earlier
    sessions are recalled into the prompt.
  * **metrics** - each directive is timed end to end and broadcast.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from .agents.base import Runtime
from .agents.ollama_runtime import OllamaRuntime
from .agents.simulated_runtime import SimulatedRuntime
from .config import settings
from .intents import Intent, parse_intent
from .logbus import LogBus
from .memory import Memory
from .metrics import Metrics, Run
from .personas import PERSONALITIES, resolve as resolve_persona
from .registry import Registry, voice_label
from .skills import SkillKit, parse_when
from .tts.manager import TTSManager

_HELP_TEXT = (
    "Give me any directive and I will answer. For control, say: settings, to open the panel; "
    "list models, or switch model to a name; change voice to a name; speak faster or slower; "
    "be more concise, or switch persona to Friday; remember that, followed by a fact; "
    "what do you know about, followed by a subject; make a note; remind me to do something "
    "in ten minutes; performance report; status, for a system report; enable or disable "
    "thinking; stop, to cut me off; or new conversation, to clear my memory."
)


def _ago(ts: float) -> str:
    """Human-scale age of a timestamp, for spoken recall."""
    seconds = max(0.0, time.time() - ts)
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{round(minutes)} minutes ago"
    hours = minutes / 60
    if hours < 36:
        return f"{round(hours)} hours ago"
    return f"{round(hours / 24)} days ago"


class Orchestrator:
    def __init__(self, bus: LogBus, tts: TTSManager, registry: Registry,
                 memory: Memory | None = None, kit: SkillKit | None = None,
                 neural: Any | None = None, metrics: Metrics | None = None,
                 vitals_provider: Any | None = None) -> None:
        self.bus = bus
        #: callable returning the latest host metrics, for reflex grounding
        self.vitals_provider = vitals_provider
        self.tts = tts
        self.registry = registry
        self.memory = memory
        self.kit = kit
        self.neural = neural
        self.metrics = metrics
        self.runtime: Runtime | None = None
        self._lock = asyncio.Lock()  # serialise workflows - one crew at a time
        self._worker: asyncio.Task | None = None
        self._reminders: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=16)
        self._last_answer = ""
        self._run: Run | None = None
        #: row id of the directive currently being answered, kept out of its own recall
        self._live_turn_id: int | None = None

    async def start(self) -> None:
        await self.registry.refresh_models()
        runtime = await OllamaRuntime.probe(
            self.bus, registry=self.registry, kit=self.kit, neural=self.neural
        )
        if runtime is None:
            runtime = SimulatedRuntime(self.bus, model_hint=self.registry.model)
            self.bus.publish(
                "warn",
                "brain",
                f"running SIMULATED crew - start ollama ({settings.ollama_base_url}) for live reasoning",
            )
        else:
            runtime.on_delta = self._emit_delta
            runtime.on_reset = self._emit_reset
            runtime.on_tool = self._emit_tool
            runtime.vitals_provider = self.vitals_provider
            if self.memory is not None:
                runtime.context_provider = self._recall_context
        self.runtime = runtime
        self._worker = asyncio.create_task(self._worker_loop(), name="orchestrator")
        if self.memory is not None:
            self._reminders = asyncio.create_task(self._reminder_loop(), name="reminders")

    async def stop(self) -> None:
        for task in (self._worker, self._reminders):
            if task:
                task.cancel()
        await asyncio.gather(
            *[t for t in (self._worker, self._reminders) if t], return_exceptions=True
        )

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

    # -- durable recall --------------------------------------------------------

    def _recall_context(self, text: str) -> str:
        """Build the grounding block injected ahead of a directive.

        Runs in a worker thread (SQLite is synchronous), and stays deliberately
        small: recall that floods the context window costs latency on every
        single turn and buries the actual question.
        """
        if self.memory is None or not self.registry.recall:
            return ""
        try:
            lines: list[str] = []
            facts = self.memory.all_facts(limit=12)
            if facts:
                lines.append("Durable facts you have been told about the operator:")
                lines.extend(f"- {f['key']}: {f['value']}" for f in facts)

            exclude = (self._live_turn_id,) if self._live_turn_id else ()
            hits = [
                h for h in self.memory.search(text, limit=settings.recall_limit,
                                              exclude_ids=exclude)
                if h.score >= settings.recall_threshold
            ]
            if hits:
                lines.append("Relevant fragments of earlier conversations:")
                for hit in hits:
                    who = "operator" if hit.role == "user" else "you"
                    lines.append(f"- ({_ago(hit.ts)}) {who}: {hit.text[:280]}")
            if not lines:
                return ""
            lines.append(
                "These are a record of what was said, not a source of truth: an earlier answer "
                "of yours may have been wrong. Use them only when actually relevant, never "
                "recite them unprompted, and prefer a tool call over repeating a remembered "
                "figure."
            )
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - recall must never break a turn
            self.bus.publish("warn", "memory", f"recall failed: {type(exc).__name__}")
            return ""

    # -- streaming answer ------------------------------------------------------

    def _emit_delta(self, delta: str) -> None:
        """Called from the model worker thread as answer tokens arrive."""
        if self._run is not None:
            self._run.mark_first_token()
        if self.neural is not None:
            self.neural.fire("out.answer", 0.25)
        self.bus.push_frame({"type": "answer.delta", "text": delta})

    def _emit_reset(self) -> None:
        """A tool round intervened: discard the partial caption."""
        self.bus.push_frame({"type": "answer.start"})

    def _emit_tool(self, name: str, arguments: dict, result: dict) -> None:
        if self._run is not None:
            self._run.tool_calls += 1
            if name not in self._run.tools_used:
                self._run.tools_used.append(name)
        self.bus.push_frame({
            "type": "tool",
            "name": name,
            "args": arguments,
            "ok": bool(result.get("ok")),
            "detail": str(result.get("error") or result.get("result"))[:400],
            "elapsed_ms": result.get("elapsed_ms", 0),
            "ts": time.time(),
        })

    def _emit_answer(self, text: str) -> None:
        self.bus.push_frame({"type": "answer", "text": text})

    def _fire(self, *nodes: str, intensity: float = 1.0) -> None:
        if self.neural is not None:
            if len(nodes) == 1:
                self.neural.fire(nodes[0], intensity)
            else:
                self.neural.path(*nodes, intensity=intensity)

    def _persist(self, role: str, text: str, *, latency_ms: int | None = None,
                 origin: str = "text") -> int | None:
        if self.memory is None or not settings.persist:
            return None
        try:
            turn_id = self.memory.add_turn(role, text, model=self.registry.model,
                                           latency_ms=latency_ms, origin=origin)
            if role == "assistant":
                self._fire("mem.write", 0.8)
            return turn_id
        except Exception as exc:  # noqa: BLE001
            self.bus.publish("warn", "memory", f"persist failed: {type(exc).__name__}")
            return None

    # -- main entry ------------------------------------------------------------

    async def handle(self, text: str, origin: str = "text") -> None:
        text = " ".join(text.split()).strip()
        if not text:
            return

        self._fire("sense.voice" if origin == "voice" else "sense.text", "intake.intent")
        intent = parse_intent(text)

        # Transport intents must not queue behind a running crew.
        if intent is not None and intent.kind == "stop":
            await self._handle_stop()
            return

        async with self._lock:
            source = "stt" if origin == "voice" else "you"
            self.bus.publish("voice" if origin == "voice" else "info", source, f'"{text}"')
            self.bus.push_frame({"type": "transcript", "role": "user", "text": text})
            self._live_turn_id = self._persist("user", text, origin=origin)

            if intent is not None:
                self._fire("intake.intent", "intake.control")
                run = None
                if self.metrics is not None:
                    run = self.metrics.begin(text, origin, self.registry.model,
                                             self.mode, kind="control")
                    self._run = run
                try:
                    await self._handle_intent(intent)
                finally:
                    if run is not None and self.metrics is not None:
                        run.chars = len(self._last_answer)
                        self.metrics.finish(run)
                        self._run = None
                return

            started = time.monotonic()
            run = None
            if self.metrics is not None:
                run = self.metrics.begin(text, origin, self.registry.model, self.mode)
                self._run = run

            self._fire("intake.intent", "intake.route")
            self._broadcast_status("thinking", text)
            self.bus.push_frame({"type": "answer.start"})
            if isinstance(self.runtime, SimulatedRuntime):
                self.runtime.model_hint = self.registry.model
            try:
                answer = await self.runtime.run(text)
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("error", "workflow", f"crew failed: {type(exc).__name__}: {exc}")
                if run is not None and self.metrics is not None:
                    run.error = type(exc).__name__
                    self.metrics.finish(run)
                    self._run = None
                await self._say("I ran into an error reaching the model. Check the telemetry log.")
                return

            elapsed = time.monotonic() - started
            if run is not None:
                run.composed = time.monotonic()
                run.chars = len(answer)
                # Report the reasoning timings now: the run itself is not
                # closed until speech ends, which can be many seconds later.
                if self.metrics is not None:
                    self.metrics.publish(run)
            self.bus.publish(
                "success", "workflow",
                f"crew finished in {elapsed:.1f}s"
                + (f" using {run.tool_calls} tool call(s)" if run and run.tool_calls else "")
                + " - routing to voice",
            )
            self._last_answer = answer
            self._emit_answer(answer)
            self.bus.publish("info", "answer", answer)
            self.bus.push_frame({"type": "transcript", "role": "assistant", "text": answer})
            self._persist("assistant", answer, latency_ms=int(elapsed * 1000), origin=origin)
            self._fire("out.answer", "out.tts")
            self._broadcast_status("speaking", answer[:120])
            await self.tts.speak(answer)
            self._fire("out.tts", "out.audio")
            if run is not None and self.metrics is not None:
                run.spoken = time.monotonic()
                self.metrics.finish(run)
                self._run = None
            self._broadcast_status("idle")

    # -- reminders ---------------------------------------------------------------

    async def _reminder_loop(self) -> None:
        """Fire due reminders aloud. The only path where J.A.R.V.I.S. speaks
        without having been spoken to, so it is deliberately conservative:
        nothing fires twice, and nothing interrupts an utterance in flight."""
        assert self.memory is not None
        while True:
            await asyncio.sleep(max(5.0, settings.reminder_interval_s))
            try:
                due = await asyncio.to_thread(self.memory.due_reminders)
                for item in due:
                    # Wait for a gap rather than talking over the assistant.
                    while self.tts.speaking:
                        await asyncio.sleep(1.0)
                    await asyncio.to_thread(self.memory.mark_fired, item["id"])
                    self._fire("mem.recall", "out.answer")
                    self.bus.publish("success", "reminder", item["text"])
                    self.bus.push_frame({
                        "type": "reminder", "id": item["id"], "text": item["text"],
                        "due_ts": item["due_ts"],
                    })
                    await self._say(f"Reminder, {settings.operator}: {item['text']}.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("warn", "reminder", f"scheduler hiccup: {type(exc).__name__}")

    # ------------------------------------------------------------------ intents

    async def _handle_stop(self) -> None:
        stopped = await self.tts.stop(reason="barge-in")
        self.bus.publish("warn", "control", "stop - speech halted" if stopped else "stop - nothing in flight")
        self._broadcast_status("idle")

    async def _handle_intent(self, intent: Intent) -> None:  # noqa: PLR0911, PLR0912, PLR0915
        bus, registry, tts = self.bus, self.registry, self.tts
        kind = intent.kind

        if kind == "settings_show":
            bus.push_frame({"type": "ui", "action": "open_settings"})
            await self._say(
                f"Settings panel open. Model {registry.model or 'none'}, "
                f"voice {voice_label(registry.voice)}, speed {registry.speed:.1f}, "
                f"persona {resolve_persona(registry.persona).label}."
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
            persona = resolve_persona(registry.persona)
            await self._say(
                f"I am {settings.name}, running entirely on this machine. "
                f"My reasoning uses {registry.model or 'no model yet'} through {self.runtime_name}, "
                f"I speak with {tts.engine.name}, and I am currently in the {persona.label} disposition."
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
            await self._say(self._status_report())
            return

        # ---- persona ---------------------------------------------------------

        if kind == "persona_list":
            for persona in PERSONALITIES.values():
                marker = " ●" if persona.key == registry.persona else ""
                bus.publish("info", "settings", f"  {persona.label}: {persona.blurb}{marker}")
            names = ", ".join(p.label for p in PERSONALITIES.values())
            await self._say(
                f"I have {len(PERSONALITIES)} dispositions: {names}. "
                f"Currently {resolve_persona(registry.persona).label}."
            )
            return

        if kind == "persona_set" and intent.value:
            result = registry.apply(persona=intent.value)
            if result["errors"]:
                await self._deny(result["errors"][0])
                return
            persona = resolve_persona(registry.persona)
            if result["applied"]:
                await self._say(f"{persona.label} disposition engaged. {persona.blurb}")
            else:
                await self._say(f"Already in the {persona.label} disposition.")
            return

        # ---- tools -----------------------------------------------------------

        if kind == "tools_list":
            if self.kit is None:
                await self._deny("No skill kit is loaded.")
                return
            catalogue = self.kit.catalogue()
            for skill in catalogue:
                bus.publish("info", "skills", f"  {skill['name']} — {skill['description'][:110]}")
            state = (
                "armed" if registry.tools_active
                else "available but the current model cannot call them"
                if registry.tools else "disabled"
            )
            names = ", ".join(s["name"].replace("_", " ") for s in catalogue[:6])
            await self._say(
                f"I have {len(catalogue)} skills, currently {state}: {names}, and more in the log."
            )
            return

        if kind in ("tools_on", "tools_off"):
            want = kind == "tools_on"
            if want and not registry.tools_supported:
                await self._deny(
                    f"{registry.model} does not advertise tool calling. "
                    "Switch to a tool-capable model such as qwen3 or llama3.1 first."
                )
                return
            result = registry.apply(tools=want)
            if result["errors"]:
                await self._deny(result["errors"][0])
            else:
                await self._say(
                    "Skills armed. I will call them when a question needs real data."
                    if want else "Skills disabled. I will answer from the model alone."
                )
            return

        # ---- durable memory ---------------------------------------------------

        if kind in ("recall_on", "recall_off"):
            registry.apply(recall=kind == "recall_on")
            await self._say(
                "Long-term recall enabled. I will draw on our earlier conversations."
                if kind == "recall_on"
                else "Long-term recall disabled. I will use only this conversation."
            )
            return

        if kind == "remember" and intent.value:
            if self.memory is None:
                await self._deny("My long-term memory is not available.")
                return
            key, value = self._split_fact(intent.value)
            await asyncio.to_thread(self.memory.remember_fact, key, value)
            self._fire("mem.write", 1.0)
            bus.publish("success", "memory", f"stored '{key}' = {value[:80]}")
            await self._say(f"Noted. I will remember that {value.rstrip('.')}.")
            return

        if kind == "recall_query" and intent.value:
            if self.memory is None:
                await self._deny("My long-term memory is not available.")
                return
            self._fire("mem.recall", 1.0)
            query = intent.value
            fact = await asyncio.to_thread(self.memory.recall_fact, query)
            hits = await asyncio.to_thread(self.memory.search, query, 4)
            if not fact and not hits:
                await self._say(f"I have nothing on record about {query}.")
                return
            for hit in hits:
                bus.publish("info", "memory", f"  ({_ago(hit.ts)}) {hit.role}: {hit.text[:140]}")
            if fact:
                await self._say(f"You told me that {fact}.")
            else:
                best = hits[0]
                who = "You said" if best.role == "user" else "I said"
                await self._say(
                    f"{who}, {_ago(best.ts)}: {best.text[:220]}"
                    + (f" There are {len(hits) - 1} more matches in the log." if len(hits) > 1 else "")
                )
            return

        if kind == "memory_stats":
            if self.memory is None:
                await self._deny("My long-term memory is not available.")
                return
            stats = await asyncio.to_thread(self.memory.stats)
            bus.publish("info", "memory", f"store at {stats['path']} ({stats['size_kb']} KB)")
            since = (
                datetime.fromtimestamp(stats["since"]).strftime("%-d %B")
                if stats.get("since") else "today"
            )
            await self._say(
                f"I am holding {stats['turns']} exchanges across {stats['sessions']} sessions "
                f"since {since}, plus {stats['facts']} facts, {stats['notes']} notes "
                f"and {stats['reminders']} pending reminders."
            )
            return

        if kind == "memory_wipe":
            if self.memory is None:
                await self._deny("My long-term memory is not available.")
                return
            removed = await asyncio.to_thread(self.memory.forget_all)
            if hasattr(self.runtime, "clear_memory"):
                self.runtime.clear_memory()
            bus.push_frame({"type": "ui", "action": "clear_transcript"})
            bus.publish("warn", "memory", f"long-term memory erased - {removed} turn(s) removed")
            await self._say(
                f"Long-term memory erased. {removed} exchanges, every stored fact, note and "
                "reminder are gone."
            )
            return

        # ---- notes -------------------------------------------------------------

        if kind == "note_add" and intent.value:
            if self.memory is None:
                await self._deny("My note store is not available.")
                return
            note_id = await asyncio.to_thread(self.memory.add_note, intent.value, "")
            self._fire("mem.write", 1.0)
            bus.publish("success", "notes", f"#{note_id} {intent.value[:120]}")
            await self._say("Note saved.")
            return

        if kind == "notes_list":
            if self.memory is None:
                await self._deny("My note store is not available.")
                return
            notes = await asyncio.to_thread(self.memory.list_notes, 20)
            if not notes:
                await self._say("You have no notes.")
                return
            for note in notes:
                bus.publish("info", "notes", f"  #{note['id']} {note['text'][:140]}")
            head = notes[0]["text"]
            await self._say(
                f"You have {len(notes)} notes. The most recent: {head[:200]}"
                if len(notes) > 1 else f"One note: {head[:220]}"
            )
            return

        # ---- reminders -----------------------------------------------------------

        if kind == "reminder_add" and intent.value:
            if self.memory is None:
                await self._deny("My reminder scheduler is not available.")
                return
            when = parse_when(intent.value)
            if when is None:
                await self._deny(
                    "I could not read a time in that. Try: remind me to stretch in twenty minutes."
                )
                return
            # Strip the time phrase out of the reminder body so the spoken
            # reminder reads "call Dr Chen", not "call Dr Chen in 20 minutes".
            body = self._strip_when(intent.value)
            reminder_id = await asyncio.to_thread(self.memory.add_reminder, body, when.timestamp())
            bus.publish("success", "reminder", f"#{reminder_id} at {when:%a %H:%M} - {body[:100]}")
            bus.push_frame({"type": "reminder.set", "id": reminder_id, "text": body,
                            "due_ts": when.timestamp()})
            await self._say(f"I will remind you to {body} at {when.strftime('%-I:%M %p').lower()}.")
            return

        if kind == "reminders_list":
            if self.memory is None:
                await self._deny("My reminder scheduler is not available.")
                return
            pending = await asyncio.to_thread(self.memory.pending_reminders, 20)
            if not pending:
                await self._say("You have no pending reminders.")
                return
            for item in pending:
                due = datetime.fromtimestamp(item["due_ts"]).strftime("%a %-I:%M %p")
                bus.publish("info", "reminder", f"  #{item['id']} {due} - {item['text'][:100]}")
            first = pending[0]
            due = datetime.fromtimestamp(first["due_ts"]).strftime("%-I:%M %p").lower()
            await self._say(
                f"{len(pending)} pending. The next is {first['text']}, at {due}."
            )
            return

        # ---- volume ---------------------------------------------------------------

        if kind in ("volume_up", "volume_down"):
            step = 0.12 if kind == "volume_up" else -0.12
            registry.apply(volume=round(min(1.0, max(0.0, registry.volume + step)), 2))
            await self._say(f"Volume {round(registry.volume * 100)} percent.")
            return

        if kind == "volume_set" and intent.value:
            result = registry.apply(volume=float(intent.value))
            if result["errors"]:
                await self._deny(result["errors"][0])
            else:
                await self._say(f"Volume set to {round(registry.volume * 100)} percent.")
            return

        if kind == "volume_mute":
            registry.apply(volume=0.0)
            bus.publish("warn", "control", "output muted")
            return

        if kind == "volume_unmute":
            registry.apply(volume=max(0.6, registry.volume))
            await self._say("Audio restored.")
            return

        # ---- crew mode --------------------------------------------------------------

        if kind in ("crew_on", "crew_off"):
            settings.crew_mode = "crew" if kind == "crew_on" else "fast"
            registry.broadcast()
            bus.publish("info", "settings", f"crew mode -> {settings.crew_mode}")
            await self._say(
                "Full crew engaged. Router, analyst, engineer and synthesiser will each take a pass, "
                "which is slower but more thorough."
                if kind == "crew_on"
                else "Back to the fast single pass. Conversational latency restored."
            )
            return

        # ---- metrics -------------------------------------------------------------------

        if kind == "metrics":
            await self._say(self._metrics_report())
            return

        await self._deny("That control command is not recognised. Say help for options.")

    # -- reports ------------------------------------------------------------------

    def _status_report(self) -> str:
        registry, tts = self.registry, self.tts
        engine_label = getattr(tts.engine, "label", tts.engine.name)
        persona = resolve_persona(registry.persona)
        parts = [
            f"Voice engine {engine_label}.",
            f"Reasoning through {self.runtime_name} in {self.mode} mode on "
            f"{registry.model or 'no model'}, "
            f"{'verified' if registry.model_verified else 'not installed'}.",
            f"Disposition {persona.label}.",
            f"Extended thinking is {'on' if registry.think_active else 'off'}.",
        ]
        if self.kit is not None:
            parts.append(
                f"{len(self.kit.skills)} skills "
                f"{'armed' if registry.tools_active else 'standing by'}."
            )
        if self.memory is not None:
            try:
                stats = self.memory.stats()
                parts.append(f"{stats['turns']} exchanges in long-term memory.")
            except Exception:  # noqa: BLE001
                pass
        if self.neural is not None:
            parts.append(f"{len(self.neural.nodes)} cognitive nodes online.")
        parts.append("All primary systems nominal.")
        return " ".join(parts)

    def _metrics_report(self) -> str:
        if self.metrics is None or not self.metrics.runs:
            return "I have not answered anything yet, so there is nothing to measure."
        snap = self.metrics.snapshot()
        ttft = snap["ttft_ms"]["p50"] / 1000
        total = snap["total_ms"]["p50"] / 1000
        rate = snap["tok_s"]["avg"]
        line = (
            f"Across {snap['commands']} directives, my median time to first word is "
            f"{ttft:.1f} seconds and a full answer takes {total:.1f} seconds, "
            f"generating about {rate:.0f} tokens per second."
        )
        if snap["tool_calls"]:
            line += f" I have made {snap['tool_calls']} tool calls."
        if snap["errors"]:
            line += f" {snap['errors']} runs failed."
        return line

    # -- helpers ---------------------------------------------------------------------

    @staticmethod
    def _split_fact(text: str) -> tuple[str, str]:
        """Derive a stable key from a spoken fact.

        "my sister is called Ada" -> key "my sister", value "my sister is
        called Ada". Keeping the whole sentence as the value means recall reads
        back naturally; the key exists only so a later correction overwrites
        rather than duplicates.
        """
        lowered = text.lower()
        for marker in (" is called ", " is named ", " is ", " are ", " was ", " lives ", " likes "):
            index = lowered.find(marker)
            if 0 < index <= 60:
                return text[:index].strip(" ,.").lower(), text.strip()
        return " ".join(text.split()[:4]).lower().strip(" ,."), text.strip()

    @staticmethod
    def _strip_when(text: str) -> str:
        """Remove the scheduling clause from a reminder body."""
        import re  # noqa: PLC0415 - one small local use

        body = re.sub(
            r"\s+(?:in|at)\s+(?:\d[\w:. ]*|an?\s+\w+)(?:\s*(?:am|pm))?\s*$", "", text, flags=re.I
        )
        body = re.sub(r"\s+tomorrow(?:\s+at\s+[\w: ]+)?\s*$", "", body, flags=re.I)
        return body.strip(" ,.") or text

    async def _say(self, text: str) -> None:
        """Speak a control response and mirror it into the transcript."""
        self.bus.publish("voice", "control", text)
        self.bus.push_frame({"type": "transcript", "role": "assistant", "text": text})
        self._emit_answer(text)
        self._last_answer = text
        self._persist("assistant", text)
        self._fire("intake.control", "out.answer")
        self._broadcast_status("speaking", text[:120])
        self._fire("out.answer", "out.tts")
        await self.tts.speak(text)
        self._fire("out.tts", "out.audio")
        self._broadcast_status("idle")

    async def _deny(self, text: str) -> None:
        self.bus.publish("error", "control", text)
        self.bus.push_frame({"type": "transcript", "role": "assistant", "text": text})
        self._last_answer = text
        self._broadcast_status("speaking")
        await self.tts.speak(text)
        self._broadcast_status("idle")

    def _broadcast_status(self, status: str, detail: str = "") -> None:
        self.bus.push_frame({"type": "status", "status": status, "detail": detail})
