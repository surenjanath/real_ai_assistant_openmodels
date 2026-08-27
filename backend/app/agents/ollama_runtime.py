"""Live agent runtime backed by a local Ollama server.

Three strategies, selected by `JARVIS_CREW_MODE` / `JARVIS_USE_CREWAI`:

1. **fast** (default) - a single streamed call in the J.A.R.V.I.S. persona with
   short-term conversational memory, durable recall and native tool calling.
   One model round-trip when no tool is needed, so replies start speaking in
   about a second. This is what you want for actual conversation.
2. **crew** - the four-persona pipeline (Router -> Analyst -> Engineer ->
   Synthesiser), each step streamed to the telemetry panel. Slower but shows
   real multi-agent reasoning.
3. **crewai** - the same crew built with the real `crewai` package.

Tool calling
------------
When the selected model advertises the `tools` capability, the skill kit's
JSON schemas are attached to the request and the runtime runs the full loop:
model -> `tool_calls` -> execute locally -> feed results back -> model. Each
executed tool grows and fires a node on the neural graph, so the interface
shows capability being exercised rather than merely declared.

Thinking models
---------------
Modern local models (qwen3.x, deepseek-r1, and friends) emit chain-of-thought
inside `<think>...</think>` before the answer. Left alone that reasoning gets
spoken aloud, which is both wrong and very long. `ThinkFilter` below splits the
stream: reasoning is routed to the telemetry panel as `agent/thought` lines,
and only the post-`</think>` prose reaches the voice engine.

Only stdlib HTTP is used (urllib in a worker thread), because the whole point
of this project is to run lean on a laptop.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import platform
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..config import settings
from ..logbus import LogBus
from ..personas import resolve as resolve_persona
from ..reflexes import as_prompt as reflex_prompt, detect as detect_reflexes
from .base import AgentPersonas, Persona

if TYPE_CHECKING:
    from ..registry import Registry

#: The invariant half of the persona prompt: grounding the model cannot infer.
JARVIS_SYSTEM = (
    "You are {name}, a locally hosted AI assistant, speaking directly to {operator}. "
    "The current date and time is {now}, and you are running on {host}. Use these when asked - "
    "you genuinely do know them. "
    "Your replies are converted to speech, so write plain spoken prose: no markdown, no bullet "
    "points, no code blocks, no emoji, no stage directions. {style} "
    "Never mention that you are a language model. If you do not know something, say so "
    "plainly rather than inventing it."
)

#: Measured, not guessed: with a softer phrasing ("call them rather than
#: guessing whenever...") qwen3.5:9b answered `48271 * 9912` from memory and got
#: it wrong by two thousand. Naming the triggers explicitly, and calling a guess
#: a failure, is what actually moves the model to emit a tool call.
_TOOL_HINT = (
    " You have tools. You MUST call a tool instead of answering from memory whenever the "
    "question depends on: the current date or time; live machine state; arithmetic on numbers "
    "longer than two digits; the contents of the operator's files; or anything said in an "
    "earlier conversation. Guessing at any of these is a failure. Call the tool first, then "
    "answer in plain spoken prose using the result, without mentioning the mechanics."
)

_SYSTEM_TMPL = (
    "You are {role} on the J.A.R.V.I.S. crew. Goal: {goal}. Backstory: {backstory} "
    "Answer in at most 6 short lines. No markdown, no preamble, no sign-off."
)
_SYNTH_TMPL = (
    "You are the Response Synthesizer for J.A.R.V.I.S. Compose the final spoken answer "
    "to the user's command using the crew findings below. Speak with composed British butler-AI "
    "tone, under 90 words, plain prose only (it will be spoken aloud). "
    "Command: {command}\nFindings: {findings}"
)

def _max_tool_rounds() -> int:
    """Hard ceiling on model -> tool -> model round-trips for one directive."""
    return max(1, min(8, settings.tool_rounds))


def _http_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@dataclass
class StreamEvent:
    """One decoded frame from Ollama's streaming chat endpoint."""

    kind: str  # "delta" | "think" | "tools"
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


def _stream_chat(url: str, payload: dict) -> Iterator[StreamEvent]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=settings.agent_step_timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            # Some builds surface reasoning in a dedicated field.
            thinking = message.get("thinking") or ""
            if thinking:
                yield StreamEvent("think", thinking)
            calls = message.get("tool_calls") or []
            if calls:
                yield StreamEvent("tools", tool_calls=list(calls))
            delta = message.get("content") or event.get("response") or ""
            if delta:
                yield StreamEvent("delta", delta)
            if event.get("done"):
                return


class ThinkFilter:
    """Splits a token stream into reasoning and answer halves.

    Handles `<think>`/`</think>` tags arriving split across chunk boundaries by
    holding back any trailing partial-tag text until it can be resolved.
    """

    _OPEN = re.compile(r"<think(?:ing)?>", re.I)
    _CLOSE = re.compile(r"</think(?:ing)?>", re.I)

    def __init__(self) -> None:
        self.in_think = False
        self._buf = ""

    def feed(self, delta: str) -> tuple[str, str]:
        """Return (answer_text, thought_text) released by this delta."""
        answer: list[str] = []
        thought: list[str] = []
        self._buf += delta
        while self._buf:
            pattern = self._CLOSE if self.in_think else self._OPEN
            match = pattern.search(self._buf)
            if match:
                head, self._buf = self._buf[: match.start()], self._buf[match.end():]
                (thought if self.in_think else answer).append(head)
                self.in_think = not self.in_think
                continue
            # No complete tag. Hold back a possible partial tag at the tail.
            hold = 0
            tail = self._buf[-12:]
            idx = tail.rfind("<")
            if idx != -1 and ">" not in tail[idx:]:
                hold = len(tail) - idx
            emit = self._buf[: len(self._buf) - hold] if hold else self._buf
            self._buf = self._buf[len(self._buf) - hold:] if hold else ""
            (thought if self.in_think else answer).append(emit)
            break
        return "".join(answer), "".join(thought)

    def flush(self) -> tuple[str, str]:
        rest, self._buf = self._buf, ""
        return ("", rest) if self.in_think else (rest, "")


def strip_think(text: str) -> str:
    """Remove complete (or unterminated) reasoning blocks from a full string."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", " ", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think(?:ing)?>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<think(?:ing)?>.*$", " ", text, flags=re.S | re.I)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Turn:
    """What one `_consume` pass produced."""

    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    chars: int = 0


class OllamaRuntime:
    name = "ollama"

    def __init__(self, bus: LogBus, base_url: str | None = None, model: str | None = None,
                 registry: "Registry | None" = None, kit: Any | None = None,
                 neural: Any | None = None) -> None:
        self.bus = bus
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._static_model = model
        self.registry = registry
        #: skill kit for native tool calling (optional)
        self.kit = kit
        #: neural graph, fired as the signal passes through each stage
        self.neural = neural
        #: chat history as [{"role": "user"|"assistant", "content": str}]
        self.memory: list[dict[str, str]] = []
        #: optional sink for streamed answer deltas (wired by the orchestrator)
        self.on_delta: Callable[[str], None] | None = None
        #: called when a partial answer must be discarded (a tool round intervened)
        self.on_reset: Callable[[], None] | None = None
        #: called as (name, arguments, result) for every executed tool
        self.on_tool: Callable[[str, dict, dict], None] | None = None
        #: extra grounding injected ahead of the user turn (durable recall)
        self.context_provider: Callable[[str], str] | None = None
        #: latest host metrics, so reflexes can ground live-state questions
        self.vitals_provider: Callable[[], dict] | None = None

    @property
    def model(self) -> str:
        """Current crew model - follows live registry switches."""
        if self.registry is not None:
            return self.registry.model
        return self._static_model or settings.ollama_model

    @property
    def think(self) -> bool:
        """Whether to let the model emit chain-of-thought before answering."""
        if self.registry is not None:
            return bool(getattr(self.registry, "think_active", False))
        return settings.think

    @property
    def tools_enabled(self) -> bool:
        """Tools require a kit, the user's consent, and model support."""
        if self.kit is None or self.registry is None:
            return False
        if not getattr(self.registry, "tools", True):
            return False
        return bool(getattr(self.registry, "tools_supported", False))

    def _apply_think(self, payload: dict) -> dict:
        """Attach Ollama's `think` flag when the model actually supports it.

        Reasoning models default to a long chain-of-thought that can add tens of
        seconds before the first spoken word. Sending `think: false` suppresses
        it. The flag is only sent when /api/tags advertised the capability -
        Ollama rejects it outright on models that lack one.
        """
        supported = (
            getattr(self.registry, "think_supported", False)
            if self.registry is not None
            else True
        )
        if supported:
            payload["think"] = self.think
        return payload

    @property
    def mode(self) -> str:
        if settings.use_crewai:
            return "crewai"
        return "crew" if settings.crew_mode.lower() == "crew" else "fast"

    def clear_memory(self) -> None:
        self.memory.clear()

    def _remember(self, role: str, content: str) -> None:
        if not content:
            return
        self.memory.append({"role": role, "content": content})
        limit = max(2, settings.memory_turns * 2)
        if len(self.memory) > limit:
            del self.memory[: len(self.memory) - limit]

    def _fire(self, node: str, intensity: float = 1.0) -> None:
        if self.neural is not None:
            self.neural.fire(node, intensity)

    # -- probe -----------------------------------------------------------------

    @classmethod
    async def probe(cls, bus: LogBus, base_url: str | None = None, model: str | None = None,
                    registry: "Registry | None" = None, kit: Any | None = None,
                    neural: Any | None = None) -> "OllamaRuntime | None":
        runtime = cls(bus, base_url, model, registry, kit, neural)
        ok = await runtime._probe()
        return runtime if ok else None

    async def _probe(self) -> bool:
        try:
            info = await asyncio.to_thread(_http_json, f"{self.base_url}/api/tags", None, 4.0)
        except Exception as exc:  # noqa: BLE001
            self.bus.publish("warn", "brain", f"ollama not reachable at {self.base_url} ({type(exc).__name__})")
            return False

        models = sorted(m.get("name", "") for m in info.get("models", []) if m.get("name"))
        if self.registry is not None:
            await self.registry.refresh_models()
            self.registry.autoselect_model()
        self.bus.publish(
            "success",
            "brain",
            f"ollama online at {self.base_url} - {len(models)} model(s) installed",
        )
        current = self.model
        if current and models and current not in models:
            self.bus.publish("warn", "brain", f"model '{current}' not installed - run: ollama pull {current}")
        self.bus.publish("info", "brain", f"crew mode '{self.mode}' on {current or 'no model'}")
        if self.kit is not None:
            if self.tools_enabled:
                self.bus.publish(
                    "success", "brain",
                    f"tool calling armed - {len(self.kit.skills)} skill(s) available to {current}",
                )
            else:
                self.bus.publish(
                    "info", "brain",
                    f"'{current}' does not advertise tool calling - skills stay manual",
                )
        return True

    # -- Runtime API -------------------------------------------------------------

    async def run(self, text: str) -> str:
        if not self.model:
            return "No language model is installed. Pull one with ollama pull, then ask me again."
        mode = self.mode
        if mode == "crewai":
            try:
                answer = await self._run_crewai(text)
                self._remember("user", text)
                self._remember("assistant", answer)
                return answer
            except ImportError:
                self.bus.publish("warn", "agent/router", "crewai not installed - using direct crew")
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("error", "agent/router", f"crewai failed ({type(exc).__name__}) - falling back")
            mode = "crew"

        answer = await (self._run_crew(text) if mode == "crew" else self._run_fast(text))
        self._remember("user", text)
        self._remember("assistant", answer)
        return answer

    # -- fast conversational path -------------------------------------------------

    def _system_prompt(self) -> str:
        """Persona prompt, stamped with live context the model cannot infer."""
        persona = resolve_persona(getattr(self.registry, "persona", None))
        prompt = JARVIS_SYSTEM.format(
            name=settings.name,
            operator=settings.operator,
            now=datetime.now().strftime("%A, %d %B %Y, %H:%M"),
            host=f"{platform.system()} {platform.machine()}",
            style=persona.style,
        )
        if self.tools_enabled:
            prompt += _TOOL_HINT
        return prompt

    def _temperature(self) -> float:
        return resolve_persona(getattr(self.registry, "persona", None)).temperature

    async def _run_fast(self, text: str) -> str:
        self._fire("mem.working", 0.7)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]

        # Durable recall: anything relevant the user said in a past session.
        if self.context_provider is not None:
            context = await asyncio.to_thread(self.context_provider, text)
            if context:
                self._fire("mem.recall", 1.0)
                messages.append({"role": "system", "content": context})
                self.bus.publish("info", "memory", f"recalled {context.count(chr(10)) + 1} prior fragment(s)")

        # Reflex arc: anything this machine can compute exactly is computed
        # now, so the answer cannot depend on the model choosing to call a tool.
        reflexes = detect_reflexes(
            text,
            vitals=(self.vitals_provider() if self.vitals_provider else None),
            now_provider=(self.kit.get_datetime if self.kit is not None else None),
        )
        if reflexes:
            self._fire("intake.reflex", 1.0)
            for reflex in reflexes:
                self._fire(reflex.node, 0.8)
                self.bus.publish("success", "reflex", f"{reflex.label}: {reflex.detail}")
            messages.append({"role": "system", "content": reflex_prompt(reflexes)})

        messages.extend(self.memory)
        messages.append({"role": "user", "content": text})

        tools = self.kit.schemas() if self.tools_enabled else None
        self.bus.publish(
            "info",
            "agent/jarvis",
            f"reasoning with {self.model}…"
            + (" [thinking]" if self.think else "")
            + (f" [{len(tools)} tools]" if tools else ""),
        )

        answer = ""
        rounds = _max_tool_rounds()
        for round_index in range(rounds):
            payload: dict[str, Any] = {
                "model": self.model,
                "stream": True,
                "messages": messages,
                "options": {"temperature": self._temperature(), "top_p": 0.9},
            }
            if tools:
                payload["tools"] = tools
            self._apply_think(payload)

            self._fire("cortex.jarvis", 1.0)
            turn = await asyncio.to_thread(self._consume, "jarvis", payload, True, self.on_delta)

            if not turn.tool_calls:
                answer = strip_think(turn.answer)
                break

            # A tool round: whatever prose leaked out is scaffolding, not the
            # answer, so tell the interface to clear the caption before we go on.
            if turn.answer.strip() and self.on_reset is not None:
                self.on_reset()

            messages.append({
                "role": "assistant",
                "content": turn.answer,
                "tool_calls": turn.tool_calls,
            })
            await self._execute_tools(turn.tool_calls, messages)

            if round_index == rounds - 1:
                self.bus.publish("warn", "agent/tools", "tool round limit reached - composing now")
                messages.append({
                    "role": "system",
                    "content": "Tool budget exhausted. Answer now with what you have.",
                })
                tools = None
        else:  # pragma: no cover - loop always breaks or exhausts above
            answer = strip_think(answer)

        return answer or "I am afraid I could not compose a response."

    async def _execute_tools(self, calls: list[dict], messages: list[dict[str, Any]]) -> None:
        """Run every tool the model asked for and append the results."""
        for call in calls:
            function = (call or {}).get("function") or {}
            name = str(function.get("name") or "").strip()
            raw_args = function.get("arguments")
            arguments = raw_args if isinstance(raw_args, dict) else raw_args
            if not name:
                continue

            node = None
            if self.neural is not None:
                node = self.neural.ensure_tool_node(name)
                self.neural.signal("cortex.jarvis", "motor.tools", 1.0)
                self.neural.signal("motor.tools", node, 1.0)

            pretty = json.dumps(arguments, default=str)[:160] if arguments else "{}"
            self.bus.publish("info", "agent/tools", f"▶ {name}({pretty})")

            result = await asyncio.to_thread(self.kit.invoke, name, arguments)

            if node is not None and self.neural is not None:
                self.neural.signal(node, "cortex.jarvis", 1.0)
            if result.get("ok"):
                summary = json.dumps(result.get("result"), default=str)
                self.bus.publish(
                    "success", "agent/tools",
                    f"◀ {name} → {summary[:180]}{'…' if len(summary) > 180 else ''} "
                    f"({result.get('elapsed_ms', 0)}ms)",
                )
            else:
                self.bus.publish("warn", "agent/tools", f"◀ {name} failed: {result.get('error')}")

            if self.on_tool is not None:
                self.on_tool(name, arguments if isinstance(arguments, dict) else {}, result)

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, default=str)[:6000],
            })

    # -- full crew path ------------------------------------------------------------

    async def _run_crew(self, text: str) -> str:
        self._fire("intake.route", 1.0)
        plan = await self._persona_step(AgentPersonas.ROUTER, text)
        findings: list[str] = []

        self._fire("cortex.analyst", 1.0)
        analyst_out = await self._persona_step(
            AgentPersonas.ANALYST,
            f"Command: {text}\nRouter plan:\n{plan}\nProduce key findings (max 5 bullets, plain text).",
        )
        findings.append(analyst_out)

        self._fire("cortex.engineer", 1.0)
        engineer_out = await self._persona_step(
            AgentPersonas.ENGINEER,
            f"Command: {text}\nFindings so far:\n{analyst_out}\n"
            "Describe exactly which tools/commands you would execute (do not actually run them). "
            "If no execution is needed, say 'no tooling required'.",
        )
        findings.append(engineer_out)

        self.bus.publish("info", "agent/synth", "composing final spoken answer")
        self._fire("cortex.synth", 1.0)
        answer = await self._persona_step(
            AgentPersonas.SYNTH,
            _SYNTH_TMPL.format(command=text, findings="\n---\n".join(findings)),
            log_sentences=False,
            on_delta=self.on_delta,
        )
        return strip_think(answer) or "I am afraid I could not compose a response."

    async def _persona_step(self, persona: Persona, prompt: str, log_sentences: bool = True,
                            on_delta: Callable[[str], None] | None = None) -> str:
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": _SYSTEM_TMPL.format(
                    role=persona.role, goal=persona.goal, backstory=persona.backstory
                )},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.4},
        }
        self._apply_think(payload)
        self.bus.publish("info", f"agent/{persona.key}", f"thinking ({self.model})…")
        started = time.monotonic()
        turn = await asyncio.to_thread(self._consume, persona.key, payload, log_sentences, on_delta)
        self.bus.publish("info", f"agent/{persona.key}", f"step complete in {time.monotonic() - started:.1f}s")
        return strip_think(turn.answer)

    def _consume(self, key: str, payload: dict, log_sentences: bool,
                 on_delta: Callable[[str], None] | None = None) -> Turn:
        """Blocking stream consumer. Runs in a worker thread."""
        filt = ThinkFilter()
        turn = Turn()
        answer_parts: list[str] = []
        sentence_buf = ""
        thought_buf = ""
        thought_logged = False
        since_spike = 0

        def flush_sentences(final: bool = False) -> None:
            nonlocal sentence_buf
            while True:
                match = re.search(r"[.!?;:](?:\s|$)", sentence_buf)
                if not match:
                    break
                sentence = sentence_buf[: match.end()].strip()
                sentence_buf = sentence_buf[match.end():]
                if sentence and log_sentences:
                    self.bus.publish("info", f"agent/{key}", sentence)
            if final and sentence_buf.strip() and log_sentences:
                self.bus.publish("info", f"agent/{key}", sentence_buf.strip())
                sentence_buf = ""

        def flush_thoughts(final: bool = False) -> None:
            nonlocal thought_buf, thought_logged
            while True:
                match = re.search(r"[.!?](?:\s|$)", thought_buf)
                if not match:
                    break
                line = thought_buf[: match.end()].strip()
                thought_buf = thought_buf[match.end():]
                if line:
                    thought_logged = True
                    self._fire("cortex.think", 0.6)
                    self.bus.publish("info", "agent/thought", line[:200])
            if final:
                if thought_buf.strip():
                    self.bus.publish("info", "agent/thought", thought_buf.strip()[:200])
                thought_buf = ""

        def take(event: StreamEvent) -> None:
            nonlocal sentence_buf, thought_buf, since_spike
            if event.kind == "tools":
                turn.tool_calls.extend(event.tool_calls)
                return
            if event.kind == "think":
                thought_buf += event.text
                flush_thoughts()
                return
            answer_delta, thought_delta = filt.feed(event.text)
            if thought_delta:
                thought_buf += thought_delta
                flush_thoughts()
            if answer_delta:
                answer_parts.append(answer_delta)
                turn.chars += len(answer_delta)
                if on_delta is not None:
                    on_delta(answer_delta)
                # Firing per token would be thousands of dict writes a second
                # for no visual gain; the graph coalesces at 20 Hz anyway.
                since_spike += len(answer_delta)
                if since_spike > 24:
                    since_spike = 0
                    self._fire(f"cortex.{key}" if key != "jarvis" else "cortex.jarvis", 0.35)
                sentence_buf += answer_delta
                flush_sentences()

        try:
            try:
                stream = _stream_chat(f"{self.base_url}/api/chat", payload)
                first = next(stream, None)
            except urllib.error.HTTPError as exc:
                # Some builds reject `think` (or `tools`) even when tags advertised them.
                retryable = exc.code == 400 and ("think" in payload or "tools" in payload)
                if retryable:
                    dropped = [k for k in ("think", "tools") if payload.pop(k, None) is not None]
                    self.bus.publish(
                        "warn", f"agent/{key}",
                        f"server rejected {'/'.join(dropped)} - retrying without",
                    )
                    stream = _stream_chat(f"{self.base_url}/api/chat", payload)
                    first = next(stream, None)
                else:
                    raise
            # `[first, *stream]` would materialise the whole generator before
            # the loop body ran even once, so every token arrived in one burst
            # at the end and nothing actually streamed. Chain it lazily.
            for event in (() if first is None else itertools.chain((first,), stream)):
                take(event)
            tail_answer, tail_thought = filt.flush()
            if tail_thought:
                thought_buf += tail_thought
            if tail_answer:
                answer_parts.append(tail_answer)
                turn.chars += len(tail_answer)
                if on_delta is not None:
                    on_delta(tail_answer)
                sentence_buf += tail_answer
            flush_thoughts(final=True)
            flush_sentences(final=True)
            if thought_logged:
                self.bus.publish("info", "agent/thought", "— reasoning complete, answer follows —")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200] if hasattr(exc, "read") else str(exc)
            self.bus.publish("error", f"agent/{key}", f"ollama HTTP {exc.code}: {detail}")
            raise
        except Exception as exc:  # noqa: BLE001
            self.bus.publish("error", f"agent/{key}", f"model call failed: {type(exc).__name__}: {exc}")
            raise
        turn.answer = "".join(answer_parts)
        return turn

    # -- real CrewAI path ---------------------------------------------------------------

    async def _run_crewai(self, text: str) -> str:
        from crewai import Agent, Crew, LLM, Task  # noqa: PLC0415 - optional dependency

        bus = self.bus
        llm = LLM(model=f"ollama/{self.model}", base_url=self.base_url)

        def make_agent(p: Persona) -> Any:
            return Agent(role=p.role, goal=p.goal, backstory=p.backstory, llm=llm, verbose=False)

        router, analyst, engineer, synth = (make_agent(p) for p in AgentPersonas.ALL)

        def step_log(payload: Any) -> None:
            msg = str(payload).strip()
            if msg:
                bus.publish("info", "crew", msg[:220])

        tasks = [
            Task(description=f"Decompose this command into a short numbered plan:\n{text}",
                 expected_output="Numbered plan, max 5 steps.", agent=router),
            Task(description="Given the plan, list the key facts or data needed.",
                 expected_output="Up to 5 plain-text findings.", agent=analyst),
            Task(description="State which tools or commands the plan requires (do not execute).",
                 expected_output="Short list of tool calls or 'no tooling required'.", agent=engineer),
            Task(description=_SYNTH_TMPL.format(command=text, findings="<from prior tasks>"),
                 expected_output="Final spoken answer, plain prose under 90 words.", agent=synth),
        ]

        crew_kwargs: dict[str, Any] = dict(agents=[router, analyst, engineer, synth], tasks=tasks, verbose=False)
        try:
            crew = Crew(**crew_kwargs, step_callback=step_log)
        except TypeError:  # older/newer crewai without step_callback
            crew = Crew(**crew_kwargs)

        bus.publish("info", "crew", f"crew assembled - 4 agents on {self.model}")
        result = await asyncio.to_thread(crew.kickoff, text)
        answer = strip_think(str(getattr(result, "raw", result) or ""))
        bus.publish("success", "crew", "crew run complete")
        return answer
