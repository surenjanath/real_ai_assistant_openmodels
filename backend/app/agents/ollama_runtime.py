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
import contextlib
import itertools
import json
import platform
import re
import threading
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
#:
#: `today` is deliberately day-resolution, not the minute. Ollama reuses the
#: KV cache only for the longest *common prefix*, and this is message zero - so
#: a clock that ticks in the system prompt invalidates everything behind it,
#: including ~3,000 tokens of tool schema, on every turn that crosses a minute
#: boundary. Measured at roughly 0.8s of extra prompt evaluation per turn, paid
#: for a figure that is stale a minute later anyway. The exact time still
#: reaches the model whenever it matters, from the reflex arc and the
#: `get_datetime` skill, both of which are accurate to the second.
JARVIS_SYSTEM = (
    "You are {name}, a locally hosted AI assistant, speaking directly to {operator}. "
    "Today is {today}, and you are running on {host}. Use these when asked - "
    "you genuinely do know them. For the time of day, consult your tools rather "
    "than assuming. "
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
    "question depends on: the current date or time; how many days lie between two dates, or "
    "what date falls a given distance from another; live machine state; arithmetic on numbers "
    "longer than two digits; a cryptographic digest; a genuinely random choice; the contents "
    "of the operator's files; or anything said in an earlier conversation. Guessing at any of "
    "these is a failure - each one is something you cannot actually compute, however confident "
    "the answer feels. Call the tool first, then answer in plain spoken prose using the "
    "result, copying any value it returns exactly as given."
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


def _options(temperature: float, **extra: Any) -> dict[str, Any]:
    """Sampling + runtime options sent with every chat request.

    `num_ctx` is pinned rather than left to the server default so Ollama's
    prompt cache actually hits between turns - a changing context size forces
    a full re-evaluation of the prompt on every request. `num_predict` is a
    seatbelt: a model that decides to write an essay should not hang the
    assistant for a minute with nothing spoken.
    """
    options: dict[str, Any] = {"temperature": temperature, "top_p": 0.9}
    if settings.num_ctx > 0:
        options["num_ctx"] = settings.num_ctx
    if settings.num_predict > 0:
        options["num_predict"] = settings.num_predict
    options.update(extra)
    return options


#: Transport failures worth retrying: the server was momentarily busy loading
#: a model, or the socket was reaped. A protocol error (HTTPError) is not here
#: on purpose - retrying a 400 just fails again.
_TRANSIENT = (urllib.error.URLError, ConnectionError, TimeoutError, OSError)


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

    kind: str  # "delta" | "think" | "tools" | "done"
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    #: only on the final "done" frame
    prompt_tokens: int = 0
    eval_tokens: int = 0
    done_reason: str = ""


def _stream_chat(url: str, payload: dict) -> Iterator[StreamEvent]:
    payload = {"keep_alive": settings.ollama_keep_alive, **payload}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
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
                # The final frame carries the accounting. It is the only place
                # two silent failures are visible: an answer cut off because it
                # hit the token ceiling, and a prompt so large that Ollama
                # quietly dropped the oldest messages to make it fit.
                yield StreamEvent(
                    "done",
                    prompt_tokens=int(event.get("prompt_eval_count") or 0),
                    eval_tokens=int(event.get("eval_count") or 0),
                    done_reason=str(event.get("done_reason") or ""),
                )
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
    #: the operator interrupted; whatever is above is a half-finished thought
    aborted: bool = False
    #: accounting from the final stream frame, for the truncation warnings
    prompt_tokens: int = 0
    eval_tokens: int = 0
    done_reason: str = ""


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
        #: called as (prompt_tokens, eval_tokens) once Ollama reports them
        self.on_usage: Callable[[int, int], None] | None = None
        #: extra grounding injected ahead of the user turn (durable recall)
        self.context_provider: Callable[[str], str] | None = None
        #: latest host metrics, so reflexes can ground live-state questions
        self.vitals_provider: Callable[[], dict] | None = None
        #: raised to cut a generation short. Set from the event loop, read by
        #: the worker thread between tokens - which is why it is an Event and
        #: not a plain bool.
        self._abort = threading.Event()
        #: monotonic stamp of the last directive, for the idle context reset
        self._last_turn_at: float = 0.0
        #: cached token cost of the tool schemas - they do not change between
        #: model switches, and re-serialising 29 of them per turn to measure
        #: something constant would be its own small waste
        self._tool_tokens: int | None = None

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

    def shed_style(self) -> int:
        """Drop our own past replies from the working context.

        Called when the disposition changes. The persona lives in the system
        prompt, but the conversation behind it still holds several turns spoken
        in the *old* voice, and a model imitates its own recent output far more
        readily than it follows an instruction: switching back to the butler
        after a few furious turns produced a butler who told the operator to
        pull themselves together.

        The operator's turns stay, so the thread of the conversation survives -
        it is only the assistant's own delivery that is forgotten. Same
        principle as durable recall: our past output is not evidence, and
        re-reading it is how a style, or a mistake, becomes permanent.
        """
        before = len(self.memory)
        self.memory[:] = [m for m in self.memory if m.get("role") != "assistant"]
        return before - len(self.memory)

    def abort(self) -> None:
        """Stop generating: the operator is talking again.

        Ollama has no cancel, so this is cooperative - the consumer thread
        notices between tokens and closes the response, which is what actually
        frees the model. Without it a barge-in silences the voice but leaves
        the whole answer being written anyway, and the directive the operator
        interrupted with waits out a reply nobody will ever hear.
        """
        self._abort.set()

    @staticmethod
    def _tokens(text: str) -> int:
        """Rough token count. Four characters per token, as used for tok/s."""
        return max(1, len(text) // 4)

    def _tool_token_cost(self) -> int:
        """What the tool schemas cost in the prompt, every single request."""
        if not self.tools_enabled or self.kit is None:
            return 0
        if self._tool_tokens is None:
            self._tool_tokens = self._tokens(json.dumps(self.kit.schemas()))
        return self._tool_tokens

    def _context_budget(self) -> int:
        """How many tokens of conversation still fit, after everything fixed.

        Trimming by *turn count* was the wrong unit: eight turns of "yes" and
        eight turns of a three-hundred-word explanation are the same number and
        wildly different prompts. What actually costs time is tokens - the
        model re-reads the whole prompt before emitting anything, so time to
        first token grows with the context whether or not it is being used.
        """
        if settings.num_ctx <= 0:
            return 0  # server default in force; not ours to second-guess
        room = settings.num_ctx - self._tool_token_cost() - settings.context_reserve_tokens
        return max(512, room)

    def _remember(self, role: str, content: str) -> None:
        if not content:
            return
        self.memory.append({"role": role, "content": content})
        limit = max(2, settings.memory_turns * 2)
        if len(self.memory) > limit:
            del self.memory[: len(self.memory) - limit]

        # ...and then again by size, because that is what is actually scarce.
        budget = self._context_budget()
        if budget <= 0:
            return
        used = sum(self._tokens(m.get("content", "")) for m in self.memory)
        dropped = 0
        while used > budget and len(self.memory) > 2:
            used -= self._tokens(self.memory.pop(0).get("content", ""))
            dropped += 1
        if dropped:
            self.bus.publish(
                "info", "memory",
                f"context trimmed - dropped {dropped} old turn(s) to stay within "
                f"{budget} tokens (every turn re-reads the whole prompt)",
            )

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

    # -- residency ---------------------------------------------------------------

    async def preload(self, model: str | None = None) -> bool:
        """Load the model into memory now, so the first directive does not.

        An 8B model costs two to ten seconds to page in. Paying that at boot
        (and again the moment the operator switches model) is the difference
        between an assistant that answers in a second and one that appears to
        have frozen the first time you talk to it.
        """
        target = model or self.model
        if not target:
            return False

        def load() -> Any:
            return _http_json(
                f"{self.base_url}/api/chat",
                {"model": target, "messages": [], "stream": False,
                 "keep_alive": settings.ollama_keep_alive},
                timeout=180.0,
            )

        started = time.monotonic()
        try:
            await asyncio.to_thread(load)
        except Exception as exc:  # noqa: BLE001 - a cold model is not fatal
            self.bus.publish("warn", "brain", f"could not preload {target}: {type(exc).__name__}")
            return False
        self.bus.publish(
            "success", "brain",
            f"{target} resident in {time.monotonic() - started:.1f}s "
            f"(kept warm for {settings.ollama_keep_alive})",
        )
        return True

    # -- Runtime API -------------------------------------------------------------

    async def run(self, text: str) -> str:
        # Ageing out a stale context comes first: it is true regardless of
        # whether this particular directive can be answered.
        # A conversation nobody has touched for a quarter of an hour is over.
        # Carrying it forward costs prompt-evaluation time on every turn of the
        # next one, for context the operator has long since stopped meaning.
        now = time.monotonic()
        idle = settings.context_idle_reset_s
        if self.memory and idle > 0 and self._last_turn_at and now - self._last_turn_at > idle:
            gap = int((now - self._last_turn_at) / 60)
            self.bus.publish(
                "info", "memory",
                f"starting fresh - {gap} minute(s) since the last directive "
                f"({len(self.memory)} turn(s) released)",
            )
            self.memory.clear()
        self._last_turn_at = now

        if not self.model:
            return "No language model is installed. Pull one with ollama pull, then ask me again."
        # Any interrupt that arrived before this directive existed was aimed at
        # the previous one.
        self._abort.clear()
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
        if self._abort.is_set():
            # A half-written answer to a question the operator moved on from is
            # worse than no answer: remembering it would have the model carry
            # the abandoned thread into everything that follows.
            self.bus.publish("warn", "agent/jarvis", "generation interrupted by the operator")
            return ""
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
            today=datetime.now().strftime("%A, %d %B %Y"),
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

        # Only the tools this directive could plausibly need. Describing all of
        # them costs ~1.3s of time-to-first-token every single time, used or
        # not, because the model reads the whole prompt before answering.
        tools = self.kit.schemas(text) if self.tools_enabled else None
        total = len(self.kit.skills) if self.kit is not None else 0
        self.bus.publish(
            "info",
            "agent/jarvis",
            f"reasoning with {self.model}…"
            + (" [thinking]" if self.think else "")
            + (f" [{len(tools)} of {total} tools]" if tools else ""),
        )

        answer = ""
        rounds = _max_tool_rounds()
        for round_index in range(rounds):
            payload: dict[str, Any] = {
                "model": self.model,
                "stream": True,
                "messages": messages,
                "options": _options(self._temperature()),
            }
            if tools:
                payload["tools"] = tools
            self._apply_think(payload)

            self._fire("cortex.jarvis", 1.0)
            turn = await asyncio.to_thread(self._consume, "jarvis", payload, True, self.on_delta)

            if turn.aborted:
                return ""
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
            if self._abort.is_set():
                return ""

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
        if self._abort.is_set():
            return ""
        findings: list[str] = []

        self._fire("cortex.analyst", 1.0)
        analyst_out = await self._persona_step(
            AgentPersonas.ANALYST,
            f"Command: {text}\nRouter plan:\n{plan}\nProduce key findings (max 5 bullets, plain text).",
        )
        findings.append(analyst_out)
        if self._abort.is_set():
            return ""

        self._fire("cortex.engineer", 1.0)
        engineer_out = await self._persona_step(
            AgentPersonas.ENGINEER,
            f"Command: {text}\nFindings so far:\n{analyst_out}\n"
            "Describe exactly which tools/commands you would execute (do not actually run them). "
            "If no execution is needed, say 'no tooling required'.",
        )
        findings.append(engineer_out)
        if self._abort.is_set():
            return ""

        self.bus.publish("info", "agent/synth", "composing final spoken answer")
        self._fire("cortex.synth", 1.0)
        answer = await self._persona_step(
            AgentPersonas.SYNTH,
            _SYNTH_TMPL.format(command=text, findings="\n---\n".join(findings)),
            log_sentences=False,
            on_delta=self.on_delta,
        )
        if self._abort.is_set():
            return ""
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
            "options": _options(0.4),
        }
        self._apply_think(payload)
        self.bus.publish("info", f"agent/{persona.key}", f"thinking ({self.model})…")
        started = time.monotonic()
        turn = await asyncio.to_thread(self._consume, persona.key, payload, log_sentences, on_delta)
        if turn.aborted:
            return ""
        self.bus.publish("info", f"agent/{persona.key}", f"step complete in {time.monotonic() - started:.1f}s")
        return strip_think(turn.answer)

    def _open_stream(self, key: str, payload: dict) -> tuple[Iterator[StreamEvent], StreamEvent | None]:
        """Start the chat stream, negotiating away rejected flags and retrying
        transient transport failures.

        Retrying is only safe here, before a single token has been handed to
        the caller - once the answer has started streaming, a reconnect would
        duplicate text into the caption and the voice.
        """
        url = f"{self.base_url}/api/chat"
        attempts = max(1, settings.ollama_retries + 1)
        for attempt in range(attempts):
            try:
                stream = _stream_chat(url, payload)
                return stream, next(stream, None)
            except urllib.error.HTTPError as exc:
                # Some builds reject `think` (or `tools`) even when tags advertised them.
                if exc.code == 400 and ("think" in payload or "tools" in payload):
                    dropped = [k for k in ("think", "tools") if payload.pop(k, None) is not None]
                    self.bus.publish(
                        "warn", f"agent/{key}",
                        f"server rejected {'/'.join(dropped)} - retrying without",
                    )
                    continue
                raise
            except _TRANSIENT as exc:
                if attempt >= attempts - 1:
                    raise
                delay = 0.4 * (attempt + 1)
                self.bus.publish(
                    "warn", f"agent/{key}",
                    f"ollama transport error ({type(exc).__name__}) - retry {attempt + 1}/{attempts - 1} in {delay:.1f}s",
                )
                time.sleep(delay)
        raise RuntimeError("ollama stream could not be opened")

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
            if event.kind == "done":
                turn.prompt_tokens = event.prompt_tokens
                turn.eval_tokens = event.eval_tokens
                turn.done_reason = event.done_reason
                return
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
            stream, first = self._open_stream(key, payload)
            # `[first, *stream]` would materialise the whole generator before
            # the loop body ran even once, so every token arrived in one burst
            # at the end and nothing actually streamed. Chain it lazily.
            for event in (() if first is None else itertools.chain((first,), stream)):
                if self._abort.is_set():
                    turn.aborted = True
                    # Close the generator rather than merely dropping it: that
                    # is what unwinds `_stream_chat`'s `with` block and hangs
                    # up on Ollama, instead of leaving the model generating
                    # into a socket nobody reads until it finishes.
                    with contextlib.suppress(Exception):
                        stream.close()
                    break
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
        self._warn_if_squeezed(key, turn)
        if self.on_usage is not None and turn.prompt_tokens:
            with contextlib.suppress(Exception):
                self.on_usage(turn.prompt_tokens, turn.eval_tokens)
        return turn

    def _warn_if_squeezed(self, key: str, turn: Turn) -> None:
        """Say so when the answer was cut, or the prompt nearly filled the window.

        Both failures are otherwise invisible: a truncated answer simply reads
        as a short one, and an over-long prompt is silently trimmed by Ollama
        from the oldest end - so the model loses the conversation rather than
        reporting that it did. Tool schemas are the usual cause; 33 of them
        cost roughly 3,200 tokens before anything is said.
        """
        if turn.done_reason == "length":
            self.bus.publish(
                "warn", f"agent/{key}",
                f"answer cut off at the {settings.num_predict}-token ceiling - "
                "raise JARVIS_NUM_PREDICT if this recurs",
            )
        window = settings.num_ctx
        if window and turn.prompt_tokens >= window * 0.8:
            self.bus.publish(
                "warn", f"agent/{key}",
                f"prompt used {turn.prompt_tokens} of {window} context tokens - "
                "older turns are being dropped; raise JARVIS_NUM_CTX or disable tools",
            )

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
