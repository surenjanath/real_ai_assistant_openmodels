"""Live agent runtime backed by a local Ollama server.

Three strategies, selected by `JARVIS_CREW_MODE` / `JARVIS_USE_CREWAI`:

1. **fast** (default) - a single streamed call in the J.A.R.V.I.S. persona with
   short-term conversational memory. One model round-trip, so replies start
   speaking in about a second. This is what you want for actual conversation.
2. **crew** - the four-persona pipeline (Router -> Analyst -> Engineer ->
   Synthesiser), each step streamed to the telemetry panel. Slower but shows
   real multi-agent reasoning.
3. **crewai** - the same crew built with the real `crewai` package.

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
import json
import platform
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator

from ..config import settings
from ..logbus import LogBus
from .base import AgentPersonas, Persona

if TYPE_CHECKING:
    from ..registry import Registry

JARVIS_SYSTEM = (
    "You are J.A.R.V.I.S., a locally hosted AI assistant, speaking directly to {operator}. "
    "The current date and time is {now}, and you are running on {host}. Use these when asked - "
    "you genuinely do know them. "
    "Your replies are converted to speech, so write plain spoken prose: no markdown, no bullet "
    "points, no code blocks, no emoji, no stage directions. Be composed, precise and dryly witty, "
    "in the manner of a British butler-engineer. Answer in two or three sentences unless asked for "
    "detail. Never mention that you are a language model. If you do not know something, say so "
    "plainly rather than inventing it."
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


def _stream_chat(url: str, payload: dict) -> Iterator[str]:
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
                yield f"\x00think\x00{thinking}"
            delta = message.get("content") or event.get("response") or ""
            if delta:
                yield delta
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
        if delta.startswith("\x00think\x00"):
            return "", delta[7:]
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


class OllamaRuntime:
    name = "ollama"

    def __init__(self, bus: LogBus, base_url: str | None = None, model: str | None = None,
                 registry: "Registry | None" = None) -> None:
        self.bus = bus
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._static_model = model
        self.registry = registry
        #: chat history as [{"role": "user"|"assistant", "content": str}]
        self.memory: list[dict[str, str]] = []
        #: optional sink for streamed answer deltas (wired by the orchestrator)
        self.on_delta: Callable[[str], None] | None = None

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

    # -- probe -----------------------------------------------------------------

    @classmethod
    async def probe(cls, bus: LogBus, base_url: str | None = None, model: str | None = None,
                    registry: "Registry | None" = None) -> "OllamaRuntime | None":
        runtime = cls(bus, base_url, model, registry)
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
        return JARVIS_SYSTEM.format(
            operator=settings.operator,
            now=datetime.now().strftime("%A, %d %B %Y, %H:%M"),
            host=f"{platform.system()} {platform.machine()}",
        )

    async def _run_fast(self, text: str) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.memory)
        messages.append({"role": "user", "content": text})
        payload = {
            "model": self.model,
            "stream": True,
            "messages": messages,
            "options": {"temperature": 0.6, "top_p": 0.9},
        }
        self._apply_think(payload)
        self.bus.publish(
            "info",
            "agent/jarvis",
            f"reasoning with {self.model}…" + (" [thinking]" if self.think else ""),
        )
        answer = await asyncio.to_thread(self._consume, "jarvis", payload, True, self.on_delta)
        answer = strip_think(answer)
        return answer or "I am afraid I could not compose a response."

    # -- full crew path ------------------------------------------------------------

    async def _run_crew(self, text: str) -> str:
        plan = await self._persona_step(AgentPersonas.ROUTER, text)
        findings: list[str] = []

        analyst_out = await self._persona_step(
            AgentPersonas.ANALYST,
            f"Command: {text}\nRouter plan:\n{plan}\nProduce key findings (max 5 bullets, plain text).",
        )
        findings.append(analyst_out)

        engineer_out = await self._persona_step(
            AgentPersonas.ENGINEER,
            f"Command: {text}\nFindings so far:\n{analyst_out}\n"
            "Describe exactly which tools/commands you would execute (do not actually run them). "
            "If no execution is needed, say 'no tooling required'.",
        )
        findings.append(engineer_out)

        self.bus.publish("info", "agent/synth", "composing final spoken answer")
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
        out = await asyncio.to_thread(self._consume, persona.key, payload, log_sentences, on_delta)
        self.bus.publish("info", f"agent/{persona.key}", f"step complete in {time.monotonic() - started:.1f}s")
        return strip_think(out)

    def _consume(self, key: str, payload: dict, log_sentences: bool,
                 on_delta: Callable[[str], None] | None = None) -> str:
        """Blocking stream consumer. Runs in a worker thread."""
        filt = ThinkFilter()
        answer_parts: list[str] = []
        sentence_buf = ""
        thought_buf = ""
        thought_logged = False

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
                    self.bus.publish("info", "agent/thought", line[:200])
            if final:
                if thought_buf.strip():
                    self.bus.publish("info", "agent/thought", thought_buf.strip()[:200])
                thought_buf = ""

        try:
            try:
                stream = _stream_chat(f"{self.base_url}/api/chat", payload)
                first = next(stream, None)
            except urllib.error.HTTPError as exc:
                # Some builds reject `think` even when tags advertised it.
                if exc.code == 400 and "think" in payload:
                    payload.pop("think", None)
                    self.bus.publish("warn", f"agent/{key}", "server rejected think flag - retrying without it")
                    stream = _stream_chat(f"{self.base_url}/api/chat", payload)
                    first = next(stream, None)
                else:
                    raise
            for delta in ([] if first is None else [first, *stream]):
                answer_delta, thought_delta = filt.feed(delta)
                if thought_delta:
                    thought_buf += thought_delta
                    flush_thoughts()
                if answer_delta:
                    answer_parts.append(answer_delta)
                    if on_delta is not None:
                        on_delta(answer_delta)
                    sentence_buf += answer_delta
                    flush_sentences()
            tail_answer, tail_thought = filt.flush()
            if tail_thought:
                thought_buf += tail_thought
            if tail_answer:
                answer_parts.append(tail_answer)
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
        return "".join(answer_parts)

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
