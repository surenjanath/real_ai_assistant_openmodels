"""Live multi-agent runtime backed by a local Ollama server.

Strategy
--------
1. If `crewai` is installed and JARVIS_USE_CREWAI=1, build a real Crew
   (Router / Analyst / Engineer / Synthesizer) with an Ollama LLM and stream
   its step callbacks to the log bus. This is the full PRD §4 stack.
2. Otherwise drive Ollama's /api/chat directly, one persona at a time, in a
   lightweight "poor man's crew" that keeps dependencies at zero. Token
   deltas are buffered to sentence boundaries and published as reasoning logs
   so the terminal shows genuine model thinking.

Only stdlib HTTP is used (urllib in a worker thread), because the whole point
of this project is to run lean on a laptop.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from typing import Any, Iterator

from ..config import settings
from ..logbus import LogBus
from .base import AgentPersonas, Persona, Runtime

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
            delta = (event.get("message") or {}).get("content") or event.get("response") or ""
            if delta:
                yield delta
            if event.get("done"):
                return


class OllamaRuntime:
    name = "ollama"

    def __init__(self, bus: LogBus, base_url: str | None = None, model: str | None = None) -> None:
        self.bus = bus
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    # -- probe -----------------------------------------------------------------

    @classmethod
    async def probe(cls, bus: LogBus, base_url: str | None = None, model: str | None = None) -> "OllamaRuntime | None":
        runtime = cls(bus, base_url, model)
        ok = await runtime._probe()
        return runtime if ok else None

    async def _probe(self) -> bool:
        try:
            info = await asyncio.to_thread(
                _http_json, f"{self.base_url}/api/tags", None, 3.0
            )
            models = [m.get("name", "") for m in info.get("models", [])]
            self.bus.publish(
                "success",
                "brain",
                f"ollama online at {self.base_url} - {len(models)} model(s): {', '.join(models[:6]) or 'none'}",
            )
            if models and not any(m.startswith(self.model.split(':')[0]) for m in models):
                self.bus.publish("warn", "brain", f"model '{self.model}' not pulled yet - run: ollama pull {self.model}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.bus.publish("warn", "brain", f"ollama not reachable at {self.base_url} ({type(exc).__name__})")
            return False

    # -- Runtime API -------------------------------------------------------------

    async def run(self, text: str) -> str:
        if settings.use_crewai:
            try:
                return await self._run_crewai(text)
            except ImportError:
                self.bus.publish("warn", "agent/router", "crewai not installed - using direct ollama crew")
            except Exception as exc:  # noqa: BLE001
                self.bus.publish("error", "agent/router", f"crewai run failed ({type(exc).__name__}: {exc}) - falling back")
        return await self._run_direct(text)

    # -- direct ollama crew ---------------------------------------------------------

    async def _run_direct(self, text: str) -> str:
        bus = self.bus
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

        bus.publish("info", "agent/synth", "composing final spoken answer")
        answer = await self._persona_step(
            AgentPersonas.SYNTH,
            _SYNTH_TMPL.format(command=text, findings="\n---\n".join(findings)),
            log_sentences=False,
        )
        return answer.strip() or "I am afraid I could not compose a response."

    async def _persona_step(self, persona: Persona, prompt: str, log_sentences: bool = True) -> str:
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
        self.bus.publish("info", f"agent/{persona.key}", f"thinking ({self.model})…")
        return await asyncio.to_thread(self._consume_stream, persona.key, payload, log_sentences)

    def _consume_stream(self, key: str, payload: dict, log_sentences: bool) -> str:
        buffer = ""
        full = []

        def flush(final: bool = False) -> None:
            nonlocal buffer
            while final or re.search(r"[.!?;:]\s", buffer):
                if final and not re.search(r"[.!?;:]\s", buffer):
                    break
                match = re.search(r"[.!?;:]\s", buffer)
                if not match:
                    break
                sentence = buffer[: match.end()].strip()
                buffer = buffer[match.end():]
                if sentence and log_sentences:
                    self.bus.publish("info", f"agent/{key}", sentence)

        try:
            for delta in _stream_chat(f"{self.base_url}/api/chat", payload):
                full.append(delta)
                if log_sentences:
                    buffer += delta
                    flush()
            buffer += " "
            flush(final=True)
        except Exception as exc:  # noqa: BLE001
            self.bus.publish("error", f"agent/{key}", f"model call failed: {type(exc).__name__}: {exc}")
            raise
        return "".join(full)

    # -- real CrewAI path ---------------------------------------------------------------

    async def _run_crewai(self, text: str) -> str:
        from crewai import Agent, Crew, Task, LLM  # noqa: PLC0415 - optional dependency

        bus = self.bus
        llm = LLM(model=f"ollama/{self.model}", base_url=self.base_url + "/api")

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
        answer = str(getattr(result, "raw", result) or "").strip()
        bus.publish("success", "crew", "crew run complete")
        return answer
