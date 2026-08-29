"""Deterministic simulated multi-agent trace (no model server required).

Produces a realistic CrewAI-style reasoning log - decomposition, OSINT sweep,
tool calls, synthesis - tailored to the command text, then returns a spoken
answer. Used whenever Ollama is unreachable so the entire product loop works
on any host.
"""

from __future__ import annotations

import asyncio
import hashlib
import random

from ..logbus import LogBus
from .base import Runtime


class _Interrupted(Exception):
    """Raised out of a pause when the operator cuts the trace short."""


def _short(text: str, limit: int = 42) -> str:
    text = " ".join(text.split())
    return (text[: limit - 1] + "…") if len(text) > limit else text


class SimulatedRuntime:
    name = "simulated"

    def __init__(self, bus: LogBus, model_hint: str = "") -> None:
        self.bus = bus
        self.model_hint = model_hint
        self._aborted = False

    def abort(self) -> None:
        """Cut the trace short - see `Runtime.abort`."""
        self._aborted = True

    async def run(self, text: str) -> str:
        self._aborted = False
        try:
            return await self._trace(text)
        except _Interrupted:
            return ""

    async def _trace(self, text: str) -> str:
        bus = self.bus
        model_note = f" [model {self.model_hint} unverified - ollama offline]" if self.model_hint else ""
        seed = int(hashlib.blake2b(text.encode(), digest_size=3).hexdigest(), 16)
        rng = random.Random(seed)
        topic = _short(text)
        plan = self._plan(text, rng)

        await self._step(0.5)
        bus.publish("info", "agent/router", f"command parsed -> intent '{topic}' confidence 0.{rng.randint(86, 97)}{model_note}")
        await self._step(0.4)
        bus.publish("info", "agent/router", "plan: " + " | ".join(plan))

        await self._step(0.7)
        bus.publish("info", "agent/analyst", "osint sweep initiated - sources: local-index, memory-store, tool-registry")
        await self._step(rng.uniform(0.6, 1.2))
        findings = rng.randint(3, 9)
        bus.publish("info", "agent/analyst", f"osint sweep complete - {findings} relevant fragments, avg score 0.{rng.randint(71, 94)}")

        await self._step(0.5)
        bus.publish("info", "agent/engineer", "tool call -> mcp.exec(command='jarvis-inspect --topic') via n8n bridge")
        await self._step(rng.uniform(0.7, 1.4))
        bus.publish("success", "tool", f"mcp tool returned exit 0 - {rng.randint(2, 6)} rows, {rng.uniform(4, 48):.1f}ms")

        await self._step(0.6)
        bus.publish("info", "agent/synth", "synthesising spoken response (target < 90 words)")
        await self._step(0.3)

        return self._answer(text, rng)

    async def _step(self, delay: float) -> None:
        """A pause in the trace, and the only place an interrupt can land."""
        await asyncio.sleep(delay)
        if self._aborted:
            raise _Interrupted

    @staticmethod
    def _plan(text: str, rng: random.Random) -> list[str]:
        t = text.lower()
        steps = ["decompose_request"]
        if any(w in t for w in ("find", "search", "who", "what", "why", "news", "latest")):
            steps.append("gather_osint")
        if any(w in t for w in ("run", "exec", "script", "open", "kill", "start", "build", "deploy", "check")):
            steps.append("execute_tool")
        if not steps or len(steps) == 1:
            steps.append("gather_osint")
        steps += ["verify", "synthesise_answer"]
        return steps

    def _answer(self, text: str, rng: random.Random) -> str:
        topic = _short(text, 60)
        openers = [
            "Certainly.",
            "Right away, sir.",
            "I have completed the analysis.",
            "As you wish.",
        ]
        bodies = [
            f"Regarding \"{topic}\" — all subsystems report nominal and the workflow finished {rng.uniform(0.4, 3.9):.1f} seconds under budget. Shall I prepare a detailed digest?",
            f"I looked into \"{topic}\". The crew reached consensus: {rng.randint(2, 5)} viable approaches, one recommended. I can proceed on your word.",
            f"\"{topic}\" has been processed. Three data fragments were retrieved, verified, and summarised. Standing by for further instructions.",
        ]
        return f"{rng.choice(openers)} {rng.choice(bodies)}"
