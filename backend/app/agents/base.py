"""Agent runtime interface and personas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..logbus import LogBus


@dataclass(frozen=True)
class Persona:
    key: str
    role: str
    goal: str
    backstory: str


class AgentPersonas:
    """The standing crew - mirrors the PRD's specialised agents."""

    ROUTER = Persona(
        key="router",
        role="Task Decomposer",
        goal="Translate a user command into a minimal, ordered plan for the crew.",
        backstory="You are the J.A.R.V.I.S. router. You think in numbered steps and never speculate.",
    )
    ANALYST = Persona(
        key="analyst",
        role="OSINT & Data Specialist",
        goal="Gather context and quantitative facts relevant to the plan.",
        backstory="You are meticulous. You cite sources and quantify uncertainty.",
    )
    ENGINEER = Persona(
        key="engineer",
        role="Systems Engineer",
        goal="Execute tools, scripts and system commands required by the plan.",
        backstory="You run code and shell commands inside a sandbox and report exact output.",
    )
    SYNTH = Persona(
        key="synth",
        role="Response Synthesizer",
        goal="Compose the final spoken answer: concise, precise, addressed as J.A.R.V.I.S.",
        backstory="You speak like a composed British butler-AI. Under 90 words. No markdown.",
    )

    ALL = (ROUTER, ANALYST, ENGINEER, SYNTH)

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(p.key for p in cls.ALL)


class Runtime(Protocol):
    name: str  # "ollama" | "simulated" | "crewai"

    async def run(self, text: str) -> str:
        """Execute the command; publish reasoning logs; return the answer."""
        ...  # pragma: no cover


def persona_for(bus: LogBus) -> str:
    """Stable short prefix used in log lines, e.g. 'agent/engineer'."""
    return "agent"
