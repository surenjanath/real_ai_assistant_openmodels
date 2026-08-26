"""Agentic layer (Phase 5).

Two interchangeable runtimes implement the same interface:

  * OllamaRuntime      - live multi-agent reasoning against a local Ollama
                         server, with an optional CrewAI crew when the
                         `crewai` package is installed and JARVIS_USE_CREWAI=1.
  * SimulatedRuntime   - deterministic, dependency-free trace so the full
                         product (logs -> terminal, answer -> voice) works on
                         any host before models are pulled.

Both publish their reasoning steps onto the LogBus and return the final
natural-language answer, which the orchestrator routes to the vocal engine.
"""

from .base import AgentPersonas, Runtime  # noqa: F401
from .ollama_runtime import OllamaRuntime  # noqa: F401
from .simulated_runtime import SimulatedRuntime  # noqa: F401

__all__ = ["AgentPersonas", "Runtime", "OllamaRuntime", "SimulatedRuntime"]
