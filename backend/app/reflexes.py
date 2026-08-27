"""Reflex arcs: deterministic grounding that fires before the cortex.

A tool-capable model *decides* whether to call a tool, and that decision is
unreliable in exactly the cases that matter. Measured here on qwen3.5:9b, with
the strongest prompt wording tried and the `calculate` schema attached, the
model called the tool for `48271 * 9912` in only four runs out of six — and
every run where it declined produced a different wrong number. Lowering the
temperature did not help (2/6 at 0.3, 3/6 at 0.1): it is a judgement failure,
not a sampling one.

So the arithmetic is not left to judgement. A reflex is a cheap, local,
*deterministic* detector: if the directive plainly depends on a fact this
machine can compute exactly, the fact is computed up front and injected as
grounding. The model still composes the prose and may still call tools of its
own accord — it simply can no longer get the number wrong.

Biologically the metaphor is honest, too: a reflex arc is the spinal shortcut
that acts before the signal reaches the cortex at all.

Each firing lights the `intake.reflex` node on the neural graph, so the
interface shows the shortcut being taken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .skills import SkillError, safe_eval

#: Spoken arithmetic -> operators. Ordered longest-first so "multiplied by"
#: is consumed before "by" could ever be considered.
_WORD_OPS: list[tuple[str, str]] = [
    (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bto the power of\b", "**"),
    (r"\bpercent of\b", "*0.01*"),
    (r"%\s*of\b", "*0.01*"),
    (r"\btimes\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bover\b", "/"),
    (r"\bsquared\b", "**2"),
    (r"\bcubed\b", "**3"),
    (r"[×✕]", "*"),
    (r"[÷]", "/"),
    (r"(?<=\d)\s*x\s*(?=\d)", "*"),
]

#: An expression worth grounding: at least two operands and one operator.
_EXPRESSION = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:\*\*|[-+*/])\s*\d[\d,]*(?:\.\d+)?)+)"
    r"(?![\w.])"
)

_HOST_WORDS = re.compile(
    r"\b(cpu|processor|memory|ram|disk|storage|battery|power|load|uptime|"
    r"network|throughput|cores?|processes|swap|temperature)\b",
    re.I,
)
_LIVE_WORDS = re.compile(
    r"\b(right now|currently|at the moment|live|this machine|my machine|"
    r"my (?:mac|laptop|computer)|are we|am i|is it)\b",
    re.I,
)
_TIME_WORDS = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (?:week|month|year)|"
    r"how (?:long|many days)|what day|current (?:date|time))\b",
    re.I,
)


@dataclass(frozen=True)
class Reflex:
    """One grounded fact, ready to be handed to the model."""

    node: str
    label: str
    detail: str

    def as_line(self) -> str:
        return f"- {self.label}: {self.detail}"


def _normalise_arithmetic(text: str) -> str:
    lowered = text.lower()
    for pattern, replacement in _WORD_OPS:
        lowered = re.sub(pattern, replacement, lowered)
    return lowered


def find_expressions(text: str) -> list[str]:
    """Extract arithmetic worth computing exactly.

    Deliberately conservative: a lone number, a year, or a version string is
    not arithmetic, and grounding trivia like "2 + 2" only adds noise.
    """
    normalised = _normalise_arithmetic(text)
    found: list[str] = []
    for match in _EXPRESSION.finditer(normalised):
        expression = match.group(1).replace(",", "").strip()
        operands = re.findall(r"\d+(?:\.\d+)?", expression)
        if len(operands) < 2:
            continue
        # Skip mental arithmetic the model gets right anyway; the reflex exists
        # for the magnitudes where it silently does not.
        if all(len(o.split(".")[0]) <= 2 for o in operands):
            continue
        if expression not in found:
            found.append(expression)
    return found[:3]


def detect(text: str, *, vitals: dict[str, Any] | None = None,
           now_provider: Callable[[], dict[str, Any]] | None = None) -> list[Reflex]:
    """Return every fact this directive can be grounded with, deterministically."""
    reflexes: list[Reflex] = []

    for expression in find_expressions(text):
        try:
            value = safe_eval(expression)
        except (SkillError, ZeroDivisionError, OverflowError, ValueError):
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        rendered = f"{value:,}" if isinstance(value, int) else f"{value:,.6g}"
        reflexes.append(Reflex("motor.tools", "exact arithmetic",
                               f"{expression} = {rendered}"))

    if vitals and _HOST_WORDS.search(text) and _LIVE_WORDS.search(text):
        parts = []
        if vitals.get("cpu") is not None:
            parts.append(f"cpu {vitals['cpu']:.0f}% over {vitals.get('cores', '?')} cores")
        if vitals.get("mem_used_gb") is not None and vitals.get("mem_total_gb"):
            parts.append(
                f"memory {vitals['mem_used_gb']:.1f} of {vitals['mem_total_gb']:.0f} GB "
                f"({vitals.get('mem', 0):.0f}%)"
            )
        elif vitals.get("mem") is not None:
            parts.append(f"memory {vitals['mem']:.0f}%")
        if vitals.get("disk") is not None:
            parts.append(f"disk {vitals['disk']:.0f}% used, "
                         f"{vitals.get('disk_free_gb', 0):.0f} GB free")
        if vitals.get("battery") is not None:
            parts.append(f"battery {vitals['battery']:.0f}% on {vitals.get('power', 'unknown')}")
        if vitals.get("load"):
            parts.append("load " + " ".join(f"{v:.2f}" for v in vitals["load"][:3]))
        if parts:
            reflexes.append(Reflex("sense.host", "live host state", "; ".join(parts)))

    if now_provider is not None and _TIME_WORDS.search(text):
        stamp = now_provider()
        reflexes.append(Reflex("sense.host", "exact current date and time",
                               f"{stamp.get('spoken')} ({stamp.get('timezone', '')})".strip()))

    return reflexes


def as_prompt(reflexes: list[Reflex]) -> str:
    """Render grounded facts as the system message injected before the turn."""
    if not reflexes:
        return ""
    lines = [
        "Facts computed directly on this machine for this question. They are exact and "
        "take precedence over your own estimate:",
    ]
    lines.extend(r.as_line() for r in reflexes)
    lines.append(
        "Quote these figures exactly as written, in digits. Do not recalculate them, do "
        "not round them, and do not spell them out as words — a spelled-out long number "
        "is where you make mistakes, and the speech engine reads digits correctly. "
        "Answer every part of the question, and do not mention how you obtained these."
    )
    return "\n".join(lines)
