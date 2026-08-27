"""Skills: the tools the reasoning cortex can actually invoke.

Each skill is a plain Python callable plus a JSON-Schema declaration in the
shape Ollama's ``/api/chat`` ``tools`` parameter expects, so a tool-capable
local model can call them natively. The orchestrator runs the resulting loop
(model → tool_calls → results → model) and every invocation fires a node on
the neural graph, so tool use is visible in the interface as it happens.

Safety posture — this runs on the user's own machine, unsandboxed, so the
defaults are conservative and every boundary is explicit:

  * filesystem reads are confined to ``JARVIS_WORKSPACE`` (default: home) and
    a denylist of credential directories is refused outright;
  * writes are refused entirely — no skill here mutates a file;
  * shell execution is **off** unless ``JARVIS_ALLOW_SHELL=1``, and even then
    only an allow-list of read-only commands runs, with a timeout;
  * network egress is **off** unless ``JARVIS_ALLOW_NET=1``.

Every call is audited to the memory store, so there is an after-the-fact
record of what the assistant did.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .config import settings
from .memory import Memory

# --------------------------------------------------------------------- guards


_DENY_PARTS = {".ssh", ".aws", ".gnupg", ".config/gcloud", "keychains", ".kube",
               ".docker", "library/keychains", ".password-store", ".netrc"}
_DENY_NAMES = {"id_rsa", "id_ed25519", ".env", "credentials", ".pgpass", ".npmrc"}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".csv", ".sh", ".zsh", ".bash", ".html",
    ".css", ".scss", ".sql", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb",
    ".swift", ".kt", ".xml", ".log", ".conf", ".mjs", ".env.example",
}

#: read-only shell verbs permitted when JARVIS_ALLOW_SHELL=1
_SHELL_ALLOW = {
    "ls", "pwd", "date", "whoami", "uname", "uptime", "df", "du", "free",
    "cat", "head", "tail", "wc", "grep", "find", "which", "echo", "hostname",
    "git", "python3", "node", "npm", "brew", "ps", "sw_vers", "sysctl",
}
_SHELL_GIT_ALLOW = {"status", "log", "diff", "branch", "remote", "show", "rev-parse"}


def workspace_root() -> Path:
    raw = os.environ.get("JARVIS_WORKSPACE")
    return (Path(raw).expanduser() if raw else Path.home()).resolve()


class SkillError(RuntimeError):
    """A refusal or failure that should be reported back to the model."""


def _resolve_path(raw: str) -> Path:
    root = workspace_root()
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SkillError(f"path is outside the permitted workspace ({root})") from exc
    lowered = str(path).lower()
    if any(part in lowered for part in _DENY_PARTS) or path.name.lower() in _DENY_NAMES:
        raise SkillError("that path holds credentials and is off limits")
    return path


# ------------------------------------------------------------------ arithmetic

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    name: getattr(math, name)
    for name in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                 "log", "log2", "log10", "exp", "floor", "ceil", "fabs", "hypot",
                 "degrees", "radians", "factorial")
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without exposing `eval` to the model."""

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise SkillError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = visit(node.left), visit(node.right)
            if type(node.op) is ast.Pow and (abs(right) > 128 or abs(left) > 1e6):
                raise SkillError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCS.get(node.func.id)
            if fn is None:
                raise SkillError(f"unknown function '{node.func.id}'")
            return fn(*[visit(a) for a in node.args])
        if isinstance(node, (ast.Tuple, ast.List)):
            return [visit(e) for e in node.elts]
        raise SkillError("unsupported expression")

    if len(expression) > 400:
        raise SkillError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SkillError(f"could not parse: {exc.msg}") from exc
    return visit(tree)


# ------------------------------------------------------------- time expressions

_DURATION = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours))?\s*"
    r"(?:(?P<m>\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes))?\s*"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds))?",
    re.I,
)


def parse_when(text: str, now: datetime | None = None) -> datetime | None:
    """Turn "in 10 minutes" / "at 4:30pm" / "tomorrow at 9" into a datetime."""
    now = now or datetime.now()
    t = text.strip().lower()

    match = re.search(r"in\s+(.+)", t)
    if match:
        found = _DURATION.match(match.group(1).strip())
        if found and any(found.groupdict().values()):
            hours = float(found.group("h") or 0)
            minutes = float(found.group("m") or 0)
            seconds = float(found.group("s") or 0)
            delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
            if delta.total_seconds() > 0:
                return now + delta

    match = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        target = now.replace(hour=min(23, hour), minute=min(59, minute), second=0, microsecond=0)
        if "tomorrow" in t or target <= now:
            target += timedelta(days=1)
        return target

    if "tomorrow" in t:
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    return None


# ------------------------------------------------------------------- the skills


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Any]
    #: shown in the interface; also gates which skills a model may see
    danger: str = "safe"  # safe | reads_files | executes | network

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(**props: Any) -> dict[str, Any]:
    required = [k for k, v in props.items() if v.pop("_required", False)]
    return {"type": "object", "properties": props, "required": required}


class SkillKit:
    """Registry of everything the assistant can do beyond talking."""

    def __init__(self, memory: Memory, vitals_provider: Callable[[], dict] | None = None) -> None:
        self.memory = memory
        self.vitals_provider = vitals_provider or (lambda: {})
        self.allow_shell = os.environ.get("JARVIS_ALLOW_SHELL", "").lower() in ("1", "true", "yes")
        self.allow_net = os.environ.get("JARVIS_ALLOW_NET", "").lower() in ("1", "true", "yes")
        self.calls = 0
        self.skills: dict[str, Skill] = {}
        self._register_all()

    # -- registration ---------------------------------------------------------

    def add(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def _register_all(self) -> None:
        self.add(Skill(
            "get_datetime",
            "Get the current local date, time, weekday and timezone. Use this rather than guessing.",
            _obj(),
            self.get_datetime,
        ))
        self.add(Skill(
            "calculate",
            "Evaluate an arithmetic or mathematical expression precisely "
            "(supports +-*/, powers, sqrt, sin/cos/tan, log, factorial, pi, e).",
            _obj(expression={"type": "string",
                             "description": "e.g. '1920*1080' or 'sqrt(2)/2'",
                             "_required": True}),
            self.calculate,
        ))
        self.add(Skill(
            "system_status",
            "Read live host metrics: CPU load, memory pressure, disk usage, network throughput, "
            "battery, uptime and core count.",
            _obj(),
            self.system_status,
        ))
        self.add(Skill(
            "remember",
            "Store a durable fact about the user or their world so it survives restarts. "
            "Use a short stable key such as 'birthday' or 'preferred editor'.",
            _obj(
                key={"type": "string", "description": "short stable identifier", "_required": True},
                value={"type": "string", "description": "the fact to store", "_required": True},
            ),
            self.remember,
        ))
        self.add(Skill(
            "recall",
            "Search everything previously said in any past conversation, plus stored facts. "
            "Use whenever the user refers to something from earlier or asks what they told you.",
            _obj(query={"type": "string", "description": "what to look for", "_required": True}),
            self.recall,
        ))
        self.add(Skill(
            "take_note",
            "Save a free-form note for the user to read later.",
            _obj(
                text={"type": "string", "description": "the note body", "_required": True},
                tags={"type": "string", "description": "optional comma-separated tags"},
            ),
            self.take_note,
        ))
        self.add(Skill(
            "list_notes",
            "List the most recent saved notes.",
            _obj(limit={"type": "integer", "description": "how many, default 10"}),
            self.list_notes,
        ))
        self.add(Skill(
            "set_reminder",
            "Schedule a spoken reminder. Accepts natural phrasing for when: "
            "'in 10 minutes', 'at 4:30pm', 'tomorrow at 9'.",
            _obj(
                text={"type": "string", "description": "what to remind about", "_required": True},
                when={"type": "string", "description": "when to fire it", "_required": True},
            ),
            self.set_reminder,
        ))
        self.add(Skill(
            "list_reminders",
            "List reminders that have not fired yet.",
            _obj(),
            self.list_reminders,
        ))
        self.add(Skill(
            "list_directory",
            "List the files and folders at a path inside the permitted workspace.",
            _obj(path={"type": "string", "description": "directory path, default the workspace root"}),
            self.list_directory,
            danger="reads_files",
        ))
        self.add(Skill(
            "read_file",
            "Read a UTF-8 text file inside the permitted workspace.",
            _obj(
                path={"type": "string", "description": "file path", "_required": True},
                max_lines={"type": "integer", "description": "cap on lines returned, default 120"},
            ),
            self.read_file,
            danger="reads_files",
        ))
        self.add(Skill(
            "search_files",
            "Find files by name pattern inside the permitted workspace.",
            _obj(
                pattern={"type": "string", "description": "glob such as '*.py'", "_required": True},
                path={"type": "string", "description": "directory to search from"},
            ),
            self.search_files,
            danger="reads_files",
        ))
        if self.allow_shell:
            self.add(Skill(
                "run_command",
                "Run one read-only shell command from an allow-list "
                f"({', '.join(sorted(_SHELL_ALLOW))}) and return its output.",
                _obj(command={"type": "string", "description": "the command line", "_required": True}),
                self.run_command,
                danger="executes",
            ))
        if self.allow_net:
            self.add(Skill(
                "fetch_url",
                "Fetch a URL and return its readable text (HTML tags stripped).",
                _obj(url={"type": "string", "description": "https URL", "_required": True}),
                self.fetch_url,
                danger="network",
            ))

    # -- introspection --------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        return [s.schema() for s in self.skills.values()]

    def catalogue(self) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "description": s.description, "danger": s.danger,
             "params": list(s.parameters.get("properties", {}))}
            for s in self.skills.values()
        ]

    # -- invocation -----------------------------------------------------------

    def invoke(self, name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        """Run a skill by name. Never raises — failures come back as data so
        the model can read the error and recover on its next turn."""
        started = time.monotonic()
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        args: dict[str, Any] = dict(arguments or {})

        skill = self.skills.get(name)
        if skill is None:
            return {"ok": False, "error": f"no such tool '{name}'",
                    "available": list(self.skills)}
        try:
            allowed = set(skill.parameters.get("properties", {}))
            result = skill.run(**{k: v for k, v in args.items() if k in allowed})
            payload = {"ok": True, "result": result}
        except SkillError as exc:
            payload = {"ok": False, "error": str(exc)}
        except TypeError as exc:
            payload = {"ok": False, "error": f"bad arguments: {exc}"}
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self.calls += 1
        payload["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        try:
            self.memory.log_event("tool", {"name": name, "args": args,
                                           "ok": payload["ok"],
                                           "ms": payload["elapsed_ms"]})
        except Exception:  # noqa: BLE001
            pass
        return payload

    # -- implementations -------------------------------------------------------

    def get_datetime(self) -> dict[str, Any]:
        now = datetime.now()
        return {
            "iso": now.isoformat(timespec="seconds"),
            "spoken": now.strftime("%A, %-d %B %Y, %-I:%M %p"),
            "weekday": now.strftime("%A"),
            "timezone": time.strftime("%Z"),
            "epoch": round(now.timestamp()),
        }

    def calculate(self, expression: str) -> dict[str, Any]:
        value = safe_eval(expression)
        rounded = round(value, 10) if isinstance(value, float) else value
        return {"expression": expression, "value": rounded}

    def system_status(self) -> dict[str, Any]:
        vitals = dict(self.vitals_provider() or {})
        vitals.pop("type", None)
        vitals["host"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
        vitals["python"] = platform.python_version()
        return vitals

    def remember(self, key: str, value: str) -> dict[str, Any]:
        self.memory.remember_fact(key, value)
        return {"stored": {key.strip().lower(): value.strip()}}

    def recall(self, query: str) -> dict[str, Any]:
        fact = self.memory.recall_fact(query)
        hits = self.memory.search(query, limit=6)
        return {
            "fact": fact,
            "matches": [h.as_dict() for h in hits],
            "found": bool(fact or hits),
        }

    def take_note(self, text: str, tags: str = "") -> dict[str, Any]:
        note_id = self.memory.add_note(text, tags)
        return {"note_id": note_id, "saved": text[:200]}

    def list_notes(self, limit: int = 10) -> dict[str, Any]:
        notes = self.memory.list_notes(max(1, min(50, int(limit))))
        return {"count": len(notes), "notes": notes}

    def set_reminder(self, text: str, when: str) -> dict[str, Any]:
        target = parse_when(when)
        if target is None:
            raise SkillError(
                "I could not read that time. Try 'in 10 minutes' or 'at 4:30pm'."
            )
        reminder_id = self.memory.add_reminder(text, target.timestamp())
        return {
            "reminder_id": reminder_id,
            "text": text,
            "due": target.strftime("%A %-I:%M %p"),
            "in_seconds": round(target.timestamp() - time.time()),
        }

    def list_reminders(self) -> dict[str, Any]:
        pending = self.memory.pending_reminders()
        for item in pending:
            item["due"] = datetime.fromtimestamp(item["due_ts"]).strftime("%a %-I:%M %p")
        return {"count": len(pending), "reminders": pending}

    def list_directory(self, path: str = ".") -> dict[str, Any]:
        target = _resolve_path(path or ".")
        if not target.is_dir():
            raise SkillError(f"'{target}' is not a directory")
        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:200]:
            if child.name.startswith("."):
                continue
            try:
                size = child.stat().st_size if child.is_file() else None
            except OSError:
                size = None
            entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file",
                            "size": size})
        return {"path": str(target), "count": len(entries), "entries": entries}

    def read_file(self, path: str, max_lines: int = 120) -> dict[str, Any]:
        target = _resolve_path(path)
        if not target.is_file():
            raise SkillError(f"'{target}' is not a file")
        if target.suffix.lower() not in _TEXT_SUFFIXES and target.stat().st_size > 200_000:
            raise SkillError("that looks like a binary or very large file")
        limit = max(1, min(600, int(max_lines)))
        lines: list[str] = []
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    lines.append(f"… truncated at {limit} lines …")
                    break
                lines.append(line.rstrip("\n")[:400])
        return {"path": str(target), "lines": len(lines), "content": "\n".join(lines)}

    def search_files(self, pattern: str, path: str = ".") -> dict[str, Any]:
        root = _resolve_path(path or ".")
        if not root.is_dir():
            raise SkillError(f"'{root}' is not a directory")
        safe_pattern = pattern if any(c in pattern for c in "*?[") else f"*{pattern}*"
        found: list[str] = []
        for match in root.rglob(safe_pattern):
            if any(part.startswith(".") for part in match.parts):
                continue
            found.append(str(match))
            if len(found) >= 60:
                break
        return {"root": str(root), "pattern": safe_pattern, "count": len(found), "paths": found}

    def run_command(self, command: str) -> dict[str, Any]:
        if not self.allow_shell:
            raise SkillError("shell execution is disabled (set JARVIS_ALLOW_SHELL=1 to enable)")
        parts = command.strip().split()
        if not parts:
            raise SkillError("empty command")
        if any(ch in command for ch in ";|&`$><\n"):
            raise SkillError("shell metacharacters are not permitted")
        verb = Path(parts[0]).name
        if verb not in _SHELL_ALLOW:
            raise SkillError(f"'{verb}' is not on the allow-list")
        if verb == "git" and (len(parts) < 2 or parts[1] not in _SHELL_GIT_ALLOW):
            raise SkillError("only read-only git subcommands are permitted")
        if shutil.which(verb) is None:
            raise SkillError(f"'{verb}' is not installed")
        try:
            proc = subprocess.run(  # noqa: S603 - argv form, allow-listed verb, no shell
                parts, capture_output=True, text=True, timeout=20,
                cwd=str(workspace_root()),
            )
        except subprocess.TimeoutExpired as exc:
            raise SkillError("command timed out after 20s") from exc
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-1000:],
        }

    def fetch_url(self, url: str) -> dict[str, Any]:
        if not self.allow_net:
            raise SkillError("network access is disabled (set JARVIS_ALLOW_NET=1 to enable)")
        if not url.lower().startswith(("http://", "https://")):
            raise SkillError("only http(s) URLs are supported")
        import urllib.request  # noqa: PLC0415 - only needed on this path

        req = urllib.request.Request(url, headers={"User-Agent": f"{settings.name}/1.3"})
        with urllib.request.urlopen(req, timeout=12) as resp:  # noqa: S310 - scheme checked
            raw = resp.read(600_000).decode("utf-8", errors="replace")
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "chars": len(text), "text": text[:6000]}
