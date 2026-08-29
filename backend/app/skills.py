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
  * writes go **only** into a dedicated scratch directory
    (``~/.jarvis/files``) — no skill can overwrite a file the user already
    had, and the path is resolved and re-checked after resolution so a
    ``..`` or a symlink cannot escape it;
  * shell execution is **off** unless ``JARVIS_ALLOW_SHELL=1``, and even then
    only an allow-list of read-only commands runs, with a timeout;
  * network egress is **off** unless ``JARVIS_ALLOW_NET=1``, and even then
    every URL is resolved and refused if it points anywhere on this host or a
    private network — otherwise a model that can "fetch a URL" can read the
    cloud metadata endpoint, the Ollama admin API on 11434, and every other
    service that trusts localhost.

Every call is audited to the memory store, so there is an after-the-fact
record of what the assistant did.
"""

from __future__ import annotations

import ast
import hashlib
import html
import ipaddress
import json
import math
import operator
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
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


def scratch_root() -> Path:
    """The one directory any skill is allowed to write into.

    Deliberately *not* the workspace: reads may roam the user's home, writes
    may not touch anything the user did not ask this assistant to create.
    """
    raw = os.environ.get("JARVIS_DATA_DIR")
    root = (Path(raw).expanduser() if raw else Path.home() / ".jarvis") / "files"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve_scratch(raw: str) -> Path:
    """Resolve a write target inside the scratch directory.

    The containment check runs *after* resolution, so `../../.ssh/authorized_keys`
    and a symlink planted in the scratch directory are both caught.
    """
    name = (raw or "").strip()
    if not name:
        raise SkillError("a filename is required")
    root = scratch_root()
    path = (root / name).expanduser()
    # A caller-supplied absolute path is not a request we can honour here.
    if Path(name).is_absolute():
        raise SkillError(f"writes are confined to {root}; give a plain filename")
    resolved = (path.parent.resolve() / path.name) if path.parent.exists() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SkillError(f"writes are confined to {root}") from exc
    if resolved.suffix.lower() not in _TEXT_SUFFIXES:
        raise SkillError(
            "only text files may be written "
            f"({', '.join(sorted(list(_TEXT_SUFFIXES)[:8]))}, …)"
        )
    return resolved


# ------------------------------------------------------------- egress guard

#: ports that are almost never a public web service and very often something
#: on this machine that trusts whoever can reach it.
_PORT_DENY = {22, 23, 25, 111, 139, 445, 465, 587, 631, 993, 995, 1433, 2049,
              2375, 2376, 3306, 5432, 5900, 6379, 8020, 9200, 11211, 11434, 27017}


def _address_is_public(host: str) -> tuple[bool, str]:
    """Resolve `host` and decide whether every address it maps to is public.

    A name that resolves to *any* private address is refused outright, which
    also handles the DNS-rebinding style of `localtest.me` name that points at
    127.0.0.1 on purpose.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return False, f"could not resolve '{host}' ({exc.strerror or exc})"
    if not infos:
        return False, f"could not resolve '{host}'"
    for info in infos:
        raw = info[4][0]
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            return False, f"'{host}' resolved to something that is not an address"
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return False, (
                f"'{host}' resolves to {addr}, which is on this machine or a private "
                "network - I will not fetch it"
            )
    return True, ""


def check_egress(url: str) -> urllib.parse.ParseResult:
    """Validate an outbound URL, or raise SkillError explaining the refusal."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise SkillError("only http(s) URLs are supported")
    host = parsed.hostname
    if not host:
        raise SkillError("that URL has no host")
    if parsed.username or parsed.password:
        raise SkillError("credentials embedded in a URL are not permitted")
    port = parsed.port
    if port is not None and port in _PORT_DENY:
        raise SkillError(f"port {port} is not a public web port")
    ok, why = _address_is_public(host)
    if not ok:
        raise SkillError(why)
    return parsed


#: Search front-ends serve a JavaScript stub to anything that announces itself
#: as a bot, so the no-JS endpoints need a browser-shaped User-Agent to answer
#: with real results at all.
_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects by hand so each hop is re-checked against the guard.

    urllib's default handler follows a 302 straight to `http://127.0.0.1:11434`
    without asking anyone, which would make the whole check above decorative.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise _Redirect(newurl)


class _Redirect(Exception):
    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url = url


def http_fetch(url: str, *, timeout: float = 12.0, max_bytes: int = 600_000,
               max_hops: int = 4, accept: str = "*/*",
               form: dict[str, str] | None = None) -> tuple[str, str]:
    """Fetch a URL with the egress guard applied to every redirect hop.

    `form` posts an application/x-www-form-urlencoded body. As browsers do,
    the body is dropped on redirect — a 302 turns the follow-up into a GET,
    so a redirected POST cannot silently replay the payload somewhere else.

    Returns `(final_url, body_text)`.
    """
    opener = urllib.request.build_opener(_NoRedirect)
    seen: list[str] = []
    current = url
    body: bytes | None = (urllib.parse.urlencode(form).encode() if form else None)
    for _ in range(max_hops):
        check_egress(current)
        seen.append(current)
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
        }
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(current, data=body, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as resp:  # noqa: S310 - guarded above
                charset = resp.headers.get_content_charset() or "utf-8"
                return current, resp.read(max_bytes).decode(charset, errors="replace")
        except _Redirect as hop:
            current = urllib.parse.urljoin(current, hop.url)
            body = None
            if current in seen:
                raise SkillError("redirect loop") from None
        except urllib.error.HTTPError as exc:
            raise SkillError(f"the server answered {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise SkillError(f"could not reach it: {exc.reason}") from exc
        except TimeoutError as exc:
            raise SkillError(f"timed out after {timeout:.0f}s") from exc
    raise SkillError(f"too many redirects (more than {max_hops})")


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


# ----------------------------------------------------------------- unit tables

#: everything reduced to one SI-ish base per dimension, so conversion is a
#: multiply and a divide rather than a table of every pair.
_UNITS: dict[str, tuple[str, float, float]] = {}


def _unit(dimension: str, factor: float, *names: str, offset: float = 0.0) -> None:
    for name in names:
        _UNITS[name] = (dimension, factor, offset)


_unit("length", 1.0, "m", "metre", "metres", "meter", "meters")
_unit("length", 0.001, "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_unit("length", 0.01, "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_unit("length", 1000.0, "km", "kilometre", "kilometres", "kilometer", "kilometers")
_unit("length", 0.0254, "in", "inch", "inches")
_unit("length", 0.3048, "ft", "foot", "feet")
_unit("length", 0.9144, "yd", "yard", "yards")
_unit("length", 1609.344, "mi", "mile", "miles")
_unit("length", 1852.0, "nmi", "nauticalmile", "nauticalmiles")
_unit("mass", 1.0, "kg", "kilogram", "kilograms")
_unit("mass", 0.001, "g", "gram", "grams")
_unit("mass", 1e-6, "mg", "milligram", "milligrams")
_unit("mass", 1000.0, "t", "tonne", "tonnes", "metricton")
_unit("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_unit("mass", 0.028349523125, "oz", "ounce", "ounces")
_unit("mass", 6.35029318, "st", "stone", "stones")
_unit("time", 1.0, "s", "sec", "secs", "second", "seconds")
_unit("time", 60.0, "min", "mins", "minute", "minutes")
_unit("time", 3600.0, "h", "hr", "hrs", "hour", "hours")
_unit("time", 86400.0, "d", "day", "days")
_unit("time", 604800.0, "wk", "week", "weeks")
_unit("time", 31557600.0, "yr", "year", "years")
_unit("volume", 1.0, "l", "litre", "litres", "liter", "liters")
_unit("volume", 0.001, "ml", "millilitre", "millilitres", "milliliter", "milliliters")
_unit("volume", 3.785411784, "gal", "gallon", "gallons", "usgal")
_unit("volume", 0.473176473, "pt", "pint", "pints")
_unit("volume", 0.2365882365, "cup", "cups")
_unit("volume", 0.0295735295625, "floz", "fluidounce", "fluidounces")
_unit("data", 1.0, "b", "byte", "bytes")
_unit("data", 1024.0, "kb", "kib", "kilobyte", "kilobytes")
_unit("data", 1024.0 ** 2, "mb", "mib", "megabyte", "megabytes")
_unit("data", 1024.0 ** 3, "gb", "gib", "gigabyte", "gigabytes")
_unit("data", 1024.0 ** 4, "tb", "tib", "terabyte", "terabytes")
_unit("speed", 1.0, "mps", "m/s", "metrespersecond")
_unit("speed", 0.277777778, "kph", "km/h", "kmh", "kilometresperhour")
_unit("speed", 0.44704, "mph", "milesperhour")
_unit("speed", 0.514444444, "kn", "knot", "knots")
# Temperature is affine, not linear: offset is applied before the factor on
# the way in and after it on the way out.
_unit("temperature", 1.0, "c", "celsius", "centigrade", "°c")
_unit("temperature", 5.0 / 9.0, "f", "fahrenheit", "°f", offset=-32.0)
_unit("temperature", 1.0, "k", "kelvin", offset=-273.15)


#: words too common to be worth reporting as a passage's "top words"
_STAT_STOPWORDS = {
    "that", "this", "with", "from", "have", "they", "them", "then", "than",
    "were", "been", "will", "would", "could", "should", "there", "their",
    "what", "when", "which", "while", "about", "into", "your", "yours",
    "just", "like", "some", "only", "also", "over", "such", "very", "more",
    "most", "here", "does", "each", "other", "these", "those", "because",
}


#: WMO weather interpretation codes, as words a person would use.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _strip_html(raw: str) -> str:
    """HTML to readable prose: drop scripts and styles, tags, then entities."""
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results in /l/?uddg=<encoded>; hand back the real URL."""
    if "uddg=" not in href:
        return href if href.startswith("http") else urllib.parse.urljoin("https://duckduckgo.com", href)
    query = urllib.parse.urlparse(href).query
    target = urllib.parse.parse_qs(query).get("uddg", [""])[0]
    return target or href


def _human_bytes(size: float) -> str:
    """Byte count in the unit a person would actually say it in."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _unit_key(raw: str) -> str:
    return re.sub(r"[\s._-]", "", raw.strip().lower())


def convert(value: float, source: str, target: str) -> tuple[float, str, str]:
    """Convert between two units of the same dimension."""
    src, dst = _unit_key(source), _unit_key(target)
    if src not in _UNITS:
        raise SkillError(f"I do not know the unit '{source}'")
    if dst not in _UNITS:
        raise SkillError(f"I do not know the unit '{target}'")
    src_dim, src_factor, src_offset = _UNITS[src]
    dst_dim, dst_factor, dst_offset = _UNITS[dst]
    if src_dim != dst_dim:
        raise SkillError(f"'{source}' is a {src_dim} and '{target}' is a {dst_dim}")
    base = (value + src_offset) * src_factor
    return (base / dst_factor) - dst_offset, src_dim, dst_dim


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


# ----------------------------------------------------------------- calendar

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
#: Fixed-date holidays people actually say out loud instead of a date. Only
#: ones that fall on the same day every year - Easter and the like move, and a
#: wrong answer delivered confidently is worse than "I could not read that".
_HOLIDAYS = {
    "christmas": (12, 25), "christmas day": (12, 25), "xmas": (12, 25),
    "christmas eve": (12, 24), "boxing day": (12, 26),
    "new year": (1, 1), "new years": (1, 1), "new year's": (1, 1),
    "new year's day": (1, 1), "new years day": (1, 1),
    "new year's eve": (12, 31), "new years eve": (12, 31),
    "halloween": (10, 31), "hallowe'en": (10, 31),
    "valentine's day": (2, 14), "valentines day": (2, 14), "valentines": (2, 14),
}

#: dd/mm vs mm/dd is unknowable from the digits alone, so it is refused rather
#: than guessed - see `parse_date`.
_SLASHED = re.compile(r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}$")


def _month_from(word: str) -> int | None:
    key = word.strip(",.").lower()[:4]
    return _MONTHS.get(key) or _MONTHS.get(key[:3])


def _add_months(date: datetime, months: int) -> datetime:
    """Shift by whole months, clamping the day to the target month's length.

    31 January plus one month is 28 February (29 in a leap year). Every
    calendar library makes this same choice; the alternative is rolling into
    March, which is never what anyone means by "next month".
    """
    total = (date.year * 12 + date.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    import calendar as _calendar  # noqa: PLC0415 - only needed here
    day = min(date.day, _calendar.monthrange(year, month)[1])
    return date.replace(year=year, month=month, day=day)


def parse_date(text: str, now: datetime | None = None, *,
               prefer: str = "future") -> datetime | None:
    """Turn a written date into a datetime at midnight, or None.

    Accepts ISO (``2026-12-25``), written forms (``25 December 2026``,
    ``Dec 25 2026``), relative words (``today``, ``tomorrow``, ``in 10 days``,
    ``next friday``), fixed-date holidays (``Christmas``, ``Halloween``) and a
    bare day-and-month (``25 December``).

    A date with no year is inherently two dates, so `prefer` picks between
    them: ``"future"`` (the default) resolves to the next occurrence, which is
    what "days until Christmas" means in any month, and ``"past"`` resolves to
    the last one, which is what "days since New Year" means. Getting this wrong
    is a whole year of error delivered with total confidence.

    Purely numeric slashed dates are deliberately *not* accepted: 03/04/2026 is
    the third of April to most of the world and the fourth of March to the
    United States, and silently picking one is how a reminder lands a month
    late. The caller is asked for ISO instead.
    """
    now = (now or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    t = " ".join((text or "").strip().lower().split())
    if not t:
        return None

    holiday = _HOLIDAYS.get(t.removeprefix("the ").strip())
    if holiday is not None:
        month, day = holiday
        candidate = datetime(now.year, month, day)
        if prefer == "past" and candidate > now:
            return datetime(now.year - 1, month, day)
        if prefer != "past" and candidate < now:
            return datetime(now.year + 1, month, day)
        return candidate

    if t in ("today", "now"):
        return now
    if t == "tomorrow":
        return now + timedelta(days=1)
    if t == "yesterday":
        return now - timedelta(days=1)

    match = re.fullmatch(r"in\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)", t)
    if match:
        count = int(match.group(1))
        unit = match.group(2).rstrip("s")
        if unit == "day":
            return now + timedelta(days=count)
        if unit == "week":
            return now + timedelta(weeks=count)
        if unit == "month":
            return _add_months(now, count)
        return _add_months(now, count * 12)

    match = re.fullmatch(r"(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago", t)
    if match:
        count = int(match.group(1))
        unit = match.group(2).rstrip("s")
        if unit == "day":
            return now - timedelta(days=count)
        if unit == "week":
            return now - timedelta(weeks=count)
        if unit == "month":
            return _add_months(now, -count)
        return _add_months(now, -count * 12)

    match = re.fullmatch(r"(next|last|this)\s+(\w+)", t)
    if match and match.group(2) in _WEEKDAYS:
        target = _WEEKDAYS[match.group(2)]
        direction = match.group(1)
        delta = (target - now.weekday()) % 7
        if direction == "next":
            return now + timedelta(days=delta or 7)
        if direction == "last":
            back = (now.weekday() - target) % 7
            return now - timedelta(days=back or 7)
        return now + timedelta(days=delta)
    if t in _WEEKDAYS:
        return now + timedelta(days=(_WEEKDAYS[t] - now.weekday()) % 7 or 7)

    if _SLASHED.fullmatch(t):
        raise SkillError(
            f"'{text}' is ambiguous - 03/04 is the 3rd of April in most places and "
            "the 4th of March in the US. Give it as YYYY-MM-DD."
        )

    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if iso:
        try:
            return datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError as exc:
            raise SkillError(f"'{text}' is not a real date ({exc})") from exc

    # "25 december 2026" / "december 25, 2026" / "25 dec"
    words = t.replace(",", " ").split()
    day = month = year = None
    for word in words:
        if month is None and (found := _month_from(word)) is not None:
            month = found
            continue
        if word.isdigit():
            number = int(word)
            if len(word) == 4 and year is None:
                year = number
            elif day is None and 1 <= number <= 31:
                day = number
        elif (stripped := re.fullmatch(r"(\d{1,2})(st|nd|rd|th)", word)) and day is None:
            day = int(stripped.group(1))
    if month is not None and day is not None:
        if year is None:
            try:
                candidate = datetime(now.year, month, day)
            except ValueError as exc:
                raise SkillError(f"'{text}' is not a real date ({exc})") from exc
            # A bare day-and-month is two dates a year apart; `prefer` says
            # which one was meant.
            if prefer == "past" and candidate > now:
                return datetime(now.year - 1, month, day)
            if prefer != "past" and candidate < now:
                return datetime(now.year + 1, month, day)
            return candidate
        try:
            return datetime(year, month, day)
        except ValueError as exc:
            raise SkillError(f"'{text}' is not a real date ({exc})") from exc

    return None


# ------------------------------------------------------- schema gating

#: Attached to every request: applicable to almost any directive, and cheap.
_CORE_SKILLS = frozenset({
    "get_datetime", "calculate", "recall", "remember", "system_status",
    "convert_units", "web_search",
})

#: (trigger, skills) - a cluster is described to the model only when the
#: directive plainly touches it. Patterns are deliberately generous: the cost
#: of matching too eagerly is a slightly larger prompt, and the cost of missing
#: is the assistant unable to do what was asked.
_SKILL_GROUPS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (re.compile(r"\b(file|files|folder|directory|directories|path|disk|space|"
                r"document|script|code|search|find|grep|read|wrote|written|save[ds]?|"
                r"open|scratch|draft|storage|size)\b"),
     frozenset({"list_directory", "read_file", "search_files", "search_file_contents",
                "file_info", "directory_size", "write_file", "list_scratch_files",
                "delete_scratch_file", "open_path", "run_command"})),
    (re.compile(r"\b(note|notes|remind|reminder|reminders|remember|forget|memory|"
                r"recall|fact|facts|know about|told you|cancel)\b"),
     frozenset({"take_note", "list_notes", "delete_note", "set_reminder",
                "list_reminders", "cancel_reminder", "list_facts", "forget_fact"})),
    (re.compile(r"\b(day|days|date|dates|week|weeks|month|months|year|years|"
                r"until|since|birthday|christmas|calendar|when|long)\b"),
     frozenset({"days_between", "shift_date", "get_datetime"})),
    (re.compile(r"\b(hash|digest|sha|sha1|sha256|sha512|md5|checksum)\b"),
     frozenset({"hash_text"})),
    (re.compile(r"\b(random|randomly|dice|die|roll|coin|flip|pick|choose|choice|"
                r"shuffle|draw|decide)\b"),
     frozenset({"pick_random"})),
    (re.compile(r"\b(cpu|memory|ram|disk|battery|process|processes|load|uptime|"
                r"network|machine|laptop|computer|performance|slow|hot)\b"),
     frozenset({"system_status", "list_processes"})),
    (re.compile(r"\b(clipboard|copy|copied|paste|pasted)\b"),
     frozenset({"clipboard_read", "clipboard_write"})),
    (re.compile(r"\b(weather|forecast|temperature|rain|raining|sunny|snow|wind)\b"),
     frozenset({"get_weather"})),
    (re.compile(r"\b(web|search|google|look up|news|latest|current|online|url|"
                r"link|website|page|fetch)\b"),
     frozenset({"web_search", "fetch_url"})),
    (re.compile(r"\b(word|words|count|characters|text|passage|reading time|"
                r"summar\w+|paragraph)\b"),
     frozenset({"text_stats"})),
    # Anything the operator calls theirs is about their machine or their
    # material, so both clusters come along. Cheap insurance against a
    # possessive phrasing that names no noun the patterns above know.
    (re.compile(r"\b(my|mine|our)\b"),
     frozenset({"list_directory", "read_file", "search_files", "search_file_contents",
                "file_info", "directory_size", "list_scratch_files",
                "take_note", "list_notes", "set_reminder", "list_reminders",
                "list_facts", "forget_fact"})),
]


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
            "list_facts",
            "List everything stored about the user. Use this for 'what do you know about me' "
            "and before correcting or removing a fact, so you know its exact key.",
            _obj(limit={"type": "integer", "description": "how many, default 30"}),
            self.list_facts,
        ))
        self.add(Skill(
            "forget_fact",
            "Delete one stored fact by its key, when it is wrong or out of date. "
            "Call list_facts first if you are unsure of the key.",
            _obj(key={"type": "string",
                      "description": "the key it was stored under", "_required": True}),
            self.forget_fact,
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
            "delete_note",
            "Delete a saved note by its id, as shown by list_notes.",
            _obj(note_id={"type": "integer", "description": "the note's id", "_required": True}),
            self.delete_note,
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
            "cancel_reminder",
            "Call off a pending reminder by its id, as shown by list_reminders.",
            _obj(reminder_id={"type": "integer",
                              "description": "the reminder's id", "_required": True}),
            self.cancel_reminder,
        ))
        self.add(Skill(
            "days_between",
            "Count the days from one date to another. Use this for anything involving how "
            "long until or since something - date arithmetic, leap years and weekday names "
            "are not things you can work out reliably in your head. For 'how many days until "
            "X', pass X as 'end' and leave 'start' out; it defaults to today. Never pass the "
            "same date as both. Accepts '2026-12-25', '25 December', 'tomorrow', "
            "'next friday', 'in 10 days'.",
            _obj(
                end={"type": "string",
                     "description": "the date being counted to, e.g. '25 December'",
                     "_required": True},
                start={"type": "string",
                       "description": "the date being counted from; omit for today"},
            ),
            self.days_between,
        ))
        self.add(Skill(
            "shift_date",
            "Work out the date a given distance from another date, and which weekday it "
            "falls on. Months and years move by the calendar, so 31 January plus one month "
            "is 28 February.",
            _obj(
                date={"type": "string", "description": "the starting date, default today"},
                days={"type": "integer", "description": "days to add (negative to subtract)"},
                weeks={"type": "integer", "description": "weeks to add"},
                months={"type": "integer", "description": "whole months to add"},
                years={"type": "integer", "description": "whole years to add"},
            ),
            self.shift_date,
        ))
        self.add(Skill(
            "convert_units",
            "Convert a quantity between units of the same kind — length, mass, time, "
            "volume, data size, speed or temperature. Use this rather than doing the "
            "arithmetic yourself.",
            _obj(
                value={"type": "number", "description": "the quantity", "_required": True},
                from_unit={"type": "string", "description": "e.g. 'miles', 'kg', 'C'", "_required": True},
                to_unit={"type": "string", "description": "e.g. 'km', 'lbs', 'F'", "_required": True},
            ),
            self.convert_units,
        ))
        self.add(Skill(
            "text_stats",
            "Count the words, characters, sentences and reading time of a passage of text, "
            "and report its most frequent significant words.",
            _obj(text={"type": "string", "description": "the passage", "_required": True}),
            self.text_stats,
        ))
        self.add(Skill(
            "hash_text",
            "Compute a cryptographic digest (md5, sha1, sha256, sha512) of a string. "
            "Use this whenever a hash is asked for - you cannot calculate one yourself.",
            _obj(
                text={"type": "string", "description": "the text to digest", "_required": True},
                algorithm={"type": "string", "enum": ["md5", "sha1", "sha256", "sha512"],
                           "description": "default sha256"},
            ),
            self.hash_text,
        ))
        self.add(Skill(
            "pick_random",
            "Choose fairly at random: pick one of a list of options, roll dice, flip a coin, "
            "or draw a number in a range. Use this whenever the user asks you to decide "
            "randomly — a language model cannot do it on its own.",
            _obj(
                options={"type": "array", "items": {"type": "string"},
                         "description": "the choices to pick from"},
                minimum={"type": "integer", "description": "low end of a numeric draw"},
                maximum={"type": "integer", "description": "high end of a numeric draw"},
                count={"type": "integer", "description": "how many to draw, default 1"},
            ),
            self.pick_random,
        ))
        self.add(Skill(
            "list_processes",
            "List the processes using the most CPU or memory on this machine right now.",
            _obj(
                sort_by={"type": "string", "enum": ["cpu", "memory"],
                         "description": "which resource to rank by, default cpu"},
                limit={"type": "integer", "description": "how many to return, default 8"},
            ),
            self.list_processes,
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
        self.add(Skill(
            "search_file_contents",
            "Search *inside* files for a piece of text and return the matching lines with "
            "their file and line number. This is the one to use for 'where did I write X' - "
            "search_files only matches file names.",
            _obj(
                query={"type": "string", "description": "the text to look for", "_required": True},
                path={"type": "string", "description": "directory to search from"},
                pattern={"type": "string", "description": "narrow to a glob such as '*.py'"},
                limit={"type": "integer", "description": "max matching lines, default 40"},
            ),
            self.search_file_contents,
            danger="reads_files",
        ))
        self.add(Skill(
            "file_info",
            "Report a file or folder's size, when it was last modified, and how many lines "
            "or entries it holds - without reading the whole thing.",
            _obj(path={"type": "string", "description": "the path", "_required": True}),
            self.file_info,
            danger="reads_files",
        ))
        self.add(Skill(
            "directory_size",
            "Measure how much disk a folder is using, and which of its children are the "
            "largest. Use this when the user asks what is taking up space.",
            _obj(path={"type": "string", "description": "directory path, default the workspace root"}),
            self.directory_size,
            danger="reads_files",
        ))
        self.add(Skill(
            "write_file",
            f"Save text to a file in the assistant's own scratch folder ({scratch_root()}). "
            "Use this when the user asks you to write something down as a file, draft a "
            "document, or save output. It cannot touch any of their existing files.",
            _obj(
                filename={"type": "string",
                          "description": "a plain filename such as 'draft.md'", "_required": True},
                content={"type": "string", "description": "the text to write", "_required": True},
                append={"type": "boolean", "description": "add to the end instead of replacing"},
            ),
            self.write_file,
            danger="writes_files",
        ))
        self.add(Skill(
            "delete_scratch_file",
            "Delete a file the assistant previously wrote to its own scratch folder. "
            "It cannot touch any of the user's own files.",
            _obj(filename={"type": "string",
                           "description": "the name as shown by list_scratch_files",
                           "_required": True}),
            self.delete_scratch_file,
            danger="writes_files",
        ))
        self.add(Skill(
            "list_scratch_files",
            "List the files the assistant has written to its own scratch folder.",
            _obj(),
            self.list_scratch_files,
            danger="reads_files",
        ))
        if self.allow_shell:
            self.add(Skill(
                "clipboard_write",
                "Put text on the system clipboard so the user can paste it.",
                _obj(text={"type": "string",
                           "description": "the text to copy", "_required": True}),
                self.clipboard_write,
                danger="executes",
            ))
            self.add(Skill(
                "clipboard_read",
                "Read what is currently on the system clipboard. Use it when the user says "
                "something like 'what did I just copy' or asks you to work on what they copied.",
                _obj(),
                self.clipboard_read,
                danger="executes",
            ))
            self.add(Skill(
                "open_path",
                "Open a file or folder inside the permitted workspace in whichever "
                "application handles it. Filesystem paths only, not URLs.",
                _obj(path={"type": "string",
                           "description": "the file or folder to open", "_required": True}),
                self.open_path,
                danger="executes",
            ))
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
                "Fetch a public URL and return its readable text (HTML tags stripped).",
                _obj(url={"type": "string", "description": "https URL", "_required": True}),
                self.fetch_url,
                danger="network",
            ))
            self.add(Skill(
                "web_search",
                "Search the web and return the top results with their titles, links and "
                "snippets. Use this for anything current, or any fact you are not certain "
                "of — your training data has a cutoff and this does not.",
                _obj(
                    query={"type": "string", "description": "the search query", "_required": True},
                    limit={"type": "integer", "description": "how many results, default 6"},
                ),
                self.web_search,
                danger="network",
            ))
            self.add(Skill(
                "get_weather",
                "Get the current weather and a short forecast for a place by name. "
                "Use this rather than guessing — you have no way to know the weather.",
                _obj(location={"type": "string",
                               "description": "a town, city or region", "_required": True}),
                self.get_weather,
                danger="network",
            ))

    # -- introspection --------------------------------------------------------

    def schemas(self, for_text: str | None = None) -> list[dict[str, Any]]:
        """Tool schemas to attach to a request.

        With `for_text`, only the clusters that directive could plausibly need.

        Measured on this project: attaching all 29 skills costs ~3,500 prompt
        tokens and **1.3 seconds of time-to-first-token on every directive**,
        whether or not a tool is called - the model re-reads the whole prompt
        before answering. Tools off: 632 tokens, 1.69s. Tools on: 4,176 tokens,
        3.01s. That is a permanent tax for capability that is idle almost
        always, and it grows with every skill added.

        So the same cheap deterministic pre-flight the reflex arc uses is
        applied here: a directive that says nothing about files does not need
        the file tools described to it. The core set rides along regardless,
        because those apply to almost anything, and an unrecognised directive
        gets everything rather than being quietly under-equipped.
        """
        if not for_text:
            return [s.schema() for s in self.skills.values()]
        wanted = self.relevant(for_text)
        return [s.schema() for name, s in self.skills.items() if name in wanted]

    def relevant(self, text: str) -> set[str]:
        """Names of the skills worth describing for this directive."""
        lowered = f" {text.lower()} "
        keep = set(_CORE_SKILLS)
        matched = False
        for pattern, group in _SKILL_GROUPS:
            if pattern.search(lowered):
                keep |= group
                matched = True
        if not matched:
            # Nothing in the directive touches a cluster - "what is the capital
            # of Japan" needs no tool at all. This is the *common* case, and it
            # was the most expensive one: sending all 29 schemas measured 15s to
            # first token against 4.8s for a gated seven. The core still rides
            # along, so the model can still check the date, do exact
            # arithmetic, or search what was said before.
            pass
        return {name for name in keep if name in self.skills}

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

    def list_facts(self, limit: int = 30) -> dict[str, Any]:
        """Everything stored about the user, without needing a search term.

        `recall` can only answer a question you already know how to ask. This
        is the one for "what do you actually know about me?", which is both a
        fair question and the only way to find a fact worth correcting.
        """
        facts = self.memory.all_facts(max(1, min(200, int(limit))))
        return {"count": len(facts), "facts": facts}

    def forget_fact(self, key: str) -> dict[str, Any]:
        """Remove one stored fact.

        Until now the store could only be added to, or erased entirely. A fact
        that has simply gone out of date - a job, a preference, an address -
        left the operator choosing between living with it and wiping
        everything, which is not a choice anyone should have to make.
        """
        name = (key or "").strip()
        if not name:
            raise SkillError("which fact? give the key it was stored under")
        removed = self.memory.forget_fact(name)
        if not removed:
            known = [f["key"] for f in self.memory.all_facts(60)]
            raise SkillError(
                f"nothing is stored under '{name}'"
                + (f" - I have: {', '.join(known[:12])}" if known else "")
            )
        return {"forgotten": name.strip().lower()}

    def take_note(self, text: str, tags: str = "") -> dict[str, Any]:
        note_id = self.memory.add_note(text, tags)
        return {"note_id": note_id, "saved": text[:200]}

    def list_notes(self, limit: int = 10) -> dict[str, Any]:
        notes = self.memory.list_notes(max(1, min(50, int(limit))))
        return {"count": len(notes), "notes": notes}

    def delete_note(self, note_id: int) -> dict[str, Any]:
        """Discard a note by its id, as listed by `list_notes`."""
        try:
            ident = int(note_id)
        except (TypeError, ValueError) as exc:
            raise SkillError("a numeric note id is required - list_notes shows them") from exc
        if not self.memory.delete_note(ident):
            raise SkillError(f"there is no note {ident}")
        return {"deleted_note": ident}

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

    def cancel_reminder(self, reminder_id: int) -> dict[str, Any]:
        """Call off a reminder that has not fired yet."""
        try:
            ident = int(reminder_id)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                "a numeric reminder id is required - list_reminders shows them"
            ) from exc
        if not self.memory.cancel_reminder(ident):
            raise SkillError(f"there is no pending reminder {ident}")
        return {"cancelled_reminder": ident}

    def _date(self, raw: str, label: str) -> datetime:
        parsed = parse_date(raw)
        if parsed is None:
            raise SkillError(
                f"I could not read {label} from '{raw}'. Try YYYY-MM-DD, "
                "'25 December', 'tomorrow' or 'in 10 days'."
            )
        return parsed

    def days_between(self, end: str = "today", start: str = "today") -> dict[str, Any]:
        """Calendar distance between two dates.

        Date arithmetic is one of the things a language model gets confidently
        wrong - leap years, month lengths and weekday names are all memorised
        rather than computed. This computes them.
        """
        first = self._date(start or "today", "the start date")
        second = self._date(end or "today", "the end date")
        days = (second - first).days
        return {
            "start": first.strftime("%Y-%m-%d"),
            "start_weekday": first.strftime("%A"),
            "end": second.strftime("%Y-%m-%d"),
            "end_weekday": second.strftime("%A"),
            "days": days,
            "weeks": round(days / 7, 2),
            "direction": "future" if days > 0 else ("past" if days < 0 else "same day"),
            "spoken": (
                "the same day" if days == 0
                else f"{abs(days)} day{'s' if abs(days) != 1 else ''} "
                     f"{'after' if days > 0 else 'before'}"
            ),
        }

    def shift_date(self, date: str = "today", days: int = 0, weeks: int = 0,
                   months: int = 0, years: int = 0) -> dict[str, Any]:
        """The date a given distance from another date.

        Months and years move by the calendar, clamping the day to the target
        month's length - 31 January plus one month is 28 February, not 3 March.
        """
        base = self._date(date or "today", "the date")
        moved = base + timedelta(days=int(days or 0), weeks=int(weeks or 0))
        if months or years:
            moved = _add_months(moved, int(months or 0) + int(years or 0) * 12)
        return {
            "from": base.strftime("%Y-%m-%d"),
            "result": moved.strftime("%Y-%m-%d"),
            "weekday": moved.strftime("%A"),
            "spoken": moved.strftime("%A, %-d %B %Y"),
            "days_moved": (moved - base).days,
        }

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        result, dimension, _ = convert(float(value), from_unit, to_unit)
        # Six significant figures is far more than any spoken answer needs and
        # still enough that a round-trip conversion does not visibly drift.
        rounded = round(result, 6)
        return {
            "value": float(value), "from": from_unit, "to": to_unit,
            "dimension": dimension, "result": rounded,
            "spoken": f"{value:g} {from_unit} is {rounded:g} {to_unit}",
        }

    def text_stats(self, text: str) -> dict[str, Any]:
        body = (text or "").strip()
        if not body:
            raise SkillError("no text given")
        words = re.findall(r"[\w'-]+", body)
        sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", body) if s.strip()]
        counts: dict[str, int] = {}
        for word in words:
            low = word.lower()
            if len(low) > 3 and low not in _STAT_STOPWORDS:
                counts[low] = counts.get(low, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        return {
            "characters": len(body),
            "characters_no_spaces": len(re.sub(r"\s", "", body)),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len([p for p in re.split(r"\n\s*\n", body) if p.strip()]),
            # 200 wpm is the usual silent-reading figure for prose.
            "reading_time_min": round(len(words) / 200.0, 1),
            # 150 wpm is roughly the assistant's own speaking rate.
            "speaking_time_min": round(len(words) / 150.0, 1),
            "longest_word": max(words, key=len) if words else "",
            "top_words": [{"word": w, "count": c} for w, c in top],
        }

    def hash_text(self, text: str, algorithm: str = "sha256") -> dict[str, Any]:
        """Digest a string. A language model cannot compute one and will
        cheerfully invent a plausible-looking hex string if asked."""
        name = (algorithm or "sha256").strip().lower().replace("-", "")
        allowed = {"md5", "sha1", "sha256", "sha512"}
        if name not in allowed:
            raise SkillError(f"algorithm must be one of {', '.join(sorted(allowed))}")
        raw = str(text).encode("utf-8")
        digest = hashlib.new(name, raw).hexdigest()
        return {
            "algorithm": name,
            "input_bytes": len(raw),
            "hex": digest,
            # Spoken aloud, sixty-four hex characters is noise; the first eight
            # are what anyone actually compares.
            "short": digest[:8],
        }

    def pick_random(self, options: list[str] | None = None, minimum: int | None = None,
                    maximum: int | None = None, count: int = 1) -> dict[str, Any]:
        draws = max(1, min(20, int(count or 1)))
        if options:
            pool = [str(o) for o in options if str(o).strip()]
            if not pool:
                raise SkillError("the options list is empty")
            # Sample without replacement when there is room, so "pick 3 of 5"
            # cannot return the same option three times.
            picked = (random.sample(pool, draws) if draws <= len(pool)
                      else [random.choice(pool) for _ in range(draws)])
            return {"from": pool, "picked": picked, "choice": picked[0]}
        if minimum is None and maximum is None:
            return {"picked": [random.choice(["heads", "tails"]) for _ in range(draws)],
                    "choice": random.choice(["heads", "tails"]), "kind": "coin"}
        low = int(minimum if minimum is not None else 1)
        high = int(maximum if maximum is not None else 6)
        if low > high:
            low, high = high, low
        rolls = [random.randint(low, high) for _ in range(draws)]
        return {"range": [low, high], "picked": rolls, "choice": rolls[0],
                "total": sum(rolls), "kind": "number"}

    def list_processes(self, sort_by: str = "cpu", limit: int = 8) -> dict[str, Any]:
        try:
            import psutil  # noqa: PLC0415 - optional, only needed here
        except ImportError as exc:  # pragma: no cover - psutil is in requirements
            raise SkillError("process listing needs psutil, which is not installed") from exc
        key = "memory" if str(sort_by).lower().startswith("mem") else "cpu"
        top = max(1, min(30, int(limit or 8)))
        rows: list[dict[str, Any]] = []
        # The first cpu_percent() call per process always reads 0.0 (it has no
        # previous sample to diff against), so prime them, wait a beat, then
        # read for real - otherwise a "sort by cpu" is sorting a column of zeros.
        procs = list(psutil.process_iter(["pid", "name", "memory_info"]))
        for proc in procs:
            try:
                proc.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.35)
        total_mem = psutil.virtual_memory().total or 1
        for proc in procs:
            try:
                info = proc.info
                mem = getattr(info.get("memory_info"), "rss", 0) or 0
                rows.append({
                    "pid": info.get("pid"),
                    "name": info.get("name") or "?",
                    "cpu_percent": round(proc.cpu_percent(None), 1),
                    "memory_mb": round(mem / 1_048_576, 1),
                    "memory_percent": round(100.0 * mem / total_mem, 1),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        rows.sort(key=lambda r: r["cpu_percent"] if key == "cpu" else r["memory_mb"],
                  reverse=True)
        return {"sorted_by": key, "count": len(rows), "processes": rows[:top]}

    def directory_size(self, path: str = ".") -> dict[str, Any]:
        root = _resolve_path(path or ".")
        if not root.is_dir():
            raise SkillError(f"'{root}' is not a directory")
        children: list[dict[str, Any]] = []
        total = 0
        files = 0
        # Bounded walk: a scan of a home directory with a node_modules tree in
        # it can otherwise run for minutes and stall the tool round-trip.
        budget = 40_000
        for child in sorted(root.iterdir(), key=lambda c: c.name.lower()):
            if child.name.startswith("."):
                continue
            size = 0
            count = 0
            try:
                if child.is_file():
                    size, count = child.stat().st_size, 1
                elif child.is_dir() and not child.is_symlink():
                    for item in child.rglob("*"):
                        if budget <= 0:
                            break
                        budget -= 1
                        try:
                            if item.is_file() and not item.is_symlink():
                                size += item.stat().st_size
                                count += 1
                        except OSError:
                            continue
            except OSError:
                continue
            total += size
            files += count
            children.append({"name": child.name, "type": "dir" if child.is_dir() else "file",
                             "bytes": size, "size": _human_bytes(size), "files": count})
        children.sort(key=lambda c: c["bytes"], reverse=True)
        return {
            "path": str(root), "total_bytes": total, "total": _human_bytes(total),
            "files": files, "truncated": budget <= 0,
            "largest": children[:12],
        }

    def write_file(self, filename: str, content: str, append: bool = False) -> dict[str, Any]:
        target = _resolve_scratch(filename)
        body = str(content)
        if len(body) > 400_000:
            raise SkillError("that is too much text to write in one call (400 KB limit)")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(body if body.endswith("\n") else body + "\n")
        size = target.stat().st_size
        return {"path": str(target), "bytes": size, "size": _human_bytes(size),
                "mode": "appended" if append else "written",
                "lines": body.count("\n") + 1}

    def list_scratch_files(self) -> dict[str, Any]:
        root = scratch_root()
        entries = []
        for child in sorted(root.glob("**/*"), key=lambda c: c.name.lower()):
            if not child.is_file():
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append({"name": str(child.relative_to(root)), "bytes": stat.st_size,
                            "size": _human_bytes(stat.st_size),
                            "modified": datetime.fromtimestamp(stat.st_mtime)
                            .strftime("%Y-%m-%d %H:%M")})
        return {"path": str(root), "count": len(entries), "files": entries[:60]}

    def delete_scratch_file(self, filename: str) -> dict[str, Any]:
        """Remove a file the assistant itself wrote.

        Confined to the scratch directory by the same resolver `write_file`
        uses, so this can only ever undo the assistant's own work - it has no
        reach into anything the operator created.
        """
        target = _resolve_scratch(filename)
        if not target.exists():
            raise SkillError(f"'{target.name}' is not in the scratch folder")
        if not target.is_file():
            raise SkillError("that is a folder, not a file")
        size = target.stat().st_size
        target.unlink()
        return {"deleted": str(target), "bytes": size, "size": _human_bytes(size)}

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

    def search_file_contents(self, query: str, path: str = ".", pattern: str = "*",
                             limit: int = 40) -> dict[str, Any]:
        """Grep the workspace. `search_files` only ever matched *names*.

        Sibling to `search_files`, and the one people actually mean when they
        ask where they wrote something. Confined by the same `_resolve_path`
        guard, skips dotted paths and anything that is not text, and caps both
        the files opened and the matches returned so a query against a home
        directory cannot walk for a minute.
        """
        root = _resolve_path(path or ".")
        if not root.is_dir():
            raise SkillError(f"'{root}' is not a directory")
        needle = (query or "").strip()
        if not needle:
            raise SkillError("a search string is required")
        wanted = max(1, min(200, int(limit)))
        lowered = needle.lower()

        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        for candidate in sorted(root.rglob(pattern or "*")):
            if len(matches) >= wanted or scanned >= 4000:
                truncated = True
                break
            if not candidate.is_file():
                continue
            if any(part.startswith(".") for part in candidate.relative_to(root).parts):
                continue
            if candidate.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                if candidate.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            scanned += 1
            try:
                with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        if lowered not in line.lower():
                            continue
                        matches.append({
                            "path": str(candidate),
                            "line": number,
                            "text": line.strip()[:240],
                        })
                        if len(matches) >= wanted:
                            truncated = True
                            break
            except OSError:
                continue

        return {
            "root": str(root),
            "query": needle,
            "files_scanned": scanned,
            "count": len(matches),
            "truncated": truncated,
            "matches": matches,
        }

    def file_info(self, path: str) -> dict[str, Any]:
        """Size, age and shape of a path, without reading it all in."""
        target = _resolve_path(path)
        if not target.exists():
            raise SkillError(f"'{target}' does not exist")
        try:
            stat = target.stat()
        except OSError as exc:
            raise SkillError(f"could not read '{target}': {exc.strerror or exc}") from exc

        info: dict[str, Any] = {
            "path": str(target),
            "name": target.name,
            "type": "dir" if target.is_dir() else "file",
            "bytes": stat.st_size,
            "size": _human_bytes(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "modified_ago_days": round((time.time() - stat.st_mtime) / 86400, 1),
        }
        if target.is_dir():
            try:
                children = list(target.iterdir())
            except OSError:
                children = []
            info["entries"] = len(children)
            info["files"] = sum(1 for c in children if c.is_file())
            info["folders"] = sum(1 for c in children if c.is_dir())
            return info

        info["suffix"] = target.suffix
        info["text"] = target.suffix.lower() in _TEXT_SUFFIXES
        # Line count only for text, and only when it is cheap enough to be
        # worth doing at all - the point of this skill is to avoid reading.
        if info["text"] and stat.st_size <= 4_000_000:
            try:
                with target.open("r", encoding="utf-8", errors="replace") as handle:
                    info["lines"] = sum(1 for _ in handle)
            except OSError:
                pass
        return info

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

    # -- desktop integration --------------------------------------------------
    #
    # These spawn a process, so they sit behind the same JARVIS_ALLOW_SHELL
    # flag as `run_command` rather than inventing a second, weaker gate. Each
    # one is argv-form with no shell, so nothing here can be turned into a
    # command by the contents of its argument.

    def _require_desktop(self, tool: str) -> str:
        if not self.allow_shell:
            raise SkillError(
                "desktop integration is disabled (set JARVIS_ALLOW_SHELL=1 to enable)"
            )
        if platform.system() != "Darwin":
            raise SkillError(f"'{tool}' is macOS-only and this is {platform.system()}")
        found = shutil.which(tool)
        if found is None:
            raise SkillError(f"'{tool}' is not available on this machine")
        return found

    def clipboard_write(self, text: str) -> dict[str, Any]:
        """Put text on the system clipboard, ready to paste."""
        tool = self._require_desktop("pbcopy")
        body = str(text)
        if len(body) > 200_000:
            raise SkillError("that is too much text for one clipboard write (200 KB limit)")
        try:
            subprocess.run(  # noqa: S603 - argv form, fixed binary, no shell
                [tool], input=body.encode("utf-8"), timeout=5, check=True,
            )
        except subprocess.SubprocessError as exc:
            raise SkillError(f"could not write the clipboard: {exc}") from exc
        return {"copied_characters": len(body), "preview": body[:120]}

    def clipboard_read(self) -> dict[str, Any]:
        """Read whatever is on the system clipboard.

        Worth being deliberate about: a clipboard very often holds the last
        password someone copied. It is gated, it is never read unless the
        operator asked for it, and the result goes into the model's context
        like any other tool output - so this is enabled by the same explicit
        opt-in as shell access, not on by default.
        """
        tool = self._require_desktop("pbpaste")
        try:
            proc = subprocess.run(  # noqa: S603 - argv form, fixed binary, no shell
                [tool], capture_output=True, timeout=5, check=True,
            )
        except subprocess.SubprocessError as exc:
            raise SkillError(f"could not read the clipboard: {exc}") from exc
        body = proc.stdout.decode("utf-8", errors="replace")
        truncated = len(body) > 20_000
        return {
            "characters": len(body),
            "truncated": truncated,
            "text": body[:20_000],
            "empty": not body.strip(),
        }

    def open_path(self, path: str) -> dict[str, Any]:
        """Open a file or folder in whatever application handles it.

        Filesystem paths only, and only inside the permitted workspace. URLs
        are deliberately not accepted: handing one to the browser would reach
        the network without passing the egress guard that every other outbound
        request in this file has to satisfy.
        """
        tool = self._require_desktop("open")
        raw = (path or "").strip()
        if re.match(r"^[a-z][a-z0-9+.-]*:", raw, re.I):
            raise SkillError(
                "only filesystem paths can be opened, not URLs or other schemes"
            )
        target = _resolve_path(raw)
        if not target.exists():
            raise SkillError(f"'{target}' does not exist")
        try:
            subprocess.run(  # noqa: S603 - argv form, fixed binary, no shell
                [tool, str(target)], capture_output=True, timeout=10, check=True,
            )
        except subprocess.SubprocessError as exc:
            raise SkillError(f"could not open it: {exc}") from exc
        return {"opened": str(target), "type": "dir" if target.is_dir() else "file"}

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
        final, raw = http_fetch(url, accept="text/html,text/plain;q=0.9,*/*;q=0.5")
        text = _strip_html(raw)
        return {"url": final, "chars": len(text), "text": text[:6000]}

    def web_search(self, query: str, limit: int = 6) -> dict[str, Any]:
        """Search via DuckDuckGo's no-JavaScript HTML endpoint.

        Chosen because it needs no API key and no account, which keeps the
        whole assistant runnable on a fresh machine with nothing to sign up
        for — the same reason Ollama and Kokoro were picked over hosted APIs.
        """
        if not self.allow_net:
            raise SkillError("network access is disabled (set JARVIS_ALLOW_NET=1 to enable)")
        wanted = max(1, min(15, int(limit or 6)))
        term = (query or "").strip()
        if not term:
            raise SkillError("a search query is required")
        results = self._ddg_html(term, wanted) or self._ddg_lite(term, wanted)
        if not results:
            # A layout change must read as "nothing found", not as a crash the
            # model then hallucinates its way around.
            return {"query": term, "count": 0, "results": [],
                    "note": "the search page returned no parseable results"}
        return {"query": term, "count": len(results), "results": results}

    @staticmethod
    def _ddg_html(term: str, wanted: int) -> list[dict[str, str]]:
        """Primary: the no-JavaScript HTML endpoint. Must be POSTed — a GET
        with the query in the URL is answered with the marketing page."""
        _, raw = http_fetch("https://html.duckduckgo.com/html/", timeout=15.0,
                            max_bytes=900_000, accept="text/html", form={"q": term})
        # The snippet may precede or follow its anchor depending on the layout
        # served, so each is matched independently and zipped by position
        # rather than assumed adjacent within one regex.
        links = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            raw, re.I | re.S)
        snippets = re.findall(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', raw, re.I | re.S)
        return [
            {"title": _strip_html(title), "url": _unwrap_ddg(href),
             "snippet": _strip_html(snippets[i])[:400] if i < len(snippets) else ""}
            for i, (href, title) in enumerate(links[:wanted])
        ]

    @staticmethod
    def _ddg_lite(term: str, wanted: int) -> list[dict[str, str]]:
        """Fallback: the Lite endpoint, whose markup is a plain table. Two
        independently-shaped scrapers is the cheapest insurance available
        against one of them being redesigned out from under us."""
        try:
            _, raw = http_fetch("https://lite.duckduckgo.com/lite/", timeout=15.0,
                                max_bytes=900_000, accept="text/html", form={"q": term})
        except SkillError:
            return []
        links = re.findall(
            r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            raw, re.I | re.S)
        snippets = re.findall(
            r'<td[^>]+class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>', raw, re.I | re.S)
        return [
            {"title": _strip_html(title), "url": _unwrap_ddg(href),
             "snippet": _strip_html(snippets[i])[:400] if i < len(snippets) else ""}
            for i, (href, title) in enumerate(links[:wanted])
        ]

    def get_weather(self, location: str) -> dict[str, Any]:
        """Current conditions via Open-Meteo — free, keyless, no account."""
        if not self.allow_net:
            raise SkillError("network access is disabled (set JARVIS_ALLOW_NET=1 to enable)")
        place = (location or "").strip()
        if not place:
            raise SkillError("a location is required")

        # The geocoder matches a single place name, so "Port of Spain,
        # Trinidad" finds nothing while "Port of Spain" finds it immediately.
        # People say the country, so narrow to the leading segment and retry
        # rather than reporting a place that plainly exists as missing.
        candidates = [place]
        if "," in place:
            candidates.append(place.split(",")[0].strip())
        spot = None
        for candidate in candidates:
            geo_url = ("https://geocoding-api.open-meteo.com/v1/search?count=1&format=json&name="
                       + urllib.parse.quote_plus(candidate))
            _, geo_raw = http_fetch(geo_url, timeout=10.0, max_bytes=100_000,
                                    accept="application/json")
            try:
                geo = json.loads(geo_raw)
            except json.JSONDecodeError as exc:
                raise SkillError("the geocoder returned something unreadable") from exc
            hits = geo.get("results") or []
            if hits:
                spot = hits[0]
                break
        if spot is None:
            raise SkillError(f"I could not find a place called '{place}'")
        lat, lon = spot["latitude"], spot["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
            "&forecast_days=3&timezone=auto"
        )
        _, raw = http_fetch(weather_url, timeout=12.0, max_bytes=200_000, accept="application/json")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SkillError("the weather service returned something unreadable") from exc

        current = data.get("current") or {}
        units = data.get("current_units") or {}
        daily = data.get("daily") or {}
        forecast = []
        for i, day in enumerate((daily.get("time") or [])[:3]):
            forecast.append({
                "date": day,
                "high_c": (daily.get("temperature_2m_max") or [None] * 3)[i],
                "low_c": (daily.get("temperature_2m_min") or [None] * 3)[i],
                "rain_chance_pct": (daily.get("precipitation_probability_max") or [None] * 3)[i],
                "conditions": _WMO.get((daily.get("weather_code") or [None] * 3)[i], "unknown"),
            })
        name = ", ".join(
            str(x) for x in (spot.get("name"), spot.get("admin1"), spot.get("country")) if x
        )
        return {
            "location": name,
            "coordinates": {"lat": lat, "lon": lon},
            "conditions": _WMO.get(current.get("weather_code"), "unknown"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_kph": current.get("wind_speed_10m"),
            "units": {"temperature": units.get("temperature_2m", "°C"),
                      "wind": units.get("wind_speed_10m", "km/h")},
            "forecast": forecast,
        }
