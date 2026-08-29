"""Durable operator preferences.

Everything in the registry — model, voice, speed, persona, tools, recall,
volume, speech streaming — was previously in-memory only, so every restart
threw away the operator's choices and fell back to whatever the environment
happened to say. That is the wrong default for a daemon you leave running for
weeks: the machine should come back the way you left it.

This is a deliberately dumb JSON file next to the memory database. It is
written atomically (temp file + rename), never raises at the call site, and
treats a corrupt or unreadable file as "no preferences" rather than a fatal
boot error — a bad settings file must never cost you the assistant.

Environment variables still win at *first* boot (there are no saved prefs
yet); after that the saved value is authoritative, because it represents a
deliberate act by the operator rather than a shell default.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

#: only these keys are ever persisted — an unknown key in the file is dropped
#: rather than blindly re-applied, so a future rename cannot resurrect a stale
#: setting under a name that now means something else.
PERSISTED_KEYS = (
    "model", "voice", "speed", "think", "persona", "tools", "recall",
    "volume", "stream_speech", "theme",
)


def prefs_path() -> Path:
    raw = os.environ.get("JARVIS_DATA_DIR")
    root = Path(raw).expanduser() if raw else Path.home() / ".jarvis"
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


class Prefs:
    """Load-once / save-on-change preference file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or prefs_path()
        self.values: dict[str, Any] = {}
        self.loaded_at: float | None = None
        self.error: str | None = None
        self.load()

    # -- io -------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Read the file. A missing or broken file yields an empty mapping."""
        self.values = {}
        self.error = None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self.values
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self.values
        if not isinstance(raw, dict):
            self.error = "settings file is not an object"
            return self.values
        self.values = {k: v for k, v in raw.items() if k in PERSISTED_KEYS}
        self.loaded_at = raw.get("saved_at") if isinstance(raw.get("saved_at"), (int, float)) else None
        return self.values

    def save(self, values: dict[str, Any]) -> bool:
        """Merge `values` in and rewrite the file atomically.

        Returns False on any failure — a read-only home directory degrades the
        assistant to session-only settings, it does not break it.
        """
        merged = dict(self.values)
        merged.update({k: v for k, v in values.items() if k in PERSISTED_KEYS})
        if merged == self.values and self.path.exists():
            return True
        payload = dict(merged)
        payload["saved_at"] = time.time()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Temp file in the same directory so the rename is atomic on the
            # same filesystem; a crash mid-write can never truncate the real
            # file to zero bytes.
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".settings-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        self.values = merged
        self.error = None
        return True

    def clear(self) -> bool:
        """Forget every saved preference; the next boot uses env defaults."""
        self.values = {}
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    # -- access ---------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "saved": dict(self.values),
                "saved_at": self.loaded_at, "error": self.error}

