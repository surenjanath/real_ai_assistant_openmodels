"""Long-term memory: a local SQLite hippocampus.

The previous build remembered a conversation only for as long as the process
lived, and `JARVIS_MEMORY_TURNS` turns of it at that. This module gives the
assistant durable recall across restarts, entirely on-device:

  * **episodic**  — every conversational turn, with the model, latency and
                    session it belonged to.
  * **semantic**  — facts the user asked to be remembered ("my sister's name
                    is Ada"), keyed so they can be overwritten rather than
                    duplicated.
  * **notes**     — free-form dictation the user asked to be kept.
  * **reminders** — text plus a due timestamp; the scheduler speaks them.

Recall is keyword-ranked over SQLite FTS5 where the interpreter ships with it
(CPython on macOS does) and degrades to a scored LIKE scan where it does not,
so the feature never becomes a hard dependency.

Everything here is synchronous and cheap; callers wrap it in
``asyncio.to_thread`` so the event loop never blocks on disk.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at   REAL,
    title      TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    ts         REAL NOT NULL,
    model      TEXT,
    latency_ms INTEGER,
    origin     TEXT
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);

CREATE TABLE IF NOT EXISTS facts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    key     TEXT NOT NULL UNIQUE,
    value   TEXT NOT NULL,
    ts      REAL NOT NULL,
    hits    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS notes (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    text  TEXT NOT NULL,
    tags  TEXT,
    ts    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    due_ts     REAL NOT NULL,
    created_ts REAL NOT NULL,
    fired      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders(fired, due_ts);

CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    REAL NOT NULL,
    kind  TEXT NOT NULL,
    data  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events(kind, ts);
"""

_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    text, content='turns', content_rowid='id', tokenize='porter'
);
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
"""

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "what", "who", "when", "did", "do",
    "does", "was", "were", "is", "are", "i", "you", "me", "my", "your", "to",
    "of", "in", "on", "for", "it", "that", "this", "about", "tell", "say",
    "again", "please", "can", "could", "would", "we", "they", "he", "she",
}


def data_dir() -> Path:
    raw = os.environ.get("JARVIS_DATA_DIR")
    path = Path(raw).expanduser() if raw else Path.home() / ".jarvis"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Recollection:
    """One remembered exchange, with a relevance score."""

    role: str
    text: str
    ts: float
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "ts": self.ts,
                "score": round(self.score, 3)}


class Memory:
    """Durable store. One connection, guarded by SQLite's own locking."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "jarvis.db")
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.fts = self._try_fts()
        self.db.commit()
        self.session_id = self._open_session()

    def _try_fts(self) -> bool:
        try:
            self.db.executescript(_FTS)
            # Backfill anything written before FTS became available.
            self.db.execute(
                "INSERT INTO turns_fts(rowid, text) "
                "SELECT id, text FROM turns WHERE id NOT IN (SELECT rowid FROM turns_fts)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    def _open_session(self) -> int:
        cur = self.db.execute("INSERT INTO sessions(started_at) VALUES (?)", (time.time(),))
        self.db.commit()
        return int(cur.lastrowid or 0)

    def close(self) -> None:
        try:
            self.db.execute(
                "UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), self.session_id)
            )
            self.db.commit()
            self.db.close()
        except Exception:  # noqa: BLE001
            pass

    # -- episodic ------------------------------------------------------------

    def add_turn(self, role: str, text: str, *, model: str = "", latency_ms: int | None = None,
                 origin: str = "text") -> int:
        cur = self.db.execute(
            "INSERT INTO turns(session_id, role, text, ts, model, latency_ms, origin) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.session_id, role, text, time.time(), model, latency_ms, origin),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def recent_turns(self, limit: int = 20, session_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT role, text, ts, model, origin FROM turns"
        args: tuple = ()
        if session_only:
            sql += " WHERE session_id=?"
            args = (self.session_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(sql, (*args, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 6,
               exclude_ids: tuple[int, ...] = (),
               roles: tuple[str, ...] = ()) -> list[Recollection]:
        """Rank past turns against a natural-language query.

        `exclude_ids` keeps the turn currently being answered out of its own
        recall — without it the assistant "remembers" the question it was just
        asked, one second ago, and dutifully reports it back.

        `roles` narrows to particular speakers. It exists for a specific
        failure: an assistant turn is not evidence of anything — it is an
        unverified generation — and feeding one back as context lets a single
        wrong answer harden into a permanent belief. See
        `Orchestrator._recall_context`.
        """
        terms = [t for t in re.findall(r"[a-z0-9']{3,}", query.lower()) if t not in _STOPWORDS]
        if not terms:
            return []
        skip = tuple(int(i) for i in exclude_ids)
        wanted = tuple(roles)
        # Over-fetch so excluded rows do not eat into the requested limit.
        fetch = limit + len(skip) + (limit * 3 if wanted else 0)

        def keep(role: str, row_id: int) -> bool:
            return row_id not in skip and (not wanted or role in wanted)

        if self.fts:
            match = " OR ".join(terms)
            try:
                rows = self.db.execute(
                    "SELECT t.id, t.role, t.text, t.ts, bm25(turns_fts) AS rank "
                    "FROM turns_fts JOIN turns t ON t.id = turns_fts.rowid "
                    "WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, fetch),
                ).fetchall()
                # bm25 is "lower is better"; map into a 0..1 relevance.
                return [
                    Recollection(r["role"], r["text"], r["ts"], 1.0 / (1.0 + abs(r["rank"])))
                    for r in rows if keep(r["role"], r["id"])
                ][:limit]
            except sqlite3.OperationalError:
                self.fts = False

        rows = self.db.execute(
            "SELECT id, role, text, ts FROM turns ORDER BY id DESC LIMIT 800"
        ).fetchall()
        scored: list[Recollection] = []
        for row in rows:
            if not keep(row["role"], row["id"]):
                continue
            hay = row["text"].lower()
            hits = sum(1 for t in terms if t in hay)
            if hits:
                scored.append(Recollection(row["role"], row["text"], row["ts"], hits / len(terms)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    # -- sessions ------------------------------------------------------------

    def title_session(self, text: str, session_id: int | None = None) -> None:
        """Name a session after the first thing said in it.

        Only ever sets a title that is still empty, so the opening directive
        names the conversation and nothing later overwrites it.
        """
        title = " ".join(str(text).split())[:80]
        if not title:
            return
        self.db.execute(
            "UPDATE sessions SET title=? WHERE id=? AND (title IS NULL OR title='')",
            (title, session_id if session_id is not None else self.session_id),
        )
        self.db.commit()

    def sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Past conversations, newest first, with their size and span.

        Empty sessions are excluded: every process start opens one, so a
        machine that has been rebooted a few times would otherwise show a list
        of conversations that never happened.
        """
        rows = self.db.execute(
            "SELECT s.id, s.started_at, s.ended_at, s.title, "
            "       COUNT(t.id) AS turns, MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts "
            "FROM sessions s JOIN turns t ON t.session_id = s.id "
            "GROUP BY s.id HAVING turns > 0 ORDER BY s.id DESC LIMIT ?",
            (max(1, min(200, limit)),),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["current"] = item["id"] == self.session_id
            if not item.get("title"):
                # Fall back to the first thing the operator said in it, so a
                # session recorded before titling existed still reads as
                # something rather than as "Session 12".
                first = self.db.execute(
                    "SELECT text FROM turns WHERE session_id=? AND role='user' "
                    "ORDER BY id LIMIT 1", (item["id"],),
                ).fetchone()
                item["title"] = (" ".join(first["text"].split())[:80] if first else "")
            out.append(item)
        return out

    def session_turns(self, session_id: int, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, role, text, ts, model, latency_ms, origin FROM turns "
            "WHERE session_id=? ORDER BY id LIMIT ?",
            (int(session_id), max(1, min(2000, limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: int) -> int:
        """Erase one conversation and everything recorded inside it."""
        session_id = int(session_id)
        removed = int(self.db.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id=?", (session_id,)
        ).fetchone()[0])
        self.db.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        # The live session's row must survive: `add_turn` has a foreign key on
        # it in spirit, and deleting it would orphan the rest of this run.
        if session_id != self.session_id:
            self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.db.commit()
        return removed

    def stats(self) -> dict[str, Any]:
        def count(table: str, where: str = "") -> int:
            return int(self.db.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0])

        first = self.db.execute("SELECT MIN(ts) FROM turns").fetchone()[0]
        return {
            "turns": count("turns"),
            # Only sessions that actually contain something: a restart opens a
            # row whether or not anyone speaks, and counting those inflates
            # "conversations remembered" every time the process bounces.
            "sessions": count("sessions", "WHERE id IN (SELECT DISTINCT session_id FROM turns)"),
            "facts": count("facts"),
            "notes": count("notes"),
            "reminders": count("reminders", "WHERE fired=0"),
            "since": first,
            "path": str(self.path),
            "fts": self.fts,
            "size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
        }

    def forget_all(self) -> int:
        removed = int(self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
        self.db.executescript(
            "DELETE FROM turns; DELETE FROM facts; DELETE FROM notes; DELETE FROM reminders;"
        )
        if self.fts:
            try:
                self.db.execute("INSERT INTO turns_fts(turns_fts) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass
        self.db.commit()
        return removed

    # -- semantic facts -------------------------------------------------------

    def remember_fact(self, key: str, value: str) -> None:
        key = key.strip().lower()[:120]
        self.db.execute(
            "INSERT INTO facts(key, value, ts) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (key, value.strip(), time.time()),
        )
        self.db.commit()

    def recall_fact(self, key: str) -> str | None:
        key = key.strip().lower()
        # Select the matched row's own key, not the lookup term: a fuzzy hit
        # ("sister" matching "sister's name") would otherwise credit the hit to
        # a key that does not exist, so the counter stayed at zero forever and
        # `hits` was useless as a signal of what actually gets recalled.
        row = self.db.execute("SELECT key, value FROM facts WHERE key=?", (key,)).fetchone()
        if row is None:
            row = self.db.execute(
                "SELECT key, value FROM facts WHERE key LIKE ? ORDER BY ts DESC LIMIT 1",
                (f"%{key}%",),
            ).fetchone()
        if row is not None:
            self.db.execute("UPDATE facts SET hits=hits+1 WHERE key=?", (row["key"],))
            self.db.commit()
            return str(row["value"])
        return None

    def all_facts(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT key, value, ts, hits FROM facts ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def forget_fact(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM facts WHERE key=?", (key.strip().lower(),))
        self.db.commit()
        return cur.rowcount > 0

    # -- notes ---------------------------------------------------------------

    def add_note(self, text: str, tags: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO notes(text, tags, ts) VALUES (?,?,?)", (text.strip(), tags, time.time())
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def list_notes(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, text, tags, ts FROM notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int) -> bool:
        cur = self.db.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- reminders -----------------------------------------------------------

    def add_reminder(self, text: str, due_ts: float) -> int:
        cur = self.db.execute(
            "INSERT INTO reminders(text, due_ts, created_ts) VALUES (?,?,?)",
            (text.strip(), due_ts, time.time()),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def due_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        rows = self.db.execute(
            "SELECT id, text, due_ts FROM reminders WHERE fired=0 AND due_ts<=? ORDER BY due_ts",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_fired(self, reminder_id: int) -> None:
        self.db.execute("UPDATE reminders SET fired=1 WHERE id=?", (reminder_id,))
        self.db.commit()

    def pending_reminders(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, text, due_ts, created_ts FROM reminders WHERE fired=0 "
            "ORDER BY due_ts LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def cancel_reminder(self, reminder_id: int) -> bool:
        cur = self.db.execute("DELETE FROM reminders WHERE id=?", (reminder_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- events (metrics / audit) ---------------------------------------------

    def log_event(self, kind: str, data: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO events(ts, kind, data) VALUES (?,?,?)",
            (time.time(), kind, json.dumps(data, default=str)),
        )
        self.db.commit()

    def events(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if kind:
            rows = self.db.execute(
                "SELECT ts, kind, data FROM events WHERE kind=? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT ts, kind, data FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row["data"])
            except json.JSONDecodeError:
                payload = {}
            out.append({"ts": row["ts"], "kind": row["kind"], "data": payload})
        return out
