"use client";

/**
 * The archive: every conversation this assistant has ever had, and every skill
 * it has ever run.
 *
 * The backend has recorded both since the hippocampus was added — sessions in
 * `sessions`/`turns`, invocations in `events` — but neither had a way in.
 * Search finds a conversation only when you already remember a word from it;
 * what people actually remember is *when*, so the index leads with the day and
 * the opening directive.
 *
 * The audit tab exists for a different reason. This thing reads files, writes
 * files and (when armed) makes network requests on the operator's own machine.
 * A record of that which can only be read with `sqlite3` is not a record
 * anybody checks, so it is here, in front of them, with the arguments each
 * call was actually made with.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type AuditEvent,
  type SessionSummary,
  type SessionTurn,
} from "@/lib/api";
import { useJarvis } from "@/state/jarvis";

type Tab = "sessions" | "audit";

/* ------------------------------------------------------------- formatting -- */

function dayLabel(ts: number): string {
  const date = new Date(ts * 1000);
  const today = new Date();
  const yesterday = new Date(today.getTime() - 86_400_000);
  const same = (a: Date, b: Date) => a.toDateString() === b.toDateString();
  if (same(date, today)) return "TODAY";
  if (same(date, yesterday)) return "YESTERDAY";
  return date
    .toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })
    .toUpperCase();
}

function clock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function span(from: number, to: number): string {
  const seconds = Math.max(0, to - from);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/** Compact rendering of the arguments a tool was called with. */
function argSummary(args: Record<string, unknown> | undefined): string {
  if (!args || Object.keys(args).length === 0) return "no arguments";
  return Object.entries(args)
    .map(([key, value]) => {
      const text =
        typeof value === "string" ? value : JSON.stringify(value ?? null);
      return `${key}=${text.length > 60 ? `${text.slice(0, 60)}…` : text}`;
    })
    .join("  ");
}

/* --------------------------------------------------------------- sessions -- */

function SessionsTab() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [turns, setTurns] = useState<SessionTurn[] | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const pushToast = useJarvis((s) => s.pushToast);
  const memoryVersion = useJarvis((s) => s.memoryVersion);
  const bodyRef = useRef<HTMLDivElement>(null);

  const reload = useCallback(async () => {
    const result = await api.sessions();
    setSessions(result?.sessions ?? []);
  }, []);

  // Reloads on mount and again whenever the durable store changes — including
  // when it was another client that erased something.
  useEffect(() => {
    void reload();
  }, [reload, memoryVersion]);

  const open = async (id: number) => {
    if (openId === id) {
      setOpenId(null);
      setTurns(null);
      return;
    }
    setOpenId(id);
    setTurns(null);
    const result = await api.session(id);
    setTurns(result?.turns ?? []);
    // Bring the expanded conversation into view; a session near the bottom of
    // a long index otherwise unfolds entirely off-screen.
    requestAnimationFrame(() =>
      bodyRef.current
        ?.querySelector(`[data-session="${id}"]`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" }),
    );
  };

  const remove = async (id: number) => {
    // Two clicks, because this is the one control here that destroys data and
    // there is no undo behind it.
    if (confirming !== id) {
      setConfirming(id);
      window.setTimeout(() => setConfirming((c) => (c === id ? null : c)), 4000);
      return;
    }
    setConfirming(null);
    const result = await api.deleteSession(id);
    if (result?.ok) {
      pushToast("warn", "Conversation erased", `${result.removed} exchange(s) removed`);
      if (openId === id) {
        setOpenId(null);
        setTurns(null);
      }
      void reload();
    } else {
      pushToast("error", "Could not erase", "the backend refused the request");
    }
  };

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle || !sessions) return sessions ?? [];
    return sessions.filter((s) => (s.title ?? "").toLowerCase().includes(needle));
  }, [sessions, filter]);

  if (sessions === null) return <div className="overlay-empty">reading the archive…</div>;
  if (sessions.length === 0) {
    return (
      <div className="overlay-empty">
        nothing archived yet — conversations appear here once you have had one.
      </div>
    );
  }

  const totalTurns = sessions.reduce((sum, s) => sum + s.turns, 0);

  return (
    <>
      <p className="overlay-note">
        {sessions.length} conversation{sessions.length === 1 ? "" : "s"} · {totalTurns} exchanges,
        all on this machine. Select one to read it back.
      </p>

      <input
        className="archive-filter"
        placeholder="filter by opening directive…"
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
        spellCheck={false}
      />

      <div className="archive-list" ref={bodyRef}>
        {visible.length === 0 && (
          <div className="overlay-empty">no conversation opens with that.</div>
        )}
        {visible.map((session, index) => {
          const previous = visible[index - 1];
          const newDay =
            !previous || dayLabel(previous.first_ts) !== dayLabel(session.first_ts);
          const expanded = openId === session.id;
          return (
            <div key={session.id}>
              {newDay && <div className="archive-day">{dayLabel(session.first_ts)}</div>}
              <div
                className={`archive-row ${expanded ? "on" : ""}`}
                data-session={session.id}
              >
                <button
                  type="button"
                  className="archive-open"
                  onClick={() => void open(session.id)}
                  aria-expanded={expanded}
                >
                  <span className="archive-title">
                    {session.title || <em>untitled conversation</em>}
                    {session.current && <b className="archive-live">live</b>}
                  </span>
                  <span className="archive-meta">
                    {clock(session.first_ts)} · {session.turns} exchange
                    {session.turns === 1 ? "" : "s"} · {span(session.first_ts, session.last_ts)}
                  </span>
                </button>
                <button
                  type="button"
                  className={`archive-del ${confirming === session.id ? "armed" : ""}`}
                  onClick={() => void remove(session.id)}
                  title={
                    confirming === session.id
                      ? "Click again to erase permanently"
                      : "Erase this conversation"
                  }
                >
                  {confirming === session.id ? "sure?" : "✕"}
                </button>
              </div>

              {expanded && (
                <div className="archive-transcript">
                  {turns === null && <div className="overlay-empty">loading…</div>}
                  {turns?.map((turn) => (
                    <div className={`turn turn-${turn.role}`} key={turn.id}>
                      <span className="turn-who">
                        {turn.role === "user" ? "YOU" : "J.A.R.V.I.S."}
                        <em className="turn-stamp">{clock(turn.ts)}</em>
                      </span>
                      <span className="turn-text">{turn.text}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ audit -- */

function AuditTab() {
  const [events, setEvents] = useState<AuditEvent[] | null>(null);
  const [onlyFailed, setOnlyFailed] = useState(false);

  useEffect(() => {
    void api.auditTrail(200).then((result) => setEvents(result?.events ?? []));
  }, []);

  const shown = useMemo(
    () => (events ?? []).filter((e) => !onlyFailed || e.data.ok === false),
    [events, onlyFailed],
  );

  /** How many times each skill has ever been run — the honest capability list. */
  const tally = useMemo(() => {
    const counts = new Map<string, number>();
    for (const event of events ?? []) {
      const name = event.data.name ?? "?";
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [events]);

  if (events === null) return <div className="overlay-empty">reading the audit trail…</div>;
  if (events.length === 0) {
    return (
      <div className="overlay-empty">
        no skill has been invoked yet — this fills in as tools are actually used.
      </div>
    );
  }

  const failures = events.filter((e) => e.data.ok === false).length;

  return (
    <>
      <p className="overlay-note">
        The last {events.length} skill invocation{events.length === 1 ? "" : "s"}, recorded
        whether the cortex called {events.length === 1 ? "it" : "them"} or you did.{" "}
        {failures > 0 ? `${failures} failed.` : "None failed."}
      </p>

      <div className="audit-tally">
        {tally.map(([name, count]) => (
          <span className="audit-chip" key={name}>
            {name.replace(/_/g, " ")}
            <b>{count}</b>
          </span>
        ))}
      </div>

      <div className="archive-controls">
        <button
          type="button"
          className={`filter ${onlyFailed ? "on" : ""}`}
          onClick={() => setOnlyFailed((v) => !v)}
        >
          failures only
        </button>
      </div>

      <div className="audit-list">
        {shown.length === 0 && <div className="overlay-empty">no failures recorded.</div>}
        {shown.map((event, index) => (
          <div className={`audit-row ${event.data.ok === false ? "bad" : ""}`} key={index}>
            <span className="audit-when">{clock(event.ts)}</span>
            <span className="audit-name">{(event.data.name ?? "?").replace(/_/g, " ")}</span>
            <span className="audit-args">{argSummary(event.data.args)}</span>
            <span className="audit-ms">
              {event.data.ms != null ? `${event.data.ms}ms` : ""}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------- root -- */

export default function Archive() {
  const [tab, setTab] = useState<Tab>("sessions");

  return (
    <>
      <div className="archive-tabs">
        {(
          [
            ["sessions", "CONVERSATIONS"],
            ["audit", "AUDIT TRAIL"],
          ] as Array<[Tab, string]>
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`tab ${tab === key ? "on" : ""}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "sessions" ? <SessionsTab /> : <AuditTab />}
    </>
  );
}
