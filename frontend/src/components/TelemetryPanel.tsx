"use client";

/**
 * Telemetry panel (PRD §2): floating semi-transparent terminal anchored to
 * the right edge. Monospace, auto-scrolling, color-coded:
 *   gray = processing · blue = STT/TTS · green = workflow success
 * (plus amber warnings / red errors for completeness).
 *
 * Four tabs, in the order you actually reach for them:
 *   TELEMETRY  the raw operational log
 *   TRANSCRIPT the conversation
 *   TOOLS      every skill the cortex executed, with arguments and results
 *   MIND       what is in long-term memory: facts, notes, pending reminders,
 *              and a keyword search across every past session
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Fact, type Note, type Recollection, type Reminder } from "@/lib/api";
import { useJarvis } from "@/state/jarvis";

const LEVEL_CLASS: Record<string, string> = {
  info: "log-info",
  voice: "log-voice",
  success: "log-success",
  warn: "log-warn",
  error: "log-error",
};

type Tab = "log" | "chat" | "tools" | "mind";
type Filter = "all" | "voice" | "agent" | "issues";

function clock(ts: number): string {
  const d = new Date(ts);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/* ------------------------------------------------------------------ MIND -- */

function MindTab() {
  const memoryStats = useJarvis((s) => s.memoryStats);
  const [facts, setFacts] = useState<Fact[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Recollection[] | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    setBusy(true);
    const [f, n, r] = await Promise.all([api.facts(), api.notes(), api.reminders()]);
    setFacts(f?.facts ?? []);
    setNotes(n?.notes ?? []);
    setReminders(r?.reminders ?? []);
    setBusy(false);
  }, []);

  // `memoryVersion` ticks on every `memory.changed` frame, so a fact stored by
  // voice while this tab is open appears without a reload. It is the same
  // signal the archive watches, and this tab was simply not listening.
  const memoryVersion = useJarvis((s) => s.memoryVersion);
  useEffect(() => {
    void reload();
  }, [reload, memoryVersion]);

  const search = async (event: React.FormEvent) => {
    event.preventDefault();
    const q = query.trim();
    if (!q) {
      setMatches(null);
      return;
    }
    setBusy(true);
    const result = await api.searchMemory(q);
    setMatches(result?.matches ?? []);
    setBusy(false);
  };

  return (
    <div className="mind">
      <form className="mind-search" onSubmit={search}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search everything ever said…"
          spellCheck={false}
          aria-label="Search long-term memory"
        />
        <button type="submit" className="mind-btn" disabled={busy}>
          {busy ? "…" : "recall"}
        </button>
        {matches !== null && (
          <button
            type="button"
            className="mind-btn ghost"
            onClick={() => {
              setMatches(null);
              setQuery("");
            }}
          >
            clear
          </button>
        )}
      </form>

      {matches !== null ? (
        <>
          <div className="mind-head">
            {matches.length} match{matches.length === 1 ? "" : "es"}
          </div>
          {matches.map((hit, i) => (
            <div className={`mind-row recall-${hit.role}`} key={`${hit.ts}-${i}`}>
              <span className="mind-meta">
                {hit.role === "user" ? "you" : "jarvis"} · {when(hit.ts)}
              </span>
              <span className="mind-text">{hit.text}</span>
            </div>
          ))}
          {matches.length === 0 && <div className="log-line log-info">nothing on record.</div>}
        </>
      ) : (
        <>
          {memoryStats && (
            <div className="mind-stats">
              <span>{memoryStats.turns} exchanges</span>
              <span>{memoryStats.sessions} sessions</span>
              <span>{memoryStats.size_kb} KB</span>
              <span>{memoryStats.fts ? "fts5" : "scan"}</span>
            </div>
          )}

          <div className="mind-head">
            facts <b>{facts.length}</b>
          </div>
          {facts.map((fact) => (
            <div className="mind-row" key={fact.key}>
              <span className="mind-key">{fact.key}</span>
              <span className="mind-text">{fact.value}</span>
              <button
                type="button"
                className="mind-x"
                title="Forget this"
                onClick={async () => {
                  await api.forgetFact(fact.key);
                  void reload();
                }}
              >
                ×
              </button>
            </div>
          ))}
          {facts.length === 0 && (
            <div className="log-line log-info">
              nothing stored — try “remember that I take my coffee black”.
            </div>
          )}

          <div className="mind-head">
            reminders <b>{reminders.length}</b>
          </div>
          {reminders.map((reminder) => (
            <div className="mind-row" key={reminder.id}>
              <span className="mind-key">{when(reminder.due_ts)}</span>
              <span className="mind-text">{reminder.text}</span>
              <button
                type="button"
                className="mind-x"
                title="Cancel"
                onClick={async () => {
                  await api.cancelReminder(reminder.id);
                  void reload();
                }}
              >
                ×
              </button>
            </div>
          ))}
          {reminders.length === 0 && <div className="log-line log-info">none pending.</div>}

          <div className="mind-head">
            notes <b>{notes.length}</b>
          </div>
          {notes.map((note) => (
            <div className="mind-row" key={note.id}>
              <span className="mind-key">{when(note.ts)}</span>
              <span className="mind-text">{note.text}</span>
              <button
                type="button"
                className="mind-x"
                title="Delete"
                onClick={async () => {
                  await api.deleteNote(note.id);
                  void reload();
                }}
              >
                ×
              </button>
            </div>
          ))}
          {notes.length === 0 && <div className="log-line log-info">no notes.</div>}

          <div className="mind-actions">
            <a className="mind-btn ghost" href={api.exportUrl("text")} target="_blank" rel="noreferrer">
              export transcript
            </a>
            <button type="button" className="mind-btn ghost" onClick={() => void reload()}>
              refresh
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- panel -- */

export default function TelemetryPanel() {
  const logs = useJarvis((s) => s.logs);
  const transcript = useJarvis((s) => s.transcript);
  const tools = useJarvis((s) => s.tools);
  const connected = useJarvis((s) => s.logsConnected);
  const clearLogs = useJarvis((s) => s.clearLogs);

  const [tab, setTab] = useState<Tab>("log");
  const [filter, setFilter] = useState<Filter>("all");
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const visible = useMemo(() => {
    if (filter === "all") return logs;
    if (filter === "voice") return logs.filter((l) => l.level === "voice" || l.source === "tts" || l.source === "stt");
    if (filter === "agent")
      return logs.filter(
        (l) =>
          l.source.startsWith("agent") ||
          l.source === "crew" ||
          l.source === "brain" ||
          l.source === "reflex" ||
          l.source === "memory",
      );
    return logs.filter((l) => l.level === "warn" || l.level === "error");
  }, [logs, filter]);

  useEffect(() => {
    // The MIND tab is browsed, not streamed: yanking it to the bottom on every
    // render would fight the reader.
    if (tab !== "mind" && stick.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible, transcript, tools, tab]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const counts = useMemo(() => {
    let issues = 0;
    for (const l of logs) if (l.level === "warn" || l.level === "error") issues++;
    return { issues };
  }, [logs]);

  const TABS: Array<[Tab, string]> = [
    ["log", "TELEMETRY"],
    ["chat", `TRANSCRIPT${transcript.length ? ` ${transcript.length}` : ""}`],
    ["tools", `TOOLS${tools.length ? ` ${tools.length}` : ""}`],
    ["mind", "MIND"],
  ];

  return (
    <section className="telemetry" data-connected={connected}>
      <header className="telemetry-header">
        <span className="telemetry-tabs">
          {TABS.map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`tab ${tab === key ? "on" : ""}`}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </span>
        {/* Archive, reference and settings used to live here as bare glyphs.
            They are labelled buttons in the ACCESS card now, under the vitals,
            which leaves this header to say the one thing it is for: whether
            the uplink is live. */}
        <span className="telemetry-header-actions">
          <span className={`telemetry-link ${connected ? "linked" : ""}`}>
            {connected ? "LINKED" : "RECONNECTING"}
          </span>
        </span>
      </header>

      {tab === "log" && (
        <div className="telemetry-filters">
          {(["all", "voice", "agent", "issues"] as Filter[]).map((f) => (
            <button
              key={f}
              type="button"
              className={`filter ${filter === f ? "on" : ""} ${f === "issues" && counts.issues ? "has-issues" : ""}`}
              onClick={() => setFilter(f)}
            >
              {f}
              {f === "issues" && counts.issues ? ` ${counts.issues}` : ""}
            </button>
          ))}
          <button type="button" className="filter ghost" onClick={clearLogs} title="Clear log">
            clear
          </button>
        </div>
      )}

      <div className="telemetry-scroll" ref={scrollRef} onScroll={onScroll}>
        {tab === "log" && (
          <>
            {visible.map((line) => (
              <div className={`log-line ${LEVEL_CLASS[line.level] ?? "log-info"}`} key={line.id}>
                <span className="log-ts">{clock(line.ts)}</span>
                <span className="log-source">{line.source}</span>
                <span className="log-msg">{line.msg}</span>
              </div>
            ))}
            {visible.length === 0 && <div className="log-line log-info">no matching lines…</div>}
          </>
        )}

        {tab === "chat" && (
          <>
            {transcript.map((turn) => (
              <div className={`turn turn-${turn.role}`} key={turn.id}>
                <span className="turn-who">{turn.role === "user" ? "YOU" : "J.A.R.V.I.S."}</span>
                <span className="turn-text">{turn.text}</span>
              </div>
            ))}
            {transcript.length === 0 && <div className="log-line log-info">no exchanges yet…</div>}
          </>
        )}

        {tab === "tools" && (
          <>
            {tools.map((tool, i) => (
              <div className={`tool-row ${tool.ok ? "ok" : "bad"}`} key={`${tool.ts}-${i}`}>
                <span className="tool-head">
                  <span className="tool-name">{tool.name}</span>
                  <span className="tool-ms">{tool.elapsed_ms}ms</span>
                </span>
                {Object.keys(tool.args ?? {}).length > 0 && (
                  <span className="tool-args">{JSON.stringify(tool.args)}</span>
                )}
                <span className="tool-detail">{tool.detail}</span>
              </div>
            ))}
            {tools.length === 0 && (
              <div className="log-line log-info">
                no skills executed yet — ask something that needs real data.
              </div>
            )}
          </>
        )}

        {tab === "mind" && <MindTab />}
      </div>
    </section>
  );
}
