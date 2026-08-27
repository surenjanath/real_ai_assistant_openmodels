"use client";

/**
 * Telemetry panel (PRD §2): floating semi-transparent terminal anchored to
 * the right edge. Monospace, auto-scrolling, color-coded:
 *   gray = processing · blue = STT/TTS · green = workflow success
 * (plus amber warnings / red errors for completeness).
 *
 * Two tabs: the raw operational log, and the conversation transcript.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useJarvis } from "@/state/jarvis";
import type { LogLevel } from "@/lib/protocol";

const LEVEL_CLASS: Record<string, string> = {
  info: "log-info",
  voice: "log-voice",
  success: "log-success",
  warn: "log-warn",
  error: "log-error",
};

type Filter = "all" | "voice" | "agent" | "issues";

function clock(ts: number): string {
  const d = new Date(ts);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

export default function TelemetryPanel() {
  const logs = useJarvis((s) => s.logs);
  const transcript = useJarvis((s) => s.transcript);
  const connected = useJarvis((s) => s.logsConnected);
  const clearLogs = useJarvis((s) => s.clearLogs);

  const [tab, setTab] = useState<"log" | "chat">("log");
  const [filter, setFilter] = useState<Filter>("all");
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const visible = useMemo(() => {
    if (filter === "all") return logs;
    if (filter === "voice") return logs.filter((l) => l.level === "voice" || l.source === "tts" || l.source === "stt");
    if (filter === "agent") return logs.filter((l) => l.source.startsWith("agent") || l.source === "crew" || l.source === "brain");
    return logs.filter((l) => l.level === "warn" || l.level === "error");
  }, [logs, filter]);

  useEffect(() => {
    if (stick.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible, transcript, tab]);

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

  return (
    <section className="telemetry" data-connected={connected}>
      <header className="telemetry-header">
        <span className="telemetry-tabs">
          <button
            type="button"
            className={`tab ${tab === "log" ? "on" : ""}`}
            onClick={() => setTab("log")}
          >
            TELEMETRY
          </button>
          <button
            type="button"
            className={`tab ${tab === "chat" ? "on" : ""}`}
            onClick={() => setTab("chat")}
          >
            TRANSCRIPT{transcript.length ? ` (${transcript.length})` : ""}
          </button>
        </span>
        <span className="telemetry-header-actions">
          <button
            className="gear"
            onClick={() => useJarvis.getState().setSettingsOpen(true)}
            title="Settings (or say “settings”)"
            aria-label="Open settings"
            type="button"
          >
            ⚙
          </button>
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
        {tab === "log" ? (
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
        ) : (
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
      </div>
    </section>
  );
}
