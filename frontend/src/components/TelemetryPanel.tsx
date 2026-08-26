"use client";

/**
 * Telemetry panel (PRD §2): floating semi-transparent terminal anchored to
 * the right edge. Monospace, max-width 350px, auto-scrolling, color-coded:
 *   gray = processing · blue = STT/TTS · green = workflow success
 * (plus amber warnings / red errors for completeness).
 *
 * Includes the command bar (typed + voice) that drives the agent pipeline.
 */

import { useEffect, useRef, useState } from "react";
import { useJarvis } from "@/state/jarvis";
import { sendCommand } from "@/hooks/useJarvisConnection";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";

const LEVEL_CLASS: Record<string, string> = {
  info: "log-info",
  voice: "log-voice",
  success: "log-success",
  warn: "log-warn",
  error: "log-error",
};

function clock(ts: number): string {
  const d = new Date(ts);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

export default function TelemetryPanel() {
  const logs = useJarvis((s) => s.logs);
  const status = useJarvis((s) => s.status);
  const connected = useJarvis((s) => s.logsConnected);
  const pushLog = useJarvis((s) => s.pushLog);
  const setStatus = useJarvis((s) => s.setStatus);

  const [command, setCommand] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const { listening, supported: micSupported, toggle: toggleMic } = useSpeechRecognition({
    onTranscript: (text) => {
      pushLog("voice", "stt", `"${text}"`);
      setStatus("thinking");
      if (!sendCommand(text, "voice")) {
        pushLog("error", "client", "uplink unavailable - command dropped");
      }
    },
    onError: (message) => pushLog("error", "stt", message),
  });

  useEffect(() => {
    if (stick.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const submit = () => {
    const text = command.trim();
    if (!text) return;
    setCommand("");
    stick.current = true;
    pushLog(listening ? "voice" : "info", listening ? "stt" : "you", `"${text}"`);
    setStatus("thinking");
    if (!sendCommand(text, listening ? "voice" : "text")) {
      pushLog("error", "client", "uplink unavailable - command dropped");
    }
  };

  return (
    <section className="telemetry" data-connected={connected}>
      <header className="telemetry-header">
        <span className="telemetry-title">OPERATIONAL TELEMETRY</span>
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

      <div className="telemetry-scroll" ref={scrollRef} onScroll={onScroll}>
        {logs.map((line) => (
          <div className={`log-line ${LEVEL_CLASS[line.level] ?? "log-info"}`} key={line.id}>
            <span className="log-ts">{clock(line.ts)}</span>
            <span className="log-source">{line.source}</span>
            <span className="log-msg">{line.msg}</span>
          </div>
        ))}
        {logs.length === 0 && <div className="log-line log-info">awaiting uplink…</div>}
      </div>

      <footer className="telemetry-input">
        <span className="telemetry-prompt">❯</span>
        <input
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") submit();
          }}
          placeholder={status === "boot" ? "establishing uplink…" : "issue a directive"}
          spellCheck={false}
          autoComplete="off"
          aria-label="Command input"
        />
        {micSupported && (
          <button
            className={`mic ${listening ? "active" : ""}`}
            onClick={toggleMic}
            title={listening ? "Stop listening" : "Voice command"}
            aria-label="Toggle voice command"
            type="button"
          >
            {listening ? "◉" : "◎"}
          </button>
        )}
      </footer>
    </section>
  );
}
