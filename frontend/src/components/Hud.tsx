"use client";

/** Top-left identity cluster: wordmark, live status pill, engine badges. */

import { useJarvis } from "@/state/jarvis";
import type { AssistantStatus } from "@/lib/protocol";

const STATUS_STYLE: Record<AssistantStatus, { label: string; color: string }> = {
  boot: { label: "BOOTING", color: "#8a919e" },
  idle: { label: "STANDBY", color: "#5d6773" },
  listening: { label: "LISTENING", color: "#5eb0ff" },
  thinking: { label: "PROCESSING", color: "#e8b446" },
  speaking: { label: "TRANSMITTING", color: "#4ade80" },
};

export default function Hud() {
  const status = useJarvis((s) => s.status);
  const engines = useJarvis((s) => s.engines);
  const logsConnected = useJarvis((s) => s.logsConnected);
  const audioConnected = useJarvis((s) => s.audioConnected);
  const latency = useJarvis((s) => s.latencyMs);

  const style = STATUS_STYLE[status];

  return (
    <div className="hud">
      <div className="hud-wordmark">
        J.A.R.V.I.S.
        <span className="hud-sub">Just A Rather Very Intelligent System</span>
      </div>
      <div className="hud-row">
        <span className="status-pill">
          <span className="status-dot" style={{ background: style.color, boxShadow: `0 0 8px ${style.color}` }} />
          {style.label}
        </span>
      </div>
      <div className="hud-row hud-meta">
        <span>tts/{engines.tts}</span>
        <span>crew/{engines.agents}</span>
        {engines.model ? <span>model/{engines.model}</span> : null}
      </div>
      <div className="hud-row hud-meta dim">
        <span className={logsConnected ? "on" : "off"}>◉ telemetry {logsConnected ? "linked" : "offline"}</span>
        <span className={audioConnected ? "on" : "off"}>◉ audio {audioConnected ? "linked" : "offline"}</span>
        {latency !== null && <span>{latency}ms</span>}
      </div>
    </div>
  );
}
