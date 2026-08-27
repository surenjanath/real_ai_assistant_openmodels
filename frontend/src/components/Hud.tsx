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

function personaLabel(settings: { persona: string; personas: Array<{ key: string; label: string }> }): string {
  const found = settings.personas?.find((p) => p.key === settings.persona);
  return (found?.label ?? settings.persona ?? "—").toLowerCase();
}

function uptime(seconds?: number): string {
  if (!seconds) return "";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function Hud() {
  const status = useJarvis((s) => s.status);
  const engines = useJarvis((s) => s.engines);
  const settings = useJarvis((s) => s.settings);
  const logsConnected = useJarvis((s) => s.logsConnected);
  const audioConnected = useJarvis((s) => s.audioConnected);
  const latency = useJarvis((s) => s.latencyMs);
  const vitals = useJarvis((s) => s.vitals);
  const memoryStats = useJarvis((s) => s.memoryStats);
  const skills = useJarvis((s) => s.skills);

  const style = STATUS_STYLE[status];
  const voiceReal = engines.ttsMode === "kokoro";

  return (
    <div className="hud">
      <div className="hud-wordmark">
        J.A.R.V.I.S.
        <span className="hud-sub">Just A Rather Very Intelligent System</span>
      </div>

      <div className="hud-row">
        <span className="status-pill" data-status={status}>
          <span
            className="status-dot"
            style={{ background: style.color, boxShadow: `0 0 10px ${style.color}` }}
          />
          {style.label}
        </span>
      </div>

      <div className="hud-row hud-meta">
        <span className={`badge ${voiceReal ? "good" : "warnish"}`} title={engines.ttsLabel || engines.tts}>
          voice · {voiceReal ? "kokoro-82M" : "fallback synth"}
        </span>
        <span className="badge" title={`runtime ${engines.agents}`}>
          crew · {engines.mode || engines.agents}
        </span>
        {settings.think_active && (
          <span className="badge warnish" title="Extended chain-of-thought is on - answers are slower">
            deep think
          </span>
        )}
      </div>

      <div className="hud-row hud-meta">
        <span className="badge" title="Disposition - say “be more concise” or “switch persona to Friday”">
          persona · {personaLabel(settings)}
        </span>
        <span
          className={`badge ${settings.tools_active ? "good" : ""}`}
          title={
            settings.tools_active
              ? "The cortex calls skills itself when a question needs real data"
              : settings.tools_supported
                ? "Skills are disabled in settings"
                : "This model does not advertise tool calling"
          }
        >
          skills · {settings.tools_active ? `${skills.length} armed` : "off"}
        </span>
      </div>

      <div className="hud-row hud-meta">
        <span className={`badge ${settings.model_verified ? "good" : "warnish"}`}>
          model · {engines.model || settings.model || "none"}
        </span>
        {memoryStats && (
          <span
            className={`badge ${settings.recall ? "" : "warnish"}`}
            title={
              settings.recall
                ? `Recalling across ${memoryStats.sessions} past sessions`
                : "Long-term recall is off - only this conversation is used"
            }
          >
            memory · {settings.recall ? `${memoryStats.turns}` : "off"}
          </span>
        )}
      </div>

      <div className="hud-row hud-meta dim">
        <span className={logsConnected ? "on" : "off"}>◉ link {logsConnected ? "up" : "down"}</span>
        <span className={audioConnected ? "on" : "off"}>◉ audio {audioConnected ? "up" : "down"}</span>
        {latency !== null && <span>{latency}ms</span>}
        {vitals?.uptime_s ? <span>up {uptime(vitals.uptime_s)}</span> : null}
      </div>
    </div>
  );
}
