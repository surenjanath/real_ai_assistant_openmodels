"use client";

/**
 * Settings panel - model / voice / speed, switchable live.
 *
 * Opened by the ⚙ button in the telemetry header, or by saying "settings"
 * (the backend routes the voice command to a `ui` frame). Changes POST to
 * /api/settings; every connected client (incl. this one) then receives a
 * `settings.update` frame, so panels stay in sync and the change is spoken
 * back by the assistant.
 */

import { useEffect, useState } from "react";
import { useJarvis } from "@/state/jarvis";
import type { LogLevel } from "@/lib/protocol";

type SaveState = "idle" | "saving" | "ok" | "error";

export default function SettingsPanel() {
  const open = useJarvis((s) => s.settingsOpen);
  const setOpen = useJarvis((s) => s.setSettingsOpen);
  const settings = useJarvis((s) => s.settings);
  const setSettings = useJarvis((s) => s.setSettings);
  const engines = useJarvis((s) => s.engines);
  const pushLog = useJarvis((s) => s.pushLog);
  const [save, setSave] = useState<SaveState>("idle");
  const [detail, setDetail] = useState("");

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open) return null;

  const apply = async (delta: { model?: string; voice?: string; speed?: number }) => {
    setSave("saving");
    setDetail("");
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(delta),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.errors?.[0] ?? `HTTP ${res.status}`);
      if (data.settings) setSettings(data.settings);
      setSave("ok");
      const changed = Object.entries(data.applied ?? {})
        .map(([k, v]) => `${k} → ${v}`)
        .join(", ");
      setDetail(changed ? `applied ${changed}` : "no change");
      const level: LogLevel = changed ? "success" : "info";
      pushLog(level, "settings", changed ? `panel: ${changed}` : "panel: no changes");
    } catch (err) {
      setSave("error");
      setDetail(err instanceof Error ? err.message : String(err));
      pushLog("error", "settings", `panel: ${detail || "update failed"}`);
    } finally {
      setTimeout(() => setSave("idle"), 2400);
    }
  };

  const statusText =
    save === "saving" ? "APPLYING…" : save === "ok" ? "APPLIED ✓" : save === "error" ? "REJECTED" : "";

  return (
    <>
      <div className="settings-scrim" onClick={() => setOpen(false)} />
      <section className="settings-panel" role="dialog" aria-label="Settings">
        <header className="telemetry-header">
          <span className="telemetry-title">SYSTEM CONFIGURATION</span>
          <button className="settings-close" onClick={() => setOpen(false)} aria-label="Close settings" type="button">
            ✕
          </button>
        </header>

        <div className="settings-body">
          <label className="settings-field">
            <span className="settings-label">
              CREW MODEL
              <em className="settings-note">
                {settings.model_verified ? "verified on ollama" : "unverified — ollama offline or not pulled"}
              </em>
            </span>
            <select
              value={settings.model}
              onChange={(event) => apply({ model: event.target.value })}
              disabled={save === "saving"}
            >
              {!settings.models.includes(settings.model) && (
                <option value={settings.model}>{settings.model}</option>
              )}
              {settings.models.map((model) => (
                <option key={model} value={model}>
                  {model}
                  {model === settings.model ? " ●" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              VOICE
              <em className="settings-note">applies on the next utterance</em>
            </span>
            <select
              value={settings.voice}
              onChange={(event) => apply({ voice: event.target.value })}
              disabled={save === "saving"}
            >
              {!settings.voices.includes(settings.voice) && (
                <option value={settings.voice}>{settings.voice}</option>
              )}
              {settings.voices.map((voice) => (
                <option key={voice} value={voice}>
                  {voice}
                  {voice === settings.voice ? " ●" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              SPEECH SPEED <b className="settings-value">{settings.speed.toFixed(2)}×</b>
            </span>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={settings.speed}
              onChange={(event) => setSettings({ speed: Number(event.target.value) })}
              onPointerUp={(event) => apply({ speed: Number((event.target as HTMLInputElement).value) })}
              onKeyUp={(event) => apply({ speed: Number((event.target as HTMLInputElement).value) })}
              disabled={save === "saving"}
            />
            <span className="settings-scale">
              <span>0.5×</span>
              <span>1.0×</span>
              <span>2.0×</span>
            </span>
          </label>

          <div className="settings-status">
            <span>tts/{engines.tts}</span>
            <span>crew/{engines.agents}</span>
          </div>

          <p className="settings-hint">
            Voice: say <i>“settings”</i>, <i>“list models”</i>, <i>“switch model to qwen3
            8b”</i>, <i>“change voice to af heart”</i>, or <i>“speak slower”</i>.
          </p>
        </div>

        <footer className="settings-footer">
          <span className={`settings-save ${save === "error" ? "bad" : "good"}`}>{statusText}</span>
          <span className="settings-detail">{detail}</span>
        </footer>
      </section>
    </>
  );
}
