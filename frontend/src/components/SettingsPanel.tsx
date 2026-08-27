"use client";

/**
 * Settings panel - model / voice / speed / volume, switchable live.
 *
 * Opened by the ⚙ button in the telemetry header, or by saying "settings"
 * (the backend routes the voice command to a `ui` frame). Changes POST to
 * /api/settings; every connected client then receives a `settings.update`
 * frame, so panels stay in sync and the change is spoken back.
 *
 * Models are split into what Ollama actually has installed versus mere
 * suggestions, because picking an un-pulled tag is the single easiest way to
 * end up with a crew that cannot answer.
 */

import { useEffect, useState } from "react";
import { audioEngine } from "@/audio/engine";
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
  const [volume, setVolume] = useState(0.9);
  const [muted, setMuted] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open) return null;

  const apply = async (delta: {
    model?: string;
    voice?: string;
    speed?: number;
    think?: boolean;
  }) => {
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
      const message = err instanceof Error ? err.message : String(err);
      setSave("error");
      setDetail(message);
      pushLog("error", "settings", `panel: ${message}`);
    } finally {
      setTimeout(() => setSave("idle"), 2400);
    }
  };

  const testVoice = () => {
    audioEngine.unlock();
    void fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "Voice check. All systems are functioning within normal parameters.",
      }),
    }).catch(() => undefined);
  };

  const statusText =
    save === "saving" ? "APPLYING…" : save === "ok" ? "APPLIED ✓" : save === "error" ? "REJECTED" : "";

  const installed = settings.installed ?? [];
  const suggestions = (settings.models ?? []).filter((m) => !installed.includes(m));
  const label = (voice: string) => settings.voice_labels?.[voice] ?? voice;

  return (
    <>
      <div className="settings-scrim" onClick={() => setOpen(false)} />
      <section className="settings-panel" role="dialog" aria-label="Settings">
        <header className="telemetry-header">
          <span className="telemetry-title">SYSTEM CONFIGURATION</span>
          <button
            className="settings-close"
            onClick={() => setOpen(false)}
            aria-label="Close settings"
            type="button"
          >
            ✕
          </button>
        </header>

        <div className="settings-body">
          <label className="settings-field">
            <span className="settings-label">
              CREW MODEL
              <em className={`settings-note ${settings.model_verified ? "" : "bad"}`}>
                {settings.model_verified
                  ? `installed on ollama · ${installed.length} available`
                  : "NOT installed — pull it or pick one below"}
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
              {installed.length > 0 && (
                <optgroup label="Installed">
                  {installed.map((model) => (
                    <option key={model} value={model}>
                      {model}
                      {model === settings.model ? "  ●" : ""}
                    </option>
                  ))}
                </optgroup>
              )}
              {suggestions.length > 0 && (
                <optgroup label="Not installed (ollama pull …)">
                  {suggestions.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              VOICE
              <em className="settings-note">
                {engines.ttsMode === "kokoro"
                  ? "kokoro-82M · applies on the next utterance"
                  : "fallback synth — install pykokoro for real speech"}
              </em>
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
                  {label(voice)}
                  {voice === settings.voice ? "  ●" : ""}
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

          <label className="settings-field">
            <span className="settings-label">
              OUTPUT VOLUME <b className="settings-value">{Math.round(volume * 100)}%</b>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volume}
              onChange={(event) => {
                const v = Number(event.target.value);
                setVolume(v);
                audioEngine.setVolume(v);
              }}
            />
          </label>

          <label className="settings-field">
            <span className="settings-label">
              EXTENDED THINKING
              <em className={`settings-note ${settings.think_supported ? "" : "bad"}`}>
                {settings.think_supported
                  ? "slower, more careful answers"
                  : "this model has no thinking mode"}
              </em>
            </span>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle ${!settings.think ? "on" : ""}`}
                onClick={() => apply({ think: false })}
                disabled={save === "saving"}
              >
                OFF · fast
              </button>
              <button
                type="button"
                className={`toggle ${settings.think ? "on" : ""}`}
                onClick={() => apply({ think: true })}
                disabled={save === "saving" || !settings.think_supported}
                title={
                  settings.think_supported
                    ? "Let the model reason before answering"
                    : "Unavailable on this model"
                }
              >
                ON · deep
              </button>
            </div>
          </label>

          <div className="settings-actions">
            <button type="button" className="settings-btn" onClick={testVoice}>
              ▶ Test voice
            </button>
            <button
              type="button"
              className={`settings-btn ${muted ? "on" : ""}`}
              onClick={() => {
                const next = !muted;
                setMuted(next);
                audioEngine.setMuted(next);
              }}
            >
              {muted ? "🔇 Muted" : "🔊 Sound on"}
            </button>
          </div>

          <div className="settings-status">
            <span>tts/{engines.ttsLabel || engines.tts}</span>
            <span>crew/{engines.agents}</span>
            {engines.mode && <span>mode/{engines.mode}</span>}
          </div>

          <p className="settings-hint">
            Voice: say <i>“settings”</i>, <i>“list models”</i>, <i>“switch model to qwen3
            8b”</i>, <i>“change voice to af heart”</i>, <i>“speak slower”</i>,{" "}
            <i>“think harder”</i>, <i>“answer faster”</i>, or <i>“stop”</i>.
            <br />
            Keys: <b>/</b> focus · <b>Esc</b> stop speech · <b>⌘K</b> wake word.
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
